import sys, os
import time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from utils_folder.SIREN import SIREN
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import gymnasium as gym
import casadi as ca

# ================== 图像设置 =======================
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['text.usetex'] = False
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 24
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['axes.labelsize'] = 24
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.alpha'] = 0.5
mpl.rcParams['grid.color'] = 'gray'

class Args:
    env_id = "environment:nonlinear2d-v0"
    learning_rate = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    filename_prefix = "sp_nonlinear2d_"

def dynamics(X, U):
    x1 = X[:, 0:1]
    x2 = X[:, 1:2]
    c = torch.cos(2*x1) + 2
    x1_dot = -x1 + x2
    x2_dot = -0.5*x1 - 0.5*x2*(1 - torch.square(c)) + c * U
    return torch.cat([x1_dot, x2_dot], dim=1)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class SPNet(nn.Module):
    def __init__(self, env):
        self.env = env
        super().__init__()
        self.net = SIREN(in_features=2, out_features=1, hidden_features=64, hidden_layers=2, norm_scale=4.0)

    def forward(self, x):
        return self.net(x)

def setup_casadi_mpc():
    opti = ca.Opti()
    N = 30
    dt = 0.1
    x = opti.variable(2, N+1)
    u = opti.variable(1, N)
    x0 = opti.parameter(2, 1)

    opti.subject_to(x[:, 0] == x0)
    cost = 0
    for k in range(N):
        x1 = x[0, k]
        x2 = x[1, k]
        uk = u[0, k]
        c = ca.cos(2*x1) + 2
        dx1 = -x1 + x2
        dx2 = -0.5*x1 - 0.5*x2*(1 - c**2) + c*uk
        
        opti.subject_to(x[0, k+1] == x1 + dx1*dt)
        opti.subject_to(x[1, k+1] == x2 + dx2*dt)
        
        cost += (x1**2 + x2**2 + uk**2) * dt
    
    cost += (x[0, N]**2 + x[1, N]**2)
    opti.minimize(cost)
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
    opti.solver('ipopt', opts)
    return opti, x0, u

def train_sp():
    env = gym.make(Args.env_id)
    model = SPNet(env).to(Args.device)
    optimizer = optim.Adam(model.parameters(), lr=Args.learning_rate)

    print("--- Phase 1: Data Generation using MPC ---")
    start_time = time.time()
    
    opti, x0_param, u_var = setup_casadi_mpc()
    
    X_data = []
    U_data = []
    
    while time.time() - start_time < 15:
        x_sample = np.random.uniform(-4.0, 4.0, 2)
        opti.set_value(x0_param, x_sample)
        try:
            sol = opti.solve()
            u_opt = sol.value(u_var[0, 0])
            X_data.append(x_sample)
            U_data.append([u_opt])
        except Exception:
            pass 

    X_data = np.array(X_data, dtype=np.float32)
    U_data = np.array(U_data, dtype=np.float32)
    print(f"Generated {len(X_data)} samples in 15 seconds.")
    
    if len(X_data) == 0:
        print("No data generated. Exiting.")
        return

    X_tensor_all = torch.from_numpy(X_data).to(Args.device)
    U_tensor_all = torch.from_numpy(U_data).to(Args.device)

    print("--- Phase 2: Training Network ---")
    train_start = time.time()
    batch_size = min(Args.batch_size if hasattr(Args, 'batch_size') else 128, len(X_data))
    
    step = 0
    while time.time() - train_start < 15:
        idx = np.random.choice(len(X_data), batch_size)
        X_batch = X_tensor_all[idx]
        U_batch = U_tensor_all[idx]
        
        u_pred = model(X_batch)
        loss = torch.nn.functional.mse_loss(u_pred, U_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.6f}")
        step += 1

    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "sp_nonlinear2d.pth")
    torch.save(model.state_dict(), model_path)
    print(f"SP model saved to {model_path}. Total training steps: {step}")


def plot_value():
    model_dir = "model"
    model_path = os.path.join(model_dir, "sp_nonlinear2d.pth")
    args = Args()
    env = gym.make(args.env_id)
    model = SPNet(env).to(Args.device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=Args.device))
    else:
        print(f"Model file {model_path} not found. Running with untrained model.")
    model.eval()

    x1 = np.linspace(-3, 3, 50)
    x2 = np.linspace(-3, 3, 50)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.stack([X1.ravel(), X2.ravel()], axis=1)

    X_tensor = torch.from_numpy(X_grid).float().to(Args.device)

    dt = 0.05
    T_steps = 400
    V_sp = torch.zeros(X_tensor.shape[0], device=Args.device)
    curr_x = X_tensor.clone()
    
    print("Simulating grid for SP value function...")
    with torch.no_grad():
        for _ in range(T_steps):
            u = model(curr_x)
            x1_t = curr_x[:, 0:1]
            x2_t = curr_x[:, 1:2]
            cost = x1_t**2 + x2_t**2 + u**2
            V_sp += cost.squeeze() * dt
            
            k1 = dynamics(curr_x, u)
            x2_rk = curr_x + 0.5 * dt * k1
            u2 = model(x2_rk)
            k2 = dynamics(x2_rk, u2)
            x3_rk = curr_x + 0.5 * dt * k2
            u3 = model(x3_rk)
            k3 = dynamics(x3_rk, u3)
            x4_rk = curr_x + dt * k3
            u4 = model(x4_rk)
            k4 = dynamics(x4_rk, u4)
            
            curr_x = curr_x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    sp_values = V_sp.cpu().numpy().reshape(X1.shape)
    
    dx1 = x1[1] - x1[0]
    dx2 = x2[1] - x2[0]
    dV_dx2, dV_dx1 = np.gradient(sp_values, dx2, dx1)
    
    with torch.no_grad():
        u_grid = model(X_tensor)
        fxu_grid = dynamics(X_tensor, u_grid).cpu().numpy()
    
    fxu_x1 = fxu_grid[:, 0].reshape(X1.shape)
    fxu_x2 = fxu_grid[:, 1].reshape(X1.shape)
    V_dot_sp = dV_dx1 * fxu_x1 + dV_dx2 * fxu_x2

    # ------------------ 计算 Optimal 值函数作为参考 ------------------
    P = np.array([[0.5, 0.0], [0.0, 1.0]])
    opt_values = np.einsum('ij,jk,ik->i', X_grid, P, X_grid).reshape(X1.shape)
    
    X_grid_torch = torch.from_numpy(X_grid).float()
    P_torch = torch.from_numpy(P).float()
    gradV_opt = 2 * torch.matmul(X_grid_torch, P_torch)  # (N, 2)
    x1_grid = X_grid_torch[:, 0:1]
    g2_grid = torch.cos(2 * x1_grid) + 2
    u_star_opt = -0.5 * g2_grid * gradV_opt[:, 1:2]  # (N, 1)
    with torch.no_grad():
        fxu_opt = dynamics(X_grid_torch, u_star_opt)
    V_dot_opt = (gradV_opt * fxu_opt).sum(dim=1).reshape(X1.shape).detach().numpy()

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1.4])
    ax = fig.add_subplot(gs[0], projection='3d')

    surf1 = ax.plot_surface(X1, X2, sp_values, cmap='Blues', alpha=0.85, linewidth=0, antialiased=True)
    surf1_1 = ax.plot_surface(X1, X2, V_dot_sp, cmap='Blues', alpha=0.85, linewidth=0, antialiased=True)
    surf2 = ax.plot_surface(X1, X2, opt_values, cmap='Oranges', alpha=0.47, linewidth=0, antialiased=True, rstride=1, cstride=1)
    surf2_1 = ax.plot_surface(X1, X2, V_dot_opt, cmap='Oranges', alpha=0.46, linewidth=0, antialiased=True, rstride=1, cstride=1)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=plt.cm.Blues(0.7), lw=2, label='SPNet'),
        Line2D([0], [0], color=plt.cm.Oranges(0.7), lw=2, label='Optimal')
    ]
    ax.legend(handles=legend_elements, ncol=2, loc='upper right')

    ax.set_xlabel(r'$x_1$', labelpad=10)
    ax.set_ylabel(r'$x_2$', labelpad=10)
    ax.set_zlabel(r'$V$ or $\dot V$', labelpad=10)
    ax.view_init(elev=10, azim=48, roll=0)

    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim([-3.5, 3.2])
    ax2.set_ylim([-3.5, 3.2])
    
    x1_lin = np.linspace(-3, 3, 22)
    x2_lin = np.linspace(-3, 3, 22)
    X1_grid, X2_grid = np.meshgrid(x1_lin, x2_lin)
    U = np.zeros_like(X1_grid)
    Vv = np.zeros_like(X2_grid)

    for i in range(X1_grid.shape[0]):
        for j in range(X1_grid.shape[1]):
            x_point = np.array([X1_grid[i, j], X2_grid[i, j]])
            x_tensor = torch.from_numpy(x_point).float().unsqueeze(0).to(Args.device)
            with torch.no_grad():
                u = model(x_tensor)
            f_vec = dynamics(x_tensor, u).cpu().numpy().squeeze()
            U[i, j] = f_vec[0]
            Vv[i, j] = f_vec[1]

    speed = np.sqrt(U**2 + Vv**2)
    color = speed
    step = 2
    Q_quiver = ax2.quiver(
        X1_grid[::step, ::step], X2_grid[::step, ::step], 
        U[::step, ::step], Vv[::step, ::step], color[::step, ::step],
        cmap='plasma', angles='xy', scale=100, width=0.005
    )
    ax2.set_xlabel(r"$x_1$")
    ax2.set_ylabel(r"$x_2$")
    ax2.grid(True, linestyle='--', alpha=0.5)

    os.makedirs("image", exist_ok=True)
    filename = "image/" + args.filename_prefix + "value_function.svg"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)

    # 画V的等高线图
    fig3 = plt.figure(figsize=(6,5))
    ax3 = fig3.add_subplot(111, projection='3d')
    z_min = min(np.min(sp_values), np.min(sp_values))
    ax3.plot_surface(X1, X2, sp_values, cmap='RdBu_r', alpha=0.8)
    ax3.contour(X1, X2, sp_values, zdir='z', offset=z_min, cmap='RdBu_r', alpha=1, levels=20)
    ax3.set_xlabel(r'$x_1$', labelpad=10)
    ax3.set_ylabel(r'$x_2$', labelpad=10)
    ax3.set_zlabel(r'$V$', labelpad=10, rotation=90)
    filename = "image/" + args.filename_prefix + "sp_value.svg"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)

    # 画V_dot的等高线图
    fig4 = plt.figure(figsize=(6,5))
    ax4 = fig4.add_subplot(111, projection='3d')
    z_min = min(np.min(V_dot_sp), np.min(V_dot_sp))
    ax4.plot_surface(X1, X2, V_dot_sp, cmap='RdBu_r', alpha=0.8)
    ax4.contour(X1, X2, V_dot_sp, zdir='z', offset=z_min, cmap='RdBu_r', alpha=1, levels=20)
    ax4.set_xlabel(r'$x_1$', labelpad=10)
    ax4.set_ylabel(r'$x_2$', labelpad=10)
    ax4.set_zlabel(r'$\dot V$', labelpad=20)
    filename = "image/" + args.filename_prefix + "sp_value_dot.svg"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)

    # 单独画相轨迹图
    fig5 = plt.figure(figsize=(6,5))
    ax5 = fig5.add_subplot(111)
    Q_quiver2 = ax5.quiver(
        X1_grid[::step, ::step], X2_grid[::step, ::step], 
        U[::step, ::step], Vv[::step, ::step], color[::step, ::step],
        cmap='RdBu_r', angles='xy', scale=100, width=0.005
    )
    cb = plt.colorbar(Q_quiver2, ax=ax5, fraction=0.045, pad=0.04)
    cb.set_label(r"Magnitude")
    ax5.set_xlabel(r'$x_1$', labelpad=10)
    ax5.set_ylabel(r'$x_2$', labelpad=10)
    ax5.axis('equal')
    filename = "image/" + args.filename_prefix + "quiver.svg"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)

    # 单独画控制量曲面对比
    u_sp_surface = u_grid.cpu().numpy().reshape(X1.shape)
    u_opt_surface = u_star_opt.cpu().numpy().reshape(X1.shape)
    
    fig6 = plt.figure(figsize=(10, 8))
    ax6 = fig6.add_subplot(111, projection='3d')
    ax6.plot_surface(X1, X2, u_sp_surface, cmap='Blues', alpha=0.85, linewidth=0, antialiased=True)
    ax6.plot_surface(X1, X2, u_opt_surface, cmap='Oranges', alpha=0.47, linewidth=0, antialiased=True)
    ax6.set_xlabel(r'$x_1$', labelpad=10)
    ax6.set_ylabel(r'$x_2$', labelpad=10)
    ax6.set_zlabel(r'$u$', labelpad=10)
    ax6.view_init(elev=10, azim=48, roll=0)
    
    from matplotlib.lines import Line2D
    legend_elements_u = [
        Line2D([0], [0], color=plt.cm.Blues(0.7), lw=2, label='SPNet Control'),
        Line2D([0], [0], color=plt.cm.Oranges(0.7), lw=2, label='Optimal Control')
    ]
    ax6.legend(handles=legend_elements_u, loc='upper right')
    
    filename_u = "image/" + args.filename_prefix + "control_comparison.svg"
    plt.savefig(filename_u, dpi=600, bbox_inches='tight', pad_inches=0.5)

    plt.show()

def simulate_once():
    env = gym.make(Args.env_id)
    obs, _ = env.reset(seed=1318)
    obs = np.array(obs, dtype=np.float32)
    
    model = SPNet(env)
    model_path = "model/" + "sp_nonlinear2d.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=Args.device))
    else:
        print(f"Model file {model_path} not found.")
        return
    model.to(Args.device)
    model.eval()

    T = 200
    state_traj = [obs.copy()]
    control_traj = []
    time_traj = [0.0]
    for t in range(T):
        x_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(Args.device)
        with torch.no_grad():
            u = model(x_tensor).cpu().numpy().squeeze()
        
        obs, _, terminated, truncated, _ = env.step(u)
        obs = np.array(obs, dtype=np.float32)
        state_traj.append(obs.copy())
        control_traj.append(u)
        time_traj.append((t+1)*0.05)

    state_traj = np.array(state_traj)
    control_traj = np.array(control_traj)
    time_traj = np.array(time_traj)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(time_traj, state_traj[:, 0], label='x1')
    axs[0].plot(time_traj, state_traj[:, 1], label='x2')
    axs[0].set_ylabel('State')
    axs[0].set_title('State Trajectory (SPNet)')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(time_traj[:-1], control_traj, label='u', color='tab:orange')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Control')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_sp()
    plot_value()
    simulate_once()
