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
    env_id = "environment:angle3d-v0"
    learning_rate = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    filename_prefix = "sp_angle3d_"
    batch_size = 2048

def dynamics(X, U):
    phi = X[:, 0:1]
    theta = X[:, 1:2]
    p = U[:, 0:1]
    q = U[:, 1:2]
    r = U[:, 2:3]

    cos_theta = torch.cos(theta)
    cos_theta = torch.clamp(cos_theta, min=1e-6)
    tan_theta = torch.tan(theta)
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)

    dphi = p + tan_theta * sin_phi * q + tan_theta * cos_phi * r
    dtheta = cos_phi * q - sin_phi * r
    dpsi = (sin_phi / cos_theta) * q + (cos_phi / cos_theta) * r

    return torch.cat([dphi, dtheta, dpsi], dim=1)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class SPNet(nn.Module):
    def __init__(self, env):
        self.env = env
        super().__init__()
        self.net = SIREN(in_features=3, out_features=3, hidden_features=128, hidden_layers=3, norm_scale=1.0)

    def forward(self, x):
        return self.net(x)

def setup_casadi_mpc():
    opti = ca.Opti()
    N = 30
    dt = 0.1
    x = opti.variable(3, N+1)
    u = opti.variable(3, N)
    x0 = opti.parameter(3, 1)

    opti.subject_to(x[:, 0] == x0)
    cost = 0
    for k in range(N):
        phi = x[0, k]
        theta = x[1, k]
        psi = x[2, k]
        p = u[0, k]
        q = u[1, k]
        r = u[2, k]
        
        cos_theta = ca.cos(theta)
        cos_theta_safe = ca.if_else(ca.fabs(cos_theta) < 1e-6, 1e-6, cos_theta)
        tan_theta = ca.sin(theta) / cos_theta_safe
        sin_phi = ca.sin(phi)
        cos_phi = ca.cos(phi)
        
        dphi = p + tan_theta * sin_phi * q + tan_theta * cos_phi * r
        dtheta = cos_phi * q - sin_phi * r
        dpsi = (sin_phi / cos_theta_safe) * q + (cos_phi / cos_theta_safe) * r
        
        opti.subject_to(x[0, k+1] == phi + dphi * dt)
        opti.subject_to(x[1, k+1] == theta + dtheta * dt)
        opti.subject_to(x[2, k+1] == psi + dpsi * dt)
        
        cost += (phi**2 + theta**2 + psi**2 + p**2 + q**2 + r**2) * dt
    
    cost += (x[0, N]**2 + x[1, N]**2 + x[2, N]**2)
    opti.minimize(cost)
    
    # State and control bounds
    opti.subject_to(opti.bounded(-3.0, x, 3.0))
    opti.subject_to(opti.bounded(-10.0, u, 10.0))

    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes', 'ipopt.max_iter': 500}
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
        x_sample = np.random.uniform(-1.0, 1.0, 3)
        opti.set_value(x0_param, x_sample)
        opti.set_initial(u_var, np.zeros((3, 30)))
        try:
            sol = opti.solve()
            u_opt = sol.value(u_var[:, 0])
            X_data.append(x_sample)
            U_data.append(u_opt)
        except Exception:
            pass 

    X_data = np.array(X_data, dtype=np.float32)
    U_data = np.array(U_data, dtype=np.float32)
    print(f"Generated {len(X_data)} samples in 30 seconds.")
    
    if len(X_data) == 0:
        print("No data generated. Exiting.")
        return

    X_tensor_all = torch.from_numpy(X_data).to(Args.device)
    U_tensor_all = torch.from_numpy(U_data).to(Args.device)

    print("--- Phase 2: Training Network ---")
    train_start = time.time()
    batch_size = min(Args.batch_size, len(X_data))
    
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
    model_path = os.path.join(model_dir, "sp_angle3d.pth")
    torch.save(model.state_dict(), model_path)
    print(f"SP model saved to {model_path}. Total training steps: {step}")


def monte_carlo_simulation():
    args = Args()
    device = args.device
    env = gym.make(args.env_id)

    # Load SPNet model
    sp_model = SPNet(env).to(device)
    sp_path = os.path.join("model", "sp_angle3d.pth")
    if os.path.exists(sp_path):
        sp_model.load_state_dict(torch.load(sp_path, map_location=device))
    else:
        print(f"SPNet model file {sp_path} not found. Running with untrained model.")
    sp_model.eval()

    num_episodes = 50
    horizon = 200

    def sp_action(x_np):
        x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)
        with torch.no_grad():
            u = sp_model(x).squeeze(0).cpu().numpy()
        return u

    def lqr_action(x_np):
        return -x_np

    try:
        from pihnn_experiments_plot.p3_hnn_angle3d import HNN, get_pfpu
        hnn_model = HNN(env).to(device)
        hnn_path = os.path.join("model", "hnn_angle3d.pth")
        if os.path.exists(hnn_path):
            hnn_model.load_state_dict(torch.load(hnn_path, map_location=device))
        hnn_model.eval()

        R_np = env.unwrapped.R if hasattr(env, 'unwrapped') else getattr(env, 'R', np.eye(3))
        R_torch = torch.tensor(R_np, dtype=torch.float32, device=device)
        R_inv = torch.linalg.inv(R_torch)

        def hnn_action(x_np):
            x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)
            with torch.no_grad():
                Lambda = hnn_model.get_lambda(x)
            pfpu = get_pfpu(x)
            pfpu_T = pfpu.transpose(1, 2)
            u_tmp = -0.5 * torch.matmul(pfpu_T, Lambda.unsqueeze(2)).squeeze(-1)
            u = (R_inv @ u_tmp.T).T.squeeze(0)
            return u.detach().cpu().numpy()
        hnn_loaded = True
    except Exception as e:
        print(f"Could not load HNN: {e}")
        hnn_loaded = False

    returns_sp = []
    returns_lqr = []
    returns_hnn = []
    traj_u_sp = []
    traj_u_lqr = []
    traj_u_hnn = []
    traj_x_sp = []
    traj_x_lqr = []
    traj_x_hnn = []

    print("Starting Monte Carlo simulation...")
    for ep in range(num_episodes):
        if (ep+1) % 10 == 0:
            print(f"Episode {ep+1}/{num_episodes}")
            
        seed = int(np.random.randint(0, 2**31 - 1))
        env_sp = gym.make(args.env_id)
        env_lqr = gym.make(args.env_id)
        env_hnn = gym.make(args.env_id)
        obs_sp, _ = env_sp.reset(seed=seed, theoretic_mode=True)
        obs_lqr, _ = env_lqr.reset(seed=seed, theoretic_mode=True)
        obs_hnn, _ = env_hnn.reset(seed=seed, theoretic_mode=True)

        ret_sp = 0.0
        ret_lqr = 0.0
        ret_hnn = 0.0

        u_seq_sp = []
        u_seq_lqr = []
        u_seq_hnn = []

        x_seq_sp = [obs_sp.copy()]
        x_seq_lqr = [obs_lqr.copy()]
        x_seq_hnn = [obs_hnn.copy()]

        for _ in range(horizon):
            u_s = sp_action(obs_sp)
            next_obs_s, r_s, term_s, trunc_s, _ = env_sp.step(u_s.astype(np.float32))
            ret_sp += float(r_s)
            u_seq_sp.append(u_s)
            obs_sp = next_obs_s if not (term_s or trunc_s) else obs_sp
            x_seq_sp.append(obs_sp.copy())

            u_l = lqr_action(obs_lqr)
            next_obs_l, r_l, term_l, trunc_l, _ = env_lqr.step(u_l.astype(np.float32))
            ret_lqr += float(r_l)
            u_seq_lqr.append(u_l)
            obs_lqr = next_obs_l if not (term_l or trunc_l) else obs_lqr
            x_seq_lqr.append(obs_lqr.copy())

            if hnn_loaded:
                u_h = hnn_action(obs_hnn)
                next_obs_h, r_h, term_h, trunc_h, _ = env_hnn.step(u_h.astype(np.float32))
                ret_hnn += float(r_h)
                u_seq_hnn.append(u_h)
                obs_hnn = next_obs_h if not (term_h or trunc_h) else obs_hnn
                x_seq_hnn.append(obs_hnn.copy())

        env_sp.close()
        env_lqr.close()
        if hnn_loaded:
            env_hnn.close()

        traj_u_sp.append(np.stack(u_seq_sp, axis=0))
        traj_u_lqr.append(np.stack(u_seq_lqr, axis=0))
        if hnn_loaded:
            traj_u_hnn.append(np.stack(u_seq_hnn, axis=0))

        traj_x_sp.append(np.stack(x_seq_sp, axis=0))
        traj_x_lqr.append(np.stack(x_seq_lqr, axis=0))
        if hnn_loaded:
            traj_x_hnn.append(np.stack(x_seq_hnn, axis=0))

        returns_sp.append(ret_sp)
        returns_lqr.append(ret_lqr)
        if hnn_loaded:
            returns_hnn.append(ret_hnn)

    traj_u_sp = np.stack(traj_u_sp, axis=0)
    traj_u_lqr = np.stack(traj_u_lqr, axis=0)
    if hnn_loaded:
        traj_u_hnn = np.stack(traj_u_hnn, axis=0)

    traj_x_sp = np.stack(traj_x_sp, axis=0)
    traj_x_lqr = np.stack(traj_x_lqr, axis=0)
    if hnn_loaded:
        traj_x_hnn = np.stack(traj_x_hnn, axis=0)

    def plot_mean_std(ax, data, label, color):
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        t = np.arange(len(mean))
        ax.plot(t, mean, label=label, color=color)
        ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.2)

    def plot_mean_std_x(ax, data, label, color):
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        t = np.arange(len(mean))
        ax.plot(t, mean, label=label, color=color)
        ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.2)

    colors = {"SPNet": "tab:orange", "LQR": "tab:green", "HNN": "tab:blue"}

    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(base_dir, "image")
    os.makedirs(image_dir, exist_ok=True)
    fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    labels_u = ['p', 'q', 'r']
    for i in range(3):
        ax = axs[i]
        plot_mean_std(ax, traj_u_sp[:, :, i], "SPNet", colors["SPNet"])
        plot_mean_std(ax, traj_u_lqr[:, :, i], "LQR", colors["LQR"])
        if hnn_loaded:
            plot_mean_std(ax, traj_u_hnn[:, :, i], "HNN", colors["HNN"])
        ax.set_ylabel(labels_u[i])
        if i == 0:
            ax.legend(loc='upper right')
        ax.grid(True)
    axs[2].set_xlabel("Time step")
    fig.tight_layout()
    plt.savefig(os.path.join(image_dir, args.filename_prefix + "traj_u.svg"), dpi=600, bbox_inches='tight', pad_inches=0.5)

    fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    labels_x = [r'$\phi$', r'$\theta$', r'$\psi$']
    for i in range(3):
        ax = axs[i]
        plot_mean_std_x(ax, traj_x_sp[:, :, i], "SPNet", colors["SPNet"])
        plot_mean_std_x(ax, traj_x_lqr[:, :, i], "LQR", colors["LQR"])
        if hnn_loaded:
            plot_mean_std_x(ax, traj_x_hnn[:, :, i], "HNN", colors["HNN"])
        ax.set_ylabel(labels_x[i])
        if i == 0:
            ax.legend(loc='upper right')
        ax.grid(True)
    axs[2].set_xlabel("Time step")
    fig.tight_layout()
    plt.savefig(os.path.join(image_dir, args.filename_prefix + "traj_x.svg"), dpi=600, bbox_inches='tight', pad_inches=0.5)

    returns_sp = np.array(returns_sp)
    returns_lqr = np.array(returns_lqr)
    if hnn_loaded:
        returns_hnn = np.array(returns_hnn)

    print(f"SPNet: mean={returns_sp.mean():.3f}, std={returns_sp.std():.3f}")
    print(f"LQR: mean={returns_lqr.mean():.3f}, std={returns_lqr.std():.3f}")
    if hnn_loaded:
        print(f"HNN: mean={returns_hnn.mean():.3f}, std={returns_hnn.std():.3f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    if hnn_loaded:
        min_val = min(returns_sp.min(), returns_lqr.min(), returns_hnn.min())
        max_val = max(returns_sp.max(), returns_lqr.max(), returns_hnn.max())
    else:
        min_val = min(returns_sp.min(), returns_lqr.min())
        max_val = max(returns_sp.max(), returns_lqr.max())
    bins = np.linspace(min_val, max_val, 20)
    
    if hnn_loaded:
        plt.hist(returns_hnn, bins=bins, alpha=0.5, label="HNN", color=colors["HNN"])
    plt.hist(returns_sp, bins=bins, alpha=0.5, label="SPNet", color=colors["SPNet"])
    plt.hist(returns_lqr, bins=bins, alpha=0.5, label="LQR", color=colors["LQR"])
    plt.xlabel("Total Return")
    plt.ylabel("Frequency")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, args.filename_prefix + "hist_returns.svg"), dpi=600, bbox_inches='tight', pad_inches=0.5)

    plt.show()

if __name__ == "__main__":
    train_sp()
    monte_carlo_simulation()
