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

# ================== 图像设置 =======================
mpl.rcParams['font.family'] = 'Times New Roman'  # 默认字体
# mpl.rcParams['font.weight'] = 'bold'  # 加粗
mpl.rcParams['text.usetex'] = False  # 是否使用tex渲染
mpl.rcParams['axes.unicode_minus'] = False  # 解决无法显示负号
mpl.rcParams['xtick.direction'] = 'in'  # x轴刻度线朝内
mpl.rcParams['ytick.direction'] = 'in'  # y轴刻度线朝内
mpl.rcParams['xtick.top'] = True  # 显示上方的坐标轴
mpl.rcParams['ytick.right'] = True  # 显示右侧的坐标轴

mpl.rcParams['legend.frameon'] = False  # legend不显示边框
mpl.rcParams['legend.fontsize'] = 24  # legend默认size

mpl.rcParams['xtick.labelsize'] = 24  # x坐标默认size
mpl.rcParams['ytick.labelsize'] = 24  # y坐标默认size
mpl.rcParams['axes.labelsize'] = 24  # 轴标题默认size
mpl.rcParams['lines.linewidth'] = 2  # 线条宽度
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.linestyle'] = '--'  # 虚线
mpl.rcParams['grid.alpha'] = 0.5       # 透明度
mpl.rcParams['grid.color'] = 'gray'    # 网格颜色

# =================== 参数设置 ======================
class Args:
    # 环境名称
    env_id = "environment:linear2d-v0"
    # 学习率
    learning_rate = 1e-3
    # 总训练次数
    time_steps = 20000
    # 训练批次
    batch_size = 1280
    # 是否使用GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 文件名前缀
    filename_prefix = "sp_linear2d_"


# 动力学相关
def dynamics(X, U):
    # X: (batch_size, 2)
    # U: (batch_size, 1) or (batch_size,)
    # A: 2x2, B: 2x1
    A = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=X.dtype, device=X.device)
    B = torch.tensor([[0.0], [1.0]], dtype=X.dtype, device=X.device)
    # Ensure U is (batch_size, 1)
    if U.dim() == 1:
        U = U.unsqueeze(1)

    fxu = torch.matmul(X, A.T) + torch.matmul(U, B.T)
    return fxu


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SPNet(nn.Module):
    def __init__(self, env):
        self.env = env
        super().__init__()
        # 仅学习从状态 x 到控制量 u 的映射
        self.net = SIREN(in_features=env.observation_space.shape[0], out_features=1, hidden_features=64, hidden_layers=2, norm_scale=3.0)

    def forward(self, x):
        return self.net(x)


def train_sp():
    env = gym.make(Args.env_id)
    model = SPNet(env).to(Args.device)
    optimizer = optim.Adam(model.parameters(), lr=Args.learning_rate)

    # 线性系统最优值函数矩阵 P
    P = np.array([[np.sqrt(3), 1.0], [1.0, np.sqrt(3)]])
    P_torch = torch.tensor(P, dtype=torch.float32, device=Args.device)
    B_torch = torch.tensor([[0.0], [1.0]], dtype=torch.float32, device=Args.device)

    start_time = time.time()
    print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    for step in range(Args.time_steps):
        if time.time() - start_time > 30:
            print(f"Time limit of 30s reached at step {step}. Stopping training.")
            model_dir = "model"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "sp_linear2d.pth")
            torch.save(model.state_dict(), model_path)
            print(f"SP model saved to {model_path}")
            break

        # 采样数据
        low = -3.0
        high = 3.0
        x_batch = np.random.uniform(low, high, (Args.batch_size, env.observation_space.shape[0]))
        X = torch.from_numpy(x_batch).float().to(Args.device)

        # ================== 生成数据 =======================
        # 最优控制量 u* = - R^{-1} B^T P x (R=1) => u* = - B^T P x
        # 注意: torch.matmul(X, P_torch) 等价于 (P x)^T，再乘以 B 等价于 x^T P B
        u_star = -torch.matmul(torch.matmul(X, P_torch), B_torch)

        # ================== 训练最优控制器 ==================
        u_pred = model(X)

        # 监督学习：直接拟合最优控制量
        loss = torch.nn.functional.mse_loss(u_pred, u_star)

        # 更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印损失
        if step % 100 == 0:
            print(f"Step {step+1}/{Args.time_steps}, Loss: {loss.item():.6f}")

        # 保存模型
        if step > 1000 and (step % 5000 == 0 or step == (Args.time_steps-1)):
            model_dir = "model"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "sp_linear2d.pth")
            torch.save(model.state_dict(), model_path)
            print(f"SP model saved to {model_path}")


# ================== 在网格化场景下测试值函数与画图 ==================
def plot_value():
    model_dir = "model"
    model_path = os.path.join(model_dir, "sp_linear2d.pth")
    args = Args()
    env = gym.make(args.env_id)
    model = SPNet(env).to(Args.device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=Args.device))
    else:
        print(f"Model file {model_path} not found. Running with untrained model.")
    model.eval()

    # 构建网格
    x1 = np.linspace(-3, 3, 50)
    x2 = np.linspace(-3, 3, 50)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.stack([X1.ravel(), X2.ravel()], axis=1)

    X_tensor = torch.from_numpy(X_grid).float().to(Args.device)

    # ------------------ 通过网格化测试获取SP值函数 ------------------
    # 由于SP网络只输出 u，其值函数需要通过仿真积分获得 V(x) = \int (x^TQx + u^TRu) dt
    dt = 0.05
    T_steps = 400  # 仿真20秒，足够收敛到原点附近
    V_sp = torch.zeros(X_tensor.shape[0], device=Args.device)
    curr_x = X_tensor.clone()
    
    Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=Args.device)
    R = torch.tensor([[1.0]], device=Args.device)
    
    print("Simulating grid for SP value function...")
    with torch.no_grad():
        for _ in range(T_steps):
            u = model(curr_x)
            # 累加 cost
            cost = torch.sum(curr_x * (curr_x @ Q), dim=1, keepdim=True) + torch.sum(u * (u @ R), dim=1, keepdim=True)
            V_sp += cost.squeeze() * dt
            
            # RK4 积分动力学
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
    
    # ------------------ 计算 SP 值函数的时间导数 \dot{V} ------------------
    # 对仿真得到的值函数网格进行数值求导
    dx1 = x1[1] - x1[0]
    dx2 = x2[1] - x2[0]
    dV_dx2, dV_dx1 = np.gradient(sp_values, dx2, dx1)
    
    # \dot{V} = dV/dx * f(x, u)
    with torch.no_grad():
        u_grid = model(X_tensor)
        fxu_grid = dynamics(X_tensor, u_grid).cpu().numpy()
    
    fxu_x1 = fxu_grid[:, 0].reshape(X1.shape)
    fxu_x2 = fxu_grid[:, 1].reshape(X1.shape)
    V_dot_sp = dV_dx1 * fxu_x1 + dV_dx2 * fxu_x2

    # LQR值函数: V(x) = x^T P x
    P = np.array([[np.sqrt(3), 1.0], [1.0, np.sqrt(3)]])
    lqr_values = np.einsum('ij,jk,ik->i', X_grid, P, X_grid).reshape(X1.shape)
    
    # LQR V_dot
    X_grid_torch = torch.from_numpy(X_grid).float()
    P_torch = torch.from_numpy(P).float()
    gradV_lqr = 2 * torch.matmul(X_grid_torch, P_torch)  # (N,2)
    B_lqr = torch.tensor([[0.0], [1.0]])
    u_star_lqr = -torch.matmul(torch.matmul(X_grid_torch, P_torch), B_lqr)  # (N,1)
    fxu_lqr = dynamics(X_grid_torch, u_star_lqr)
    V_dot_lqr = (gradV_lqr * fxu_lqr).sum(dim=1).reshape(X1.shape).detach().numpy()

    # 绘制对比曲面
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1.4])
    ax = fig.add_subplot(gs[0], projection='3d')

    surf1 = ax.plot_surface(X1, X2, sp_values, cmap='Blues', alpha=0.85, linewidth=0, antialiased=True)
    surf1_1 = ax.plot_surface(X1, X2, V_dot_sp, cmap='Blues', alpha=0.85, linewidth=0, antialiased=True)
    surf2 = ax.plot_surface(X1, X2, lqr_values, cmap='Oranges', alpha=0.55, linewidth=0, antialiased=True)
    surf2_1 = ax.plot_surface(X1, X2, V_dot_lqr, cmap='Oranges', alpha=0.55, linewidth=0, antialiased=True)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=plt.cm.Blues(0.7), lw=2, label='SPNet'),
        Line2D([0], [0], color=plt.cm.Oranges(0.7), lw=2, label='Optimal')
    ]
    ax.legend(handles=legend_elements, ncol=2, loc='upper right')

    ax.set_xlabel(r'$x_1$', labelpad=10)
    ax.set_ylabel(r'$x_2$', labelpad=10)
    ax.set_zlabel(r'$V$ or $\dot V$', labelpad=10)
    ax.view_init(elev=5, azim=30, roll=0)


    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim([-3, 3])
    ax2.set_ylim([-3, 3])
    
    # 相轨迹图
    x1_lin = np.linspace(-3, 3, 25)
    x2_lin = np.linspace(-3, 3, 25)
    X1_grid, X2_grid = np.meshgrid(x1_lin, x2_lin)
    U = np.zeros_like(X1_grid)
    V = np.zeros_like(X2_grid)

    for i in range(X1_grid.shape[0]):
        for j in range(X1_grid.shape[1]):
            x_point = np.array([X1_grid[i, j], X2_grid[i, j]])
            x_tensor = torch.from_numpy(x_point).float().unsqueeze(0).to(Args.device)
            with torch.no_grad():
                u = model(x_tensor).cpu().numpy().squeeze()
            U[i, j] = x_point[1]
            V[i, j] = u

    speed = np.sqrt(U**2 + V**2)
    color = speed
    step = 2
    Q_quiver = ax2.quiver(
        X1_grid[::step, ::step], X2_grid[::step, ::step], 
        U[::step, ::step], V[::step, ::step], color[::step, ::step],
        cmap='plasma', angles='xy', scale=100, width=0.005
    )
    cb = plt.colorbar(Q_quiver, ax=ax2, fraction=0.045, pad=0.04)
    cb.set_label(r"Magnitude")
    
    ax2.set_xlabel(r"$x_1$")
    ax2.set_ylabel(r"$x_2$")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.axis('equal')

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
        U[::step, ::step], V[::step, ::step], color[::step, ::step],
        cmap='RdBu_r', angles='xy', scale=100, width=0.005
    )
    cb = plt.colorbar(Q_quiver2, ax=ax5, fraction=0.045, pad=0.04)
    cb.set_label(r"Magnitude")
    ax5.set_xlabel(r'$x_1$', labelpad=10)
    ax5.set_ylabel(r'$x_2$', labelpad=10)
    ax5.axis('equal')
    filename = "image/" + args.filename_prefix + "quiver.svg"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)

    plt.show()


def simulate_once():
    """
    加载训练好的SP网络和环境，进行一次仿真，并画出状态和控制的时间曲线
    """
    env = gym.make(Args.env_id)
    obs, _ = env.reset(seed=1318)
    obs = np.array(obs, dtype=np.float32)
    
    model = SPNet(env)
    model_path = "model/" + "sp_linear2d.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=Args.device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model file {model_path} not found. Please train the model first.")
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
        time_traj.append((t+1)*env.dt if hasattr(env, "dt") else (t+1)*0.05)

    state_traj = np.array(state_traj)
    control_traj = np.array(control_traj)
    time_traj = np.array(time_traj)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(time_traj, state_traj[:, 0], label='x1')
    axs[0].plot(time_traj, state_traj[:, 1], label='x2')
    axs[0].set_ylabel('State')
    axs[0].set_title('State Trajectory')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(time_traj[:-1], control_traj, label='u', color='tab:orange')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Control')
    axs[1].set_title('Control Input')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = Args()
    # 训练模型请取消注释下面一行
    train_sp()
    plot_value()
    simulate_once()