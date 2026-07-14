'''
使用 Furfaro et al. (2022) 论文中的 X-TFC (Extreme Theory of Functional Connections) PINN 方法
解决 3D Quadrotor 非线性最优控制问题的 HJB 方程。
（已修复 L-BFGS 闭包随机性、激活饱和与采样空间问题）
'''
import sys, os
import time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import gymnasium as gym
from scipy.linalg import solve_continuous_are
from utils import DroneVisualizer
import pandas as pd
import seaborn as sns

# ================== 图像设置 =======================
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['text.usetex'] = False
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['axes.titlesize'] = 22
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.alpha'] = 0.5
mpl.rcParams['grid.color'] = 'gray'

# =================== 参数设置 ======================
class Args:
    env_id = "environment:quadrotor3d_TR-v0"
    learning_rate = 1.0    # L-BFGS 默认学习率
    epochs = 2000          # 训练 Epoch 数（每次重采样一个新的Batch）
    batch_size = 1280      # 增大 Batch Size 覆盖 9 维空间
    hidden_dim = 500      # 增加隐藏层节点容量，提升高维函数拟合能力
    device = "cuda" if torch.cuda.is_available() else "cpu"
    filename_prefix = "xtfc_quadrotor3d_"

_env_tmp = gym.make(Args.env_id)
m = _env_tmp.metadata['m']
g = _env_tmp.metadata['g']

# 动力学相关
def dynamics(X, U):
    vx = X[:, 3]
    vy = X[:, 4]
    vz = X[:, 5]
    phi = X[:, 6]
    theta = X[:, 7]
    psi = X[:, 8]

    T = U[:, 0]
    p = U[:, 1]
    q = U[:, 2]
    r = U[:, 3]

    N = X.shape[0]
    fxu = torch.zeros(N, X.shape[1], dtype=X.dtype, device=X.device)

    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    tan_theta = torch.tan(theta)
    sin_psi = torch.sin(psi)
    cos_psi = torch.cos(psi)

    fxu[:, 0] += vx
    fxu[:, 1] += vy
    fxu[:, 2] += vz
    fxu[:, 3] += T * (cos_phi * sin_theta * cos_psi + sin_phi * sin_psi) / m
    fxu[:, 4] += T * (cos_phi * sin_theta * sin_psi - sin_phi * cos_psi) / m
    fxu[:, 5] += T * (cos_phi * cos_theta) / m - g
    fxu[:, 6] += p + sin_phi * tan_theta * q + cos_phi * tan_theta * r
    fxu[:, 7] += cos_phi * q - sin_phi * r
    fxu[:, 8] += sin_phi / cos_theta * q + cos_phi / cos_theta * r

    return fxu


def get_pfpu(X):
    phi = X[:, 6]
    theta = X[:, 7]
    psi = X[:, 8]

    N = X.shape[0]
    pfpu = torch.zeros(N, 9, 4, dtype=X.dtype, device=X.device)

    pfpu[:, 3, 0] += (torch.sin(phi)*torch.sin(psi) + torch.sin(theta)*torch.cos(phi)*torch.cos(psi))/m
    pfpu[:, 4, 0] += (-torch.sin(phi)*torch.cos(psi) + torch.sin(psi)*torch.sin(theta)*torch.cos(phi))/m
    pfpu[:, 5, 0] += torch.cos(phi)*torch.cos(theta)/m
    pfpu[:, 6, 1] += 1
    pfpu[:, 6, 2] += torch.sin(phi) * torch.tan(theta)
    pfpu[:, 6, 3] += torch.cos(phi) * torch.tan(theta)
    pfpu[:, 7, 2] += torch.cos(phi)
    pfpu[:, 7, 3] += -torch.sin(phi)
    pfpu[:, 8, 2] += torch.sin(phi) / torch.cos(theta)
    pfpu[:, 8, 3] += torch.cos(phi) / torch.cos(theta)

    return pfpu


# =====================================================================
# X-TFC (Extreme Theory of Functional Connections) PINN 架构
# =====================================================================
class XTFC_PINN(nn.Module):
    def __init__(self, env, hidden_dim=1000):
        super().__init__()
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.hidden_dim = hidden_dim

        # 【修改点 1】：缩放系数从 1.5 降至 0.6，防止 tanh 激活函数进入饱和区导致 dH -> 0
        self.W = nn.Parameter(torch.randn(hidden_dim, self.state_dim) * 0.6, requires_grad=False)
        self.b = nn.Parameter(torch.randn(hidden_dim) * 0.6, requires_grad=False)
        self.beta = nn.Parameter(torch.zeros(hidden_dim, 1, requires_grad=True))
        self.act = torch.tanh

    def _get_H(self, x):
        return self.act(torch.matmul(x, self.W.t()) + self.b)

    def get_value(self, x):
        H_x = self._get_H(x)
        H_0 = self._get_H(torch.zeros_like(x))
        V = torch.matmul(H_x - H_0, self.beta)
        return V

    def get_lambda(self, x):
        Z = torch.matmul(x, self.W.t()) + self.b
        H = self.act(Z)
        dH = 1.0 - H ** 2 

        grads = []
        for i in range(self.state_dim):
            dH_dxi = dH * self.W[:, i].unsqueeze(0) 
            dV_dxi = torch.matmul(dH_dxi, self.beta)
            grads.append(dV_dxi)

        Lambda = torch.cat(grads, dim=1) 
        return Lambda

    def get_action(self, X):
        R = self.env.R
        u0 = torch.tensor(self.env.u0, dtype=X.dtype, device=X.device)
        Lambda = self.get_lambda(X).unsqueeze(2)  
        pfpu = get_pfpu(X)                        
        
        R_inv = torch.tensor(np.linalg.inv(R), dtype=X.dtype, device=X.device)
        pfpu_T = pfpu.transpose(1, 2)
        
        u_star = -0.5 * torch.matmul(pfpu_T, Lambda) 
        u_star = torch.matmul(R_inv, u_star.squeeze(-1).T).T + u0
        return u_star

    def warm_start_lqr(self, X_sample):
        with torch.no_grad():
            A_mat = self.env.A
            B_mat = self.env.B
            Q_mat = self.env.Q
            R_mat = self.env.R
            P_np = solve_continuous_are(A_mat, B_mat, Q_mat, R_mat)
            P_torch = torch.tensor(P_np, dtype=X_sample.dtype, device=X_sample.device)
            
            V_guess = torch.einsum('bi,ij,bj->b', X_sample, P_torch, X_sample).unsqueeze(1)
            
            H_x = self._get_H(X_sample)
            H_0 = self._get_H(torch.zeros_like(X_sample))
            A = H_x - H_0
            
            beta_init = torch.linalg.lstsq(A, V_guess).solution
            self.beta.data.copy_(beta_init)
            print("X-TFC: Warm-start LQR initialization completed via Least-Squares.")


def get_hjb_residual_loss(model, X):
    Lambda = model.get_lambda(X)
    u_star = model.get_action(X)
    
    Q_torch = torch.tensor(model.env.Q, dtype=X.dtype, device=X.device)
    R_torch = torch.tensor(model.env.R, dtype=X.dtype, device=X.device)
    u0 = torch.tensor(model.env.u0, dtype=X.dtype, device=X.device)
    delta_u = u_star - u0
    
    xQx = torch.einsum('bi,ij,bj->b', X, Q_torch, X)
    uRu = torch.einsum('bi,ij,bj->b', delta_u, R_torch, delta_u)
    
    fxu = dynamics(X, u_star)
    lambda_fxu = (Lambda * fxu).sum(dim=1)
    
    H_residual = xQx + uRu + lambda_fxu
    loss = (H_residual ** 2).mean()
    return loss


def sample_training_data(batch_size, state_dim, low=-1.0, high=1.0):
    """
    【修改点 2】：混合采样策略
    一半数据进行均匀全局探索，另一半数据围绕原点（平衡位置）高斯分布采样。
    强化重点区域（控制目标点附近）的拟合精度。
    """
    half_bs = batch_size // 2
    # 均匀分布采样
    x_uniform = np.random.uniform(low, high, (half_bs, state_dim))
    # 正态分布采样（集中在 0 附近，标准差设为 0.35）
    x_normal = np.random.normal(loc=0.0, scale=0.35, size=(batch_size - half_bs, state_dim))
    x_normal = np.clip(x_normal, low, high)
    
    x_batch = np.vstack([x_uniform, x_normal])
    return x_batch


def train_xtfc():
    args = Args()
    env = gym.make(args.env_id)
    state_dim = env.observation_space.shape[0]
    xtfc = XTFC_PINN(env, hidden_dim=args.hidden_dim).to(args.device)

    low, high = -1.0, 1.0
    
    # 初始 LQR Warm Start
    x_init = sample_training_data(args.batch_size * 2, state_dim, low, high)
    X_train_init = torch.from_numpy(x_init).float().to(args.device)
    xtfc.warm_start_lqr(X_train_init)

    # L-BFGS 优化器
    optimizer = optim.LBFGS(
        [xtfc.beta], 
        lr=args.learning_rate, 
        max_iter=20,          # 单次 step 最多执行的迭代数
        max_eval=25, 
        history_size=50, 
        line_search_fn="strong_wolfe"
    )
    
    print("Starting X-TFC Iterative Least-Squares Training...")
    start_time = time.time()
    
    for epoch in range(args.epochs):  
        if time.time() - start_time > 300:
            print(f"Time limit of 300s reached at epoch {epoch}. Stopping training.")
            break
            
        # 【修改点 3】：在 closure 外部生成当前 Epoch 的固定数据集！
        # L-BFGS 计算二阶导数近似时，必须保证目标函数在闭包执行期间保持绝对静态。
        x_rand = sample_training_data(args.batch_size, state_dim, low, high)
        X_curr = torch.from_numpy(x_rand).float().to(args.device)

        def closure():
            optimizer.zero_grad()
            loss = get_hjb_residual_loss(xtfc, X_curr)
            loss.backward()
            return loss

        # 执行固定 Batch 下的拟牛顿法优化
        loss = optimizer.step(closure)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{args.epochs}, HJB Residual MSE Loss: {loss.item():.2e}")

    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "xtfc_quadrotor3d.pth")
    torch.save(xtfc.state_dict(), model_path)
    print(f"X-TFC model successfully saved to {model_path}")


def simulate_once():
    # 仿真LQR和X-TFC PINN控制效果
    args = Args()
    device = args.device
    env_probe = gym.make(args.env_id, theoretic_mode=True)

    # 仿真参数
    horizon = int(env_probe.T / env_probe.dt)
    A_np = env_probe.A
    B_np = env_probe.B
    Q_np = env_probe.Q
    R_np = env_probe.R
    u0 = env_probe.u0

    # 计算LQR控制器
    P = solve_continuous_are(A_np, B_np, Q_np, R_np)
    K = np.linalg.inv(R_np) @ B_np.T @ P
    def lqr_action(x_np):
        u = -K @ x_np + u0
        u = np.clip(u, env_probe.action_space.low, env_probe.action_space.high)
        return np.array(u, dtype=np.float32)

    # 加载X-TFC模型
    xtfc_path = "model/xtfc_quadrotor3d.pth"
    env_xtfc = gym.make(args.env_id, theoretic_mode=True)
    xtfc_model = XTFC_PINN(env_xtfc, hidden_dim=args.hidden_dim).to(device)
    xtfc_model.load_state_dict(torch.load(xtfc_path, map_location=device))
    xtfc_model.eval()

    # X-TFC控制器
    def xtfc_action(x_np):
        x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)
        with torch.no_grad():
            u = xtfc_model.get_action(x).squeeze(0).cpu().numpy()
        u = np.clip(u, env_probe.action_space.low, env_probe.action_space.high)
        return np.array(u, dtype=np.float32)

    # 初始化环境
    seed = np.random.randint(0, 10000000)
    env_lqr = gym.make(args.env_id, theoretic_mode=True)
    obs_lqr, _ = env_lqr.reset(seed=seed)
    env_xtfc_sim = gym.make(args.env_id, theoretic_mode=True)
    obs_xtfc, _ = env_xtfc_sim.reset(seed=seed)  # 用同一个初始条件

    ret_lqr = 0.0
    u_seq_lqr = []
    x_seq_lqr = [obs_lqr.copy()]

    ret_xtfc = 0.0
    u_seq_xtfc = []
    x_seq_xtfc = [obs_xtfc.copy()]

    lqr_trunc_count = 0
    xtfc_trunc_count = 0
    for t in range(horizon):
        # LQR
        u_l = lqr_action(obs_lqr.astype(np.float32))
        next_obs_l, r_l, term_l, trunc_l, _ = env_lqr.step(u_l)
        ret_lqr += float(r_l)
        u_seq_lqr.append(u_l)
        obs_lqr = next_obs_l
        x_seq_lqr.append(obs_lqr.copy())
        if not(term_l or trunc_l):
            lqr_trunc_count += 1
        # X-TFC
        u_x = xtfc_action(obs_xtfc.astype(np.float32))
        next_obs_x, r_x, term_x, trunc_x, _ = env_xtfc_sim.step(u_x)
        ret_xtfc += float(r_x)
        u_seq_xtfc.append(u_x)
        obs_xtfc = next_obs_x
        x_seq_xtfc.append(obs_xtfc.copy())
        if not(term_x or trunc_x):
            xtfc_trunc_count +=1

    env_lqr.close()
    env_xtfc_sim.close()

    traj_u_lqr = np.vstack(u_seq_lqr)
    traj_x_lqr = np.vstack(x_seq_lqr)
    traj_u_xtfc = np.vstack(u_seq_xtfc)
    traj_x_xtfc = np.vstack(x_seq_xtfc)

    # 绘制控制输入轨迹
    time_u = np.arange(horizon)
    plt.figure(figsize=(12, 6))
    for i in range(4):
        plt.plot(time_u[:], traj_u_lqr[:, i], label=f"LQR u{i+1}", linestyle='--')
        plt.plot(time_u[:], traj_u_xtfc[:, i], label=f"X-TFC u{i+1}")
    plt.xlabel("Time step")
    plt.ylabel("u")
    plt.title("Control trajectories (single episode)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    # 绘制状态轨迹
    time_x = np.arange(horizon + 1)
    fig, axs = plt.subplots(9, 1, figsize=(12, 14), sharex=True)
    for i, ax in enumerate(axs):
        ax.plot(time_x[:], traj_x_lqr[:, i], label="LQR", color="tab:green")
        ax.plot(time_x[:], traj_x_xtfc[:, i], label="X-TFC", color="tab:blue")
        ax.set_ylabel(f"x{i+1}")
        ax.grid(True, linestyle='--', alpha=0.4)
        if i == 0:
            ax.legend()
    axs[-1].set_xlabel("Time step")
    axs[2].set_ylim([-1, 1])
    fig.suptitle("State trajectories (single episode)")
    plt.tight_layout()

    # 打印回报
    print("Single episode return:")
    print(f"LQR: {ret_lqr:.3f}")
    print(f"X-TFC: {ret_xtfc:.3f}")

    os.makedirs('image', exist_ok=True)
    # 绘制无人机三维轨迹
    try:
        import cmaps
        cmap_xtfc = cmaps.MPL_PuOr_r
        cmap_lqr = cmaps.MPL_RdYlGn_r
        drone_visualizer = DroneVisualizer(traj=traj_x_xtfc[:, 0:3], alpha_range=[0.2, 1])
        drone_visualizer.add_one_traj(traj=traj_x_xtfc[:xtfc_trunc_count, 0:3], attitudes=np.rad2deg(traj_x_xtfc[:xtfc_trunc_count, 6:9]), stride=50, colormap=cmap_xtfc) 
        drone_visualizer.add_one_traj(traj=traj_x_lqr[:lqr_trunc_count, 0:3], attitudes=np.rad2deg(traj_x_lqr[:lqr_trunc_count, 6:9]), stride=50, colormap=cmap_lqr)
        drone_visualizer.ax.set_zlim([-1, 1])
        plt.savefig('image/quad_traj_single_xtfc.svg', dpi=600, bbox_inches='tight', pad_inches=0.5, transparent=True)
    except Exception as e:
        print(f"跳过 3D 轨迹绘制: {e}")

    # 控制量轨迹
    control_labels = ['$T$\,(N)', '$p$\,(rad/s)', '$q$\,(rad/s)', '$r$\,(rad/s)']
    nature_green = "#389826"   # LQR
    nature_purple = "#4B8BBE"  # X-TFC
    fig, axs = plt.subplots(4, 1, figsize=(6, 6.3), sharex=True)
    for i, ax in enumerate(axs):
        ax.plot(time_u[:], traj_u_lqr[:, i], color=nature_green, alpha=0.6, linewidth=3, label="LQR" if i==0 else "")
        ax.plot(time_u[:], traj_u_xtfc[:, i], color=nature_purple, alpha=0.8, linewidth=3, label="X-TFC" if i==0 else "")
        ax.set_ylabel(control_labels[i])
        ax.grid(True, linestyle='--', alpha=0.36)
    axs[-1].set_xlabel("$t$ (s)")
    plt.tight_layout()
    filename = "image/" + args.filename_prefix + "control_traj_single_v2.svg"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5, transparent=True)

    plt.show()

# =====================================================================
# Monte Carlo Simulation
# =====================================================================
def monte_carlo_simulation():
    np.random.seed(42)
    args = Args()
    device = args.device
    num_episodes = 50

    env_lqr = gym.make(args.env_id, theoretic_mode=True)
    env_xtfc = gym.make(args.env_id, theoretic_mode=True)
    horizon = int(env_lqr.T / env_lqr.dt)

    A = env_lqr.A
    B = env_lqr.B
    Q = env_lqr.Q
    R = env_lqr.R
    u0 = env_lqr.u0

    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P

    xtfc_model = XTFC_PINN(env_xtfc, hidden_dim=args.hidden_dim).to(device)
    xtfc_path = os.path.join("model", "xtfc_quadrotor3d.pth")
    xtfc_model.load_state_dict(torch.load(xtfc_path, map_location=device))
    xtfc_model.eval()

    def xtfc_action(x_np):
        x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)
        with torch.no_grad():
            u = xtfc_model.get_action(x).squeeze(0).cpu().numpy()
        return u

    all_traj_x_lqr = []
    all_traj_u_lqr = []
    all_traj_x_xtfc = []
    all_traj_u_xtfc = []
    returns_lqr = []
    returns_xtfc = []

    print("Starting Monte Carlo simulation...")
    for ep in range(num_episodes):
        if (ep+1) % 10 == 0:
            print(f"Episode {ep+1}/{num_episodes}")
            
        seed = np.random.randint(0, 1000000)
        
        obs_lqr, _ = env_lqr.reset(seed=seed)
        obs_xtfc, _ = env_xtfc.reset(seed=seed)

        x_seq_lqr = [obs_lqr.copy()]
        u_seq_lqr = []
        x_seq_xtfc = [obs_xtfc.copy()]
        u_seq_xtfc = []

        ret_lqr = 0.0
        ret_xtfc = 0.0

        # LQR episode
        obs_lqr_ep = obs_lqr.copy()
        for t in range(horizon):
            u_l = -K @ obs_lqr_ep + u0
            u_l = np.clip(u_l, env_lqr.action_space.low, env_lqr.action_space.high)
            next_obs_l, reward_l, terminated_l, truncated_l, _ = env_lqr.step(u_l)
            u_seq_lqr.append(u_l)
            obs_lqr_ep = next_obs_l
            x_seq_lqr.append(obs_lqr_ep.copy())
            ret_lqr += float(reward_l)

        # X-TFC PINN episode
        obs_xtfc_ep = obs_xtfc.copy()
        for t in range(horizon):
            u_x = xtfc_action(obs_xtfc_ep)
            u_x = np.clip(u_x, env_xtfc.action_space.low, env_xtfc.action_space.high)
            next_obs_x, reward_x, terminated_x, truncated_x, _ = env_xtfc.step(u_x)
            u_seq_xtfc.append(u_x)
            obs_xtfc_ep = next_obs_x
            x_seq_xtfc.append(obs_xtfc_ep.copy())
            ret_xtfc += float(reward_x)

        all_traj_x_lqr.append(np.vstack(x_seq_lqr))
        all_traj_u_lqr.append(np.vstack(u_seq_lqr))
        returns_lqr.append(ret_lqr)

        all_traj_x_xtfc.append(np.vstack(x_seq_xtfc))
        all_traj_u_xtfc.append(np.vstack(u_seq_xtfc))
        returns_xtfc.append(ret_xtfc)

    env_lqr.close()
    env_xtfc.close()
    
    control_labels = [r'$T$', r'$p$', r'$q$', r'$r$']
    time_u = np.arange(horizon)

    nature_green = "#389826"   # LQR
    nature_purple = "#4B8BBE"  # X-TFC 
    
    image_dir = "image"
    os.makedirs(image_dir, exist_ok=True)
    
    # 画所有回合的控制曲线叠加图
    fig, axs = plt.subplots(4, 1, figsize=(6, 6.3), sharex=True)
    for i, ax in enumerate(axs):
        for ep in range(num_episodes):
            ax.plot(time_u*env_lqr.dt, all_traj_u_lqr[ep][:, i], color=nature_green, alpha=0.22, linewidth=1)
            ax.plot(time_u*env_xtfc.dt, all_traj_u_xtfc[ep][:, i], color=nature_purple, alpha=0.22, linewidth=1)
        mean_lqr = np.stack([all_traj_u_lqr[ep][:, i] for ep in range(num_episodes)]).mean(axis=0)
        mean_xtfc = np.stack([all_traj_u_xtfc[ep][:, i] for ep in range(num_episodes)]).mean(axis=0)
        ax.plot(time_u*env_lqr.dt, mean_lqr, color=nature_green, linewidth=2.6, label="LQR" if i==0 else "")
        ax.plot(time_u*env_xtfc.dt, mean_xtfc, color=nature_purple, linewidth=2.6, label="X-TFC" if i==0 else "")
        ax.set_ylabel(control_labels[i])
        ax.grid(True, linestyle='--', alpha=0.36)
    axs[-1].set_xlabel("$t$ (s)")
    axs[0].legend(loc="best")
    plt.tight_layout()
    filename = "image/" + args.filename_prefix + "control_traj.svg"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)
    
    # 画控制曲线单回合(第一回合)
    fig, axs = plt.subplots(4, 1, figsize=(6, 6.3), sharex=True)
    for i, ax in enumerate(axs):
        ax.plot(time_u*env_lqr.dt, all_traj_u_lqr[0][:, i], color=nature_green, alpha=0.8, linewidth=3, linestyle='--', label="LQR" if i==0 else "")
        ax.plot(time_u*env_xtfc.dt, all_traj_u_xtfc[0][:, i], color=nature_purple, alpha=0.8, linewidth=3, label="X-TFC" if i==0 else "")
        ax.set_ylabel(control_labels[i])
        ax.grid(True, linestyle='--', alpha=0.36)
    axs[-1].set_xlabel("$t$ (s)")
    plt.tight_layout()
    filename = "image/" + args.filename_prefix + "control_traj_single.svg"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5, transparent=True)

    # ========== 箱线图及小提琴图统计 ==========
    final_dist_lqr = np.array([np.linalg.norm(traj[-1, 0:3]) for traj in all_traj_x_lqr])
    final_dist_xtfc = np.array([np.linalg.norm(traj[-1, 0:3]) for traj in all_traj_x_xtfc])
    success_mask_lqr = final_dist_lqr <= 0.1
    success_mask_xtfc = final_dist_xtfc <= 0.1
    
    filtered_returns_lqr = np.array(returns_lqr)[success_mask_lqr]
    filtered_returns_xtfc = np.array(returns_xtfc)[success_mask_xtfc]

    df_box = pd.DataFrame({
        "Return": np.concatenate((-filtered_returns_lqr, -filtered_returns_xtfc)),
        "Controller": ["LQR"]*len(filtered_returns_lqr) + ["X-TFC"]*len(filtered_returns_xtfc)
    })

    palette = [nature_green, nature_purple]

    fig, ax = plt.subplots(figsize=(5, 7))
    sns.violinplot(
        x="Controller", y="Return", data=df_box,
        order=["LQR", "X-TFC"],
        palette=palette, inner=None, alpha=0.22, cut=0,
        linewidth=1.2, ax=ax
    )
    sns.boxplot(
        x="Controller", y="Return", data=df_box,
        order=["LQR", "X-TFC"],
        palette=palette, width=0.25,
        showcaps=True, showbox=True, showfliers=False, medianprops=dict(color="#6D2D2B", linewidth=2),
        boxprops=dict(alpha=0.7, edgecolor='k', linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        ax=ax
    )
    sns.stripplot(
        x="Controller", y="Return", data=df_box,
        order=["LQR", "X-TFC"], 
        palette=palette,
        size=7, alpha=0.4, jitter=0.22, linewidth=0.2, edgecolor='#444', ax=ax
    )
    ax.set_xticklabels(["LQR", "X-TFC"])
    ax.set_ylabel(r"$J$")
    ax.grid(True, linestyle='--', alpha=0.45)
    sns.despine(top=False, right=False, left=False, bottom=False, ax=ax)
    filename = "image/" + args.filename_prefix + "return_boxplot.svg"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)
   
    print("Monte Carlo 50 episodes return statistics:")
    print(f"LQR: mean={np.mean(returns_lqr):.3f}, std={np.std(returns_lqr):.3f}, min={np.min(returns_lqr):.3f}, max={np.max(returns_lqr):.3f}")
    print(f"X-TFC: mean={np.mean(returns_xtfc):.3f}, std={np.std(returns_xtfc):.3f}, min={np.min(returns_xtfc):.3f}, max={np.max(returns_xtfc):.3f}")

    plt.show()


if __name__ == "__main__":
    train_xtfc()
    simulate_once()
    monte_carlo_simulation()