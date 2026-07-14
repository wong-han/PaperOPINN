'''
使用 Furfaro et al. (2022) 论文中的 X-TFC (Extreme Theory of Functional Connections) PINN 方法
解决 3D Quadrotor 受限非线性最优控制问题的 HJB 方程。
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
    env_id = "environment:quadrotor3d_TR_constrained"
    learning_rate = 1e-2  
    time_steps = 80000     
    batch_size = 4096
    hidden_dim = 400      
    device = "cuda" if torch.cuda.is_available() else "cpu"
    filename_prefix = "xtfc_quadrotor3d_constrained_"

_env_tmp = gym.make(Args.env_id)
m = _env_tmp.metadata['m']
g = _env_tmp.metadata['g']

# 动力学相关
def dynamics(X, U):
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
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


def get_penalty(X):
    y = X[:, 1]
    p = 30 * (torch.exp(torch.relu(-y)) - 1)
    return p


# =====================================================================
# X-TFC (Extreme Theory of Functional Connections) PINN 架构
# =====================================================================
class XTFC_PINN(nn.Module):
    def __init__(self, env, hidden_dim=400):
        super().__init__()
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.hidden_dim = hidden_dim

        # 1. ELM 随机初始化隐藏层权重并冻结 (不参与梯度更新)
        # 使用 normal 或 uniform 映射到合适的非线性区间
        self.W = nn.Parameter(torch.randn(hidden_dim, self.state_dim) * 1.5, requires_grad=False)
        self.b = nn.Parameter(torch.randn(hidden_dim) * 1.5, requires_grad=False)

        # 2. 唯一可训练参数：线性输出权重 \beta
        self.beta = nn.Parameter(torch.zeros(hidden_dim, 1, requires_grad=True))

        # 激活函数采用论文经典推荐：Tanh
        self.act = torch.tanh

    def _get_H(self, x):
        """计算 ELM 隐藏层特征矩阵 H(x) = tanh(X W^T + b)"""
        return self.act(torch.matmul(x, self.W.t()) + self.b)

    def get_value(self, x):
        """
        通过 TFC Constraint Expression 解析满足 V(0) = 0:
        V_CE(x) = g(x) - g(0) = (H(x) - H(0)) * \beta
        """
        H_x = self._get_H(x)
        H_0 = self._get_H(torch.zeros_like(x))
        V = torch.matmul(H_x - H_0, self.beta)
        return V

    def get_lambda(self, x):
        """
        计算协状态 \Lambda = \nabla V(x)。
        利用 ELM 结构进行解析快速求导 (极大地节约计算图开销并提升精度)：
        d/dx_i [tanh(Z)] = (1 - tanh^2(Z)) * W[:, i]
        """
        Z = torch.matmul(x, self.W.t()) + self.b
        H = self.act(Z)
        dH = 1.0 - H ** 2  # (batch_size, hidden_dim)

        grads = []
        for i in range(self.state_dim):
            # 对每一维状态求偏导
            dH_dxi = dH * self.W[:, i].unsqueeze(0)  # (batch_size, hidden_dim)
            dV_dxi = torch.matmul(dH_dxi, self.beta) # (batch_size, 1)
            grads.append(dV_dxi)

        Lambda = torch.cat(grads, dim=1)  # (batch_size, state_dim)
        return Lambda

    def get_action(self, X):
        """通过最优控制律公式求解 u^* = -0.5 * R^{-1} * G(X)^T * \Lambda"""
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
        """
        论文 IV.A/IV.B 核心技巧：利用 LQR 二次函数 V_guess = X^T P X 对 \beta 进行 Least-Squares 预训练，
        提供高质量初始猜测，防止 HJB 非线性项导数发散。
        """
        with torch.no_grad():
            A_mat = self.env.A
            B_mat = self.env.B
            Q_mat = self.env.Q
            R_mat = self.env.R
            P_np = solve_continuous_are(A_mat, B_mat, Q_mat, R_mat)
            P_torch = torch.tensor(P_np, dtype=X_sample.dtype, device=X_sample.device)
            # 目标猜测值 V_guess
            V_guess = torch.einsum('bi,ij,bj->b', X_sample, P_torch, X_sample).unsqueeze(1)
            
            # 线性方程组 A * \beta = Y
            H_x = self._get_H(X_sample)
            H_0 = self._get_H(torch.zeros_like(X_sample))
            A = H_x - H_0
            
            # 最小二乘求解 \beta
            beta_init = torch.linalg.lstsq(A, V_guess).solution
            self.beta.data.copy_(beta_init)
            print("X-TFC: Warm-start LQR initialization completed via Least-Squares.")


def get_hjb_residual_loss(model, X):
    """
    计算 HJB PDE 残差损失 (论文 Eq. 7 & Eq. 28)
    H(X, \Lambda) = L(X, u^*) + \Lambda^T f(X, u^*) = 0
    由于 TFC 已经解析满足了 V(0)=0，此处绝对不需要添加边界条件损失！
    """
    Lambda = model.get_lambda(X)
    u_star = model.get_action(X)
    
    Q_torch = torch.tensor(model.env.Q, dtype=X.dtype, device=X.device)
    R_torch = torch.tensor(model.env.R, dtype=X.dtype, device=X.device)
    u0 = torch.tensor(model.env.u0, dtype=X.dtype, device=X.device)
    delta_u = u_star - u0
    
    # 运行成本 L(X, u) = X^T Q X + u^T R u
    xQx = torch.einsum('bi,ij,bj->b', X, Q_torch, X)
    uRu = torch.einsum('bi,ij,bj->b', delta_u, R_torch, delta_u)
    
    # 动力学项 \Lambda^T \dot{X}
    fxu = dynamics(X, u_star)
    lambda_fxu = (Lambda * fxu).sum(dim=1)
    penalty = get_penalty(X)
    
    # H = 0
    H_residual = xQx + uRu + lambda_fxu + penalty
    loss = (H_residual ** 2).mean()
    return loss


def train_xtfc():
    args = Args()
    env = gym.make(args.env_id)
    xtfc = XTFC_PINN(env, hidden_dim=args.hidden_dim).to(args.device)

    # 采样均匀分布的训练配点 (Collocation Points)
    low, high = -1.0, 1.0
    x_batch = np.random.uniform(low, high, (args.batch_size, env.observation_space.shape[0]))
    X_train = torch.from_numpy(x_batch).float().to(args.device)

    # 1. 执行 Least-Squares 热启动
    xtfc.warm_start_lqr(X_train)

    # 2. 使用 L-BFGS (拟牛顿迭代最小二乘法) 或 Adam 优化唯一变量 \beta
    # 论文指出 X-TFC 的参数呈线性化特征，迭代收敛极快
    optimizer = optim.LBFGS([xtfc.beta], lr=1.0, max_iter=20, history_size=50, line_search_fn="strong_wolfe")
    
    print("Starting X-TFC Iterative Least-Squares Training...")
    start_time = time.time()
    
    for step in range(args.time_steps // 20):  
        if time.time() - start_time > 300:
            print(f"Time limit of 300s reached at step {step}. Stopping training.")
            break
            
        # 为防止过拟合固定网格，每步微调重采样一部分点（但在一次 L-BFGS 步长线搜索内必须固定，否则 Hessian 发散）
        x_rand = np.random.uniform(low, high, (args.batch_size, env.observation_space.shape[0]))
        X_curr = torch.from_numpy(x_rand).float().to(args.device)
        def closure():
            optimizer.zero_grad()
            loss = get_hjb_residual_loss(xtfc, X_curr)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        
        if step % 5 == 0:
            print(f"Iter {step*20}/{args.time_steps}, HJB Residual MSE Loss: {loss.item():.2e}")

    # 保存 X-TFC 模型参数
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "xtfc_quadrotor3d_constrained.pth")
    torch.save(xtfc.state_dict(), model_path)
    print(f"X-TFC model successfully saved to {model_path}")


# =====================================================================
# Monte Carlo Simulation (Adapted from p4_chnn_quadrotor3d_TR.py)
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
    xtfc_path = os.path.join("model", "xtfc_quadrotor3d_constrained.pth")
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
        
        obs_lqr, _ = env_lqr.reset(seed=seed, theoretic_mode=True)
        obs_xtfc, _ = env_xtfc.reset(seed=seed, theoretic_mode=True)

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
    
    # 绘图部分使用 CHNN 相同的风格
    control_labels = [r'$T$', r'$p$', r'$q$', r'$r$']
    time_u = np.arange(horizon)

    nature_green = "#389826"   # LQR
    nature_purple = "#4B8BBE"  # X-TFC 
    
    image_dir = "image"
    os.makedirs(image_dir, exist_ok=True)
    
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

    # ========== 箱线图及统计 ==========
    final_dist_lqr = np.array([np.linalg.norm(traj[-1, 0:3]) for traj in all_traj_x_lqr])
    final_dist_xtfc = np.array([np.linalg.norm(traj[-1, 0:3]) for traj in all_traj_x_xtfc])
    success_mask_lqr = final_dist_lqr <= 0.1
    success_mask_xtfc = final_dist_xtfc <= 0.1
    filtered_returns_lqr = np.array(returns_lqr)[success_mask_lqr]
    filtered_returns_xtfc = np.array(returns_xtfc)[success_mask_xtfc]

    min_y_lqr = np.array([np.min(traj[:, 1]) for traj in all_traj_x_lqr])
    min_y_xtfc = np.array([np.min(traj[:, 1]) for traj in all_traj_x_xtfc])
    filtered_min_y_lqr = min_y_lqr[success_mask_lqr]
    filtered_min_y_xtfc = min_y_xtfc[success_mask_xtfc]

    import seaborn as sns
    palette = ["#f6c85f", "#88bde6"]  
    boxprops = dict(linewidth=2, color='k')
    medianprops = dict(linewidth=2, color="#d24d57")
    whiskerprops = dict(linewidth=1.5, color='k')
    capprops = dict(linewidth=1.5, color='k')
    flierprops = dict(marker='o', markerfacecolor='#ff7f0e', markersize=7, linestyle='none', alpha=0.3, markeredgecolor='k')

    fig, ax = plt.subplots(figsize=(5, 8))
    bp = ax.boxplot(
        [-filtered_returns_lqr, -filtered_returns_xtfc],
        labels=[r"LQR" , r"X-TFC"],
        widths=0.5,
        patch_artist=True,
        boxprops=dict(facecolor=palette[0], edgecolor='k', **boxprops),
        medianprops=medianprops,
        whiskerprops=whiskerprops,
        capprops=capprops, 
        flierprops=flierprops,
        showfliers=False
    )
    for patch, color in zip(bp['boxes'], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.38)
    data_pts = [-filtered_returns_lqr, -filtered_returns_xtfc]
    for i, y_data in enumerate(data_pts):
        x_jitter = np.random.normal(0, 0.07, size=len(y_data))
        ax.scatter(np.full(len(y_data), i+1) + x_jitter, y_data, color=palette[i], alpha=0.33, s=50, edgecolor='k', linewidth=0.2, zorder=2)
    ax.set_ylabel(r"$J$")
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')
    ax.grid(True, linestyle='--', alpha=0.45)
    sns.despine(top=False, right=False, left=False, bottom=False, ax=ax)
    filename = "image/" + args.filename_prefix + "return_boxplot.svg"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)

    import pandas as pd
    violin_palette = [nature_green, nature_purple]
    eps = 1e-8 
    log_abs_ymin_lqr = np.log10(np.abs(filtered_min_y_lqr) + eps)
    log_abs_ymin_xtfc = np.log10(np.abs(filtered_min_y_xtfc) + eps)
    df_logymin = pd.DataFrame({
        r"$\log_{10}|y_{\mathrm{min}}|$": np.concatenate([log_abs_ymin_lqr, log_abs_ymin_xtfc]),
        "Controller": ["LQR"] * len(log_abs_ymin_lqr) + ["X-TFC"] * len(log_abs_ymin_xtfc)
    })
    fig2, ax2 = plt.subplots(figsize=(5, 7))
    sns.violinplot(
        data=df_logymin, x="Controller", y=r"$\log_{10}|y_{\mathrm{min}}|$",
        palette=violin_palette, inner=None, linewidth=0, alpha=0.3, ax=ax2
    )
    sns.boxplot(
        data=df_logymin, x="Controller", y=r"$\log_{10}|y_{\mathrm{min}}|$",
        width=0.2, boxprops={'zorder': 2, 'facecolor': 'none', 'edgecolor': 'k', 'linewidth': 1.5},
        whiskerprops={'linewidth': 1.5, 'color': 'k'},
        capprops={'linewidth': 1.5, 'color': 'k'},
        medianprops={'linewidth': 2, 'color': '#d24d57'},
        showfliers=False, ax=ax2
    )
    sns.stripplot(
        data=df_logymin, x="Controller", y=r"$\log_{10}|y_{\mathrm{min}}|$",
        palette=violin_palette, alpha=0.4, size=6, jitter=0.15, edgecolor='k', linewidth=0.5, zorder=3, ax=ax2
    )
    ax2.axhline(np.log10(1e-2), color='gray', linestyle='--', linewidth=1.5, zorder=0)
    ax2.set_xlabel("")
    ax2.set_ylabel(r"$\log_{10}|y_{\mathrm{min}}|$")
    ax2.tick_params(axis='x')
    ax2.grid(True, linestyle='--', alpha=0.45)
    sns.despine(top=False, right=False, left=False, bottom=False, ax=ax2)
    filename2 = "image/" + args.filename_prefix + "ymin_violinplot.svg"
    plt.savefig(filename2, dpi=600, bbox_inches='tight', pad_inches=0.5)
    plt.show()

if __name__ == "__main__":
    train_xtfc()
    monte_carlo_simulation()