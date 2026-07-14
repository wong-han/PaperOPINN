'''
使用 Furfaro et al. (2022) 论文中的 X-TFC (Extreme Theory of Functional Connections) PINN 方法
解决 3D 姿态角非线性最优控制问题的 HJB 方程。
核心改动：
1. 使用 ELM (随机固定隐藏层参数，仅训练输出权重 \beta)。
2. 构造 Constraint Expression (CE): V(x) = g(x) - g(0)，解析满足 V(0) = 0 边界条件。
3. 采用二次 LQR 函数进行 \beta 的 Least-Squares 热启动初始化，避免非线性 HJB 梯度发散。
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
from sup_angle3d import PINNNetwork


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
    env_id = "environment:angle3d-v0"
    learning_rate = 1e-2  # X-TFC 使用 L-BFGS 或快速 Adam，学习率可放大
    time_steps = 10000     # ELM 收敛极大加速，无需 10000 步
    batch_size = 1280     # 配点数量 (Collocation points)
    hidden_dim = 400      # 论文推荐 X-TFC 隐藏节点数 100~800
    device = "cuda" if torch.cuda.is_available() else "cpu"
    filename_prefix = "xtfc_angle3d_"


# 动力学相关
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

    fxu = torch.cat([dphi, dtheta, dpsi], dim=1)
    return fxu


def get_pfpu(X):
    phi = X[:, 0]
    theta = X[:, 1]
    N = X.shape[0]
    pfpu = torch.zeros(N, 3, 3, dtype=X.dtype, device=X.device)

    pfpu[:, 0, 0] = 1.0
    pfpu[:, 0, 1] = torch.tan(theta) * torch.sin(phi)
    pfpu[:, 0, 2] = torch.tan(theta) * torch.cos(phi)

    pfpu[:, 1, 0] = 0.0
    pfpu[:, 1, 1] = torch.cos(phi)
    pfpu[:, 1, 2] = -torch.sin(phi)

    cos_theta = torch.cos(theta)
    pfpu[:, 2, 0] = 0.0
    pfpu[:, 2, 1] = torch.sin(phi) / cos_theta
    pfpu[:, 2, 2] = torch.cos(phi) / cos_theta

    return pfpu


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
        Lambda = self.get_lambda(X).unsqueeze(2)  # (batch_size, 3, 1)
        pfpu = get_pfpu(X)                        # (batch_size, 3, 3)
        
        R_inv = torch.tensor(np.linalg.inv(R), dtype=X.dtype, device=X.device)
        pfpu_T = pfpu.transpose(1, 2)
        
        u_star = -0.5 * torch.matmul(pfpu_T, Lambda) # (batch_size, 3, 1)
        u_star = torch.matmul(R_inv, u_star.squeeze(-1).T).T  # (batch_size, 3)
        return u_star

    def warm_start_lqr(self, X_sample):
        """
        论文 IV.A/IV.B 核心技巧：利用 LQR 二次函数 V_guess = X^T Q X 对 \beta 进行 Least-Squares 预训练，
        提供高质量初始猜测，防止 HJB 非线性项导数发散。
        """
        with torch.no_grad():
            Q_np = self.env.Q
            Q_torch = torch.tensor(Q_np, dtype=X_sample.dtype, device=X_sample.device)
            # 目标猜测值 V_guess
            V_guess = torch.einsum('bi,ij,bj->b', X_sample, Q_torch, X_sample).unsqueeze(1)
            
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
    
    # 运行成本 L(X, u) = X^T Q X + u^T R u
    xQx = torch.einsum('bi,ij,bj->b', X, Q_torch, X)
    uRu = torch.einsum('bi,ij,bj->b', u_star, R_torch, u_star)
    
    # 动力学项 \Lambda^T \dot{X}
    fxu = dynamics(X, u_star)
    lambda_fxu = (Lambda * fxu).sum(dim=1)
    
    # HJB 残差 H = 0
    H_residual = xQx + uRu + lambda_fxu
    
    # X-TFC 损失函数只需最小化 HJB 均方残差
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
    
    for step in range(args.time_steps // 20):  # L-BFGS 每步包含多次内部迭代
        if time.time() - start_time > 30:
            print(f"Time limit of 30s reached at step {step}. Stopping training.")
            break
            
        def closure():
            optimizer.zero_grad()
            # 为防止过拟合固定网格，每步微调重采样一部分点
            x_rand = np.random.uniform(low, high, (args.batch_size, env.observation_space.shape[0]))
            X_curr = torch.from_numpy(x_rand).float().to(args.device)
            loss = get_hjb_residual_loss(xtfc, X_curr)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        
        if step % 5 == 0:
            print(f"Iter {step*20}/{args.time_steps}, HJB Residual MSE Loss: {loss.item():.2e}")

    # 保存 X-TFC 模型参数
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "xtfc_angle3d.pth")
    torch.save(xtfc.state_dict(), model_path)
    print(f"X-TFC model successfully saved to {model_path}")


# =====================================================================
# 可视化与对比测试函数 (保持原有评估流程，无缝替换模型)
# =====================================================================
def plot_value():
    model_dir = "model"
    model_path = os.path.join(model_dir, "xtfc_angle3d.pth")
    args = Args()
    env = gym.make(args.env_id)
    xtfc = XTFC_PINN(env, hidden_dim=args.hidden_dim)
    xtfc.load_state_dict(torch.load(model_path, map_location="cpu"))
    xtfc.eval()

    x1 = np.linspace(-1, 1, 50)
    x2 = np.linspace(-1, 1, 50)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.stack([X1.ravel(), X2.ravel(), np.zeros_like(X1.ravel())], axis=1)

    X_tensor = torch.from_numpy(X_grid).float()
    with torch.no_grad():
        xtfc_values = xtfc.get_value(X_tensor).cpu().numpy().reshape(X1.shape)

    P = np.eye(3)
    lqr_values = np.einsum('ij,jk,ik->i', X_grid, P, X_grid).reshape(X1.shape)

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X1, X2, xtfc_values, cmap='Blues', alpha=0.85, linewidth=0, antialiased=True)
    ax.plot_surface(X1, X2, lqr_values, cmap='Greens', alpha=0.55, linewidth=0, antialiased=True)

    legend_elements = [
        Line2D([0], [0], color=plt.cm.Blues(0.7), lw=2, label='X-TFC PINN'),
        Line2D([0], [0], color=plt.cm.Greens(0.7), lw=2, label='LQR')
    ]
    ax.legend(handles=legend_elements)
    ax.set_xlabel(r'$\phi$', labelpad=10)
    ax.set_ylabel(r'$\theta$', labelpad=10)
    ax.set_zlabel(r'$V(x)$', labelpad=10)
    ax.set_title(r'X-TFC Value Function Surface ($\psi=0$)')
    ax.view_init(elev=28, azim=-50)
    
    os.makedirs("image", exist_ok=True)
    plt.tight_layout()
    plt.savefig("image/" + args.filename_prefix + "value_function.png", dpi=300)
    plt.show()


def plot_lambda():
    model_dir = "model"
    model_path = os.path.join(model_dir, "xtfc_angle3d.pth")
    args = Args()
    env = gym.make(args.env_id)
    xtfc = XTFC_PINN(env, hidden_dim=args.hidden_dim)
    xtfc.load_state_dict(torch.load(model_path, map_location="cpu"))
    xtfc.eval()

    supervised_model = PINNNetwork(input_dim=3, output_dim=3, hidden_dim=32, num_layers=3)
    sup_path = "model/supervised_angle3d_model.pth"
    if os.path.exists(sup_path):
        supervised_model.load_state_dict(torch.load(sup_path, map_location="cpu"))
    supervised_model.eval()

    x1 = np.linspace(-1, 1, 50)
    x2 = np.linspace(-1, 1, 50)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.stack([X1.ravel(), X2.ravel(), np.zeros_like(X1.ravel())], axis=1)
    X_tensor = torch.from_numpy(X_grid).float()

    with torch.no_grad():
        xtfc_lambda = xtfc.get_lambda(X_tensor).cpu().numpy()
        sup_lambda = supervised_model(X_tensor).cpu().numpy()

    lqr_lambda = 2 * (X_grid @ np.eye(3))

    lambda_names = [r'$\lambda_\phi$', r'$\lambda_\theta$', r'$\lambda_\psi$']
    model_colors = ['Purples', 'Oranges', 'Greens']

    for i in range(3):
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        lam_x = xtfc_lambda[:, i].reshape(X1.shape)
        lam_s = sup_lambda[:, i].reshape(X1.shape)
        
        ax.plot_surface(X1, X2, lam_x, cmap='Purples', alpha=0.8, label='X-TFC')
        ax.plot_surface(X1, X2, lam_s, cmap='Oranges', alpha=0.5, label='Supervised')
        
        ax.set_xlabel(r'$\phi$')
        ax.set_ylabel(r'$\theta$')
        ax.set_zlabel(lambda_names[i])
        ax.set_title(f"Costate Comparison ({lambda_names[i]})")
        
        os.makedirs("image", exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"image/{args.filename_prefix}lambda_{i+1}.png", dpi=300)
        plt.show()


def monte_carlo_simulation():
    args = Args()
    device = args.device
    env = gym.make(args.env_id)

    xtfc = XTFC_PINN(env, hidden_dim=args.hidden_dim).to(device)
    xtfc_path = os.path.join("model", "xtfc_angle3d.pth")
    if os.path.exists(xtfc_path):
        xtfc.load_state_dict(torch.load(xtfc_path, map_location=device))
    xtfc.eval()

    supervised_model = PINNNetwork(input_dim=3, output_dim=3, hidden_dim=32, num_layers=3).to(device)
    sup_path = os.path.join("model", "supervised_angle3d_model.pth")
    if os.path.exists(sup_path):
        supervised_model.load_state_dict(torch.load(sup_path, map_location=device))
    supervised_model.eval()

    num_episodes = 50
    horizon = 200
    R_inv = torch.linalg.inv(torch.tensor(env.R, dtype=torch.float32, device=device))

    def xtfc_action(x_np):
        x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)
        with torch.no_grad():
            u = xtfc.get_action(x)
        return u.squeeze(0).cpu().numpy()

    def sup_action(x_np):
        x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)
        with torch.no_grad():
            Lambda = supervised_model(x)
        pfpu = get_pfpu(x).transpose(1, 2)
        u_tmp = -0.5 * torch.matmul(pfpu, Lambda.unsqueeze(2)).squeeze(-1)
        u = (R_inv @ u_tmp.T).T.squeeze(0)
        return u.cpu().numpy()

    def lqr_action(x_np):
        return -x_np

    returns_xtfc, returns_sup, returns_lqr = [], [], []
    traj_u_xtfc, traj_u_sup, traj_u_lqr = [], [], []
    traj_x_xtfc, traj_x_sup, traj_x_lqr = [], [], []

    print("Running Monte Carlo Simulations over 50 Episodes...")
    for ep in range(num_episodes):
        seed = int(np.random.randint(0, 2**31 - 1))
        env_x = gym.make(args.env_id)
        env_s = gym.make(args.env_id)
        env_l = gym.make(args.env_id)
        
        obs_x, _ = env_x.reset(seed=seed, theoretic_mode=True)
        obs_s, _ = env_s.reset(seed=seed, theoretic_mode=True)
        obs_l, _ = env_l.reset(seed=seed, theoretic_mode=True)

        ret_x, ret_s, ret_l = 0.0, 0.0, 0.0
        u_seq_x, u_seq_s, u_seq_l = [], [], []
        x_seq_x, x_seq_s, x_seq_l = [obs_x.copy()], [obs_s.copy()], [obs_l.copy()]

        for t in range(horizon):
            u_x = xtfc_action(obs_x)
            obs_x, r_x, term_x, trunc_x, _ = env_x.step(u_x.astype(np.float32))
            ret_x += float(r_x)
            u_seq_x.append(u_x)
            x_seq_x.append(obs_x.copy())

            u_s = sup_action(obs_s)
            obs_s, r_s, term_s, trunc_s, _ = env_s.step(u_s.astype(np.float32))
            ret_s += float(r_s)
            u_seq_s.append(u_s)
            x_seq_s.append(obs_s.copy())

            u_l = lqr_action(obs_l)
            obs_l, r_l, term_l, trunc_l, _ = env_l.step(u_l.astype(np.float32))
            ret_l += float(r_l)
            u_seq_l.append(u_l)
            x_seq_l.append(obs_l.copy())

        env_x.close()
        env_s.close()
        env_l.close()

        traj_u_xtfc.append(np.stack(u_seq_x, axis=0))
        traj_u_sup.append(np.stack(u_seq_s, axis=0))
        traj_u_lqr.append(np.stack(u_seq_l, axis=0))

        traj_x_xtfc.append(np.stack(x_seq_x, axis=0))
        traj_x_sup.append(np.stack(x_seq_s, axis=0))
        traj_x_lqr.append(np.stack(x_seq_l, axis=0))

        returns_xtfc.append(ret_x)
        returns_sup.append(ret_s)
        returns_lqr.append(ret_l)

    traj_u_xtfc = np.stack(traj_u_xtfc, axis=0)
    traj_u_sup = np.stack(traj_u_sup, axis=0)
    traj_u_lqr = np.stack(traj_u_lqr, axis=0)

    time = np.arange(horizon)
    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    def plot_mean_std(ax, data, label, color):
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        ax.plot(time, mean, label=label, color=color)
        ax.fill_between(time, mean - std, mean + std, color=color, alpha=0.2)

    colors = {"X-TFC": "tab:blue", "Supervised": "tab:orange", "LQR": "tab:green"}
    for i, ax in enumerate(axs):
        plot_mean_std(ax, traj_u_xtfc[:, :, i], "X-TFC PINN", colors["X-TFC"])
        plot_mean_std(ax, traj_u_sup[:, :, i], "Supervised", colors["Supervised"])
        plot_mean_std(ax, traj_u_lqr[:, :, i], "LQR", colors["LQR"])
        ax.set_ylabel(f"Control $u_{i+1}$")
        ax.grid(True, linestyle='--', alpha=0.4)
        if i == 0:
            ax.legend()
    axs[-1].set_xlabel("Time step")
    fig.suptitle("Control Trajectories (Mean ± Std over 50 Episodes)")
    plt.tight_layout()
    plt.show()

    returns_xtfc = np.array(returns_xtfc)
    returns_sup = np.array(returns_sup)
    returns_lqr = np.array(returns_lqr)

    print("\n=================== Cumulative Returns ===================")
    print(f"X-TFC PINN : mean = {returns_xtfc.mean():.3f}, std = {returns_xtfc.std():.3f}")
    print(f"Supervised : mean = {returns_sup.mean():.3f}, std = {returns_sup.std():.3f}")
    print(f"LQR        : mean = {returns_lqr.mean():.3f}, std = {returns_lqr.std():.3f}")
    print("==========================================================")


if __name__ == "__main__":
    train_xtfc()
    plot_value()
    plot_lambda()
    monte_carlo_simulation()