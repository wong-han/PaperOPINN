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
import casadi as ca
from scipy.linalg import solve_continuous_are

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
    env_id = "environment:quadrotor3d_TR-v0"
    learning_rate = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    filename_prefix = "sp_quadrotor3d_"
    batch_size = 2048

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class SPNet(nn.Module):
    def __init__(self, env):
        self.env = env
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(9, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 4), std=1.0),
        )

    def forward(self, x):
        return self.net(x)

N = 20
def setup_casadi_mpc(env):
    opti = ca.Opti()
    dt = 0.2
    x = opti.variable(9, N+1)
    u = opti.variable(4, N)
    x0 = opti.parameter(9, 1)

    opti.subject_to(x[:, 0] == x0)
    
    Q = env.Q
    R = env.R
    u0 = env.u0
    m = env.metadata['m']
    g = env.metadata['g']
    
    x_min = env.state_low
    x_max = env.state_high
    u_min = env.action_space.low
    u_max = env.action_space.high
    
    cost = 0
    for k in range(N):
        x_k = x[:, k]
        u_k = u[:, k]
        
        x_pos = x_k[0]; y_pos = x_k[1]; z_pos = x_k[2]
        vx = x_k[3]; vy = x_k[4]; vz = x_k[5]
        phi = x_k[6]; theta = x_k[7]; psi = x_k[8]
        
        T = u_k[0]; p = u_k[1]; q = u_k[2]; r = u_k[3]
        
        cos_phi = ca.cos(phi); sin_phi = ca.sin(phi)
        cos_theta = ca.cos(theta); sin_theta = ca.sin(theta)
        cos_theta_safe = ca.if_else(ca.fabs(cos_theta) < 1e-6, 1e-6, cos_theta)
        tan_theta = sin_theta / cos_theta_safe
        cos_psi = ca.cos(psi); sin_psi = ca.sin(psi)
        
        dx = vx
        dy = vy
        dz = vz
        dvx = T * (cos_phi * sin_theta * cos_psi + sin_phi * sin_psi) / m
        dvy = T * (cos_phi * sin_theta * sin_psi - sin_phi * cos_psi) / m
        dvz = T * (cos_phi * cos_theta) / m - g
        dphi = p + tan_theta * sin_phi * q + tan_theta * cos_phi * r
        dtheta = cos_phi * q - sin_phi * r
        dpsi = (sin_phi / cos_theta_safe) * q + (cos_phi / cos_theta_safe) * r
        
        opti.subject_to(x[0, k+1] == x_pos + dx * dt)
        opti.subject_to(x[1, k+1] == y_pos + dy * dt)
        opti.subject_to(x[2, k+1] == z_pos + dz * dt)
        opti.subject_to(x[3, k+1] == vx + dvx * dt)
        opti.subject_to(x[4, k+1] == vy + dvy * dt)
        opti.subject_to(x[5, k+1] == vz + dvz * dt)
        opti.subject_to(x[6, k+1] == phi + dphi * dt)
        opti.subject_to(x[7, k+1] == theta + dtheta * dt)
        opti.subject_to(x[8, k+1] == psi + dpsi * dt)
        
        x_err = x_k
        u_err = u_k - u0
        cost += ca.mtimes([x_err.T, Q, x_err]) * dt + ca.mtimes([u_err.T, R, u_err]) * dt
    
    P_ref = env.P
    x_err_N = x[:, N]
    cost += ca.mtimes([x_err_N.T, P_ref, x_err_N])
    
    opti.minimize(cost)
    
    for i in range(9):
        opti.subject_to(opti.bounded(x_min[i], x[i, :], x_max[i]))
    for i in range(4):
        opti.subject_to(opti.bounded(u_min[i], u[i, :], u_max[i]))

    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes', 'ipopt.max_iter': 500}
    opti.solver('ipopt', opts)
    return opti, x0, u


def train_sp():
    env = gym.make(Args.env_id)
    model = SPNet(env).to(Args.device)
    optimizer = optim.Adam(model.parameters(), lr=Args.learning_rate)

    # print("--- Phase 1: Data Generation using MPC ---")
    # start_time = time.time()
    
    # opti, x0_param, u_var = setup_casadi_mpc(env)
    
    # X_data = []
    # U_data = []
    # X_trajs = []
    # U_trajs = []
    
    # while time.time() - start_time < 3600:
    #     # 结合 MPC 在环境中的闭环仿真采集数据集 (Behavioral Cloning 分布匹配)
    #     init_state = np.random.uniform(-1.0, 1.0, 9)
    #     state, _ = env.reset(theoretic_mode=True, options={'init_state': init_state})
    #     u_init = np.zeros((4, N))
    #     for i in range(N):
    #         u_init[:, i] = env.u0
            
    #     x_traj = []
    #     u_traj = []
            
    #     episode_length = int(env.T / env.dt)
    #     for step in range(episode_length):
    #         opti.set_value(x0_param, state)
    #         opti.set_initial(u_var, u_init)
            
    #         try:
    #             sol = opti.solve()
    #             u_opt = sol.value(u_var[:, 0])
    #             x_traj.append(state.copy())
    #             u_traj.append(u_opt)
                
    #             # Warm start 给下一步使用
    #             u_seq = sol.value(u_var)
    #             if u_seq.ndim == 1:
    #                 u_seq = u_seq.reshape(-1, 1)
    #             u_init[:, :-1] = u_seq[:, 1:]
    #             u_init[:, -1] = u_seq[:, -1]
                
    #         except Exception:
    #             # 求解失败，直接放弃该回合剩余部分，重新 reset
    #             break 
                
    #         # 环境前向演化
    #         state, reward, terminated, truncated, _ = env.step(u_opt)
    #         if terminated or truncated:
    #             break
                
    #         if time.time() - start_time >= 3000:
    #             break
                
    #     if len(x_traj) > 0:
    #         X_trajs.append(np.array(x_traj))
    #         U_trajs.append(np.array(u_traj))
    #         X_data.extend(x_traj)
    #         U_data.extend(u_traj)

    # X_data = np.array(X_data, dtype=np.float32)
    # U_data = np.array(U_data, dtype=np.float32)
    # print(f"Generated {len(X_data)} samples across {len(X_trajs)} trajectories.")

    # # 保存轨迹到data文件夹，画图画出所有轨迹
    # import os
    # if not os.path.exists('data'):
    #     os.makedirs('data')
        
    # # 因为轨迹长度不一致，使用 dtype=object 搭配 allow_pickle=True 保存变长数组的列表
    # np.save('data/X_trajs.npy', np.array(X_trajs, dtype=object), allow_pickle=True)
    # np.save('data/U_trajs.npy', np.array(U_trajs, dtype=object), allow_pickle=True)
    # print("Trajectories saved to data/X_trajs.npy and data/U_trajs.npy")
    
    # # 画图：所有轨迹的位置变化
    # plt.figure(figsize=(10, 8))
    # ax = plt.axes(projection='3d')
    # for traj in X_trajs:
    #     ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.6)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title(f'MPC Expert Trajectories (N={len(X_trajs)})')
    # plt.savefig('image/mpc_expert_trajectories_3d.png')
    # print("Saved 3D trajectory plot to mpc_expert_trajectories_3d.png")
    # plt.close()
    
    # =====================================================================
    # 快速调试加载数据：
    # 可以随意注释掉上方的 `while time.time() - start_time < 3000:` 循环。
    # =====================================================================
    print("Loading expert trajectories from disk...")
    X_trajs_raw = np.load('data/X_trajs.npy', allow_pickle=True)
    U_trajs_raw = np.load('data/U_trajs.npy', allow_pickle=True)
    X_data = np.vstack([traj for traj in X_trajs_raw]).astype(np.float32)
    U_data = np.vstack([traj for traj in U_trajs_raw]).astype(np.float32)
    print(f"Successfully loaded {len(X_data)} samples from {len(X_trajs_raw)} trajectories.")

    if len(X_data) == 0:
        print("No data generated. Exiting.")
        return

    X_tensor_all = torch.from_numpy(X_data).to(Args.device)
    U_tensor_all = torch.from_numpy(U_data).to(Args.device)

    print("--- Phase 2: Training Network ---")
    train_start = time.time()
    batch_size = min(Args.batch_size, len(X_data))
    
    step = 0
    patience = 500
    patience_counter = 0

    while True:
        idx = np.random.choice(len(X_data), batch_size)
        X_batch = X_tensor_all[idx]
        U_batch = U_tensor_all[idx]
        
        u_pred = model(X_batch)
        loss = torch.nn.functional.mse_loss(u_pred, U_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        
        if current_loss < 0.0005:
            patience_counter += 1
        else:
            patience_counter = 0
            
        if step % 100 == 0:
            print(f"Step {step}, Loss: {current_loss:.6f}, Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Loss maintained below 0.0001 for {patience} steps. Stopping training.")
            break
            
        if step > 100000:
            print(f"Max steps (100000) reached. Stopping training.")
            break

        step += 1

    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "sp_quadrotor3d.pth")
    torch.save(model.state_dict(), model_path)
    print(f"SP model saved to {model_path}. Total training steps: {step}")
    print(f"Total training time: {time.time() - train_start:.2f} seconds")


def monte_carlo_simulation():
    np.random.seed(42)
    args = Args()
    device = args.device
    num_episodes = 50

    env_lqr = gym.make(args.env_id, theoretic_mode=True)
    env_sp = gym.make(args.env_id, theoretic_mode=True)
    horizon = int(env_lqr.T / env_lqr.dt)

    A = env_lqr.A
    B = env_lqr.B
    Q = env_lqr.Q
    R = env_lqr.R
    u0 = env_lqr.u0

    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P

    sp_model = SPNet(env_sp).to(device)
    sp_path = os.path.join("model", "sp_quadrotor3d.pth")
    if os.path.exists(sp_path):
        sp_model.load_state_dict(torch.load(sp_path, map_location=device))
    sp_model.eval()

    def sp_action(x_np):
        x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)
        with torch.no_grad():
            u = sp_model(x).squeeze(0).cpu().numpy()
        return u

    try:
        from pihnn_experiments_plot.p4_hnn_quadrotor3d_TR import HNN, get_pfpu
        env_hnn = gym.make(args.env_id, theoretic_mode=True)
        hnn_model = HNN(env_hnn).to(device)
        hnn_path = os.path.join("model", "hnn_quadrotor3d_TR_v2_paperplot.pth")
        hnn_model.load_state_dict(torch.load(hnn_path, map_location=device))
        hnn_model.eval()

        R_torch = torch.tensor(R, dtype=torch.float32, device=device)
        R_inv = torch.linalg.inv(R_torch)

        def hnn_action(x_np):
            x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)
            with torch.no_grad():
                Lambda = hnn_model.get_lambda(x)
            pfpu = get_pfpu(x)
            pfpu_T = pfpu.transpose(1, 2)
            u_tmp = -0.5 * torch.matmul(pfpu_T, Lambda.unsqueeze(2)).squeeze(-1)
            u = (R_inv @ u_tmp.T).T.squeeze(0).detach().cpu().numpy() + u0
            return u
        hnn_loaded = True
    except Exception as e:
        print(f"Could not load HNN: {e}")
        hnn_loaded = False

    all_traj_x_lqr = []
    all_traj_u_lqr = []
    all_traj_x_sp = []
    all_traj_u_sp = []
    all_traj_x_hnn = []
    all_traj_u_hnn = []
    returns_lqr = []
    returns_sp = []
    returns_hnn = []

    print("Starting Monte Carlo simulation...")
    for ep in range(num_episodes):
        if (ep+1) % 1 == 0:
            print(f"Episode {ep+1}/{num_episodes}")
            
        seed = np.random.randint(0, 1000000)
        
        obs_lqr, _ = env_lqr.reset(seed=seed, theoretic_mode=True)
        obs_sp, _ = env_sp.reset(seed=seed, theoretic_mode=True)
        if hnn_loaded:
            obs_hnn, _ = env_hnn.reset(seed=seed, theoretic_mode=True)

        x_seq_lqr = [obs_lqr.copy()]
        u_seq_lqr = []
        x_seq_sp = [obs_sp.copy()]
        u_seq_sp = []
        if hnn_loaded:
            x_seq_hnn = [obs_hnn.copy()]
            u_seq_hnn = []

        ret_lqr = 0.0
        ret_sp = 0.0
        ret_hnn = 0.0

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

        # SPNet episode
        obs_sp_ep = obs_sp.copy()
        for t in range(horizon):
            u_s = sp_action(obs_sp_ep)
            u_s = np.clip(u_s, env_sp.action_space.low, env_sp.action_space.high)
            next_obs_s, reward_s, terminated_s, truncated_s, _ = env_sp.step(u_s)
            u_seq_sp.append(u_s)
            obs_sp_ep = next_obs_s
            x_seq_sp.append(obs_sp_ep.copy())
            ret_sp += float(reward_s)

        # HNN episode
        if hnn_loaded:
            obs_hnn_ep = obs_hnn.copy()
            for t in range(horizon):
                u_h = hnn_action(obs_hnn_ep)
                u_h = np.clip(u_h, env_hnn.action_space.low, env_hnn.action_space.high)
                next_obs_h, reward_h, terminated_h, truncated_h, _ = env_hnn.step(u_h)
                u_seq_hnn.append(u_h)
                obs_hnn_ep = next_obs_h
                x_seq_hnn.append(obs_hnn_ep.copy())
                ret_hnn += float(reward_h)

        all_traj_x_lqr.append(np.vstack(x_seq_lqr))
        all_traj_u_lqr.append(np.vstack(u_seq_lqr))
        returns_lqr.append(ret_lqr)

        all_traj_x_sp.append(np.vstack(x_seq_sp))
        all_traj_u_sp.append(np.vstack(u_seq_sp))
        returns_sp.append(ret_sp)

        if hnn_loaded:
            all_traj_x_hnn.append(np.vstack(x_seq_hnn))
            all_traj_u_hnn.append(np.vstack(u_seq_hnn))
            returns_hnn.append(ret_hnn)

    env_lqr.close()
    env_sp.close()
    if hnn_loaded:
        env_hnn.close()

    traj_u_sp = np.stack(all_traj_u_sp, axis=0)
    traj_x_sp = np.stack(all_traj_x_sp, axis=0)
    traj_u_lqr = np.stack(all_traj_u_lqr, axis=0)
    traj_x_lqr = np.stack(all_traj_x_lqr, axis=0)
    if hnn_loaded:
        traj_u_hnn = np.stack(all_traj_u_hnn, axis=0)
        traj_x_hnn = np.stack(all_traj_x_hnn, axis=0)

    colors = {"SPNet": "tab:orange", "LQR": "tab:green"}
    if hnn_loaded:
        colors["HNN"] = "tab:blue"

    def plot_mean_std(ax, data, label, color):
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        t = np.arange(len(mean))
        ax.plot(t, mean, label=label, color=color)
        ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.2)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(base_dir, "image")
    os.makedirs(image_dir, exist_ok=True)
    
    # Plot Traj U
    fig, axs = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
    labels_u = ['T', 'p', 'q', 'r']
    for i in range(4):
        ax = axs[i]
        plot_mean_std(ax, traj_u_sp[:, :, i], "SPNet", colors["SPNet"])
        plot_mean_std(ax, traj_u_lqr[:, :, i], "LQR", colors["LQR"])
        if hnn_loaded:
            plot_mean_std(ax, traj_u_hnn[:, :, i], "HNN", colors["HNN"])
        ax.set_ylabel(labels_u[i])
        if i == 0:
            ax.legend(loc='upper right')
        ax.grid(True)
    axs[3].set_xlabel("Time step")
    fig.tight_layout()
    plt.savefig(os.path.join(image_dir, args.filename_prefix + "traj_u.svg"), dpi=600, bbox_inches='tight', pad_inches=0.5)

    # Plot Traj X 
    fig, axs = plt.subplots(6, 1, figsize=(10, 16), sharex=True)
    labels_x = ['x', 'y', 'z', r'$\phi$', r'$\theta$', r'$\psi$']
    plot_indices = [0, 1, 2, 6, 7, 8]
    for i in range(6):
        ax = axs[i]
        idx = plot_indices[i]
        plot_mean_std(ax, traj_x_sp[:, :, idx], "SPNet", colors["SPNet"])
        plot_mean_std(ax, traj_x_lqr[:, :, idx], "LQR", colors["LQR"])
        if hnn_loaded:
            plot_mean_std(ax, traj_x_hnn[:, :, idx], "HNN", colors["HNN"])
        ax.set_ylabel(labels_x[i])
        if i == 0:
            ax.legend(loc='upper right')
        ax.grid(True)
    axs[5].set_xlabel("Time step")
    fig.tight_layout()
    plt.savefig(os.path.join(image_dir, args.filename_prefix + "traj_x.svg"), dpi=600, bbox_inches='tight', pad_inches=0.5)

    print(f"SPNet: mean={np.mean(returns_sp):.3f}, std={np.std(returns_sp):.3f}")
    print(f"LQR: mean={np.mean(returns_lqr):.3f}, std={np.std(returns_lqr):.3f}")
    if hnn_loaded:
        print(f"HNN: mean={np.mean(returns_hnn):.3f}, std={np.std(returns_hnn):.3f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    min_val = min(np.min(returns_sp), np.min(returns_lqr))
    max_val = max(np.max(returns_sp), np.max(returns_lqr))
    if hnn_loaded:
        min_val = min(min_val, np.min(returns_hnn))
        max_val = max(max_val, np.max(returns_hnn))
    bins = np.linspace(min_val, max_val, 20)
    
    plt.hist(returns_sp, bins=bins, alpha=0.5, label="SPNet", color=colors["SPNet"])
    plt.hist(returns_lqr, bins=bins, alpha=0.5, label="LQR", color=colors["LQR"])
    if hnn_loaded:
        plt.hist(returns_hnn, bins=bins, alpha=0.5, label="HNN", color=colors["HNN"])
        
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
