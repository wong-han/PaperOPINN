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
import cmaps

# ================== 图像设置 =======================
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['text.usetex'] = True
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
    env_id = "environment:quadrotor3d_TR_constrained"
    learning_rate = 3e-4
    time_steps = 1000000
    num_steps = 2048
    batch_size = 64
    ppo_epochs = 10
    clip_param = 0.2
    gamma = 0.99
    gae_lambda = 0.95
    device = "cuda" if torch.cuda.is_available() else "cpu"
    filename_prefix = "rl_quadrotor3d_constrained_"


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class RLNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def get_action(self, x):
        return self.actor_mean(x)


def setup_casadi_mpc(env):
    import casadi as ca
    opti = ca.Opti()
    dt = 0.2
    N = 20
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
    
    opti.subject_to(x[1, :] >= 0.0)
    
    for i in range(9):
        opti.subject_to(opti.bounded(x_min[i], x[i, :], x_max[i]))
    for i in range(4):
        opti.subject_to(opti.bounded(u_min[i], u[i, :], u_max[i]))

    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes', 'ipopt.max_iter': 500}
    opti.solver('ipopt', opts)
    return opti, x0, u


def monte_carlo_simulation():
    np.random.seed(2222)
    args = Args()
    device = args.device
    num_episodes = 50

    env_lqr = gym.make(args.env_id, theoretic_mode=True)
    env_sp = gym.make(args.env_id, theoretic_mode=True)
    env_rl = gym.make(args.env_id, theoretic_mode=True)
    horizon = int(env_lqr.T / env_lqr.dt)

    A = env_lqr.A
    B = env_lqr.B
    Q = env_lqr.Q
    R = env_lqr.R
    u0 = env_lqr.u0

    from scipy.linalg import solve_continuous_are
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P

    # 1. Load RL
    rl_model = RLNet(9, 4).to(device)
    rl_path = os.path.join("..", "reinforce_learn_ex_plot", "model", "rl_quadrotor3d_constrained.pth")
    if os.path.exists(rl_path):
        rl_model.load_state_dict(torch.load(rl_path, map_location=device))
    rl_model.eval()

    def rl_action(x_np):
        x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)
        with torch.no_grad():
            a = rl_model.get_action(x).squeeze(0).cpu().numpy()
        u_min = env_rl.action_space.low
        u_max = env_rl.action_space.high
        u = a * (u_max - u_min) / 2.0 + (u_max + u_min) / 2.0
        return u

    # 2. Load SPNet
    try:
        from sup_learn_ex_plot.p4_sp_quadrotor3d_TR_constrained import SPNet
        sp_model = SPNet(env_sp).to(device)
        sp_path = os.path.join("..", "sup_learn_ex_plot", "model", "sp_quadrotor3d_constrained.pth")
        if os.path.exists(sp_path):
            sp_model.load_state_dict(torch.load(sp_path, map_location=device))
        sp_model.eval()
        def sp_action(x_np):
            x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)
            with torch.no_grad():
                u = sp_model(x).squeeze(0).cpu().numpy()
            return u
        sp_loaded = True
    except Exception as e:
        print(f"Could not load SPNet: {e}")
        sp_loaded = False

    # 3. Load HNN
    try:
        from pihnn_constrained_ex_plot.p4_chnn_quadrotor3d_TR import HNN, get_pfpu
        env_hnn = gym.make(args.env_id, theoretic_mode=True)
        hnn_model = HNN(env_hnn).to(device)
        hnn_path = os.path.join("..", "pihnn_constrained_ex_plot", "model", "chnn_quadrotor3d_TR_v1.pth")
        if os.path.exists(hnn_path):
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
        print(f"HNN not loaded: {e}")
        hnn_loaded = False

    # 4. Load MPC
    env_mpc = gym.make(args.env_id, theoretic_mode=True)
    opti_mpc, x0_param_mpc, u_var_mpc = setup_casadi_mpc(env_mpc)
    mpc_loaded = True

    all_traj_x_lqr = []
    all_traj_u_lqr = []
    all_traj_x_sp = []
    all_traj_u_sp = []
    all_traj_x_hnn = []
    all_traj_u_hnn = []
    all_traj_x_rl = []
    all_traj_u_rl = []
    all_traj_x_mpc = []
    all_traj_u_mpc = []
    returns_lqr = []
    returns_sp = []
    returns_hnn = []
    returns_rl = []
    returns_mpc = []

    print("Starting Monte Carlo simulation...")
    for ep in range(num_episodes):
        if (ep+1) % 1 == 0:
            print(f"Episode {ep+1}/{num_episodes}")
            
        seed = np.random.randint(0, 1000000)
        
        obs_lqr, _ = env_lqr.reset(seed=seed, theoretic_mode=True)
        obs_rl, _ = env_rl.reset(seed=seed, theoretic_mode=True)
        if sp_loaded:
            obs_sp, _ = env_sp.reset(seed=seed, theoretic_mode=True)
        if hnn_loaded:
            obs_hnn, _ = env_hnn.reset(seed=seed, theoretic_mode=True)
        if mpc_loaded:
            obs_mpc, _ = env_mpc.reset(seed=seed, theoretic_mode=True)
            u_init_mpc = np.zeros((4, 20))
            for i in range(20):
                u_init_mpc[:, i] = env_mpc.u0

        x_seq_lqr = [obs_lqr.copy()]
        u_seq_lqr = []
        x_seq_rl = [obs_rl.copy()]
        u_seq_rl = []
        if sp_loaded:
            x_seq_sp = [obs_sp.copy()]
            u_seq_sp = []
        if hnn_loaded:
            x_seq_hnn = [obs_hnn.copy()]
            u_seq_hnn = []
        if mpc_loaded:
            x_seq_mpc = [obs_mpc.copy()]
            u_seq_mpc = []

        ret_lqr = 0.0
        ret_sp = 0.0
        ret_hnn = 0.0
        ret_rl = 0.0
        ret_mpc = 0.0

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

        # RL episode
        obs_rl_ep = obs_rl.copy()
        for t in range(horizon):
            u_r = rl_action(obs_rl_ep)
            u_r = np.clip(u_r, env_rl.action_space.low, env_rl.action_space.high)
            next_obs_r, reward_r, terminated_r, truncated_r, _ = env_rl.step(u_r)
            u_seq_rl.append(u_r)
            obs_rl_ep = next_obs_r
            x_seq_rl.append(obs_rl_ep.copy())
            ret_rl += float(reward_r)

        # SPNet episode
        if sp_loaded:
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

        # MPC episode
        if mpc_loaded:
            obs_mpc_ep = obs_mpc.copy()
            for t in range(horizon):
                opti_mpc.set_value(x0_param_mpc, obs_mpc_ep)
                opti_mpc.set_initial(u_var_mpc, u_init_mpc)
                try:
                    sol = opti_mpc.solve()
                    u_m = sol.value(u_var_mpc[:, 0])
                    u_seq = sol.value(u_var_mpc)
                    if u_seq.ndim == 1:
                        u_seq = u_seq.reshape(-1, 1)
                    u_init_mpc[:, :-1] = u_seq[:, 1:]
                    u_init_mpc[:, -1] = u_seq[:, -1]
                except Exception:
                    u_m = opti_mpc.debug.value(u_var_mpc[:, 0])
                
                u_m = np.clip(u_m, env_mpc.action_space.low, env_mpc.action_space.high)
                next_obs_m, reward_m, terminated_m, truncated_m, _ = env_mpc.step(u_m)
                u_seq_mpc.append(u_m)
                obs_mpc_ep = next_obs_m
                x_seq_mpc.append(obs_mpc_ep.copy())
                ret_mpc += float(reward_m)

        all_traj_x_lqr.append(np.vstack(x_seq_lqr))
        all_traj_u_lqr.append(np.vstack(u_seq_lqr))
        returns_lqr.append(ret_lqr)

        all_traj_x_rl.append(np.vstack(x_seq_rl))
        all_traj_u_rl.append(np.vstack(u_seq_rl))
        returns_rl.append(ret_rl)

        if sp_loaded:
            all_traj_x_sp.append(np.vstack(x_seq_sp))
            all_traj_u_sp.append(np.vstack(u_seq_sp))
            returns_sp.append(ret_sp)

        if hnn_loaded:
            all_traj_x_hnn.append(np.vstack(x_seq_hnn))
            all_traj_u_hnn.append(np.vstack(u_seq_hnn))
            returns_hnn.append(ret_hnn)

        if mpc_loaded:
            all_traj_x_mpc.append(np.vstack(x_seq_mpc))
            all_traj_u_mpc.append(np.vstack(u_seq_mpc))
            returns_mpc.append(ret_mpc)

    env_lqr.close()
    env_rl.close()
    if sp_loaded:
        env_sp.close()
    if hnn_loaded:
        env_hnn.close()
    if mpc_loaded:
        env_mpc.close()

    # ========== 计算成功率并过滤发散轨迹 ==========
    final_dist_lqr = np.array([np.linalg.norm(traj[-1, 0:3]) for traj in all_traj_x_lqr])
    final_dist_rl = np.array([np.linalg.norm(traj[-1, 0:3]) for traj in all_traj_x_rl])
    if sp_loaded:
        final_dist_sp = np.array([np.linalg.norm(traj[-1, 0:3]) for traj in all_traj_x_sp])
    if hnn_loaded:
        final_dist_hnn = np.array([np.linalg.norm(traj[-1, 0:3]) for traj in all_traj_x_hnn])
    if mpc_loaded:
        final_dist_mpc = np.array([np.linalg.norm(traj[-1, 0:3]) for traj in all_traj_x_mpc])
        
    # 认为 距离 > 0.1 为失败
    success_mask_lqr = final_dist_lqr <= 0.1
    success_mask_rl = final_dist_rl <= 0.1
    if sp_loaded:
        success_mask_sp = final_dist_sp <= 0.1
    if hnn_loaded:
        success_mask_hnn = final_dist_hnn <= 0.1
    if mpc_loaded:
        success_mask_mpc = final_dist_mpc <= 0.1

    print("\nMonte Carlo 50 episodes Success Rate:")
    print(f"LQR: {np.mean(success_mask_lqr)*100:.1f}%")
    if sp_loaded:
        print(f"SPNet: {np.mean(success_mask_sp)*100:.1f}%")
    print(f"RL: {np.mean(success_mask_rl)*100:.1f}%")
    if hnn_loaded:
        print(f"HNN: {np.mean(success_mask_hnn)*100:.1f}%")

    def filter_traj(traj_list, mask):
        filtered = [t for i, t in enumerate(traj_list) if mask[i]]
        if len(filtered) == 0:
            return np.zeros((1, traj_list[0].shape[0], traj_list[0].shape[1]))
        return np.stack(filtered, axis=0)

    traj_x_lqr = filter_traj(all_traj_x_lqr, success_mask_lqr)
    traj_x_rl = filter_traj(all_traj_x_rl, success_mask_rl)
    if sp_loaded:
        traj_x_sp = filter_traj(all_traj_x_sp, success_mask_sp)
    if hnn_loaded:
        traj_x_hnn = filter_traj(all_traj_x_hnn, success_mask_hnn)

    # =============== 提取成功轨迹的回报 ===============
    filtered_returns_lqr = np.array(returns_lqr)[success_mask_lqr]
    filtered_returns_rl = np.array(returns_rl)[success_mask_rl]
    if sp_loaded:
        filtered_returns_sp = np.array(returns_sp)[success_mask_sp]
    if hnn_loaded:
        filtered_returns_hnn = np.array(returns_hnn)[success_mask_hnn]
    if mpc_loaded:
        filtered_returns_mpc = np.array(returns_mpc)[success_mask_mpc]

    print("\nMonte Carlo 50 episodes Return Statistics (Successful only):")
    print(f"LQR: mean={np.mean(filtered_returns_lqr):.3f}, std={np.std(filtered_returns_lqr):.3f}")
    if mpc_loaded:
        print(f"MPC: mean={np.mean(filtered_returns_mpc):.3f}, std={np.std(filtered_returns_mpc):.3f}")
    if sp_loaded:
        print(f"SPNet: mean={np.mean(filtered_returns_sp):.3f}, std={np.std(filtered_returns_sp):.3f}")
    print(f"RL: mean={np.mean(filtered_returns_rl):.3f}, std={np.std(filtered_returns_rl):.3f}")
    if hnn_loaded:
        print(f"HNN: mean={np.mean(filtered_returns_hnn):.3f}, std={np.std(filtered_returns_hnn):.3f}")

    min_y_lqr = np.array([np.min(traj[:, 1]) for traj in all_traj_x_lqr])
    min_y_rl = np.array([np.min(traj[:, 1]) for traj in all_traj_x_rl])
    if sp_loaded:
        min_y_sp = np.array([np.min(traj[:, 1]) for traj in all_traj_x_sp])
    if hnn_loaded:
        min_y_hnn = np.array([np.min(traj[:, 1]) for traj in all_traj_x_hnn])
    if mpc_loaded:
        min_y_mpc = np.array([np.min(traj[:, 1]) for traj in all_traj_x_mpc])
        
    filtered_min_y_lqr = min_y_lqr[success_mask_lqr]
    filtered_min_y_rl = min_y_rl[success_mask_rl]
    if sp_loaded:
        filtered_min_y_sp = min_y_sp[success_mask_sp]
    if hnn_loaded:
        filtered_min_y_hnn = min_y_hnn[success_mask_hnn]
    if mpc_loaded:
        filtered_min_y_mpc = min_y_mpc[success_mask_mpc]

    eps = 1e-8
    log_abs_ymin_lqr = np.log10(np.abs(filtered_min_y_lqr) + eps)
    log_abs_ymin_rl = np.log10(np.abs(filtered_min_y_rl) + eps)
    if sp_loaded:
        log_abs_ymin_sp = np.log10(np.abs(filtered_min_y_sp) + eps)
    if hnn_loaded:
        log_abs_ymin_hnn = np.log10(np.abs(filtered_min_y_hnn) + eps)
    if mpc_loaded:
        log_abs_ymin_mpc = np.log10(np.abs(filtered_min_y_mpc) + eps)

    import pandas as pd
    import seaborn as sns

    # 准备 log_y 数据
    data_logy = []
    data_controllers_logy = []
    
    data_logy.extend(log_abs_ymin_lqr)
    data_controllers_logy.extend(["LQR"] * len(log_abs_ymin_lqr))
    
    if mpc_loaded:
        data_logy.extend(log_abs_ymin_mpc)
        data_controllers_logy.extend(["MPC"] * len(log_abs_ymin_mpc))
        
    if sp_loaded:
        data_logy.extend(log_abs_ymin_sp)
        data_controllers_logy.extend(["SPNet"] * len(log_abs_ymin_sp))
        
    data_logy.extend(log_abs_ymin_rl)
    data_controllers_logy.extend(["RL"] * len(log_abs_ymin_rl))
    
    if hnn_loaded:
        data_logy.extend(log_abs_ymin_hnn)
        data_controllers_logy.extend(["CHNN"] * len(log_abs_ymin_hnn))
        
    df_logy = pd.DataFrame({
        r"$\log_{10}|y_{\mathrm{min}}|$": data_logy,
        "Controller": data_controllers_logy
    })
    
    # 颜色设定
    order = ["LQR"]
    palette = ["#F39B7F"]
    if mpc_loaded:
        order.append("MPC")
        palette.append("#E64B35")
    if sp_loaded:
        order.append("SPNet")
        palette.append("#389826")
    order.append("RL")
    palette.append("#4B8BBE")
    if hnn_loaded:
        order.append("CHNN")
        palette.append("#9558B2")
        
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(base_dir, "image")
    os.makedirs(image_dir, exist_ok=True)
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # =============== 自动保存绘图数据 ===============
    # 1. 保存供 Seaborn 直接使用的 DataFrame
    df_logy.to_csv(os.path.join(data_dir, args.filename_prefix + "logy_boxplot_data.csv"), index=False)
    
    # 因为下面才计算 df_ret，所以把这个放在更后面保存，但这里先保存 raw 数据
    
    # 2. 保存原始回报和成功掩码，方便以后扩展分析
    np.savez(os.path.join(data_dir, args.filename_prefix + "raw_returns_and_logy.npz"),
             returns_lqr=returns_lqr, success_mask_lqr=success_mask_lqr, log_abs_ymin_lqr=log_abs_ymin_lqr,
             returns_rl=returns_rl, success_mask_rl=success_mask_rl, log_abs_ymin_rl=log_abs_ymin_rl,
             returns_sp=returns_sp if sp_loaded else [], success_mask_sp=success_mask_sp if sp_loaded else [],
             returns_hnn=returns_hnn if hnn_loaded else [], success_mask_hnn=success_mask_hnn if hnn_loaded else [],
             returns_mpc=returns_mpc if mpc_loaded else [], success_mask_mpc=success_mask_mpc if mpc_loaded else [])
             
    # 3. 保存所有状态轨迹数据（处理成 numpy object 数组以允许不规则长度）
    np.save(os.path.join(data_dir, args.filename_prefix + "all_traj_x_lqr.npy"), np.array(all_traj_x_lqr, dtype=object), allow_pickle=True)
    np.save(os.path.join(data_dir, args.filename_prefix + "all_traj_x_rl.npy"), np.array(all_traj_x_rl, dtype=object), allow_pickle=True)
    if sp_loaded:
        np.save(os.path.join(data_dir, args.filename_prefix + "all_traj_x_sp.npy"), np.array(all_traj_x_sp, dtype=object), allow_pickle=True)
    if hnn_loaded:
        np.save(os.path.join(data_dir, args.filename_prefix + "all_traj_x_hnn.npy"), np.array(all_traj_x_hnn, dtype=object), allow_pickle=True)
    if mpc_loaded:
        np.save(os.path.join(data_dir, args.filename_prefix + "all_traj_x_mpc.npy"), np.array(all_traj_x_mpc, dtype=object), allow_pickle=True)
    
    print(f"All Monte Carlo simulation data successfully saved to {data_dir}/")

    # 画 log y min 箱线图 (Nature 风格融合)
    fig_y, ax_y = plt.subplots(figsize=(8, 7))
    sns.violinplot(
        x="Controller", y=r"$\log_{10}|y_{\mathrm{min}}|$", data=df_logy,
        order=order, palette=palette, inner=None, alpha=0.22, cut=0,
        linewidth=1.2, ax=ax_y
    )
    sns.boxplot(
        x="Controller", y=r"$\log_{10}|y_{\mathrm{min}}|$", data=df_logy,
        order=order, palette=palette, width=0.25,
        showcaps=True, showbox=True, showfliers=False, medianprops=dict(color="#6D2D2B", linewidth=2),
        boxprops=dict(alpha=0.7, edgecolor='k', linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        ax=ax_y
    )
    sns.stripplot(
        x="Controller", y=r"$\log_{10}|y_{\mathrm{min}}|$", data=df_logy,
        order=order, palette=palette,
        size=7, alpha=0.4, jitter=0.22, linewidth=0.2, edgecolor='#444', ax=ax_y
    )
    ax_y.set_xticklabels(order)
    ax_y.set_ylabel(r"$\log_{10}|y_{\mathrm{min}}|$")
    ax_y.grid(True, linestyle='--', alpha=0.45)
    sns.despine(top=False, right=False, left=False, bottom=False, ax=ax_y)
    
    fig_y.tight_layout()
    plt.savefig(os.path.join(image_dir, args.filename_prefix + "min_y_boxplot.svg"), dpi=600, bbox_inches='tight', pad_inches=0.5)

    # 画 Return 箱线图
    data_ret = []
    data_controllers_ret = []
    
    data_ret.extend(filtered_returns_lqr)
    data_controllers_ret.extend(["LQR"] * len(filtered_returns_lqr))
    
    if mpc_loaded:
        data_ret.extend(filtered_returns_mpc)
        data_controllers_ret.extend(["MPC"] * len(filtered_returns_mpc))
        
    if sp_loaded:
        data_ret.extend(filtered_returns_sp)
        data_controllers_ret.extend(["SPNet"] * len(filtered_returns_sp))
        
    data_ret.extend(filtered_returns_rl)
    data_controllers_ret.extend(["RL"] * len(filtered_returns_rl))
    
    if hnn_loaded:
        data_ret.extend(filtered_returns_hnn)
        data_controllers_ret.extend(["CHNN"] * len(filtered_returns_hnn))
        
    df_ret = pd.DataFrame({
        "Return": data_ret,
        "Controller": data_controllers_ret
    })
    
    df_ret.to_csv(os.path.join(data_dir, args.filename_prefix + "return_boxplot_data.csv"), index=False)

    fig_r, ax_r = plt.subplots(figsize=(8, 7))
    sns.violinplot(
        x="Controller", y="Return", data=df_ret,
        order=order, palette=palette, inner=None, alpha=0.22, cut=0,
        linewidth=1.2, ax=ax_r
    )
    sns.boxplot(
        x="Controller", y="Return", data=df_ret,
        order=order, palette=palette, width=0.25,
        showcaps=True, showbox=True, showfliers=False, medianprops=dict(color="#6D2D2B", linewidth=2),
        boxprops=dict(alpha=0.7, edgecolor='k', linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        ax=ax_r
    )
    sns.stripplot(
        x="Controller", y="Return", data=df_ret,
        order=order, palette=palette,
        size=7, alpha=0.4, jitter=0.22, linewidth=0.2, edgecolor='#444', ax=ax_r
    )
    
    ax_r.set_xticklabels(order)
    ax_r.set_ylabel("Return")
    ax_r.grid(True, linestyle='--', alpha=0.45)
    sns.despine(top=False, right=False, left=False, bottom=False, ax=ax_r)
    
    fig_r.tight_layout()
    plt.savefig(os.path.join(image_dir, args.filename_prefix + "return_boxplot.svg"), dpi=600, bbox_inches='tight', pad_inches=0.5)
    plt.show()

if __name__ == "__main__":
    monte_carlo_simulation()
