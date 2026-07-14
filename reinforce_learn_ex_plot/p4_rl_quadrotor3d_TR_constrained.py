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
from utils import DroneVisualizer
import cmaps

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


def train_rl():
    env = gym.make(Args.env_id, theoretic_mode=True, rl_mode=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = RLNet(state_dim, action_dim).to(Args.device)
    optimizer = optim.Adam(agent.parameters(), lr=Args.learning_rate, eps=1e-5)
    
    num_steps = Args.num_steps
    obs = torch.zeros((num_steps, state_dim)).to(Args.device)
    actions = torch.zeros((num_steps, action_dim)).to(Args.device)
    logprobs = torch.zeros((num_steps,)).to(Args.device)
    rewards = torch.zeros((num_steps,)).to(Args.device)
    dones = torch.zeros((num_steps,)).to(Args.device)
    values = torch.zeros((num_steps,)).to(Args.device)

    global_step = 0
    num_updates = Args.time_steps // num_steps
    
    next_obs, _ = env.reset(theoretic_mode=True, rl_mode=True)
    next_obs = torch.Tensor(next_obs).to(Args.device)
    next_done = torch.zeros(1).to(Args.device)

    start_time = time.time()
    print(f"Starting PPO training for {Args.time_steps} steps...")

    for update in range(1, num_updates + 1):
        # if time.time() - start_time > 3000:
        #     print(f"Time limit of 30s reached at update {update}. Stopping training.")
        #     break

        for step in range(0, num_steps):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done
            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs.unsqueeze(0))
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            
            action_np = action.cpu().numpy().flatten()
            
            # 将网络输出的 [-1, 1] 动作映射到真实动作空间 [u_min, u_max]
            u_min = env.action_space.low
            u_max = env.action_space.high
            scaled_action = action_np * (u_max - u_min) / 2.0 + (u_max + u_min) / 2.0
            
            next_obs_np, reward, terminated, truncated, info = env.step(scaled_action)
            
            rewards[step] = torch.tensor(reward).to(Args.device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(Args.device)
            next_done = torch.Tensor([terminated or truncated]).to(Args.device)
            
            if terminated or truncated:
                next_obs_np, _ = env.reset()
                next_obs = torch.Tensor(next_obs_np).to(Args.device)
                
        with torch.no_grad():
            next_value = agent.get_value(next_obs.unsqueeze(0)).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(Args.device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + Args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + Args.gamma * Args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1, state_dim))
        b_logprobs = logprobs.reshape((-1,))
        b_actions = actions.reshape((-1, action_dim))
        b_advantages = advantages.reshape((-1,))
        b_returns = returns.reshape((-1,))
        b_values = values.reshape((-1,))

        b_inds = np.arange(num_steps)
        for epoch in range(Args.ppo_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, num_steps, Args.batch_size):
                end = start + Args.batch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - Args.clip_param, 1 + Args.clip_param)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - 0.01 * entropy_loss + v_loss * 0.5

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
                
        if update % 5 == 0:
            print(f"Update: {update}/{num_updates}, Global Step: {global_step}, Policy Loss: {pg_loss.item():.4f}, Value Loss: {v_loss.item():.4f}")

    print("Total training time: ", time.time() - start_time, " seconds")

    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "rl_quadrotor3d_constrained.pth")
    torch.save(agent.state_dict(), model_path)
    print(f"RL PPO model saved to {model_path}")



def simulate_once():
    args = Args()
    device = args.device

    env_lqr = gym.make(args.env_id, theoretic_mode=True)
    env_rl = gym.make(args.env_id, theoretic_mode=True)
    env_sp = gym.make(args.env_id, theoretic_mode=True)
    env_hnn = gym.make(args.env_id, theoretic_mode=True)

    horizon = int(env_lqr.T / env_lqr.dt)
    A_np = env_lqr.A
    B_np = env_lqr.B
    Q_np = env_lqr.Q
    R_np = env_lqr.R
    m = env_lqr.metadata['m']
    g = env_lqr.metadata['g']
    u0 = env_lqr.u0

    from scipy.linalg import solve_continuous_are
    P = solve_continuous_are(A_np, B_np, Q_np, R_np)
    K = np.linalg.inv(R_np) @ B_np.T @ P

    def lqr_action(x_np):
        u = -K @ x_np + u0
        u = np.clip(u, env_lqr.action_space.low, env_lqr.action_space.high)
        return np.array(u, dtype=np.float32)

    # Load RL
    rl_model = RLNet(9, 4).to(device)
    rl_path = os.path.join("model", "rl_quadrotor3d_constrained.pth")
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
        u = np.clip(u, u_min, u_max)
        return np.array(u, dtype=np.float32)

    # Load SPNet
    try:
        from sup_learn_ex_plot.p4_sp_quadrotor3d_TR import SPNet
        sp_model = SPNet(env_sp).to(device)
        sp_path = os.path.join("model", "sp_quadrotor3d.pth")
        if os.path.exists(sp_path):
            sp_model.load_state_dict(torch.load(sp_path, map_location=device))
        sp_model.eval()
        def sp_action(x_np):
            x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)
            with torch.no_grad():
                u = sp_model(x).squeeze(0).cpu().numpy()
            u = np.clip(u, env_sp.action_space.low, env_sp.action_space.high)
            return np.array(u, dtype=np.float32)
        sp_loaded = True
    except Exception as e:
        print(f"SPNet not loaded: {e}")
        sp_loaded = False

    # Load HNN
    try:
        from pihnn_experiments_plot.p4_hnn_quadrotor3d_TR import HNN, get_pfpu
        hnn_model = HNN(env_hnn).to(device)
        hnn_path = os.path.join("model", "hnn_quadrotor3d_TR_v2_paperplot.pth")
        if os.path.exists(hnn_path):
            hnn_model.load_state_dict(torch.load(hnn_path, map_location=device))
        hnn_model.eval()
        R_torch = torch.tensor(R_np, dtype=torch.float32, device=device)
        R_inv = torch.linalg.inv(R_torch)
        def hnn_action(x_np):
            x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)
            with torch.no_grad():
                Lambda = hnn_model.get_lambda(x)
            pfpu = get_pfpu(x)
            pfpu_T = pfpu.transpose(1, 2)
            u_tmp = -0.5 * torch.matmul(pfpu_T, Lambda.unsqueeze(2)).squeeze(-1)
            u = (R_inv @ u_tmp.T).T.squeeze(0).detach().cpu().numpy() + u0
            u = np.clip(u, env_hnn.action_space.low, env_hnn.action_space.high)
            return np.array(u, dtype=np.float32)
        hnn_loaded = True
    except Exception as e:
        print(f"HNN not loaded: {e}")
        hnn_loaded = False

    seed = np.random.randint(0, 10000000)
    obs_lqr, _ = env_lqr.reset(seed=seed, theoretic_mode=True)
    obs_rl, _ = env_rl.reset(seed=seed, theoretic_mode=True)
    if sp_loaded: obs_sp, _ = env_sp.reset(seed=seed, theoretic_mode=True)
    if hnn_loaded: obs_hnn, _ = env_hnn.reset(seed=seed, theoretic_mode=True)

    ret_lqr = 0.0
    u_seq_lqr = []
    x_seq_lqr = [obs_lqr.copy()]
    
    ret_rl = 0.0
    u_seq_rl = []
    x_seq_rl = [obs_rl.copy()]

    ret_sp = 0.0
    u_seq_sp = []
    x_seq_sp = [obs_sp.copy()] if sp_loaded else []

    ret_hnn = 0.0
    u_seq_hnn = []
    x_seq_hnn = [obs_hnn.copy()] if hnn_loaded else []

    lqr_trunc_count = 0
    rl_trunc_count = 0
    sp_trunc_count = 0
    hnn_trunc_count = 0

    print("Running simulate_once...")
    for t in range(horizon):
        # LQR
        u_l = lqr_action(obs_lqr.astype(np.float32))
        next_obs_l, r_l, term_l, trunc_l, _ = env_lqr.step(u_l)
        ret_lqr += float(r_l)
        u_seq_lqr.append(u_l)
        obs_lqr = next_obs_l
        x_seq_lqr.append(obs_lqr.copy())
        if not(term_l or trunc_l): lqr_trunc_count += 1
        
        # RL
        u_r = rl_action(obs_rl.astype(np.float32))
        next_obs_r, r_r, term_r, trunc_r, _ = env_rl.step(u_r)
        ret_rl += float(r_r)
        u_seq_rl.append(u_r)
        obs_rl = next_obs_r
        x_seq_rl.append(obs_rl.copy())
        if not(term_r or trunc_r): rl_trunc_count += 1

        # SPNet
        if sp_loaded:
            u_s = sp_action(obs_sp.astype(np.float32))
            next_obs_s, r_s, term_s, trunc_s, _ = env_sp.step(u_s)
            ret_sp += float(r_s)
            u_seq_sp.append(u_s)
            obs_sp = next_obs_s
            x_seq_sp.append(obs_sp.copy())
            if not(term_s or trunc_s): sp_trunc_count += 1
            
        # HNN
        if hnn_loaded:
            u_h = hnn_action(obs_hnn.astype(np.float32))
            next_obs_h, r_h, term_h, trunc_h, _ = env_hnn.step(u_h)
            ret_hnn += float(r_h)
            u_seq_hnn.append(u_h)
            obs_hnn = next_obs_h
            x_seq_hnn.append(obs_hnn.copy())
            if not(term_h or trunc_h): hnn_trunc_count += 1

    env_lqr.close()
    env_rl.close()
    if sp_loaded: env_sp.close()
    if hnn_loaded: env_hnn.close()

    traj_u_lqr = np.vstack(u_seq_lqr)
    traj_x_lqr = np.vstack(x_seq_lqr)
    traj_u_rl = np.vstack(u_seq_rl)
    traj_x_rl = np.vstack(x_seq_rl)
    
    if sp_loaded:
        traj_u_sp = np.vstack(u_seq_sp)
        traj_x_sp = np.vstack(x_seq_sp)
    if hnn_loaded:
        traj_u_hnn = np.vstack(u_seq_hnn)
        traj_x_hnn = np.vstack(x_seq_hnn)

    # Draw trajectories
    time = np.arange(horizon)
    plt.figure(figsize=(12, 6))
    for i in range(4):
        plt.plot(time, traj_u_lqr[:, i], label=f"LQR u{i+1}", linestyle='--')
        plt.plot(time, traj_u_rl[:, i], label=f"RL u{i+1}")
    plt.xlabel("Time step")
    plt.ylabel("u")
    plt.title("Control trajectories (single episode)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    time_x = np.arange(horizon + 1)
    fig = plt.figure(figsize=(12, 14))
    axs = fig.subplots(9, 1, sharex=True)
    for i, ax in enumerate(axs):
        ax.plot(time_x, traj_x_lqr[:, i], label="LQR", color="tab:green")
        ax.plot(time_x, traj_x_rl[:, i], label="RL", color="tab:red")
        if sp_loaded:
            ax.plot(time_x, traj_x_sp[:, i], label="SPNet", color="tab:orange")
        if hnn_loaded:
            ax.plot(time_x, traj_x_hnn[:, i], label="HNN", color="tab:blue")
        ax.set_ylabel(f"x{i+1}")
        ax.grid(True, linestyle='--', alpha=0.4)
        if i == 0:
            ax.legend()
    axs[-1].set_xlabel("Time step")
    axs[2].set_ylim([-1, 1])
    fig.suptitle("State trajectories (single episode)")
    # plt.tight_layout()

    print("Single episode return:")
    print(f"LQR: {ret_lqr:.3f}")
    print(f"RL: {ret_rl:.3f}")
    if sp_loaded: print(f"SPNet: {ret_sp:.3f}")
    if hnn_loaded: print(f"HNN: {ret_hnn:.3f}")

    # 3D Drone Visualizer
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(base_dir, "image")
    os.makedirs(image_dir, exist_ok=True)
    
    cmap_lqr = cmaps.MPL_RdYlGn_r
    cmap_rl = cmaps.MPL_Reds
    drone_visualizer = DroneVisualizer(traj=traj_x_rl[:, 0:3], alpha_range=[0.2, 1])
    drone_visualizer.add_one_traj(traj=traj_x_rl[:rl_trunc_count, 0:3], attitudes=np.rad2deg(traj_x_rl[:rl_trunc_count, 6:9]), stride=50, colormap=cmap_rl) 
    drone_visualizer.add_one_traj(traj=traj_x_lqr[:lqr_trunc_count, 0:3], attitudes=np.rad2deg(traj_x_lqr[:lqr_trunc_count, 6:9]), stride=50, colormap=cmap_lqr)
    
    if hnn_loaded:
        cmap_hnn = cmaps.MPL_PuOr_r
        drone_visualizer.add_one_traj(traj=traj_x_hnn[:hnn_trunc_count, 0:3], attitudes=np.rad2deg(traj_x_hnn[:hnn_trunc_count, 6:9]), stride=50, colormap=cmap_hnn)

    drone_visualizer.ax.set_zlim([-1, 1])
    plt.savefig(os.path.join(image_dir, args.filename_prefix + 'quad_traj_single.svg'), dpi=600, bbox_inches='tight', pad_inches=0.5, transparent=True)

    # Control input styling (Nature inspired)
    nature_green = "#389826"   # LQR
    nature_red = "#E64B35"     # RL
    nature_purple = "#9558B2"  # HNN
    nature_orange = "#F39B7F"  # SP
    
    fig, axs = plt.subplots(4, 1, figsize=(6, 6.3), sharex=True)
    for i, ax in enumerate(axs):
        ax.plot(time, traj_u_lqr[:, i], color=nature_green, alpha=0.6, linewidth=3, label="LQR" if i==0 else "")
        ax.plot(time, traj_u_rl[:, i], color=nature_red, alpha=0.8, linewidth=3, label="RL" if i==0 else "")
        if hnn_loaded:
            ax.plot(time, traj_u_hnn[:, i], color=nature_purple, alpha=0.8, linewidth=3, label="HNN" if i==0 else "")
        if sp_loaded:
            ax.plot(time, traj_u_sp[:, i], color=nature_orange, alpha=0.8, linewidth=3, label="SPNet" if i==0 else "")
            
        ax.set_ylabel(['$T$\,(N)', '$p$\,(rad/s)', '$q$\,(rad/s)', '$r$\,(rad/s)'][i])
        ax.grid(True, linestyle='--', alpha=0.4)
        if i == 0:
            ax.legend(loc='upper right', ncol=2)
    axs[3].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, args.filename_prefix + 'control_single.svg'), dpi=600, bbox_inches='tight', pad_inches=0.5, transparent=True)
    
    plt.show()

def monte_carlo_simulation():
    np.random.seed(200)
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
    rl_path = os.path.join("model", "rl_quadrotor3d_constrained.pth")
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
        from sup_learn_ex_plot.p4_sp_quadrotor3d_TR import SPNet
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
        sp_loaded = True
    except Exception as e:
        print(f"Could not load SPNet: {e}")
        sp_loaded = False

    # 3. Load HNN
    try:
        from pihnn_experiments_plot.p4_hnn_quadrotor3d_TR import HNN, get_pfpu
        env_hnn = gym.make(args.env_id, theoretic_mode=True)
        hnn_model = HNN(env_hnn).to(device)
        hnn_path = os.path.join("model", "hnn_quadrotor3d_TR_v2_paperplot.pth")
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

    all_traj_x_lqr = []
    all_traj_u_lqr = []
    all_traj_x_sp = []
    all_traj_u_sp = []
    all_traj_x_hnn = []
    all_traj_u_hnn = []
    all_traj_x_rl = []
    all_traj_u_rl = []
    returns_lqr = []
    returns_sp = []
    returns_hnn = []
    returns_rl = []

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

        ret_lqr = 0.0
        ret_sp = 0.0
        ret_hnn = 0.0
        ret_rl = 0.0

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

    env_lqr.close()
    env_rl.close()
    if sp_loaded:
        env_sp.close()
    if hnn_loaded:
        env_hnn.close()

    # ========== 计算成功率并过滤发散轨迹 ==========
    final_dist_lqr = np.array([np.linalg.norm(traj[-1, 0:3]) for traj in all_traj_x_lqr])
    final_dist_rl = np.array([np.linalg.norm(traj[-1, 0:3]) for traj in all_traj_x_rl])
    if sp_loaded:
        final_dist_sp = np.array([np.linalg.norm(traj[-1, 0:3]) for traj in all_traj_x_sp])
    if hnn_loaded:
        final_dist_hnn = np.array([np.linalg.norm(traj[-1, 0:3]) for traj in all_traj_x_hnn])
        
    # 认为 距离 > 0.1 为失败
    success_mask_lqr = final_dist_lqr <= 0.2
    success_mask_rl = final_dist_rl <= 0.2
    if sp_loaded:
        success_mask_sp = final_dist_sp <= 0.2
    if hnn_loaded:
        success_mask_hnn = final_dist_hnn <= 0.2

    print("\nMonte Carlo 50 episodes Success Rate:")
    print(f"LQR: {np.mean(success_mask_lqr)*100:.1f}%")
    print(f"RL: {np.mean(success_mask_rl)*100:.1f}%")
    if sp_loaded:
        print(f"SPNet: {np.mean(success_mask_sp)*100:.1f}%")
    if hnn_loaded:
        print(f"HNN: {np.mean(success_mask_hnn)*100:.1f}%")

    def filter_traj(traj_list, mask):
        filtered = [t for i, t in enumerate(traj_list) if mask[i]]
        if len(filtered) == 0:
            return np.zeros((1, traj_list[0].shape[0], traj_list[0].shape[1]))
        return np.stack(filtered, axis=0)

    traj_u_lqr = filter_traj(all_traj_u_lqr, success_mask_lqr)
    traj_x_lqr = filter_traj(all_traj_x_lqr, success_mask_lqr)
    traj_u_rl = filter_traj(all_traj_u_rl, success_mask_rl)
    traj_x_rl = filter_traj(all_traj_x_rl, success_mask_rl)
    
    if sp_loaded:
        traj_u_sp = filter_traj(all_traj_u_sp, success_mask_sp)
        traj_x_sp = filter_traj(all_traj_x_sp, success_mask_sp)
    if hnn_loaded:
        traj_u_hnn = filter_traj(all_traj_u_hnn, success_mask_hnn)
        traj_x_hnn = filter_traj(all_traj_x_hnn, success_mask_hnn)

    colors = {"RL": "tab:red", "LQR": "tab:green"}
    if sp_loaded:
        colors["SPNet"] = "tab:orange"
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
        plot_mean_std(ax, traj_u_rl[:, :, i], "RL", colors["RL"])
        if sp_loaded:
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
        plot_mean_std(ax, traj_x_rl[:, :, idx], "RL", colors["RL"])
        if sp_loaded:
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

    filtered_returns_rl = np.array(returns_rl)[success_mask_rl]
    filtered_returns_lqr = np.array(returns_lqr)[success_mask_lqr]
    if sp_loaded:
        filtered_returns_sp = np.array(returns_sp)[success_mask_sp]
    if hnn_loaded:
        filtered_returns_hnn = np.array(returns_hnn)[success_mask_hnn]

    print("\nMonte Carlo 50 episodes Return Statistics (Successful only):")
    print(f"RL: mean={np.mean(filtered_returns_rl):.3f}, std={np.std(filtered_returns_rl):.3f}")
    if sp_loaded:
        print(f"SPNet: mean={np.mean(filtered_returns_sp):.3f}, std={np.std(filtered_returns_sp):.3f}")
    print(f"LQR: mean={np.mean(filtered_returns_lqr):.3f}, std={np.std(filtered_returns_lqr):.3f}")
    if hnn_loaded:
        print(f"HNN: mean={np.mean(filtered_returns_hnn):.3f}, std={np.std(filtered_returns_hnn):.3f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    
    def get_min_max(arrs):
        valid_arrs = [a for a in arrs if len(a) > 0]
        if not valid_arrs:
            return 0, 1
        return min([np.min(a) for a in valid_arrs]), max([np.max(a) for a in valid_arrs])

    arrs_to_check = [filtered_returns_rl, filtered_returns_lqr]
    if sp_loaded:
        arrs_to_check.append(filtered_returns_sp)
    if hnn_loaded:
        arrs_to_check.append(filtered_returns_hnn)
        
    min_val, max_val = get_min_max(arrs_to_check)
    bins = np.linspace(min_val, max_val, 20)
    
    if len(filtered_returns_rl) > 0:
        plt.hist(filtered_returns_rl, bins=bins, alpha=0.5, label="RL", color=colors["RL"])
    if sp_loaded and len(filtered_returns_sp) > 0:
        plt.hist(filtered_returns_sp, bins=bins, alpha=0.5, label="SPNet", color=colors["SPNet"])
    if len(filtered_returns_lqr) > 0:
        plt.hist(filtered_returns_lqr, bins=bins, alpha=0.5, label="LQR", color=colors["LQR"])
    if hnn_loaded and len(filtered_returns_hnn) > 0:
        plt.hist(filtered_returns_hnn, bins=bins, alpha=0.5, label="HNN", color=colors["HNN"])
        
    plt.xlabel("Total Return")
    plt.ylabel("Frequency")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, args.filename_prefix + "hist_returns.svg"), dpi=600, bbox_inches='tight', pad_inches=0.5)
    plt.show()

if __name__ == "__main__":
    train_rl()
    # simulate_once()
    monte_carlo_simulation()
