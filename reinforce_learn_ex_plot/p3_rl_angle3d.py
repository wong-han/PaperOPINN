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
    learning_rate = 3e-4
    time_steps = 200000
    num_steps = 2048
    batch_size = 64
    ppo_epochs = 10
    clip_param = 0.2
    gamma = 0.99
    gae_lambda = 0.95
    device = "cuda" if torch.cuda.is_available() else "cpu"
    filename_prefix = "rl_angle3d_"


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
    env = gym.make(Args.env_id, theoretic_mode=True)
    state_dim = env.observation_space.shape[0]
    action_dim = 3  # For angle3d
    
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
    
    next_obs, _ = env.reset()
    next_obs = torch.Tensor(next_obs).to(Args.device)
    next_done = torch.zeros(1).to(Args.device)

    start_time = time.time()
    print(f"Starting PPO training for {Args.time_steps} steps...")

    for update in range(1, num_updates + 1):
        # if time.time() - start_time > 300:
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
            next_obs_np, reward, terminated, truncated, info = env.step(action_np)
            
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

    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "rl_angle3d.pth")
    torch.save(agent.state_dict(), model_path)
    print(f"RL PPO model saved to {model_path}")


def monte_carlo_simulation():
    args = Args()
    device = args.device
    env = gym.make(args.env_id)

    # 1. Load RL
    rl_model = RLNet(3, 3).to(device)
    rl_path = os.path.join("model", "rl_angle3d.pth")
    if os.path.exists(rl_path):
        rl_model.load_state_dict(torch.load(rl_path, map_location=device))
    rl_model.eval()

    def rl_action(x_np):
        x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)
        with torch.no_grad():
            u = rl_model.get_action(x).squeeze(0).cpu().numpy()
        return u

    # 2. Load SPNet
    try:
        from sup_learn_ex_plot.p3_sp_angle3d import SPNet
        sp_model = SPNet(env).to(device)
        sp_path = os.path.join("model", "sp_angle3d.pth")
        if os.path.exists(sp_path):
            sp_model.load_state_dict(torch.load(sp_path, map_location=device))
        sp_model.eval()
        def sp_action(x_np):
            x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)
            with torch.no_grad():
                u = sp_model(x).squeeze(0).cpu().numpy()
            return u
        sp_loaded = True
    except:
        sp_loaded = False

    # 3. Load HNN
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
        print(f"HNN not loaded: {e}")
        hnn_loaded = False

    def lqr_action(x_np):
        return -x_np

    num_episodes = 50
    horizon = 200

    results = {"RL": {"returns": [], "traj_u": [], "traj_x": []},
               "LQR": {"returns": [], "traj_u": [], "traj_x": []}}
    if sp_loaded:
        results["SPNet"] = {"returns": [], "traj_u": [], "traj_x": []}
    if hnn_loaded:
        results["HNN"] = {"returns": [], "traj_u": [], "traj_x": []}

    actions_dict = {"RL": rl_action, "LQR": lqr_action}
    if sp_loaded:
        actions_dict["SPNet"] = sp_action
    if hnn_loaded:
        actions_dict["HNN"] = hnn_action

    print("Starting Monte Carlo simulation...")
    for ep in range(num_episodes):
        if (ep+1) % 10 == 0:
            print(f"Episode {ep+1}/{num_episodes}")
            
        seed = int(np.random.randint(0, 2**31 - 1))
        
        env_dict = {name: gym.make(args.env_id) for name in actions_dict.keys()}
        obs_dict = {}
        for name in env_dict.keys():
            obs, _ = env_dict[name].reset(seed=seed, theoretic_mode=True)
            obs_dict[name] = obs
            
        ret_dict = {name: 0.0 for name in actions_dict.keys()}
        u_seq_dict = {name: [] for name in actions_dict.keys()}
        x_seq_dict = {name: [obs_dict[name].copy()] for name in actions_dict.keys()}

        for _ in range(horizon):
            for name in actions_dict.keys():
                u = actions_dict[name](obs_dict[name])
                next_obs, r, term, trunc, _ = env_dict[name].step(u.astype(np.float32))
                ret_dict[name] += float(r)
                u_seq_dict[name].append(u)
                obs_dict[name] = next_obs if not (term or trunc) else obs_dict[name]
                x_seq_dict[name].append(obs_dict[name].copy())

        for name in actions_dict.keys():
            env_dict[name].close()
            results[name]["traj_u"].append(np.stack(u_seq_dict[name], axis=0))
            results[name]["traj_x"].append(np.stack(x_seq_dict[name], axis=0))
            results[name]["returns"].append(ret_dict[name])

    for name in actions_dict.keys():
        results[name]["traj_u"] = np.stack(results[name]["traj_u"], axis=0)
        results[name]["traj_x"] = np.stack(results[name]["traj_x"], axis=0)
        results[name]["returns"] = np.array(results[name]["returns"])

    colors = {"RL": "tab:red", "SPNet": "tab:orange", "LQR": "tab:green", "HNN": "tab:blue"}

    def plot_mean_std(ax, data, label, color):
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        t = np.arange(len(mean))
        ax.plot(t, mean, label=label, color=color)
        ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.2)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(base_dir, "image")
    os.makedirs(image_dir, exist_ok=True)
    fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    labels_u = ['p', 'q', 'r']
    for i in range(3):
        ax = axs[i]
        for name in actions_dict.keys():
            plot_mean_std(ax, results[name]["traj_u"][:, :, i], name, colors[name])
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
        for name in actions_dict.keys():
            plot_mean_std(ax, results[name]["traj_x"][:, :, i], name, colors[name])
        ax.set_ylabel(labels_x[i])
        if i == 0:
            ax.legend(loc='upper right')
        ax.grid(True)
    axs[2].set_xlabel("Time step")
    fig.tight_layout()
    plt.savefig(os.path.join(image_dir, args.filename_prefix + "traj_x.svg"), dpi=600, bbox_inches='tight', pad_inches=0.5)

    for name in actions_dict.keys():
        print(f"{name}: mean={results[name]['returns'].mean():.3f}, std={results[name]['returns'].std():.3f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    min_val = min([results[name]["returns"].min() for name in actions_dict.keys()])
    max_val = max([results[name]["returns"].max() for name in actions_dict.keys()])
    bins = np.linspace(min_val, max_val, 20)
    
    for name in actions_dict.keys():
        plt.hist(results[name]["returns"], bins=bins, alpha=0.5, label=name, color=colors[name])
    plt.xlabel("Total Return")
    plt.ylabel("Frequency")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, args.filename_prefix + "hist_returns.svg"), dpi=600, bbox_inches='tight', pad_inches=0.5)
    plt.show()

if __name__ == "__main__":
    train_rl()
    monte_carlo_simulation()
