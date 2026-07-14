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
import pandas as pd
import seaborn as sns

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
    env_id = "environment:quadrotor3d_TR-v0"
    learning_rate = 3e-4
    time_steps = 1000000
    num_steps = 2048
    batch_size = 64
    ppo_epochs = 10
    clip_param = 0.2
    gamma = 0.99
    gae_lambda = 0.95
    device = "cuda" if torch.cuda.is_available() else "cpu"
    filename_prefix = "rl_quadrotor3d_"


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
    rl_path = os.path.join("..", "reinforce_learn_ex_plot", "model", "rl_quadrotor3d.pth")
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
    from sup_learn_ex_plot.p4_sp_quadrotor3d_TR import SPNet
    sp_model = SPNet(env_sp).to(device)
    sp_path = os.path.join("..", "sup_learn_ex_plot", "model", "sp_quadrotor3d.pth")
    sp_model.load_state_dict(torch.load(sp_path, map_location=device))
    sp_model.eval()
    def sp_action(x_np):
        x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)
        with torch.no_grad():
            u = sp_model(x).squeeze(0).cpu().numpy()
        return u
    sp_loaded = True

    # 3. Load HNN
    from pihnn_experiments_plot.p4_hnn_quadrotor3d_TR import HNN, get_pfpu
    env_hnn = gym.make(args.env_id, theoretic_mode=True)
    hnn_model = HNN(env_hnn).to(device)
    hnn_path = os.path.join("..", "pihnn_experiments_plot", "model", "hnn_quadrotor3d_TR_v2_paperplot.pth")
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

    # 4. Load SIREN
    env_siren = gym.make(args.env_id, theoretic_mode=True)
    try:
        from sup_learn_ex_plot_with_siren.p4_sp_quadrotor3d_TR import SPNet as SIRENNet
        siren_model = SIRENNet(env_siren).to(device)
        siren_path = os.path.join("..", "sup_learn_ex_plot_with_siren", "model", "sp_quadrotor3d.pth")
        if os.path.exists(siren_path):
            siren_model.load_state_dict(torch.load(siren_path, map_location=device))
        siren_model.eval()
        def siren_action(x_np):
            x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)
            with torch.no_grad():
                u = siren_model(x).squeeze(0).cpu().numpy()
            return u
        siren_loaded = True
    except Exception as e:
        print(f"Could not load SIREN: {e}")
        siren_loaded = False

    all_traj_x_lqr = []
    all_traj_u_lqr = []
    all_traj_x_sp = []
    all_traj_u_sp = []
    all_traj_x_hnn = []
    all_traj_u_hnn = []
    all_traj_x_rl = []
    all_traj_u_rl = []
    all_traj_x_siren = []
    all_traj_u_siren = []
    returns_lqr = []
    returns_sp = []
    returns_hnn = []
    returns_rl = []
    returns_siren = []

#     print("Starting Monte Carlo simulation...")
#     for ep in range(num_episodes):
#         if (ep+1) % 1 == 0:
#             print(f"Episode {ep+1}/{num_episodes}")
#             
#         seed = np.random.randint(0, 1000000)
#         
#         obs_lqr, _ = env_lqr.reset(seed=seed, theoretic_mode=True)
#         obs_rl, _ = env_rl.reset(seed=seed, theoretic_mode=True)
#         if sp_loaded:
#             obs_sp, _ = env_sp.reset(seed=seed, theoretic_mode=True)
#         if hnn_loaded:
#             obs_hnn, _ = env_hnn.reset(seed=seed, theoretic_mode=True)
#         if siren_loaded:
#             obs_siren, _ = env_siren.reset(seed=seed, theoretic_mode=True)
# 
# 
#         x_seq_lqr = [obs_lqr.copy()]
#         u_seq_lqr = []
#         x_seq_rl = [obs_rl.copy()]
#         u_seq_rl = []
#         if sp_loaded:
#             x_seq_sp = [obs_sp.copy()]
#             u_seq_sp = []
#         if hnn_loaded:
#             x_seq_hnn = [obs_hnn.copy()]
#             u_seq_hnn = []
#         if siren_loaded:
#             x_seq_siren = [obs_siren.copy()]
#             u_seq_siren = []
# 
#         ret_lqr = 0.0
#         ret_sp = 0.0
#         ret_hnn = 0.0
#         ret_rl = 0.0
#         ret_siren = 0.0
# 
#         # LQR episode
#         obs_lqr_ep = obs_lqr.copy()
#         for t in range(horizon):
#             u_l = -K @ obs_lqr_ep + u0
#             u_l = np.clip(u_l, env_lqr.action_space.low, env_lqr.action_space.high)
#             next_obs_l, reward_l, terminated_l, truncated_l, _ = env_lqr.step(u_l)
#             u_seq_lqr.append(u_l)
#             obs_lqr_ep = next_obs_l
#             x_seq_lqr.append(obs_lqr_ep.copy())
#             ret_lqr += float(reward_l)
# 
#         # RL episode
#         obs_rl_ep = obs_rl.copy()
#         for t in range(horizon):
#             u_r = rl_action(obs_rl_ep)
#             u_r = np.clip(u_r, env_rl.action_space.low, env_rl.action_space.high)
#             next_obs_r, reward_r, terminated_r, truncated_r, _ = env_rl.step(u_r)
#             u_seq_rl.append(u_r)
#             obs_rl_ep = next_obs_r
#             x_seq_rl.append(obs_rl_ep.copy())
#             ret_rl += float(reward_r)
# 
#         # SPNet episode
#         if sp_loaded:
#             obs_sp_ep = obs_sp.copy()
#             for t in range(horizon):
#                 u_s = sp_action(obs_sp_ep)
#                 u_s = np.clip(u_s, env_sp.action_space.low, env_sp.action_space.high)
#                 next_obs_s, reward_s, terminated_s, truncated_s, _ = env_sp.step(u_s)
#                 u_seq_sp.append(u_s)
#                 obs_sp_ep = next_obs_s
#                 x_seq_sp.append(obs_sp_ep.copy())
#                 ret_sp += float(reward_s)
# 
#         # HNN episode
#         if hnn_loaded:
#             obs_hnn_ep = obs_hnn.copy()
#             for t in range(horizon):
#                 u_h = hnn_action(obs_hnn_ep)
#                 u_h = np.clip(u_h, env_hnn.action_space.low, env_hnn.action_space.high)
#                 next_obs_h, reward_h, terminated_h, truncated_h, _ = env_hnn.step(u_h)
#                 u_seq_hnn.append(u_h)
#                 obs_hnn_ep = next_obs_h
#                 x_seq_hnn.append(obs_hnn_ep.copy())
#                 ret_hnn += float(reward_h)
# 
#         # SIREN episode
#         if siren_loaded:
#             obs_siren_ep = obs_siren.copy()
#             for t in range(horizon):
#                 u_m = siren_action(obs_siren_ep)
#                 u_m = np.clip(u_m, env_siren.action_space.low, env_siren.action_space.high)
#                 next_obs_m, reward_m, terminated_m, truncated_m, _ = env_siren.step(u_m)
#                 u_seq_siren.append(u_m)
#                 obs_siren_ep = next_obs_m
#                 x_seq_siren.append(obs_siren_ep.copy())
#                 ret_siren += float(reward_m)
# 
#         all_traj_x_lqr.append(np.vstack(x_seq_lqr))
#         all_traj_u_lqr.append(np.vstack(u_seq_lqr))
#         returns_lqr.append(ret_lqr)
# 
#         all_traj_x_rl.append(np.vstack(x_seq_rl))
#         all_traj_u_rl.append(np.vstack(u_seq_rl))
#         returns_rl.append(ret_rl)
# 
#         if sp_loaded:
#             all_traj_x_sp.append(np.vstack(x_seq_sp))
#             all_traj_u_sp.append(np.vstack(u_seq_sp))
#             returns_sp.append(ret_sp)
# 
#         if hnn_loaded:
#             all_traj_x_hnn.append(np.vstack(x_seq_hnn))
#             all_traj_u_hnn.append(np.vstack(u_seq_hnn))
#             returns_hnn.append(ret_hnn)
# 
#         if siren_loaded:
#             all_traj_x_siren.append(np.vstack(x_seq_siren))
#             all_traj_u_siren.append(np.vstack(u_seq_siren))
#             returns_siren.append(ret_siren)
# 
#     env_lqr.close()
#     env_rl.close()
#     if sp_loaded:
#         env_sp.close()
#     if hnn_loaded:
#         env_hnn.close()
#     if siren_loaded:
#         env_siren.close()
# 
#     # ========== 计算成功率并过滤发散轨迹 ==========
#     final_dist_lqr = np.array([np.linalg.norm(traj[-1, 0:3]) for traj in all_traj_x_lqr])
#     final_dist_rl = np.array([np.linalg.norm(traj[-1, 0:3]) for traj in all_traj_x_rl])
#     if sp_loaded:
#         final_dist_sp = np.array([np.linalg.norm(traj[-1, 0:3]) for traj in all_traj_x_sp])
#     if hnn_loaded:
#         final_dist_hnn = np.array([np.linalg.norm(traj[-1, 0:3]) for traj in all_traj_x_hnn])
#     if siren_loaded:
#         final_dist_siren = np.array([np.linalg.norm(traj[-1, 0:3]) for traj in all_traj_x_siren])
#         
#     # 认为 距离 > 0.1 为失败
#     success_mask_lqr = final_dist_lqr <= 0.1
#     success_mask_rl = final_dist_rl <= 0.1
#     if sp_loaded:
#         success_mask_sp = final_dist_sp <= 0.1
#     if hnn_loaded:
#         success_mask_hnn = final_dist_hnn <= 0.1
#     if siren_loaded:
#         success_mask_siren = final_dist_siren <= 0.1
# 
#     print("\nMonte Carlo 50 episodes Success Rate:")
#     print(f"LQR: {np.mean(success_mask_lqr)*100:.1f}%")
#     if siren_loaded:
#         print(f"SIREN: {np.mean(success_mask_siren)*100:.1f}%")
#     if sp_loaded:
#         print(f"SPNet: {np.mean(success_mask_sp)*100:.1f}%")
#     print(f"RL: {np.mean(success_mask_rl)*100:.1f}%")
#     if hnn_loaded:
#         print(f"HNN: {np.mean(success_mask_hnn)*100:.1f}%")
# 
#     filtered_returns_lqr = np.array(returns_lqr)[success_mask_lqr]
#     filtered_returns_rl = np.array(returns_rl)[success_mask_rl]
#     if sp_loaded:
#         filtered_returns_sp = np.array(returns_sp)[success_mask_sp]
#     if hnn_loaded:
#         filtered_returns_hnn = np.array(returns_hnn)[success_mask_hnn]
#     if siren_loaded:
#         filtered_returns_siren = np.array(returns_siren)[success_mask_siren]
# 
#     print("\nMonte Carlo 50 episodes Return Statistics (Successful only):")
#     print(f"LQR: mean={np.mean(filtered_returns_lqr):.3f}, std={np.std(filtered_returns_lqr):.3f}")
#     if siren_loaded:
#         print(f"SIREN: mean={np.mean(filtered_returns_siren):.3f}, std={np.std(filtered_returns_siren):.3f}")
#     if sp_loaded:
#         print(f"SPNet: mean={np.mean(filtered_returns_sp):.3f}, std={np.std(filtered_returns_sp):.3f}")
#     print(f"RL: mean={np.mean(filtered_returns_rl):.3f}, std={np.std(filtered_returns_rl):.3f}")
#     if hnn_loaded:
#         print(f"HNN: mean={np.mean(filtered_returns_hnn):.3f}, std={np.std(filtered_returns_hnn):.3f}")
# 
#     # Boxplot
#     data_returns = []
#     data_controllers = []
#     
#     data_returns.extend(-filtered_returns_lqr)
#     data_controllers.extend(["LQR"] * len(filtered_returns_lqr))
#     
#     if siren_loaded:
#         data_returns.extend(-filtered_returns_siren)
#         data_controllers.extend(["SIREN"] * len(filtered_returns_siren))
#         
#     if sp_loaded:
#         data_returns.extend(-filtered_returns_sp)
#         data_controllers.extend(["SPNet"] * len(filtered_returns_sp))
#         
#     data_returns.extend(-filtered_returns_rl)
#     data_controllers.extend(["RL"] * len(filtered_returns_rl))
#     
#     if hnn_loaded:
#         data_returns.extend(-filtered_returns_hnn)
#         data_controllers.extend(["HNN"] * len(filtered_returns_hnn))
#         
#     df_box = pd.DataFrame({"Return": data_returns, "Controller": data_controllers})
    
    # 颜色: HNN用紫色，RL用蓝色，SPNet用绿色，SIREN用红色，LQR用橙色
    order = ["LQR"]
    palette = ["#F39B7F"]  # Orange for LQR
    if siren_loaded:
        order.append("SIREN")
        palette.append("#E64B35")  # Red for SIREN
    if sp_loaded:
        order.append("MLP")
        palette.append("#389826")  # Green for SPNet
    order.append("RL")
    palette.append("#4B8BBE")  # Blue for RL
    if hnn_loaded:
        order.append("OPINN")
        palette.append("#9558B2")  # Purple for HNN
        
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(base_dir, "image")
    os.makedirs(image_dir, exist_ok=True)
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

#     # =============== 自动保存绘图数据 ===============
#     # 1. 保存供 Seaborn 直接使用的 DataFrame
#     df_box.to_csv(os.path.join(data_dir, args.filename_prefix + "return_boxplot_data.csv"), index=False)
#     
#     # 2. 保存原始回报和成功掩码，方便以后扩展分析
#     np.savez(os.path.join(data_dir, args.filename_prefix + "raw_returns.npz"),
#              returns_lqr=returns_lqr, success_mask_lqr=success_mask_lqr,
#              returns_rl=returns_rl, success_mask_rl=success_mask_rl,
#              returns_sp=returns_sp if sp_loaded else [], success_mask_sp=success_mask_sp if sp_loaded else [],
#              returns_hnn=returns_hnn if hnn_loaded else [], success_mask_hnn=success_mask_hnn if hnn_loaded else [],
#              returns_siren=returns_siren if siren_loaded else [], success_mask_siren=success_mask_siren if siren_loaded else [])
#              
#     # 3. 保存所有状态轨迹数据（处理成 numpy object 数组以允许不规则长度）
#     np.save(os.path.join(data_dir, args.filename_prefix + "all_traj_x_lqr.npy"), np.array(all_traj_x_lqr, dtype=object), allow_pickle=True)
#     np.save(os.path.join(data_dir, args.filename_prefix + "all_traj_x_rl.npy"), np.array(all_traj_x_rl, dtype=object), allow_pickle=True)
#     if sp_loaded:
#         np.save(os.path.join(data_dir, args.filename_prefix + "all_traj_x_sp.npy"), np.array(all_traj_x_sp, dtype=object), allow_pickle=True)
#     if hnn_loaded:
#         np.save(os.path.join(data_dir, args.filename_prefix + "all_traj_x_hnn.npy"), np.array(all_traj_x_hnn, dtype=object), allow_pickle=True)
#     if siren_loaded:
#         np.save(os.path.join(data_dir, args.filename_prefix + "all_traj_x_siren.npy"), np.array(all_traj_x_siren, dtype=object), allow_pickle=True)
#     
#     print(f"All Monte Carlo simulation data successfully saved to {data_dir}/")

    df_box = pd.read_csv(os.path.join(data_dir, args.filename_prefix + "return_boxplot_data.csv"))
    print("\n=== Unconstrained (Quadratic Cost) Average J ===")
    print(df_box.groupby("Controller")["Return"].mean())
    print("================================================\n")
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.violinplot(
        x="Controller", y="Return", data=df_box,
        order=order, palette=palette, inner=None, alpha=0.22, cut=0,
        linewidth=1.2, ax=ax
    )
    sns.boxplot(
        x="Controller", y="Return", data=df_box,
        order=order, palette=palette, width=0.25,
        showcaps=True, showbox=True, showfliers=False, medianprops=dict(color="#6D2D2B", linewidth=2),
        boxprops=dict(alpha=0.7, edgecolor='k', linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        ax=ax
    )
    sns.stripplot(
        x="Controller", y="Return", data=df_box,
        order=order, palette=palette,
        size=7, alpha=0.4, jitter=0.22, linewidth=0.2, edgecolor='#444', ax=ax
    )
    
    # Customize layout
    ax.set_xticklabels(order)
    ax.set_xlabel("")
    ax.set_ylabel(r"$J$", usetex=True)
    ax.grid(True, linestyle='--', alpha=0.45)
    sns.despine(top=False, right=False, left=False, bottom=False, ax=ax)
    
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, args.filename_prefix + "return_boxplot.svg"), dpi=600, bbox_inches='tight', pad_inches=0.5)
    plt.show()

if __name__ == "__main__":
    monte_carlo_simulation()
