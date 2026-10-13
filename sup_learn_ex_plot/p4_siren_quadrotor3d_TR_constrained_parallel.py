import sys, os
import json
import time
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils_folder.SIREN import SIREN
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import gymnasium as gym
from scipy.linalg import solve_continuous_are
from supervised_training_common import train_from_saved_trajectories


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

    learning_rate = 1e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    filename_prefix = "siren_quadrotor3d_constrained_"
    batch_size = 2048


# ================== Unified experiment settings ==================
EXPERIMENT_SEED = 2026
MONTE_CARLO_EPISODES = 50
MONTE_CARLO_SEED = 100
MONTE_CARLO_SUCCESS_TOL = 0.1

EVAL_INTERVAL = 100
MAX_TRAIN_STEPS = 100000
TARGET_TRAINING_MSE = 1e-4
REQUIRED_TARGET_CHECKS = 5

EXPERIMENT_NAME = "constrained_siren"


def _sync_device():
    if Args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _save_json(path, payload):
    if not os.path.isabs(path):
        path = os.path.join(SCRIPT_DIR, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _script_path(*parts):
    return os.path.join(SCRIPT_DIR, *parts)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class SPNet(nn.Module):
    def __init__(self, env):
        self.env = env.unwrapped if hasattr(env, 'unwrapped') else env
        super().__init__()


        self.net = SIREN(in_features=9, out_features=4, hidden_features=64, hidden_layers=3, norm_scale=1.0, omega_0=10)

    def forward(self, x):
        return self.net(x)

def train_sp():
    """Train SIREN from the shared constrained dataset; no data generation occurs here."""
    np.random.seed(EXPERIMENT_SEED)
    torch.manual_seed(EXPERIMENT_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(EXPERIMENT_SEED)
    env = gym.make(Args.env_id).unwrapped
    model = SPNet(env).to(Args.device)
    optimizer = optim.Adam(model.parameters(), lr=Args.learning_rate)
    return train_from_saved_trajectories(
        model=model,
        optimizer=optimizer,
        device=Args.device,
        experiment_name=EXPERIMENT_NAME,
        x_path=_script_path("data", "X_trajs_constrained.npy"),
        u_path=_script_path("data", "U_trajs_constrained.npy"),
        model_path=_script_path("model", "siren_quadrotor3d_constrained.pth"),
        result_path=_script_path("results", "constrained_siren_training.json"),
        training_seed=EXPERIMENT_SEED,
        requested_batch_size=Args.batch_size,
        evaluation_interval=EVAL_INTERVAL,
        maximum_training_steps=MAX_TRAIN_STEPS,
        target_training_mse=TARGET_TRAINING_MSE,
        required_target_checks=REQUIRED_TARGET_CHECKS,
    )


def monte_carlo_simulation():
    np.random.seed(MONTE_CARLO_SEED)
    args = Args()
    device = args.device
    num_episodes = MONTE_CARLO_EPISODES

    env_lqr = gym.make(args.env_id, theoretic_mode=True).unwrapped
    env_sp = gym.make(args.env_id, theoretic_mode=True).unwrapped
    horizon = int(env_lqr.T / env_lqr.dt)

    A = env_lqr.A
    B = env_lqr.B
    Q = env_lqr.Q
    R = env_lqr.R
    u0 = env_lqr.u0

    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P

    sp_model = SPNet(env_sp).to(device)
    sp_path = _script_path("model", "siren_quadrotor3d_constrained.pth")
    if not os.path.exists(sp_path):
        raise FileNotFoundError(f"Trained model not found: {sp_path}")
    sp_model.load_state_dict(torch.load(sp_path, map_location=device))
    sp_model.eval()

    def sp_action(x_np):
        x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)
        with torch.no_grad():
            u = sp_model(x).squeeze(0).cpu().numpy()
        return u

    try:
        from pihnn_experiments_plot.p4_hnn_quadrotor3d_TR import HNN, get_pfpu
        env_hnn = gym.make(args.env_id, theoretic_mode=True).unwrapped
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
    finally:
        os.chdir(SCRIPT_DIR)

    all_traj_x_lqr = []
    all_traj_u_lqr = []
    all_traj_x_sp = []
    all_traj_u_sp = []
    all_traj_x_hnn = []
    all_traj_u_hnn = []
    returns_lqr = []
    returns_sp = []
    returns_hnn = []
    episode_seeds = []

    print("Starting Monte Carlo simulation...")
    _sync_device()
    monte_carlo_start = time.perf_counter()
    for ep in range(num_episodes):
        if (ep+1) % 1 == 0:
            print(f"Episode {ep+1}/{num_episodes}")
            
        seed = np.random.randint(0, 1000000)
        episode_seeds.append(int(seed))
        
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

    _sync_device()
    monte_carlo_seconds = time.perf_counter() - monte_carlo_start

    env_lqr.close()
    env_sp.close()
    if hnn_loaded:
        env_hnn.close()


    final_dist_lqr = np.array([np.linalg.norm(traj[-1, 0:3]) for traj in all_traj_x_lqr])
    final_dist_sp = np.array([np.linalg.norm(traj[-1, 0:3]) for traj in all_traj_x_sp])
    if hnn_loaded:
        final_dist_hnn = np.array([np.linalg.norm(traj[-1, 0:3]) for traj in all_traj_x_hnn])
        

    success_mask_lqr = final_dist_lqr <= MONTE_CARLO_SUCCESS_TOL
    success_mask_sp = final_dist_sp <= MONTE_CARLO_SUCCESS_TOL
    if hnn_loaded:
        success_mask_hnn = final_dist_hnn <= MONTE_CARLO_SUCCESS_TOL

    print("\nMonte Carlo 50 episodes Success Rate:")
    print(f"LQR: {np.mean(success_mask_lqr)*100:.1f}%")
    print(f"SPNet: {np.mean(success_mask_sp)*100:.1f}%")
    if hnn_loaded:
        print(f"HNN: {np.mean(success_mask_hnn)*100:.1f}%")

    def filter_traj(traj_list, mask):
        filtered = [t for i, t in enumerate(traj_list) if mask[i]]
        if len(filtered) == 0:
            return np.zeros((1, traj_list[0].shape[0], traj_list[0].shape[1]))
        return np.stack(filtered, axis=0)

    traj_u_sp = filter_traj(all_traj_u_sp, success_mask_sp)
    traj_x_sp = filter_traj(all_traj_x_sp, success_mask_sp)
    traj_u_lqr = filter_traj(all_traj_u_lqr, success_mask_lqr)
    traj_x_lqr = filter_traj(all_traj_x_lqr, success_mask_lqr)
    if hnn_loaded:
        traj_u_hnn = filter_traj(all_traj_u_hnn, success_mask_hnn)
        traj_x_hnn = filter_traj(all_traj_x_hnn, success_mask_hnn)

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

    filtered_returns_lqr = np.array(returns_lqr)[success_mask_lqr]
    filtered_returns_sp = np.array(returns_sp)[success_mask_sp]
    if hnn_loaded:
        filtered_returns_hnn = np.array(returns_hnn)[success_mask_hnn]

    print("\nMonte Carlo 50 episodes Return Statistics (Successful only):")
    print(f"LQR: mean={np.mean(filtered_returns_lqr):.3f}, std={np.std(filtered_returns_lqr):.3f}")
    print(f"SPNet: mean={np.mean(filtered_returns_sp):.3f}, std={np.std(filtered_returns_sp):.3f}")
    if hnn_loaded:
        print(f"HNN: mean={np.mean(filtered_returns_hnn):.3f}, std={np.std(filtered_returns_hnn):.3f}")


    min_y_lqr = np.array([np.min(traj[:, 1]) for traj in all_traj_x_lqr])
    min_y_sp = np.array([np.min(traj[:, 1]) for traj in all_traj_x_sp])
    if hnn_loaded:
        min_y_hnn = np.array([np.min(traj[:, 1]) for traj in all_traj_x_hnn])

    raw_results = {
        "episode_seeds": np.asarray(episode_seeds, dtype=np.int64),
        "returns_lqr": np.asarray(returns_lqr),
        "returns_siren": np.asarray(returns_sp),
        "final_position_error_lqr": final_dist_lqr,
        "final_position_error_siren": final_dist_sp,
        "minimum_y_lqr": min_y_lqr,
        "minimum_y_siren": min_y_sp,
        "success_lqr": success_mask_lqr,
        "success_siren": success_mask_sp,
    }
    controller_summary = {
        "lqr": {
            "return_mean": float(np.mean(returns_lqr)),
            "return_std": float(np.std(returns_lqr)),
            "final_position_error_mean": float(np.mean(final_dist_lqr)),
            "final_position_error_std": float(np.std(final_dist_lqr)),
            "minimum_y_mean": float(np.mean(min_y_lqr)),
            "constraint_violation_rate": float(np.mean(min_y_lqr < 0.0)),
            "success_rate": float(np.mean(success_mask_lqr)),
        },
        "siren": {
            "return_mean": float(np.mean(returns_sp)),
            "return_std": float(np.std(returns_sp)),
            "final_position_error_mean": float(np.mean(final_dist_sp)),
            "final_position_error_std": float(np.std(final_dist_sp)),
            "minimum_y_mean": float(np.mean(min_y_sp)),
            "constraint_violation_rate": float(np.mean(min_y_sp < 0.0)),
            "success_rate": float(np.mean(success_mask_sp)),
        },
    }
    if hnn_loaded:
        raw_results.update({
            "returns_hnn": np.asarray(returns_hnn),
            "final_position_error_hnn": final_dist_hnn,
            "minimum_y_hnn": min_y_hnn,
            "success_hnn": success_mask_hnn,
        })
        controller_summary["hnn"] = {
            "return_mean": float(np.mean(returns_hnn)),
            "return_std": float(np.std(returns_hnn)),
            "final_position_error_mean": float(np.mean(final_dist_hnn)),
            "final_position_error_std": float(np.std(final_dist_hnn)),
            "minimum_y_mean": float(np.mean(min_y_hnn)),
            "constraint_violation_rate": float(np.mean(min_y_hnn < 0.0)),
            "success_rate": float(np.mean(success_mask_hnn)),
        }
    os.makedirs(_script_path("results"), exist_ok=True)
    np.savez_compressed(_script_path("results", "constrained_siren_monte_carlo_raw.npz"), **raw_results)
    with open(_script_path("results", "constrained_siren_training.json"), "r", encoding="utf-8") as file:
        training_record = json.load(file)
    _save_json("results/constrained_siren_monte_carlo_summary.json", {
        "experiment": EXPERIMENT_NAME,
        "seed": MONTE_CARLO_SEED,
        "episodes": num_episodes,
        "horizon_steps": horizon,
        "success_position_l2_tolerance": MONTE_CARLO_SUCCESS_TOL,
        "monte_carlo_seconds": monte_carlo_seconds,
        "seconds_per_episode": monte_carlo_seconds / num_episodes,
        "training_seconds": training_record["training_seconds"],
        "training_steps": training_record["steps"],
        "training_stop_reason": training_record["stop_reason"],
        "controllers": controller_summary,
    })
    print(f"Monte Carlo testing finished in {monte_carlo_seconds:.2f} seconds.")
        
    filtered_min_y_lqr = min_y_lqr[success_mask_lqr]
    filtered_min_y_sp = min_y_sp[success_mask_sp]
    if hnn_loaded:
        filtered_min_y_hnn = min_y_hnn[success_mask_hnn]

    import seaborn as sns
    if hnn_loaded:
        palette = ["#f6c85f", "#ff9f43", "#88bde6"]  # LQR, SPNet, HNN
        data_to_plot_J = [-filtered_returns_lqr, -filtered_returns_sp, -filtered_returns_hnn]
        labels_J = [r"LQR", r"SPNet", r"HNN"]
    else:
        palette = ["#f6c85f", "#ff9f43"]
        data_to_plot_J = [-filtered_returns_lqr, -filtered_returns_sp]
        labels_J = [r"LQR", r"SPNet"]
        
    boxprops = dict(linewidth=2, color='k')
    medianprops = dict(linewidth=2, color="#d24d57")
    whiskerprops = dict(linewidth=1.5, color='k')
    capprops = dict(linewidth=1.5, color='k')
    flierprops = dict(marker='o', markerfacecolor='#ff7f0e', markersize=7, linestyle='none', alpha=0.3, markeredgecolor='k')


    fig, ax = plt.subplots(figsize=(6, 8))
    bp = ax.boxplot(
        data_to_plot_J,
        tick_labels=labels_J,
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
    for i, y_data in enumerate(data_to_plot_J):
        x_jitter = np.random.normal(0, 0.07, size=len(y_data))
        ax.scatter(np.full(len(y_data), i+1) + x_jitter, y_data, color=palette[i], alpha=0.33, s=50, edgecolor='k', linewidth=0.2, zorder=2)
    ax.set_ylabel(r"$J$")
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')
    ax.grid(True, linestyle='--', alpha=0.45)
    sns.despine(top=False, right=False, left=False, bottom=False, ax=ax)
    plt.savefig(os.path.join(image_dir, args.filename_prefix + "return_boxplot.svg"), dpi=600, bbox_inches='tight', pad_inches=0.5)


    import pandas as pd
    eps = 1e-8
    log_abs_ymin_lqr = np.log10(np.abs(filtered_min_y_lqr) + eps)
    log_abs_ymin_sp = np.log10(np.abs(filtered_min_y_sp) + eps)
    
    if hnn_loaded:
        log_abs_ymin_hnn = np.log10(np.abs(filtered_min_y_hnn) + eps)
        df_logymin = pd.DataFrame({
            r"$\log_{10}|y_{\mathrm{min}}|$": np.concatenate([log_abs_ymin_lqr, log_abs_ymin_sp, log_abs_ymin_hnn]),
            "Controller": ["LQR"] * len(log_abs_ymin_lqr) + ["SPNet"] * len(log_abs_ymin_sp) + ["HNN"] * len(log_abs_ymin_hnn)
        })
        violin_palette = ["#389826", "#e67e22", "#9558B2"]
    else:
        df_logymin = pd.DataFrame({
            r"$\log_{10}|y_{\mathrm{min}}|$": np.concatenate([log_abs_ymin_lqr, log_abs_ymin_sp]),
            "Controller": ["LQR"] * len(log_abs_ymin_lqr) + ["SPNet"] * len(log_abs_ymin_sp)
        })
        violin_palette = ["#389826", "#e67e22"]

    fig2, ax2 = plt.subplots(figsize=(6, 7))
    sns.violinplot(
        data=df_logymin, x="Controller", y=r"$\log_{10}|y_{\mathrm{min}}|$",
        hue="Controller", palette=violin_palette, legend=False,
        inner=None, linewidth=0, alpha=0.3, ax=ax2
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
        hue="Controller", palette=violin_palette, legend=False,
        alpha=0.4, size=6, jitter=0.15, edgecolor='k', linewidth=0.5, zorder=3, ax=ax2
    )
    
    ax2.axhline(np.log10(1e-2), color='gray', linestyle='--', linewidth=1.5, zorder=0)
    ax2.set_xlabel("")
    ax2.set_ylabel(r"$\log_{10}|y_{\mathrm{min}}|$")
    ax2.tick_params(axis='x')
    ax2.grid(True, linestyle='--', alpha=0.45)
    sns.despine(top=False, right=False, left=False, bottom=False, ax=ax2)
    plt.savefig(os.path.join(image_dir, args.filename_prefix + "ymin_violinplot.svg"), dpi=600, bbox_inches='tight', pad_inches=0.5)
    plt.close('all')

if __name__ == "__main__":
    train_sp()
    # monte_carlo_simulation()
