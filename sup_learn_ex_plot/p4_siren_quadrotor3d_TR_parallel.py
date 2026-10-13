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
    # SIREN 网络对学习率非常敏感，默认 1e-3 容易导致梯度爆炸或 Loss 震荡不降，改为 1e-4
    learning_rate = 1e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    filename_prefix = "siren_quadrotor3d_"
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

EXPERIMENT_NAME = "quadratic_siren"


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
        # 控制策略函数通常比较平滑，SIREN 默认的 omega_0=30 频率太高容易拟合出剧烈震荡的无效策略
        # 我们把 omega_0 降为 10（甚至 5）以匹配控制任务的平滑特性，大幅提高拟合精度。
        self.net = SIREN(in_features=9, out_features=4, hidden_features=64, hidden_layers=3, norm_scale=1.0, omega_0=10)

    def forward(self, x):
        return self.net(x)

def train_sp():
    """Train SIREN from the shared quadratic dataset; no data generation occurs here."""
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
        x_path=_script_path("data", "X_trajs.npy"),
        u_path=_script_path("data", "U_trajs.npy"),
        model_path=_script_path("model", "siren_quadrotor3d.pth"),
        result_path=_script_path("results", "quadratic_siren_training.json"),
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
    sp_path = _script_path("model", "siren_quadrotor3d.pth")
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

    final_error_lqr = np.array([np.linalg.norm(traj[-1, :3]) for traj in all_traj_x_lqr])
    final_error_sp = np.array([np.linalg.norm(traj[-1, :3]) for traj in all_traj_x_sp])
    success_lqr = final_error_lqr <= MONTE_CARLO_SUCCESS_TOL
    success_sp = final_error_sp <= MONTE_CARLO_SUCCESS_TOL
    raw_results = {
        "episode_seeds": np.asarray(episode_seeds, dtype=np.int64),
        "returns_lqr": np.asarray(returns_lqr),
        "returns_siren": np.asarray(returns_sp),
        "final_position_error_lqr": final_error_lqr,
        "final_position_error_siren": final_error_sp,
        "success_lqr": success_lqr,
        "success_siren": success_sp,
    }
    controller_summary = {
        "lqr": {
            "return_mean": float(np.mean(returns_lqr)),
            "return_std": float(np.std(returns_lqr)),
            "final_position_error_mean": float(np.mean(final_error_lqr)),
            "final_position_error_std": float(np.std(final_error_lqr)),
            "success_rate": float(np.mean(success_lqr)),
        },
        "siren": {
            "return_mean": float(np.mean(returns_sp)),
            "return_std": float(np.std(returns_sp)),
            "final_position_error_mean": float(np.mean(final_error_sp)),
            "final_position_error_std": float(np.std(final_error_sp)),
            "success_rate": float(np.mean(success_sp)),
        },
    }
    if hnn_loaded:
        final_error_hnn = np.array([np.linalg.norm(traj[-1, :3]) for traj in all_traj_x_hnn])
        success_hnn = final_error_hnn <= MONTE_CARLO_SUCCESS_TOL
        raw_results.update({
            "returns_hnn": np.asarray(returns_hnn),
            "final_position_error_hnn": final_error_hnn,
            "success_hnn": success_hnn,
        })
        controller_summary["hnn"] = {
            "return_mean": float(np.mean(returns_hnn)),
            "return_std": float(np.std(returns_hnn)),
            "final_position_error_mean": float(np.mean(final_error_hnn)),
            "final_position_error_std": float(np.std(final_error_hnn)),
            "success_rate": float(np.mean(success_hnn)),
        }
    os.makedirs(_script_path("results"), exist_ok=True)
    np.savez_compressed(_script_path("results", "quadratic_siren_monte_carlo_raw.npz"), **raw_results)
    with open(_script_path("results", "quadratic_siren_training.json"), "r", encoding="utf-8") as file:
        training_record = json.load(file)
    _save_json("results/quadratic_siren_monte_carlo_summary.json", {
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
    plt.close('all')

if __name__ == "__main__":
    train_sp()
    # monte_carlo_simulation()
