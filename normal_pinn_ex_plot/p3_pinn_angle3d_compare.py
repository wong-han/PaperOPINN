'''
Comparison of HNN (OPINN), X-TFC PINN, and a supervised numerical-optimal-solution baseline.
Training is excluded; the HNN architecture and pretrained model are imported directly
from ``pihnn_experiments_plot``.
'''
import sys, os
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import gymnasium as gym

from sup_angle3d import PINNNetwork
from p3_pinn_angle3d import XTFC_PINN
from pihnn_experiments_plot.p3_2_hnn_angle3d import HNN, dynamics, get_pfpu


# Plot settings
mpl.rcParams['font.family'] = 'Times New Roman'  # Default font
mpl.rcParams['text.usetex'] = True  # Use TeX rendering
mpl.rcParams['axes.unicode_minus'] = False  # Render minus signs correctly
mpl.rcParams['xtick.direction'] = 'in'  # Draw ticks inward
mpl.rcParams['ytick.direction'] = 'in'  # Draw ticks inward
mpl.rcParams['xtick.top'] = True  # Show top ticks
mpl.rcParams['ytick.right'] = True  # Show right ticks

mpl.rcParams['legend.frameon'] = False  # Hide the legend frame
mpl.rcParams['legend.fontsize'] = 24  # Default legend font size

mpl.rcParams['axes.titlesize'] = 24   # Default title size
mpl.rcParams['xtick.labelsize'] = 24  # Default tick-label size
mpl.rcParams['ytick.labelsize'] = 24  # Default tick-label size
mpl.rcParams['axes.labelsize'] = 24  # Default axis-label size
mpl.rcParams['lines.linewidth'] = 2  # Line width
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.linestyle'] = '--'  # Dashed grid lines
mpl.rcParams['grid.alpha'] = 0.5       # Transparency
mpl.rcParams['grid.color'] = 'gray'    # Grid color


# Parameters and paths
class Args:
    env_id = "environment:angle3d-v0"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    filename_prefix = "hnn_angle3d_"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HNN_MODEL_PATH = os.path.join(BASE_DIR, "..", "pihnn_experiments_plot", "model", "hnn_angle3d.pth")
if not os.path.exists(HNN_MODEL_PATH):
    HNN_MODEL_PATH = os.path.join(BASE_DIR, "model", "hnn_angle3d.pth")

SUP_MODEL_PATH = os.path.join(BASE_DIR, "model", "supervised_angle3d_model.pth")
if not os.path.exists(SUP_MODEL_PATH):
    SUP_MODEL_PATH = os.path.join(BASE_DIR, "..", "pihnn_experiments_plot", "model", "supervised_angle3d_model.pth")

XTFC_MODEL_PATH = os.path.join(BASE_DIR, "model", "xtfc_angle3d.pth")



def plot_value():
    args = Args()
    env = gym.make(args.env_id)


    hnn = HNN(env)
    hnn.load_state_dict(torch.load(HNN_MODEL_PATH, map_location="cpu"))
    hnn.eval()

    # Construct a three-dimensional state slice
    x1 = np.linspace(-1, 1, 50)
    x2 = np.linspace(-1, 1, 50)
    X1, X2 = np.meshgrid(x1, x2)
    X1_flat = X1.ravel()
    X2_flat = X2.ravel()
    X3_flat = np.zeros_like(X1_flat)
    X_grid = np.stack([X1_flat, X2_flat, X3_flat], axis=1)  # shape (N, 3)

    X_tensor = torch.from_numpy(X_grid).float()
    with torch.no_grad():
        hnn_values = hnn.get_value(X_tensor).cpu().numpy()
    hnn_values = hnn_values.reshape(X1.shape)


    xtfc = XTFC_PINN(env, hidden_dim=400).to("cpu")
    if os.path.exists(XTFC_MODEL_PATH):
        xtfc.load_state_dict(torch.load(XTFC_MODEL_PATH, map_location="cpu"))
    xtfc.eval()

    with torch.no_grad():
        xtfc_values = xtfc.get_value(X_tensor).cpu().numpy().reshape(X1.shape)

    # Plot the comparison surfaces
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(111, projection='3d')


    ax.plot_surface(
        X1, X2, hnn_values, cmap='Purples', alpha=0.85, linewidth=0, antialiased=True, rstride=1, cstride=1
    )
    ax.plot_surface(
        X1, X2, xtfc_values, cmap='Blues', alpha=1.0, linewidth=0, antialiased=True, rstride=1, cstride=1
    )

    legend_elements = [
        Line2D([0], [0], color=plt.cm.Purples(0.7), lw=2, label='HNN (OPINN)'),
        Line2D([0], [0], color=plt.cm.Blues(0.7), lw=2, label='X-TFC PINN')
    ]
    ax.legend(handles=legend_elements)

    ax.set_xlabel(r'$x_1$', labelpad=10)
    ax.set_ylabel(r'$x_2$', labelpad=10)
    ax.set_zlabel(r'$V(x)$', labelpad=10)
    ax.set_title(r'Value Function Surface ($x_3=0$)')
    ax.view_init(elev=28, azim=-50, roll=0)
    plt.tight_layout()
    os.makedirs("image", exist_ok=True)
    filename = "image/" + args.filename_prefix + "value_function.png"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)
    plt.show()


def plot_lambda():
    args = Args()
    env = gym.make(args.env_id)


    hnn = HNN(env)
    hnn.load_state_dict(torch.load(HNN_MODEL_PATH, map_location="cpu"))
    hnn.eval()


    supervised_model = PINNNetwork(input_dim=3, output_dim=3, hidden_dim=32, num_layers=3)
    supervised_model.load_state_dict(torch.load(SUP_MODEL_PATH, map_location="cpu"))
    supervised_model.eval()


    xtfc = XTFC_PINN(env, hidden_dim=400).to("cpu")
    if os.path.exists(XTFC_MODEL_PATH):
        xtfc.load_state_dict(torch.load(XTFC_MODEL_PATH, map_location="cpu"))
    xtfc.eval()


    x1 = np.linspace(-1, 1, 50)
    x2 = np.linspace(-1, 1, 50)
    X1, X2 = np.meshgrid(x1, x2)
    X1_flat = X1.ravel()
    X2_flat = X2.ravel()
    X3_flat = np.zeros_like(X1_flat)
    X_grid = np.stack([X1_flat, X2_flat, X3_flat], axis=1)

    X_tensor = torch.from_numpy(X_grid).float()

    with torch.no_grad():
        hnn_lambda = hnn.get_lambda(X_tensor).cpu().numpy()
        sup_lambda = supervised_model(X_tensor).cpu().numpy()
        xtfc_lambda = xtfc.get_lambda(X_tensor).cpu().numpy()

    hnn_lambda1 = hnn_lambda[:, 0].reshape(X1.shape)
    hnn_lambda2 = hnn_lambda[:, 1].reshape(X1.shape)
    hnn_lambda3 = hnn_lambda[:, 2].reshape(X1.shape)

    sup_lambda1 = sup_lambda[:, 0].reshape(X1.shape)
    sup_lambda2 = sup_lambda[:, 1].reshape(X1.shape)
    sup_lambda3 = sup_lambda[:, 2].reshape(X1.shape)

    xtfc_lambda1 = xtfc_lambda[:, 0].reshape(X1.shape)
    xtfc_lambda2 = xtfc_lambda[:, 1].reshape(X1.shape)
    xtfc_lambda3 = xtfc_lambda[:, 2].reshape(X1.shape)

    # Compute errors
    err_hnn1 = np.abs(hnn_lambda1 - sup_lambda1)
    err_hnn2 = np.abs(hnn_lambda2 - sup_lambda2)
    err_hnn3 = np.abs(hnn_lambda3 - sup_lambda3)
    err_xtfc1 = np.abs(xtfc_lambda1 - sup_lambda1)
    err_xtfc2 = np.abs(xtfc_lambda2 - sup_lambda2)
    err_xtfc3 = np.abs(xtfc_lambda3 - sup_lambda3)
    err_list = [err_hnn1, err_hnn2, err_hnn3, err_xtfc1, err_xtfc2, err_xtfc3]

    vmin = min([err.min() for err in err_list])
    vmax = max([err.max() for err in err_list])
    levels = np.linspace(vmin, vmax, 21)

    lambda_names = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$']
    lambda_all = [
        (hnn_lambda1, sup_lambda1, xtfc_lambda1),
        (hnn_lambda2, sup_lambda2, xtfc_lambda2),
        (hnn_lambda3, sup_lambda3, xtfc_lambda3),
    ]
    model_cmaps = ["Purples", "Oranges", "Blues"]
    model_alphas = [1, 1, 1]

    os.makedirs("image", exist_ok=True)


    figs = []
    for i in range(3):
        figs.append(plt.figure(figsize=(6, 5)))

    for i in range(3):
        fig = figs[i]
        ax = fig.add_subplot(111, projection='3d')
        for j in [0, 1]:
            lam = lambda_all[i][j]
            cmap = model_cmaps[j]
            alpha = model_alphas[j]
            ax.plot_surface(
                X1, X2, lam,
                cmap=cmap,
                edgecolor='none',
                alpha=alpha,
                rstride=1, cstride=1, linewidth=0, antialiased=True
            )
            if i == 1:
                ax.view_init(elev=30, azim=120, roll=0)
        z_min = ax.get_zlim()[0]
        ax.contourf(X1, X2, err_list[i], levels=levels, cmap='RdBu_r', vmin=vmin, vmax=vmax, zdir='z', offset=z_min)
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_zlabel(lambda_names[i])

        filename = "image/" + args.filename_prefix + "lambda" + str(i + 1) + ".svg"
        fig.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)


    figs_xtfc = []
    for i in range(3):
        figs_xtfc.append(plt.figure(figsize=(6, 5)))

    error_cf = None
    for i in range(3):
        fig = figs_xtfc[i]
        ax = fig.add_subplot(111, projection='3d')
        for j in [1, 2]:
            lam = lambda_all[i][j]
            cmap = model_cmaps[j]
            alpha = model_alphas[j]
            ax.plot_surface(
                X1, X2, lam,
                cmap=cmap,
                edgecolor='none',
                alpha=alpha,
                rstride=1, cstride=1, linewidth=0, antialiased=True
            )
            if i == 1:
                ax.view_init(elev=30, azim=120, roll=0)
        z_min = ax.get_zlim()[0]
        error_cf = ax.contourf(X1, X2, err_list[i + 3], levels=levels, cmap='RdBu_r', vmin=vmin, vmax=vmax, zdir='z', offset=z_min)
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_zlabel(lambda_names[i])

        filename = "image/" + args.filename_prefix + "lambda_xtfc" + str(i + 1) + ".svg"
        fig.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)


    if error_cf is not None:
        fig2 = plt.figure(figsize=(6, 10))
        ax = fig2.add_subplot(111)
        fig2.colorbar(error_cf, ax=ax, orientation='vertical', fraction=0.05, pad=0.02, location='right')
        filename = "image/" + args.filename_prefix + "lambda_colorbar.svg"
        fig2.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)

    plt.show()


def monte_carlo_simulation():
    args = Args()
    device = args.device
    env = gym.make(args.env_id)

    # 1. Load HNN
    hnn = HNN(env).to(device)
    if os.path.exists(HNN_MODEL_PATH):
        hnn.load_state_dict(torch.load(HNN_MODEL_PATH, map_location=device))
    hnn.eval()

    # 2. Load Supervised
    supervised_model = PINNNetwork(input_dim=3, output_dim=3, hidden_dim=32, num_layers=3).to(device)
    if os.path.exists(SUP_MODEL_PATH):
        supervised_model.load_state_dict(torch.load(SUP_MODEL_PATH, map_location=device))
    supervised_model.eval()

    # 3. Load XTFC PINN
    xtfc = XTFC_PINN(env, hidden_dim=400).to(device)
    if os.path.exists(XTFC_MODEL_PATH):
        xtfc.load_state_dict(torch.load(XTFC_MODEL_PATH, map_location=device))
    xtfc.eval()

    num_episodes = 50
    horizon = 200
    R_np = env.R
    R_torch = torch.tensor(R_np, dtype=torch.float32, device=device)
    R_inv = torch.linalg.inv(R_torch)

    def hnn_action(x_np):
        x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)
        with torch.no_grad():
            Lambda = hnn.get_lambda(x)
        pfpu = get_pfpu(x)
        pfpu_T = pfpu.transpose(1, 2)
        u_tmp = -0.5 * torch.matmul(pfpu_T, Lambda.unsqueeze(2)).squeeze(-1)
        u = (R_inv @ u_tmp.T).T.squeeze(0)
        return u.cpu().numpy()

    def sup_action(x_np):
        x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)
        with torch.no_grad():
            Lambda = supervised_model(x)
        pfpu = get_pfpu(x).transpose(1, 2)
        u_tmp = -0.5 * torch.matmul(pfpu, Lambda.unsqueeze(2)).squeeze(-1)
        u = (R_inv @ u_tmp.T).T.squeeze(0)
        return u.cpu().numpy()

    def xtfc_action(x_np):
        x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)
        with torch.no_grad():
            u = xtfc.get_action(x)
        return u.squeeze(0).cpu().numpy()

    returns_hnn, returns_sup, returns_xtfc = [], [], []
    traj_u_hnn, traj_u_sup, traj_u_xtfc = [], [], []
    traj_x_hnn, traj_x_sup, traj_x_xtfc = [], [], []

    print("Running Monte Carlo Simulations over 50 Episodes...")
    for ep in range(num_episodes):
        seed = int(np.random.randint(0, 2**31 - 1))
        env_h = gym.make(args.env_id)
        env_s = gym.make(args.env_id)
        env_x = gym.make(args.env_id)

        obs_h, _ = env_h.reset(seed=seed, theoretic_mode=True)
        obs_s, _ = env_s.reset(seed=seed, theoretic_mode=True)
        obs_x, _ = env_x.reset(seed=seed, theoretic_mode=True)

        ret_h, ret_s, ret_x = 0.0, 0.0, 0.0
        u_seq_h, u_seq_s, u_seq_x = [], [], []
        x_seq_h, x_seq_s, x_seq_x = [obs_h.copy()], [obs_s.copy()], [obs_x.copy()]

        for t in range(horizon):
            u_h = hnn_action(obs_h)
            obs_h, r_h, _, _, _ = env_h.step(u_h.astype(np.float32))
            ret_h += float(r_h)
            u_seq_h.append(u_h)
            x_seq_h.append(obs_h.copy())

            u_s = sup_action(obs_s)
            obs_s, r_s, _, _, _ = env_s.step(u_s.astype(np.float32))
            ret_s += float(r_s)
            u_seq_s.append(u_s)
            x_seq_s.append(obs_s.copy())

            u_x = xtfc_action(obs_x)
            obs_x, r_x, _, _, _ = env_x.step(u_x.astype(np.float32))
            ret_x += float(r_x)
            u_seq_x.append(u_x)
            x_seq_x.append(obs_x.copy())

        returns_hnn.append(ret_h)
        returns_sup.append(ret_s)
        returns_xtfc.append(ret_x)

        traj_u_hnn.append(u_seq_h)
        traj_u_sup.append(u_seq_s)
        traj_u_xtfc.append(u_seq_x)

        traj_x_hnn.append(x_seq_h)
        traj_x_sup.append(x_seq_s)
        traj_x_xtfc.append(x_seq_x)

    traj_u_hnn = np.array(traj_u_hnn)
    traj_u_sup = np.array(traj_u_sup)
    traj_u_xtfc = np.array(traj_u_xtfc)

    traj_x_hnn = np.array(traj_x_hnn)
    traj_x_sup = np.array(traj_x_sup)
    traj_x_xtfc = np.array(traj_x_xtfc)

    time_steps = np.arange(horizon)
    colors = {"HNN": "tab:purple", "Supervised": "tab:orange", "X-TFC": "tab:blue"}


    fig_u, axs_u = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    def plot_mean_std(ax, data, label, color):
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        ax.plot(time_steps, mean, label=label, color=color)
        ax.fill_between(time_steps, mean - std, mean + std, color=color, alpha=0.2)

    for i, ax in enumerate(axs_u):
        plot_mean_std(ax, traj_u_hnn[:, :, i], "HNN", colors["HNN"])
        plot_mean_std(ax, traj_u_sup[:, :, i], "Supervised", colors["Supervised"])
        plot_mean_std(ax, traj_u_xtfc[:, :, i], "X-TFC PINN", colors["X-TFC"])
        ax.set_ylabel(f"Control $u_{i+1}$")
        ax.grid(True, linestyle='--', alpha=0.4)
        if i == 0:
            ax.legend()
    axs_u[-1].set_xlabel("Time step")
    fig_u.suptitle("Control trajectories (mean ± std over 50 episodes)")
    plt.tight_layout()
    plt.show()


    time_x = np.arange(horizon + 1)
    fig_x, axs_x = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    def plot_mean_std_x(ax, data, label, color):
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        ax.plot(time_x, mean, label=label, color=color)
        ax.fill_between(time_x, mean - std, mean + std, color=color, alpha=0.2)

    for i, ax in enumerate(axs_x):
        plot_mean_std_x(ax, traj_x_hnn[:, :, i], "HNN", colors["HNN"])
        plot_mean_std_x(ax, traj_x_sup[:, :, i], "Supervised", colors["Supervised"])
        plot_mean_std_x(ax, traj_x_xtfc[:, :, i], "X-TFC PINN", colors["X-TFC"])
        ax.set_ylabel(f"$x_{i+1}$")
        ax.grid(True, linestyle='--', alpha=0.4)
        if i == 0:
            ax.legend()
    axs_x[-1].set_xlabel("Time step")
    fig_x.suptitle("State trajectories (mean ± std over 50 episodes)")
    plt.tight_layout()
    plt.show()

    returns_hnn = np.array(returns_hnn)
    returns_sup = np.array(returns_sup)
    returns_xtfc = np.array(returns_xtfc)

    print("\n=================== Cumulative Returns (50 Episodes) ===================")
    print(f"HNN (OPINN): mean = {returns_hnn.mean():.3f}, std = {returns_hnn.std():.3f}")
    print(f"Supervised : mean = {returns_sup.mean():.3f}, std = {returns_sup.std():.3f}")
    print(f"X-TFC PINN : mean = {returns_xtfc.mean():.3f}, std = {returns_xtfc.std():.3f}")
    print("=========================================================================")


if __name__ == "__main__":
    args = Args()
    plot_value()
    plot_lambda()
    # monte_carlo_simulation()
