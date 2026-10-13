'''
Implement the X-TFC (Extreme Theory of Functional Connections) PINN method of Furfaro et al. (2022).
Solve the HJB equation for the nonlinear three-dimensional attitude-control problem.
Key implementation details:
1. Use an ELM with fixed random hidden-layer parameters and train only the output weights ``beta``.
2. Construct the constrained expression ``V(x) = g(x) - g(0)`` so that ``V(0) = 0`` holds analytically.
3. Warm-start ``beta`` with a least-squares fit to the quadratic LQR value function to prevent
   gradient divergence in the nonlinear HJB residual.
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


# Plot settings
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


# Parameters
class Args:
    # Random seed
    seed = 42
    env_id = "environment:angle3d-v0"
    learning_rate = 1  # Learning rate
    time_steps = 10000
    batch_size = 512
    hidden_dim = 400
    device = "cuda" if torch.cuda.is_available() else "cpu"
    filename_prefix = "xtfc_angle3d_"

args = Args()


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# Dynamics
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

# =====================================================================
class XTFC_PINN(nn.Module):
    def __init__(self, env, hidden_dim=400):
        super().__init__()
        self.env = env.unwrapped if hasattr(env, 'unwrapped') else env
        self.state_dim = env.observation_space.shape[0]
        self.hidden_dim = hidden_dim

        # Hidden layers

        self.W = nn.Parameter(torch.randn(hidden_dim, self.state_dim) * 1.5, requires_grad=False)
        self.b = nn.Parameter(torch.randn(hidden_dim) * 1.5, requires_grad=False)


        self.beta = nn.Parameter(torch.zeros(hidden_dim, 1, requires_grad=True))


        self.act = torch.tanh

    def _get_H(self, x):
        """Compute the ELM hidden-feature matrix H(x) = tanh(X W^T + b)."""
        return self.act(torch.matmul(x, self.W.t()) + self.b)

    def get_value(self, x):
        """
        Use the TFC constrained expression to satisfy V(0) = 0 analytically:
        V_CE(x) = g(x) - g(0) = (H(x) - H(0)) * \beta
        """
        H_x = self._get_H(x)
        H_0 = self._get_H(torch.zeros_like(x))
        V = torch.matmul(H_x - H_0, self.beta)
        return V

    def get_lambda(self, x):
        """
        Compute the costate \Lambda = \nabla V(x).
        Use the ELM structure for efficient analytical differentiation:
        d/dx_i [tanh(Z)] = (1 - tanh^2(Z)) * W[:, i]
        """
        Z = torch.matmul(x, self.W.t()) + self.b
        H = self.act(Z)
        dH = 1.0 - H ** 2  # (batch_size, hidden_dim)

        grads = []
        for i in range(self.state_dim):

            dH_dxi = dH * self.W[:, i].unsqueeze(0)  # (batch_size, hidden_dim)
            dV_dxi = torch.matmul(dH_dxi, self.beta) # (batch_size, 1)
            grads.append(dV_dxi)

        Lambda = torch.cat(grads, dim=1)  # (batch_size, state_dim)
        return Lambda

    def get_action(self, X):
        """Evaluate u^* = -0.5 * R^{-1} * G(X)^T * \Lambda from the optimal control law."""
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
        Use the quadratic LQR function V_guess = X^T Q X to warm-start \beta by least squares, following Sections IV.A--IV.B.
        This provides a high-quality initial guess and prevents divergence of nonlinear HJB derivatives.
        """
        with torch.no_grad():
            Q_np = self.env.Q
            Q_torch = torch.tensor(Q_np, dtype=X_sample.dtype, device=X_sample.device)

            V_guess = torch.einsum('bi,ij,bj->b', X_sample, Q_torch, X_sample).unsqueeze(1)
            

            H_x = self._get_H(X_sample)
            H_0 = self._get_H(torch.zeros_like(X_sample))
            A = H_x - H_0
            

            beta_init = torch.linalg.lstsq(A, V_guess).solution
            self.beta.data.copy_(beta_init)
            print("X-TFC: Warm-start LQR initialization completed via Least-Squares.")


def get_hjb_residual_loss(model, X):
    """
    Compute the HJB PDE residual loss (Eqs. 7 and 28 of the reference).
    H(X, \Lambda) = L(X, u^*) + \Lambda^T f(X, u^*) = 0
    Because the TFC expression satisfies V(0)=0 analytically, no additional boundary-condition loss is required.
    """
    Lambda = model.get_lambda(X)
    u_star = model.get_action(X)
    
    Q_torch = torch.tensor(model.env.Q, dtype=X.dtype, device=X.device)
    R_torch = torch.tensor(model.env.R, dtype=X.dtype, device=X.device)
    

    xQx = torch.einsum('bi,ij,bj->b', X, Q_torch, X)
    uRu = torch.einsum('bi,ij,bj->b', u_star, R_torch, u_star)
    

    fxu = dynamics(X, u_star)
    lambda_fxu = (Lambda * fxu).sum(dim=1)
    

    H_residual = xQx + uRu + lambda_fxu
    

    loss = (H_residual ** 2).mean()
    return loss


def train_xtfc(seed=None):
    args = Args()
    if seed is None:
        seed = getattr(args, 'seed', 42)
    set_seed(seed)

    env = gym.make(args.env_id).unwrapped
    xtfc = XTFC_PINN(env, hidden_dim=args.hidden_dim).to(args.device)


    low, high = -1.1, 1.1
    x_batch = np.random.uniform(low, high, (args.batch_size, env.observation_space.shape[0]))
    X_train = torch.from_numpy(x_batch).float().to(args.device)


    xtfc.warm_start_lqr(X_train)



    optimizer = optim.LBFGS([xtfc.beta], lr=args.learning_rate, max_iter=20, history_size=50, line_search_fn="strong_wolfe")
    
    start_time = time.time()
    
    for step in range(args.time_steps):
        def closure():
            optimizer.zero_grad()

            x_rand = np.random.uniform(low, high, (args.batch_size, env.observation_space.shape[0]))
            X_curr = torch.from_numpy(x_rand).float().to(args.device)
            loss = get_hjb_residual_loss(xtfc, X_curr)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        
        if (step + 1) % 100 == 0 or step == 0:
            print(f"Step {step+1}/{args.time_steps}, HJB Residual MSE Loss: {loss.item():.6e}")

    end_time = time.time()
    print(f"Training completed. Total training time: {end_time - start_time:.2f} seconds.")

    # Model parameters
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "xtfc_angle3d.pth")
    torch.save(xtfc.state_dict(), model_path)
    print(f"X-TFC model successfully saved to {model_path}")


# =====================================================================

# =====================================================================
def plot_value():
    model_dir = "model"
    model_path = os.path.join(model_dir, "xtfc_angle3d.pth")
    args = Args()
    env = gym.make(args.env_id).unwrapped
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
    env = gym.make(args.env_id).unwrapped
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
    env = gym.make(args.env_id).unwrapped

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
        env_x = gym.make(args.env_id).unwrapped
        env_s = gym.make(args.env_id).unwrapped
        env_l = gym.make(args.env_id).unwrapped
        
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
