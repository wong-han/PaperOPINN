"""
18-DoF Three-UAV Coupled Formation Control with Distance Rigidity and Collision Avoidance
- State dimension: 18 (3D position and velocity of three UAVs)
- Control dimension: 9 (3D acceleration/thrust command for three UAVs)
- Features: a high-order nonlinear coupled cost with quartic rigidity potential and
  a nonsmooth collision-avoidance barrier, implemented without a Gym dependency.
"""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import time
import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

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
mpl.rcParams['xtick.labelsize'] = 24  # Default tick-label size
mpl.rcParams['ytick.labelsize'] = 24  # Default tick-label size
mpl.rcParams['axes.labelsize'] = 24  # Default axis-label size
mpl.rcParams['lines.linewidth'] = 2  # Line width
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.linestyle'] = '--'  # Dashed grid lines
mpl.rcParams['grid.alpha'] = 0.5       # Transparency
mpl.rcParams['grid.color'] = 'gray'    # Grid color


# Parameters
class Args:
    model_name = "opinn_18d_swarm_formation.pth"
    train_range = 1.5           # State space
    learning_rate = 1e-3
    time_steps = 50000
    batch_size = 128
    T = 12.0
    forward_steps = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_uavs = 5
    state_dim = 6 * num_uavs
    action_dim = 3 * num_uavs



class UAVSwarmFormationEnv:
    """
    Cooperative 3D formation-control environment for three UAVs.

    The UAVs start from perturbed initial states and form a hovering equilateral
    triangle with side length ``D = 2.0 m``. The 18-dimensional state is
    ``X = [p1, v1, p2, v2, p3, v3]`` relative to the target formation, and the
    9-dimensional control is ``U = [a1, a2, a3]`` containing 3D acceleration
    commands.
    """
    def __init__(self, num_uavs=3, dt=0.05):
        self.num_uavs = num_uavs
        self.dt = dt
        self.state_dim = 6 * num_uavs
        self.action_dim = 3 * num_uavs
        

        self.target_dist = 2.0
        self.d_safe = 0.5
        self.w_form = 2.0
        self.w_coll = 15.0
        

        self.obstacles_numpy = [
            (np.array([ 0.5,  0.5,  0.5], dtype=np.float32), 0.35),
            (np.array([-0.5, -0.5,  0.5], dtype=np.float32), 0.35),
            (np.array([ 0.5, -0.5, -0.5], dtype=np.float32), 0.35),
            (np.array([-0.5,  0.5, -0.5], dtype=np.float32), 0.35),
        ]
        self.d_safe_obs = 0.2
        self.w_obs = 20.0


        A_single = np.zeros((6, 6), dtype=np.float32)
        A_single[0:3, 3:6] = np.eye(3)
        B_single = np.zeros((6, 3), dtype=np.float32)
        B_single[3:6, 0:3] = np.eye(3)
        

        self.A = np.kron(np.eye(num_uavs, dtype=np.float32), A_single)
        self.B = np.kron(np.eye(num_uavs, dtype=np.float32), B_single)
        

        Q_single = np.diag([8.0, 8.0, 8.0, 1.0, 1.0, 1.0]).astype(np.float32)
        R_single = np.diag([1.0, 1.0, 1.0]).astype(np.float32)
        self.Q = np.kron(np.eye(num_uavs, dtype=np.float32), Q_single)
        self.R = np.kron(np.eye(num_uavs, dtype=np.float32), R_single)
        self.R_inv = np.linalg.inv(self.R)
        

        self.u_max = 6.0
        self.u_min = -6.0
        self.state_max = 6.0
        self.state_min = -6.0
        

        self.P = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R).astype(np.float32)



        R = self.target_dist / (2 * np.sin(np.pi / self.num_uavs))
        p_star_list = []
        for i in range(self.num_uavs):
            theta = np.pi/2 + i * 2 * np.pi / self.num_uavs
            p_star_list.append([R * np.cos(theta), R * np.sin(theta), 0.0])
        self.p_star = np.array(p_star_list, dtype=np.float32)

    def to_tensors(self, device):
        self.A_tensor = torch.tensor(self.A, dtype=torch.float32, device=device)
        self.B_tensor = torch.tensor(self.B, dtype=torch.float32, device=device)
        self.Q_tensor = torch.tensor(self.Q, dtype=torch.float32, device=device)
        self.R_tensor = torch.tensor(self.R, dtype=torch.float32, device=device)
        self.R_inv_tensor = torch.tensor(self.R_inv, dtype=torch.float32, device=device)
        self.P_tensor = torch.tensor(self.P, dtype=torch.float32, device=device)
        self.p_star_tensor = torch.tensor(self.p_star, dtype=torch.float32, device=device)
        if hasattr(self, 'obstacles_numpy'):
            self.obstacles_tensor = [(torch.tensor(c, dtype=torch.float32, device=device), r) for c, r in self.obstacles_numpy]



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class OPINN(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.env = env.unwrapped if hasattr(env, 'unwrapped') else env
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        
        # Shared layers
        self.shared_net = nn.Sequential(
            layer_init(nn.Linear(self.state_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
        )
        

        self.value_part = nn.Sequential(
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )


        self.lambda_part = nn.Sequential(
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.state_dim), std=1.0),
        )


        self.hamilton_part = nn.Sequential(
            layer_init(nn.Linear(2 * self.state_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )

    def get_value(self, x):

        zero_point = torch.zeros_like(x)
        x_feature = self.shared_net(x)
        zero_feature = self.shared_net(zero_point)
        return self.value_part(x_feature) - self.value_part(zero_feature)

    def get_lambda(self, x):

        zero_point = torch.zeros_like(x)
        x_feature = self.shared_net(x)
        zero_feature = self.shared_net(zero_point)
        return self.lambda_part(x_feature) - self.lambda_part(zero_feature)

    def get_action(self, x):
        """ Analytical PMP control law: u* = -0.5 * R^-1 * B^T * \lambda """
        Lambda = self.get_lambda(x)
        u_star = -0.5 * torch.matmul(Lambda, self.env.B_tensor)
        u_star = torch.matmul(u_star, self.env.R_inv_tensor)
        return torch.clamp(u_star, self.env.u_min, self.env.u_max)



def dynamics_tensor(env, X, U):
    return torch.matmul(X, env.A_tensor.T) + torch.matmul(U, env.B_tensor.T)

def running_cost_tensor(env, X, U):
    """ Compute the coupled cost: quadratic term, quartic formation potential, and collision-avoidance barrier. """
    xQx = torch.einsum('bi,ij,bj->b', X, env.Q_tensor, X)
    uRu = torch.einsum('bi,ij,bj->b', U, env.R_tensor, U)
    

    pos = torch.zeros((X.shape[0], env.num_uavs, 3), device=X.device)
    for i in range(env.num_uavs):
        pos[:, i, :] = X[:, i*6 : i*6+3] + env.p_star_tensor[i]
        
    loss_form = 0.0
    loss_coll = 0.0
    for i in range(env.num_uavs):
        for j in range(i+1, env.num_uavs):

            dij_sq = torch.sum((pos[:, i, :] - pos[:, j, :])**2, dim=1)
            D_ij_sq = torch.sum((env.p_star_tensor[i] - env.p_star_tensor[j])**2)
            loss_form += (dij_sq - D_ij_sq)**2
            

            dij = torch.sqrt(dij_sq + 1e-8)
            loss_coll += torch.relu(env.d_safe - dij)**2

    loss_form = loss_form * env.w_form
    loss_coll = loss_coll * env.w_coll
    
    loss_obs = 0.0
    if hasattr(env, 'obstacles_tensor'):
        for i in range(env.num_uavs):
            for obs_center, obs_radius in env.obstacles_tensor:
                dist_obs_sq = torch.sum((pos[:, i, :] - obs_center)**2, dim=1)


                loss_obs += torch.exp(-10.0 * dist_obs_sq)

        loss_obs = loss_obs * getattr(env, 'w_obs', 1000.0)

    return xQx + uRu + loss_form + loss_coll + loss_obs



def get_all_loss(model, env, X):
    V = model.get_value(X)
    Lambda = model.get_lambda(X)
    Lambda_detach = Lambda.detach().clone().requires_grad_(True)


    pVpX = torch.autograd.grad(
        V, X, grad_outputs=torch.ones_like(V), create_graph=True, retain_graph=True
    )[0]


    hamilton_input = torch.cat([X, Lambda_detach], dim=-1)
    H = model.hamilton_part(hamilton_input)
    pHpLambda = torch.autograd.grad(
        H, Lambda_detach, grad_outputs=torch.ones_like(H), create_graph=True, retain_graph=True
    )[0]
    pHpX = torch.autograd.grad(
        H, X, grad_outputs=torch.ones_like(H), create_graph=True, retain_graph=True
    )[0]


    u_star = model.get_action(X)
    fxu = dynamics_tensor(env, X, u_star)


    loss1 = nn.functional.mse_loss(Lambda, pVpX)               # \lambda = dV/dx
    loss2 = torch.relu(-V).mean()                              # V(X) >= 0
    loss3 = nn.functional.mse_loss(fxu.detach(), pHpLambda)    # dx/dt = dH/d\lambda
    

    cost_val = running_cost_tensor(env, X, u_star.detach())
    dl_dX = torch.autograd.grad(
        cost_val, X, grad_outputs=torch.ones_like(cost_val), create_graph=True, retain_graph=True
    )[0]
    lambda_dot_true = -(dl_dX + torch.matmul(Lambda_detach, env.A_tensor))
    loss4 = nn.functional.mse_loss(-pHpX, lambda_dot_true.detach())
    

    H_true = cost_val + (Lambda * fxu).sum(dim=1)
    loss5 = (H_true ** 2).mean()

    return loss1 + 10.0 * loss2 + loss3 + loss4 + 0.1 * loss5

def lambda_jacobian_loss(model, env, device):
    """ Costate-Jacobian regularization at the origin: (\partial \lambda / \partial x)|_{0} = 2P_LQR """
    zero_point = torch.zeros(1, env.state_dim, dtype=torch.float32, device=device, requires_grad=True)
    Lambda = model.get_lambda(zero_point)
    
    grads = [torch.autograd.grad(Lambda[0, i], zero_point, retain_graph=True, create_graph=True)[0] 
             for i in range(env.state_dim)]
    grad_Lambda = torch.stack(grads, dim=1).squeeze(0)
    return nn.functional.mse_loss(grad_Lambda, 2 * env.P_tensor)



def train_opinn(args, env):
    print(f"--- [Training Start] {env.state_dim}-DoF {env.num_uavs}-UAV Formation Control on {args.device} ---")
    model_path = f"model/opinn_{env.state_dim}d_model.pth"
    env.to_tensors(args.device)
    model = OPINN(env).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    
    dt = args.T / (args.forward_steps * 10)
    start_time = time.time()
    best_loss = float('inf')
    ema_loss = None

    for step in range(args.time_steps):

        x_np = np.random.uniform(-args.train_range, args.train_range, (args.batch_size, env.state_dim))
        X = torch.from_numpy(x_np).float().to(args.device)
        X.requires_grad_(True)
        

        loss = get_all_loss(model, env, X) + lambda_jacobian_loss(model, env, args.device)
        

        x_roll = X.detach().clone()
        for _ in range(args.forward_steps):
            u_roll = model.get_action(x_roll)
            fxu = dynamics_tensor(env, x_roll, u_roll)
            x_roll = x_roll + fxu * dt
            x_roll = torch.clamp(x_roll, env.state_min, env.state_max)
            loss += 10 * running_cost_tensor(env, x_roll, u_roll).mean() * dt
            

        xPx = torch.einsum('bi,ij,bj->b', x_roll, env.P_tensor, x_roll)
        loss += 10 * xPx.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        curr_loss = loss.item()
        ema_loss = curr_loss if ema_loss is None else 0.95 * ema_loss + 0.05 * curr_loss

        if (step + 1) % 500 == 0 or step == 0:
            print(f"Step [{step+1:5d}/{args.time_steps}], Loss: {curr_loss:.4f}, EMA Loss: {ema_loss:.4f}")
            if ema_loss < best_loss:
                best_loss = ema_loss
                os.makedirs("model", exist_ok=True)
                torch.save(model.state_dict(), model_path)
    
    print(f"Saved trained model to {model_path}")
    print(f"--- [Training Complete] Elapsed Time: {time.time() - start_time:.2f} s ---")
    return model



def simulate_and_plot(args, env):
    print("--- [Running Simulation and Plotting] ---")
    device = torch.device("cpu")
    env.to_tensors(device)

    model_path = f"model/opinn_{env.state_dim}d_model.pth"
    model = OPINN(env).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded trained OPINN model successfully.")
    except Exception as e:
        print(f"Failed to load model {model_path}: {e}")
        print("Please ensure you have trained the model first!")
        return
    model.eval()


    K = np.linalg.inv(env.R) @ env.B.T @ env.P




    np.random.seed(345)
    x0 = np.zeros((env.state_dim,), dtype=np.float32)
    for i in range(args.num_uavs):

        p_init_i = -env.p_star[i] + np.random.uniform(-0.5, 0.5, size=(3,))

        p_init_i[2] = np.random.uniform(-2.5, 2.5) 
        x0[i*6 : i*6 + 3] = p_init_i - env.p_star[i]
        x0[i*6 + 3 : i*6 + 6] = 0.0

    steps = int(args.T / env.dt)
    t_axis = np.linspace(0, args.T, steps)
    
    traj_opinn = np.zeros((steps, env.state_dim))
    traj_lqr = np.zeros((steps, env.state_dim))
    x_curr_opinn = x0.copy()
    x_curr_lqr = x0.copy()
    
    cost_opinn = 0.0
    cost_lqr = 0.0
    
    for t in range(steps):
        traj_opinn[t] = x_curr_opinn
        traj_lqr[t] = x_curr_lqr
        

        x_tensor = torch.from_numpy(x_curr_opinn).float().unsqueeze(0)
        with torch.no_grad():
            u_opinn = model.get_action(x_tensor).squeeze(0).numpy()
            

        u_lqr = np.clip(-K @ x_curr_lqr, env.u_min, env.u_max)
        

        u_opinn_tensor = torch.tensor(u_opinn, dtype=torch.float32, device=device).unsqueeze(0)
        x_opinn_tensor = torch.tensor(x_curr_opinn, dtype=torch.float32, device=device).unsqueeze(0)
        cost_opinn += running_cost_tensor(env, x_opinn_tensor, u_opinn_tensor).item() * env.dt
        
        u_lqr_tensor = torch.tensor(u_lqr, dtype=torch.float32, device=device).unsqueeze(0)
        x_lqr_tensor = torch.tensor(x_curr_lqr, dtype=torch.float32, device=device).unsqueeze(0)
        cost_lqr += running_cost_tensor(env, x_lqr_tensor, u_lqr_tensor).item() * env.dt


        x_curr_opinn += (env.A @ x_curr_opinn + env.B @ u_opinn) * env.dt
        x_curr_lqr += (env.A @ x_curr_lqr + env.B @ u_lqr) * env.dt


    pos_opinn = np.zeros((steps, env.num_uavs, 3))
    pos_lqr = np.zeros((steps, env.num_uavs, 3))
    for i in range(env.num_uavs):
        pos_opinn[:, i, :] = traj_opinn[:, i*6 : i*6+3] + env.p_star[i]
        pos_lqr[:, i, :] = traj_lqr[:, i*6 : i*6+3] + env.p_star[i]

    os.makedirs("image", exist_ok=True)


    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.get_cmap('tab10', env.num_uavs)(np.linspace(0, 1, env.num_uavs))
    
    for i in range(args.num_uavs):

        ax.plot(pos_opinn[:, i, 0], pos_opinn[:, i, 1], pos_opinn[:, i, 2], 
                color=colors[i], linestyle='-', linewidth=2.5)

        ax.plot(pos_lqr[:, i, 0], pos_lqr[:, i, 1], pos_lqr[:, i, 2], 
                color=colors[i], linestyle='--', linewidth=1.5, alpha=0.6)

        ax.scatter(pos_opinn[0, i, 0], pos_opinn[0, i, 1], pos_opinn[0, i, 2], color=colors[i], marker='o', s=60)
        ax.scatter(env.p_star[i, 0], env.p_star[i, 1], env.p_star[i, 2], color=colors[i], marker='*', s=200, edgecolor='k')


    if hasattr(env, 'obstacles_numpy'):
        for center, radius in env.obstacles_numpy:
            u, v = np.mgrid[0:2*np.pi:15j, 0:np.pi:10j]
            x = center[0] + radius * np.cos(u) * np.sin(v)
            y = center[1] + radius * np.sin(u) * np.sin(v)
            z = center[2] + radius * np.cos(v)
            ax.plot_surface(x, y, z, color='gray', alpha=0.4, edgecolor='none')

    ax.set_xlabel('$X$ (m)')
    ax.set_ylabel('$Y$ (m)')
    ax.set_zlabel('$Z$ (m)')
    ax.view_init(elev=25, azim=-142, roll=-2)
    plt.tight_layout()
    plt.savefig("image/opinn_18d_swarm_3d.pdf", format='pdf', dpi=300)
    print("Saved 3D formation trajectory plot to 'image/opinn_18d_swarm_3d.pdf'.")


    min_dist_arr_opinn = np.min([
        np.linalg.norm(pos_opinn[:, i, :] - pos_opinn[:, j, :], axis=1)
        for i in range(env.num_uavs) for j in range(i+1, env.num_uavs)
    ], axis=0)
    min_dist_arr_lqr = np.min([
        np.linalg.norm(pos_lqr[:, i, :] - pos_lqr[:, j, :], axis=1)
        for i in range(env.num_uavs) for j in range(i+1, env.num_uavs)
    ], axis=0)

    min_dist_opinn = np.min(min_dist_arr_opinn)
    min_dist_lqr = np.min(min_dist_arr_lqr)
    

    min_obs_arr_opinn = np.full(steps, np.inf)
    min_obs_arr_lqr = np.full(steps, np.inf)
    if hasattr(env, 'obstacles_numpy'):
        for i in range(env.num_uavs):
            for center, radius in env.obstacles_numpy:
                dist_o = np.linalg.norm(pos_opinn[:, i, :] - center, axis=1) - radius
                dist_l = np.linalg.norm(pos_lqr[:, i, :] - center, axis=1) - radius
                min_obs_arr_opinn = np.minimum(min_obs_arr_opinn, dist_o)
                min_obs_arr_lqr = np.minimum(min_obs_arr_lqr, dist_l)
                
    min_obs_opinn = np.min(min_obs_arr_opinn)
    min_obs_lqr = np.min(min_obs_arr_lqr)
    
    print("\\n" + "="*50)
    print("[OPINN vs. LQR performance comparison]")
    print(f"OPINN cumulative cost (including collision avoidance and formation stiffness): {cost_opinn:.2f}")
    print(f"LQR cumulative cost (including collision avoidance and formation stiffness): {cost_lqr:.2f}")
    print(f"OPINN minimum inter-agent distance: {min_dist_opinn:.3f} m (safety distance: {env.d_safe} m)")
    print(f"LQR minimum inter-agent distance: {min_dist_lqr:.3f} m (safety distance: {env.d_safe} m)")
    if hasattr(env, 'obstacles_numpy'):
        print(f"OPINN minimum obstacle clearance: {min_obs_opinn:.3f} m (positive values are safe)")
        print(f"LQR minimum obstacle clearance: {min_obs_lqr:.3f} m (positive values are safe)")
    if min_dist_lqr < env.d_safe or (hasattr(env, 'obstacles_numpy') and min_obs_lqr < 0):
        print(">> LQR collision detected.")
    if min_dist_opinn >= env.d_safe and (not hasattr(env, 'obstacles_numpy') or min_obs_opinn >= 0):
        print(">> OPINN achieved collision-free formation control.")
    print("="*50 + "\\n")


    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5.5))
    


    pairs = [(i, (i+1)%env.num_uavs, f'UAV {i+1}-{(i+1)%env.num_uavs+1}') for i in range(env.num_uavs)]
    cmap = plt.cm.get_cmap('Set1', env.num_uavs)
    
    for idx, (i, j, label) in enumerate(pairs):
        dist_opinn = np.linalg.norm(pos_opinn[:, i, :] - pos_opinn[:, j, :], axis=1)
        dist_lqr = np.linalg.norm(pos_lqr[:, i, :] - pos_lqr[:, j, :], axis=1)
        
        ax1.plot(t_axis, dist_opinn, color=cmap(idx), linewidth=2.5, label=f'OPINN {label}')
        ax1.plot(t_axis, dist_lqr, color=cmap(idx), linewidth=1.5, linestyle='--', alpha=0.6)

    ax1.axhline(y=env.target_dist, color='black', linestyle='-.', linewidth=2.0, label='Target Distance $D=2.0$m')
    ax1.axhline(y=env.d_safe, color='red', linestyle=':', linewidth=2.5, label='Safety Barrier $d_{safe}=0.5$m')
    ax1.set_xlabel('Time $t$ (s)')
    ax1.set_ylabel('Adjacent Inter-Agent Distance (m)')
    ax1.set_title('Convergence of Formation Rigidity (Edges)')
    ax1.legend(ncol=2, loc='lower right', fontsize=8)


    ax2.plot(t_axis, min_dist_arr_opinn, color='#1f77b4', linewidth=3.0, label='OPINN Min Distance')
    ax2.plot(t_axis, min_dist_arr_lqr, color='#ff7f0e', linewidth=2.5, linestyle='--', label='LQR Min Distance')
    ax2.axhline(y=env.d_safe, color='red', linestyle='-', linewidth=2.0, label='Collision Region')
    ax2.fill_between(t_axis, 0, env.d_safe, color='red', alpha=0.15)
    ax2.set_xlabel('Time $t$ (s)')
    ax2.set_ylabel('Global Minimum Distance (m)')
    ax2.set_title('UAV-UAV Collision Avoidance')
    ax2.legend()
    

    if hasattr(env, 'obstacles_numpy'):
        ax3.plot(t_axis, min_obs_arr_opinn, color='#1f77b4', linewidth=3.0, label='OPINN Min Obs Dist')
        ax3.plot(t_axis, min_obs_arr_lqr, color='#ff7f0e', linewidth=2.5, linestyle='--', label='LQR Min Obs Dist')
        ax3.axhline(y=0.0, color='red', linestyle='-', linewidth=2.0, label='Obstacle Surface')
        ax3.fill_between(t_axis, -1.0, 0.0, color='red', alpha=0.15)
        ax3.set_xlabel('Time $t$ (s)')
        ax3.set_ylabel('Distance to Obstacle Surface (m)')
        ax3.set_title('UAV-Obstacle Collision Avoidance')
        ax3.set_ylim([-0.5, 2.0])
        ax3.legend()
    
    plt.tight_layout()
    plt.savefig(f"image/opinn_{env.state_dim}d_swarm_rigidity.png", dpi=300)
    print(f"Saved formation distance rigidity plot to 'image/opinn_{env.state_dim}d_swarm_rigidity.png'.")
    plt.show()



def monte_carlo_simulation(args, env, num_episodes=50):
    print(f"\n--- [Running Monte Carlo Simulation: {num_episodes} Episodes] ---")
    device = torch.device("cpu")
    env.to_tensors(device)
    model_path = f"model/opinn_{env.state_dim}d_model.pth"
    model = OPINN(env).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print("Model loading failed.")
        return
    model.eval()

    K = np.linalg.inv(env.R) @ env.B.T @ env.P

    min_dists_lqr = []
    min_dists_opinn = []
    min_obs_lqr = []
    min_obs_opinn = []

    steps = int(args.T / env.dt)
    
    np.random.seed(42)

    for ep in range(num_episodes):
        x0 = np.zeros((env.state_dim,), dtype=np.float32)
        for i in range(args.num_uavs):
            p_init_i = -env.p_star[i] + np.random.uniform(-0.5, 0.5, size=(3,))
            p_init_i[2] = np.random.uniform(-2.5, 2.5) 
            x0[i*6 : i*6 + 3] = p_init_i - env.p_star[i]
            x0[i*6 + 3 : i*6 + 6] = 0.0
            
        x_curr_opinn = x0.copy()
        x_curr_lqr = x0.copy()

        pos_opinn = np.zeros((steps, env.num_uavs, 3))
        pos_lqr = np.zeros((steps, env.num_uavs, 3))

        for t in range(steps):
            for i in range(env.num_uavs):
                pos_opinn[t, i] = x_curr_opinn[i*6 : i*6 + 3] + env.p_star[i]
                pos_lqr[t, i] = x_curr_lqr[i*6 : i*6 + 3] + env.p_star[i]
                
            x_tensor = torch.from_numpy(x_curr_opinn).float().unsqueeze(0)
            with torch.no_grad():
                u_opinn = model.get_action(x_tensor).squeeze(0).numpy()
            u_lqr = np.clip(-K @ x_curr_lqr, env.u_min, env.u_max)

            x_curr_opinn += (env.A @ x_curr_opinn + env.B @ u_opinn) * env.dt
            x_curr_lqr += (env.A @ x_curr_lqr + env.B @ u_lqr) * env.dt

        ep_min_dist_opinn = np.min([
            np.min(np.linalg.norm(pos_opinn[:, i, :] - pos_opinn[:, j, :], axis=1))
            for i in range(env.num_uavs) for j in range(i+1, env.num_uavs)
        ])
        ep_min_dist_lqr = np.min([
            np.min(np.linalg.norm(pos_lqr[:, i, :] - pos_lqr[:, j, :], axis=1))
            for i in range(env.num_uavs) for j in range(i+1, env.num_uavs)
        ])

        ep_min_obs_opinn = np.inf
        ep_min_obs_lqr = np.inf
        if hasattr(env, 'obstacles_numpy'):
            obs_dists_opinn = []
            obs_dists_lqr = []
            for center, radius in env.obstacles_numpy:
                for i in range(env.num_uavs):
                    dist_opinn = np.linalg.norm(pos_opinn[:, i, :] - center, axis=1) #- radius
                    dist_lqr = np.linalg.norm(pos_lqr[:, i, :] - center, axis=1) #- radius
                    obs_dists_opinn.append(np.min(dist_opinn))
                    obs_dists_lqr.append(np.min(dist_lqr))
            ep_min_obs_opinn = np.min(obs_dists_opinn)
            ep_min_obs_lqr = np.min(obs_dists_lqr)
            
        min_dists_opinn.append(ep_min_dist_opinn)
        min_dists_lqr.append(ep_min_dist_lqr)
        min_obs_opinn.append(ep_min_obs_opinn)
        min_obs_lqr.append(ep_min_obs_lqr)

    import seaborn as sns
    import pandas as pd

    df_dist = pd.DataFrame({
        "Distance": np.concatenate((min_dists_lqr, min_dists_opinn)),
        "Controller": ["LQR"]*num_episodes + ["OPINN"]*num_episodes
    })

    df_obs = pd.DataFrame({
        "Distance": np.concatenate((min_obs_lqr, min_obs_opinn)),
        "Controller": ["LQR"]*num_episodes + ["OPINN"]*num_episodes
    })

    nature_green = "#389826"   # LQR
    nature_purple = "#9558B2"  # OPINN
    palette = [nature_green, nature_purple]


    fig1, ax1 = plt.subplots(figsize=(5, 7))
    sns.violinplot(x="Controller", y="Distance", data=df_dist, order=["LQR", "OPINN"],
                   palette=palette, inner=None, alpha=0.22, cut=0, linewidth=1.2, ax=ax1)
    sns.boxplot(x="Controller", y="Distance", data=df_dist, order=["LQR", "OPINN"],
                palette=palette, width=0.25, showcaps=True, showbox=True, showfliers=False,
                medianprops=dict(color="#6D2D2B", linewidth=2),
                boxprops=dict(alpha=0.7, edgecolor='k', linewidth=2), ax=ax1)
    sns.stripplot(x="Controller", y="Distance", data=df_dist, order=["LQR", "OPINN"],
                  palette=palette, size=7, alpha=0.4, jitter=0.22, linewidth=0.2, edgecolor='#444', ax=ax1)
    ax1.set_ylabel("Minimum Inter-Agent Distance (m)")
    # ax1.set_title("Inter-Agent Collision Avoidance")
    ax1.grid(True, linestyle='--', alpha=0.45)
    plt.tight_layout()
    import os
    os.makedirs('image', exist_ok=True)
    plt.savefig('image/opinn_30d_mc_inter_agent.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig1)


    fig2, ax2 = plt.subplots(figsize=(5, 7))
    sns.violinplot(x="Controller", y="Distance", data=df_obs, order=["LQR", "OPINN"],
                   palette=palette, inner=None, alpha=0.22, cut=0, linewidth=1.2, ax=ax2)
    sns.boxplot(x="Controller", y="Distance", data=df_obs, order=["LQR", "OPINN"],
                palette=palette, width=0.25, showcaps=True, showbox=True, showfliers=False,
                medianprops=dict(color="#6D2D2B", linewidth=2),
                boxprops=dict(alpha=0.7, edgecolor='k', linewidth=2), ax=ax2)
    sns.stripplot(x="Controller", y="Distance", data=df_obs, order=["LQR", "OPINN"],
                  palette=palette, size=7, alpha=0.4, jitter=0.22, linewidth=0.2, edgecolor='#444', ax=ax2)
    ax2.set_ylabel("Minimum Obstacle Distance (m)")
    # ax2.set_title("Obstacle Avoidance")
    ax2.grid(True, linestyle='--', alpha=0.45)

    plt.tight_layout()
    plt.savefig('image/opinn_30d_mc_obstacle.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print("Saved separate Monte Carlo constraints plots to 'image/opinn_30d_mc_inter_agent.pdf' and 'image/opinn_30d_mc_obstacle.pdf'")
    
    print("\n--- [Monte Carlo Simulation Results] ---")
    print(f"OPINN Inter-Agent Dist: mean={np.mean(min_dists_opinn):.3f}, min={np.min(min_dists_opinn):.3f}")
    print(f"LQR   Inter-Agent Dist: mean={np.mean(min_dists_lqr):.3f}, min={np.min(min_dists_lqr):.3f}")
    print(f"OPINN Obstacle Dist   : mean={np.mean(min_obs_opinn):.3f}, min={np.min(min_obs_opinn):.3f}")
    print(f"LQR   Obstacle Dist   : mean={np.mean(min_obs_lqr):.3f}, min={np.min(min_obs_lqr):.3f}")

if __name__ == "__main__":
    args = Args()
    env = UAVSwarmFormationEnv(num_uavs=args.num_uavs)
    

    # train_opinn(args, env)
    

    simulate_and_plot(args, env)
    

    monte_carlo_simulation(args, env, num_episodes=50)
