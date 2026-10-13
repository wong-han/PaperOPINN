'''
Combine all loss terms and use a positive-definite value-network design for the quadrotor.
'''
import sys, os
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
import time
from scipy.linalg import solve_continuous_are
from utils import DroneVisualizer
from copy import deepcopy

# Plot settings
mpl.rcParams['font.family'] = 'Times New Roman'  # Default font
# Bold font
mpl.rcParams['text.usetex'] = False  # Use TeX rendering
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
    # Environment identifier
    env_id = "environment:quadrotor3d_TR-v0"
    # Model name
    model_name = "hnn_quadrotor3d_TR_v2_paperplot.pth"
    # Training domain
    train_range = 1.0
    # Learning rate
    learning_rate = 1e-3
    # Total training iterations
    time_steps = 30000
    # Training batch size
    batch_size = 2048

    T = 1

    forward_steps = 5
    # Select the compute device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Output filename prefix
    filename_prefix = "hnn_quadrotor3d_"


def dynamics(X, U):
    # X: (N, 9) -> [x, y, z, vx, vy, vz, phi, theta, psi]
    # U: (N, 4) -> [T, p, q, r]

    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
    vx = X[:, 3]
    vy = X[:, 4]
    vz = X[:, 5]
    phi = X[:, 6]
    theta = X[:, 7]
    psi = X[:, 8]

    T = U[:, 0]
    p = U[:, 1]
    q = U[:, 2]
    r = U[:, 3]

    N = X.shape[0]
    fxu = torch.zeros(N, X.shape[1], dtype=X.dtype, device=X.device)

    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    tan_theta = torch.tan(theta)
    sin_psi = torch.sin(psi)
    cos_psi = torch.cos(psi)

    fxu[:, 0] += vx
    fxu[:, 1] += vy
    fxu[:, 2] += vz
    fxu[:, 3] += T * (cos_phi * sin_theta * cos_psi + sin_phi * sin_psi) / m
    fxu[:, 4] += T * (cos_phi * sin_theta * sin_psi - sin_phi * cos_psi) / m
    fxu[:, 5] += T * (cos_phi * cos_theta) / m - g
    fxu[:, 6] += p + sin_phi * tan_theta * q + cos_phi * tan_theta * r
    fxu[:, 7] += cos_phi * q - sin_phi * r
    fxu[:, 8] += sin_phi / cos_theta * q + cos_phi / cos_theta * r

    return fxu

_env_tmp = gym.make(Args.env_id).unwrapped
m = _env_tmp.metadata['m']
g = _env_tmp.metadata['g']

# Compute the control Jacobian
def get_pfpu(X):
    """
    Compute the control Jacobian of the three-dimensional quadrotor dynamics for each sample.
    X: (N, 12) -> [x, y, z, vx, vy, vz, phi, theta, psi]
    Returns: (N, 12, 4)
    """

    # m, I, g
    phi = X[:, 6]
    theta = X[:, 7]
    psi = X[:, 8]

    N = X.shape[0]
    pfpu = torch.zeros(N, 9, 4, dtype=X.dtype, device=X.device)

    pfpu[:, 3, 0] += (torch.sin(phi)*torch.sin(psi) + torch.sin(theta)*torch.cos(phi)*torch.cos(psi))/m
    pfpu[:, 4, 0] += (-torch.sin(phi)*torch.cos(psi) + torch.sin(psi)*torch.sin(theta)*torch.cos(phi))/m
    pfpu[:, 5, 0] += torch.cos(phi)*torch.cos(theta)/m
    pfpu[:, 6, 1] += 1
    pfpu[:, 6, 2] += torch.sin(phi) * torch.tan(theta)
    pfpu[:, 6, 3] += torch.cos(phi) * torch.tan(theta)
    pfpu[:, 7, 2] += torch.cos(phi)
    pfpu[:, 7, 3] += -torch.sin(phi)
    pfpu[:, 8, 2] += torch.sin(phi) / torch.cos(theta)
    pfpu[:, 8, 3] += torch.cos(phi) / torch.cos(theta)

    return pfpu

# Compute the state Jacobian
def get_pfpx(X, U):
    """
    Compute the state Jacobian of the three-dimensional quadrotor dynamics for each sample.
    X: (N, 12)
    U: (N, 4)
    Returns: (N, 12, 12)
    """

    phi = X[:, 6]
    theta = X[:, 7]
    psi = X[:, 8]
    
    T = U[:, 0]
    p = U[:, 1]
    q = U[:, 2]
    r = U[:, 3]

    N = X.shape[0]
    pfpx = torch.zeros(N, 9, 9, dtype=X.dtype, device=X.device)

    pfpx[:, 0, 3] += 1
    pfpx[:, 1, 4] += 1
    pfpx[:, 2, 5] += 1
    pfpx[:, 3, 6] += T*(-torch.sin(phi)*torch.sin(theta)*torch.cos(psi) + torch.sin(psi)*torch.cos(phi))/m
    pfpx[:, 3, 7] += T*torch.cos(phi)*torch.cos(psi)*torch.cos(theta)/m
    pfpx[:, 3, 8] += T*(torch.sin(phi)*torch.cos(psi) - torch.sin(psi)*torch.sin(theta)*torch.cos(phi))/m
    pfpx[:, 4, 6] += T*(-torch.sin(phi)*torch.sin(psi)*torch.sin(theta) - torch.cos(phi)*torch.cos(psi))/m
    pfpx[:, 4, 7] += T*torch.sin(psi)*torch.cos(phi)*torch.cos(theta)/m
    pfpx[:, 4, 8] += T*(torch.sin(phi)*torch.sin(psi) + torch.sin(theta)*torch.cos(phi)*torch.cos(psi))/m
    pfpx[:, 5, 6] += -T*torch.sin(phi)*torch.cos(theta)/m
    pfpx[:, 5, 7] += -T*torch.sin(theta)*torch.cos(phi)/m
    pfpx[:, 6, 6] += q*torch.cos(phi)*torch.tan(theta) - r*torch.sin(phi)*torch.tan(theta)
    pfpx[:, 6, 7] += q*(torch.tan(theta)**2 + 1)*torch.sin(phi) + r*(torch.tan(theta)**2 + 1)*torch.cos(phi)
    pfpx[:, 7, 6] += -q*torch.sin(phi) - r*torch.cos(phi)
    pfpx[:, 8, 6] += q*torch.cos(phi)/torch.cos(theta) - r*torch.sin(phi)/torch.cos(theta)
    pfpx[:, 8, 7] += q*torch.sin(phi)*torch.sin(theta)/torch.cos(theta)**2 + r*torch.sin(theta)*torch.cos(phi)/torch.cos(theta)**2
    
    return pfpx


class HNN(nn.Module):
    def __init__(self, env):
        self.env = env.unwrapped if hasattr(env, 'unwrapped') else env
        super().__init__()

        # Shared layers
        self.shared_net = nn.Sequential(
            layer_init(nn.Linear(env.observation_space.shape[0], 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.state_dim = env.observation_space.shape[0]
        # self.value_element = nn.Sequential(
        #     layer_init(nn.Linear(64, 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, self.state_dim * (self.state_dim + 1) // 2), std=1.0),
        # )
        # self.indices = [(i, j) for i in range(self.state_dim) for j in range(i + 1)]
        self.value_part = nn.Sequential(
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.lambda_part = nn.Sequential(
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, env.observation_space.shape[0]), std=1.0),
        )

        self.hamilton_part = nn.Sequential(
            layer_init(nn.Linear(2*env.observation_space.shape[0], 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
    
    def get_value(self, x):
        zero_point = torch.zeros_like(x)
        x_feature = self.shared_net(x)
        zero_point_feature = self.shared_net(zero_point)
        return self.value_part(x_feature) - self.value_part(zero_point_feature)
        # x = x - torch.tensor([0] * x.shape[1], dtype=x.dtype, device=x.device)
        # batch_size = x.shape[0]
        # x_feature = self.shared_net(x)
        # elements = self.value_element(x_feature)
        # L = torch.zeros(batch_size, self.state_dim, self.state_dim, device=x.device)
        # for idx, (i, j) in enumerate(self.indices):
        #     L[:, i, j] += elements[:, idx]

        # L_T = L.transpose(1, 2)
        # intermediate = torch.bmm(x.unsqueeze(1), L_T).squeeze(1)
        # value = (intermediate ** 2).sum(dim=1)
        # value = value.unsqueeze(dim=1)
        # return value

    
    def get_lambda(self, x):
        zero_point = torch.zeros_like(x)
        x_feature = self.shared_net(x)
        zero_point_feature = self.shared_net(zero_point)
        return self.lambda_part(x_feature) - self.lambda_part(zero_point_feature)

    def get_hamilton(self, x):
        Lambda = self.get_lambda(x)
        hamilton_input = torch.cat([x, Lambda], dim=-1)
        return self.hamilton_part(hamilton_input)
    
    def get_pHpLambda(self, x):
        x = x.requires_grad_(True)
        Lambda = self.get_lambda(x)
        Lambda = Lambda.detach().requires_grad_(True)
        hamilton_input = torch.cat([x, Lambda], dim=-1)
        H = self.hamilton_part(hamilton_input).squeeze(-1)  # (batch,)
        grads = []
        for i in range(H.shape[0]):
            grad = torch.autograd.grad(
                H[i], Lambda, retain_graph=True, create_graph=True, allow_unused=True
            )[0]
            grads.append(grad[i])
        grads = torch.stack(grads, dim=0)  # (batch_size, state_dim)
        return grads
    
    def get_pHpX(self, x):
        x = x.requires_grad_(True)
        Lambda = self.get_lambda(x).detach()  # Detach the costate output from the state graph
        hamilton_input = torch.cat([x, Lambda], dim=-1)
        H = self.hamilton_part(hamilton_input).squeeze(-1)  # (batch,)
        grads = []
        for i in range(H.shape[0]):
            grad = torch.autograd.grad(
                H[i], x, retain_graph=True, create_graph=True, allow_unused=True
            )[0]
            grads.append(grad[i])
        grads = torch.stack(grads, dim=0)  # (batch_size, state_dim)
        return grads
    
    def get_action(self, X):
        R = self.env.R
        Lambda = self.get_lambda(X) 
        Lambda = Lambda.unsqueeze(2)  
        
        pfpu = get_pfpu(X)  
        

        R_inv = torch.tensor(np.linalg.inv(R), dtype=X.dtype, device=X.device)  
        pfpu_T = pfpu.transpose(1, 2)      
        u_star = -0.5 * torch.matmul(pfpu_T, Lambda)  
        u_star = torch.matmul(R_inv, u_star.squeeze(-1).T).T  
        u_star = u_star + u0
        
        return u_star


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer


def get_all_loss(model, X):
    # Compute the quantities required by the loss
    V = model.get_value(X)
    Lambda = model.get_lambda(X)
    Lambda_unsq = Lambda.unsqueeze(2)
    Lambda_detach = Lambda.detach().clone().requires_grad_(True)
    Lambda_detach_unsq = Lambda_detach.unsqueeze(2)  

    hamilton_input = torch.cat([X, Lambda_detach], dim=-1)
    H = model.hamilton_part(hamilton_input)

    # pVpX
    pVpX = torch.autograd.grad(
        V, X,
        grad_outputs=torch.ones_like(V),
        create_graph=True,
        retain_graph=True
    )[0]  # (batch, state_dim)

    # pHpλ
    pHpLambda = torch.autograd.grad(
        H, Lambda_detach, 
        grad_outputs=torch.ones_like(H), 
        create_graph=True, 
        retain_graph=True
    )[0]  # (batch, state_dim)

    # pHpX
    pHpX = torch.autograd.grad(
        H, X, 
        grad_outputs=torch.ones_like(H), 
        create_graph=True, 
        retain_graph=True
    )[0]  # (batch, state_dim)

    # action
    R = model.env.R
    R_torch = torch.tensor(R, dtype=X.dtype, device=X.device)
    pfpu = get_pfpu(X)  
    R_inv = torch.tensor(np.linalg.inv(R), dtype=X.dtype, device=X.device)  
    pfpu_T = pfpu.transpose(1, 2)      
    delta_u = -0.5 * torch.matmul(pfpu_T, Lambda_unsq)  
    delta_u = torch.matmul(R_inv, delta_u.squeeze(-1).T).T  
    u_star = delta_u + u0_tensor
    u_star = torch.clamp(u_star, u_min_tensor, u_max_tensor)

    # Assemble the loss terms

    loss1 = torch.nn.functional.mse_loss(Lambda, pVpX)
    

    loss2 = torch.relu(-V).mean()


    fxu = dynamics(X, u_star)
    fxu_detach = fxu.detach().clone()
    loss3 = torch.nn.functional.mse_loss(fxu_detach, pHpLambda)


    Q = model.env.Q
    Q_torch = torch.tensor(Q, dtype=X.dtype, device=X.device)
    pfpx = get_pfpx(X, u_star) 
    term1 = 2 * torch.matmul(X, Q_torch.T)  
    pfpx_T = pfpx.transpose(1, 2)  
    term2 = torch.matmul(pfpx_T, Lambda_detach_unsq).squeeze(-1)  
    Lambda_dot = -(term1 + term2)  
    loss4 = torch.nn.functional.mse_loss(-pHpX, Lambda_dot)


    xQx = torch.einsum('bi,ij,bj->b', X, Q_torch, X)  # (batch_size,)
    uRu = torch.einsum('bi,ij,bj->b', delta_u, R_torch, delta_u)  # (batch_size,)
    gradV_fxu = (Lambda * fxu).sum(dim=1)
    H_eq = xQx + uRu + gradV_fxu  # The Hamiltonian residual should vanish
    loss5 = (H_eq ** 2).mean()
    

    # loss6 = torch.nn.functional.mse_loss(H_eq, H)

    return loss1 + loss2 + loss3 + loss4 + loss5 #+ loss6


def lambda_zero_loss(model):
    zero_point = torch.zeros(1, model.env.observation_space.shape[0], device=next(model.parameters()).device)
    zero_point_Lambda = model.get_lambda(zero_point)
    # Differentiate the costate output at the origin
    zero_point = zero_point.clone().detach().requires_grad_(True)
    Lambda = model.get_lambda(zero_point)  # (1, state_dim)
    grad_Lambda = []
    for i in range(Lambda.shape[1]):
        grad = torch.autograd.grad(
            Lambda[0, i], zero_point,
            retain_graph=True,
            create_graph=True
        )[0]  # (1, state_dim)
        grad_Lambda.append(grad)
    grad_Lambda = torch.stack(grad_Lambda, dim=1)  # (1, state_dim, state_dim)
    zero_loss = torch.nn.functional.mse_loss(zero_point_Lambda, torch.zeros_like(zero_point_Lambda))

    P_ref_tensor = torch.tensor(P_ref, dtype=grad_Lambda.dtype, device=grad_Lambda.device)
    grad_lambda_loss = torch.nn.functional.mse_loss(grad_Lambda.squeeze(0), 2*P_ref_tensor)
    # loss = zero_loss + grad_lambda_loss
    loss = grad_lambda_loss
    return loss


def train_hnn():
    env = gym.make(args.env_id).unwrapped
    hnn = HNN(env).to(args.device)
    model_dir = "model"
    model_path = os.path.join(model_dir, args.model_name)
    # hnn.load_state_dict(torch.load(model_path, map_location=args.device))
    optimizer = optim.Adam(hnn.parameters(), lr=args.learning_rate, weight_decay=0.01)


    dt = args.T / args.forward_steps
    Q = env.metadata['Q']
    R = env.metadata['R']
    P = env.P
    Q_torch = torch.tensor(Q, dtype=torch.float32, device=args.device)
    R_torch = torch.tensor(R, dtype=torch.float32, device=args.device)
    P_torch = torch.tensor(P, dtype=torch.float32, device=args.device)
    
    start_time = time.time()
    best_loss = float('inf')
    patience = 500
    patience_counter = 0
    min_steps = 1000
    ema_loss = None

    for step in range(args.time_steps):

        # Sample training data
        low = -args.train_range
        high = args.train_range
        x_batch = np.random.uniform(low, high, (args.batch_size, env.observation_space.shape[0]))
        X = torch.from_numpy(x_batch).float().to(args.device)
        X.requires_grad_(True)
        # Compute the loss
        loss = get_all_loss(hnn, X) + lambda_zero_loss(hnn)

        # Running-cost loss
        x = X.detach().clone()
        for i in range(args.forward_steps):
            pfpu = get_pfpu(x)
            Lambda = hnn.get_lambda(x) 
            Lambda_unsq = Lambda.unsqueeze(2) 
            R_inv = torch.tensor(np.linalg.inv(R), dtype=x.dtype, device=x.device)  
            pfpu_T = pfpu.transpose(1, 2)      
            delta_u = -0.5 * torch.matmul(pfpu_T, Lambda_unsq)
            delta_u = torch.matmul(R_inv, delta_u.squeeze(-1).T).T  
            u_star = delta_u + u0_tensor
            fxu = dynamics(x, u_star)
            x = x + fxu * dt
            x = torch.clamp(x, x_min_tensor, x_max_tensor)
            xQx = torch.einsum('bi,ij,bj->b', x, Q_torch, x)
            uRu = torch.einsum('bi,ij,bj->b', u_star-u0_tensor, R_torch, u_star-u0_tensor)
            loss += (xQx + uRu).mean() * dt
        # Terminal-cost loss
        xPx =  torch.einsum('bi,ij,bj->b', x, P_torch, x)
        loss += xPx.mean()

        # Update model parameters
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        current_loss = loss.item()
        if ema_loss is None:
            ema_loss = current_loss
        else:
            ema_loss = 0.95 * ema_loss + 0.05 * current_loss

        # Report the loss
        if step % 100 == 0:
            if ema_loss < best_loss - 1:
                best_loss = ema_loss
                patience_counter = 0
            else:
                patience_counter += 100
                
            print(f"Step {step+1}/{args.time_steps}, Loss: {current_loss:.6f}, EMA Loss: {ema_loss:.6f}, Patience: {patience_counter}/{patience}")

        if step > min_steps and patience_counter >= patience:
            end_time = time.time()
            print(f"EMA Loss has not improved significantly for {patience} steps. Stopping training at step {step+1}.")
            print(f"Total training time: {end_time - start_time:.2f} seconds.")
            model_dir = "model"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, args.model_name)
            torch.save(hnn.state_dict(), model_path)
            print(f"HNN model saved to {model_path}")
            break

        # Save the PINN checkpoint in the model directory
        if step > 1000 and (step % 5000 == 0 or step == (args.time_steps-1)):
            model_dir = "model"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, args.model_name)
            torch.save(hnn.state_dict(), model_path)
            print(f"HNN model saved to {model_path}")
            if step == (args.time_steps-1):
                end_time = time.time()
                print(f"Training completed. Total training time: {end_time - start_time:.2f} seconds.")


def plot_value():
    # Load model parameters
    args = Args()
    env = gym.make(args.env_id).unwrapped
    hnn = HNN(env)
    model_dir = "model"
    model_path = os.path.join(model_dir, args.model_name)
    hnn.load_state_dict(torch.load(model_path, map_location="cpu"))
    hnn.eval()

    # Construct a nine-dimensional state slice
    x1 = np.linspace(-args.train_range, args.train_range, 50)
    x2 = np.linspace(-args.train_range, args.train_range, 50)
    X1, X2 = np.meshgrid(x1, x2)
    X1_flat = X1.ravel()
    X2_flat = X2.ravel()
    # Set the remaining seven dimensions to zero
    X_rest = [np.zeros_like(X1_flat) for _ in range(7)]
    X_grid = np.stack([X1_flat, X2_flat] + X_rest, axis=1)  # shape (N, 9)


    X_tensor = torch.from_numpy(X_grid).float()
    with torch.no_grad():
        hnn_values = hnn.get_value(X_tensor).cpu().numpy()
    hnn_values = hnn_values.reshape(X1.shape)


    P = env.P  # Read the LQR solution from the environment
    lqr_values = np.einsum('ij,jk,ik->i', X_grid, P, X_grid).reshape(X1.shape)

    # Plot the comparison surfaces
    fig = plt.figure(figsize=(16,6))

    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    surf1 = ax1.plot_surface(X1, X2, hnn_values, cmap='viridis', alpha=1)
    surf2 = ax1.plot_surface(X1, X2, lqr_values, cmap='plasma', alpha=0.5)
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('Value')
    ax1.set_title('Value Surface (slice $x_3$-$x_{9}=0$)')

    # Add and label color bars
    mappable1 = plt.cm.ScalarMappable(cmap='viridis')
    mappable1.set_array(hnn_values)
    cbar1 = fig.colorbar(mappable1, ax=ax1, shrink=0.5, pad=0.1)
    cbar1.set_label('HNN Value', fontsize=12)

    mappable2 = plt.cm.ScalarMappable(cmap='plasma')
    mappable2.set_array(lqr_values)
    cbar2 = fig.colorbar(mappable2, ax=ax1, shrink=0.5, pad=0.05)
    cbar2.set_label('LQR Value', fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_lambda():
    # Model parameters
    args = Args()
    env = gym.make(args.env_id).unwrapped
    hnn = HNN(env)
    model_dir = "model"
    model_path = os.path.join(model_dir, args.model_name)
    hnn.load_state_dict(torch.load(model_path, map_location="cpu"))
    hnn.eval()


    P = env.P

    # Construct a nine-dimensional state slice
    x1 = np.linspace(-args.train_range, args.train_range, 30)
    x2 = np.linspace(-args.train_range, args.train_range, 30)
    X1, X2 = np.meshgrid(x1, x2)
    X1_flat = X1.ravel()
    X2_flat = X2.ravel()
    X_rest = [np.zeros_like(X1_flat) for _ in range(7)]
    X_grid = np.stack([X1_flat, X2_flat] + X_rest, axis=1)  # shape (N, 9)

    X_tensor = torch.from_numpy(X_grid).float()

    # HNN Lambda
    with torch.no_grad():
        hnn_lambda = hnn.get_lambda(X_tensor).cpu().numpy()  # shape (N, 9)
    # Reshape each state component
    hnn_lambda_list = [hnn_lambda[:, i].reshape(X1.shape) for i in range(9)]

    # LQR Lambda: Lambda = 2 * P * X
    lqr_lambda = 2 * (X_grid @ P)  # shape (N, 9)
    lqr_lambda_list = [lqr_lambda[:, i].reshape(X1.shape) for i in range(9)]

    lambda_names = [rf'$\lambda_{i+1}$' for i in range(9)]


    fig = plt.figure(figsize=(24, 12))
    for i in range(9):
        ax = fig.add_subplot(3, 3, i+1, projection='3d')
        surf1 = ax.plot_surface(X1, X2, hnn_lambda_list[i], cmap='Blues', alpha=0.8, edgecolor='none')
        surf2 = ax.plot_surface(X1, X2, lqr_lambda_list[i], cmap='Oranges', alpha=0.5, edgecolor='none')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel(lambda_names[i])
        ax.set_title(f"{lambda_names[i]} (slice $x_3$-$x_{{9}}=0$)")
        # Add the color bar only for OPINN
        mappable1 = plt.cm.ScalarMappable(cmap='Blues')
        mappable1.set_array(hnn_lambda_list[i])
        cbar1 = fig.colorbar(mappable1, ax=ax, shrink=0.6, pad=0.05)
        cbar1.set_label('HNN', fontsize=10)
        mappable2 = plt.cm.ScalarMappable(cmap='Oranges')
        mappable2.set_array(lqr_lambda_list[i])
        cbar2 = fig.colorbar(mappable2, ax=ax, shrink=0.6, pad=0.01)
        cbar2.set_label('LQR', fontsize=10)
        # Legend
        custom_lines = [
            plt.Line2D([0], [0], color='tab:blue', lw=4, label='HNN'),
            plt.Line2D([0], [0], color='tab:orange', lw=4, label='LQR'),
        ]
        ax.legend(handles=custom_lines, loc='upper left')
    plt.tight_layout()
    plt.show()


def simulate_once():
    # Simulate LQR and OPINN closed-loop control
    args = Args()
    device = args.device
    env_probe = gym.make(args.env_id, theoretic_mode=True).unwrapped

    # Simulation parameters
    horizon = int(env_probe.T / env_probe.dt)
    A_np = env_probe.A
    B_np = env_probe.B
    Q_np = env_probe.Q
    R_np = env_probe.R
    m = env_probe.metadata['m']
    g = env_probe.metadata['g']
    u0 = env_probe.u0

    # Compute the LQR controller
    P = solve_continuous_are(A_np, B_np, Q_np, R_np)
    K = np.linalg.inv(R_np) @ B_np.T @ P
    def lqr_action(x_np):
        u = -K @ x_np + u0
        u = np.clip(u, env_probe.action_space.low, env_probe.action_space.high)
        return np.array(u, dtype=np.float32)

    # Load the OPINN model
    hnn_path = "model/" + args.model_name
    env_hnn = gym.make(args.env_id, theoretic_mode=True).unwrapped
    hnn = HNN(env_hnn).to(device)
    hnn.load_state_dict(torch.load(hnn_path, map_location=device))
    hnn.eval()

    # OPINN controller
    R_torch = torch.tensor(R_np, dtype=torch.float32, device=device)
    R_inv = torch.linalg.inv(R_torch)
    def hnn_action(x_np):
        x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)  # (1, 9)
        with torch.no_grad():
            Lambda = hnn.get_lambda(x)  # (1, 9)
        pfpu = get_pfpu(x)  # (1, 9, 4)
        pfpu_T = pfpu.transpose(1, 2)  # (1, 4, 9)
        u_tmp = -0.5 * torch.matmul(pfpu_T, Lambda.unsqueeze(2)).squeeze(-1)  # (1, 4)
        u = (R_inv @ u_tmp.T).T.squeeze(0)  # (4,)
        u = u.detach().cpu().numpy()
        u = u + u0  # Add gravity compensation
        u = np.clip(u, env_probe.action_space.low, env_probe.action_space.high)
        return np.array(u, dtype=np.float32)

    # Initialize the environments
    seed = np.random.randint(0, 10000000)
    env_lqr = gym.make(args.env_id, theoretic_mode=True).unwrapped
    obs_lqr, _ = env_lqr.reset(seed=seed)
    env_hnn = gym.make(args.env_id, theoretic_mode=True).unwrapped
    obs_hnn, _ = env_hnn.reset(seed=seed)  # Use the same initial condition

    ret_lqr = 0.0
    u_seq_lqr = []
    x_seq_lqr = [obs_lqr.copy()]

    ret_hnn = 0.0
    u_seq_hnn = []
    x_seq_hnn = [obs_hnn.copy()]

    lqr_trunc_count = 0
    hnn_trunc_count = 0
    for t in range(horizon):
        print(f"{t}/{horizon}")
        # LQR
        u_l = lqr_action(obs_lqr.astype(np.float32))
        next_obs_l, r_l, term_l, trunc_l, _ = env_lqr.step(u_l)
        ret_lqr += float(r_l)
        u_seq_lqr.append(u_l)
        obs_lqr = next_obs_l
        x_seq_lqr.append(obs_lqr.copy())
        if not(term_l or trunc_l):
            lqr_trunc_count += 1
        # HNN
        u_h = hnn_action(obs_hnn.astype(np.float32))
        next_obs_h, r_h, term_h, trunc_h, _ = env_hnn.step(u_h)
        ret_hnn += float(r_h)
        u_seq_hnn.append(u_h)
        obs_hnn = next_obs_h
        x_seq_hnn.append(obs_hnn.copy())
        if not(term_l or trunc_l):
            hnn_trunc_count +=1

    env_lqr.close()
    env_hnn.close()

    traj_u_lqr = np.vstack(u_seq_lqr)
    traj_x_lqr = np.vstack(x_seq_lqr)
    traj_u_hnn = np.vstack(u_seq_hnn)
    traj_x_hnn = np.vstack(x_seq_hnn)

    # Plot control trajectories
    time = np.arange(horizon)
    plt.figure(figsize=(12, 6))
    for i in range(4):
        plt.plot(time[:], traj_u_lqr[:, i], label=f"LQR u{i+1}", linestyle='--')
        plt.plot(time[:], traj_u_hnn[:, i], label=f"HNN u{i+1}")
    plt.xlabel("Time step")
    plt.ylabel("u")
    plt.title("Control trajectories (single episode)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    # Plot state trajectories
    time_x = np.arange(horizon + 1)
    fig, axs = plt.subplots(9, 1, figsize=(12, 14), sharex=True)
    for i, ax in enumerate(axs):
        ax.plot(time_x[:], traj_x_lqr[:, i], label="LQR", color="tab:green")
        ax.plot(time_x[:], traj_x_hnn[:, i], label="HNN", color="tab:blue")
        ax.set_ylabel(f"x{i+1}")
        ax.grid(True, linestyle='--', alpha=0.4)
        if i == 0:
            ax.legend()
    axs[-1].set_xlabel("Time step")
    axs[2].set_ylim([-1, 1])
    fig.suptitle("State trajectories (single episode)")
    plt.tight_layout()

    # Report returns
    print("Single episode return:")
    print(f"LQR: {ret_lqr:.3f}")
    print(f"HNN: {ret_hnn:.3f}")

    # drone_visualizer = DroneVisualizer(traj=traj_x_hnn[:hnn_trunc_count, 0:3], attitudes=np.rad2deg(traj_x_hnn[:hnn_trunc_count, 6:9]), alpha_range=(0.25, 1.0), stride=60, pause=0.04)
    # drone_visualizer.show()


    import cmaps
    cmap_hnn = cmaps.MPL_PuOr_r
    cmap_lqr = cmaps.MPL_RdYlGn_r
    drone_visualizer = DroneVisualizer(traj=traj_x_hnn[:, 0:3], alpha_range=[0.2, 1])
    drone_visualizer.add_one_traj(traj=traj_x_hnn[:hnn_trunc_count, 0:3], attitudes=np.rad2deg(traj_x_hnn[:hnn_trunc_count, 6:9]), stride=50, colormap=cmap_hnn) 
    drone_visualizer.add_one_traj(traj=traj_x_lqr[:lqr_trunc_count, 0:3], attitudes=np.rad2deg(traj_x_lqr[:lqr_trunc_count, 6:9]), stride=50, colormap=cmap_lqr)
    drone_visualizer.ax.set_zlim([-1, 1])
    plt.savefig('image/quad_traj_single.svg', dpi=600, bbox_inches='tight', pad_inches=0.5, transparent=True)

    # Control trajectories
    control_labels = ['$T$\,(N)', '$p$\,(rad/s)', '$q$\,(rad/s)', '$r$\,(rad/s)']

    nature_green = "#389826"   # LQR
    nature_purple = "#9558B2"  # HNN
    fig, axs = plt.subplots(4, 1, figsize=(6, 6.3), sharex=True)
    for i, ax in enumerate(axs):

        ax.plot(time[:], traj_u_lqr[:, i], color=nature_green, alpha=0.6, linewidth=3, label="LQR" if i==0 else "")

        ax.plot(time[:], traj_u_hnn[:, i], color=nature_purple, alpha=0.8, linewidth=3, label="HNN" if i==0 else "")
        ax.set_ylabel(control_labels[i])
        ax.grid(True, linestyle='--', alpha=0.36)
    axs[-1].set_xlabel("$t$ (s)")
    plt.tight_layout()
    filename = "image/" + args.filename_prefix + "control_traj_single.svg"

    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5, transparent=True)

    plt.show()


def monte_carlo_simulation():
    # seed = 200
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    num_episodes = 50

    env_lqr = gym.make(args.env_id, theoretic_mode=True).unwrapped
    env_hnn = gym.make(args.env_id, theoretic_mode=True).unwrapped
    horizon = int(env_lqr.T / env_lqr.dt)

    m = env_lqr.metadata['m']
    g = env_lqr.metadata['g']
    Q = env_lqr.metadata['Q']
    R = env_lqr.metadata['R']
    u0 = env_lqr.u0

    # LQR controller
    A = env_lqr.A
    B = env_lqr.B
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P


    device = "cpu"
    hnn = HNN(env_hnn).to(device)
    hnn.load_state_dict(torch.load("model/"+args.model_name, map_location=device))
    hnn.eval()

    # OPINN controller
    R_torch = torch.tensor(R, dtype=torch.float32, device=device)
    R_inv = torch.linalg.inv(R_torch)
    def hnn_action(x_np):
        x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)  # (1, 9)
        with torch.no_grad():
            Lambda = hnn.get_lambda(x)  # (1, 9)
        pfpu = get_pfpu(x)  # (1, 9, 4)
        pfpu_T = pfpu.transpose(1, 2)  # (1, 4, 9)
        u_tmp = -0.5 * torch.matmul(pfpu_T, Lambda.unsqueeze(2)).squeeze(-1)  # (1, 4)
        u = (R_inv @ u_tmp.T).T.squeeze(0)  # (4,)
        u = u.detach().cpu().numpy()
        u = u + u0
        u = np.clip(u, env_hnn.action_space.low, env_hnn.action_space.high)
        return np.array(u, dtype=np.float32)

    all_traj_x_lqr = []
    all_traj_u_lqr = []
    all_traj_x_hnn = []
    all_traj_u_hnn = []
    returns_lqr = []
    returns_hnn = []

    drone_visualizer = None
    for ep in range(num_episodes):
        print(f"{ep}/{num_episodes}")


        seed = np.random.randint(0, 1000000)
        obs_lqr, _ = env_lqr.reset(seed=seed, theoretic_mode=True)
        obs_hnn, _ = env_hnn.reset(seed=seed, theoretic_mode=True)
        x_seq_lqr = [obs_lqr.copy()]
        u_seq_lqr = []
        x_seq_hnn = [obs_hnn.copy()]
        u_seq_hnn = []
        ret_lqr = 0.0
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
            ret_lqr += reward_l

        # HNN episode
        hnn_truncated_count = 0
        obs_hnn_ep = obs_hnn.copy()
        for t in range(horizon):
            u_h = hnn_action(obs_hnn_ep)
            u_h = np.clip(u_h, env_hnn.action_space.low, env_hnn.action_space.high)
            next_obs_h, reward_h, terminated_h, truncated_h, _ = env_hnn.step(u_h)
            u_seq_hnn.append(u_h)
            obs_hnn_ep = next_obs_h
            x_seq_hnn.append(obs_hnn_ep.copy())
            ret_hnn += reward_h
            if not(terminated_h or truncated_h):
                hnn_truncated_count += 1

        all_traj_x_lqr.append(np.vstack(x_seq_lqr))
        all_traj_u_lqr.append(np.vstack(u_seq_lqr))
        all_traj_x_hnn.append(np.vstack(x_seq_hnn))
        all_traj_u_hnn.append(np.vstack(u_seq_hnn))
        returns_lqr.append(ret_lqr)
        returns_hnn.append(ret_hnn)

        traj_x_hnn = np.vstack(x_seq_hnn)
        if ep < 3:
            if drone_visualizer is None:
                drone_visualizer = DroneVisualizer(traj=traj_x_hnn[:, 0:3], alpha_range=[0.2, 1])
                drone_visualizer.add_one_traj(traj=traj_x_hnn[:hnn_truncated_count, 0:3], attitudes=np.rad2deg(traj_x_hnn[:hnn_truncated_count, 6:9]), stride=50)
            else:
                drone_visualizer.add_one_traj(traj=traj_x_hnn[:hnn_truncated_count, 0:3], attitudes=np.rad2deg(traj_x_hnn[:hnn_truncated_count, 6:9]), stride=50)
        else:
            drone_visualizer.add_only_traj(traj=deepcopy(traj_x_hnn[:hnn_truncated_count, 0:3]))
    # plt.savefig('image/quad_traj.svg', dpi=600, bbox_inches='tight', pad_inches=0.5)

    env_lqr.close()
    env_hnn.close()


    np.savez(
        "image/" + args.filename_prefix + "all_traj_mc.npz",
        all_traj_x_lqr=np.array(all_traj_x_lqr, dtype=object),
        all_traj_u_lqr=np.array(all_traj_u_lqr, dtype=object),
        all_traj_x_hnn=np.array(all_traj_x_hnn, dtype=object),
        all_traj_u_hnn=np.array(all_traj_u_hnn, dtype=object),
        returns_lqr=np.array(returns_lqr),
        returns_hnn=np.array(returns_hnn),
    )


    data = np.load("image/" + args.filename_prefix + "all_traj_mc.npz", allow_pickle=True)
    all_traj_x_lqr = data['all_traj_x_lqr']
    all_traj_u_lqr = data['all_traj_u_lqr']
    all_traj_x_hnn = data['all_traj_x_hnn']
    all_traj_u_hnn = data['all_traj_u_hnn']
    returns_lqr = data['returns_lqr']
    returns_hnn = data['returns_hnn']


    time_x = np.arange(horizon + 1)
    time_u = np.arange(horizon)

    # State trajectories
    fig, axs = plt.subplots(9, 1, figsize=(12, 14), sharex=True)
    for i, ax in enumerate(axs):
        for ep in range(num_episodes):
            # ax.plot(time_x*env.dt, all_traj_x_lqr[ep][:, i], color="tab:green", alpha=0.2)
            ax.plot(time_x*env.dt, all_traj_x_hnn[ep][:, i], color="tab:blue", alpha=0.2)
        ax.set_ylabel(f"$x_{i+1}$")
        ax.grid(True, linestyle='--', alpha=0.4)
        # if i == 0:
        #     ax.set_title("State trajectories (50 episodes)")
    axs[-1].set_xlabel("$t(s)$")
    plt.tight_layout()
    filename = "image/" + args.filename_prefix + "state_traj.png"
    # plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)

    # Control trajectories
    control_labels = ['$T$\,(N)', '$p$\,(rad/s)', '$q$\,(rad/s)', '$r$\,(rad/s)']

    nature_green = "#389826"   # LQR
    nature_purple = "#9558B2"  # HNN
    fig, axs = plt.subplots(4, 1, figsize=(6, 6.3), sharex=True)
    for i, ax in enumerate(axs):

        for ep in range(num_episodes):
            ax.plot(time_u*env.dt, all_traj_u_lqr[ep][:, i], color=nature_green, alpha=0.22, linewidth=1)

        for ep in range(num_episodes):
            ax.plot(time_u*env.dt, all_traj_u_hnn[ep][:, i], color=nature_purple, alpha=0.22, linewidth=1)
        # Bold font
        mean_lqr = np.stack([all_traj_u_lqr[ep][:, i] for ep in range(num_episodes)]).mean(axis=0)
        mean_hnn = np.stack([all_traj_u_hnn[ep][:, i] for ep in range(num_episodes)]).mean(axis=0)
        ax.plot(time_u*env.dt, mean_lqr, color=nature_green, linewidth=2.6, label="LQR" if i==0 else "")
        ax.plot(time_u*env.dt, mean_hnn, color=nature_purple, linewidth=2.6, label="HNN" if i==0 else "")
        ax.set_ylabel(control_labels[i])
        ax.grid(True, linestyle='--', alpha=0.36)
    axs[-1].set_xlabel("$t$ (s)")
    plt.tight_layout()
    filename = "image/" + args.filename_prefix + "control_traj.svg"
    # plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)



    final_dist_lqr = np.array([np.linalg.norm(traj[-1, 0:3]) for traj in all_traj_x_lqr])
    final_dist_hnn = np.array([np.linalg.norm(traj[-1, 0:3]) for traj in all_traj_x_hnn])
    # Treat terminal distances above 0.1 as failures
    success_mask_lqr = final_dist_lqr <= 0.1
    success_mask_hnn = final_dist_hnn <= 0.1
    filtered_returns_lqr = returns_lqr[success_mask_lqr]
    filtered_returns_hnn = returns_hnn[success_mask_hnn]
    success_rate_lqr = np.sum(success_mask_lqr) / len(success_mask_lqr)
    success_rate_hnn = np.sum(success_mask_hnn) / len(success_mask_hnn)
    import seaborn as sns
    import pandas as pd


    df_box = pd.DataFrame({
        "Return": np.concatenate((-filtered_returns_lqr, -filtered_returns_hnn)),
        "Controller": ["LQR"]*len(filtered_returns_lqr) + ["HNN"]*len(filtered_returns_hnn)
    })

    # lqr_label = r"LQR:%.1f$\%%$" % (success_rate_lqr*100)
    # hnn_label = r"HNN:%.1f$\%%$" % (success_rate_hnn*100)
    lqr_label = r"LQR"
    hnn_label = r"HNN"
    show_labels = [lqr_label, hnn_label]


    nature_green = "#389826"   # LQR
    nature_purple = "#9558B2"  # HNN
    palette = [nature_green, nature_purple]


    fig, ax = plt.subplots(figsize=(5, 7))

    sns.violinplot(
        x="Controller", y="Return", data=df_box,
        order=["LQR", "HNN"],
        palette=palette, inner=None, alpha=0.22, cut=0,
        linewidth=1.2, ax=ax
    )

    sns.boxplot(
        x="Controller", y="Return", data=df_box,
        order=["LQR", "HNN"],
        palette=palette, width=0.25,
        showcaps=True, showbox=True, showfliers=False, medianprops=dict(color="#6D2D2B", linewidth=2),
        boxprops=dict(alpha=0.7, edgecolor='k', linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        ax=ax
    )


    sns.stripplot(
        x="Controller", y="Return", data=df_box,
        order=["LQR", "HNN"], 
        palette=palette,
        size=7, alpha=0.4, jitter=0.22, linewidth=0.2, edgecolor='#444', ax=ax
    )
    ax.set_xticklabels(show_labels)
    ax.set_ylabel(r"$J$")
    ax.grid(True, linestyle='--', alpha=0.45)
    sns.despine(top=False, right=False, left=False, bottom=False, ax=ax)
    filename = "image/" + args.filename_prefix + "return_boxplot.svg"
    # plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)
   
    # Report return statistics
    print("Monte Carlo 50 episodes return statistics:")
    print(f"LQR: mean={np.mean(returns_lqr):.3f}, std={np.std(returns_lqr):.3f}, min={np.min(returns_lqr):.3f}, max={np.max(returns_lqr):.3f}")
    print(f"HNN: mean={np.mean(returns_hnn):.3f}, std={np.std(returns_hnn):.3f}, min={np.min(returns_hnn):.3f}, max={np.max(returns_hnn):.3f}")

    plt.show()



if __name__ == "__main__":
    args = Args()
    env = gym.make(args.env_id).unwrapped
    m = env.metadata['m']
    g = env.metadata['g']
    P_ref = env.P
    u0_tensor = torch.tensor(env.u0, dtype=torch.float32, device=args.device)
    u0 = env.u0
    u_min_tensor = torch.tensor(env.u_min, dtype=torch.float32, device=args.device)
    u_max_tensor = torch.tensor(env.u_max, dtype=torch.float32, device=args.device)
    x_min_tensor = torch.tensor(env.state_low, dtype=torch.float32, device=args.device)
    x_max_tensor = torch.tensor(env.state_high, dtype=torch.float32, device=args.device)


    train_hnn()
    # plot_value()
    # plot_lambda()
    # simulate_once()
    monte_carlo_simulation()
