'''
Combine all loss terms and use a positive-definite value-network design.
'''
import sys, os
import time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from sympy.geometry.entity import x
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import gymnasium as gym


# Plot settings
mpl.rcParams['font.family'] = 'Times New Roman'  # Default font
# Bold font
mpl.rcParams['text.usetex'] = True  # Enable TeX rendering
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
    env_id = "environment:linear2d-v0"
    # Learning rate
    learning_rate = 1e-3
    # Total training iterations
    time_steps = 20000
    # Training batch size
    batch_size = 2048
    # Select the compute device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Output filename prefix
    filename_prefix = "hnn_linear2d_"


# Dynamics
def dynamics(X, U):

    # X: (batch_size, 2)
    # U: (batch_size, 1) or (batch_size,)
    # A: 2x2, B: 2x1
    A = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=X.dtype, device=X.device)
    B = torch.tensor([[0.0], [1.0]], dtype=X.dtype, device=X.device)
    # Ensure U is (batch_size, 1)
    if U.dim() == 1:
        U = U.unsqueeze(1)

    fxu = torch.matmul(X, A.T) + torch.matmul(U, B.T)
    return fxu


# Compute the control Jacobian
def get_pfpu(X):
    # X: (batch_size, n)
    batch_size, n = X.shape


    B = torch.tensor([[0.0], [1.0]], device=X.device, dtype=X.dtype).expand(batch_size, 2, 1)
    
    return B


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
        self.value_part = nn.Sequential(
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # self.state_dim = env.observation_space.shape[0]
        # self.value_element = nn.Sequential(
        #     layer_init(nn.Linear(64, 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, self.state_dim * (self.state_dim + 1) // 2), std=1.0),
        # )
        # self.indices = [(i, j) for i in range(self.state_dim) for j in range(i + 1)]

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
        # x = x - torch.tensor([0, 0], dtype=x.dtype, device=x.device)
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
        Lambda = self.get_lambda(X) # (batch_size, 2)
        Lambda = Lambda.unsqueeze(2)  # (batch_size, 2, 1)
        
        pfpu = get_pfpu(X)  # (batch_size, 2, 1)
        

        # R: numpy 1x1, pfpu: (batch_size, 2, 1), grad_V: (batch_size, 2)
        R_inv = torch.tensor(np.linalg.inv(R), dtype=X.dtype, device=X.device)  # (1, 1)
        pfpu_T = pfpu.transpose(1, 2)      # (batch_size, 1, 2)
        # (batch_size, 1, 2) @ (batch_size, 2, 1) -> (batch_size, 1, 1)
        u_star = -0.5 * torch.matmul(pfpu_T, Lambda)  # (batch_size, 1, 1)
        u_star = torch.matmul(R_inv, u_star.squeeze(-1).T).T  # (batch_size, 1)
        
        return u_star


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer



def value_lambda_loss(model, X):
    V = model.get_value(X).squeeze(-1)  # (batch,)
    grad_V = torch.autograd.grad(
        V, X,
        grad_outputs=torch.ones_like(V),
        create_graph=True,
        retain_graph=True
    )[0]  # (batch, state_dim)
    Lambda = model.get_lambda(X)  # (batch, state_dim)
    loss = torch.nn.functional.mse_loss(Lambda, grad_V)
    return loss



def positive_v_loss(model, X):

    V = model.get_value(X).squeeze(-1)  # (batch,)

    loss = torch.relu(-V).mean()
    return loss



def dynamics_loss(model, X):
    u_star = model.get_action(X)
    fxu = dynamics(X, u_star)
    pHpLambda = model.get_pHpLambda(X)
    loss = torch.nn.functional.mse_loss(fxu, pHpLambda)
    return loss



def lambda_dot_loss(model, X):
    Q = model.env.Q
    Q_torch = torch.tensor(Q, dtype=X.dtype, device=X.device)

    pfpx = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=X.dtype, device=X.device)

    Lambda = model.get_lambda(X)
    pHpX = model.get_pHpX(X)


    # X: (batch_size, 2)
    # Q_torch: (2,2)
    # pfpx: (2,2)
    # Lambda: (batch_size, 2)
    # pfpx^T: (2,2)
    # Lambda.unsqueeze(2): (batch_size, 2, 1)
    # pfpx^T @ Lambda^T: (2,2) @ (batch_size,2,1) -> (batch_size,2,1) -> squeeze(-1) -> (batch_size,2)
    term1 = 2 * torch.matmul(X, Q_torch.T)  # (batch_size, 2)
    pfpx_T = pfpx.T  # (2,2)
    Lambda_unsq = Lambda.unsqueeze(2)  # (batch_size, 2, 1)
    term2 = torch.matmul(pfpx_T, Lambda_unsq).squeeze(-1)  # (batch_size, 2)
    Lambda_dot = -(term1 + term2)  # (batch_size, 2)
    loss = torch.nn.functional.mse_loss(-pHpX, Lambda_dot)  # Return the Hamiltonian gradient; the loss is assembled externally
    return loss


def hamiltonian_loss(model, X):
    Q = model.env.Q
    R = model.env.R

    u_star = model.get_action(X)
    grad_V = model.get_lambda(X)


    fxu = dynamics(X, u_star)


    # x: (batch_size, 2), Q: (2,2), u_star: (batch_size, 1), R: (1,1), grad_V: (batch_size, 2), fxu: (batch_size, 2)
    Q_torch = torch.tensor(Q, dtype=X.dtype, device=X.device)
    R_torch = torch.tensor(R, dtype=X.dtype, device=X.device)

    # x^T Q x
    xQx = torch.einsum('bi,ij,bj->b', X, Q_torch, X)  # (batch_size,)

    # u_star^T R u_star
    uRu = torch.einsum('bi,ij,bj->b', u_star, R_torch, u_star)  # (batch_size,)

    # grad_V^T fxu
    gradV_fxu = (grad_V * fxu).sum(dim=1)  # (batch_size,)

    H = xQx + uRu + gradV_fxu  # The Hamiltonian residual should vanish
    H_loss = (H ** 2).mean()
    return H_loss


def direct_H_loss(model, X):
    H = model.get_hamilton(X)
    H_loss = (H ** 2).mean()
    return H_loss


def get_all_loss(model, X):
    # Compute the quantities required by the loss
    V = model.get_value(X)
    Lambda = model.get_lambda(X)
    Lambda_unsq = Lambda.unsqueeze(2)
    Lambda_detach = Lambda.detach().clone().requires_grad_(True)
    Lambda_detach_unsq = Lambda_detach.unsqueeze(2)  # (batch_size, 2, 1)

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
    pfpu = get_pfpu(X)  # (batch_size, 2, 1)
    R_inv = torch.tensor(np.linalg.inv(R), dtype=X.dtype, device=X.device)  # (1, 1)
    pfpu_T = pfpu.transpose(1, 2)      # (batch_size, 1, 2)
    u_star = -0.5 * torch.matmul(pfpu_T, Lambda_unsq)  # (batch_size, 1, 1)
    u_star = torch.matmul(R_inv, u_star.squeeze(-1).T).T  # (batch_size, 1)

    # Assemble the loss terms

    loss1 = torch.nn.functional.mse_loss(Lambda, pVpX)
    

    loss2 = torch.relu(-V).mean()


    fxu = dynamics(X, u_star)
    fxu_detach = fxu.detach().clone()
    loss3 = torch.nn.functional.mse_loss(fxu_detach, pHpLambda)


    Q = model.env.Q
    Q_torch = torch.tensor(Q, dtype=X.dtype, device=X.device)
    pfpx = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=X.dtype, device=X.device)
    term1 = 2 * torch.matmul(X, Q_torch.T)  # (batch_size, 2)
    pfpx_T = pfpx.T  # (2,2)
    term2 = torch.matmul(pfpx_T, Lambda_detach_unsq).squeeze(-1)  # (batch_size, 2)
    Lambda_dot = -(term1 + term2)  # (batch_size, 2)
    loss4 = torch.nn.functional.mse_loss(-pHpX, Lambda_dot)


    xQx = torch.einsum('bi,ij,bj->b', X, Q_torch, X)  # (batch_size,)
    u_star_connect = u_star = -0.5 * torch.matmul(pfpu_T, Lambda_unsq)  # (batch_size, 1, 1)
    u_star_connect = torch.matmul(R_inv, u_star_connect.squeeze(-1).T).T  # (batch_size, 1)
    uRu = torch.einsum('bi,ij,bj->b', u_star_connect, R_torch, u_star_connect)  # (batch_size,)
    gradV_fxu = (Lambda * fxu).sum(dim=1)
    H = xQx + uRu + gradV_fxu  # The Hamiltonian residual should vanish
    loss5 = (H ** 2).mean()

    return loss1 + loss2 + loss3 + loss4 + loss5


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

    P_ref = [[np.sqrt(3), 1], [1, np.sqrt(3)]]
    P_ref_tensor = torch.tensor(P_ref, dtype=grad_Lambda.dtype, device=grad_Lambda.device)
    grad_lambda_loss = torch.nn.functional.mse_loss(grad_Lambda.squeeze(0), 2*P_ref_tensor)
    # loss = zero_loss + grad_lambda_loss
    loss = grad_lambda_loss
    return loss



def train_hnn():
    env = gym.make(args.env_id)
    hnn = HNN(env).to(args.device)
    optimizer = optim.Adam(hnn.parameters(), lr=args.learning_rate)

    # start_time = time.time()
    # print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    for step in range(args.time_steps):
        # if time.time() - start_time > 30:
        #     print(f"Time limit of 30s reached at step {step}. Stopping training.")
        #     model_dir = "model"
        #     os.makedirs(model_dir, exist_ok=True)
        #     model_path = os.path.join(model_dir, "hnn_linear2d.pth")
        #     torch.save(hnn.state_dict(), model_path)
        #     print(f"HNN model saved to {model_path}")
        #     break

        # Sample training data
        low = -4.0
        high = 4.0
        x_batch = np.random.uniform(low, high, (args.batch_size, env.observation_space.shape[0]))
        X = torch.from_numpy(x_batch).float().to(args.device)
        X.requires_grad_(True)
        # Compute the loss
        # posi_v_loss = positive_v_loss(hnn, X)
        # v_lam_loss = value_lambda_loss(hnn, X)
        # # lam_zero_loss = lambda_zero_loss(pinn)
        # dyn_loss = dynamics_loss(hnn, X)
        # lam_dot_loss = lambda_dot_loss(hnn, X)
        # H_loss = direct_H_loss(hnn, X)
        # loss = posi_v_loss + v_lam_loss + dyn_loss + lam_dot_loss + H_loss
        loss = get_all_loss(hnn, X) + lambda_zero_loss(hnn)

        # Update model parameters
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Report the loss
        if step % 100 == 0:
            print(f"Step {step+1}/{args.time_steps}, Loss: {loss.item():.6f}")

        # Save the PINN checkpoint in the model directory
        if step > 1000 and (step % 5000 == 0 or step == (args.time_steps-1)):
            model_dir = "model"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "hnn_linear2d.pth")
            torch.save(hnn.state_dict(), model_path)
            print(f"HNN model saved to {model_path}")


def plot_value():
    os.makedirs("image", exist_ok=True)
    # Load model parameters
    model_dir = "model"
    model_path = os.path.join(model_dir, "hnn_linear2d.pth")
    args = Args()
    env = gym.make(args.env_id)
    hnn = HNN(env)
    hnn.load_state_dict(torch.load(model_path, map_location="cpu"))
    hnn.eval()

    # Construct the evaluation grid
    x1 = np.linspace(-3, 3, 50)
    x2 = np.linspace(-3, 3, 50)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.stack([X1.ravel(), X2.ravel()], axis=1)


    X_tensor = torch.from_numpy(X_grid).float()
    with torch.no_grad():
        hnn_values = hnn.get_value(X_tensor).cpu().numpy()
    hnn_values = hnn_values.reshape(X1.shape)


    X_tensor.requires_grad_(True)

    Lambda = hnn.get_lambda(X_tensor)  # (N, 2)

    B = torch.tensor([[0.0], [1.0]], dtype=X_tensor.dtype, device=X_tensor.device)
    R = torch.tensor([[1.0]], dtype=X_tensor.dtype, device=X_tensor.device)

    u_star = -0.5*torch.matmul(Lambda, B)  # (N, 1)

    fxu = dynamics(X_tensor, u_star)

    V = hnn.get_value(X_tensor)
    V_sum = V.sum()
    pVpx = torch.autograd.grad(V_sum, X_tensor, create_graph=False)[0]  # (N, 2)

    V_dot_hnn = (pVpx * fxu).sum(dim=1).reshape(X1.shape).detach().numpy()


    P = np.array([[np.sqrt(3), 1.0], [1.0, np.sqrt(3)]])
    lqr_values = np.einsum('ij,jk,ik->i', X_grid, P, X_grid).reshape(X1.shape)

    # u_star_lqr = -0.5*R^{-1} B^T (2Px)
    X_grid_torch = torch.from_numpy(X_grid).float()
    P_torch = torch.from_numpy(P).float()
    B_lqr = torch.tensor([[0.0], [1.0]])
    R_inv_lqr = torch.tensor([[1.0]])
    gradV_lqr = 2 * torch.matmul(X_grid_torch, P_torch)  # (N,2)

    u_star_lqr = -0.5*torch.matmul(gradV_lqr, B_lqr)  # (N,1)
    fxu_lqr = dynamics(X_grid_torch, u_star_lqr)
    V_dot_lqr = (gradV_lqr * fxu_lqr).sum(dim=1).reshape(X1.shape).detach().numpy()

    # Plot the comparison surfaces
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1.4])
    ax = fig.add_subplot(gs[0], projection='3d')


    surf1 = ax.plot_surface(
        X1, X2, hnn_values, cmap='Purples', alpha=0.85, linewidth=0, antialiased=True, rstride=1, cstride=1
    )
    surf1_1 = ax.plot_surface(
        X1, X2, V_dot_hnn, cmap='Purples', alpha=0.85, linewidth=0, antialiased=True, rstride=1, cstride=1
    )
    surf2 = ax.plot_surface(
        X1, X2, lqr_values, cmap='Oranges', alpha=0.85, linewidth=0, antialiased=True, rstride=1, cstride=1
    )
    surf2_1 = ax.plot_surface(
        X1, X2, V_dot_lqr, cmap='Oranges', alpha=0.85, linewidth=0, antialiased=True, rstride=1, cstride=1
    )

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=plt.cm.Purples(0.7), lw=2, label='OCNet'),
        Line2D([0], [0], color=plt.cm.Oranges(0.7), lw=2, label='Optimal')
    ]
    ax.legend(handles=legend_elements, ncol=2, loc='upper right')


    ax.set_xlabel(r'$x_1$', labelpad=10)
    ax.set_ylabel(r'$x_2$', labelpad=10)
    ax.set_zlabel(r'$V$ or $\dot V$', labelpad=10)
    ax.view_init(elev=5, azim=30, roll=0)

    # plt.tight_layout()


    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim([-3, 3])
    ax2.set_ylim([-3, 3])



    x1_lin = np.linspace(-3, 3, 25)
    x2_lin = np.linspace(-3, 3, 25)
    X1_grid, X2_grid = np.meshgrid(x1_lin, x2_lin)
    U = np.zeros_like(X1_grid)
    V = np.zeros_like(X2_grid)


    for i in range(X1_grid.shape[0]):
        for j in range(X1_grid.shape[1]):
            x_point = np.array([X1_grid[i, j], X2_grid[i, j]])
            x_tensor = torch.from_numpy(x_point).float().unsqueeze(0)
            with torch.no_grad():
                Lambda = hnn.get_lambda(x_tensor).numpy().squeeze()
            B = np.array([[0], [1]])
            R = np.array([[1.0]])
            u = -np.linalg.inv(R) @ B.T @ Lambda.reshape(-1, 1)
            u = u.squeeze()
            U[i, j] = x_point[1]
            V[i, j] = u


    speed = np.sqrt(U**2 + V**2)
    color = speed


    step = 2
    Q = ax2.quiver(
        X1_grid[::step, ::step], X2_grid[::step, ::step], 
        U[::step, ::step], V[::step, ::step], color[::step, ::step],
        cmap='plasma', angles='xy', scale=100, width=0.005
    )
    cb = plt.colorbar(Q, ax=ax2, fraction=0.045, pad=0.04)
    cb.set_label(r"Magnitude")
    
    ax2.set_xlabel(r"$x_1$")
    ax2.set_ylabel(r"$x_2$")
    ax2.grid(True, linestyle='--', alpha=0.5)

    ax2.axis('equal')


    # plt.tight_layout()

    filename = "image/" + args.filename_prefix + "value_function.svg"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)


    fig2 = plt.figure(figsize=(6,5))
    ax2_3d = fig2.add_subplot(111, projection='3d')
    surf_fig2_1 = ax2_3d.plot_surface(
        X1, X2, hnn_values, cmap='Purples', alpha=0.8, linewidth=0, antialiased=True, rstride=1, cstride=1
    )
    surf_fig2_1_1 = ax2_3d.plot_surface(
        X1, X2, V_dot_hnn, cmap='Purples', alpha=0.8, linewidth=0, antialiased=True, rstride=1, cstride=1
    )
    surf_fig2_2 = ax2_3d.plot_surface(
        X1, X2, lqr_values, cmap='Oranges', alpha=1, linewidth=0, antialiased=True, rstride=1, cstride=1
    )
    surf_fig2_2_1 = ax2_3d.plot_surface(
        X1, X2, V_dot_lqr, cmap='Oranges', alpha=1, linewidth=0, antialiased=True, rstride=1, cstride=1
    )
    # ax2_3d.legend(handles=legend_elements, ncol=2, loc='upper right')
    ax2_3d.set_xlabel(r'$x_1$', labelpad=10)
    ax2_3d.set_ylabel(r'$x_2$', labelpad=10)
    ax2_3d.set_zlabel(r'$V$ and $\dot V$', labelpad=10)
    ax2_3d.view_init(elev=12, azim=124, roll=0)
    ax2_3d.set_xlim([-3, 3])
    ax2_3d.set_ylim([-3, 3])
    ax2_3d.set_zlim([-100, 50])
    filename2 = "image/" + args.filename_prefix + "value_function_3d_standalone.svg"
    plt.savefig(filename2, dpi=600, bbox_inches='tight', pad_inches=0.5)

    print("Max abs error of value function (HNN vs LQR):", np.max(np.abs(hnn_values - lqr_values)))
    print("Max abs error of time derivative (V_dot_hnn vs V_dot_lqr):", np.max(np.abs(V_dot_hnn - V_dot_lqr)))

    # Plot value-derivative contours
    fig3 = plt.figure(figsize=(6,5))
    ax3 = fig3.add_subplot(111, projection='3d')

    z_min = min(np.min(hnn_values), np.min(hnn_values))

    ax3.plot_surface(
        X1, X2, hnn_values, cmap='RdBu_r', alpha=0.8
    )

    surf3_1 = ax3.contour(
        X1, X2, hnn_values, zdir='z', offset=z_min, cmap='RdBu_r', alpha=1, levels=20
    )
    ax3.set_xlabel(r'$x_1$', labelpad=10)
    ax3.set_ylabel(r'$x_2$', labelpad=10)
    ax3.set_zlabel(r'$V$', labelpad=10, rotation=90)
    # ax3.view_init(elev=5, azim=30, roll=0)
    filename = "image/" + args.filename_prefix + "hnn_value.svg"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)

    # Plot value-function contours
    fig4 = plt.figure(figsize=(6,5))
    ax4 = fig4.add_subplot(111, projection='3d')

    z_min = min(np.min(V_dot_hnn), np.min(V_dot_hnn))

    ax4.plot_surface(
        X1, X2, V_dot_hnn, cmap='RdBu_r', alpha=0.8
    )

    surf4_1 = ax4.contour(
        X1, X2, V_dot_hnn, zdir='z', offset=z_min, cmap='RdBu_r', alpha=1, levels=20
    )
    ax4.set_xlabel(r'$x_1$', labelpad=10)
    ax4.set_ylabel(r'$x_2$', labelpad=10)
    ax4.set_zlabel(r'$\dot V$', labelpad=20)
    # ax4.view_init(elev=5, azim=30, roll=0)
    filename = "image/" + args.filename_prefix + "hnn_value_dot.svg"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)

    # Plot the phase portrait
    fig5 = plt.figure(figsize=(6,5))
    ax5 = fig5.add_subplot(111)

    step = 2
    Q = ax5.quiver(
        X1_grid[::step, ::step], X2_grid[::step, ::step], 
        U[::step, ::step], V[::step, ::step], color[::step, ::step],
        cmap='RdBu_r', angles='xy', scale=100, width=0.005
    )
    cb = plt.colorbar(Q, ax=ax5, fraction=0.045, pad=0.04)
    cb.set_label(r"Magnitude")
    ax5.set_xlabel(r'$x_1$', labelpad=10)
    ax5.set_ylabel(r'$x_2$', labelpad=10)
    ax5.axis('equal')
    filename = "image/" + args.filename_prefix + "quiver.svg"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)

    plt.show()


def simulate_once():
    """
    Load the trained OPINN and environment, run one simulation, and plot the state and control histories.
    """
    # Load the environment
    env = gym.make(Args.env_id)
    obs, _ = env.reset(seed=1318)
    print(obs)
    obs = np.array(obs, dtype=np.float32)
    
    # Load the network
    hnn = HNN(env)
    model_path = "model/" + "hnn_linear2d.pth"
    if os.path.exists(model_path):
        hnn.load_state_dict(torch.load(model_path, map_location=Args.device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model file {model_path} not found. Please train the model first.")
        return
    hnn.to(Args.device)
    hnn.eval()

    # Simulation parameters
    T = 200
    state_traj = [obs.copy()]
    control_traj = []
    time_traj = [0.0]
    for t in range(T):
        x_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(Args.device)

        with torch.no_grad():
            Lambda = hnn.get_lambda(x_tensor).cpu().numpy().squeeze()

        B = np.array([[0], [1]])
        R = np.diag([1])

        u = -np.linalg.inv(R) @ B.T @ Lambda.reshape(-1, 1)
        u = u.squeeze()
        # Update the state
        obs, _, terminated, truncated, _ = env.step(u)
        obs = np.array(obs, dtype=np.float32)
        state_traj.append(obs.copy())
        control_traj.append(u)
        time_traj.append((t+1)*env.dt if hasattr(env, "dt") else (t+1)*0.05)
        # if terminated or truncated:
        #     break
    state_traj = np.array(state_traj)
    control_traj = np.array(control_traj)
    time_traj = np.array(time_traj)


    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(time_traj, state_traj[:, 0], label='x1')
    axs[0].plot(time_traj, state_traj[:, 1], label='x2')
    axs[0].set_ylabel('State')
    axs[0].set_title('State Trajectory')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(time_traj[:-1], control_traj, label='u', color='tab:orange')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Control')
    axs[1].set_title('Control Input')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    args = Args()
    train_hnn()
    plot_value()
    # simulate_once()
