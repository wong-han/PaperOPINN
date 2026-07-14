'''
Kuramoto Oscillator Regulation with OPINN
'''
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.linalg import solve_continuous_are

# ================== 图像设置 =======================
mpl.rcParams['font.family'] = 'Times New Roman'  # 默认字体
mpl.rcParams['text.usetex'] = True  # 使用TEX
mpl.rcParams['axes.unicode_minus'] = False  # 解决无法显示负号
mpl.rcParams['xtick.direction'] = 'in'  # x轴刻度线朝内
mpl.rcParams['ytick.direction'] = 'in'  # y轴刻度线朝内
mpl.rcParams['xtick.top'] = True  # 显示上方的坐标轴
mpl.rcParams['ytick.right'] = True  # 显示右侧的坐标轴
mpl.rcParams['legend.frameon'] = False  # legend不显示边框
mpl.rcParams['legend.fontsize'] = 24  # legend默认size
mpl.rcParams['xtick.labelsize'] = 24  # x坐标默认size
mpl.rcParams['ytick.labelsize'] = 24  # y坐标默认size
mpl.rcParams['axes.labelsize'] = 24  # 轴标题默认size
mpl.rcParams['lines.linewidth'] = 2  # 线条宽度
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.linestyle'] = '--'  # 虚线
mpl.rcParams['grid.alpha'] = 0.5       # 透明度
mpl.rcParams['grid.color'] = 'gray'    # 网格颜色

# =================== 参数设置 ======================
class Args:
    # 环境参数
    N = 50
    K = 0.5
    omega_var = 0.2
    dt = 0.05
    T = 10.0
    # 模型名称
    model_name = "hnn_kuramoto_v1.pth"
    # 训练范围
    train_range = np.pi / 2
    # 训练参数
    time_steps = 15000
    batch_size = 1280
    learning_rate = 1e-3
    forward_steps = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = Args()

# Fix random seed for reproducibility of omega
np.random.seed(42)
OMEGA = np.random.normal(0, np.sqrt(args.omega_var), args.N)
OMEGA_TENSOR = torch.tensor(OMEGA, dtype=torch.float32, device=args.device)

# Q and R matrices
Q_np = np.eye(args.N) * 1.0
R_np = np.eye(args.N) * 1.0
Q_torch = torch.tensor(Q_np, dtype=torch.float32, device=args.device)
R_torch = torch.tensor(R_np, dtype=torch.float32, device=args.device)

# Equilibrium control
u0_np = -OMEGA
u0_tensor = torch.tensor(u0_np, dtype=torch.float32, device=args.device)

# =================== 动力学和 HNN 网络 ======================
def dynamics(x, u):
    # x: (B, N)
    # u: (B, N)
    B, N = x.shape
    sin_x = torch.sin(x)
    cos_x = torch.cos(x)
    
    sum_sin = torch.sum(sin_x, dim=1, keepdim=True)
    sum_cos = torch.sum(cos_x, dim=1, keepdim=True)
    
    coupling = (args.K / N) * (cos_x * sum_sin - sin_x * sum_cos)
    x_dot = OMEGA_TENSOR + coupling + u
    return x_dot

def get_pfpu(x):
    # x: (B, N)
    # df/du is Identity matrix since \dot{x} = f(x) + u
    B, N = x.shape
    I = torch.eye(N, dtype=x.dtype, device=x.device)
    return I.unsqueeze(0).repeat(B, 1, 1)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def get_pfpx(x, u):
    B, N = x.shape
    diff = x.unsqueeze(1) - x.unsqueeze(2) # shape: (B, N, N)
    C = (args.K / N) * torch.cos(diff)
    pfpx = C.clone()
    mask = torch.eye(N, dtype=torch.bool, device=x.device).unsqueeze(0).expand(B, -1, -1)
    pfpx[mask] = 0.0
    sum_C = pfpx.sum(dim=2)
    pfpx[mask] = -sum_C.flatten()
    return pfpx

class HNN(nn.Module):
    def __init__(self, state_dim=args.N):
        super().__init__()
        self.state_dim = state_dim
        
        self.shared_net = nn.Sequential(
            layer_init(nn.Linear(state_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
        )
        self.value_part = nn.Sequential(
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0)
        )
        self.lambda_part = nn.Sequential(
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, state_dim), std=1.0)
        )
        self.hamilton_part = nn.Sequential(
            layer_init(nn.Linear(2*state_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        
    def get_value(self, x):
        zero_point = torch.zeros_like(x)
        x_feature = self.shared_net(x)
        zero_point_feature = self.shared_net(zero_point)
        return self.value_part(x_feature) - self.value_part(zero_point_feature)
        
    def get_lambda(self, x):
        zero_point = torch.zeros_like(x)
        x_feature = self.shared_net(x)
        zero_point_feature = self.shared_net(zero_point)
        return self.lambda_part(x_feature) - self.lambda_part(zero_point_feature)

    def get_hamilton(self, x):
        Lambda = self.get_lambda(x)
        hamilton_input = torch.cat([x, Lambda], dim=-1)
        return self.hamilton_part(hamilton_input)

# =================== LQR Controller ======================
def get_lqr_gain():
    A = np.full((args.N, args.N), args.K / args.N)
    np.fill_diagonal(A, -args.K * (args.N - 1) / args.N)
    B = np.eye(args.N)
    P = solve_continuous_are(A, B, Q_np, R_np)
    K_lqr = np.linalg.inv(R_np) @ B.T @ P
    return P, K_lqr

P_ref_np, K_lqr_np = get_lqr_gain()
P_ref_tensor = torch.tensor(P_ref_np, dtype=torch.float32, device=args.device)

# =================== 损失函数 ======================
def get_all_loss(model, X):
    # 先把需要的量都算出来
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

    # pHpLambda
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
    pfpu = get_pfpu(X)  
    R_inv = torch.inverse(R_torch)
    pfpu_T = pfpu.transpose(1, 2)      
    delta_u = -0.5 * torch.matmul(pfpu_T, Lambda_unsq)  
    delta_u = torch.matmul(R_inv, delta_u.squeeze(-1).T).T  
    u_star = delta_u + u0_tensor

    # 再计算损失
    # 1 pVpx = lambda损失
    loss1 = torch.nn.functional.mse_loss(Lambda, pVpX)
    
    # 2 V > 0损失
    loss2 = torch.relu(-V).mean()

    # 3 x_dot = pHpLambda损失
    fxu = dynamics(X, u_star)
    fxu_detach = fxu.detach().clone()
    loss3 = torch.nn.functional.mse_loss(fxu_detach, pHpLambda)

    # 4 lambda_dot = -pHpX损失
    pfpx = get_pfpx(X, u_star) 
    term1 = 2 * torch.matmul(X, Q_torch.T)  
    pfpx_T = pfpx.transpose(1, 2)  
    term2 = torch.matmul(pfpx_T, Lambda_detach_unsq).squeeze(-1)  
    Lambda_dot = -(term1 + term2)  
    loss4 = torch.nn.functional.mse_loss(-pHpX, Lambda_dot)

    # 5 H = 0损失
    xQx = torch.einsum('bi,ij,bj->b', X, Q_torch, X)  # (batch_size,)
    uRu = torch.einsum('bi,ij,bj->b', delta_u, R_torch, delta_u)  # (batch_size,)
    gradV_fxu = (Lambda * fxu).sum(dim=1)
    H_eq = xQx + uRu + gradV_fxu  # 希望H=0
    loss5 = (H_eq ** 2).mean()
    
    # 6 神经网络直接的H=0损失
    loss6 = torch.nn.functional.mse_loss(H_eq, H)

    return loss1 + loss2 + loss3 + loss4 + loss5 + loss6

def lambda_zero_loss(model):
    zero_point = torch.zeros(1, model.state_dim, device=args.device)
    zero_point_Lambda = model.get_lambda(zero_point)
    # 计算零点处Lambda对输入的导数
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

    grad_lambda_loss = torch.nn.functional.mse_loss(grad_Lambda.squeeze(0), 2*P_ref_tensor)
    loss = zero_loss + grad_lambda_loss
    return loss

# =================== 训练函数 ======================
def train_hnn():
    hnn = HNN().to(args.device)
    optimizer = optim.Adam(hnn.parameters(), lr=args.learning_rate)

    start_time = time.time()
    best_loss = float('inf')
    patience = 500
    patience_counter = 0
    min_steps = 1000
    ema_loss = None

    dt = args.T / args.forward_steps
    for step in range(args.time_steps):
        # 采样数据
        low = -args.train_range
        high = args.train_range
        x_batch = np.random.uniform(low, high, (args.batch_size, args.N))
        X = torch.from_numpy(x_batch).float().to(args.device)
        X.requires_grad_(True)
        
        # 计算损失函数
        loss = get_all_loss(hnn, X) + lambda_zero_loss(hnn)

        # 累计代价损失
        x = X.detach().clone()
        for i in range(args.forward_steps):
            pfpu = get_pfpu(x)
            Lambda = hnn.get_lambda(x) 
            Lambda_unsq = Lambda.unsqueeze(2) 
            R_inv = torch.inverse(R_torch)
            pfpu_T = pfpu.transpose(1, 2)      
            delta_u = -0.5 * torch.matmul(pfpu_T, Lambda_unsq)
            delta_u = torch.matmul(R_inv, delta_u.squeeze(-1).T).T  
            u_star = delta_u + u0_tensor
            fxu = dynamics(x, u_star)
            x = x + fxu * dt
            xQx = torch.einsum('bi,ij,bj->b', x, Q_torch, x)
            uRu = torch.einsum('bi,ij,bj->b', u_star-u0_tensor, R_torch, u_star-u0_tensor)
            loss += 10 * (xQx + uRu).mean() * dt
        # 终端代价损失
        xPx =  torch.einsum('bi,ij,bj->b', x, P_ref_tensor, x)
        loss += 10 * xPx.mean()

        # 更新参数
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        current_loss = loss.item()
        if ema_loss is None:
            ema_loss = current_loss
        else:
            ema_loss = 0.95 * ema_loss + 0.05 * current_loss

        # 打印损失
        if step % 100 == 0:
            if ema_loss < best_loss - 0.01:
                best_loss = ema_loss
                patience_counter = 0
            else:
                patience_counter += 100*0
                
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

        if step == (args.time_steps-1):
            model_dir = "model"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, args.model_name)
            torch.save(hnn.state_dict(), model_path)
            end_time = time.time()
            print(f"Training completed. Total training time: {end_time - start_time:.2f} seconds.")
            print(f"HNN model saved to {model_path}")

# =================== 画图与仿真 ======================
def simulate_once():
    # 仿真标称控制 (u0) 和 OPINN 控制效果
    hnn = HNN().to(args.device)
    hnn_path = "model/" + args.model_name
    hnn.load_state_dict(torch.load(hnn_path, map_location=args.device))
    hnn.eval()

    # 初始化状态
    np.random.seed(100)
    x0 = np.random.uniform(-np.pi/2, np.pi/2, args.N)
    
    horizon = int(args.T / args.dt)
    
    # 仿真 Nominal Control (u0)
    x_u0 = np.zeros((horizon, args.N))
    u_u0_hist = np.zeros((horizon, args.N))
    curr_x = x0.copy()
    for t in range(horizon):
        x_u0[t] = curr_x
        # 仅使用稳态前馈控制 u = u0_np
        u = 0
        u_u0_hist[t] = u
        
        # 动力学步进
        sin_x = np.sin(curr_x)
        cos_x = np.cos(curr_x)
        sum_sin = np.sum(sin_x)
        sum_cos = np.sum(cos_x)
        coupling = (args.K / args.N) * (cos_x * sum_sin - sin_x * sum_cos)
        x_dot = OMEGA + coupling + u
        curr_x = curr_x + x_dot * args.dt
        
    # 仿真 OPINN
    x_hnn = np.zeros((horizon, args.N))
    u_hnn_hist = np.zeros((horizon, args.N))
    curr_x = x0.copy()
    
    R_inv = torch.inverse(R_torch)
    
    for t in range(horizon):
        x_hnn[t] = curr_x
        curr_x_tensor = torch.from_numpy(curr_x).float().unsqueeze(0).to(args.device)
        curr_x_tensor.requires_grad_(True)
        
        Lambda = hnn.get_lambda(curr_x_tensor) # (1, N)
        delta_u = -0.5 * torch.matmul(R_inv, Lambda.T).T # (1, N)
        delta_u_np = delta_u.detach().cpu().numpy()[0]
        u = delta_u_np + u0_np
        u_hnn_hist[t] = u
        
        # 动力学步进
        sin_x = np.sin(curr_x)
        cos_x = np.cos(curr_x)
        sum_sin = np.sum(sin_x)
        sum_cos = np.sum(cos_x)
        coupling = (args.K / args.N) * (cos_x * sum_sin - sin_x * sum_cos)
        x_dot = OMEGA + coupling + u
        curr_x = curr_x + x_dot * args.dt
        
    # ================= 画图: 单独画两个状态曲线并保存为SVG =================
    time_steps = np.arange(horizon) * args.dt
    
    # 按照每个振子初始相位分配连续渐变色
    norm = plt.Normalize(vmin=np.min(x0), vmax=np.max(x0))
    cmap = plt.get_cmap('Spectral_r')
    
    import os
    os.makedirs('image', exist_ok=True)
    
    # 1. 绘制 Nominal Control 状态曲线
    fig_u0 = plt.figure(figsize=(8, 6))
    ax_u0 = fig_u0.add_subplot(111)
    for i in range(args.N):
        c = cmap(norm(x0[i]))
        ax_u0.plot(time_steps, x_u0[:, i], color=c, alpha=0.8, linewidth=2.0)
    ax_u0.set_xlabel('Time (s)')
    ax_u0.set_ylabel('Phase $\\theta_i$ (rad)')
    plt.tight_layout()
    plt.savefig('image/kuramoto_phase_u0.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig_u0)
    
    # 2. 绘制 OPINN 状态曲线
    fig_opinn = plt.figure(figsize=(8, 6))
    ax_opinn = fig_opinn.add_subplot(111)
    for i in range(args.N):
        c = cmap(norm(x0[i]))
        ax_opinn.plot(time_steps, x_hnn[:, i], color=c, alpha=0.8, linewidth=2.0)
    ax_opinn.set_xlabel('Time (s)')
    ax_opinn.set_ylabel('Phase $\\theta_i$ (rad)')
    plt.tight_layout()
    plt.savefig('image/kuramoto_phase_opinn.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig_opinn)
    
    print("Saved individual phase trajectory plots to 'image/kuramoto_phase_u0.pdf' and 'image/kuramoto_phase_opinn.pdf'.")
    
    # 统计 Cost
    cost_u0 = np.sum(np.sum(x_u0**2, axis=1) + np.sum((u_u0_hist - u0_np)**2, axis=1)) * args.dt
    cost_hnn = np.sum(np.sum(x_hnn**2, axis=1) + np.sum((u_hnn_hist - u0_np)**2, axis=1)) * args.dt
    print(f"Nominal Control (u0) Total Cost: {cost_u0:.4f}")
    print(f"OPINN Total Cost: {cost_hnn:.4f}")

if __name__ == "__main__":
    # 确保保存路径存在
    import os
    os.makedirs("image", exist_ok=True)
    # train_hnn()
    simulate_once()