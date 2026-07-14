'''
简化所有loss计算为一体; 使用正定值函数网络设计；二维无人机
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

# ================== 图像设置 =======================
mpl.rcParams['font.family'] = 'Times New Roman'  # 默认字体
# mpl.rcParams['font.weight'] = 'bold'  # 加粗
mpl.rcParams['text.usetex'] = False  # 使用TEX
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
    # 环境名称
    env_id = "environment:quadrotor3d_TR-v0"
    # 模型名称
    model_name = "hnn_quadrotor3d_TR_v2_paperplot.pth"
    # 训练范围
    train_range = 1.0
    # 学习率
    learning_rate = 1e-3
    # 总训练次数
    time_steps = 30000
    # 训练批次
    batch_size = 1280
    # 时间范围
    T = 1
    # 轨迹前向长度
    forward_steps = 5
    # 是否使用GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 文件名前缀
    filename_prefix = "hnn_quadrotor3d_"


def dynamics(X, U):
    # X: (N, 9) -> [x, y, z, vx, vy, vz, phi, theta, psi]
    # U: (N, 4) -> [T, p, q, r]
    # 参考quadrotor3d_TR.py的动力学
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

_env_tmp = gym.make(Args.env_id)
m = _env_tmp.metadata['m']
g = _env_tmp.metadata['g']

# 计算控制矩阵pfpu
def get_pfpu(X):
    """
    计算三维无人机动力学对u的雅可比矩阵（每个样本一个9x4矩阵）
    X: (N, 12) -> [x, y, z, vx, vy, vz, phi, theta, psi]
    返回: (N, 12, 4)
    """
    # 物理参数
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

# 计算pfpx
def get_pfpx(X, U):
    """
    计算三维无人机动力学对X的雅可比矩阵（每个样本一个12x12矩阵）
    X: (N, 12)
    U: (N, 4)
    返回: (N, 12, 12)
    """
    # 这里只实现主要的非零项，详细推导可参考quadrotor3d_TM.py
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


# Hamilton神经网络
class HNN(nn.Module):
    def __init__(self, env):
        self.env = env
        super().__init__()

        # 共享层
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
        Lambda = self.get_lambda(x).detach()  # 切断Lambda对x的梯度连接
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
        
        # 计算 u_star = -0.5 * inv(R) * pfpu.T * grad_V
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
    Q = model.env.Q
    Q_torch = torch.tensor(Q, dtype=X.dtype, device=X.device)
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
    # loss6 = torch.nn.functional.mse_loss(H_eq, H)

    return loss1 + loss2 + loss3 + loss4 + loss5 #+ loss6


def lambda_zero_loss(model):
    zero_point = torch.zeros(1, model.env.observation_space.shape[0], device=next(model.parameters()).device)
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

    P_ref_tensor = torch.tensor(P_ref, dtype=grad_Lambda.dtype, device=grad_Lambda.device)
    grad_lambda_loss = torch.nn.functional.mse_loss(grad_Lambda.squeeze(0), 2*P_ref_tensor)
    # loss = zero_loss + grad_lambda_loss
    loss = grad_lambda_loss
    return loss


def train_hnn():
    env = gym.make(args.env_id)
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

        # 采样数据
        low = -args.train_range
        high = args.train_range
        x_batch = np.random.uniform(low, high, (args.batch_size, env.observation_space.shape[0]))
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
        # 终端代价损失
        xPx =  torch.einsum('bi,ij,bj->b', x, P_torch, x)
        loss += xPx.mean()

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

        # 保存pinn模型到model文件夹
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
    # 加载模型参数
    args = Args()
    env = gym.make(args.env_id)
    hnn = HNN(env)
    model_dir = "model"
    model_path = os.path.join(model_dir, args.model_name)
    hnn.load_state_dict(torch.load(model_path, map_location="cpu"))
    hnn.eval()

    # 构建9维状态的切片网格，x1,x2在-train_range到train_range之间，其他维度为0
    x1 = np.linspace(-args.train_range, args.train_range, 50)
    x2 = np.linspace(-args.train_range, args.train_range, 50)
    X1, X2 = np.meshgrid(x1, x2)
    X1_flat = X1.ravel()
    X2_flat = X2.ravel()
    # 其余7维为0
    X_rest = [np.zeros_like(X1_flat) for _ in range(7)]
    X_grid = np.stack([X1_flat, X2_flat] + X_rest, axis=1)  # shape (N, 9)

    # HNN值函数
    X_tensor = torch.from_numpy(X_grid).float()
    with torch.no_grad():
        hnn_values = hnn.get_value(X_tensor).cpu().numpy()
    hnn_values = hnn_values.reshape(X1.shape)

    # LQR值函数: V(x) = x^T P x, 其中P为环境的LQR解
    P = env.P  # 直接从环境获取LQR解
    lqr_values = np.einsum('ij,jk,ik->i', X_grid, P, X_grid).reshape(X1.shape)

    # 绘制对比曲面
    fig = plt.figure(figsize=(16,6))

    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    surf1 = ax1.plot_surface(X1, X2, hnn_values, cmap='viridis', alpha=1)
    surf2 = ax1.plot_surface(X1, X2, lqr_values, cmap='plasma', alpha=0.5)
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('Value')
    ax1.set_title('Value Surface (slice $x_3$-$x_{9}=0$)')

    # 添加colorbar并分别标注
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
    # 加载hnnv2_quadrotor3d_TM模型参数
    args = Args()
    env = gym.make(args.env_id)
    hnn = HNN(env)
    model_dir = "model"
    model_path = os.path.join(model_dir, args.model_name)
    hnn.load_state_dict(torch.load(model_path, map_location="cpu"))
    hnn.eval()

    # LQR P矩阵（直接从环境获取）
    P = env.P

    # 构建9维状态的切片网格，x1,x2在-train_range到train_range之间，其余为0
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
    # 每个维度reshape成(X1.shape)
    hnn_lambda_list = [hnn_lambda[:, i].reshape(X1.shape) for i in range(9)]

    # LQR Lambda: Lambda = 2 * P * X
    lqr_lambda = 2 * (X_grid @ P)  # shape (N, 9)
    lqr_lambda_list = [lqr_lambda[:, i].reshape(X1.shape) for i in range(9)]

    lambda_names = [rf'$\lambda_{i+1}$' for i in range(9)]

    # ----------- 9个子图，每个子图画HNN和LQR对应维度的lambda曲面 -----------
    fig = plt.figure(figsize=(24, 12))
    for i in range(9):
        ax = fig.add_subplot(3, 3, i+1, projection='3d')
        surf1 = ax.plot_surface(X1, X2, hnn_lambda_list[i], cmap='Blues', alpha=0.8, edgecolor='none')
        surf2 = ax.plot_surface(X1, X2, lqr_lambda_list[i], cmap='Oranges', alpha=0.5, edgecolor='none')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel(lambda_names[i])
        ax.set_title(f"{lambda_names[i]} (slice $x_3$-$x_{{9}}=0$)")
        # colorbar只加HNN的
        mappable1 = plt.cm.ScalarMappable(cmap='Blues')
        mappable1.set_array(hnn_lambda_list[i])
        cbar1 = fig.colorbar(mappable1, ax=ax, shrink=0.6, pad=0.05)
        cbar1.set_label('HNN', fontsize=10)
        mappable2 = plt.cm.ScalarMappable(cmap='Oranges')
        mappable2.set_array(lqr_lambda_list[i])
        cbar2 = fig.colorbar(mappable2, ax=ax, shrink=0.6, pad=0.01)
        cbar2.set_label('LQR', fontsize=10)
        # 图例
        custom_lines = [
            plt.Line2D([0], [0], color='tab:blue', lw=4, label='HNN'),
            plt.Line2D([0], [0], color='tab:orange', lw=4, label='LQR'),
        ]
        ax.legend(handles=custom_lines, loc='upper left')
    plt.tight_layout()
    plt.show()


def simulate_once():
    # 仿真LQR和HNN控制效果
    args = Args()
    device = args.device
    env_probe = gym.make(args.env_id, theoretic_mode=True)

    # 仿真参数
    horizon = int(env_probe.T / env_probe.dt)
    A_np = env_probe.A
    B_np = env_probe.B
    Q_np = env_probe.Q
    R_np = env_probe.R
    m = env_probe.metadata['m']
    g = env_probe.metadata['g']
    u0 = env_probe.u0

    # 计算LQR控制器
    P = solve_continuous_are(A_np, B_np, Q_np, R_np)
    K = np.linalg.inv(R_np) @ B_np.T @ P
    def lqr_action(x_np):
        u = -K @ x_np + u0
        u = np.clip(u, env_probe.action_space.low, env_probe.action_space.high)
        return np.array(u, dtype=np.float32)

    # 加载HNN模型
    hnn_path = "model/" + args.model_name
    env_hnn = gym.make(args.env_id, theoretic_mode=True)
    hnn = HNN(env_hnn).to(device)
    hnn.load_state_dict(torch.load(hnn_path, map_location=device))
    hnn.eval()

    # HNN控制器
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
        u = u + u0  # 加重力补偿
        u = np.clip(u, env_probe.action_space.low, env_probe.action_space.high)
        return np.array(u, dtype=np.float32)

    # 初始化环境
    seed = np.random.randint(0, 10000000)
    env_lqr = gym.make(args.env_id, theoretic_mode=True)
    obs_lqr, _ = env_lqr.reset(seed=seed)
    env_hnn = gym.make(args.env_id, theoretic_mode=True)
    obs_hnn, _ = env_hnn.reset(seed=seed)  # 用同一个初始条件

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

    # 绘制控制输入轨迹
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

    # 绘制状态轨迹
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

    # 打印回报
    print("Single episode return:")
    print(f"LQR: {ret_lqr:.3f}")
    print(f"HNN: {ret_hnn:.3f}")

    # drone_visualizer = DroneVisualizer(traj=traj_x_hnn[:hnn_trunc_count, 0:3], attitudes=np.rad2deg(traj_x_hnn[:hnn_trunc_count, 6:9]), alpha_range=(0.25, 1.0), stride=60, pause=0.04)
    # drone_visualizer.show()

    # 绘制无人机三维轨迹
    import cmaps
    cmap_hnn = cmaps.MPL_PuOr_r
    cmap_lqr = cmaps.MPL_RdYlGn_r
    drone_visualizer = DroneVisualizer(traj=traj_x_hnn[:, 0:3], alpha_range=[0.2, 1])
    drone_visualizer.add_one_traj(traj=traj_x_hnn[:hnn_trunc_count, 0:3], attitudes=np.rad2deg(traj_x_hnn[:hnn_trunc_count, 6:9]), stride=50, colormap=cmap_hnn) 
    drone_visualizer.add_one_traj(traj=traj_x_lqr[:lqr_trunc_count, 0:3], attitudes=np.rad2deg(traj_x_lqr[:lqr_trunc_count, 6:9]), stride=50, colormap=cmap_lqr)
    drone_visualizer.ax.set_zlim([-1, 1])
    plt.savefig('image/quad_traj_single.svg', dpi=600, bbox_inches='tight', pad_inches=0.5, transparent=True)

    # 控制量轨迹（使用nature配色和风格，所有episode透明叠加线）
    control_labels = ['$T$\,(N)', '$p$\,(rad/s)', '$q$\,(rad/s)', '$r$\,(rad/s)']
    # Nature绿色和紫色（LQR和HNN），如上文定义
    nature_green = "#389826"   # LQR
    nature_purple = "#9558B2"  # HNN
    fig, axs = plt.subplots(4, 1, figsize=(6, 6.3), sharex=True)
    for i, ax in enumerate(axs):
        # 画LQR（绿色），单条
        ax.plot(time[:], traj_u_lqr[:, i], color=nature_green, alpha=0.6, linewidth=3, label="LQR" if i==0 else "")
        # 画HNN（紫色），单条
        ax.plot(time[:], traj_u_hnn[:, i], color=nature_purple, alpha=0.8, linewidth=3, label="HNN" if i==0 else "")
        ax.set_ylabel(control_labels[i])
        ax.grid(True, linestyle='--', alpha=0.36)
    axs[-1].set_xlabel("$t$ (s)")
    plt.tight_layout()
    filename = "image/" + args.filename_prefix + "control_traj_single.svg"
    # 保存为透明背景的svg
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5, transparent=True)

    plt.show()


def monte_carlo_simulation():
    # seed = 200
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    num_episodes = 50

    env_lqr = gym.make(args.env_id, theoretic_mode=True)
    env_hnn = gym.make(args.env_id, theoretic_mode=True)
    horizon = int(env_lqr.T / env_lqr.dt)

    m = env_lqr.metadata['m']
    g = env_lqr.metadata['g']
    Q = env_lqr.metadata['Q']
    R = env_lqr.metadata['R']
    u0 = env_lqr.u0

    # LQR控制器
    A = env_lqr.A
    B = env_lqr.B
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P

    # 加载HNN
    device = "cpu"
    hnn = HNN(env_hnn).to(device)
    hnn.load_state_dict(torch.load("model/"+args.model_name, map_location=device))
    hnn.eval()

    # HNN控制器
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

        # 统一seed
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

    # 保存所有轨迹
    np.savez(
        "image/" + args.filename_prefix + "all_traj_mc.npz",
        all_traj_x_lqr=np.array(all_traj_x_lqr, dtype=object),
        all_traj_u_lqr=np.array(all_traj_u_lqr, dtype=object),
        all_traj_x_hnn=np.array(all_traj_x_hnn, dtype=object),
        all_traj_u_hnn=np.array(all_traj_u_hnn, dtype=object),
        returns_lqr=np.array(returns_lqr),
        returns_hnn=np.array(returns_hnn),
    )

    # 加载所有轨迹
    data = np.load("image/" + args.filename_prefix + "all_traj_mc.npz", allow_pickle=True)
    all_traj_x_lqr = data['all_traj_x_lqr']
    all_traj_u_lqr = data['all_traj_u_lqr']
    all_traj_x_hnn = data['all_traj_x_hnn']
    all_traj_u_hnn = data['all_traj_u_hnn']
    returns_lqr = data['returns_lqr']
    returns_hnn = data['returns_hnn']

    # 绘制所有轨迹
    time_x = np.arange(horizon + 1)
    time_u = np.arange(horizon)

    # 状态轨迹
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

    # 控制量轨迹（使用nature配色和风格，所有episode透明叠加线）
    control_labels = ['$T$\,(N)', '$p$\,(rad/s)', '$q$\,(rad/s)', '$r$\,(rad/s)']
    # Nature绿色和紫色（LQR和HNN），如上文定义
    nature_green = "#389826"   # LQR
    nature_purple = "#9558B2"  # HNN
    fig, axs = plt.subplots(4, 1, figsize=(6, 6.3), sharex=True)
    for i, ax in enumerate(axs):
        # 画LQR（绿色），50条
        for ep in range(num_episodes):
            ax.plot(time_u*env.dt, all_traj_u_lqr[ep][:, i], color=nature_green, alpha=0.22, linewidth=1)
        # 画HNN（紫色），50条
        for ep in range(num_episodes):
            ax.plot(time_u*env.dt, all_traj_u_hnn[ep][:, i], color=nature_purple, alpha=0.22, linewidth=1)
        # 平均轨迹线条加粗展示
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

    # 绘制LQR和HNN的return的箱体+小提琴+散点图，主色调使用Nature自然杂志配色
    # LQR使用绿色，HNN使用紫色。依然选用nature配色体系（绿色和紫色的组合）。
    final_dist_lqr = np.array([np.linalg.norm(traj[-1, 0:3]) for traj in all_traj_x_lqr])
    final_dist_hnn = np.array([np.linalg.norm(traj[-1, 0:3]) for traj in all_traj_x_hnn])
    # 认为 距离 > 0.1 为失败
    success_mask_lqr = final_dist_lqr <= 0.1
    success_mask_hnn = final_dist_hnn <= 0.1
    filtered_returns_lqr = returns_lqr[success_mask_lqr]
    filtered_returns_hnn = returns_hnn[success_mask_hnn]
    success_rate_lqr = np.sum(success_mask_lqr) / len(success_mask_lqr)
    success_rate_hnn = np.sum(success_mask_hnn) / len(success_mask_hnn)
    import seaborn as sns
    import pandas as pd

    # ==== 数据准备 ====
    df_box = pd.DataFrame({
        "Return": np.concatenate((-filtered_returns_lqr, -filtered_returns_hnn)),
        "Controller": ["LQR"]*len(filtered_returns_lqr) + ["HNN"]*len(filtered_returns_hnn)
    })

    # lqr_label = r"LQR:%.1f$\%%$" % (success_rate_lqr*100)
    # hnn_label = r"HNN:%.1f$\%%$" % (success_rate_hnn*100)
    lqr_label = r"LQR"
    hnn_label = r"HNN"
    show_labels = [lqr_label, hnn_label]

    # Nature推荐绿色 #389826，紫色 #9558B2 （来源见nature artwork guide），分别给LQR和HNN
    nature_green = "#389826"   # LQR
    nature_purple = "#9558B2"  # HNN
    palette = [nature_green, nature_purple]

    # ---- Nature风格: Seaborn小提琴+箱线+散点 ----
    fig, ax = plt.subplots(figsize=(5, 7))
    # 小提琴底色
    sns.violinplot(
        x="Controller", y="Return", data=df_box,
        order=["LQR", "HNN"],
        palette=palette, inner=None, alpha=0.22, cut=0,
        linewidth=1.2, ax=ax
    )
    # 箱线（叠加）
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
    # 散点（透明叠加）
    # 用更低对比度灰色点
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
   
    # 打印回报统计
    print("Monte Carlo 50 episodes return statistics:")
    print(f"LQR: mean={np.mean(returns_lqr):.3f}, std={np.std(returns_lqr):.3f}, min={np.min(returns_lqr):.3f}, max={np.max(returns_lqr):.3f}")
    print(f"HNN: mean={np.mean(returns_hnn):.3f}, std={np.std(returns_hnn):.3f}, min={np.min(returns_hnn):.3f}, max={np.max(returns_hnn):.3f}")

    plt.show()



if __name__ == "__main__":
    args = Args()
    env = gym.make(args.env_id)
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