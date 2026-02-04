'''
简化所有loss计算为一体; 使用正定值函数网络设计；
'''
import sys, os
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


# ================== 图像设置 =======================
mpl.rcParams['font.family'] = 'Times New Roman'  # 默认字体
# mpl.rcParams['font.weight'] = 'bold'  # 加粗
mpl.rcParams['text.usetex'] = True  # 解决无法显示负号
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
    env_id = "environment:linear2d-v0"
    # 学习率
    learning_rate = 1e-3
    # 总训练次数
    time_steps = 20000
    # 训练批次
    batch_size = 1280
    # 是否使用GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 文件名前缀
    filename_prefix = "hnn_linear2d_"


# 动力学相关
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


# 计算控制矩阵pfpu
def get_pfpu(X):
    # X: (batch_size, n)
    batch_size, n = X.shape

    # 生成 batch_size 个 2x1 的列向量 B，每个都是 [0, 1]
    B = torch.tensor([[0.0], [1.0]], device=X.device, dtype=X.dtype).expand(batch_size, 2, 1)
    
    return B


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
        Lambda = self.get_lambda(X) # (batch_size, 2)
        Lambda = Lambda.unsqueeze(2)  # (batch_size, 2, 1)
        
        pfpu = get_pfpu(X)  # (batch_size, 2, 1)
        
        # 计算 u_star = -0.5 * inv(R) * pfpu.T * grad_V
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


# 计算pVpx = lambda损失
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


# 计算V > 0损失
def positive_v_loss(model, X):
    # 计算pinn对X的输出，要求输出大于0的loss
    V = model.get_value(X).squeeze(-1)  # (batch,)
    # 只惩罚V<=0的部分
    loss = torch.relu(-V).mean()
    return loss


# 计算x_dot = pHpLambda损失
def dynamics_loss(model, X):
    u_star = model.get_action(X)
    fxu = dynamics(X, u_star)
    pHpLambda = model.get_pHpLambda(X)
    loss = torch.nn.functional.mse_loss(fxu, pHpLambda)
    return loss


# 计算lambda_dot = -pHpX损失
def lambda_dot_loss(model, X):
    Q = model.env.Q
    Q_torch = torch.tensor(Q, dtype=X.dtype, device=X.device)

    pfpx = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=X.dtype, device=X.device)

    Lambda = model.get_lambda(X)
    pHpX = model.get_pHpX(X)

    # 计算 pHpX = 2*Q*X + pfpx^T * Lambda
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
    loss = torch.nn.functional.mse_loss(-pHpX, Lambda_dot)  # 这里返回pHpX，实际损失在外部定义
    return loss


def hamiltonian_loss(model, X):
    Q = model.env.Q
    R = model.env.R

    u_star = model.get_action(X)
    grad_V = model.get_lambda(X)

    # 计算 fxu
    fxu = dynamics(X, u_star)

    # 计算 H_loss = x^T Q x + u_star^T R u_star + grad_V^T fxu
    # x: (batch_size, 2), Q: (2,2), u_star: (batch_size, 1), R: (1,1), grad_V: (batch_size, 2), fxu: (batch_size, 2)
    Q_torch = torch.tensor(Q, dtype=X.dtype, device=X.device)
    R_torch = torch.tensor(R, dtype=X.dtype, device=X.device)

    # x^T Q x
    xQx = torch.einsum('bi,ij,bj->b', X, Q_torch, X)  # (batch_size,)

    # u_star^T R u_star
    uRu = torch.einsum('bi,ij,bj->b', u_star, R_torch, u_star)  # (batch_size,)

    # grad_V^T fxu
    gradV_fxu = (grad_V * fxu).sum(dim=1)  # (batch_size,)

    H = xQx + uRu + gradV_fxu  # 希望H=0
    H_loss = (H ** 2).mean()
    return H_loss


def direct_H_loss(model, X):
    H = model.get_hamilton(X)
    H_loss = (H ** 2).mean()
    return H_loss


def get_all_loss(model, X):
    # 先把需要的量都算出来
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
    pfpx = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=X.dtype, device=X.device)
    term1 = 2 * torch.matmul(X, Q_torch.T)  # (batch_size, 2)
    pfpx_T = pfpx.T  # (2,2)
    term2 = torch.matmul(pfpx_T, Lambda_detach_unsq).squeeze(-1)  # (batch_size, 2)
    Lambda_dot = -(term1 + term2)  # (batch_size, 2)
    loss4 = torch.nn.functional.mse_loss(-pHpX, Lambda_dot)

    # 5 H = 0损失
    xQx = torch.einsum('bi,ij,bj->b', X, Q_torch, X)  # (batch_size,)
    u_star_connect = u_star = -0.5 * torch.matmul(pfpu_T, Lambda_unsq)  # (batch_size, 1, 1)
    u_star_connect = torch.matmul(R_inv, u_star_connect.squeeze(-1).T).T  # (batch_size, 1)
    uRu = torch.einsum('bi,ij,bj->b', u_star_connect, R_torch, u_star_connect)  # (batch_size,)
    gradV_fxu = (Lambda * fxu).sum(dim=1)
    H = xQx + uRu + gradV_fxu  # 希望H=0
    loss5 = (H ** 2).mean()

    return loss1 + loss2 + loss3 + loss4 + loss5


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

    for step in range(args.time_steps):
        # 采样数据
        low = -4.0
        high = 4.0
        x_batch = np.random.uniform(low, high, (args.batch_size, env.observation_space.shape[0]))
        X = torch.from_numpy(x_batch).float().to(args.device)
        X.requires_grad_(True)
        # 计算损失函数
        # posi_v_loss = positive_v_loss(hnn, X)
        # v_lam_loss = value_lambda_loss(hnn, X)
        # # lam_zero_loss = lambda_zero_loss(pinn)
        # dyn_loss = dynamics_loss(hnn, X)
        # lam_dot_loss = lambda_dot_loss(hnn, X)
        # H_loss = direct_H_loss(hnn, X)
        # loss = posi_v_loss + v_lam_loss + dyn_loss + lam_dot_loss + H_loss
        loss = get_all_loss(hnn, X) + lambda_zero_loss(hnn)

        # 更新参数
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 打印损失
        if step % 100 == 0:
            print(f"Step {step+1}/{args.time_steps}, Loss: {loss.item():.6f}")

        # 保存pinn模型到model文件夹
        if step > 1000 and (step % 5000 == 0 or step == (args.time_steps-1)):
            model_dir = "model"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "hnn_linear2d.pth")
            torch.save(hnn.state_dict(), model_path)
            print(f"HNN model saved to {model_path}")


def plot_value():
    # 加载模型参数
    model_dir = "model"
    model_path = os.path.join(model_dir, "hnn_linear2d.pth")
    args = Args()
    env = gym.make(args.env_id)
    hnn = HNN(env)
    hnn.load_state_dict(torch.load(model_path, map_location="cpu"))
    hnn.eval()

    # 构建网格
    x1 = np.linspace(-3, 3, 50)
    x2 = np.linspace(-3, 3, 50)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.stack([X1.ravel(), X2.ravel()], axis=1)

    # HNN值函数
    X_tensor = torch.from_numpy(X_grid).float()
    with torch.no_grad():
        hnn_values = hnn.get_value(X_tensor).cpu().numpy()
    hnn_values = hnn_values.reshape(X1.shape)

    # 计算HNN值函数的V_dot = pVpx * f(x, u*), 其中u*用hnn-Lambda闭环控制
    X_tensor.requires_grad_(True)
    # 计算Lambda = dV/dx for closed-loop control
    Lambda = hnn.get_lambda(X_tensor)  # (N, 2)
    # 线性系统: B= [[0],[1]], R=1
    B = torch.tensor([[0.0], [1.0]], dtype=X_tensor.dtype, device=X_tensor.device)
    R = torch.tensor([[1.0]], dtype=X_tensor.dtype, device=X_tensor.device)
    # 最优闭环控制 u* = -R^{-1} B^T Lambda^T
    u_star = -torch.matmul(Lambda, B)  # (N, 1)
    # 动力学
    fxu = dynamics(X_tensor, u_star)
    # 计算V(x)对x的梯度
    V = hnn.get_value(X_tensor)
    V_sum = V.sum()
    pVpx = torch.autograd.grad(V_sum, X_tensor, create_graph=False)[0]  # (N, 2)
    # V_dot (点乘)：每个点pVpx * fxu 求和
    V_dot_hnn = (pVpx * fxu).sum(dim=1).reshape(X1.shape).detach().numpy()

    # LQR值函数: V(x) = x^T P x, 其中P=[[sqrt(3), 1],[1, sqrt(3)]]
    P = np.array([[np.sqrt(3), 1.0], [1.0, np.sqrt(3)]])
    lqr_values = np.einsum('ij,jk,ik->i', X_grid, P, X_grid).reshape(X1.shape)
    # 计算LQR值函数的V_dot = (2*P*x)^T * f(x, u)
    # u_star_lqr = -R^{-1} B^T (2Px)
    X_grid_torch = torch.from_numpy(X_grid).float()
    P_torch = torch.from_numpy(P).float()
    B_lqr = torch.tensor([[0.0], [1.0]])
    R_inv_lqr = torch.tensor([[1.0]])
    gradV_lqr = 2 * torch.matmul(X_grid_torch, P_torch)  # (N,2)
    # 计算lqr闭环控制
    u_star_lqr = -torch.matmul(gradV_lqr, B_lqr)  # (N,1)
    fxu_lqr = dynamics(X_grid_torch, u_star_lqr)
    V_dot_lqr = (gradV_lqr * fxu_lqr).sum(dim=1).reshape(X1.shape).detach().numpy()

    # 绘制对比曲面在同一张图上
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1.4])
    ax = fig.add_subplot(gs[0], projection='3d')

    # 使用单色渐变: HNN用Blues, LQR用Oranges
    surf1 = ax.plot_surface(
        X1, X2, hnn_values, cmap='Blues', alpha=0.85, linewidth=0, antialiased=True, rstride=1, cstride=1
    )
    surf1_1 = ax.plot_surface(
        X1, X2, V_dot_hnn, cmap='Blues', alpha=0.85, linewidth=0, antialiased=True, rstride=1, cstride=1
    )
    surf2 = ax.plot_surface(
        X1, X2, lqr_values, cmap='Oranges', alpha=0.55, linewidth=0, antialiased=True, rstride=1, cstride=1
    )
    surf2_1 = ax.plot_surface(
        X1, X2, V_dot_lqr, cmap='Oranges', alpha=0.55, linewidth=0, antialiased=True, rstride=1, cstride=1
    )

    # 手动添加图例，颜色与曲面一致
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=plt.cm.Blues(0.7), lw=2, label='OCNet'),
        Line2D([0], [0], color=plt.cm.Oranges(0.7), lw=2, label='Optimal')
    ]
    ax.legend(handles=legend_elements, ncol=2, loc='upper right')

    # 坐标轴和标题
    ax.set_xlabel(r'$x_1$', labelpad=10)
    ax.set_ylabel(r'$x_2$', labelpad=10)
    ax.set_zlabel(r'$V$ or $\dot V$', labelpad=10)
    ax.view_init(elev=5, azim=30, roll=0)
    # # 美化
    # plt.tight_layout()


    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim([-3, 3])
    ax2.set_ylim([-3, 3])
    # 以HNN闭环控制-矢量场短箭头和彩色箭头作相图

    # 定义网格
    x1_lin = np.linspace(-3, 3, 25)
    x2_lin = np.linspace(-3, 3, 25)
    X1_grid, X2_grid = np.meshgrid(x1_lin, x2_lin)
    U = np.zeros_like(X1_grid)
    V = np.zeros_like(X2_grid)

    # 计算向量场
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

    # 归一化箭头方向用于着色
    speed = np.sqrt(U**2 + V**2)
    color = speed
    # "inferno", "plasma", "viridis"，可自选
    # 稀疏化箭头（每隔2个点画一个箭头），并调整箭头大小
    step = 2  # 每隔2个点取一次，提升稀疏度
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

    # # 美化
    # plt.tight_layout()

    filename = "image/" + args.filename_prefix + "value_function.svg"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)


    # 画V_dot的等高线图（投影到最低面z=min）
    fig3 = plt.figure(figsize=(6,5))
    ax3 = fig3.add_subplot(111, projection='3d')
    # 找到等高线投影的最低z面
    z_min = min(np.min(hnn_values), np.min(hnn_values))
    # 主表面
    ax3.plot_surface(
        X1, X2, hnn_values, cmap='RdBu_r', alpha=0.8
    )
    # 在z=z_min做等高线投影
    surf3_1 = ax3.contour(
        X1, X2, hnn_values, zdir='z', offset=z_min, cmap='RdBu_r', alpha=1, levels=20
    )
    ax3.set_xlabel(r'$x_1$', labelpad=10)
    ax3.set_ylabel(r'$x_2$', labelpad=10)
    ax3.set_zlabel(r'$V$', labelpad=10, rotation=90)
    # ax3.view_init(elev=5, azim=30, roll=0)
    filename = "image/" + args.filename_prefix + "hnn_value.svg"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)

    # 画V的等高线图（投影到最低面z=min）
    fig4 = plt.figure(figsize=(6,5))
    ax4 = fig4.add_subplot(111, projection='3d')
    # 找到等高线投影的最低z面
    z_min = min(np.min(V_dot_hnn), np.min(V_dot_hnn))
    # 主表面
    ax4.plot_surface(
        X1, X2, V_dot_hnn, cmap='RdBu_r', alpha=0.8
    )
    # 在z=z_min做等高线投影
    surf4_1 = ax4.contour(
        X1, X2, V_dot_hnn, zdir='z', offset=z_min, cmap='RdBu_r', alpha=1, levels=20
    )
    ax4.set_xlabel(r'$x_1$', labelpad=10)
    ax4.set_ylabel(r'$x_2$', labelpad=10)
    ax4.set_zlabel(r'$\dot V$', labelpad=20)
    # ax4.view_init(elev=5, azim=30, roll=0)
    filename = "image/" + args.filename_prefix + "hnn_value_dot.svg"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)

    # 画相轨迹图
    fig5 = plt.figure(figsize=(6,5))
    ax5 = fig5.add_subplot(111)
    # 在z=z_min做等高线投影
    step = 2  # 每隔2个点取一次，提升稀疏度
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
    加载训练好的HNN网络和环境，进行一次仿真，并画出状态和控制的时间曲线
    """
    # 加载环境
    env = gym.make(Args.env_id)
    obs, _ = env.reset(seed=1318)
    print(obs)
    obs = np.array(obs, dtype=np.float32)
    
    # 加载网络
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

    # 仿真参数
    T = 200
    state_traj = [obs.copy()]
    control_traj = []
    time_traj = [0.0]
    for t in range(T):
        x_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(Args.device)
        # 直接用get_lambda，不用对V求导
        with torch.no_grad():
            Lambda = hnn.get_lambda(x_tensor).cpu().numpy().squeeze()
        # 线性系统参数
        B = np.array([[0], [1]])
        R = np.diag([1])
        # 最优控制律 u* = -R^{-1} B^T Lambda
        u = -np.linalg.inv(R) @ B.T @ Lambda.reshape(-1, 1)
        u = u.squeeze()
        # 状态更新通过环境
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

    # 绘制状态和控制的时间曲线
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
    simulate_once()