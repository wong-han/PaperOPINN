'''
状态约束积分系统
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
    filename_prefix = "chnn_linear2d_"


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


# 计算罚函数
def get_penalty(X):
    # X: (batch_size, 2)
    x2 = X[:, 1]
    term1 = -torch.log(0.5 - x2) + np.log(0.5)
    term2 = -torch.log(0.5 + x2) + np.log(0.5)
    p = term1 + term2  # (batch_size,)
    return p


# 计算pPpx
def get_pPpx(X):
    X2 = X[:, 1]  # (batch_size,)
    pP1px = torch.zeros(X.shape[0], 2, dtype=X.dtype, device=X.device)  # (batch_size, 2)
    pP1px[:, 1] = 1.0 / (0.5 - X2)
    pP2px = torch.zeros(X.shape[0], 2, dtype=X.dtype, device=X.device)  # (batch_size, 2)
    pP2px[:, 1] = -1.0 / (0.5 + X2)
    return pP1px + pP2px


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
    pPpx = get_pPpx(X)
    Lambda_dot = -(term1 + term2 + pPpx)  # (batch_size, 2)
    loss4 = torch.nn.functional.mse_loss(-pHpX, Lambda_dot)

    # 5 H = 0损失
    xQx = torch.einsum('bi,ij,bj->b', X, Q_torch, X)  # (batch_size,)
    u_star_connect = u_star = -0.5 * torch.matmul(pfpu_T, Lambda_unsq)  # (batch_size, 1, 1)
    u_star_connect = torch.matmul(R_inv, u_star_connect.squeeze(-1).T).T  # (batch_size, 1)
    uRu = torch.einsum('bi,ij,bj->b', u_star_connect, R_torch, u_star_connect)  # (batch_size,)
    gradV_fxu = (Lambda * fxu).sum(dim=1)
    penalty = get_penalty(X)
    H = xQx + uRu + gradV_fxu + penalty  # 希望H=0
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
        low = [-1.0, -0.49]
        high = [1.0, 0.49]
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
            model_path = os.path.join(model_dir, args.filename_prefix + "v1.pth")
            torch.save(hnn.state_dict(), model_path)
            print(f"HNN model saved to {model_path}")


def plot_value():
    args = Args()
    # 加载模型参数
    model_dir = "model"
    model_path = os.path.join(model_dir,  args.filename_prefix + "v1.pth")
    env = gym.make(args.env_id)
    hnn = HNN(env)
    hnn.load_state_dict(torch.load(model_path, map_location="cpu"))
    hnn.eval()

    # 构建网格
    x1 = np.linspace(-1, 1, 50)
    x2 = np.linspace(-1, 1, 50)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.stack([X1.ravel(), X2.ravel()], axis=1)
 
    # HNN值函数
    X_tensor = torch.from_numpy(X_grid).float()
    with torch.no_grad():
        hnn_values = hnn.get_value(X_tensor).cpu().numpy()
    hnn_values = hnn_values.reshape(X1.shape)

    # ========== 画 V(x) ===============
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    # 使用单色渐变: HNN用Blues
    surf1 = ax.plot_surface(
        X1, X2, hnn_values, cmap='RdBu_r', alpha=0.85, linewidth=0, antialiased=True, rstride=1, cstride=1
    )
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=plt.cm.Blues(0.7), lw=2, label='HNN')
    ]
    ax.legend(handles=legend_elements)
    ax.set_xlabel(r'dim1', labelpad=10)
    ax.set_ylabel(r'dim2', labelpad=10)
    ax.set_zlabel(r'V(x)', labelpad=10, fontstyle='italic')
    ax.view_init(elev=28, azim=-50, roll=0)
    plt.tight_layout()
    filename = "image/" + args.filename_prefix + "value_function.png"
    # plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)

    # ============== 画 V_dot ==============
    # 参见 hnn_linear2d_v3.py 的计算方法
    # 计算HNN值函数的V_dot = pVpx * f(x, u*), 其中u*用hnn-Lambda闭环控制
    X_tensor_vdot = torch.from_numpy(X_grid).float()
    X_tensor_vdot.requires_grad_(True)
    # Lambda = dV/dx (N,2)
    Lambda = hnn.get_lambda(X_tensor_vdot)  # (N,2)
    # 线性系统: B= [[0],[1]], R=1
    B = torch.tensor([[0.0],[1.0]], dtype=X_tensor_vdot.dtype, device=X_tensor_vdot.device)
    R = torch.tensor([[1.0]], dtype=X_tensor_vdot.dtype, device=X_tensor_vdot.device)
    # 最优闭环控制 u* = -R^{-1} B^T Lambda^T
    u_star = -torch.matmul(Lambda, B)  # (N,1)
    # 动力学 (需定义 dynamics)
    fxu = dynamics(X_tensor_vdot, u_star)
    # 对x求V的梯度
    V = hnn.get_value(X_tensor_vdot)
    V_sum = V.sum()
    pVpx = torch.autograd.grad(V_sum, X_tensor_vdot, create_graph=False)[0]  # (N,2)
    # V_dot: 每个点pVpx * fxu 求和
    V_dot_hnn = (pVpx * fxu).sum(dim=1).reshape(X1.shape).detach().cpu().numpy()

    # 画V_dot的等高线图
    fig2, ax2 = plt.subplots(figsize=(6,5))
    contour_vdot = ax2.contourf(X1, X2, V_dot_hnn, levels=30, cmap='RdBu_r', alpha=0.9)
    cbar = plt.colorbar(contour_vdot, ax=ax2)
    cbar.set_label(r'$\dot{V}$', rotation=0, labelpad=20)
    ax2.set_xlabel(r'$x_1$')
    ax2.set_ylabel(r'$x_2$')
    ax2.set_aspect('equal')
    ax2.axhline(0.5, color='k', linestyle='--', linewidth=2)
    ax2.axhline(-0.5, color='k', linestyle='--', linewidth=2)
    filename = "image/" + args.filename_prefix + "vdot_contourf.svg"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)

    # 绘制HNN值函数的等高线（contourf）图
    fig3, ax3 = plt.subplots(figsize=(6,5))
    contour_hnn = ax3.contourf(X1, X2, hnn_values, levels=30, cmap='RdBu_r', vmin=0, vmax=6, alpha=0.9)
    cbar = plt.colorbar(contour_hnn, ax=ax3)
    cbar.set_label(r'$V$', rotation=0, labelpad=20)
    ax3.set_xlabel(r'$x_1$')
    ax3.set_ylabel(r'$x_2$')
    ax3.set_aspect('equal')
    # 在x2=±0.5处画两条虚线
    ax3.axhline(0.5, color='k', linestyle='--', linewidth=2)
    ax3.axhline(-0.5, color='k', linestyle='--', linewidth=2)
    filename = "image/" + args.filename_prefix + "value_function_contourf.svg"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)

    # 绘制相轨迹图
    # ----------- 相轨迹图（Phase Portrait）绘制 ---------------
    # 构建稀疏网格用于画矢量场
    x1_lin = np.linspace(-1, 1, 25)
    x2_lin = np.linspace(-1, 1, 25)
    X1_grid, X2_grid = np.meshgrid(x1_lin, x2_lin)
    U = np.zeros_like(X1_grid)
    V = np.zeros_like(X2_grid)

    # 计算每个网格点的闭环动力学（dx1/dt, dx2/dt），u* 由 CHNN Lambda 得到
    for i in range(X1_grid.shape[0]):
        for j in range(X1_grid.shape[1]):
            x_point = np.array([X1_grid[i, j], X2_grid[i, j]], dtype=np.float32)
            x_tensor = torch.from_numpy(x_point).float().unsqueeze(0)
            with torch.no_grad():
                Lambda = hnn.get_lambda(x_tensor).cpu().numpy().squeeze()
            B = np.array([[0.0], [1.0]])
            R = np.array([[1.0]])
            u = -np.linalg.inv(R) @ B.T @ Lambda.reshape(-1, 1)
            u = u.squeeze()
            # 系统动力学: dx/dt = Ax + Bu
            U[i, j] = x_point[1]
            V[i, j] = u

    # 归一化用于着色
    speed = np.sqrt(U**2 + V**2)
    # 绘制矢量场
    fig4, ax4 = plt.subplots(figsize=(6,5))
    step = 2  # 每隔2个点画一次箭头
    Q = ax4.quiver(
        X1_grid[::step, ::step], X2_grid[::step, ::step],
        U[::step, ::step], V[::step, ::step],
        speed[::step, ::step], cmap="RdBu_r", scale=90, width=0.009
    )
    cb = plt.colorbar(Q, ax=ax4, fraction=0.045, pad=0.04)
    cb.set_label(r"Magnitude")
    ax4.set_xlabel(r"$x_1$")
    ax4.set_ylabel(r"$x_2$")
    ax4.set_aspect('equal')
    ax4.axhline(0.5, color='k', linestyle='--', linewidth=2)
    ax4.axhline(-0.5, color='k', linestyle='--', linewidth=2)

    # 可以选画几条初始轨迹（仿真流线），如下为可选，注释掉即可
    from scipy.integrate import solve_ivp
    # LQR参数
    # P由理论求解(dx/dt = Ax+Bu, Q=I, R=I)
    P_lqr = np.array([[np.sqrt(3), 1.0], [1.0, np.sqrt(3)]])
    B_lqr = np.array([[0.0], [1.0]])
    R_lqr = np.array([[1.0]])
    def closed_loop_dynamics(t, x):
        xt = torch.from_numpy(x).float().unsqueeze(0)
        with torch.no_grad():
            Lambda = hnn.get_lambda(xt).cpu().numpy().squeeze()
        u = -np.linalg.inv(R) @ B.T @ Lambda.reshape(-1, 1)
        u = u.squeeze()
        dx1 = x[1]
        dx2 = u
        return np.array([dx1, dx2])
    def lqr_closed_loop_dynamics(t, x):
        # V(x) = x^T P x, ∇V = 2Px
        gradV = 2 * P_lqr @ x.reshape(-1, 1)
        u = -np.linalg.inv(R_lqr) @ B_lqr.T @ gradV
        u = u.squeeze()
        dx1 = x[1]
        dx2 = u
        return np.array([dx1, dx2])
    # 两组初值
    x0_list = [[-0.9, -0.4], [0.9, 0.4], [-0.9, 0.4], [0.9, -0.4]]
    for x0 in x0_list:
        # HNN仿真轨迹（红色）
        sol = solve_ivp(closed_loop_dynamics, [0, 8], x0, t_eval=np.linspace(0,8,200))
        ax4.plot(sol.y[0], sol.y[1], 'r-', lw=2, label="CHNN" if x0==x0_list[0] else "")
        # # LQR仿真轨迹（灰色）
        # sol_lqr = solve_ivp(lqr_closed_loop_dynamics, [0, 8], x0, t_eval=np.linspace(0,8,200))
        # ax4.plot(sol_lqr.y[0], sol_lqr.y[1], color='gray', lw=2, linestyle='--', label="LQR" if x0==x0_list[0] else "")
    # 初始点标记出来，使用散点
    x0_arr = np.array(x0_list)
    ax4.scatter(x0_arr[:, 0], x0_arr[:, 1], color='red', s=80, marker='o', edgecolors='k')
    # # 添加图例只显示一次
    # handles, labels = ax4.get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # ax4.legend(by_label.values(), by_label.keys(), loc='upper right')
    # 原点用散点标出
    ax4.scatter([0], [0], color='blue', s=80, marker='o', edgecolors='k', zorder=10)
    filename = "image/" + args.filename_prefix + "phase_portrait.svg"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)

    plt.show()


def simulate_once():
    """
    加载训练好的HNN网络和环境，进行一次仿真，并画出状态和控制的时间曲线
    同时加入LQR的理论轨迹用于对比
    """
    seed = 1318
    # ============================== HNN 仿真 ==============================
    # 加载环境
    env = gym.make(Args.env_id)
    obs, _ = env.reset(seed=seed)
    obs = np.array(obs, dtype=np.float32)
    obs_hnn_init = obs.copy()

    # 加载网络
    hnn = HNN(env)
    model_path = "model/" + args.filename_prefix + "v1.pth"
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
    state_traj_hnn = [obs.copy()]
    control_traj_hnn = []
    time_traj = [0.0]

    # 记录HNN轨迹
    for t in range(T):
        x_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(Args.device)
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
        state_traj_hnn.append(obs.copy())
        control_traj_hnn.append(u)
        time_traj.append((t+1)*env.dt if hasattr(env, "dt") else (t+1)*0.05)
        # if terminated or truncated:
        #     break
    state_traj_hnn = np.array(state_traj_hnn)
    control_traj_hnn = np.array(control_traj_hnn)
    time_traj = np.array(time_traj)

    # ============================== LQR 理论仿真 ==============================
    # 理论LQR的参数
    P = np.array([[np.sqrt(3), 1.0], [1.0, np.sqrt(3)]])
    K = np.linalg.inv(np.diag([1])) @ np.array([[0, 1]]) @ P  # K shape (1,2)
    K = K.squeeze()  # shape (2,), K = [K1, K2]
    # 环境初始化
    env_lqr = gym.make(Args.env_id)
    obs, _ = env_lqr.reset(seed=seed)
    obs = np.array(obs, dtype=np.float32)
    state_traj_lqr = [obs.copy()]
    control_traj_lqr = []
    for t in range(T):
        # LQR控制律 u* = -K x
        u_lqr = -K @ obs
        u_lqr_scalar = np.array(u_lqr).squeeze()
        obs, _, terminated, truncated, _ = env_lqr.step(u_lqr_scalar)
        obs = np.array(obs, dtype=np.float32)
        state_traj_lqr.append(obs.copy())
        control_traj_lqr.append(u_lqr_scalar)
    state_traj_lqr = np.array(state_traj_lqr)
    control_traj_lqr = np.array(control_traj_lqr)

    # ============================== 绘图 ==============================
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    # 定义更优雅的蓝色（不纯蓝），比如tab:blue + LQR橙色为tab:orange
    hnn_color = '#1f77b4'  # matplotlib tab:blue (稍带青色)
    lqr_color = '#ff7f0e'  # matplotlib tab:orange

    # x1
    axs[0].plot(time_traj, state_traj_hnn[:, 0], label='x1 (HNN)', color=hnn_color)
    axs[0].plot(time_traj, state_traj_lqr[:, 0], label='x1 (LQR)', color=lqr_color)
    axs[0].set_ylabel('$x_1$')
    axs[0].set_title('x1 Trajectory')
    axs[0].legend()
    axs[0].grid(True)

    # x2
    axs[1].plot(time_traj, state_traj_hnn[:, 1], label='x2 (HNN)', color=hnn_color)  # tab:green
    axs[1].plot(time_traj, state_traj_lqr[:, 1], label='x2 (LQR)', color=lqr_color)
    axs[1].axhline(y=0.5, color='black', linestyle='--', linewidth=1.5)
    axs[1].axhline(y=-0.5, color='black', linestyle='--', linewidth=1.5)
    axs[1].set_ylabel('$x_2$')
    axs[1].set_title('x2 Trajectory')
    axs[1].legend()
    axs[1].grid(True)

    # control input
    axs[2].plot(time_traj[:-1], control_traj_hnn, label='u (HNN)', color=hnn_color)
    axs[2].plot(time_traj[:-1], control_traj_lqr, label='u (LQR)', color=lqr_color)
    axs[2].set_xlabel('$t(s)$')
    axs[2].set_ylabel('$u$')
    axs[2].set_title('Control Input')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    filename = "image/" + args.filename_prefix + "state_and_control_comp.png"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)
    plt.show()
    

if __name__ == "__main__":
    args = Args()
    # train_hnn()
    plot_value()
    simulate_once()