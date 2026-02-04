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
from p3_mlp_att_angle_training import MLP


# ================== 图像设置 =======================
mpl.rcParams['font.family'] = 'Times New Roman'  # 默认字体
# mpl.rcParams['font.weight'] = 'bold'  # 加粗
mpl.rcParams['text.usetex'] = True  # 使用TEX
mpl.rcParams['axes.unicode_minus'] = False  # 解决无法显示负号
mpl.rcParams['xtick.direction'] = 'in'  # x轴刻度线朝内
mpl.rcParams['ytick.direction'] = 'in'  # y轴刻度线朝内
mpl.rcParams['xtick.top'] = True  # 显示上方的坐标轴
mpl.rcParams['ytick.right'] = True  # 显示右侧的坐标轴

mpl.rcParams['legend.frameon'] = False  # legend不显示边框
mpl.rcParams['legend.fontsize'] = 24  # legend默认size

mpl.rcParams['axes.titlesize'] = 24   # 设置title默认size
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
    env_id = "environment:angle3d-v0"
    # 学习率
    learning_rate = 1e-3
    # 总训练次数
    time_steps = 10000
    # 训练批次
    batch_size = 1280
    # 是否使用GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 文件名前缀
    filename_prefix = "hnn_angle3d_"

# 动力学相关
def dynamics(X, U):
    # X: (N, 3) -> [phi, theta, psi], U: (N, 3) -> [p, q, r]
    phi = X[:, 0:1]
    theta = X[:, 1:2]
    # psi = X[:, 2:3]  # not used directly
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


# 计算控制矩阵pfpu
def get_pfpu(X):
    """
    计算动力学对u的雅可比矩阵（每个样本一个3x3矩阵）
    X: (N, 3) -> [phi, theta, psi]
    返回: (N, 3, 3)
    """
    phi = X[:, 0]
    theta = X[:, 1]

    N = X.shape[0]
    pfpu = torch.zeros(N, 3, 3, dtype=X.dtype, device=X.device)

    # dphi/du
    pfpu[:, 0, 0] = 1.0
    pfpu[:, 0, 1] = torch.tan(theta) * torch.sin(phi)
    pfpu[:, 0, 2] = torch.tan(theta) * torch.cos(phi)

    # dtheta/du
    pfpu[:, 1, 0] = 0.0
    pfpu[:, 1, 1] = torch.cos(phi)
    pfpu[:, 1, 2] = -torch.sin(phi)

    # dpsi/du
    cos_theta = torch.cos(theta)
    pfpu[:, 2, 0] = 0.0
    pfpu[:, 2, 1] = torch.sin(phi) / cos_theta
    pfpu[:, 2, 2] = torch.cos(phi) / cos_theta

    return pfpu

# 计算pfpx
def get_pfpx(X, U):
    """
    计算动力学对X的雅可比矩阵（每个样本一个3x3x3矩阵）
    X: (N, 3) -> [phi, theta, psi]
    U: (N, 3) -> [p, q, r]
    返回: (N, 3, 3)
    """
    phi = X[:, 0]
    theta = X[:, 1]
    # psi = X[:, 2]  # not used directly

    p = U[:, 0]
    q = U[:, 1]
    r = U[:, 2]

    N = X.shape[0]
    pfpx = torch.zeros(N, 3, 3, dtype=X.dtype, device=X.device)

    # d/dphi
    pfpx[:, 0, 0] = q * torch.cos(phi) * torch.tan(theta) - r * torch.sin(phi) * torch.tan(theta)
    pfpx[:, 1, 0] = -q * torch.sin(phi) - r * torch.cos(phi)
    pfpx[:, 2, 0] = q * torch.cos(phi) / torch.cos(theta) - r * torch.sin(phi) / torch.cos(theta)

    # d/dtheta
    tan_theta = torch.tan(theta)
    sec_theta_sq = tan_theta**2 + 1  # sec^2(theta) = tan^2(theta) + 1
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    pfpx[:, 0, 1] = q * sec_theta_sq * torch.sin(phi) + r * sec_theta_sq * torch.cos(phi)
    pfpx[:, 1, 1] = 0.0
    pfpx[:, 2, 1] = q * torch.sin(phi) * sin_theta / (cos_theta**2) + r * sin_theta * torch.cos(phi) / (cos_theta**2)

    # d/dpsi
    pfpx[:, 0, 2] = 0.0
    pfpx[:, 1, 2] = 0.0
    pfpx[:, 2, 2] = 0.0

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
        self.value_element = nn.Sequential(
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.state_dim * (self.state_dim + 1) // 2), std=1.0),
        )
        self.indices = [(i, j) for i in range(self.state_dim) for j in range(i + 1)]

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
        # zero_point = torch.zeros_like(x)
        # x_feature = self.shared_net(x)
        # zero_point_feature = self.shared_net(zero_point)
        # return self.value_part(x_feature) - self.value_part(zero_point_feature)
        x = x - torch.tensor([0] * x.shape[1], dtype=x.dtype, device=x.device)
        batch_size = x.shape[0]
        x_feature = self.shared_net(x)
        elements = self.value_element(x_feature)
        L = torch.zeros(batch_size, self.state_dim, self.state_dim, device=x.device)
        for idx, (i, j) in enumerate(self.indices):
            L[:, i, j] += elements[:, idx]

        L_T = L.transpose(1, 2)
        intermediate = torch.bmm(x.unsqueeze(1), L_T).squeeze(1)
        value = (intermediate ** 2).sum(dim=1)
        value = value.unsqueeze(dim=1)
        return value

    
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
    pfpx = get_pfpx(X, u_star)  # (batch_size, 2, 2)
    term1 = 2 * torch.matmul(X, Q_torch.T)  # (batch_size, 2)
    # pfpx: (batch_size, 2, 2), Lambda_detach_unsq: (batch_size, 2, 1)
    pfpx_T = pfpx.transpose(1, 2)  # (batch_size, 2, 2)
    term2 = torch.matmul(pfpx_T, Lambda_detach_unsq).squeeze(-1)  # (batch_size, 2)
    Lambda_dot = -(term1 + term2)  # (batch_size, 2)
    loss4 = torch.nn.functional.mse_loss(-pHpX, Lambda_dot)

    # 5 H = 0损失
    xQx = torch.einsum('bi,ij,bj->b', X, Q_torch, X)  # (batch_size,)
    # u_star_connect = u_star = -0.5 * torch.matmul(pfpu_T, Lambda_unsq)  # (batch_size, 1, 1)
    # u_star_connect = torch.matmul(R_inv, u_star_connect.squeeze(-1).T).T  # (batch_size, 1)
    uRu = torch.einsum('bi,ij,bj->b', u_star, R_torch, u_star)  # (batch_size,)
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

    P_ref = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
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
        low = -1.0
        high = 1.0
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
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "hnn_angle3d.pth")
    torch.save(hnn.state_dict(), model_path)
    print(f"HNN model saved to {model_path}")


def plot_value():
    # 加载模型参数
    model_dir = "model"
    model_path = os.path.join(model_dir, "hnn_angle3d.pth")
    args = Args()
    env = gym.make(args.env_id)
    hnn = HNN(env)
    hnn.load_state_dict(torch.load(model_path, map_location="cpu"))
    hnn.eval()

    # 构建三维状态的切片网格，x1,x2在-1到1之间，x3=0
    x1 = np.linspace(-1, 1, 50)
    x2 = np.linspace(-1, 1, 50)
    X1, X2 = np.meshgrid(x1, x2)
    X1_flat = X1.ravel()
    X2_flat = X2.ravel()
    X3_flat = np.zeros_like(X1_flat)
    X_grid = np.stack([X1_flat, X2_flat, X3_flat], axis=1)  # shape (N, 3)

    # HNN值函数
    X_tensor = torch.from_numpy(X_grid).float()
    with torch.no_grad():
        hnn_values = hnn.get_value(X_tensor).cpu().numpy()
    hnn_values = hnn_values.reshape(X1.shape)

    # LQR值函数: V(x) = x^T P x, 其中P为单位阵
    P = np.eye(3)
    lqr_values = np.einsum('ij,jk,ik->i', X_grid, P, X_grid).reshape(X1.shape)

    # 绘制对比曲面在同一张图上
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 使用单色渐变: HNN用Blues, LQR用Greens
    surf1 = ax.plot_surface(
        X1, X2, hnn_values, cmap='Blues', alpha=0.85, linewidth=0, antialiased=True, rstride=1, cstride=1
    )
    surf2 = ax.plot_surface(
        X1, X2, lqr_values, cmap='Greens', alpha=0.55, linewidth=0, antialiased=True, rstride=1, cstride=1
    )

    # 手动添加图例，颜色与曲面一致
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=plt.cm.Blues(0.7), lw=2, label='HNN'),
        Line2D([0], [0], color=plt.cm.Greens(0.7), lw=2, label='LQR')
    ]
    ax.legend(handles=legend_elements)

    # 坐标轴和标题
    ax.set_xlabel(r'$x_1$', labelpad=10)
    ax.set_ylabel(r'$x_2$', labelpad=10)
    ax.set_zlabel(r'$V(x)$', labelpad=10)
    ax.set_title(r'Value Function Surface ($x_3=0$)')
    ax.view_init(elev=28, azim=-50, roll=0)
    # 美化
    plt.tight_layout()
    filename = "image/" + args.filename_prefix + "value_function.png"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)
    plt.show()


def plot_lambda():
    # 加载hnn模型参数
    model_dir = "model"
    model_path = os.path.join(model_dir, "hnn_angle3d.pth")
    args = Args()
    env = gym.make(args.env_id)
    hnn = HNN(env)
    hnn.load_state_dict(torch.load(model_path, map_location="cpu"))
    hnn.eval()

    # 加载监督学习模型
    supervised_model = MLP(input_dim=3, output_dim=3, hidden_dim=32, num_layers=3)
    supervised_model.load_state_dict(torch.load("model/supervised_angle3d_model.pth", map_location="cpu"))
    supervised_model.eval()

    # 构建三维状态的切片网格，x1,x2在-1到1之间，x3=0
    x1 = np.linspace(-1, 1, 50)
    x2 = np.linspace(-1, 1, 50)
    X1, X2 = np.meshgrid(x1, x2)
    X1_flat = X1.ravel()
    X2_flat = X2.ravel()
    X3_flat = np.zeros_like(X1_flat)
    X_grid = np.stack([X1_flat, X2_flat, X3_flat], axis=1)  # shape (N, 3)

    X_tensor = torch.from_numpy(X_grid).float()

    # HNN Lambda
    with torch.no_grad():
        hnn_lambda = hnn.get_lambda(X_tensor).cpu().numpy()  # shape (N, 3)
    hnn_lambda1 = hnn_lambda[:, 0].reshape(X1.shape)
    hnn_lambda2 = hnn_lambda[:, 1].reshape(X1.shape)
    hnn_lambda3 = hnn_lambda[:, 2].reshape(X1.shape)

    # 监督模型 Lambda
    with torch.no_grad():
        sup_lambda = supervised_model(X_tensor).cpu().numpy()  # shape (N, 3)
    sup_lambda1 = sup_lambda[:, 0].reshape(X1.shape)
    sup_lambda2 = sup_lambda[:, 1].reshape(X1.shape)
    sup_lambda3 = sup_lambda[:, 2].reshape(X1.shape)

    # LQR Lambda: Lambda = 2 * P_ref * X, P_ref为单位阵
    P_ref = np.eye(3)
    lqr_lambda = 2 * (X_grid @ P_ref)  # shape (N, 3)
    lqr_lambda1 = lqr_lambda[:, 0].reshape(X1.shape)
    lqr_lambda2 = lqr_lambda[:, 1].reshape(X1.shape)
    lqr_lambda3 = lqr_lambda[:, 2].reshape(X1.shape)

    # 计算误差
    err_hnn1 = np.abs(hnn_lambda1 - sup_lambda1)
    err_hnn2 = np.abs(hnn_lambda2 - sup_lambda2)
    err_hnn3 = np.abs(hnn_lambda3 - sup_lambda3)
    err_lqr1 = np.abs(lqr_lambda1 - sup_lambda1)
    err_lqr2 = np.abs(lqr_lambda2 - sup_lambda2)
    err_lqr3 = np.abs(lqr_lambda3 - sup_lambda3)
    err_list = [err_hnn1, err_hnn2, err_hnn3, err_lqr1, err_lqr2, err_lqr3]

    # 统一colorbar范围
    all_errs = np.array([err_hnn1, err_hnn2, err_hnn3, err_lqr1, err_lqr2, err_lqr3])
    vmin = all_errs.min()
    vmax = all_errs.max()

    # ----------- 图一：三个子图，每个子图画三种模型对应维度的lambda曲面 -----------
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.lines import Line2D

    lambda_names = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$']
    lambda_all = [
        (hnn_lambda1, sup_lambda1, lqr_lambda1),
        (hnn_lambda2, sup_lambda2, lqr_lambda2),
        (hnn_lambda3, sup_lambda3, lqr_lambda3),
    ]
    model_names = ["HNN", "Supervised", "LQR"]
    model_cmaps = ["Purples", "Oranges", "Greens"]
    model_colors = [plt.cm.Blues(0.7), plt.cm.Oranges(0.7), plt.cm.Greens(0.7)]
    model_alphas = [0.85, 0.55, 0.35]
    model_alphas = [1, 1, 1]


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
            if i==2 and j == 2:
                surf = ax.plot_surface(
                        X1, X2, lam,
                        color=(0.0, 0.5, 0.0, 0.5),
                        edgecolor='none',
                        alpha=0.5,
                        rstride=1, cstride=1, linewidth=0, antialiased=True
                    )
            else:
                surf = ax.plot_surface(
                        X1, X2, lam,
                        cmap=cmap,
                        edgecolor='none',
                        alpha=alpha,
                        rstride=1, cstride=1, linewidth=0, antialiased=True
                    )
            if i == 1:
                ax.view_init(elev=30, azim=120, roll=0)
        # 找到等高线投影的最低z面
        z_min = ax.get_zlim()[0]
        ax.contourf(X1, X2, err_list[i], levels=10, cmap='RdBu_r', vmin=vmin, vmax=vmax, zdir='z', offset=z_min)
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_zlabel(lambda_names[i])

        filename = "image/" + args.filename_prefix + "lambda"+str(i+1)+".svg"
        fig.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)

    figs = []
    for i in range(3):
        figs.append(plt.figure(figsize=(6, 5)))

    for i in range(3):
        fig = figs[i]
        ax = fig.add_subplot(111, projection='3d')
        for j in [1, 2]:
            lam = lambda_all[i][j]
            cmap = model_cmaps[j]
            alpha = model_alphas[j]
            if i==2 and j == 2:
                surf = ax.plot_surface(
                        X1, X2, lam,
                        color=(0.0, 0.5, 0.0, 0.5),
                        edgecolor='none',
                        alpha=0.5,
                        rstride=1, cstride=1, linewidth=0, antialiased=True
                    )
            else:
                surf = ax.plot_surface(
                        X1, X2, lam,
                        cmap=cmap,
                        edgecolor='none',
                        alpha=alpha,
                        rstride=1, cstride=1, linewidth=0, antialiased=True
                    )
            if i == 1:
                ax.view_init(elev=30, azim=120, roll=0)
        # 找到等高线投影的最低z面
        z_min = ax.get_zlim()[0]
        error_cf = ax.contourf(X1, X2, err_list[i+3], levels=10, cmap='RdBu_r', vmin=vmin, vmax=vmax, zdir='z', offset=z_min)
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_zlabel(lambda_names[i])

        filename = "image/" + args.filename_prefix + "lambda_lqr"+str(i+1)+".svg"
        fig.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)

    # 统一colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig2 = plt.figure(figsize=(6, 10))
    ax = fig2.add_subplot(111)
    cbar = fig2.colorbar(error_cf, ax=ax, orientation='vertical', fraction=0.05, pad=0.02, location='right')
    filename = "image/" + args.filename_prefix + "lambda_colorbar.svg"
    fig2.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)

    plt.show()


def monte_carlo_simulation():
    args = Args()
    device = args.device
    env = gym.make(args.env_id)

    # Load HNN
    hnn = HNN(env).to(device)
    hnn_path = os.path.join("model", "hnn_angle3d.pth")
    if os.path.exists(hnn_path):
        hnn.load_state_dict(torch.load(hnn_path, map_location=device))
    hnn.eval()

    # Load supervised model
    supervised_model = MLP(input_dim=3, output_dim=3, hidden_dim=32, num_layers=3).to(device)
    sup_path = os.path.join("model", "supervised_angle3d_model.pth")
    if os.path.exists(sup_path):
        supervised_model.load_state_dict(torch.load(sup_path, map_location=device))
    supervised_model.eval()

    # Simulation parameters
    num_episodes = 50
    horizon = 200

    R_np = env.R
    R_torch = torch.tensor(R_np, dtype=torch.float32, device=device)
    R_inv = torch.linalg.inv(R_torch)


    def hnn_action(x_np):
        x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)  # (1, 3)
        with torch.no_grad():
            Lambda = hnn.get_lambda(x)  # (1, 3)
        pfpu = get_pfpu(x)  # (1, 3, 3)
        pfpu_T = pfpu.transpose(1, 2)  # (1, 3, 3)
        u_tmp = -0.5 * torch.matmul(pfpu_T, Lambda.unsqueeze(2)).squeeze(-1)  # (1, 3)
        u = (R_inv @ u_tmp.T).T.squeeze(0)  # (3,)
        return u.detach().cpu().numpy()

    def sup_action(x_np):
        x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)  # (1, 3)
        with torch.no_grad():
            Lambda = supervised_model(x)  # (1, 3)
        pfpu = get_pfpu(x)  # (1, 3, 3)
        pfpu_T = pfpu.transpose(1, 2)  # (1, 3, 3)
        u_tmp = -0.5 * torch.matmul(pfpu_T, Lambda.unsqueeze(2)).squeeze(-1)  # (1, 3)
        u = (R_inv @ u_tmp.T).T.squeeze(0)  # (3,)
        return u.detach().cpu().numpy()

    def lqr_action(x_np):
        # K = I, u = -K x = -x
        return -x_np

    # Collect metrics
    returns_hnn = []
    returns_sup = []
    returns_lqr = []

    traj_u_hnn = []  # (episodes, horizon, 3)
    traj_u_sup = []
    traj_u_lqr = []

    traj_x_hnn = []  # (episodes, horizon+1, 3)
    traj_x_sup = []
    traj_x_lqr = []

    for ep in range(num_episodes):
        print(f"{ep}/{num_episodes}")
        # Use same initial condition for all three controllers by sharing the seed
        seed = int(np.random.randint(0, 2**31 - 1))
        env_hnn = gym.make(args.env_id)
        env_sup = gym.make(args.env_id)
        env_lqr = gym.make(args.env_id)
        obs_hnn, _ = env_hnn.reset(seed=seed, theoretic_mode=True)
        obs_sup, _ = env_sup.reset(seed=seed, theoretic_mode=True)
        obs_lqr, _ = env_lqr.reset(seed=seed, theoretic_mode=True)

        ret_hnn = 0.0
        ret_sup = 0.0
        ret_lqr = 0.0

        u_seq_hnn = []
        u_seq_sup = []
        u_seq_lqr = []

        x_seq_hnn = [obs_hnn.copy()]
        x_seq_sup = [obs_sup.copy()]
        x_seq_lqr = [obs_lqr.copy()]

        for t in range(horizon):
            # HNN
            u_h = hnn_action(obs_hnn)
            next_obs_h, r_h, term_h, trunc_h, _ = env_hnn.step(u_h.astype(np.float32))
            ret_hnn += float(r_h)
            u_seq_hnn.append(u_h)
            obs_hnn = next_obs_h if not (term_h or trunc_h) else obs_hnn
            x_seq_hnn.append(obs_hnn.copy())

            # Supervised
            u_s = sup_action(obs_sup)
            next_obs_s, r_s, term_s, trunc_s, _ = env_sup.step(u_s.astype(np.float32))
            ret_sup += float(r_s)
            u_seq_sup.append(u_s)
            obs_sup = next_obs_s if not (term_s or trunc_s) else obs_sup
            x_seq_sup.append(obs_sup.copy())

            # LQR
            u_l = lqr_action(obs_lqr)
            next_obs_l, r_l, term_l, trunc_l, _ = env_lqr.step(u_l.astype(np.float32))
            ret_lqr += float(r_l)
            u_seq_lqr.append(u_l)
            obs_lqr = next_obs_l if not (term_l or trunc_l) else obs_lqr
            x_seq_lqr.append(obs_lqr.copy())

        env_hnn.close()
        env_sup.close()
        env_lqr.close()

        traj_u_hnn.append(np.stack(u_seq_hnn, axis=0))
        traj_u_sup.append(np.stack(u_seq_sup, axis=0))
        traj_u_lqr.append(np.stack(u_seq_lqr, axis=0))

        traj_x_hnn.append(np.stack(x_seq_hnn, axis=0))  # (horizon+1, 3)
        traj_x_sup.append(np.stack(x_seq_sup, axis=0))
        traj_x_lqr.append(np.stack(x_seq_lqr, axis=0))

        returns_hnn.append(ret_hnn)
        returns_sup.append(ret_sup)
        returns_lqr.append(ret_lqr)

    traj_u_hnn = np.stack(traj_u_hnn, axis=0)  # (E, T, 3)
    traj_u_sup = np.stack(traj_u_sup, axis=0)
    traj_u_lqr = np.stack(traj_u_lqr, axis=0)

    traj_x_hnn = np.stack(traj_x_hnn, axis=0)  # (E, T+1, 3)
    traj_x_sup = np.stack(traj_x_sup, axis=0)
    traj_x_lqr = np.stack(traj_x_lqr, axis=0)

    # Plot mean +/- std control trajectories per dimension
    time = np.arange(horizon)
    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    def plot_mean_std(ax, data, label, color):
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        ax.plot(time, mean, label=label, color=color)
        ax.fill_between(time, mean - std, mean + std, color=color, alpha=0.2)

    colors = {"HNN": "tab:blue", "Supervised": "tab:orange", "LQR": "tab:green"}
    for i, ax in enumerate(axs):
        plot_mean_std(ax, traj_u_hnn[:, :, i], "HNN", colors["HNN"])
        plot_mean_std(ax, traj_u_sup[:, :, i], "Supervised", colors["Supervised"])
        plot_mean_std(ax, traj_u_lqr[:, :, i], "LQR", colors["LQR"])
        ax.set_ylabel(f"u{i+1}")
        ax.grid(True, linestyle='--', alpha=0.4)
        if i == 0:
            ax.legend()
    axs[-1].set_xlabel("Time step")
    fig.suptitle("Control trajectories (mean ± std over 50 episodes)")
    plt.tight_layout()
    plt.show()

    # --------- 画状态轨迹统计图 ---------
    time_x = np.arange(horizon + 1)
    fig_x, axs_x = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    def plot_mean_std_x(ax, data, label, color):
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        ax.plot(time_x, mean, label=label, color=color)
        ax.fill_between(time_x, mean - std, mean + std, color=color, alpha=0.2)

    for i, ax in enumerate(axs_x):
        plot_mean_std_x(ax, traj_x_hnn[:, :, i], "HNN", colors["HNN"])
        plot_mean_std_x(ax, traj_x_sup[:, :, i], "Supervised", colors["Supervised"])
        plot_mean_std_x(ax, traj_x_lqr[:, :, i], "LQR", colors["LQR"])
        ax.set_ylabel(f"x{i+1}")
        ax.grid(True, linestyle='--', alpha=0.4)
        if i == 0:
            ax.legend()
    axs_x[-1].set_xlabel("Time step")
    fig_x.suptitle("State trajectories (mean ± std over 50 episodes)")
    plt.tight_layout()
    plt.show()

    # Print returns and plot histogram
    returns_hnn = np.array(returns_hnn)
    returns_sup = np.array(returns_sup)
    returns_lqr = np.array(returns_lqr)

    print("Returns over 50 episodes:")
    print(f"HNN: mean={returns_hnn.mean():.3f}, std={returns_hnn.std():.3f}")
    print(f"Supervised: mean={returns_sup.mean():.3f}, std={returns_sup.std():.3f}")
    print(f"LQR: mean={returns_lqr.mean():.3f}, std={returns_lqr.std():.3f}")

    plt.figure(figsize=(10, 4))
    bins = 15
    plt.hist(returns_hnn, bins=bins, alpha=0.5, label="HNN")
    plt.hist(returns_sup, bins=bins, alpha=0.5, label="Supervised")
    plt.hist(returns_lqr, bins=bins, alpha=0.5, label="LQR")
    plt.xlabel("Return")
    plt.ylabel("Count")
    plt.title("Return distributions over 50 episodes")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = Args()
    train_hnn()
    plot_value()
    plot_lambda()
    monte_carlo_simulation()