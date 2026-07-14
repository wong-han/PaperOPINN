'''
简化所有loss计算为一体; 使用正定值函数网络设计；
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


# ================== 图像设置 =======================
mpl.rcParams['font.family'] = 'Times New Roman'  # 默认字体
# mpl.rcParams['font.weight'] = 'bold'  # 加粗
mpl.rcParams['axes.unicode_minus'] = False  # 解决无法显示负号
mpl.rcParams['text.usetex'] = True  # Tex
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
    env_id = "environment:nonlinear2d-v0"
    # 学习率
    learning_rate = 1e-3
    # 总训练次数
    time_steps = 20000
    # 训练批次
    batch_size = 1280
    # 是否使用GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 文件名前缀
    filename_prefix = "hnn_nonlinear2d_"


# 动力学相关
def dynamics(X, U):
    x1 = X[:, 0:1]
    x2 = X[:, 1:2]
    x1_dot = -x1 + x2
    x2_dot = -0.5*x1 - 0.5*x2*(1 - torch.square(torch.cos(2*x1)+2)) + (torch.cos(2*x1)+2) * U
    fxu = torch.cat([x1_dot, x2_dot], dim=1)
    return fxu


# 计算控制矩阵pfpu
def get_pfpu(X):
    x1 = X[:, 0]
    pfpu = torch.zeros(X.shape[0], 2, 1, dtype=X.dtype, device=X.device)
    pfpu[:, 1, 0] = torch.cos(2*x1) + 2
    return pfpu

# 计算pfpx
def get_pfpx(X, U):
    x1 = X[:, 0]
    x2 = X[:, 1]
    u_sq = U.squeeze()

    pfpx = torch.zeros(X.shape[0], 2, 2, dtype=X.dtype, device=X.device)
    pfpx[:, 0, 0] = -1
    pfpx[:, 0, 1] = 1
    pfpx[:, 1, 0] = -0.5 - 2*x2*torch.sin(2*x1)*(torch.cos(2*x1)+2) - 2*u_sq*torch.sin(2*x1)
    pfpx[:, 1, 1] = -0.5 + 0.5*torch.square(torch.cos(2*x1)+2)

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
        x = x - torch.tensor([0, 0], dtype=x.dtype, device=x.device)
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

    P_ref = [[0.5, 0], [0, 1]]
    P_ref_tensor = torch.tensor(P_ref, dtype=grad_Lambda.dtype, device=grad_Lambda.device)
    grad_lambda_loss = torch.nn.functional.mse_loss(grad_Lambda.squeeze(0), 2*P_ref_tensor)
    # loss = zero_loss + grad_lambda_loss
    loss = grad_lambda_loss
    return loss



def train_hnn():
    env = gym.make(args.env_id)
    hnn = HNN(env).to(args.device)
    optimizer = optim.Adam(hnn.parameters(), lr=args.learning_rate)

    start_time = time.time()
    print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    for step in range(args.time_steps):
        if time.time() - start_time > 30:
            print(f"Time limit of 30s reached at step {step}. Stopping training.")
            model_dir = "model"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "hnn_nonlinear2d.pth")
            torch.save(hnn.state_dict(), model_path)
            print(f"HNN model saved to {model_path}")
            break

        # 采样数据
        low = -3.0
        high = 3.0
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
        loss = get_all_loss(hnn, X) #+ lambda_zero_loss(hnn)

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
    model_path = os.path.join(model_dir, "hnn_nonlinear2d.pth")
    torch.save(hnn.state_dict(), model_path)
    print(f"HNN model saved to {model_path}")


def plot_value():
    # 加载模型参数
    model_dir = "model"
    model_path = os.path.join(model_dir, "hnn_nonlinear2d.pth")
    args = Args()
    env = gym.make(args.env_id)
    hnn = HNN(env)
    hnn.load_state_dict(torch.load(model_path, map_location="cpu"))
    hnn.eval()

    # 构建网格
    x1 = np.linspace(-2.7, 2.7, 50)
    x2 = np.linspace(-2.7, 2.7, 50)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.stack([X1.ravel(), X2.ravel()], axis=1)

    # HNN值函数
    X_tensor = torch.from_numpy(X_grid).float()
    X_tensor.requires_grad_(True)
    with torch.no_grad():
        hnn_values = hnn.get_value(X_tensor).cpu().numpy()
    hnn_values = hnn_values.reshape(X1.shape)

    # 计算HNN的 V_dot；最优控制 u* = -0.5 * g^T * pVpx，其中 g = [0, cos(2*x1)+2]
    X_tensor.requires_grad_(True)
    V = hnn.get_value(X_tensor)
    V_sum = V.sum()
    pVpx = torch.autograd.grad(V_sum, X_tensor, create_graph=False)[0]  # (N, 2)
    x1 = X_tensor[:, 0:1]  # (N, 1)
    g2 = torch.cos(2 * x1) + 2  # (N, 1)
    u_star = -0.5 * g2 * pVpx[:, 1:2]  # (N, 1)，u* = -0.5 * g^T pVpx
    with torch.no_grad():
        fxu = dynamics(X_tensor, u_star)
    V_dot_hnn = (pVpx * fxu).sum(dim=1).reshape(X1.shape).detach().numpy()

    # 最优值函数 V(x) = x^T P x；最优 Lambda = 2Px；最优控制 u* = -0.5 * g^T * (2Px)
    P = np.array([[0.5, 0.0], [0.0, 1.0]])
    opt_values = np.einsum('ij,jk,ik->i', X_grid, P, X_grid).reshape(X1.shape)

    X_grid_torch = torch.from_numpy(X_grid).float()
    P_torch = torch.from_numpy(P).float()
    gradV_opt = 2 * torch.matmul(X_grid_torch, P_torch)  # (N, 2)
    x1_grid = X_grid_torch[:, 0:1]
    g2_grid = torch.cos(2 * x1_grid) + 2
    u_star_opt = -0.5 * g2_grid * gradV_opt[:, 1:2]  # (N, 1)
    with torch.no_grad():
        fxu_opt = dynamics(X_grid_torch, u_star_opt)
    V_dot_opt = (gradV_opt * fxu_opt).sum(dim=1).reshape(X1.shape).detach().numpy()

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1.4])

    # ----------- 3D曲面图：值函数和V_dot ----------------
    ax = fig.add_subplot(gs[0], projection='3d')

    surf1 = ax.plot_surface(
        X1, X2, hnn_values, cmap='Blues', alpha=0.83, linewidth=0, antialiased=True, rstride=1, cstride=1
    )
    surf2 = ax.plot_surface(
        X1, X2, opt_values, cmap='Oranges', alpha=0.47, linewidth=0, antialiased=True, rstride=1, cstride=1
    )

    # v_dot 曲面
    surf3 = ax.plot_surface(
        X1, X2, V_dot_hnn, cmap='Blues', alpha=0.46, linewidth=0, antialiased=True, rstride=1, cstride=1
    )

    surf3 = ax.plot_surface(
        X1, X2, V_dot_opt, cmap='Oranges', alpha=0.46, linewidth=0, antialiased=True, rstride=1, cstride=1
    )

    # 手动添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=plt.cm.Blues(0.7), lw=2, label='OCNet'),
        Line2D([0], [0], color=plt.cm.Oranges(0.7), lw=2, label='Optimal'),
    ]
    ax.legend(handles=legend_elements, ncol=2, loc='upper right')
    ax.set_xlabel(r'$x_1$', labelpad=10)
    ax.set_ylabel(r'$x_2$', labelpad=10)
    ax.set_zlabel(r'$V$ or $\dot V$', labelpad=10)
    ax.view_init(elev=10, azim=48, roll=0)

    # ----------- 相空间流线图（相轨迹streamplot/quiver）----------------
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim([-3.5, 3.2])
    ax2.set_ylim([-3.5, 3.2])
    # 设置较粗网格
    x1_stream = np.linspace(-3, 3, 22)
    x2_stream = np.linspace(-3, 3, 22)
    Xs1, Xs2 = np.meshgrid(x1_stream, x2_stream)
    U = np.zeros_like(Xs1)
    Vv = np.zeros_like(Xs2)

    for i in range(Xs1.shape[0]):
        for j in range(Xs1.shape[1]):
            x_point = np.array([Xs1[i, j], Xs2[i, j]])
            x_tensor = torch.from_numpy(x_point).float().unsqueeze(0)
            with torch.no_grad():
                Lambda = hnn.get_lambda(x_tensor).cpu().numpy().squeeze()
            B_np = np.array([[0], [1]])
            u = -B_np.T @ Lambda.reshape(-1, 1)
            u = float(u.squeeze())
            f_vec = dynamics(x_tensor, torch.tensor([[u]])).cpu().numpy().squeeze()
            U[i, j] = f_vec[0]
            Vv[i, j] = f_vec[1]

    speed = np.sqrt(U ** 2 + Vv ** 2)
    color=speed
    step = 2  # 每隔2个点取一次，提升稀疏度
    Q = ax2.quiver(
        Xs1[::step, ::step], Xs2[::step, ::step], 
        U[::step, ::step], Vv[::step, ::step], color[::step, ::step],
        cmap='plasma', angles='xy', scale=100, width=0.005
    )
    ax2.set_xlabel(r'$x_1$')
    ax2.set_ylabel(r'$x_2$')

    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.axis('equal')
    cb = plt.colorbar(Q, ax=ax2, fraction=0.045, pad=0.04)
    cb.set_label("Magnitude")

    filename = "image/" + args.filename_prefix + "value_function.svg"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)

    # 单独画对比曲面（图1第一个子图的单独版）
    fig2 = plt.figure(figsize=(6,5))
    ax2_3d = fig2.add_subplot(111, projection='3d')
    surf_fig2_1 = ax2_3d.plot_surface(
        X1, X2, hnn_values, cmap='Purples', alpha=1, linewidth=0, antialiased=True, rstride=1, cstride=1
    )
    surf_fig2_1_1 = ax2_3d.plot_surface(
        X1, X2, V_dot_hnn, cmap='Purples', alpha=1, linewidth=0, antialiased=True, rstride=1, cstride=1
    )
    surf_fig2_2 = ax2_3d.plot_surface(
        X1, X2, opt_values, cmap='Oranges', alpha=0.4, linewidth=0, antialiased=True, rstride=1, cstride=1
    )
    surf_fig2_2_1 = ax2_3d.plot_surface(
        X1, X2, V_dot_opt, cmap='Oranges', alpha=0.8, linewidth=0, antialiased=True, rstride=1, cstride=1
    )
    # ax2_3d.legend(handles=legend_elements, ncol=2, loc='upper right')
    ax2_3d.set_xlabel(r'$x_1$', labelpad=10)
    ax2_3d.set_ylabel(r'$x_2$', labelpad=10)
    ax2_3d.set_zlabel(r'$V$ and $\dot V$', labelpad=10)
    ax2_3d.view_init(elev=9, azim=-64, roll=0)
    ax2_3d.set_xlim([-3, 3])
    ax2_3d.set_ylim([-3, 3])
    ax2_3d.set_zlim([-80, 20])
    filename2 = "image/" + args.filename_prefix + "value_function_3d_standalone.svg"
    plt.savefig(filename2, dpi=600, bbox_inches='tight', pad_inches=0.5)

    print("Max abs error of value function (HNN vs OPT):", np.max(np.abs(hnn_values - opt_values)))
    print("Max abs error of time derivative (V_dot_hnn vs V_dot_opt):", np.max(np.abs(V_dot_hnn - V_dot_opt)))

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
        Xs1[::step, ::step], Xs2[::step, ::step], 
        U[::step, ::step], Vv[::step, ::step], color[::step, ::step],
        cmap='RdBu_r', angles='xy', scale=100, width=0.005
    )
    cb = plt.colorbar(Q, ax=ax5, fraction=0.045, pad=0.04)
    cb.set_label(r"Magnitude")
    ax5.set_xlabel(r'$x_1$', labelpad=10)
    ax5.set_ylabel(r'$x_2$', labelpad=10)
    ax5.axis('equal')
    filename = "image/" + args.filename_prefix + "quiver.svg"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)

    # 单独画控制量曲面对比
    u_hnn_surface = u_star.detach().cpu().numpy().reshape(X1.shape)
    u_opt_surface = u_star_opt.cpu().numpy().reshape(X1.shape)
    
    fig6 = plt.figure(figsize=(10, 8))
    ax6 = fig6.add_subplot(111, projection='3d')
    ax6.plot_surface(X1, X2, u_hnn_surface, cmap='Blues', alpha=0.85, linewidth=0, antialiased=True)
    ax6.plot_surface(X1, X2, u_opt_surface, cmap='Oranges', alpha=0.47, linewidth=0, antialiased=True)
    ax6.set_xlabel(r'$x_1$', labelpad=10)
    ax6.set_ylabel(r'$x_2$', labelpad=10)
    ax6.set_zlabel(r'$u$', labelpad=10)
    ax6.view_init(elev=10, azim=48, roll=0)
    
    from matplotlib.lines import Line2D
    legend_elements_u = [
        Line2D([0], [0], color=plt.cm.Blues(0.7), lw=2, label='HNN Control'),
        Line2D([0], [0], color=plt.cm.Oranges(0.7), lw=2, label='Optimal Control')
    ]
    ax6.legend(handles=legend_elements_u, loc='upper right')
    
    filename_u = "image/" + args.filename_prefix + "control_comparison.svg"
    plt.savefig(filename_u, dpi=600, bbox_inches='tight', pad_inches=0.5)

    plt.show()


def plot_lambda():
    """
    画出HNN和LQR的协态对比图（二维系统）。
    LQR协态: Lambda = 2 * P * X, 其中P = diag([0.5, 1])
    """
    # 加载hnn模型参数
    model_dir = "model"
    model_path = os.path.join(model_dir, "hnn_nonlinear2d.pth")
    args = Args()
    env = gym.make(args.env_id)
    hnn = HNN(env)
    hnn.load_state_dict(torch.load(model_path, map_location="cpu"))
    hnn.eval()

    # 构建二维状态网格
    x1 = np.linspace(-3, 3, 30)
    x2 = np.linspace(-3, 3, 30)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.stack([X1.ravel(), X2.ravel()], axis=1)  # shape (N, 2)

    X_tensor = torch.from_numpy(X_grid).float()

    # HNN Lambda
    with torch.no_grad():
        hnn_lambda = hnn.get_lambda(X_tensor).cpu().numpy()  # shape (N, 2)
    hnn_lambda1 = hnn_lambda[:, 0].reshape(X1.shape)
    hnn_lambda2 = hnn_lambda[:, 1].reshape(X1.shape)

    # LQR Lambda: Lambda = 2 * P * X, P = diag([0.5, 1])
    P = np.diag([0.5, 1.0])
    lqr_lambda = 2 * (X_grid @ P)  # shape (N, 2)
    lqr_lambda1 = lqr_lambda[:, 0].reshape(X1.shape)
    lqr_lambda2 = lqr_lambda[:, 1].reshape(X1.shape)

    # ----------- 图一：两个子图，分别画HNN和LQR的lambda1、lambda2曲面 -----------
    fig = plt.figure(figsize=(14, 6))
    lambda_names = [r'$\lambda_1$', r'$\lambda_2$']

    for i, (hnn_lam, lqr_lam) in enumerate(zip([hnn_lambda1, hnn_lambda2], [lqr_lambda1, lqr_lambda2])):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        surf1 = ax.plot_surface(X1, X2, hnn_lam, cmap='Blues', edgecolor='none', alpha=0.7, label='OCNet')
        surf2 = ax.plot_surface(X1, X2, lqr_lam, cmap='Oranges', edgecolor='none', alpha=0.5, label='LQR')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel(lambda_names[i])
        ax.set_title(f"{lambda_names[i]} Surface: HNN (blue) vs LQR (orange)")
        # 图例
        custom_lines = [
            plt.Line2D([0], [0], color='tab:blue', lw=4, label='OCNet'),
            plt.Line2D([0], [0], color='tab:orange', lw=4, label='LQR'),
        ]
        ax.legend(handles=custom_lines, loc='upper left')
    plt.tight_layout()

    # ----------- 图二：两个子图，画HNN和LQR协态的绝对误差 -----------
    fig2, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    err1 = np.abs(hnn_lambda1 - lqr_lambda1)
    err2 = np.abs(hnn_lambda2 - lqr_lambda2)
    vmax = max(err1.max(), err2.max())
    vmin = min(err1.min(), err2.min())
    cf1 = axs[0].contourf(X1, X2, err1, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[0].set_xlabel('x1')
    axs[0].set_ylabel('x2')
    axs[0].set_title(r'$|\lambda_1^{\mathrm{HNN}} - \lambda_1^{\mathrm{LQR}}|$')
    cf2 = axs[1].contourf(X1, X2, err2, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
    axs[1].set_xlabel('x1')
    axs[1].set_ylabel('x2')
    axs[1].set_title(r'$|\lambda_2^{\mathrm{HNN}} - \lambda_2^{\mathrm{LQR}}|$')
    # 统一colorbar
    fig2.colorbar(cf2, ax=axs, orientation='vertical', fraction=0.025, pad=0.02)
    plt.show()


def monte_carlo_simulation():
    args = Args()
    device = args.device
    env_probe = gym.make(args.env_id, theoretic_mode=True)

    # Load HNN
    hnn = HNN(env_probe).to(device)
    hnn_path = os.path.join("model", "hnn_nonlinear2d.pth")
    if os.path.exists(hnn_path):
        hnn.load_state_dict(torch.load(hnn_path, map_location=device))
    hnn.eval()

    # Simulation parameters
    num_episodes = 50
    horizon = 200

    R_np = env_probe.R
    R_torch = torch.tensor(R_np, dtype=torch.float32, device=device)
    R_inv = torch.linalg.inv(R_torch)

    def hnn_action(x_np):
        x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)  # (1, 2)
        with torch.no_grad():
            Lambda = hnn.get_lambda(x)  # (1, 2)
        pfpu = get_pfpu(x)  # (1, 2, 1)
        pfpu_T = pfpu.transpose(1, 2)  # (1, 1, 2)
        u_tmp = -0.5 * torch.matmul(pfpu_T, Lambda.unsqueeze(2)).squeeze(-1)  # (1, 1)
        u = (R_inv @ u_tmp.T).T.squeeze(0).squeeze(0)  # () scalar
        return np.array([u.detach().cpu().item()], dtype=np.float32)

    def lqr_action(x_np):
        # K = [1, 2], u = -K x
        K = np.array([1.0, 2.0], dtype=np.float32)
        u = -float(K @ x_np)
        return np.array([u], dtype=np.float32)

    returns_hnn = []
    returns_lqr = []

    traj_u_hnn = []  # (E, T, 1)
    traj_u_lqr = []
    traj_x_hnn = []  # (E, T+1, 2)
    traj_x_lqr = []

    for ep in range(num_episodes):
        # Shared initial condition via common seed
        seed = int(np.random.randint(0, 2**31 - 1))
        env_hnn = gym.make(args.env_id, theoretic_mode=True)
        env_lqr = gym.make(args.env_id, theoretic_mode=True)
        obs_hnn, _ = env_hnn.reset(seed=seed)
        obs_lqr, _ = env_lqr.reset(seed=seed)

        ret_hnn = 0.0
        ret_lqr = 0.0

        u_seq_hnn = []
        u_seq_lqr = []
        x_seq_hnn = [obs_hnn.copy()]
        x_seq_lqr = [obs_lqr.copy()]

        for t in range(horizon):
            # HNN control
            u_h = hnn_action(obs_hnn.astype(np.float32))  # (1,)
            next_obs_h, r_h, term_h, trunc_h, _ = env_hnn.step(u_h)
            ret_hnn += float(r_h)
            u_seq_hnn.append(u_h)
            obs_hnn = next_obs_h if not (term_h or trunc_h) else obs_hnn
            x_seq_hnn.append(obs_hnn.copy())

            # LQR control
            u_l = lqr_action(obs_lqr.astype(np.float32))
            next_obs_l, r_l, term_l, trunc_l, _ = env_lqr.step(u_l)
            ret_lqr += float(r_l)
            u_seq_lqr.append(u_l)
            obs_lqr = next_obs_l if not (term_l or trunc_l) else obs_lqr
            x_seq_lqr.append(obs_lqr.copy())

        env_hnn.close()
        env_lqr.close()

        traj_u_hnn.append(np.vstack(u_seq_hnn))  # (T, 1)
        traj_u_lqr.append(np.vstack(u_seq_lqr))
        traj_x_hnn.append(np.vstack(x_seq_hnn))  # (T+1, 2)
        traj_x_lqr.append(np.vstack(x_seq_lqr))
        returns_hnn.append(ret_hnn)
        returns_lqr.append(ret_lqr)

    traj_u_hnn = np.stack(traj_u_hnn, axis=0)  # (E, T, 1)
    traj_u_lqr = np.stack(traj_u_lqr, axis=0)
    traj_x_hnn = np.stack(traj_x_hnn, axis=0)  # (E, T+1, 2)
    traj_x_lqr = np.stack(traj_x_lqr, axis=0)

    # Plot control trajectories mean ± std
    time = np.arange(horizon)
    plt.figure(figsize=(9, 4))
    mean_h = traj_u_hnn.mean(axis=0).squeeze(-1)
    std_h = traj_u_hnn.std(axis=0).squeeze(-1)
    mean_l = traj_u_lqr.mean(axis=0).squeeze(-1)
    std_l = traj_u_lqr.std(axis=0).squeeze(-1)
    plt.plot(time, mean_h, label="HNN", color="tab:blue")
    plt.fill_between(time, mean_h - std_h, mean_h + std_h, color="tab:blue", alpha=0.2)
    plt.plot(time, mean_l, label="LQR (K=[1,2])", color="tab:green")
    plt.fill_between(time, mean_l - std_l, mean_l + std_l, color="tab:green", alpha=0.2)
    plt.xlabel("Time step")
    plt.ylabel("u")
    plt.title("Control trajectories (mean ± std over 50 episodes)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    # Plot state trajectories mean ± std
    time_x = np.arange(horizon + 1)
    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for i, ax in enumerate(axs):
        mean_h = traj_x_hnn[:, :, i].mean(axis=0)
        std_h = traj_x_hnn[:, :, i].std(axis=0)
        mean_l = traj_x_lqr[:, :, i].mean(axis=0)
        std_l = traj_x_lqr[:, :, i].std(axis=0)
        ax.plot(time_x, mean_h, label="HNN", color="tab:blue")
        ax.fill_between(time_x, mean_h - std_h, mean_h + std_h, color="tab:blue", alpha=0.2)
        ax.plot(time_x, mean_l, label="LQR (K=[1,2])", color="tab:green")
        ax.fill_between(time_x, mean_l - std_l, mean_l + std_l, color="tab:green", alpha=0.2)
        ax.set_ylabel(f"x{i+1}")
        ax.grid(True, linestyle='--', alpha=0.4)
        if i == 0:
            ax.legend()
    axs[-1].set_xlabel("Time step")
    fig.suptitle("State trajectories (mean ± std over 50 episodes)")
    plt.tight_layout()
    plt.show()

    # Print returns and histogram
    returns_hnn = np.array(returns_hnn)
    returns_lqr = np.array(returns_lqr)
    print("Returns over 50 episodes:")
    print(f"HNN: mean={returns_hnn.mean():.3f}, std={returns_hnn.std():.3f}")
    print(f"LQR: mean={returns_lqr.mean():.3f}, std={returns_lqr.std():.3f}")

    plt.figure(figsize=(9, 4))
    bins = 15
    plt.hist(returns_hnn, bins=bins, alpha=0.5, label="HNN")
    plt.hist(returns_lqr, bins=bins, alpha=0.5, label="LQR (K=[1,2])")
    plt.xlabel("Return")
    plt.ylabel("Count")
    plt.title("Return distributions over 50 episodes")
    plt.legend()
    plt.tight_layout()
    plt.show()


def simulate_once():
    # 仿真一次，对比HNN和LQR的控制效果
    args = Args()
    device = args.device
    env_probe = gym.make(args.env_id, theoretic_mode=True)

    # 加载HNN模型
    hnn = HNN(env_probe).to(device)
    hnn_path = os.path.join("model", "hnn_nonlinear2d.pth")
    if os.path.exists(hnn_path):
        hnn.load_state_dict(torch.load(hnn_path, map_location=device))
    hnn.eval()

    # 仿真参数
    horizon = 200

    R_np = env_probe.R
    R_torch = torch.tensor(R_np, dtype=torch.float32, device=device)
    R_inv = torch.linalg.inv(R_torch)

    def hnn_action(x_np):
        x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)  # (1, 2)
        with torch.no_grad():
            Lambda = hnn.get_lambda(x)  # (1, 2)
        pfpu = get_pfpu(x)  # (1, 2, 1)
        pfpu_T = pfpu.transpose(1, 2)  # (1, 1, 2)
        u_tmp = -0.5 * torch.matmul(pfpu_T, Lambda.unsqueeze(2)).squeeze(-1)  # (1, 1)
        u = (R_inv @ u_tmp.T).T.squeeze(0).squeeze(0)  # () scalar
        return np.array([u.detach().cpu().item()], dtype=np.float32)

    def lqr_action(x_np):
        # K = [1, 2], u = -K x
        K = np.array([1.0, 2.0], dtype=np.float32)
        u = -float(K @ x_np)
        return np.array([u], dtype=np.float32)

    # 使用相同初始条件
    # seed = int(np.random.randint(0, 2**31 - 1))
    env_hnn = gym.make(args.env_id, theoretic_mode=True)
    env_lqr = gym.make(args.env_id, theoretic_mode=True)
    obs_hnn, _ = env_hnn.reset(seed=1)
    obs_lqr, _ = env_lqr.reset(seed=1)

    ret_hnn = 0.0
    ret_lqr = 0.0

    u_seq_hnn = []
    u_seq_lqr = []
    x_seq_hnn = [obs_hnn.copy()]
    x_seq_lqr = [obs_lqr.copy()]

    for t in range(horizon):
        # HNN控制
        u_h = hnn_action(obs_hnn.astype(np.float32))  # (1,)
        next_obs_h, r_h, term_h, trunc_h, _ = env_hnn.step(u_h)
        ret_hnn += float(r_h)
        u_seq_hnn.append(u_h)
        obs_hnn = next_obs_h
        x_seq_hnn.append(obs_hnn.copy())

        # LQR控制
        u_l = lqr_action(obs_lqr.astype(np.float32))
        next_obs_l, r_l, term_l, trunc_l, _ = env_lqr.step(u_l)
        ret_lqr += float(r_l)
        u_seq_lqr.append(u_l)
        obs_lqr = next_obs_l
        x_seq_lqr.append(obs_lqr.copy())

    env_hnn.close()
    env_lqr.close()

    traj_u_hnn = np.vstack(u_seq_hnn)  # (T, 1)
    traj_u_lqr = np.vstack(u_seq_lqr)
    traj_x_hnn = np.vstack(x_seq_hnn)  # (T+1, 2)
    traj_x_lqr = np.vstack(x_seq_lqr)

    # 绘制控制输入轨迹
    time = np.arange(horizon)
    plt.figure(figsize=(9, 4))
    plt.plot(time, traj_u_hnn.squeeze(-1), label="HNN", color="tab:blue")
    plt.plot(time, traj_u_lqr.squeeze(-1), label="LQR (K=[1,2])", color="tab:green")
    plt.xlabel("Time step")
    plt.ylabel("u")
    plt.title("Control trajectories (single episode)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    # 绘制状态轨迹
    time_x = np.arange(horizon + 1)
    fig, axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for i, ax in enumerate(axs):
        ax.plot(time_x, traj_x_hnn[:, i], label="HNN", color="tab:blue")
        ax.plot(time_x, traj_x_lqr[:, i], label="LQR (K=[1,2])", color="tab:green")
        ax.set_ylabel(f"x{i+1}")
        ax.grid(True, linestyle='--', alpha=0.4)
        if i == 0:
            ax.legend()
    axs[-1].set_xlabel("Time step")
    fig.suptitle("State trajectories (single episode)")
    plt.tight_layout()
    plt.show()

    # 打印回报
    print("Single episode return:")
    print(f"HNN: {ret_hnn:.3f}")
    print(f"LQR: {ret_lqr:.3f}")


if __name__ == "__main__":
    args = Args()
    # train_hnn()
    plot_value()
    plot_lambda()
    simulate_once()
    # monte_carlo_simulation()