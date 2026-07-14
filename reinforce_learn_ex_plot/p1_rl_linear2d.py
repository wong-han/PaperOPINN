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

# ================== 图像设置 =======================
mpl.rcParams['font.family'] = 'Times New Roman'  # 默认字体
# mpl.rcParams['font.weight'] = 'bold'  # 加粗
mpl.rcParams['text.usetex'] = True  # 是否使用tex渲染
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
    learning_rate = 3e-4
    # 总训练交互步数 (PPO需要较多数据)
    time_steps = 200000
    # 每次PPO更新的步数
    num_steps = 10
    # 训练批次
    batch_size = 64
    # PPO的优化轮数
    ppo_epochs = 10
    # clip参数
    clip_param = 0.2
    # 折扣因子
    gamma = 0.99
    # GAE lambda
    gae_lambda = 0.95
    # 是否使用GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 文件名前缀
    filename_prefix = "rl_linear2d_"


# 动力学相关 (用于测试和画图时的积分推演)
def dynamics(X, U):
    A = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=X.dtype, device=X.device)
    B = torch.tensor([[0.0], [1.0]], dtype=X.dtype, device=X.device)
    if U.dim() == 1:
        U = U.unsqueeze(1)
    fxu = torch.matmul(X, A.T) + torch.matmul(U, B.T)
    return fxu


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ================== PPO 强化学习网络 ==================
class RLNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Actor 网络：预测动作均值
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        # Actor 动作对数标准差：独立可学习参数
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic 网络：预测状态价值
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def get_action(self, x):
        # 推理测试时，直接使用策略均值作为确定性动作
        return self.actor_mean(x)


# ================== PPO 训练逻辑 ==================
def train_rl():
    env = gym.make(Args.env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = 1  # 针对当前线性的2D环境，控制量为1维
    
    agent = RLNet(state_dim, action_dim).to(Args.device)
    optimizer = optim.Adam(agent.parameters(), lr=Args.learning_rate, eps=1e-5)
    
    # 初始化 rollout 缓冲区
    num_steps = Args.num_steps
    obs = torch.zeros((num_steps, state_dim)).to(Args.device)
    actions = torch.zeros((num_steps, action_dim)).to(Args.device)
    logprobs = torch.zeros((num_steps,)).to(Args.device)
    rewards = torch.zeros((num_steps,)).to(Args.device)
    dones = torch.zeros((num_steps,)).to(Args.device)
    values = torch.zeros((num_steps,)).to(Args.device)

    global_step = 0
    num_updates = Args.time_steps // num_steps
    
    next_obs, _ = env.reset()
    next_obs = torch.Tensor(next_obs).to(Args.device)
    next_done = torch.zeros(1).to(Args.device)

    # start_time = time.time()
    # print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    # print(f"Starting PPO training for {Args.time_steps} steps...")

    for update in range(1, num_updates + 1):
        # if time.time() - start_time > 30:
        #     print(f"Time limit of 30s reached at update {update}. Stopping training.")
        #     break

        # 采集数据
        for step in range(0, num_steps):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done
            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs.unsqueeze(0))
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            
            action_np = action.cpu().numpy().flatten()
            next_obs_np, reward, terminated, truncated, info = env.step(action_np)
            
            rewards[step] = torch.tensor(reward).to(Args.device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(Args.device)
            next_done = torch.Tensor([terminated or truncated]).to(Args.device)
            
            if terminated or truncated:
                next_obs_np, _ = env.reset()
                next_obs = torch.Tensor(next_obs_np).to(Args.device)
                
        # 计算优势函数 (GAE)
        with torch.no_grad():
            next_value = agent.get_value(next_obs.unsqueeze(0)).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(Args.device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + Args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + Args.gamma * Args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # 准备小批量训练数据
        b_obs = obs.reshape((-1, state_dim))
        b_logprobs = logprobs.reshape((-1,))
        b_actions = actions.reshape((-1, action_dim))
        b_advantages = advantages.reshape((-1,))
        b_returns = returns.reshape((-1,))
        b_values = values.reshape((-1,))

        # 开始进行 PPO 网络优化
        b_inds = np.arange(num_steps)
        for epoch in range(Args.ppo_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, num_steps, Args.batch_size):
                end = start + Args.batch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                # 归一化 Advantage
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy Loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - Args.clip_param, 1 + Args.clip_param)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - 0.01 * entropy_loss + v_loss * 0.5

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
                
        if update % 5 == 0:
            print(f"Update: {update}/{num_updates}, Global Step: {global_step}, Policy Loss: {pg_loss.item():.4f}, Value Loss: {v_loss.item():.4f}")

    # 保存模型
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "rl_linear2d.pth")
    torch.save(agent.state_dict(), model_path)
    print(f"RL PPO model saved to {model_path}")


# ================== 在网格化场景下测试值函数与画图 ==================
def plot_value():
    model_dir = "model"
    model_path = os.path.join(model_dir, "rl_linear2d.pth")
    args = Args()
    env = gym.make(args.env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = 1
    
    agent = RLNet(state_dim, action_dim).to(Args.device)
    
    if os.path.exists(model_path):
        agent.load_state_dict(torch.load(model_path, map_location=Args.device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model file {model_path} not found. Running with untrained model.")
    agent.eval()

    # 构建网格
    x1 = np.linspace(-3, 3, 50)
    x2 = np.linspace(-3, 3, 50)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.stack([X1.ravel(), X2.ravel()], axis=1)
    X_tensor = torch.from_numpy(X_grid).float().to(Args.device)

    # ------------------ 通过网格化测试获取 RL 值函数 ------------------
    # 由于 RL 只输出了确定的动作，我们通过积分评估该策略的价值 V(x) = \int (x^TQx + u^TRu) dt
    dt = 0.05
    T_steps = 400  # 仿真20秒
    V_rl = torch.zeros(X_tensor.shape[0], device=Args.device)
    V_next_rl = torch.zeros(X_tensor.shape[0], device=Args.device)
    curr_x = X_tensor.clone()
    
    Q = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=Args.device)
    R = torch.tensor([[1.0]], device=Args.device)
    
    print("Simulating grid to compute RL value function...")
    with torch.no_grad():
        for step in range(T_steps):
            u = agent.get_action(curr_x)
            # 累加当前步 cost
            cost = torch.sum(curr_x * (curr_x @ Q), dim=1, keepdim=True) + torch.sum(u * (u @ R), dim=1, keepdim=True)
            step_cost = cost.squeeze() * dt
            
            V_rl += step_cost
            if step > 0:
                V_next_rl += step_cost
            
            # 使用 RK4 进行状态更新以保证精度
            k1 = dynamics(curr_x, u)
            x2_rk = curr_x + 0.5 * dt * k1
            u2 = agent.get_action(x2_rk)
            k2 = dynamics(x2_rk, u2)
            x3_rk = curr_x + 0.5 * dt * k2
            u3 = agent.get_action(x3_rk)
            k3 = dynamics(x3_rk, u3)
            x4_rk = curr_x + dt * k3
            u4 = agent.get_action(x4_rk)
            k4 = dynamics(x4_rk, u4)
            
            curr_x = curr_x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
        # 给 V_next_rl 补上最后一步的 cost，保证积分长度相同
        u = agent.get_action(curr_x)
        cost = torch.sum(curr_x * (curr_x @ Q), dim=1, keepdim=True) + torch.sum(u * (u @ R), dim=1, keepdim=True)
        V_next_rl += cost.squeeze() * dt

    rl_values = V_rl.cpu().numpy().reshape(X1.shape)
    
    # ------------------ 计算 RL 值函数的时间导数 \dot{V} ------------------
    # 直接根据轨迹序列上的 \Delta V 计算 \dot{V} = (V(x_{t+dt}) - V(x_t)) / dt
    V_dot_rl = ((V_next_rl - V_rl) / dt).cpu().numpy().reshape(X1.shape)

    # ------------------ 计算 LQR 值函数 ------------------
    P = np.array([[np.sqrt(3), 1.0], [1.0, np.sqrt(3)]])
    lqr_values = np.einsum('ij,jk,ik->i', X_grid, P, X_grid).reshape(X1.shape)
    # 计算LQR值函数的V_dot = (2*P*x)^T * f(x, u)
    # u_star_lqr = -0.5*R^{-1} B^T (2Px)
    X_grid_torch = torch.from_numpy(X_grid).float()
    P_torch = torch.from_numpy(P).float()
    B_lqr = torch.tensor([[0.0], [1.0]])
    R_inv_lqr = torch.tensor([[1.0]])
    gradV_lqr = 2 * torch.matmul(X_grid_torch, P_torch)  # (N,2)
    # 计算lqr闭环控制
    u_star_lqr = -0.5*torch.matmul(gradV_lqr, B_lqr)  # (N,1)
    fxu_lqr = dynamics(X_grid_torch, u_star_lqr)
    V_dot_lqr = (gradV_lqr * fxu_lqr).sum(dim=1).reshape(X1.shape).detach().numpy()

    # ================== 绘制对比曲面 ==================
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1.4])
    ax = fig.add_subplot(gs[0], projection='3d')

    surf1 = ax.plot_surface(X1, X2, rl_values, cmap='Blues', alpha=0.85, linewidth=0, antialiased=True)
    surf1_1 = ax.plot_surface(X1, X2, V_dot_rl, cmap='Blues', alpha=0.85, linewidth=0, antialiased=True)
    surf2 = ax.plot_surface(X1, X2, lqr_values, cmap='Oranges', alpha=0.55, linewidth=0, antialiased=True)
    surf2_1 = ax.plot_surface(X1, X2, V_dot_lqr, cmap='Oranges', alpha=0.55, linewidth=0, antialiased=True)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=plt.cm.Blues(0.7), lw=2, label='RL (PPO)'),
        Line2D([0], [0], color=plt.cm.Oranges(0.7), lw=2, label='Optimal')
    ]
    ax.legend(handles=legend_elements, ncol=2, loc='upper right')

    ax.set_xlabel(r'$x_1$', labelpad=10)
    ax.set_ylabel(r'$x_2$', labelpad=10)
    ax.set_zlabel(r'$V$ or $\dot V$', labelpad=10)
    ax.view_init(elev=5, azim=30, roll=0)


    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim([-3, 3])
    ax2.set_ylim([-3, 3])
    
    # 相轨迹矢量图
    x1_lin = np.linspace(-3, 3, 25)
    x2_lin = np.linspace(-3, 3, 25)
    X1_grid, X2_grid = np.meshgrid(x1_lin, x2_lin)
    
    U_vel = np.zeros_like(X1_grid)
    V_vel = np.zeros_like(X2_grid)

    for i in range(X1_grid.shape[0]):
        for j in range(X1_grid.shape[1]):
            x_point = np.array([X1_grid[i, j], X2_grid[i, j]])
            x_tensor = torch.from_numpy(x_point).float().unsqueeze(0).to(Args.device)
            with torch.no_grad():
                u = agent.get_action(x_tensor).cpu().numpy().squeeze()
            U_vel[i, j] = x_point[1]
            V_vel[i, j] = u

    speed = np.sqrt(U_vel**2 + V_vel**2)
    color = speed
    step = 2
    Q_quiver = ax2.quiver(
        X1_grid[::step, ::step], X2_grid[::step, ::step], 
        U_vel[::step, ::step], V_vel[::step, ::step], color[::step, ::step],
        cmap='plasma', angles='xy', scale=100, width=0.005
    )
    cb = plt.colorbar(Q_quiver, ax=ax2, fraction=0.045, pad=0.04)
    cb.set_label(r"Magnitude")
    
    ax2.set_xlabel(r"$x_1$")
    ax2.set_ylabel(r"$x_2$")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.axis('equal')

    os.makedirs("image", exist_ok=True)
    filename = "image/" + args.filename_prefix + "value_function.svg"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)

    # 单独画对比曲面（图1第一个子图的单独版）
    fig2_standalone = plt.figure(figsize=(6,5))
    ax2_3d = fig2_standalone.add_subplot(111, projection='3d')
    surf_fig2_1 = ax2_3d.plot_surface(
        X1, X2, rl_values, cmap='Blues', alpha=0.8, linewidth=0, antialiased=True, rstride=1, cstride=1
    )
    surf_fig2_1_1 = ax2_3d.plot_surface(
        X1, X2, V_dot_rl, cmap='Blues', alpha=0.8, linewidth=0, antialiased=True, rstride=1, cstride=1
    )
    surf_fig2_2 = ax2_3d.plot_surface(
        X1, X2, lqr_values, cmap='Oranges', alpha=1, linewidth=0, antialiased=True, rstride=1, cstride=1
    )
    surf_fig2_2_1 = ax2_3d.plot_surface(
        X1, X2, V_dot_lqr, cmap='Oranges', alpha=1, linewidth=0, antialiased=True, rstride=1, cstride=1
    )
    ax2_3d.set_xlabel(r'$x_1$', labelpad=10)
    ax2_3d.set_ylabel(r'$x_2$', labelpad=10)
    ax2_3d.set_zlabel(r'$V$ and $\dot V$', labelpad=10)
    ax2_3d.view_init(elev=12, azim=124, roll=0)
    ax2_3d.set_xlim([-3, 3])
    ax2_3d.set_ylim([-3, 3])
    # ax2_3d.set_zlim([-200, 40])
    filename2 = "image/" + args.filename_prefix + "value_function_3d_standalone.svg"
    plt.savefig(filename2, dpi=600, bbox_inches='tight', pad_inches=0.5)


    # 单独画 RL 值函数等高线图
    fig3 = plt.figure(figsize=(6,5))
    ax3 = fig3.add_subplot(111, projection='3d')
    z_min = min(np.min(rl_values), np.min(rl_values))
    ax3.plot_surface(X1, X2, rl_values, cmap='RdBu_r', alpha=0.8)
    ax3.contour(X1, X2, rl_values, zdir='z', offset=z_min, cmap='RdBu_r', alpha=1, levels=20)
    ax3.set_xlabel(r'$x_1$', labelpad=10)
    ax3.set_ylabel(r'$x_2$', labelpad=10)
    ax3.set_zlabel(r'$V$', labelpad=10, rotation=90)
    filename = "image/" + args.filename_prefix + "rl_value.svg"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)

    # 单独画 RL \dot{V} 的等高线图
    fig4 = plt.figure(figsize=(6,5))
    ax4 = fig4.add_subplot(111, projection='3d')
    z_min = min(np.min(V_dot_rl), np.min(V_dot_rl))
    ax4.plot_surface(X1, X2, V_dot_rl, cmap='RdBu_r', alpha=0.8)
    ax4.contour(X1, X2, V_dot_rl, zdir='z', offset=z_min, cmap='RdBu_r', alpha=1, levels=20)
    ax4.set_xlabel(r'$x_1$', labelpad=10)
    ax4.set_ylabel(r'$x_2$', labelpad=10)
    ax4.set_zlabel(r'$\dot V$', labelpad=20)
    filename = "image/" + args.filename_prefix + "rl_value_dot.svg"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)

    # 单独画相图 Quiver
    fig5 = plt.figure(figsize=(6,5))
    ax5 = fig5.add_subplot(111)
    Q_quiver2 = ax5.quiver(
        X1_grid[::step, ::step], X2_grid[::step, ::step], 
        U_vel[::step, ::step], V_vel[::step, ::step], color[::step, ::step],
        cmap='RdBu_r', angles='xy', scale=100, width=0.005
    )
    cb = plt.colorbar(Q_quiver2, ax=ax5, fraction=0.045, pad=0.04)
    cb.set_label(r"Magnitude")
    ax5.set_xlabel(r'$x_1$', labelpad=10)
    ax5.set_ylabel(r'$x_2$', labelpad=10)
    ax5.axis('equal')
    filename = "image/" + args.filename_prefix + "quiver.svg"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0.5)

    # 单独画控制量曲面对比
    with torch.no_grad():
        u_rl_tensor = agent.get_action(X_tensor)
    u_rl_surface = u_rl_tensor.cpu().numpy().reshape(X1.shape)
    u_opt_surface = u_star_lqr.detach().cpu().numpy().reshape(X1.shape)
    
    fig6 = plt.figure(figsize=(10, 8))
    ax6 = fig6.add_subplot(111, projection='3d')
    ax6.plot_surface(X1, X2, u_rl_surface, cmap='Blues', alpha=0.85, linewidth=0, antialiased=True)
    ax6.plot_surface(X1, X2, u_opt_surface, cmap='Oranges', alpha=0.47, linewidth=0, antialiased=True)
    ax6.set_xlabel(r'$x_1$', labelpad=10)
    ax6.set_ylabel(r'$x_2$', labelpad=10)
    ax6.set_zlabel(r'$u$', labelpad=10)
    ax6.view_init(elev=10, azim=48, roll=0)
    
    legend_elements_u = [
        Line2D([0], [0], color=plt.cm.Blues(0.7), lw=2, label='RL (PPO) Control'),
        Line2D([0], [0], color=plt.cm.Oranges(0.7), lw=2, label='Optimal Control')
    ]
    ax6.legend(handles=legend_elements_u, loc='upper right')
    
    # filename_u = "image/" + args.filename_prefix + "control_comparison.svg"
    # plt.savefig(filename_u, dpi=600, bbox_inches='tight', pad_inches=0.5)

    plt.show()


def simulate_once():
    """
    加载训练好的RL网络和环境，进行一次仿真，并画出状态和控制的时间曲线
    """
    env = gym.make(Args.env_id)
    options = {}
    options['init_state'] = np.array([-2.5, -3.0])
    obs, _ = env.reset(options=options)
    obs = np.array(obs, dtype=np.float32)
    
    state_dim = env.observation_space.shape[0]
    action_dim = 1
    agent = RLNet(state_dim, action_dim).to(Args.device)
    
    model_path = "model/" + "rl_linear2d.pth"
    if os.path.exists(model_path):
        agent.load_state_dict(torch.load(model_path, map_location=Args.device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model file {model_path} not found. Please train the model first.")
        return
    agent.eval()

    T = 200
    state_traj = [obs.copy()]
    control_traj = []
    time_traj = [0.0]
    for t in range(T):
        x_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(Args.device)
        with torch.no_grad():
            # 测试时取确定性动作
            u = agent.get_action(x_tensor).cpu().numpy().squeeze()
            if u.ndim == 0:
                u = np.expand_dims(u, axis=0)
        
        obs, _, terminated, truncated, _ = env.step(u)
        obs = np.array(obs, dtype=np.float32)
        state_traj.append(obs.copy())
        # 取数值以便绘图
        control_traj.append(u.item() if u.size == 1 else u)
        time_traj.append((t+1)*env.dt if hasattr(env, "dt") else (t+1)*0.05)

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
    # 训练模型请取消注释下面一行
    train_rl()
    plot_value()
    # simulate_once()
