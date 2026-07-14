import numpy as np
import matplotlib.pyplot as plt


def read_data(file_path):
    """
    读取 hnn/state_data_adjusted.txt 文件, 返回数据(np.ndarray)和表头(list of str)
    第一行为表头
    """
    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        raise ValueError("文件内容为空")
    header = lines[0].strip().split()
    # 支持tab 或 空格分割
    data = np.genfromtxt(lines[1:], delimiter=None)
    return header, data



def plot_state():
    # 读取state_data_adjusted.txt和u_data.txt文件
    state_header, state_data = read_data("hnn/state_data_adjusted.txt")
    u_header, u_data = read_data("hnn/u_data.txt")

    # 提取时间列(假定均为第一列)
    state_time = state_data[:, 0]
    u_time = u_data[:, 0]

    # 画出状态文件的所有状态曲线
    plt.figure(figsize=(12, 6))
    for i in range(1, state_data.shape[1]):
        plt.plot(state_time, state_data[:, i], label=state_header[i])
    plt.xlabel("Time")
    plt.ylabel("State Value")
    plt.title("State Data Time Series")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def cal_performance():
    # 期望目标位置
    target_pos = np.array([0.58, 0.0, 0.9])
    Q = np.diag([5, 5, 20, 0.1, 0.1, 1, 0.1, 0.1, 0.1])
    R = np.diag([10, 1, 1, 1])
    u0 = np.array([0.27, 0, 0.21, 0])  # u的平衡点

    # 读取hnn文件夹的数据
    state_header_hnn, state_data_hnn = read_data("hnn/state_data_adjusted.txt")
    u_header_hnn, u_data_hnn = read_data("hnn/u_data.txt")

    # 读取lqr文件夹的数据
    state_header_lqr, state_data_lqr = read_data("lqr/state_data_adjusted.txt")
    u_header_lqr, u_data_lqr = read_data("lqr/u_data.txt")

    def compute_performance(state, u, Q, R, target_pos, u0):
        # 提取时间，并只保留起始时刻起10秒内的数据
        t = state[:, 0]
        x = state[:, 1:]
        t0 = t[0]
        # 寻找不超过10秒的有效索引
        mask = (t - t0) <= 10.0
        t = t[mask]
        x = x[mask, :]

        # 位置误差 = 实际位置 - 目标位置
        x_pos = x[:, 0:3]  # x, y, z
        x_rest = x[:, 3:]  # 其余状态
        x_error = np.hstack([x_pos - target_pos.reshape(1, 3), x_rest])  # 替换前3项为误差

        # 控制量u数据也按与state同样的起止时间处理
        u_t = u[:, 0]
        u_data = u[:, 1:]
        # u中哪些行在t0开始的10s内
        u_mask = (u_t - u_t[0]) <= 10.0
        # 由于可能起始t不完全对齐，以状态的有效t为准，找到与t最近的u的行来对齐
        # 简单做法是：只选取u的前len(t)行，如果u比t短，则用短的长度
        N = min(len(t), np.sum(u_mask))
        u_aligned = u_data[:N, :]
        t = t[:N]
        x_error = x_error[:N, :]

        # 平衡点u0，计算u-u0
        u_error = u_aligned - u0.reshape(1, -1)

        # Q指标现在针对误差变量
        xQx = np.einsum('ij,jk,ik->i', x_error, Q, x_error)
        # u_error 是 shape (N, 4), R 是 (4,4)
        uRu = np.einsum('ij,jk,ik->i', u_error, R, u_error)
        integrand = xQx + uRu

        # 不使用np.trapz, 手动计算dt并用梯形法则积分
        dt = np.diff(t)
        # 积分值即 sum(0.5*(fi+fi+1) * dt)
        J = np.sum(0.5 * (integrand[:-1] + integrand[1:]) * dt)
        return J

    # hnn性能
    J_hnn = compute_performance(state_data_hnn, u_data_hnn, Q, R, target_pos, u0)
    # lqr性能
    J_lqr = compute_performance(state_data_lqr, u_data_lqr, Q, R, target_pos, u0)

    print(f"HNN 性能指标: {J_hnn}")
    print(f"LQR 性能指标: {J_lqr}")


def cal_pos_error():
    # 读取hnn文件夹的数据
    state_header_hnn, state_data_hnn = read_data("hnn/state_data_adjusted.txt")
    # 读取lqr文件夹的数据
    state_header_lqr, state_data_lqr = read_data("lqr/state_data_adjusted.txt")

    # 期望目标位置
    target_pos = np.array([0.58, 0.0, 0.9])

    # 假定位置分量为 state_data_? 的第1~3列（即数组索引1,2,3），第0列通常为时间戳
    pos_hnn = state_data_hnn[:, 1:4]
    pos_lqr = state_data_lqr[:, 1:4]

    # 按目标位置修正后，均方误差 mean squared error (MSE)
    mse_hnn = np.mean(np.sum((pos_hnn - target_pos) ** 2, axis=1))
    mse_lqr = np.mean(np.sum((pos_lqr - target_pos) ** 2, axis=1))

    print(f"HNN 位置均方误差: {mse_hnn}")
    print(f"LQR 位置均方误差: {mse_lqr}")

    

# plot_state()
cal_performance()
cal_pos_error()
