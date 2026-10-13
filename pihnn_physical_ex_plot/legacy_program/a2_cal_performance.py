import numpy as np
import matplotlib.pyplot as plt


def read_data(file_path):
    """
    Read hnn/state_data_adjusted.txt and return the data array and column names.
    The first line contains the column names.
    """
    with open(file_path, encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        raise ValueError("The file is empty.")
    header = lines[0].strip().split()

    data = np.genfromtxt(lines[1:], delimiter=None)
    return header, data



def plot_state():

    state_header, state_data = read_data("hnn/state_data_adjusted.txt")
    u_header, u_data = read_data("hnn/u_data.txt")


    state_time = state_data[:, 0]
    u_time = u_data[:, 0]


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

    target_pos = np.array([0.58, 0.0, 0.9])
    Q = np.diag([5, 5, 20, 0.1, 0.1, 1, 0.1, 0.1, 0.1])
    R = np.diag([10, 1, 1, 1])
    u0 = np.array([0.27, 0, 0.21, 0])


    state_header_hnn, state_data_hnn = read_data("hnn/state_data_adjusted.txt")
    u_header_hnn, u_data_hnn = read_data("hnn/u_data.txt")


    state_header_lqr, state_data_lqr = read_data("lqr/state_data_adjusted.txt")
    u_header_lqr, u_data_lqr = read_data("lqr/u_data.txt")

    def compute_performance(state, u, Q, R, target_pos, u0):

        t = state[:, 0]
        x = state[:, 1:]
        t0 = t[0]

        mask = (t - t0) <= 10.0
        t = t[mask]
        x = x[mask, :]


        x_pos = x[:, 0:3]  # x, y, z
        x_rest = x[:, 3:]
        x_error = np.hstack([x_pos - target_pos.reshape(1, 3), x_rest])


        u_t = u[:, 0]
        u_data = u[:, 1:]

        u_mask = (u_t - u_t[0]) <= 10.0


        N = min(len(t), np.sum(u_mask))
        u_aligned = u_data[:N, :]
        t = t[:N]
        x_error = x_error[:N, :]


        u_error = u_aligned - u0.reshape(1, -1)


        xQx = np.einsum('ij,jk,ik->i', x_error, Q, x_error)

        uRu = np.einsum('ij,jk,ik->i', u_error, R, u_error)
        integrand = xQx + uRu


        dt = np.diff(t)

        J = np.sum(0.5 * (integrand[:-1] + integrand[1:]) * dt)
        return J


    J_hnn = compute_performance(state_data_hnn, u_data_hnn, Q, R, target_pos, u0)

    J_lqr = compute_performance(state_data_lqr, u_data_lqr, Q, R, target_pos, u0)

    print(f"OPINN performance metric: {J_hnn}")
    print(f"LQR performance metric: {J_lqr}")


def cal_pos_error():

    state_header_hnn, state_data_hnn = read_data("hnn/state_data_adjusted.txt")

    state_header_lqr, state_data_lqr = read_data("lqr/state_data_adjusted.txt")


    target_pos = np.array([0.58, 0.0, 0.9])


    pos_hnn = state_data_hnn[:, 1:4]
    pos_lqr = state_data_lqr[:, 1:4]


    mse_hnn = np.mean(np.sum((pos_hnn - target_pos) ** 2, axis=1))
    mse_lqr = np.mean(np.sum((pos_lqr - target_pos) ** 2, axis=1))

    print(f"OPINN position MSE: {mse_hnn}")
    print(f"LQR position MSE: {mse_lqr}")

    

# plot_state()
cal_performance()
cal_pos_error()
