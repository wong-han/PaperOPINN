'''
Generate optimal data for three-axis attitude control using Numba JIT and multiprocessing.
'''

import os
import time
import numpy as np
import numba
from multiprocessing import Pool, cpu_count
from p3_0_mlp_generate_data_func import build_full_walk
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'simsun'
mpl.rcParams['axes.unicode_minus'] = False  # Render minus signs correctly

mpl.rcParams['xtick.direction'] = 'in'  # Draw ticks inward
mpl.rcParams['ytick.direction'] = 'in'  # Draw ticks inward
mpl.rcParams['xtick.top'] = True  # Show top ticks
mpl.rcParams['ytick.right'] = True  # Show right ticks

mpl.rcParams['legend.frameon'] = False  # Hide the legend frame
mpl.rcParams['legend.fontsize'] = 9  # Default legend font size

mpl.rcParams['xtick.labelsize'] = 9  # Default tick-label size
mpl.rcParams['ytick.labelsize'] = 9  # Default tick-label size
mpl.rcParams['axes.labelsize'] = 9  # Default axis-label size

dt = 0.01
r1, r2, r3 = 1.0, 1.0, 1.0
q1, q2, q3 = 1.0, 1.0, 1.0
S = np.eye(3)


@numba.njit(fastmath=True)
def numba_system(Z):
    """
    Compute the state and variational Jacobian derivatives (JIT compiled).
    """
    phi = Z[0]
    theta = Z[1]
    psi = Z[2]
    lam1 = Z[3]
    lam2 = Z[4]
    lam3 = Z[5]

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    tan_th = np.tan(theta)
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)
    sec2_th = tan_th**2 + 1.0


    pfpu_T = np.zeros((3, 3))
    pfpu_T[0, 0] = 1.0
    pfpu_T[1, 0] = sin_phi * tan_th
    pfpu_T[1, 1] = cos_phi
    pfpu_T[1, 2] = sin_phi / cos_th
    pfpu_T[2, 0] = cos_phi * tan_th
    pfpu_T[2, 1] = -sin_phi
    pfpu_T[2, 2] = cos_phi / cos_th

    Lambda = np.array([lam1, lam2, lam3])
    u_star = -0.5 * (pfpu_T @ Lambda)
    p = u_star[0]
    q = u_star[1]
    r = u_star[2]


    phi_dot = p + q * sin_phi * tan_th + r * cos_phi * tan_th
    theta_dot = q * cos_phi - r * sin_phi
    psi_dot = q * sin_phi / cos_th + r * cos_phi / cos_th

    lam1_dot = -(2.0 * q1 * phi + (q * cos_phi * tan_th - r * sin_phi * tan_th) * lam1 + (-q * sin_phi - r * cos_phi) * lam2 + (q * cos_phi / cos_th - r * sin_phi / cos_th) * lam3)
    lam2_dot = -(2.0 * q2 * theta + (q * sec2_th * sin_phi + r * sec2_th * cos_phi) * lam1 + (q * sin_phi * sin_th / cos_th**2 + r * sin_th * cos_phi / cos_th**2) * lam3)
    lam3_dot = -(2.0 * q3 * psi)


    pFpZ = np.zeros((6, 6))
    pFpZ[0, 0] = q * cos_phi * tan_th - r * sin_phi * tan_th
    pFpZ[0, 1] = q * sec2_th * sin_phi + r * sec2_th * cos_phi
    pFpZ[1, 0] = -q * sin_phi - r * cos_phi
    pFpZ[2, 0] = q * cos_phi / cos_th - r * sin_phi / cos_th
    pFpZ[2, 1] = q * sin_phi * sin_th / cos_th**2 + r * sin_th * cos_phi / cos_th**2

    pFpZ[3, 0] = -lam1 * (-q * sin_phi * tan_th - r * cos_phi * tan_th) - lam2 * (-q * cos_phi + r * sin_phi) - lam3 * (-q * sin_phi / cos_th - r * cos_phi / cos_th) - 2.0 * q1
    pFpZ[3, 1] = -lam1 * (q * sec2_th * cos_phi - r * sec2_th * sin_phi) - lam3 * (q * sin_th * cos_phi / cos_th**2 - r * sin_phi * sin_th / cos_th**2)
    pFpZ[3, 3] = -q * cos_phi * tan_th + r * sin_phi * tan_th
    pFpZ[3, 4] = q * sin_phi + r * cos_phi
    pFpZ[3, 5] = -q * cos_phi / cos_th + r * sin_phi / cos_th

    pFpZ[4, 0] = -lam1 * (q * sec2_th * cos_phi - r * sec2_th * sin_phi) - lam3 * (q * sin_th * cos_phi / cos_th**2 - r * sin_phi * sin_th / cos_th**2)
    pFpZ[4, 1] = -lam1 * (q * 2.0 * sec2_th * tan_th * sin_phi + r * 2.0 * sec2_th * tan_th * cos_phi) - lam3 * (2.0 * q * sin_phi * sin_th**2 / cos_th**3 + q * sin_phi / cos_th + 2.0 * r * sin_th**2 * cos_phi / cos_th**3 + r * cos_phi / cos_th) - 2.0 * q2
    pFpZ[4, 3] = -q * sec2_th * sin_phi - r * sec2_th * cos_phi
    pFpZ[4, 5] = -q * sin_phi * sin_th / cos_th**2 - r * sin_th * cos_phi / cos_th**2
    pFpZ[5, 2] = -2.0 * q3


    pFpU = np.zeros((6, 3))
    pFpU[0, 0] = 1.0
    pFpU[0, 1] = sin_phi * tan_th
    pFpU[0, 2] = cos_phi * tan_th
    pFpU[1, 1] = cos_phi
    pFpU[1, 2] = -sin_phi
    pFpU[2, 1] = sin_phi / cos_th
    pFpU[2, 2] = cos_phi / cos_th
    pFpU[3, 1] = -lam1 * cos_phi * tan_th + lam2 * sin_phi - lam3 * cos_phi / cos_th
    pFpU[3, 2] = lam1 * sin_phi * tan_th + lam2 * cos_phi + lam3 * sin_phi / cos_th
    pFpU[4, 1] = -lam1 * sec2_th * sin_phi - lam3 * sin_phi * sin_th / cos_th**2
    pFpU[4, 2] = -lam1 * sec2_th * cos_phi - lam3 * sin_th * cos_phi / cos_th**2


    pUpZ = np.zeros((3, 6))
    pUpZ[0, 3] = -0.5 / r1
    pUpZ[1, 0] = -0.5 * lam1 * cos_phi * tan_th / r2 + 0.5 * lam2 * sin_phi / r2 - 0.5 * lam3 * cos_phi / (r2 * cos_th)
    pUpZ[1, 1] = -0.5 * lam1 * sec2_th * sin_phi / r2 - 0.5 * lam3 * sin_phi * sin_th / (r2 * cos_th**2)
    pUpZ[1, 3] = -0.5 * sin_phi * tan_th / r2
    pUpZ[1, 4] = -0.5 * cos_phi / r2
    pUpZ[1, 5] = -0.5 * sin_phi / (r2 * cos_th)
    pUpZ[2, 0] = 0.5 * lam1 * sin_phi * tan_th / r3 + 0.5 * lam2 * cos_phi / r3 + 0.5 * lam3 * sin_phi / (r3 * cos_th)
    pUpZ[2, 1] = -0.5 * lam1 * sec2_th * cos_phi / r3 - 0.5 * lam3 * sin_th * cos_phi / (r3 * cos_th**2)
    pUpZ[2, 3] = -0.5 * cos_phi * tan_th / r3
    pUpZ[2, 4] = 0.5 * sin_phi / r3
    pUpZ[2, 5] = -0.5 * cos_phi / (r3 * cos_th)

    F_jacobian = pFpZ + pFpU @ pUpZ
    PHI = Z[6:42].reshape(6, 6)
    PHI_dot = (F_jacobian @ PHI).flatten()

    res = np.empty(42)
    res[0] = phi_dot
    res[1] = theta_dot
    res[2] = psi_dot
    res[3] = lam1_dot
    res[4] = lam2_dot
    res[5] = lam3_dot
    res[6:42] = PHI_dot
    return res


@numba.njit(fastmath=True)
def reverse_integ(state, T):
    """
    Preallocate memory and use Numba for efficient continuous backward integration.
    """
    steps = int(T / dt) + 1
    data_array = np.empty((steps, 46))
    Z = np.zeros(42)
    Z[0:3] = state
    Z[3:6] = 2.0 * state  # S = eye(3) => Lambda = 2*S*state

    for i in range(6):
        Z[6 + i * 6 + i] = 1.0  # PHI(0) = eye(6)

    V = state[0]**2 + state[1]**2 + state[2]**2

    phi = state[0]
    theta = state[1]
    psi = state[2]
    Lambda = Z[3:6]
    pfpu_T = np.zeros((3, 3))
    pfpu_T[0, 0] = 1.0
    pfpu_T[1, 0] = np.sin(phi) * np.tan(theta)
    pfpu_T[1, 1] = np.cos(phi)
    pfpu_T[1, 2] = np.sin(phi) / np.cos(theta)
    pfpu_T[2, 0] = np.cos(phi) * np.tan(theta)
    pfpu_T[2, 1] = -np.sin(phi)
    pfpu_T[2, 2] = np.cos(phi) / np.cos(theta)
    action = -0.5 * (pfpu_T @ Lambda)

    data_array[0, 0:42] = Z
    data_array[0, 42:45] = action
    data_array[0, 45] = V

    idx = 0
    while idx < steps - 1:
        dZdt = numba_system(Z)
        st = Z[0:3]
        phi = st[0]
        theta = st[1]
        psi = st[2]
        Lambda = Z[3:6]

        pfpu_T = np.zeros((3, 3))
        pfpu_T[0, 0] = 1.0
        pfpu_T[1, 0] = np.sin(phi) * np.tan(theta)
        pfpu_T[1, 1] = np.cos(phi)
        pfpu_T[1, 2] = np.sin(phi) / np.cos(theta)
        pfpu_T[2, 0] = np.cos(phi) * np.tan(theta)
        pfpu_T[2, 1] = -np.sin(phi)
        pfpu_T[2, 2] = np.cos(phi) / np.cos(theta)
        action = -0.5 * (pfpu_T @ Lambda)

        V = V + (st[0]**2 + st[1]**2 + st[2]**2 + action[0]**2 + action[1]**2 + action[2]**2) * dt
        Z = Z - dZdt * dt
        idx += 1
        data_array[idx, 0:42] = Z
        data_array[idx, 42:45] = action
        data_array[idx, 45] = V

    return data_array


@numba.njit(fastmath=True)
def shoot_to_point(target_x, T=3.5, n_steps=15):
    """
    Construct an initial state guess for target_x using homotopy continuation from the origin.
    """
    cur_state = np.zeros(3)
    path = np.empty((n_steps, 3))
    for i in range(n_steps):
        path[i] = cur_state + ((i + 1.0) / n_steps) * (target_x - cur_state)

    for step_idx in range(n_steps):
        pt = path[step_idx]
        for _ in range(50):
            Z_array = reverse_integ(cur_state, T)
            x_old = Z_array[-1, 0:3]
            diff = pt - x_old
            if np.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2) < 0.01:
                break
            PHI = Z_array[-1, 6:42].reshape(6, 6)
            LU = PHI[0:3, 0:3]
            RU = PHI[0:3, 3:6]
            STM = LU + 2.0 * RU  # S = eye(3)
            cur_state = cur_state + np.linalg.solve(STM, diff)

    return cur_state


def worker_task(args_tuple):
    """
    Independent task block executed by each worker process.
    """
    worker_id, sub_walk, global_start_idx, interval, T = args_tuple
    first_target = sub_walk[0]


    state = shoot_to_point(first_target, T, n_steps=15)
    worker_data = []

    for i in range(len(sub_walk)):
        x_d = sub_walk[i]
        global_idx = global_start_idx + i
        for _ in range(50):
            Z_array = reverse_integ(state, T)
            x_old = Z_array[-1, 0:3]
            diff = x_d - x_old

            if np.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2) < 0.01:
                if global_idx % interval == 0:
                    worker_data.append(Z_array)
                break

            PHI = Z_array[-1, 6:42].reshape(6, 6)
            LU = PHI[0:3, 0:3]
            RU = PHI[0:3, 3:6]
            STM = LU + 2.0 * RU
            delta_state = np.linalg.solve(STM, diff)
            state = state + delta_state

    return worker_id, worker_data


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("image", exist_ok=True)

    PLOT = False
    T = 3.5
    xd_low, xd_high = -1.0, 1.0
    x_step = 0.11
    interval = 2


    warmup_state = np.array([-0.035, 0.024, 0.025])
    _ = reverse_integ(warmup_state, 1.0)
    _ = shoot_to_point(np.array([0.1, 0.1, 0.1]), 1.0, n_steps=2)


    state0 = np.array([-0.03523661, 0.02429563, 0.02535206])
    Z_array0 = reverse_integ(state0, T)
    x_d0 = Z_array0[-1, 0:3]
    walk = build_full_walk(x_d0, xd_low, xd_high, x_step)
    total_points = len(walk)


    total_cpus = cpu_count()
    num_workers = min(20, total_cpus)
    print(f"Detected CPU cores/threads: {total_cpus}, worker processes: {num_workers}")
    print(f"Total target points: {total_points}, sampling interval: {interval}")


    chunks = np.array_split(walk, num_workers)
    tasks = []
    cur_idx = 0
    for wid, chunk in enumerate(chunks):
        tasks.append((wid, chunk, cur_idx, interval, T))
        cur_idx += len(chunk)

    start_time = time.time()
    print("Starting parallel data generation with Numba JIT and multiprocessing...")

    with Pool(processes=num_workers) as pool:
        results = pool.map(worker_task, tasks)

    end_time = time.time()
    print(f"\n==========================================")
    print(f"Data generation completed. Total elapsed time: {end_time - start_time:.2f} s")
    print(f"==========================================")


    all_trajectories = []
    results.sort(key=lambda x: x[0])
    for wid, w_data in results:
        all_trajectories.extend(w_data)

    print(f"Total optimal trajectories collected: {len(all_trajectories)}")

    if len(all_trajectories) > 0:
        dataset = np.vstack(all_trajectories)
        index = [0, 1, 2, 3, 4, 5, -4, -3, -2, -1]
        dataset = dataset[:, index]
        save_path = "data/DATA.npz"
        np.savez(save_path, dataset=dataset)
        print(f"Dataset saved to {save_path}，shape: {dataset.shape}")


    if PLOT and len(all_trajectories) > 0:
        fig = plt.figure(figsize=[5, 5])
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)
        ax.set_xlabel(r'$\phi$(rad)')
        ax.set_ylabel(r'$\theta$(rad)')
        ax.set_zlabel(r'$\psi$(rad)')
        color_cycle = plt.cm.tab20.colors

        for idx, traj in enumerate(all_trajectories):
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                    color=color_cycle[idx % len(color_cycle)],
                    linewidth=1.0, alpha=0.7)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
