'''
三轴角度控制最优数据生成 Generate Data
'''

import numpy as np
from copy import deepcopy
from p3_mlp_generate_data_func import build_full_walk
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True  # 使用Latex语法
mpl.rcParams['font.family'] = 'simsun'  # 解决中文问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决无法显示负号

mpl.rcParams['xtick.direction'] = 'in'  # x轴刻度线朝内
mpl.rcParams['ytick.direction'] = 'in'  # y轴刻度线朝内
mpl.rcParams['xtick.top'] = True  # 显示上方的坐标轴
mpl.rcParams['ytick.right'] = True  # 显示右侧的坐标轴

mpl.rcParams['legend.frameon'] = False  # legend不显示边框
mpl.rcParams['legend.fontsize'] = 9  # legend默认size

mpl.rcParams['xtick.labelsize'] = 9  # x坐标默认size
mpl.rcParams['ytick.labelsize'] = 9  # y坐标默认size
mpl.rcParams['axes.labelsize'] = 9  # 轴标题默认size


dt = 0.01
r1 = 1
r2 = 1
r3 = 1
q1 = 1
q2 = 1
q3 = 1
S = np.eye(3)
Q = np.eye(3)
R = np.eye(3)

x_dim = 3
u_dim = 3
z_dim = x_dim * 2

def system(Z):

    z = Z[0:z_dim]
    PHI = Z[z_dim:z_dim+z_dim**2].reshape(z_dim, z_dim)

    phi, theta, psi, lam1, lam2, lam3 = z

    pfpu = np.zeros((x_dim, u_dim))
    pfpu[0][0] += 1
    pfpu[0][1] += np.sin(phi)*np.tan(theta)
    pfpu[0][2] += np.cos(phi)*np.tan(theta)
    pfpu[1][0] += 0
    pfpu[1][1] += np.cos(phi)
    pfpu[1][2] += -np.sin(phi)
    pfpu[2][0] += 0
    pfpu[2][1] += np.sin(phi)/np.cos(theta)
    pfpu[2][2] += np.cos(phi)/np.cos(theta)

    Lambda = np.array([lam1, lam2, lam3])
    u_star = -0.5 * np.linalg.inv(R) * pfpu.T @ Lambda
    p, q, r = u_star

    phi_dot = p + q*np.sin(phi)*np.tan(theta) + r*np.cos(phi)*np.tan(theta)
    theta_dot = q*np.cos(phi) - r*np.sin(phi)
    psi_dot = q*np.sin(phi)/np.cos(theta) + r*np.cos(phi)/np.cos(theta)

    lam1_dot = -(2*q1*phi + (q*np.cos(phi)*np.tan(theta) - r*np.sin(phi)*np.tan(theta)) * lam1 + (-q*np.sin(phi) - r*np.cos(phi)) * lam2 + (q*np.cos(phi)/np.cos(theta) - r*np.sin(phi)/np.cos(theta)) * lam3)
    lam2_dot = -(2*q2*theta + (q*(np.tan(theta)**2 + 1)*np.sin(phi) + r*(np.tan(theta)**2 + 1)*np.cos(phi)) * lam1 + (q*np.sin(phi)*np.sin(theta)/np.cos(theta)**2 + r*np.sin(theta)*np.cos(phi)/np.cos(theta)**2) * lam3)
    lam3_dot = -(2*q3*psi)

    z_dot = np.array([phi_dot, theta_dot, psi_dot, lam1_dot, lam2_dot, lam3_dot])

    pFpZ = np.zeros((z_dim, z_dim))
    pFpZ[0][0] += q*np.cos(phi)*np.tan(theta) - r*np.sin(phi)*np.tan(theta)
    pFpZ[0][1] += q*(np.tan(theta)**2 + 1)*np.sin(phi) + r*(np.tan(theta)**2 + 1)*np.cos(phi)
    pFpZ[1][0] += -q*np.sin(phi) - r*np.cos(phi)
    pFpZ[2][0] += q*np.cos(phi)/np.cos(theta) - r*np.sin(phi)/np.cos(theta)
    pFpZ[2][1] += q*np.sin(phi)*np.sin(theta)/np.cos(theta)**2 + r*np.sin(theta)*np.cos(phi)/np.cos(theta)**2
    pFpZ[3][0] += -lam1*(-q*np.sin(phi)*np.tan(theta) - r*np.cos(phi)*np.tan(theta)) - lam2*(-q*np.cos(phi) + r*np.sin(phi)) - lam3*(-q*np.sin(phi)/np.cos(theta) - r*np.cos(phi)/np.cos(theta)) - 2*q1
    pFpZ[3][1] += -lam1*(q*(np.tan(theta)**2 + 1)*np.cos(phi) - r*(np.tan(theta)**2 + 1)*np.sin(phi)) - lam3*(q*np.sin(theta)*np.cos(phi)/np.cos(theta)**2 - r*np.sin(phi)*np.sin(theta)/np.cos(theta)**2)
    pFpZ[3][3] += -q*np.cos(phi)*np.tan(theta) + r*np.sin(phi)*np.tan(theta)
    pFpZ[3][4] += q*np.sin(phi) + r*np.cos(phi)
    pFpZ[3][5] += -q*np.cos(phi)/np.cos(theta) + r*np.sin(phi)/np.cos(theta)
    pFpZ[4][0] += -lam1*(q*(np.tan(theta)**2 + 1)*np.cos(phi) - r*(np.tan(theta)**2 + 1)*np.sin(phi)) - lam3*(q*np.sin(theta)*np.cos(phi)/np.cos(theta)**2 - r*np.sin(phi)*np.sin(theta)/np.cos(theta)**2)
    pFpZ[4][1] += -lam1*(q*(2*np.tan(theta)**2 + 2)*np.sin(phi)*np.tan(theta) + r*(2*np.tan(theta)**2 + 2)*np.cos(phi)*np.tan(theta)) - lam3*(2*q*np.sin(phi)*np.sin(theta)**2/np.cos(theta)**3 + q*np.sin(phi)/np.cos(theta) + 2*r*np.sin(theta)**2*np.cos(phi)/np.cos(theta)**3 + r*np.cos(phi)/np.cos(theta)) - 2*q2
    pFpZ[4][3] += -q*(np.tan(theta)**2 + 1)*np.sin(phi) - r*(np.tan(theta)**2 + 1)*np.cos(phi)
    pFpZ[4][5] += -q*np.sin(phi)*np.sin(theta)/np.cos(theta)**2 - r*np.sin(theta)*np.cos(phi)/np.cos(theta)**2
    pFpZ[5][2] += -2*q3
    
    pFpU = np.zeros((z_dim, u_dim))
    pFpU[0][0] += 1
    pFpU[0][1] += np.sin(phi)*np.tan(theta)
    pFpU[0][2] += np.cos(phi)*np.tan(theta)
    pFpU[1][1] += np.cos(phi)
    pFpU[1][2] += -np.sin(phi)
    pFpU[2][1] += np.sin(phi)/np.cos(theta)
    pFpU[2][2] += np.cos(phi)/np.cos(theta)
    pFpU[3][1] += -lam1*np.cos(phi)*np.tan(theta) + lam2*np.sin(phi) - lam3*np.cos(phi)/np.cos(theta)
    pFpU[3][2] += lam1*np.sin(phi)*np.tan(theta) + lam2*np.cos(phi) + lam3*np.sin(phi)/np.cos(theta)
    pFpU[4][1] += -lam1*(np.tan(theta)**2 + 1)*np.sin(phi) - lam3*np.sin(phi)*np.sin(theta)/np.cos(theta)**2
    pFpU[4][2] += -lam1*(np.tan(theta)**2 + 1)*np.cos(phi) - lam3*np.sin(theta)*np.cos(phi)/np.cos(theta)**2

    pUpZ = np.zeros((u_dim, z_dim))
    pUpZ[0][3] += -0.5/r1
    pUpZ[1][0] += -0.5*lam1*np.cos(phi)*np.tan(theta)/r2 + 0.5*lam2*np.sin(phi)/r2 - 0.5*lam3*np.cos(phi)/(r2*np.cos(theta))
    pUpZ[1][1] += -0.5*lam1*(np.tan(theta)**2 + 1)*np.sin(phi)/r2 - 0.5*lam3*np.sin(phi)*np.sin(theta)/(r2*np.cos(theta)**2)
    pUpZ[1][3] += -0.5*np.sin(phi)*np.tan(theta)/r2
    pUpZ[1][4] += -0.5*np.cos(phi)/r2
    pUpZ[1][5] += -0.5*np.sin(phi)/(r2*np.cos(theta))
    pUpZ[2][0] += 0.5*lam1*np.sin(phi)*np.tan(theta)/r3 + 0.5*lam2*np.cos(phi)/r3 + 0.5*lam3*np.sin(phi)/(r3*np.cos(theta))
    pUpZ[2][1] += -0.5*lam1*(np.tan(theta)**2 + 1)*np.cos(phi)/r3 - 0.5*lam3*np.sin(theta)*np.cos(phi)/(r3*np.cos(theta)**2)
    pUpZ[2][3] += -0.5*np.cos(phi)*np.tan(theta)/r3
    pUpZ[2][4] += 0.5*np.sin(phi)/r3
    pUpZ[2][5] += -0.5*np.cos(phi)/(r3*np.cos(theta))

    F_jacobian = pFpZ + np.matmul(pFpU, pUpZ)

    PHI_dot = F_jacobian @ PHI

    return np.hstack((z_dot, PHI_dot.flatten()))


def reverse_integ(state, T):

    t = 0
    PHI = np.eye(z_dim)

    Lambda = 2 * S @ state
    Z = np.hstack([state, Lambda, PHI.flatten()])

    phi, theta, psi = state
    pfpu = np.zeros((x_dim, u_dim))
    pfpu[0][0] += 1
    pfpu[0][1] += np.sin(phi)*np.tan(theta)
    pfpu[0][2] += np.cos(phi)*np.tan(theta)
    pfpu[1][0] += 0
    pfpu[1][1] += np.cos(phi)
    pfpu[1][2] += -np.sin(phi)
    pfpu[2][0] += 0
    pfpu[2][1] += np.sin(phi)/np.cos(theta)
    pfpu[2][2] += np.cos(phi)/np.cos(theta)

    u_star = -0.5 * np.linalg.inv(R) * pfpu.T @ Lambda
    p, q, r = u_star

    action = np.array([p, q, r])

    V = state.T @ S @ state
    DATA = [np.hstack([Z, action, V])]

    while t < T:
        dZdt = system(Z)

        state = Z[0:3]
        Lambda = Z[3:6]
        phi, theta, psi = state

        pfpu = np.zeros((x_dim, u_dim))
        pfpu[0][0] += 1
        pfpu[0][1] += np.sin(phi)*np.tan(theta)
        pfpu[0][2] += np.cos(phi)*np.tan(theta)
        pfpu[1][0] += 0
        pfpu[1][1] += np.cos(phi)
        pfpu[1][2] += -np.sin(phi)
        pfpu[2][0] += 0
        pfpu[2][1] += np.sin(phi)/np.cos(theta)
        pfpu[2][2] += np.cos(phi)/np.cos(theta)

        u_star = -0.5 * np.linalg.inv(R) * pfpu.T @ Lambda
        p, q, r = u_star
        action = np.array([p, q, r])

        V = V + (state.T @ Q @ state + action.T @ R @ action) * dt
        dZdt = -dZdt
        Z = Z + dZdt * dt
        t = t + dt
        DATA.append(np.hstack([Z, action, V]))

    data_array = np.array(DATA)
    return data_array

PLOT = False
if PLOT:
    fig = plt.figure(figsize=[3.5, 3.5])
    ax1_0 = fig.add_subplot(111, projection='3d')
    # 设置网格样式
    ax1_0.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)

    # 定义颜色循环和线型（用于区分不同轨迹）
    color_cycle = plt.cm.tab20.colors  # 使用高对比度色板
    line_styles = ['-', '--', '-.', ':']  # 不同线型

    # 设置坐标轴标签和标题
    ax1_0.set_xlabel(r'$\phi$(rad)', usetex=True)
    ax1_0.set_ylabel(r'$\theta$(rad)', usetex=True)
    ax1_0.set_zlabel(r'$\psi$(rad)', usetex=True)

state = np.array([-0.03523661, 0.02429563, 0.02535206])
print(state)
T = 3.5
xd_low, xd_high = -1.0, 1.0
x_step = 0.11


Z_array = reverse_integ(state, T)
x_d0 = Z_array[-1, 0:x_dim]
x_d = x_d0
walk = build_full_walk(x_d0, xd_low, xd_high, x_step)


DATA = []
count = -1
traj_counter = 0
interval = 2
walk_it = enumerate(walk, start=1)
while True:
    Z_array = reverse_integ(state, T)
    x_old = Z_array[-1, 0:x_dim]

    PHI = Z_array[-1, z_dim:z_dim+z_dim**2].reshape(z_dim, z_dim)
    LU = PHI[0:x_dim, 0:x_dim]
    RU = PHI[0:x_dim, x_dim:x_dim+x_dim]
    STM = LU + 2 * RU @ S
    # print(STM)

    if np.linalg.norm(x_old-x_d) < 0.01:
        hit_flag = True
        count += 1
        if count % interval == 0 and PLOT:
            ax1_0.scatter(x_d[0], x_d[1], x_d[2], c='k', marker='+', s=15, zorder=3)
        try:
            i, x_d = next(walk_it)   # 主动获取下一个值
        except StopIteration:
            break
    else:
        hit_flag = False


    if count % interval == 0 and hit_flag:
        if PLOT:
            ax1_0.plot(Z_array[:, 0], Z_array[:, 1], Z_array[:, 2],
                       color=color_cycle[traj_counter % len(color_cycle)],
                       # linestyle=line_styles[traj_counter % len(line_styles)],
                       linewidth=1.5,
                       alpha=0.8,
                       label=f'Trajectory {traj_counter + 1}')

        DATA.append(Z_array)
        traj_counter += 1

    delta_state = np.linalg.inv(STM) @ (x_d - x_old)
    state = state + delta_state
    print(state, T)
    print(f"process: {i} / {len(walk)}")

    if PLOT:
        plt.pause(0.1)


temp_data = DATA
length = len(temp_data)
while length != 1:
    temp_temp_data = []
    for i in range(0, len(temp_data), 2):
        if i == len(temp_data) - 1:
            temp_temp_data.append(temp_data[i])
        else:
            temp_temp_data.append(np.vstack((temp_data[i], temp_data[i + 1])))
    temp_data = temp_temp_data
    length = len(temp_data)

dataset = temp_data[0]
dataset = np.array(dataset)
index = [0, 1, 2, 3, 4, 5, -4, -3, -2, -1]
dataset = dataset[:, index]
np.savez("data/DATA.npz", dataset=dataset)

# fig.savefig('image/optimal_data_ATT.pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()



