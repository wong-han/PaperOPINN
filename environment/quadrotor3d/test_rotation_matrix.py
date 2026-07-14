'''
测试四元数和欧拉角之间的旋转关系是否正确
'''


import numpy as np

def euler_kinematics(euler_angles, body_rates):
    """
    由欧拉角和机体系角速度(p, q, r)计算欧拉角变化率(dot_phi, dot_theta, dot_psi)
    euler_angles: [phi, theta, psi]
    body_rates: [p, q, r]
    返回: [dot_phi, dot_theta, dot_psi]
    """
    phi, theta, psi = euler_angles
    R = np.array([
        [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
    ])
    euler_angle_rate = R @ body_rates
    return euler_angle_rate


def quaternion_kinematics(quaternion, body_rates):
    q0, qx, qy, qz = quaternion
    T = np.array([
        [-qx, -qy, -qz],
        [q0, -qz, qy],
        [qz, q0, -qx],
        [-qy, qx, q0]
    ])
    quater_dot = 0.5 * T @ body_rates
    return quater_dot


def euler_to_quaternion(euler_angles):
    phi, theta, psi = euler_angles
    half_phi = phi / 2.0
    half_theta = theta / 2.0
    half_psi = psi / 2.0

    c_phi = np.cos(half_phi)
    s_phi = np.sin(half_phi)
    c_theta = np.cos(half_theta)
    s_theta = np.sin(half_theta)
    c_psi = np.cos(half_psi)
    s_psi = np.sin(half_psi)

    q_w = c_phi * c_theta * c_psi + s_phi * s_theta * s_psi
    q_x = s_phi * c_theta * c_psi - c_phi * s_theta * s_psi
    q_y = c_phi * s_theta * c_psi + s_phi * c_theta * s_psi
    q_z = c_phi * c_theta * s_psi - s_phi * s_theta * c_psi

    return np.array([q_w, q_x, q_y, q_z])
    

def quaternion_to_euler(quaternion):
    qw, qx, qy, qz = quaternion

    phi = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))
    theta = np.arcsin(2 * (qw * qy - qz * qx))
    psi = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))

    return np.array([phi, theta, psi])
    

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    # 仿真参数
    T = 5.0
    dt = 0.01
    N = int(T / dt)
    time = np.arange(N) * dt

    # 初始欧拉角和四元数
    euler = np.zeros(3)
    quat = euler_to_quaternion(euler)
    # 恒定角速度
    body_rates = np.array([0.3, 0.1, 0.1])

    # 记录
    euler_traj = np.zeros((N, 3))
    quat_traj = np.zeros((N, 4))
    euler_from_quat_traj = np.zeros((N, 3))
    quat_from_euler_traj = np.zeros((N, 4))

    for i in range(N):
        # 记录当前
        euler_traj[i] = euler
        quat_traj[i] = quat
        # 互相转化
        quat_from_euler = euler_to_quaternion(euler)
        euler_from_quat = quaternion_to_euler(quat)
        quat_from_euler_traj[i] = quat_from_euler
        euler_from_quat_traj[i] = euler_from_quat

        # 步进欧拉角动力学
        euler_dot = euler_kinematics(euler, body_rates)
        euler = euler + euler_dot * dt

        # 步进四元数动力学
        quat_dot = quaternion_kinematics(quat, body_rates)
        quat = quat + quat_dot * dt
        quat = quat / np.linalg.norm(quat)  # 保持单位四元数

    # 画图1：欧拉角动力学 vs 四元数动力学转欧拉角
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(time, euler_traj[:, 0], label="Euler kinematics")
    plt.plot(time, euler_from_quat_traj[:, 0], '--', label="Quaternion kinematics→Euler")
    plt.ylabel("phi (roll)")
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(time, euler_traj[:, 1])
    plt.plot(time, euler_from_quat_traj[:, 1], '--')
    plt.ylabel("theta (pitch)")
    plt.subplot(3, 1, 3)
    plt.plot(time, euler_traj[:, 2])
    plt.plot(time, euler_from_quat_traj[:, 2], '--')
    plt.ylabel("psi (yaw)")
    plt.xlabel("Time [s]")
    plt.suptitle("Euler kinematics vs Quaternion kinematics (converted to Euler angles)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 画图2：四元数动力学 vs 欧拉角动力学转四元数
    plt.figure(figsize=(10, 6))
    plt.subplot(4, 1, 1)
    plt.plot(time, quat_traj[:, 0], label="Quaternion kinematics")
    plt.plot(time, quat_from_euler_traj[:, 0], '--', label="Euler kinematics→Quaternion")
    plt.ylabel("q_w")
    plt.legend()
    plt.subplot(4, 1, 2)
    plt.plot(time, quat_traj[:, 1])
    plt.plot(time, quat_from_euler_traj[:, 1], '--')
    plt.ylabel("q_x")
    plt.subplot(4, 1, 3)
    plt.plot(time, quat_traj[:, 2])
    plt.plot(time, quat_from_euler_traj[:, 2], '--')
    plt.ylabel("q_y")
    plt.subplot(4, 1, 4)
    plt.plot(time, quat_traj[:, 3])
    plt.plot(time, quat_from_euler_traj[:, 3], '--')
    plt.ylabel("q_z")
    plt.xlabel("Time [s]")
    plt.suptitle("Quaternion kinematics vs Euler kinematics (converted to Quaternion)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show()
