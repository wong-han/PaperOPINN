import sys, os
import time
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import casadi as ca

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from p4_sp_quadrotor3d_TR import setup_casadi_mpc, N

def test_mpc():
    env = gym.make("environment:quadrotor3d_TR-v0")
    
    # 随机初始化
    init_state = np.random.uniform(-1.0, 1.0, 9)
    state, _ = env.reset(theoretic_mode=True, options={'init_state': init_state})
    
    opti, x0_param, u_var = setup_casadi_mpc(env)
    
    u_init = np.zeros((4, N))
    for i in range(N):
        u_init[:, i] = env.u0
        
    x_traj = [state.copy()]
    u_traj = []
    
    episode_length = int(env.T / env.dt)
    
    print("Starting MPC simulation...")
    for step in range(episode_length):
        opti.set_value(x0_param, state)
        opti.set_initial(u_var, u_init)
        
        try:
            sol = opti.solve()
            u_opt = sol.value(u_var[:, 0])
            
            # Warm start
            u_seq = sol.value(u_var)
            if u_seq.ndim == 1:
                u_seq = u_seq.reshape(-1, 1)
            u_init[:, :-1] = u_seq[:, 1:]
            u_init[:, -1] = u_seq[:, -1]
            
        except Exception as e:
            print(f"MPC failed at step {step}")
            break
            
        u_traj.append(u_opt.copy())
        
        state, reward, terminated, truncated, _ = env.step(u_opt)
        x_traj.append(state.copy())
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break

    x_traj = np.array(x_traj)
    u_traj = np.array(u_traj)
    
    if len(u_traj) == 0:
        print("MPC failed immediately.")
        return

    # 绘图
    time_steps = np.arange(len(x_traj)) * env.dt
    time_steps_u = np.arange(len(u_traj)) * env.dt
    
    plt.figure(figsize=(12, 10))
    
    # 位置
    plt.subplot(3, 1, 1)
    plt.plot(time_steps, x_traj[:, 0], label='x')
    plt.plot(time_steps, x_traj[:, 1], label='y')
    plt.plot(time_steps, x_traj[:, 2], label='z')
    plt.title('Position (m)')
    plt.legend()
    plt.grid(True)
    
    # 姿态
    plt.subplot(3, 1, 2)
    plt.plot(time_steps, x_traj[:, 6], label='phi')
    plt.plot(time_steps, x_traj[:, 7], label='theta')
    plt.plot(time_steps, x_traj[:, 8], label='psi')
    plt.title('Euler Angles (rad)')
    plt.legend()
    plt.grid(True)
    
    # 控制输入
    plt.subplot(3, 1, 3)
    plt.plot(time_steps_u, u_traj[:, 0], label='Thrust (T)')
    plt.plot(time_steps_u, u_traj[:, 1], label='p')
    plt.plot(time_steps_u, u_traj[:, 2], label='q')
    plt.plot(time_steps_u, u_traj[:, 3], label='r')
    plt.title('Control Inputs')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_mpc()
