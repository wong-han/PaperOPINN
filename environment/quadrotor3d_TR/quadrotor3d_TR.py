from copy import deepcopy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import sympy as sp
import scipy


class Quadrotor3D_TR(gym.Env):
    metadata = {}
    metadata['nx'] = 9
    metadata['nu'] = 4
    metadata['m'] = 0.027
    metadata['g'] = 9.81
    metadata['Q'] = np.diag([5, 5, 5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    metadata['R'] = np.diag([50, 1, 1, 1])


    def __init__(self, options={}, theoretic_mode=False, rl_mode=False):
        super().__init__()

        # 模型设置
        self.system_torch_dynamics = SystemTorchDynamics(self.metadata)
        # 状态空间: [p_x, p_y, p_z, v_x, v_y, v_z, phi, theta, psi]
        self.state_low = np.array([-1.5, -1.5, -1.5, -10, -10, -10, -np.pi/3, -np.pi/3, -np.pi/2])
        self.state_high = np.array([1.5, 1.5, 1.5, 10, 10, 10, np.pi/3, np.pi/3, np.pi/2])
        self.observation_space = spaces.Box(low=self.state_low, high=self.state_high, shape=(self.metadata['nx'],), dtype=np.float64)
        # 动作空间: [推力, p, q, r] (推力+角速度)
        omega_max = np.pi
        self.u_min = np.array([0.0, -omega_max, -omega_max, -omega_max])
        self.u_max = np.array([self.metadata['m']*self.metadata['g']*2, omega_max, omega_max, omega_max])
        self.action_space = spaces.Box(low=self.u_min, high=self.u_max, shape=(self.metadata['nu'],), dtype=np.float64)

        if rl_mode:
            self.state = np.random.uniform(-1, 1, 9)
        else:
            # 在observation_space范围内随机生成一个状态
            self.state = 1.0 * self.np_random.uniform(self.observation_space.low, self.observation_space.high)
            self.state[3:6] = 0.0  # 初始速度为0
            self.state[6:9] = 0.0  # 初始欧拉角为0
        self.rl_mode = rl_mode

        # 模型参数
        self.A, self.B = self._linearize_at_origin()

        # 控制参数
        self.u0 = np.array([self.metadata['m']*self.metadata['g'], 0.0, 0.0, 0.0])
        control_param = options.get('control_param')
        if control_param is None:
            self.Q = self.metadata['Q']
            self.R = self.metadata['R']
        else:
            self.Q = control_param['Q']
            self.R = control_param['R']

        self.P = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)

        # 仿真参数
        self.dt = 0.01
        self.t = 0.0
        self.data = {}
        self.T = 8.0  # 最大运行时间
        self.theoretic_mode = theoretic_mode

        # 扰动设置
        self.has_disturbed = False
        disturb_param = options.get('disturb_param')
        if disturb_param is not None:
            self.disturb_mode = disturb_param['disturb_mode']
            self.disturb_once()
        else:
            self.disturb_mode = None

    def _get_obs(self):
        return self.state

    def _get_info(self):
        info = {}
        info['data'] = deepcopy(self.data)
        return info

    def reset(self, seed=None, positive_y=False, options=None, theoretic_mode=None, rl_mode=None):
        super().reset(seed=seed)
        np.random.seed(seed)

        # Preserve the mode selected when the environment was constructed unless
        # the caller explicitly requests a different mode for this and later episodes.
        if theoretic_mode is not None:
            self.theoretic_mode = theoretic_mode
        if rl_mode is not None:
            self.rl_mode = rl_mode

        # 初始状态
        if options is not None:
            init_state = options.get('init_state')
        else:
            init_state = None
        if init_state is not None:
            self.state = init_state
        else:
            self.state = np.random.uniform(-1, 1, 9)
                
        self.t = 0.0
        self.data = {}
        # 控制参数
        if options is not None:
            control_param = options.get('control_param')
        else:
            control_param = None
        if control_param is not None:
            self.Q = control_param['Q']
            self.R = control_param['R']

        # 扰动设置
        if options is not None:
            self.disturb_param = options.get('disturb_param')
        else:
            self.disturb_param = None
        if self.disturb_param is not None:
            self.disturb_mode = self.disturb_param['disturb_mode']
            self.disturb_once()
        else:
            self.disturb_mode = None

        return self._get_obs(), self._get_info()

    def step(self, action):
        # 将 action 限制在 action_space 的范围内
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # 状态更新（使用无人机3D非线性系统）
        old_state = deepcopy(self.state)
        dxdt = self._nonlinear_dynamics_np(self.state, action) + self.disturbance()
        self.state = self.state + dxdt * self.dt
        self.t = self.t + self.dt
        # 奖励函数
        delta_action = action - self.u0
        reward = -(self.state.T @ self.Q @ self.state + delta_action.T @ self.R @ delta_action) * self.dt + 10 * self.dt * (1 - self.theoretic_mode)
        if np.all(np.abs(self.state) < np.array([0.2]*9)):
            terminated = True
            state_norm = np.linalg.norm(self.state)
            reward += 10 * (1 - self.theoretic_mode) + 50 * (1 - state_norm) * (1 - self.theoretic_mode)
        else:
            terminated = False
        if not self.observation_space.contains(self.state):
            crashed = True
            reward -= 10 * (1 - self.theoretic_mode)
        else:
            crashed = False
        if self.t > self.T or crashed:
            truncated = True
        else:
            truncated = False
            
        # 信息
        if 'state' not in self.data:
            self.data['state'] = [deepcopy(old_state)]
        else:
            self.data['state'].append(deepcopy(old_state))
        if 'action' not in self.data:
            self.data['action'] = [deepcopy(action)]
        else:
            self.data['action'].append(deepcopy(action))
        if 'state_dot' not in self.data:
            self.data['state_dot'] = [deepcopy(dxdt)]
        else:
            self.data['state_dot'].append(deepcopy(dxdt))
        if 'crashed' not in self.data:
            self.data['crashed'] = [crashed]
        else:
            self.data['crashed'].append(crashed)
        info = self._get_info()

        return self.state, reward, terminated, truncated, info

    def _linearize_at_origin(self):
        """
        利用SymPy进行符号表达式计算，并在平衡点处求取雅可比矩阵线性化
        """
        m = self.metadata['m']
        g = self.metadata['g']

        px, py, pz, vx, vy, vz, phi, theta, psi = sp.symbols(
            'px py pz vx vy vz phi theta psi'
        )
        T, p, q, r = sp.symbols('T p q r')

        x = sp.Matrix([px, py, pz, vx, vy, vz, phi, theta, psi])
        u = sp.Matrix([T, p, q, r])

        cphi = sp.cos(phi)
        sphi = sp.sin(phi)
        cth = sp.cos(theta)
        sth = sp.sin(theta)
        cpsi = sp.cos(psi)
        spsi = sp.sin(psi)
        R = sp.Matrix([
            [cth * cpsi, sphi * sth * cpsi - cphi * spsi, cphi * sth * cpsi + sphi * spsi],
            [cth * spsi, sphi * sth * spsi + cphi * cpsi, cphi * sth * spsi - sphi * cpsi],
            [-sth,       sphi * cth,                      cphi * cth]
        ])

        dpos = sp.Matrix([vx, vy, vz])
        dv = (1.0 / m) * (R * sp.Matrix([0, 0, T])) - sp.Matrix([0, 0, g])

        tan_theta = sp.tan(theta)
        sec_theta = 1.0 / sp.cos(theta)
        euler_dot = sp.Matrix([
            p + sphi * tan_theta * q + cphi * tan_theta * r,
            cphi * q - sphi * r,
            sphi * sec_theta * q + cphi * sec_theta * r
        ])

        f = sp.Matrix([
            dpos[0], dpos[1], dpos[2],
            dv[0], dv[1], dv[2],
            euler_dot[0], euler_dot[1], euler_dot[2]
        ])

        A_sym = f.jacobian(x)
        B_sym = f.jacobian(u)

        subs_dict = {
            px: 0.0, py: 0.0, pz: 0.0,
            vx: 0.0, vy: 0.0, vz: 0.0,
            phi: 0.0, theta: 0.0, psi: 0.0,
            T: m * g, p: 0.0, q: 0.0, r: 0.0
        }

        A = np.array(A_sym.subs(subs_dict), dtype=np.float64)
        B = np.array(B_sym.subs(subs_dict), dtype=np.float64)

        return A, B

    def _nonlinear_dynamics_np(self, state, action):
        # state: (9,), action: (4,)  action = [T, p, q, r]
        m = self.metadata['m']
        g = self.metadata['g']

        # 状态变量
        px, py, pz = state[0:3]
        vx, vy, vz = state[3:6]
        phi, theta, psi = state[6:9]

        # 动作
        T, p, q, r = action

        # 旋转矩阵
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        cth = np.cos(theta)
        sth = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)
        R = np.array([
            [cth * cpsi, sphi * sth * cpsi - cphi * spsi, cphi * sth * cpsi + sphi * spsi],
            [cth * spsi, sphi * sth * spsi + cphi * cpsi, cphi * sth * spsi - sphi * cpsi],
            [-sth,       sphi * cth,                      cphi * cth]
        ])

        # 位置微分
        dpos = np.array([vx, vy, vz])
        # 速度微分
        dv = (1.0 / m) * (R @ np.array([0, 0, T])) - np.array([0, 0, g])

        # 欧拉角微分
        tan_theta = np.tan(theta) if np.abs(np.cos(theta)) > 1e-6 else 0.0
        sec_theta = 1.0 / np.cos(theta) if np.abs(np.cos(theta)) > 1e-6 else 0.0
        euler_dot = np.array([
            p + sphi * tan_theta * q + cphi * tan_theta * r,
            cphi * q - sphi * r,
            sphi * sec_theta * q + cphi * sec_theta * r
        ])

        dxdt = np.zeros(9)
        dxdt[0:3] = dpos
        dxdt[3:6] = dv
        dxdt[6:9] = euler_dot
        return dxdt

    def disturbance(self):
        if self.disturb_mode == 'ConstantBias':
            disturbance = np.array([0.1]*self.metadata['nx'])
        elif self.disturb_mode == 'AdditiveNoise':
            disturbance = self.np_random.normal(0, 0.5, size=self.metadata['nx'])
        else:
            disturbance = np.zeros(self.metadata['nx'])
        return disturbance

    def disturb_once(self):
        if self.has_disturbed == True:
            return
        self.has_disturbed = True
        if self.disturb_mode in ['ParamBias', 'OppositeB']:
            pass

    def render(self):
        pass

    def _render_frame(self):
        pass

    def close(self):
        pass


class SystemTorchDynamics(torch.nn.Module):
    def __init__(self, metadata):
        self.metadata = metadata
        super().__init__()
        self.nx = metadata['nx']
        self.nu = metadata['nu']

    def forward(self, x, u):
        # x: (N, 9), u: (N, 4)  u = [T, p, q, r]
        m = self.metadata['m']
        g = self.metadata['g']

        px = x[:, 0]
        py = x[:, 1]
        pz = x[:, 2]
        vx = x[:, 3]
        vy = x[:, 4]
        vz = x[:, 5]
        phi = x[:, 6]
        theta = x[:, 7]
        psi = x[:, 8]

        T = u[:, 0]
        p = u[:, 1]
        q = u[:, 2]
        r = u[:, 3]

        cphi = torch.cos(phi)
        sphi = torch.sin(phi)
        cth = torch.cos(theta)
        sth = torch.sin(theta)
        cpsi = torch.cos(psi)
        spsi = torch.sin(psi)

        # 旋转矩阵
        R11 = cth * cpsi
        R12 = sphi * sth * cpsi - cphi * spsi
        R13 = cphi * sth * cpsi + sphi * spsi
        R21 = cth * spsi
        R22 = sphi * sth * spsi + cphi * cpsi
        R23 = cphi * sth * spsi - sphi * cpsi
        R31 = -sth
        R32 = sphi * cth
        R33 = cphi * cth

        # 位置微分
        dpos = torch.stack([vx, vy, vz], dim=1)
        # 速度微分
        dv = torch.stack([
            (R13 * T) / m,
            (R23 * T) / m,
            (R33 * T) / m - g
        ], dim=1)

        # 欧拉角微分
        tan_theta = torch.tan(theta)
        sec_theta = 1.0 / torch.cos(theta)
        euler_dot = torch.stack([
            p + sphi * tan_theta * q + cphi * tan_theta * r,
            cphi * q - sphi * r,
            sphi * sec_theta * q + cphi * sec_theta * r
        ], dim=1)

        dxdt = torch.cat([
            dpos, dv, euler_dot
        ], dim=1)
        return dxdt


