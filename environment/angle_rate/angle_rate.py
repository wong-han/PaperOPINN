from copy import deepcopy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import casadi as ca
from acados_template import AcadosModel


class AngleRate(gym.Env):
    metadata = {}
    metadata['nx'] = 6
    metadata['nu'] = 3
    metadata['Ixx'] = 0.3
    metadata['Iyy'] = 0.6
    metadata['Izz'] = 1
    # metadata['Q'] = np.diag([300, 300, 300, 100, 100, 100])
    metadata['Q'] = np.diag([1, 1, 1, 1, 1, 1])
    metadata['R'] = np.diag([1, 1, 1])

    def __init__(self, options={}, theoretic_mode=False):
        super().__init__()
        
        # 模型设置
        self.acados_model = NonmialAcadosModel(self.metadata)
        self.system_torch_dynamics = SystemTorchDynamics(self.metadata)

        # 状态范围：角度[-pi, pi]，角速度[-5, 5]
        state_low = np.array([-1.5, -1.5, -1.5, -10, -10, -10])
        state_high = np.array([1.5, 1.5, 1.5, 10, 10, 10])
        self.observation_space = spaces.Box(
            low=state_low, high=state_high, shape=(self.metadata['nx'],), dtype=np.float64
        )

        # 控制输入范围：力矩[-2, 2]
        self.action_space = spaces.Box(
            low=-10, high=10, shape=(self.metadata['nu'],), dtype=np.float64
        )

        # 初始化状态
        self.state = 1.0 * self.np_random.uniform(
            self.observation_space.low, self.observation_space.high
        )
        self.state[3:] = 0.0  # 初始角速度设为 0

        # 模型参数
        self.A = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
        ])

        self.B = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1 / self.metadata['Ixx'], 0, 0],
        [0, 1 / self.metadata['Iyy'], 0],
        [0, 0, 1 / self.metadata['Izz']]
        ])

        # 控制参数
        control_param = options.get('control_param')
        if control_param is None:
            self.Q = self.metadata['Q']
            self.R = self.metadata['R']
        else:
            self.Q = control_param['Q']
            self.R = control_param['R']

        # 仿真参数
        self.dt = 0.01
        self.t = 0.0
        self.data = {}
        self.T = 10  # 最大运行时间
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

    def reset(self, seed=None, options={}):
        super().reset(seed=seed)
        self.state = 1.0 * self.np_random.uniform(
            self.observation_space.low, self.observation_space.high
        )
        self.state[3:] = 0.0
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
        # 状态更新（使用非线性系统）
        old_state = deepcopy(self.state)
        dxdt = self._nonlinear_dynamics_np(self.state, action) + self.disturbance()
        self.state = self.state + dxdt * self.dt
        self.t = self.t + self.dt
        # 奖励函数
        reward = -(self.state.T @ self.Q @ self.state + action.T @ self.R @ action) * self.dt + 10 * self.dt * (1 - self.theoretic_mode)
        if np.all(np.abs(self.state) < np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])):
            terminated = True
            reward += 100 * (1 - self.theoretic_mode)
        else:
            terminated = False
        if not self.observation_space.contains(self.state):
            crashed = True
            reward -= 10 * (1 - self.theoretic_mode)
        else:
            crashed = False
        if self.t > self.T:
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

    def _nonlinear_dynamics_np(self, state, action):
        # state: (6,), action: (3,)
        phi, theta, psi, p, q, r = state
        tau_x, tau_y, tau_z = action

        # 角速度到角速度的映射矩阵
        W = np.array([
            [1, np.tan(theta) * np.sin(phi), np.tan(theta) * np.cos(phi)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]
        ])

        ang_vel = np.array([p, q, r])
        d_angles = W @ ang_vel

        # 欧拉角速度方程 (I=1)
        dp = tau_x
        dq = tau_y
        dr = tau_z

        return np.array([d_angles[0], d_angles[1], d_angles[2], dp, dq, dr])

    def disturbance(self):
        if self.disturb_mode == 'ConstantBias':
            disturbance = np.array([0.1, 0.1, 0.1, 0.0, 0.0, 0.0])
        elif self.disturb_mode == 'AdditiveNoise':
            disturbance = self.np_random.normal(0, 0.05, size=self.metadata['nx'])
        else:
            disturbance = np.zeros(self.metadata['nx'])
        return disturbance
        
    def disturb_once(self):
        if self.has_disturbed:
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
        super().__init__()
        self.metadata = metadata
        self.nx = metadata['nx']
        self.nu = metadata['nu']

    def forward(self, x, u):
        # x: (N, 6), u: (N, 3)
        phi = x[:, 0]
        theta = x[:, 1]
        psi = x[:, 2]
        p = x[:, 3]
        q = x[:, 4]
        r = x[:, 5]

        tau_x = u[:, 0]
        tau_y = u[:, 1]
        tau_z = u[:, 2]

        dphi = p + torch.tan(theta) * (torch.sin(phi) * q + torch.cos(phi) * r)
        dtheta = torch.cos(phi) * q - torch.sin(phi) * r
        dpsi = (torch.sin(phi) / torch.cos(theta)) * q + (torch.cos(phi) / torch.cos(theta)) * r

        dp = tau_x
        dq = tau_y
        dr = tau_z

        return torch.stack([dphi, dtheta, dpsi, dp, dq, dr], dim=1)


class NonmialAcadosModel():
    def __init__(self, metadata):
        self.metadata = metadata
        self.model = self.export_acados_model()

    def export_acados_model(self):
        nx = self.metadata['nx']
        nu = self.metadata['nu']

        fx = ca.SX.sym('x_dot', nx)
        x = ca.SX.sym('x', nx)
        u = ca.SX.sym('u', nu)

        phi, theta, psi, p, q, r = x[0], x[1], x[2], x[3], x[4], x[5]
        tau_x, tau_y, tau_z = u[0], u[1], u[2]

        dphi = p + ca.tan(theta) * (ca.sin(phi) * q + ca.cos(phi) * r)
        dtheta = ca.cos(phi) * q - ca.sin(phi) * r
        dpsi = (ca.sin(phi) / ca.cos(theta)) * q + (ca.cos(phi) / ca.cos(theta)) * r

        dp = tau_x
        dq = tau_y
        dr = tau_z

        x_dot = ca.vertcat(dphi, dtheta, dpsi, dp, dq, dr)

        model = AcadosModel()
        model.f_expl_expr = x_dot
        model.f_impl_expr = fx - x_dot
        model.x_dot = fx
        model.x = x
        model.u = u
        model.p = []
        model.name = "attitude_control"

        return model
