from copy import deepcopy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import casadi as ca
from acados_template import AcadosModel, AcadosOcp


class Quadrotor2D(gym.Env):
    metadata = {}
    metadata['nx'] = 6
    metadata['nu'] = 2
    metadata['m'] = 0.5
    metadata['g'] = 9.81
    metadata['r'] = 0.2
    metadata['I'] = 0.02
    metadata['Q'] = np.diag([1, 1, 1, 1, 1, 1])
    metadata['R'] = np.array([[1, 0], [0, 1]])

    def __init__(self, options={}, theoretic_mode=False):
        super().__init__()
        
        # 模型设置
        self.acados_model = NonmialAcadosModel(self.metadata)
        self.system_torch_dynamics = SystemTorchDynamics(self.metadata)
        state_low = np.array([-2, -2, -1.5, -10, -10, -10])
        state_high = np.array([2, 2, 1.5, 10, 10, 10])
        self.observation_space = spaces.Box(low=state_low, high=state_high, shape=(self.metadata['nx'],), dtype=np.float64)
        u_max = 2.5 * self.metadata['m'] * self.metadata['g'] / 2
        self.action_space = spaces.Box(low=0, high=u_max, shape=(self.metadata['nu'],), dtype=np.float64)

        # 在observation_space范围内随机生成一个状态
        self.state = 0.8 * self.np_random.uniform(self.observation_space.low, self.observation_space.high)
        self.state[3:] = 0.0

        # 模型参数
        self.u0 = 0.5 * self.metadata['m'] * self.metadata['g']
        self.A = np.zeros([self.metadata['nx'], self.metadata['nx']])
        self.A[0:3, 3:6] = np.eye(3)
        self.A[3, 2] = -1 / self.metadata['m'] * (self.u0 + self.u0)
        self.B = np.zeros([self.metadata['nx'], self.metadata['nu']])
        self.B[4, :] = 1 / self.metadata['m']
        self.B[5, 0] = self.metadata['r'] / self.metadata['I']
        self.B[5, 1] = -self.metadata['r'] / self.metadata['I']

        # 控制参数（仍然使用单位阵）
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

    def reset(self, seed=None, options={}):
        super().reset(seed=seed)
        # 初始状态
        if options is not None:
            init_state = options.get('init_state')
        else:
            init_state = None
        if init_state is not None:
            self.state = init_state
        else:
            self.state = 0.8 * self.np_random.uniform(self.observation_space.low, self.observation_space.high)
            self.state[1] = np.abs(self.state[1])
            self.state[3:] = 0.0
            # self.state[0] = 0.4
            # self.state[1] = 0.4
            # self.state[2] = 0.0
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
        # 奖励函数（Q, R为单位阵）
        delta_acion = action - self.u0
        reward = -(self.state.T @ self.Q @ self.state + delta_acion.T @ self.R @ delta_acion) * self.dt + 10 * self.dt * (1 - self.theoretic_mode)
        if np.all(np.abs(self.state) < np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])):
            terminated = True
            # terminated = False  # Do not terminate
            reward += 100 * (1 - self.theoretic_mode)
        else:
            terminated = False
        if not self.observation_space.contains(self.state):
            # print("crashed")
            crashed = True
            reward -= 10 * (1 - self.theoretic_mode)
            # self.state = old_state
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
        # state: (6,), action: (2,)
        m = self.metadata['m']
        g = self.metadata['g']
        r = self.metadata['r']
        I = self.metadata['I']

        x1, x2, x3, x4, x5, x6 = state
        u1, u2 = action

        dx1 = x4
        dx2 = x5
        dx3 = x6
        dx4 = - (u1 + u2) * np.sin(x3) / m
        dx5 = ( (u1 + u2) * np.cos(x3) - m * g ) / m
        dx6 = r * (u1 - u2) / I
        return np.array([dx1, dx2, dx3, dx4, dx5, dx6])

    def disturbance(self):
        if self.disturb_mode == 'ConstantBias':
            disturbance = np.array([0.1, 0.1, 0.1, 0.0, 0.0, 0.0])
        elif self.disturb_mode == 'AdditiveNoise':
            # disturbance = self.np_random.uniform(-2, 2, size=self.metadata['nx'])
            # disturbance = self.np_random.normal(0, 0.5, size=self.metadata['nx'])
            disturbance = self.np_random.normal(0, 0.5, size=self.metadata['nx'])
        else:
            disturbance = np.zeros(self.metadata['nx'])
        return disturbance
        
    def disturb_once(self):
        # 非线性系统不再支持对A, B参数的扰动；此处保留接口但不修改系统参数
        if self.has_disturbed == True:
            return
        self.has_disturbed = True
        if self.disturb_mode in ['ParamBias', 'OppositeB']:
            # 对于这些模式，在非线性系统中无操作
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
        # x: (N, 6), u: (N, 2)
        m = self.metadata['m']
        g = self.metadata['g']
        r = self.metadata['r']
        I = self.metadata['I']

        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
        x5 = x[:, 4]
        x6 = x[:, 5]

        u1 = u[:, 0]
        u2 = u[:, 1]

        dx1 = x4
        dx2 = x5
        dx3 = x6
        dx4 = - (u1 + u2) * torch.sin(x3) / m
        dx5 = ((u1 + u2) * torch.cos(x3) - m * g) / m
        dx6 = r * (u1 - u2) / I
        return torch.stack([dx1, dx2, dx3, dx4, dx5, dx6], dim=1)
        
class NonmialAcadosModel():
    def __init__(self, metadata):
        self.metadata = metadata
        self.model = self.export_acados_model()

    def export_acados_model(self):
        nx = self.metadata['nx']
        nu = self.metadata['nu']
        m = self.metadata['m']
        g = self.metadata['g']
        r = self.metadata['r']
        I = self.metadata['I']

        fx = ca.SX.sym('x_dot', nx)
        x = ca.SX.sym('x', nx)
        u = ca.SX.sym('u', nu)
        
        # 非线性动力学（Quadrotor 2D）
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        x4 = x[3]
        x5 = x[4]
        x6 = x[5]

        u1 = u[0]
        u2 = u[1]

        dx1 = x4
        dx2 = x5
        dx3 = x6
        dx4 = - (u1 + u2) * ca.sin(x3) / m
        dx5 = ((u1 + u2) * ca.cos(x3) - m * g) / m
        dx6 = r * (u1 - u2) / I
        x_dot = ca.vertcat(dx1, dx2, dx3, dx4, dx5, dx6)

        model = AcadosModel()
        model.f_expl_expr = x_dot
        model.f_impl_expr = fx - x_dot
        model.x_dot = fx
        model.x = x
        model.u = u
        model.p = []
        model.name = "quadrotor2d"

        return model





