from copy import deepcopy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import casadi as ca
from acados_template import AcadosModel, AcadosOcp


class Nonlinear2D(gym.Env):
    metadata = {}
    metadata['nx'] = 2
    metadata['nu'] = 1
    metadata['a1'] = 0.0
    metadata['a2'] = 1.0
    metadata['a3'] = 0.0
    metadata['a4'] = 0.0
    metadata['b1'] = 0.0
    metadata['b2'] = 1.0
    metadata['Q'] = np.diag([1, 1])
    metadata['R'] = np.diag([1])

    def __init__(self, options={}, theoretic_mode=False):
        super().__init__()
        
        # 模型设置
        self.acados_model = NonmialAcadosModel(self.metadata)
        self.system_torch_dynamics = SystemTorchDynamics(self.metadata)
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(self.metadata['nx'],), dtype=np.float64)
        self.action_space = spaces.Box(low=-10, high=10, shape=(self.metadata['nu'],), dtype=np.float64)

        # 在observation_space范围内随机生成一个状态
        self.state = 0.6 * self.np_random.uniform(self.observation_space.low, self.observation_space.high)

        # 模型参数（线性参数不再使用，保留兼容）
        model_param = options.get('model_param')
        if model_param is None:
            a1 = self.metadata['a1']
            a2 = self.metadata['a2']
            a3 = self.metadata['a3']
            a4 = self.metadata['a4']
            b1 = self.metadata['b1']
            b2 = self.metadata['b2']
        else:
            a1 = model_param['a1']
            a2 = model_param['a2']
            a3 = model_param['a3']
            a4 = model_param['a4']
            b1 = model_param['b1']
            b2 = model_param['b2']
        
        self.A = np.array([
            [a1, a2],
            [a3, a4]
        ])
        self.A_nominal = self.A

        self.B = np.array([
            [b1],
            [b2]
        ])
        self.B_nominal = self.B

        # 控制参数（仍然使用单位阵）
        control_param = options.get('control_param')
        if control_param is None:
            self.Q = self.metadata['Q']
            self.R = self.metadata['R']
        else:
            self.Q = control_param['Q']
            self.R = control_param['R']

        # 仿真参数
        self.dt = 0.05
        self.t = 0.0
        self.data = {}
        self.T = 10.0  # 最大运行时间
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
        self.state = 0.6 * self.np_random.uniform(self.observation_space.low, self.observation_space.high)
        self.t = 0.0
        self.state = np.array([0.9, 2])
        self.data = {}
        # 模型参数（线性参数不再使用，保留兼容）
        if options is not None:
            model_param = options.get('model_param')
        else:
            model_param = None
        if model_param is not None:
            a1 = model_param['a1']
            a2 = model_param['a2']
            a3 = model_param['a3']
            a4 = model_param['a4']
            b1 = model_param['b1']
            b2 = model_param['b2']
            self.A = np.array([
                [a1, a2],
                [a3, a4]
            ])
            self.B = np.array([
                [b1],
                [b2]
            ])
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
        reward = -(self.state.T @ self.Q @ self.state + action.T @ self.R @ action) * self.dt + 10 * self.dt * (1 - self.theoretic_mode)
        if np.all(np.abs(self.state) < np.array([0.01, 0.01])):
            terminated = True
            # terminated = False  # Do not terminate
            reward += 100 * (1 - self.theoretic_mode)
        else:
            terminated = False
        if not self.observation_space.contains(self.state):
            # print("crashed")
            crashed = True
            reward -= 10 * (1 - self.theoretic_mode)
            self.state = old_state
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
        # state: (2,), action: (1,) or scalar array
        x1 = state[0]
        x2 = state[1]
        u = action[0] if np.ndim(action) > 0 else action
        c = np.cos(2 * x1) + 2.0
        dx1 = -x1 + x2
        dx2 = -0.5 * x1 - 0.5 * x2 * (1.0 - c**2) + c * u
        return np.array([dx1, dx2])

    def disturbance(self):
        if self.disturb_mode == 'ConstantBias':
            disturbance = np.array([0.2, 0.3])
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
        # x: (N, 2), u: (N, 1)
        x1 = x[:, 0]
        x2 = x[:, 1]
        u_scalar = u[:, 0]
        c = torch.cos(2.0 * x1) + 2.0
        dx1 = -x1 + x2
        dx2 = -0.5 * x1 - 0.5 * x2 * (1.0 - c**2) + c * u_scalar
        return torch.stack([dx1, dx2], dim=1)
        
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
        
        # 非线性动力学
        x1 = x[0]
        x2 = x[1]
        c = ca.cos(2 * x1) + 2
        dx1 = -x1 + x2
        dx2 = -0.5 * x1 - 0.5 * x2 * (1 - c**2) + c * u[0]
        x_dot = ca.vertcat(dx1, dx2)

        model = AcadosModel()
        model.f_expl_expr = x_dot
        model.f_impl_expr = fx - x_dot
        model.x_dot = fx
        model.x = x
        model.u = u
        model.p = []
        model.name = "nonlinear_2d"

        return model




