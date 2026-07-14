from copy import deepcopy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import casadi as ca
from acados_template import AcadosModel
import scipy

class Pendulum(gym.Env):
    metadata = {}
    metadata['nx'] = 2
    metadata['nu'] = 1
    metadata['m'] = 1
    metadata['length'] = 0.5
    metadata['beta'] = 0.0
    metadata['g'] = 9.81

    metadata['Q'] = np.diag([1, 1])
    metadata['R'] = np.diag([1])

    def __init__(self, options={}, theoretic_mode=False):
        super().__init__()

        # 模型设置
        self.acados_model = NonmialAcadosModel(self.metadata)
        self.system_torch_dynamics = SystemTorchDynamics(self.metadata)
        # 状态空间: [theta, theta_dot]
        self.state_low = np.array([-np.pi, -80])
        self.state_high = np.array([np.pi, 80])
        self.observation_space = spaces.Box(low=self.state_low, high=self.state_high, shape=(self.metadata['nx'],), dtype=np.float64)
        # 动作空间: [u]
        u_max = 1.8
        self.u_min = np.array([-u_max])
        self.u_max = np.array([u_max])
        self.action_space = spaces.Box(low=self.u_min, high=self.u_max, shape=(self.metadata['nu'],), dtype=np.float64)

        # 在observation_space范围内随机生成一个状态
        self.state = 1.0 * self.np_random.uniform(self.observation_space.low, self.observation_space.high)
        self.state[1] = 0.0  # 初始角速度为0

        # 模型参数
        self.A, self.B = linearize_at_origin(self.acados_model)

        # 控制参数
        self.u0 = np.array([0.0])
        control_param = options.get('control_param')
        if control_param is None:
            self.Q = self.metadata['Q']
            self.R = self.metadata['R']
        else:
            self.Q = control_param['Q']
            self.R = control_param['R']

        self.P = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)

        # 仿真参数
        self.dt = 0.001
        self.t = 0.0
        self.data = {}
        self.T = 5.0  # 最大运行时间
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
            self.state = 1.0 * self.np_random.uniform(self.observation_space.low, self.observation_space.high)
            self.state[1] = 0.0
            # self.state[0] = 0.3
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
        # 状态更新（倒立摆非线性系统）
        old_state = deepcopy(self.state)
        dxdt = self._nonlinear_dynamics_np(self.state, action) + self.disturbance()
        self.state = self.state + dxdt * self.dt
        # 保证角度theta在[-pi, pi]范围内
        self.state[0] = (self.state[0] + np.pi) % (2 * np.pi) - np.pi
        self.t = self.t + self.dt
        # 奖励函数
        delta_action = action - self.u0
        reward = -(self.state.T @ self.Q @ self.state + delta_action.T @ self.R @ delta_action) * self.dt + 100 * self.dt * (1 - self.theoretic_mode)
        if np.all(np.abs(self.state) < np.array([0.01, 0.01])):
            terminated = True
            reward += 100 * (1 - self.theoretic_mode)
        else:
            terminated = False
        if not self.observation_space.contains(self.state):
            crashed = True
            reward -= 100 * (1 - self.theoretic_mode)
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


    def _nonlinear_dynamics_np(self, state, action):
        # state: (2,), action: (1,)
        m = self.metadata['m']
        l = self.metadata['length']
        beta = self.metadata['beta']
        g = self.metadata['g']

        theta = state[0]
        theta_dot = state[1]
        u = action[0]

        dtheta = theta_dot
        dtheta_dot = (m * g * l * np.sin(theta) - beta * theta_dot + u) / (m * l ** 2)

        dxdt = np.array([dtheta, dtheta_dot])
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
        # x: (N, 2), u: (N, 1)
        m = self.metadata['m']
        l = self.metadata['length']
        beta = self.metadata['beta']
        g = self.metadata['g']

        theta = x[:, 0]
        theta_dot = x[:, 1]
        u_val = u[:, 0]

        dtheta = theta_dot
        dtheta_dot = (m * g * l * torch.sin(theta) - beta * theta_dot + u_val) / (m * l ** 2)

        dxdt = torch.stack([dtheta, dtheta_dot], dim=1)
        return dxdt

class NonmialAcadosModel():
    def __init__(self, metadata):
        self.metadata = metadata
        self.model = self.export_acados_model()

    def export_acados_model(self):
        nx = self.metadata['nx']
        nu = self.metadata['nu']
        m = self.metadata['m']
        l = self.metadata['length']
        beta = self.metadata['beta']
        g = self.metadata['g']

        fx = ca.SX.sym('x_dot', nx)
        x = ca.SX.sym('x', nx)
        u = ca.SX.sym('u', nu)

        theta = x[0]
        theta_dot = x[1]
        u_val = u[0]

        dtheta = theta_dot
        dtheta_dot = (m * g * l * ca.sin(theta) - beta * theta_dot + u_val) / (m * l ** 2)

        x_dot = ca.vertcat(dtheta, dtheta_dot)

        model = AcadosModel()
        model.f_expl_expr = x_dot
        model.f_impl_expr = fx - x_dot
        model.x_dot = fx
        model.x = x
        model.u = u
        model.p = []
        model.name = "pendulum"

        return model


def linearize_at_origin(acados_model):
    """
    计算动力学模型在零点（x=0, u=0）的线性化矩阵A和B
    :param model: AcadosModel对象
    :return: 数值矩阵A, B
    """
    model = acados_model.model
    # 提取符号变量和表达式
    x = model.x
    u = model.u
    f_expl = model.f_expl_expr  # 显式动力学: dx/dt = f_expl(x, u)

    # 计算雅可比矩阵 (符号形式)
    A_sym = ca.jacobian(f_expl, x)  # A = df/dx
    B_sym = ca.jacobian(f_expl, u)  # B = df/du

    # 创建函数用于数值计算
    A_func = ca.Function('A_func', [x, u], [A_sym])
    B_func = ca.Function('B_func', [x, u], [B_sym])

    # 在零点求值 (x=0, u=0)
    x0 = np.zeros(model.x.size()[0])
    u0 = np.zeros(model.u.size()[0])
    A0 = A_func(x0, u0).full()  # 转换为NumPy数组
    B0 = B_func(x0, u0).full()

    return A0, B0
