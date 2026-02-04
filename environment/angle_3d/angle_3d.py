from copy import deepcopy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import casadi as ca
from acados_template import AcadosModel, AcadosOcp


class Angle3D(gym.Env):
    metadata = {}
    metadata['nx'] = 3
    metadata['nu'] = 3
    metadata['Q'] = np.diag([1, 1, 1])
    metadata['R'] = np.diag([1, 1, 1])

    def __init__(self, options={}, theoretic_mode=False):
        super().__init__()
        
        # 模型设置
        self.acados_model = NonmialAcadosModel(self.metadata)
        self.system_torch_dynamics = SystemTorchDynamics(self.metadata)
        self.observation_space = spaces.Box(low=-1.5, high=1.5, shape=(self.metadata['nx'],), dtype=np.float64)
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(self.metadata['nu'],), dtype=np.float64)

        # 在observation_space范围内随机生成一个状态
        self.state = 0.9 * self.np_random.uniform(self.observation_space.low, self.observation_space.high)

        # 控制参数（仍然使用单位阵，或按外部传入）
        control_param = options.get('control_param')
        if control_param is None:
            self.Q = self.metadata['Q']
            self.R = self.metadata['R']
        else:
            self.Q = control_param['Q']
            self.R = control_param['R']

        # 仿真参数
        self.dt = 0.02
        self.t = 0.0
        self.data = {}
        self.T = 10.0  # 最大运行时间
        self.theoretic_mode = theoretic_mode  # 理论模式：仅计算∫(x^TQx+u^TRu)dt性能指标

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

    def reset(self, seed=None, options={}, theoretic_mode=False):
        super().reset(seed=seed)
        self.state = 0.9 * self.np_random.uniform(self.observation_space.low, self.observation_space.high)
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
        
        # 仿真设置
        self.theoretic_mode = theoretic_mode
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        # 将 action 限制在 action_space 的范围内
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # 状态更新（使用三维角度非线性系统）
        old_state = deepcopy(self.state)
        dxdt = self._angle3d_dynamics_np(self.state, action) + self.disturbance()
        self.state = self.state + dxdt * self.dt
        self.t = self.t + self.dt
        # 奖励函数（Q, R为单位阵）
        reward = -(self.state.T @ self.Q @ self.state + action.T @ self.R @ action) * self.dt + 10 * self.dt * (1 - self.theoretic_mode)
        if np.all(np.abs(self.state) < np.array([0.01, 0.01, 0.01])):
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

    def _angle3d_dynamics_np(self, state, action):
        # state: (3,) -> [phi, theta, psi]; action: (3,) -> [p, q, r]
        phi = state[0]
        theta = state[1]
        # psi = state[2]  # 未直接出现在右端
        p = action[0]
        q = action[1]
        r = action[2]
        # 避免cos(theta)=0导致除零
        cos_theta = np.cos(theta)
        cos_theta = np.clip(cos_theta, 1e-6, None)
        tan_theta = np.tan(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        dphi = 1.0 * p + tan_theta * sin_phi * q + tan_theta * cos_phi * r
        dtheta = 0.0 * p + cos_phi * q + (-sin_phi) * r
        dpsi = 0.0 * p + (sin_phi / cos_theta) * q + (cos_phi / cos_theta) * r
        return np.array([dphi, dtheta, dpsi])

    def disturbance(self):
        if self.disturb_mode == 'ConstantBias':
            disturbance = np.array([0.05, 0.05, 0.05])
        elif self.disturb_mode == 'AdditiveNoise':
            disturbance = self.np_random.normal(0, 0.05, size=self.metadata['nx'])
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
        # x: (N, 3), u: (N, 3)
        phi = x[:, 0]
        theta = x[:, 1]
        # psi = x[:, 2]
        p = u[:, 0]
        q = u[:, 1]
        r = u[:, 2]
        cos_theta = torch.cos(theta)
        cos_theta = torch.clamp(cos_theta, min=1e-6)
        tan_theta = torch.tan(theta)
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)
        dphi = p + tan_theta * sin_phi * q + tan_theta * cos_phi * r
        dtheta = cos_phi * q - sin_phi * r
        dpsi = (sin_phi / cos_theta) * q + (cos_phi / cos_theta) * r
        return torch.stack([dphi, dtheta, dpsi], dim=1)
        
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
        
        # x = [phi, theta, psi], u = [p, q, r]
        phi = x[0]
        theta = x[1]
        # psi = x[2]
        p = u[0]
        q = u[1]
        r = u[2]
        cos_theta = ca.cos(theta)
        # 为了数值稳定性，避免直接除以cos_theta==0，可在外部设置约束；此处直接使用表达式
        tan_theta = ca.tan(theta)
        sin_phi = ca.sin(phi)
        cos_phi = ca.cos(phi)
        dphi = p + tan_theta * sin_phi * q + tan_theta * cos_phi * r
        dtheta = cos_phi * q - sin_phi * r
        dpsi = (sin_phi / cos_theta) * q + (cos_phi / cos_theta) * r
        x_dot = ca.vertcat(dphi, dtheta, dpsi)

        model = AcadosModel()
        model.f_expl_expr = x_dot
        model.f_impl_expr = fx - x_dot
        model.x_dot = fx
        model.x = x
        model.u = u
        model.p = []
        model.name = "angle_3d"

        return model