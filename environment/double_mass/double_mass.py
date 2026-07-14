from copy import deepcopy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import casadi as ca
from acados_template import AcadosModel, AcadosOcp

class DoubleMass(gym.Env):
    metadata = {}
    metadata['nx'] = 4
    metadata['nu'] = 2
    metadata['a1'] = 0.1
    metadata['a2'] = 0.2
    metadata['a3'] = 0.3
    metadata['beta1'] = 0.1
    metadata['beta2'] = 0.2
    metadata['b1'] = 1.0
    metadata['b2'] = 2.0
    metadata['Q'] = np.diag([1, 1, 0.1, 0.1])
    metadata['R'] = np.diag([0.1, 0.1])

    def __init__(self, options={}):
        super().__init__()
        
        # 模型设置
        self.acados_model = NonmialAcadosModel(self.metadata)
        self.system_torch_dynamics = SystemTorchDynamics(self.metadata)
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(self.metadata['nx'],), dtype=np.float64)
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(self.metadata['nu'],), dtype=np.float64)

        # 在observation_space范围内随机生成一个状态
        self.state = self.np_random.uniform(self.observation_space.low, self.observation_space.high) * 0.5

        # 模型参数
        model_param = options.get('model_param')
        if model_param is None:
            a1 = self.metadata['a1']
            a2 = self.metadata['a2']
            a3 = self.metadata['a3']
            beta1 = self.metadata['beta1']
            beta2 = self.metadata['beta2']
            b1 = self.metadata['b1']
            b2 = self.metadata['b2']
        else:
            a1 = model_param['a1']
            a2 = model_param['a2']
            a3 = model_param['a3']
            beta1 = model_param['beta1']
            beta2 = model_param['beta2']
            b1 = model_param['b1']
            b2 = model_param['b2']
        
        self.A = np.array([
            [0, 1, 0, 0],
            [-(a1 + a2), -beta1, a2, 0],
            [0, 0, 0, 1],
            [a2, 0, -(a2 + a3), beta2]
        ])

        self.B = np.array([
            [0, 0],
            [b1, 0],
            [0, 0],
            [0, b2]
            
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
        self.dt = 0.05
        self.t = 0.0
        self.data = {}
        self.T = 5.0  # 最大运行时间


    def _get_obs(self):
        return self.state

    def _get_info(self):
        info = {}
        info['data'] = deepcopy(self.data)
        return info

    def reset(self, seed=None, options={}):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(self.observation_space.low, self.observation_space.high) * 0.5
        self.t = 0.0
        self.data = {}
        # 模型参数
        if options is not None:
            model_param = options.get('model_param')
        else:
            model_param = None
        if model_param is not None:
            a1 = model_param['a1']
            a2 = model_param['a2']
            a3 = model_param['a3']
            beta1 = model_param['beta1']
            beta2 = model_param['beta2']
            b1 = model_param['b1']
            b2 = model_param['b2']
            self.A = np.array([
                [0, 1, 0, 0],
                [-(a1 + a2), -beta1, a2, 0],
                [0, 0, 0, 1],
                [a2, 0, -(a2 + a3), beta2]
            ])
            self.B = np.array([
                [0, 0],
                [b1, 0],
                [0, 0],
                [0, b2]
            ])
        # 控制参数
        if options is not None:
            control_param = options.get('control_param')
        else:
            control_param = None
        if control_param is not None:
            self.Q = control_param['Q']
            self.R = control_param['R']

        return self._get_obs(), self._get_info()

    def step(self, action):
        # 将 action 限制在 action_space 的范围内
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # 状态更新
        old_state = deepcopy(self.state)
        dxdt = self.A @ self.state + self.B @ action
        self.state = self.state + dxdt * self.dt
        self.t = self.t + self.dt
        # 奖励函数
        reward = -(self.state.T @ self.Q @ self.state + action.T @ self.R @ action) * self.dt + 10 * self.dt
        if np.all(np.abs(self.state) < np.array([0.2, 0.2, 0.2, 0.2])):
            terminated = True
            reward += 100
        else:
            terminated = False
        if not self.observation_space.contains(self.state):
            print("crashed")
            crashed = True
            reward -= 10
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
        info = self._get_info()

        return self.state, reward, terminated, truncated, info


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
        a1 = metadata['a1']
        a2 = metadata['a2']
        a3 = metadata['a3']
        beta1 = metadata['beta1']
        beta2 = metadata['beta2']
        b1 = metadata['b1']
        b2 = metadata['b2']
        self.A = np.array([
                [0, 1, 0, 0],
                [-(a1 + a2), -beta1, a2, 0],
                [0, 0, 0, 1],
                [a2, 0, -(a2 + a3), beta2]
            ])
        self.B = np.array([
            [0, 0],
            [b1, 0],
            [0, 0],
            [0, b2]
        ])

    def forward(self, x, u):
        # 将A, B转为torch tensor（如果还没转的话），并确保和x, u的dtype/device一致
        A = torch.from_numpy(self.A).to(x.device).type(x.dtype)
        B = torch.from_numpy(self.B).to(x.device).type(x.dtype)
        # x: (N, nx), u: (N, nu)
        Ax = torch.matmul(x, A.T)
        Bu = torch.matmul(u, B.T)
        return Ax + Bu
        
class NonmialAcadosModel():
    def __init__(self, metadata):
        self.metadata = metadata
        self.model = self.export_acados_model()

    def export_acados_model(self):
        a1 = self.metadata['a1']
        a2 = self.metadata['a2']
        a3 = self.metadata['a3']
        beta1 = self.metadata['beta1']
        beta2 = self.metadata['beta2']
        b1 = self.metadata['b1']
        b2 = self.metadata['b2']
        nx = self.metadata['nx']
        nu = self.metadata['nu']

        fx = ca.SX.sym('x_dot', nx)
        x = ca.SX.sym('x', nx)
        u = ca.SX.sym('u', nu)
        
        A = ca.DM([
            [0, 1, 0, 0],
            [-(a1 + a2), -beta1, a2, 0],
            [0, 0, 0, 1],
            [a2, 0, -(a2 + a3), beta2]
        ])

        B = ca.DM([
            [0, 0],
            [b1, 0],
            [0, 0],
            [0, b2]
        ])

        x_dot = A @ x + B @ u
        model = AcadosModel()
        model.f_expl_expr = x_dot
        model.f_impl_expr = fx - x_dot
        model.x_dot = fx
        model.x = x
        model.u = u
        model.p = []
        model.name = "double_mass"

        return model



