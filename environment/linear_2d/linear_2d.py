from copy import deepcopy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

class Linear2D(gym.Env):
    metadata = {}
    metadata['render_modes'] = []
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

    def __init__(self, options={}):
        super().__init__()
        
        # Model setup
        self.system_torch_dynamics = SystemTorchDynamics(self.metadata)
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(self.metadata['nx'],), dtype=np.float64)
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(self.metadata['nu'],), dtype=np.float64)

        # Default initial state; reset() uses the same [-4, 4] range.
        self.state = self.np_random.uniform(-4.0, 4.0, size=self.metadata['nx'])

        # Model parameters
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

        # Control parameters
        control_param = options.get('control_param')
        if control_param is None:
            self.Q = self.metadata['Q']
            self.R = self.metadata['R']
        else:
            self.Q = control_param['Q']
            self.R = control_param['R']

        # Simulation parameters
        # self.dt = 0.01
        self.dt = 0.1
        self.t = 0.0
        self.data = {}
        self.T = 20.0  # Maximum episode duration

        # Disturbance settings
        self.has_disturbed = False
        disturb_param = options.get('disturb_param')
        if disturb_param is not None:
            self.disturb_mode = disturb_param['disturb_mode']
            self.disturb_once()
        else:
            self.disturb_mode = None


        self.track_mode = None
        self.state_d = np.zeros_like(self.state)
        self.u_d = np.zeros_like(self.np_random.uniform(self.action_space.low, self.action_space.high))
        track_mode = options.get('track_mode')
        if track_mode is not None:
            self.track_mode = track_mode
            self.set_state_d()

    def _get_obs(self):
        return self.state

    def _get_info(self):
        info = {}
        info['data'] = deepcopy(self.data)
        return info

    def reset(self, seed=None, options={}, rl_mode=True):
        # rl_mode is retained only for API compatibility.  The linear-system
        # reset distribution and theoretical stage cost do not depend on it.
        if options is not None and 'rl_mode' in options:
            rl_mode = bool(options['rl_mode'])
        self.rl_mode = rl_mode

        self.t = 0.0
        self.data = {}
        super().reset(seed=seed)
        # Initial state
        if options is not None:
            init_state = options.get('init_state')
        else:
            init_state = None
        if init_state is not None:
            self.state = init_state
        else:
            self.state = self.np_random.uniform(-4.0, 4.0, size=self.metadata['nx'])
        # Model parameters
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
        # Control parameters
        if options is not None:
            control_param = options.get('control_param')
        else:
            control_param = None
        if control_param is not None:
            self.Q = control_param['Q']
            self.R = control_param['R']

        # Disturbance settings
        if options is not None:
            self.disturb_param = options.get('disturb_param')
        else:
            self.disturb_param = None
        if self.disturb_param is not None:
            self.disturb_mode = self.disturb_param['disturb_mode']
            self.disturb_once()
        else:
            self.disturb_mode = None

        if self.track_mode is None:
            return self._get_obs(), self._get_info()
        else:
            aug_state = np.hstack((self.state-self.state_d, self.state_d))
            return aug_state, self._get_info()
    
    def step(self, action):
        # Clip the action to the action-space bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Update the state
        old_state = deepcopy(self.state)
        self.t = self.t + self.dt
        self.stata_d, self.u_d = self.set_state_d()
        
        if self.track_mode is None:
            dxdt = self.A @ self.state + self.B @ action + self.disturbance()
            self.state = self.state + dxdt * self.dt
            # Reward function
            reward = -(self.state.T @ self.Q @ self.state + action.T @ self.R @ action) * self.dt
            if np.all(np.abs(self.state) < np.array([0.01, 0.01])):
                terminated = True
            else:
                terminated = False
        else:
            dxdt = self.A @ self.state + self.B @ (action + self.u_d) + self.disturbance()
            self.state = self.state + dxdt * self.dt
            delta_state = self.state - self.state_d
            # Reward function
            reward = -(delta_state @ self.Q @ delta_state + action @ self.R @ action) * self.dt
            if np.all(np.abs(delta_state) < np.array([0.01, 0.01])):
                terminated = True
            else:
                terminated = False
        if not self.observation_space.contains(self.state):
            # print("crashed")
            crashed = True
            self.state = old_state
        else:
            crashed = False
        if self.t > self.T:
            truncated = True
        else:
            truncated = False
        # Diagnostic information
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
    
        if self.track_mode is None:    
            return self.state, reward, terminated, truncated, info
        else:
            aug_state = np.hstack((self.state-self.state_d, self.state_d))
            return aug_state, reward, terminated, truncated, info

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
        if self.has_disturbed == True:
            return
        self.has_disturbed = True
        a1 = self.metadata['a1']
        a2 = self.metadata['a2']
        a3 = self.metadata['a3']
        a4 = self.metadata['a4']
        b1 = self.metadata['b1']
        b2 = self.metadata['b2']
        if self.disturb_mode == 'ParamBias':
            self.A = np.array([
            [a1, a2],
            [a3, a4]
            ])
            # A_disturbance = self.np_random.uniform(-0.5 * np.abs(self.A), 0.5 * np.abs(self.A))
            A_disturbance = np.array([[0, 0], [0, 0.90]])
            self.A += A_disturbance

            self.B = np.array([
                [b1],
                [b2]
            ])
            # B_disturbance = self.np_random.uniform(-0.5 * np.abs(self.B), 0.5 * np.abs(self.B))
            B_disturbance = np.array([[0], [-0.90]])
            self.B += B_disturbance

        elif self.disturb_mode == 'OppositeB':
            self.B = np.array([
                [b1],
                [b2]
            ])
            self.B = -self.B

    def set_state_d(self):
        if self.track_mode == 'linear_system':
            self.A_d = np.array([[0.0, 1.0], [-1.0, -1.0]])
            if self.t == 0:
                self.state_d = np.array([-2.0, -0.5])
                self.u_d = np.array([-self.state_d[0]-self.state_d[1]])
            else:
                dot_state_d = self.A_d @ self.state_d
                self.state_d = self.state_d + dot_state_d * self.dt
                self.u_d = np.array([-self.state_d[0]-self.state_d[1]])
        return self.state_d, self.u_d


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
        a4 = metadata['a4']
        b1 = metadata['b1']
        b2 = metadata['b2']
        self.A = np.array([
            [a1, a2],
            [a3, a4]
        ])
        self.B = np.array([
            [b1],
            [b2]
        ])

    def forward(self, x, u):

        A = torch.from_numpy(self.A).to(x.device).type(x.dtype)
        B = torch.from_numpy(self.B).to(x.device).type(x.dtype)
        # x: (N, nx), u: (N, nu)
        Ax = torch.matmul(x, A.T)
        Bu = torch.matmul(u, B.T)
        return Ax + Bu
