from copy import deepcopy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch


class Angle3D(gym.Env):
    metadata = {}
    metadata['nx'] = 3
    metadata['nu'] = 3
    metadata['Q'] = np.diag([1, 1, 1])
    metadata['R'] = np.diag([1, 1, 1])

    def __init__(self, options={}, theoretic_mode=False):
        super().__init__()
        
        # Model setup
        self.system_torch_dynamics = SystemTorchDynamics(self.metadata)
        self.observation_space = spaces.Box(low=-1.5, high=1.5, shape=(self.metadata['nx'],), dtype=np.float64)
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(self.metadata['nu'],), dtype=np.float64)


        self.state = 0.666 * self.np_random.uniform(self.observation_space.low, self.observation_space.high)

        # Control parameters
        control_param = options.get('control_param')
        if control_param is None:
            self.Q = self.metadata['Q']
            self.R = self.metadata['R']
        else:
            self.Q = control_param['Q']
            self.R = control_param['R']

        # Simulation parameters
        self.dt = 0.02
        self.t = 0.0
        self.data = {}
        self.T = 10.0  # Maximum episode duration
        self.theoretic_mode = theoretic_mode

        # Disturbance settings
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
        

        self.theoretic_mode = theoretic_mode
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        # Clip the action to the action-space bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Update the state
        old_state = deepcopy(self.state)
        dxdt = self._angle3d_dynamics_np(self.state, action) + self.disturbance()
        self.state = self.state + dxdt * self.dt
        self.t = self.t + self.dt
        # Reward function
        reward = -(self.state.T @ self.Q @ self.state + action.T @ self.R @ action) * self.dt + 10 * self.dt * (1 - self.theoretic_mode)
        if np.all(np.abs(self.state) < np.array([0.1, 0.1, 0.1])):
            state_norm = np.linalg.norm(self.state)
            reward += 10 * (1 - self.theoretic_mode) + 10 * (1 - state_norm) * (1 - self.theoretic_mode)
            if np.all(np.abs(self.state) < np.array([0.01, 0.01, 0.01])):
                terminated = True
            else:
                terminated = False
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

        return self.state, reward, terminated, truncated, info

    def _angle3d_dynamics_np(self, state, action):
        # state: (3,) -> [phi, theta, psi]; action: (3,) -> [p, q, r]
        phi = state[0]
        theta = state[1]

        p = action[0]
        q = action[1]
        r = action[2]

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
        # Parameter perturbations are not applied to this nonlinear system
        if self.has_disturbed == True:
            return
        self.has_disturbed = True
        if self.disturb_mode in ['ParamBias', 'OppositeB']:
            # No parameter change is required for these modes
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
