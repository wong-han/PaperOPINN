from copy import deepcopy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import sympy as sp
import scipy


class WingedCone(gym.Env):
    metadata = {}
    metadata['nx'] = 2
    metadata['nu'] = 1
    metadata['alpha0'] = 0.0314533
    metadata['hd'] = 110000
    metadata['Q'] = np.diag([0.0001, 0.0001])
    metadata['R'] = np.diag([1000])

    def __init__(self, options={}, theoretic_mode=False):
        super().__init__()
        
        # Model setup
        self.system_torch_dynamics = SystemTorchDynamics(self.metadata)
        self.state_low = np.array([-3000, -600])
        self.state_high = np.array([3000, 600])
        self.observation_space = spaces.Box(low=self.state_low, high=self.state_high, shape=(self.metadata['nx'],), dtype=np.float64)
        self.u_max = np.deg2rad(5)
        self.action_space = spaces.Box(low=-self.u_max, high=self.u_max, shape=(self.metadata['nu'],), dtype=np.float64)


        self.state = 0.5 * self.np_random.uniform(self.observation_space.low, self.observation_space.high)

        # Model parameters
        self.A, self.B = linearize_at_origin(self.metadata)

        # Control parameters
        control_param = options.get('control_param')
        if control_param is None:
            self.Q = self.metadata['Q']
            self.R = self.metadata['R']
        else:
            self.Q = control_param['Q']
            self.R = control_param['R']

        self.P = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)

        # Simulation parameters
        self.dt = 0.01
        self.t = 0.0
        self.data = {}
        self.T = 50.0  # Maximum episode duration
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

    def reset(self, seed=None, options={}):
        super().reset(seed=seed)
        self.state = 0.5 * self.np_random.uniform(self.observation_space.low, self.observation_space.high)
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

        return self._get_obs(), self._get_info()
    
    def step(self, action):
        # Clip the action to the action-space bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Update the state
        old_state = deepcopy(self.state)
        dxdt = self._nonlinear_dynamics_np(self.state, action) + self.disturbance()
        self.state = self.state + dxdt * self.dt
        self.t = self.t + self.dt
        # Reward function
        reward = -(self.state.T @ self.Q @ self.state + action.T @ self.R @ action) * self.dt + 10 * self.dt * (1 - self.theoretic_mode)
        if np.all(np.abs(self.state) < np.array([0.1, 0.1])):
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

    def _nonlinear_dynamics_np(self, state, action):
        # state: (2,), action: (1,) or scalar array
        x1 = state[0]
        x2 = state[1]
        alpha = action[0] if np.ndim(action) > 0 else action
        # dx1 = x2
        # dx2 = 64345.28 * np.exp(-(x1 + 110000) / 24000) * np.sqrt(1 - (x2 / 15060) ** 2) * (alpha - 0.0314533) - 20.69 * (1 - (x2 / 15060) ** 2)
        sqrt_arg = 1 - (x2 / 15060) ** 2
        dx1 = x2
        dx2 = 64345.28 * np.exp(-(x1 + 110000) / 24000) * np.sqrt(sqrt_arg) * (alpha + 0.0314533) - 20.69 * sqrt_arg
        return np.array([dx1, dx2])

    def disturbance(self):
        if self.disturb_mode == 'ConstantBias':
            disturbance = np.array([0.2, 0.3])
        elif self.disturb_mode == 'AdditiveNoise':
            disturbance = self.np_random.normal(0, 0.5, size=self.metadata['nx'])
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
        # x: (N, 2), u: (N, 1)
        x1 = x[:, 0]
        x2 = x[:, 1]
        alpha = u[:, 0]
        sqrt_arg = 1 - (x2 / 15060.0) ** 2
        dx1 = x2
        dx2 = 64345.28 * torch.exp(-(x1 + 110000) / 24000.0) * torch.sqrt(sqrt_arg) * (alpha - 0.0314533) - 20.69 * sqrt_arg
        return torch.stack([dx1, dx2], dim=1)
        
def linearize_at_origin(metadata):
    """
    Compute the linearized A and B matrices at x=0 and u=0 using SymPy.
    :param metadata: Environment-parameter dictionary.
    :return: Numerical matrices A and B.
    """
    x1, x2, alpha = sp.symbols('x1 x2 alpha')
    x = sp.Matrix([x1, x2])
    u = sp.Matrix([alpha])

    sqrt_arg = 1.0 - (x2 / 15060.0) ** 2
    dx1 = x2
    dx2 = 64345.28 * sp.exp(-(x1 + 110000.0) / 24000.0) * sp.sqrt(sqrt_arg) * (alpha - 0.0314533) - 20.69 * sqrt_arg
    f = sp.Matrix([dx1, dx2])

    A_sym = f.jacobian(x)
    B_sym = f.jacobian(u)

    subs_dict = {x1: 0.0, x2: 0.0, alpha: 0.0}
    A0 = np.array(A_sym.subs(subs_dict), dtype=np.float64)
    B0 = np.array(B_sym.subs(subs_dict), dtype=np.float64)

    return A0, B0
