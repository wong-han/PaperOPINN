from copy import deepcopy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import sympy as sp
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

        # Model setup
        self.system_torch_dynamics = SystemTorchDynamics(self.metadata)
        # State space
        self.state_low = np.array([-np.pi, -80])
        self.state_high = np.array([np.pi, 80])
        self.observation_space = spaces.Box(low=self.state_low, high=self.state_high, shape=(self.metadata['nx'],), dtype=np.float64)
        # Action space
        u_max = 1.8
        self.u_min = np.array([-u_max])
        self.u_max = np.array([u_max])
        self.action_space = spaces.Box(low=self.u_min, high=self.u_max, shape=(self.metadata['nu'],), dtype=np.float64)


        self.state = 1.0 * self.np_random.uniform(self.observation_space.low, self.observation_space.high)
        self.state[1] = 0.0  # Initialize angular velocity to zero

        # Model parameters
        self.A, self.B = linearize_at_origin(self.metadata)

        # Control parameters
        self.u0 = np.array([0.0])
        control_param = options.get('control_param')
        if control_param is None:
            self.Q = self.metadata['Q']
            self.R = self.metadata['R']
        else:
            self.Q = control_param['Q']
            self.R = control_param['R']

        self.P = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)

        # Simulation parameters
        self.dt = 0.001
        self.t = 0.0
        self.data = {}
        self.T = 5.0  # Maximum episode duration
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
        # Initial state
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

        self.state[0] = (self.state[0] + np.pi) % (2 * np.pi) - np.pi
        self.t = self.t + self.dt
        # Reward function
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

def linearize_at_origin(metadata):
    """
    Compute the linearized A and B matrices at x=0 and u=0 using SymPy.
    :param metadata: Environment-parameter dictionary.
    :return: Numerical matrices A and B.
    """
    theta, theta_dot, u_val = sp.symbols('theta theta_dot u_val')
    m = metadata['m']
    l = metadata['length']
    beta = metadata['beta']
    g = metadata['g']

    x = sp.Matrix([theta, theta_dot])
    u = sp.Matrix([u_val])

    dtheta = theta_dot
    dtheta_dot = (m * g * l * sp.sin(theta) - beta * theta_dot + u_val) / (m * l ** 2)
    f = sp.Matrix([dtheta, dtheta_dot])

    A_sym = f.jacobian(x)
    B_sym = f.jacobian(u)

    subs_dict = {theta: 0.0, theta_dot: 0.0, u_val: 0.0}
    A0 = np.array(A_sym.subs(subs_dict), dtype=np.float64)
    B0 = np.array(B_sym.subs(subs_dict), dtype=np.float64)

    return A0, B0
