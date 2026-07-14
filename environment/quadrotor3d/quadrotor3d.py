from copy import deepcopy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import casadi as ca
from acados_template import AcadosModel, AcadosOcp


class Quadrotor3D(gym.Env):
    metadata = {}
    metadata['nx'] = 12
    metadata['nu'] = 4
    metadata['m'] = 0.027
    metadata['g'] = 9.81
    metadata['l'] = 0.028
    metadata['Ixx'] = 1.4e-5
    metadata['Iyy'] = 1.4e-5
    metadata['Izz'] = 2.17e-5
    metadata['kT'] = 3.6e-10
    metadata['kM'] = 7.94e-12
    metadata['omega_max'] = 37267.8 * 0.5  # 削减最大推力为2.0 * (0.5 ** 2) = 0.5

    metadata['Q'] = np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    metadata['R'] = np.diag([10, 10, 10, 10])

    def __init__(self, options={}, theoretic_mode=False):
        super().__init__()

        # 模型设置
        self.acados_model = NonmialAcadosModel(self.metadata)
        self.system_torch_dynamics = SystemTorchDynamics(self.metadata)
        # 状态空间: [p_x, p_y, p_z, v_x, v_y, v_z, phi, theta, psi, p, q, r]
        state_low = np.array([-1.5, -1.5, -1.5, -10, -10, -10, -np.pi, -np.pi/2, -np.pi, -10, -10, -10])
        state_high = np.array([1.5, 1.5, 1.5, 10, 10, 10, np.pi, np.pi/2, np.pi, 10, 10, 10])
        self.observation_space = spaces.Box(low=state_low, high=state_high, shape=(self.metadata['nx'],), dtype=np.float64)
        # 动作空间: 4个电机推力
        u_max = self.metadata['kT'] * (self.metadata['omega_max'] ** 2)
        self.action_space = spaces.Box(low=0, high=u_max, shape=(self.metadata['nu'],), dtype=np.float64)

        # 在observation_space范围内随机生成一个状态
        self.state = 1.0 * self.np_random.uniform(self.observation_space.low, self.observation_space.high)
        self.state[3:6] = 0.0  # 初始速度为0
        self.state[6:9] = 0.0  # 初始欧拉角为0
        self.state[9:12] = 0.0 # 初始角速度为0

        # 模型参数
        self.A, self.B = linearize_at_origin(self.acados_model)

        # 控制参数
        control_param = options.get('control_param')
        if control_param is None:
            self.Q = self.metadata['Q']
            self.R = self.metadata['R']
        else:
            self.Q = control_param['Q']
            self.R = control_param['R']

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
        self.state = 1.0 * self.np_random.uniform(self.observation_space.low, self.observation_space.high)
        self.state[3:6] = 0.0
        self.state[6:9] = 0.0
        self.state[9:12] = 0.0
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
        # 状态更新（使用无人机3D非线性系统）
        old_state = deepcopy(self.state)
        dxdt = self._nonlinear_dynamics_np(self.state, action) + self.disturbance()
        self.state = self.state + dxdt * self.dt
        self.t = self.t + self.dt
        # 奖励函数
        reward = -(self.state.T @ self.Q @ self.state + action.T @ self.R @ action) * self.dt + 10 * self.dt * (1 - self.theoretic_mode)
        if np.all(np.abs(self.state) < np.array([0.01]*12)):
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
        # state: (12,), action: (4,)
        m = self.metadata['m']
        g = self.metadata['g']
        l = self.metadata['l']
        Ixx = self.metadata['Ixx']
        Iyy = self.metadata['Iyy']
        Izz = self.metadata['Izz']
        kT = self.metadata['kT']
        kM = self.metadata['kM']

        # 状态变量
        px, py, pz = state[0:3]
        vx, vy, vz = state[3:6]
        phi, theta, psi = state[6:9]
        p, q, r = state[9:12]

        # 动作为4个电机推力
        T1, T2, T3, T4 = action

        # 推力/力矩分配矩阵
        alloc = np.array([
            [1, 1, 1, 1],
            [-l, l, l, -l],
            [-l, l, -l, l],
            [-kM/kT, -kM/kT, kM/kT, kM/kT]
        ])
        motor_input = np.array([T1, T2, T3, T4])
        T, tau_x, tau_y, tau_z = alloc @ motor_input

        # 旋转矩阵
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        cth = np.cos(theta)
        sth = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)
        R = np.array([
            [cth * cpsi, sphi * sth * cpsi - cphi * spsi, cphi * sth * cpsi + sphi * spsi],
            [cth * spsi, sphi * sth * spsi + cphi * cpsi, cphi * sth * spsi - sphi * cpsi],
            [-sth,       sphi * cth,                      cphi * cth]
        ])

        # 位置微分
        dpos = np.array([vx, vy, vz])
        # 速度微分
        dv = (1.0 / m) * (R @ np.array([0, 0, T])) - np.array([0, 0, g])

        # 欧拉角微分
        tan_theta = np.tan(theta) if np.abs(np.cos(theta)) > 1e-6 else 0.0
        sec_theta = 1.0 / np.cos(theta) if np.abs(np.cos(theta)) > 1e-6 else 0.0
        euler_dot = np.array([
            1, sphi * tan_theta, cphi * tan_theta,
            0, cphi,            -sphi,
            0, sphi * sec_theta, cphi * sec_theta
        ]).reshape(3, 3) @ np.array([p, q, r])

        # 角速度微分
        p_dot = (1.0 / Ixx) * (tau_x + q * r * (Iyy - Izz))
        q_dot = (1.0 / Iyy) * (tau_y + p * r * (Izz - Ixx))
        r_dot = (1.0 / Izz) * (tau_z + p * q * (Ixx - Iyy))

        dxdt = np.zeros(12)
        dxdt[0:3] = dpos
        dxdt[3:6] = dv
        dxdt[6:9] = euler_dot
        dxdt[9] = p_dot
        dxdt[10] = q_dot
        dxdt[11] = r_dot
        return dxdt

    def disturbance(self):
        if self.disturb_mode == 'ConstantBias':
            disturbance = np.array([0.1]*12)
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
        # x: (N, 12), u: (N, 4)
        m = self.metadata['m']
        g = self.metadata['g']
        l = self.metadata['l']
        Ixx = self.metadata['Ixx']
        Iyy = self.metadata['Iyy']
        Izz = self.metadata['Izz']
        kT = self.metadata['kT']
        kM = self.metadata['kM']

        px = x[:, 0]
        py = x[:, 1]
        pz = x[:, 2]
        vx = x[:, 3]
        vy = x[:, 4]
        vz = x[:, 5]
        phi = x[:, 6]
        theta = x[:, 7]
        psi = x[:, 8]
        p = x[:, 9]
        q = x[:, 10]
        r = x[:, 11]

        T1 = u[:, 0]
        T2 = u[:, 1]
        T3 = u[:, 2]
        T4 = u[:, 3]

        # 分配矩阵
        alloc = torch.tensor([
            [1, 1, 1, 1],
            [-l, l, l, -l],
            [-l, l, -l, l],
            [-kM/kT, -kM/kT, kM/kT, kM/kT]
        ], dtype=x.dtype, device=x.device)
        motor_input = torch.stack([T1, T2, T3, T4], dim=1).T  # (4, N)
        out = torch.matmul(alloc, motor_input)  # (4, N)
        T = out[0]
        tau_x = out[1]
        tau_y = out[2]
        tau_z = out[3]

        cphi = torch.cos(phi)
        sphi = torch.sin(phi)
        cth = torch.cos(theta)
        sth = torch.sin(theta)
        cpsi = torch.cos(psi)
        spsi = torch.sin(psi)

        # 旋转矩阵
        R11 = cth * cpsi
        R12 = sphi * sth * cpsi - cphi * spsi
        R13 = cphi * sth * cpsi + sphi * spsi
        R21 = cth * spsi
        R22 = sphi * sth * spsi + cphi * cpsi
        R23 = cphi * sth * spsi - sphi * cpsi
        R31 = -sth
        R32 = sphi * cth
        R33 = cphi * cth

        # 位置微分
        dpos = torch.stack([vx, vy, vz], dim=1)
        # 速度微分
        dv = torch.stack([
            (R13 * T) / m,
            (R23 * T) / m,
            (R33 * T) / m - g
        ], dim=1)

        # 欧拉角微分
        tan_theta = torch.tan(theta)
        sec_theta = 1.0 / torch.cos(theta)
        euler_dot = torch.stack([
            p + sphi * tan_theta * q + cphi * tan_theta * r,
            cphi * q - sphi * r,
            sphi * sec_theta * q + cphi * sec_theta * r
        ], dim=1)

        # 角速度微分
        p_dot = (tau_x + q * r * (Iyy - Izz)) / Ixx
        q_dot = (tau_y + p * r * (Izz - Ixx)) / Iyy
        r_dot = (tau_z + p * q * (Ixx - Iyy)) / Izz

        dxdt = torch.cat([
            dpos, dv, euler_dot,
            p_dot.unsqueeze(1), q_dot.unsqueeze(1), r_dot.unsqueeze(1)
        ], dim=1)
        return dxdt

class NonmialAcadosModel():
    def __init__(self, metadata):
        self.metadata = metadata
        self.model = self.export_acados_model()

    def export_acados_model(self):
        nx = self.metadata['nx']
        nu = self.metadata['nu']
        m = self.metadata['m']
        g = self.metadata['g']
        l = self.metadata['l']
        Ixx = self.metadata['Ixx']
        Iyy = self.metadata['Iyy']
        Izz = self.metadata['Izz']
        kT = self.metadata['kT']
        kM = self.metadata['kM']

        fx = ca.SX.sym('x_dot', nx)
        x = ca.SX.sym('x', nx)
        u = ca.SX.sym('u', nu)

        # 状态变量
        px = x[0]
        py = x[1]
        pz = x[2]
        vx = x[3]
        vy = x[4]
        vz = x[5]
        phi = x[6]
        theta = x[7]
        psi = x[8]
        p = x[9]
        q = x[10]
        r = x[11]

        T1 = u[0]
        T2 = u[1]
        T3 = u[2]
        T4 = u[3]

        # 推力/力矩分配
        alloc = ca.vertcat(
            ca.horzcat(1, 1, 1, 1),
            ca.horzcat(-l, l, l, -l),
            ca.horzcat(-l, l, -l, l),
            ca.horzcat(-kM/kT, -kM/kT, kM/kT, kM/kT)
        )
        motor_input = ca.vertcat(T1, T2, T3, T4)
        out = alloc @ motor_input
        T = out[0]
        tau_x = out[1]
        tau_y = out[2]
        tau_z = out[3]

        cphi = ca.cos(phi)
        sphi = ca.sin(phi)
        cth = ca.cos(theta)
        sth = ca.sin(theta)
        cpsi = ca.cos(psi)
        spsi = ca.sin(psi)
        R = ca.vertcat(
            ca.horzcat(cth * cpsi, sphi * sth * cpsi - cphi * spsi, cphi * sth * cpsi + sphi * spsi),
            ca.horzcat(cth * spsi, sphi * sth * spsi + cphi * cpsi, cphi * sth * spsi - sphi * cpsi),
            ca.horzcat(-sth,       sphi * cth,                      cphi * cth)
        )

        # 位置微分
        dpos = ca.vertcat(vx, vy, vz)
        # 速度微分
        dv = (1.0 / m) * (R @ ca.vertcat(0, 0, T)) - ca.vertcat(0, 0, g)

        # 欧拉角微分
        tan_theta = ca.tan(theta)
        sec_theta = 1.0 / ca.cos(theta)
        euler_dot = ca.vertcat(
            p + sphi * tan_theta * q + cphi * tan_theta * r,
            cphi * q - sphi * r,
            sphi * sec_theta * q + cphi * sec_theta * r
        )

        # 角速度微分
        p_dot = (1.0 / Ixx) * (tau_x + q * r * (Iyy - Izz))
        q_dot = (1.0 / Iyy) * (tau_y + p * r * (Izz - Ixx))
        r_dot = (1.0 / Izz) * (tau_z + p * q * (Ixx - Iyy))

        x_dot = ca.vertcat(
            dpos,
            dv,
            euler_dot,
            p_dot,
            q_dot,
            r_dot
        )

        model = AcadosModel()
        model.f_expl_expr = x_dot
        model.f_impl_expr = fx - x_dot
        model.x_dot = fx
        model.x = x
        model.u = u
        model.p = []
        model.name = "quadrotor3d"

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
    u0 = np.ones(model.u.size()[0]) * 0.25 * acados_model.metadata['m'] * acados_model.metadata['g']
    A0 = A_func(x0, u0).full()  # 转换为NumPy数组
    B0 = B_func(x0, u0).full()

    return A0, B0

