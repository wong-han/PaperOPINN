import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
import matplotlib as mpl

# ================== 图像设置 =======================
mpl.rcParams['font.family'] = 'Times New Roman'  # 默认字体
# mpl.rcParams['font.weight'] = 'bold'  # 加粗
mpl.rcParams['text.usetex'] = True  # 使用TEX
mpl.rcParams['axes.unicode_minus'] = False  # 解决无法显示负号
mpl.rcParams['xtick.direction'] = 'in'  # x轴刻度线朝内
mpl.rcParams['ytick.direction'] = 'in'  # y轴刻度线朝内
mpl.rcParams['xtick.top'] = True  # 显示上方的坐标轴
mpl.rcParams['ytick.right'] = True  # 显示右侧的坐标轴

mpl.rcParams['legend.frameon'] = False  # legend不显示边框
mpl.rcParams['legend.fontsize'] = 24  # legend默认size

mpl.rcParams['axes.titlesize'] = 24   # 设置title默认size
mpl.rcParams['xtick.labelsize'] = 24  # x坐标默认size
mpl.rcParams['ytick.labelsize'] = 24  # y坐标默认size
mpl.rcParams['axes.labelsize'] = 24  # 轴标题默认size
mpl.rcParams['lines.linewidth'] = 2  # 线条宽度
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.linestyle'] = '--'  # 虚线
mpl.rcParams['grid.alpha'] = 0.5       # 透明度
mpl.rcParams['grid.color'] = 'gray'    # 网格颜色

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)

class DroneVisualizer:
    def __init__(self, traj, attitudes=None, alpha_range=(0.2, 1.0), stride=5, pause=0.02):
        """
        traj: N x 3 array, positions
        attitudes: N x 3 (roll, pitch, yaw in degrees) or None (all zeros)
        alpha_range: (min_alpha, max_alpha)
        stride: plot every stride points for visualization
        pause: pause time for plt.pause
        """
        self.traj = np.asarray(traj)
        self.N = len(self.traj)
        if attitudes is not None:
            self.attitudes = np.asarray(attitudes)
        else:
            self.attitudes = np.zeros((self.N, 3))
        self.alpha_range = alpha_range
        self.stride = stride
        self.pause = pause

        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('white')
        self.fig.set_facecolor('white')
        self.ax.view_init(elev=25, azim=-53, roll=0)

        # 轴范围视为-5~5, 0~10
        # self.ax.set_xlim([-1.5, 1.5])
        # self.ax.set_ylim([-1.5, 1.5])
        # self.ax.set_zlim([-1.5, 1.5])
        self.ax.set_xlabel('$x$(m)',labelpad=10)
        self.ax.set_ylabel('$y$(m)',labelpad=10)
        self.ax.set_zlabel('$z$(m)')
        # 坐标轴外观
        self.ax.xaxis.pane.fill = True
        self.ax.yaxis.pane.fill = True
        self.ax.zaxis.pane.fill = True
        self.ax.xaxis.pane.set_alpha(0.1)
        self.ax.yaxis.pane.set_alpha(0.1)
        self.ax.zaxis.pane.set_alpha(0.1)
        self.ax.xaxis.pane.set_edgecolor('lightgray')
        self.ax.yaxis.pane.set_edgecolor('lightgray')
        self.ax.zaxis.pane.set_edgecolor('lightgray')
        self.ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

        self.size_factor = 0.1

    def _rotation_matrix(self, roll, pitch, yaw):
        """roll, pitch, yaw (deg)"""
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)
        # Yaw (Z)
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        # Pitch (Y)
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        # Roll (X)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        return Rz @ Ry @ Rx

    def _create_cube(self, size):
        """创建立方体顶点"""
        v = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                     [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]) * size/2
        f = [[0,1,2,3], [4,5,6,7], [0,1,5,4], 
             [2,3,7,6], [0,3,7,4], [1,2,6,5]]
        verts = []
        for face in f:
            for idx in face:
                verts.append(v[idx])
        verts = np.array(verts)
        x = verts[:, 0].reshape(6, 4).T
        y = verts[:, 1].reshape(6, 4).T
        z = verts[:, 2].reshape(6, 4).T
        return x, y, z

    def _create_arm(self, length, thickness, angle):
        """创建旋翼臂"""
        v = np.array([
            [-thickness/2, -thickness/2, -thickness/2],
            [length/2, -thickness/2, -thickness/2],
            [length/2, thickness/2, -thickness/2],
            [-thickness/2, thickness/2, -thickness/2],
            [-thickness/2, -thickness/2, thickness/2],
            [length/2, -thickness/2, thickness/2],
            [length/2, thickness/2, thickness/2],
            [-thickness/2, thickness/2, thickness/2]
        ])
        # 旋转臂到正确方向
        rot = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        v = v @ rot.T
        f = [[0,1,2,3], [4,5,6,7], [0,1,5,4], 
             [2,3,7,6], [0,3,7,4], [1,2,6,5]]
        verts = []
        for face in f:
            for idx in face:
                verts.append(v[idx])
        verts = np.array(verts)
        x = verts[:, 0].reshape(6, 4).T
        y = verts[:, 1].reshape(6, 4).T
        z = verts[:, 2].reshape(6, 4).T
        return x, y, z

    def _create_rotor(self, radius, height):
        """创建旋翼（圆柱体）"""
        # 修正z和theta meshgrid顺序，确保网格shape为(theta, z)
        theta = np.linspace(0, 2 * np.pi, 30)
        z = np.linspace(-height / 2, height / 2, 2)
        theta_grid, z_grid = np.meshgrid(theta, z, indexing='ij')  # shape (30,2)

        x = radius * np.cos(theta_grid)  # (30,2)
        y = radius * np.sin(theta_grid)  # (30,2)
        z = z_grid                       # (30,2)
        return x, y, z

    def _draw_drone(self, position, att, alpha):
        """在指定位置和姿态绘制无人机，透明度为alpha"""
        # att: (roll, pitch, yaw) in degree
        roll, pitch, yaw = att
        R = self._rotation_matrix(roll, pitch, yaw)
        x0, y0, z0 = position

        # 机身
        cube_size = 0.5 * self.size_factor
        cube = self._create_cube(cube_size)
        cube_flat = np.array([cube[0].flatten(), cube[1].flatten(), cube[2].flatten()])
        cube_rot = R @ cube_flat
        cube_rot = cube_rot.reshape(3, 6, 4)
        body = self.ax.plot_surface(
            cube_rot[0] + x0,
            cube_rot[1] + y0,
            cube_rot[2] + z0,
            color='#4c72b0', alpha=alpha, edgecolor='black', linewidth=0.5
        )

        # 旋翼臂
        arm_length = 1.5 * self.size_factor
        arm_thickness = 0.1 * self.size_factor
        arms = []
        for angle in [0, 90, 180, 270]:
            rad = np.radians(angle)
            arm = self._create_arm(arm_length, arm_thickness, rad)
            arm_flat = np.array([arm[0].flatten(), arm[1].flatten(), arm[2].flatten()])
            arm_rot = R @ arm_flat
            arm_rot = arm_rot.reshape(3, 6, 4)
            a = self.ax.plot_surface(
                arm_rot[0] + x0,
                arm_rot[1] + y0,
                arm_rot[2] + z0,
                color='#55a868', alpha=alpha, edgecolor='black', linewidth=0.3
            )
            arms.append(a)

        # 旋翼
        rotor_radius = 0.4 * self.size_factor
        rotor_height = 0.05 * self.size_factor
        for angle in [45, 135, 225, 315]:
            rad = np.radians(angle)
            x_offset = np.cos(rad) * arm_length / 2
            y_offset = np.sin(rad) * arm_length / 2
            rotor = self._create_rotor(rotor_radius, rotor_height)
            # rotor: (x, y, z), each shape (30,2)
            rotor_points = np.array([rotor[0], rotor[1], rotor[2]])  # shape (3, 30, 2)
            rotor_flat = rotor_points.reshape(3, -1)  # (3, 60)
            rotor_rot = R @ rotor_flat  # (3, 60)
            rotor_rot = rotor_rot.reshape(3, rotor[0].shape[0], rotor[0].shape[1])  # (3,30,2)
            self.ax.plot_surface(
                rotor_rot[0] + x0 + x_offset,
                rotor_rot[1] + y0 + y_offset,
                rotor_rot[2] + z0,
                color='#c44e52', alpha=alpha, edgecolor='black', linewidth=0.3
            )

        # 方向箭头
        arrow_end = R @ np.array([1,0,0]) * self.size_factor
        arrow_props = dict(mutation_scale=15, arrowstyle='-|>', color='red', linewidth=2, alpha=alpha)
        arrow = Arrow3D(
            [x0, x0 + arrow_end[0]],
            [y0, y0 + arrow_end[1]],
            [z0, z0 + arrow_end[2]],
            **arrow_props
        )
        self.ax.add_artist(arrow)

    def show(self):
        # 绘制轨迹曲线
        self.ax.plot(self.traj[:,0], self.traj[:,1], self.traj[:,2], 'b', alpha=0.4, linewidth=1.5)
        # draw drone along trajectory, with alpha gradient
        indices = list(range(0, self.N, self.stride))
        if indices[-1] != self.N-1:
            indices.append(self.N-1)
        n = len(indices)
        alphas = np.linspace(self.alpha_range[0], self.alpha_range[1], n)
        for idx, alpha in zip(indices, alphas):
            self._draw_drone(self.traj[idx], self.attitudes[idx], alpha=alpha)
            plt.pause(self.pause)
        plt.savefig('image/quad_traj.svg', dpi=600, bbox_inches='tight', pad_inches=0.5)
        plt.show()
    
    def add_one_traj(self, traj, attitudes, stride, colormap='bwr'):
        N = traj.shape[0]
        # 根据轨迹距离原点的距离画上由红到蓝的渐变
        from matplotlib import cm
        from matplotlib.colors import Normalize

        # 计算每个点距离原点的距离
        dists = np.linalg.norm(traj, axis=1)
        norm = Normalize(vmin=np.min(dists), vmax=np.max(dists))
        cmap = cm.get_cmap(colormap)

        # 绘制带有渐变色的轨迹曲线（3D线段）
        for i in range(N-1):
            seg_traj = traj[i:i+2]
            color = cmap(norm(dists[i]))
            self.ax.plot(seg_traj[:,0], seg_traj[:,1], seg_traj[:,2], color=color, linewidth=3)

        # draw drone along trajectory, with color from RdBu according to distance
        indices = list(range(0, N, stride))
        if indices[-1] != N-1:
            indices.append(N-1)
        for idx in indices:
            color = cmap(norm(dists[idx]))
            if np.linalg.norm(traj[idx]) > 0.5:
                self._draw_drone(traj[idx], attitudes[idx], alpha=1.0)
                # 在 drone 外部画透明的球体用以区分颜色（可选）
                # self.ax.scatter(traj[idx,0], traj[idx,1], traj[idx,2], color=color, s=30, alpha=0.7)
            else:
                self._draw_drone(traj[-1], attitudes[-1], alpha=1.0)
    
    def add_only_traj(self, traj, colormap='bwr'):
        # 绘制带有渐变色的轨迹曲线（3D线段）
        from matplotlib import cm
        from matplotlib.colors import Normalize
        N = traj.shape[0]
        dists = np.linalg.norm(traj, axis=1)
        norm = Normalize(vmin=np.min(dists), vmax=np.max(dists))
        cmap = cm.get_cmap(colormap)
        for i in range(N-1):
            seg_traj = traj[i:i+2]
            color = cmap(norm(dists[i]))
            self.ax.plot(seg_traj[:,0], seg_traj[:,1], seg_traj[:,2], color=color, linewidth=1.5, alpha=0.2)

# ==============
# 示例用法/调用
# ==============

if __name__ == "__main__":
    # 轨迹示例(N x 3), 姿态示例(N x 3)
    t = np.linspace(0, 10, 101)
    traj_demo = np.stack([
        3*np.cos(0.6*t),
        3*np.sin(0.6*t),
        5 + 2*np.sin(0.3*t)
    ], axis=1)
    att_demo = np.stack([
        20*np.sin(0.2*t),   # roll
        15*np.sin(0.35*t),  # pitch
        180*np.sin(0.1*t)   # yaw
    ], axis=1)  # degrees

    viz = DroneVisualizer(traj=traj_demo, attitudes=att_demo, alpha_range=(0.25, 1.0), stride=15, pause=0.04)
    viz.show()