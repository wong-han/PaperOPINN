"""
Plot 3D and 2D trajectories for adjusted state data files.

Example:
    python plot_state_data.py \
        --inputs hnn/state_data_adjusted.txt lqr/state_data_adjusted.txt \
        --output-dir plots
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib
from matplotlib import cm
import sys
# Choose backend: use interactive backend when user passes --show
if '--show' in sys.argv:
    try:
        matplotlib.use('TkAgg')
    except Exception:
        matplotlib.use('Agg')
else:
    # Default to non-interactive backend for headless environments
    matplotlib.use('Agg')

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D projection
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import sys

# allow importing utils from workspace root (parent directory)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from utils import DroneVisualizer
except Exception:
    DroneVisualizer = None


# ==== Matplotlib style settings (match attachment style) ====
mpl.rcParams['font.family'] = 'Times New Roman'
# mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['text.usetex'] = True
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 28  # legend默认size
mpl.rcParams['axes.titlesize'] = 28   # 设置title默认size
mpl.rcParams['xtick.labelsize'] = 28  # x坐标默认size
mpl.rcParams['ytick.labelsize'] = 28  # y坐标默认size
mpl.rcParams['axes.labelsize'] = 28  # 轴标题默认size
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.alpha'] = 0.5
mpl.rcParams['grid.color'] = 'gray'

# Nature palette used across plots
NATURE_GREEN = "#389826"  # LQR
NATURE_PURPLE = "#9558B2"  # HNN


COLUMN_NAMES = [
    "time",
    "pos_x",
    "pos_y",
    "pos_z",
    "roll",
    "pitch",
    "yaw",
    "vel_x",
    "vel_y",
    "vel_z",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 2D and 3D trajectory plots for adjusted state data files."
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        type=Path,
        required=False,
        help=(
            "One or more adjusted txt files to plot. If omitted, defaults to "
            "hnn/state_data_adjusted.txt and lqr/state_data_adjusted.txt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        required=False,
        help="Directory to store generated plot images.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        required=False,
        help="If set, display the combined 3D plot using matplotlib.show() (requires interactive backend).",
    )
    return parser.parse_args()


def load_state_data(path: Path) -> Tuple[List[float], List[float], List[float], List[float]]:
    ts: List[float] = []
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", fieldnames=COLUMN_NAMES)
        next(reader, None)  # skip header
        for row in reader:
            try:
                ts.append(float(row["time"]))
                xs.append(float(row["pos_x"]))
                ys.append(float(row["pos_y"]))
                zs.append(float(row["pos_z"]))
            except (TypeError, ValueError):
                continue

    if not xs:
        raise ValueError(f"No data rows parsed from {path}")

    return ts, xs, ys, zs


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_combined_3d(datasets, output_dir: Path, show: bool = False) -> None:
    # If programmatically requested to show, attempt to switch to an interactive backend.
    if show:
        backend = matplotlib.get_backend()
        if backend.lower().startswith('agg'):
            matplotlib.use('TkAgg', force=True)

    ensure_output_dir(output_dir)

    # Require DroneVisualizer for plotting: only use DroneVisualizer methods
    if DroneVisualizer is None:
        print("Error: DroneVisualizer is not available. Cannot draw combined 3D using only DroneVisualizer.")
        return

    # Colormap choices (strings accepted by DroneVisualizer.add_one_traj)
    cmap_hnn = 'OrRd_r'
    cmap_lqr = 'winter_r'

    # How many drone models to draw along each trajectory (reduce to minimize clutter)
    target_drones = 1

    # Initialize a single DroneVisualizer using the first dataset as a base
    first_label, ts0, xs0, ys0, zs0 = datasets[0]
    traj0 = np.column_stack((np.array(xs0), np.array(ys0), np.array(zs0)))
    N0 = max(1, traj0.shape[0])
    attitudes0 = np.zeros((N0, 3))
    dv = DroneVisualizer(traj=traj0, attitudes=attitudes0, alpha_range=(0.2, 1.0), stride=max(1, N0 // target_drones))
    dv.size_factor = 0.05

    for idx, (label, ts, xs, ys, zs) in enumerate(datasets):
        lab_lower = label.lower()
        traj = np.column_stack((np.array(xs), np.array(ys), np.array(zs)))
        N = max(1, traj.shape[0])
        attitudes = np.zeros((N, 3))
        stride = max(1, N // target_drones)
        if 'hnn' in lab_lower:
            cmap = cmap_hnn
        elif 'lqr' in lab_lower:
            cmap = cmap_lqr
        else:
            cmap = 'bwr'
        # Assume correct signature for add_one_traj (stride as kwarg)
        dv.add_one_traj(traj, attitudes, stride=stride, colormap=cmap)

    dv.ax.set_xlim(-0.3, 0.9)
    dv.ax.set_ylim(-0.6, 0.6)
    dv.ax.set_zlim(0.4, 1.6)

    dv.fig.tight_layout()
    out_name = output_dir / "combined_3d.svg"

    # 设置三维坐标轴三个边的颜色为白色
    # 适配mpl 3D的边（pane、grid、tick、axis）为白色
    for axis in [dv.ax.xaxis, dv.ax.yaxis, dv.ax.zaxis]:
        # 设置坐标轴线
        axis._axinfo['axisline']['color'] = (1,1,1,1)
        # 设置坐标轴pane（背景面）颜色为透明
        axis.set_pane_color((0,0,0,0))
        # 设置网格线颜色为淡白色
        axis._axinfo['grid']['color'] =  (1,1,1,0.2)
    # 设置刻度线为白色
    dv.ax.tick_params(colors='white', which='both')

    dv.ax.set_xlabel("$x$(m)", color="white")
    dv.ax.set_ylabel("$y$(m)", color="white")
    dv.ax.set_zlabel("$z$(m)", color="white", labelpad=10)
    dv.ax.tick_params(colors='white', which='both')
    for tl in dv.ax.xaxis.get_ticklabels():
        tl.set_color('white')
    for tl in dv.ax.yaxis.get_ticklabels():
        tl.set_color('white')
    for tl in dv.ax.zaxis.get_ticklabels():
        tl.set_color('white')

    dv.fig.patch.set_facecolor('none')
    dv.fig.patch.set_alpha(0.0)
    dv.ax.set_facecolor('none')
    dv.ax.patch.set_alpha(0.0)
    dv.fig.savefig(out_name, dpi=300, transparent=True, bbox_inches='tight', pad_inches=1)

    if show:
        plt.show(block=True)
    plt.close(dv.fig)

    


def plot_multi_2d(datasets, output_dir: Path) -> None:
    ensure_output_dir(output_dir)
    # Only two subplots: X-Z and Y-Z
    # make the 2D figure flatter (wider and shorter)
    fig, axes = plt.subplots(1, 2, figsize=(24, 6))
    projections = [
        ("X-Y", lambda xi, yi, zi: (xi, yi), "$x$(m)", "$y$(m)"),
        ("Y-Z", lambda xi, yi, zi: (yi, zi), "$y$(m)", "$z$(m)"),
    ]


    for ax, (title, func, xlabel, ylabel) in zip(axes, projections):
        for label, ts, xs, ys, zs in datasets:
            # 插值与plot_xz完全一致
            x_vals = np.array(xs)
            y_vals = np.array(ys)
            z_vals = np.array(zs)
            t_vals = np.array(ts)
            traj = np.column_stack((x_vals, y_vals, z_vals))
            if traj.shape[0] < 2:
                continue
            target_spacing = 0.02
            xi = [x_vals[0]]
            yi = [y_vals[0]]
            zi = [z_vals[0]]
            ti = [t_vals[0]]
            for i in range(len(x_vals) - 1):
                x0, y0, z0 = x_vals[i], y_vals[i], z_vals[i]
                x1, y1, z1 = x_vals[i + 1], y_vals[i + 1], z_vals[i + 1]
                t0 = t_vals[i]
                t1 = t_vals[i + 1]
                dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
                seg_len = np.sqrt(dx * dx + dy * dy + dz * dz)
                if seg_len <= 0:
                    continue
                steps = max(1, int(np.ceil(seg_len / target_spacing)))
                for s in range(1, steps + 1):
                    t = s / (steps + 0.0)
                    xi.append(x0 + t * dx)
                    yi.append(y0 + t * dy)
                    zi.append(z0 + t * dz)
                    ti.append(t0 + t * (t1 - t0))
            xi = np.array(xi)
            yi = np.array(yi)
            zi = np.array(zi)
            ti = np.array(ti)
            interp_traj = np.column_stack((xi, yi, zi))
            seg_vecs = np.diff(interp_traj, axis=0)
            dt_seg = np.diff(ti)
            dt_seg[dt_seg == 0] = 1e-8
            seg_speeds = np.linalg.norm(seg_vecs, axis=1) / dt_seg  # m/s
            # 平滑速度
            window = 7
            if seg_speeds.size < window:
                window = seg_speeds.size
            if window > 1:
                kernel = np.ones(window) / window
                smooth_speeds = np.convolve(seg_speeds, kernel, mode='same')
            else:
                smooth_speeds = seg_speeds
            from matplotlib import cm
            from matplotlib.colors import Normalize
            norm = Normalize(vmin=smooth_speeds.min(), vmax=smooth_speeds.max())
            # choose colormap per dataset: HNN -> OrRd_r, LQR -> winter_r
            lab_lower = label.lower()
            if 'hnn' in lab_lower:
                cmap = cm.get_cmap('OrRd_r')
            elif 'lqr' in lab_lower:
                cmap = cm.get_cmap('winter_r')
            else:
                cmap = cm.get_cmap('OrRd_r')
            # 构建2D线段
            x_plot, y_plot = func(xi, yi, zi)
            segs = np.stack([
                np.column_stack((x_plot[:-1], y_plot[:-1])),
                np.column_stack((x_plot[1:], y_plot[1:])),
            ], axis=1)
            colors = cmap(norm(smooth_speeds))
            lc = LineCollection(segs, colors=colors, linewidths=2.2, alpha=0.95, zorder=3)
            ax.add_collection(lc)
            # 标注该轨迹的起点和终点，使用 NATURE_PURPLE 颜色
            try:
                ax.scatter(x_plot[0], y_plot[0], s=90, facecolor=NATURE_PURPLE, edgecolor='k', zorder=5)
            except Exception:
                try:
                    ax.plot(x_plot[0], y_plot[0], marker='o', markersize=8, color=NATURE_PURPLE, zorder=5)
                except Exception:
                    pass
            try:
                ax.scatter(x_plot[-1], y_plot[-1], s=120, marker='X', color=NATURE_PURPLE, zorder=6)
            except Exception:
                try:
                    ax.plot(x_plot[-1], y_plot[-1], marker='X', markersize=9, color=NATURE_PURPLE, zorder=6)
                except Exception:
                    pass
            ax.autoscale_view()
            if(title == 'X-Y'):
                ax.set_ylim([-1, 1])
            else:
                ax.set_xlim([-1,1])

        # 坐标轴、标题、标签、刻度设为白色以便与深色背景配合
        # ax.set_title(title, color='white')
        ax.set_xlabel(xlabel, color='white')
        ax.set_ylabel(ylabel, color='white')
        ax.tick_params(colors='white', which='both', width=1.8)
        for name, spine in ax.spines.items():
            spine.set_color('white')
            spine.set_linewidth(1.5)
        ax.grid(True, linestyle='--', color='white', linewidth=1.2, alpha=0.25)
        if title == "X-Z":
            ax.set_ylim([0.8, 1.5])

        # 在子图上添加 Start / End 图例（只显示一次的标识）
        try:
            start_handle = mpl.lines.Line2D([0], [0], color='none', marker='o', markerfacecolor=NATURE_PURPLE, markersize=12, linestyle='None')
            end_handle = mpl.lines.Line2D([0], [0], color='none', marker='X', markerfacecolor=NATURE_PURPLE, markersize=12, linestyle='None')
            leg = ax.legend([start_handle, end_handle], ['Start', 'End'], loc='best', frameon=False)
            # make legend text white
            try:
                for text in leg.get_texts():
                    text.set_color('white')
            except Exception:
                pass
        except Exception:
            pass

    fig.tight_layout()
    fig.savefig(output_dir / "combined_2d.svg", dpi=300, pad_inches=1)
    plt.close(fig)


def plot_xz(datasets, output_dir: Path) -> None:
    """Create a standalone X-Z projection plot and save to output_dir."""
    ensure_output_dir(output_dir)
    fig, ax = plt.subplots(figsize=(10, 5))
    
    from matplotlib.colors import Normalize

    colorbar_infos = []
    for idx, (label, ts, xs, ys, zs) in enumerate(datasets):
        lab_lower = label.lower()
        if "hnn" in lab_lower:
            base_color = NATURE_PURPLE
        elif "lqr" in lab_lower:
            base_color = NATURE_GREEN
        else:
            base_color = None

        x_vals = np.array(xs)
        z_vals = np.array(zs)
        y_vals = np.array(ys)
        t_vals = np.array(ts)

        # Interpolate trajectory to increase point density based on a target spacing
        traj = np.column_stack((x_vals, y_vals, z_vals))
        if traj.shape[0] < 2:
            continue

        target_spacing = 0.01  # meters per interpolated step (tune this for density)
        xi = [x_vals[0]]
        yi = [y_vals[0]]
        zi = [z_vals[0]]
        ti = [t_vals[0]]
        for i in range(len(x_vals) - 1):
            x0, y0, z0 = x_vals[i], y_vals[i], z_vals[i]
            x1, y1, z1 = x_vals[i + 1], y_vals[i + 1], z_vals[i + 1]
            t0 = t_vals[i]
            t1 = t_vals[i + 1]
            dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
            seg_len = np.sqrt(dx * dx + dy * dy + dz * dz)
            if seg_len <= 0:
                continue
            steps = max(1, int(np.ceil(seg_len / target_spacing)))
            # generate intermediate points (include the end point so we cover segment)
            for s in range(1, steps + 1):
                t = s / (steps + 0.0)
                xi.append(x0 + t * dx)
                yi.append(y0 + t * dy)
                zi.append(z0 + t * dz)
                ti.append(t0 + t * (t1 - t0))

        xi = np.array(xi)
        yi = np.array(yi)
        zi = np.array(zi)
        ti = np.array(ti)

        interp_traj = np.column_stack((xi, yi, zi))
        seg_vecs = np.diff(interp_traj, axis=0)
        dt_seg = np.diff(ti)
        # avoid zero dt
        dt_seg[dt_seg == 0] = 1e-8
        seg_speeds = np.linalg.norm(seg_vecs, axis=1) / dt_seg  # m/s

        # Smooth speeds to avoid abrupt color jumps (moving average)
        window = 7
        if seg_speeds.size < window:
            window = seg_speeds.size
        if window > 1:
            kernel = np.ones(window) / window
            smooth_speeds = np.convolve(seg_speeds, kernel, mode='same')
        else:
            smooth_speeds = seg_speeds

        norm = Normalize(vmin=smooth_speeds.min(), vmax=smooth_speeds.max())
        # choose colormap per dataset: HNN -> OrRd_r, LQR -> winter_r
        lab_lower = label.lower()
        if 'hnn' in lab_lower:
            cmap = mpl.cm.get_cmap('OrRd_r')
        elif 'lqr' in lab_lower:
            cmap = mpl.cm.get_cmap('winter_r')
        else:
            cmap = mpl.cm.get_cmap('OrRd_r')

        # Build 2D segments for X-Z projection and create a LineCollection from interpolated points
        segs = np.stack(
            [
                np.column_stack((xi[:-1], zi[:-1])),
                np.column_stack((xi[1:], zi[1:])),
            ],
            axis=1,
        )
        colors = cmap(norm(smooth_speeds))
        lc = LineCollection(segs, colors=colors, linewidths=3, alpha=0.95, zorder=3)
        ax.add_collection(lc)
        # autoscale to the data
        ax.autoscale_view()
        # Instead of drawing colorbars here, collect cmap/norm so a separate
        # figure can draw both colorbars together (avoids clipping/overlap).
        try:
            colorbar_infos.append({'cmap': cmap, 'norm': norm, 'label': 'Speed (m/s)'})
        except Exception:
            # ignore if we can't record colorbar info for this dataset
            pass

        # Mark start and end points: start = yellow circle, end = red X
        try:
            ax.scatter(x_vals[0], z_vals[0], s=120, facecolors="#9558B2",zorder=5)
        except Exception:
            ax.plot(x_vals[0], z_vals[0], marker='o', markersize=8, color="#9558B2", zorder=5)
        try:
            ax.scatter(x_vals[-1], z_vals[-1], s=140, marker='X', color="#9558B2", zorder=6)
        except Exception:
            ax.plot(x_vals[-1], z_vals[-1], marker='X', markersize=9, color="#9558B2", zorder=6)

    # Title and axis labels in white with larger font sizes
    # ax.set_title("X-Z Projection", color="white")
    ax.set_xlabel("$x$(m)", color="white")
    ax.set_ylabel("$z$(m)", color="white")
    ax.set_ylim([0.7, 1.5])
    ax.set_xlim([-0.5, 1.25])

    # Tick labels and tick width/size
    ax.tick_params(colors="white", which="both", labelsize=24, width=1.8)

    # Hide top and right spines; style left and bottom spines in white and thicker
    for name, spine in ax.spines.items():
        if name in ("right", "top"):
            spine.set_visible(False)
        else:
            spine.set_visible(True)
            spine.set_color("white")
            spine.set_linewidth(1.8)

    # Grid as white dashed lines with increased linewidth
    # ax.grid(True, linestyle="--", color="white", linewidth=1.8, alpha=0.6)
    ax.grid(False)

    # # Legend text to white and transparent frame with white edge
    # leg = ax.legend()
    # if leg is not None:
    #     for text in leg.get_texts():
    #         text.set_color("white")
    #     leg.get_frame().set_facecolor("none")
    #     leg.get_frame().set_edgecolor("white")

    # leave some outer margins so content is not flush with axes
    try:
        fig.tight_layout(rect=(0.12, 0.12, 0.96, 0.95))
    except Exception:
        fig.tight_layout()
    fig.savefig(output_dir / "xz_projection.svg", dpi=300, bbox_inches='tight', pad_inches=1, transparent=True)
    plt.show()
    plt.close(fig)

    # Return collected colorbar metadata for separate plotting
    return colorbar_infos


def plot_colorbars(colorbar_infos, output_dir: Path) -> None:
    """Draw the colorbars for each dataset in a separate figure (side-by-side).

    Only the right-most colorbar will show ticks and label; the left one is
    drawn as a visual companion.
    """
    if not colorbar_infos:
        return

    ensure_output_dir(output_dir)
    from matplotlib.colorbar import ColorbarBase

    # Save each colorbar as its own image file: colorbar_1.png, colorbar_2.png, ...
    for i, info in enumerate(colorbar_infos):
        try:
            # create a compact vertical figure for each colorbar
            fig, ax = plt.subplots(figsize=(1.5, 6.0))
            cb = ColorbarBase(ax, cmap=info['cmap'], norm=info['norm'])

            # show ticks for every colorbar and style them for dark background
            try:
                cb.ax.yaxis.set_tick_params(color='white', labelsize=24)
                plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')
            except Exception:
                pass

            # label each colorbar (user can ignore text if undesired)
            try:
                cb.set_label(info.get('label', ''), color='white')
            except Exception:
                pass

            try:
                cb.outline.set_edgecolor('white')
            except Exception:
                pass

            out_name = output_dir / f"colorbar_{i+1}.svg"
            try:
                fig.tight_layout()
            except Exception:
                pass
            fig.savefig(out_name, dpi=300, bbox_inches='tight', transparent=True)
            plt.close(fig)
        except Exception:
            # continue on failure to save individual colorbar
            continue


def main() -> None:
    args = parse_args()
    datasets = []
    # Default directories to look for when no explicit inputs are given
    default_dirs = ["hnn", "lqr"]

    # Build list of input files to process. If the user supplied explicit
    # --inputs, respect that. Otherwise, only include files under existing
    # `hnn` or `lqr` directories and skip missing ones quietly.
    if args.inputs:
        input_paths: List[Path] = list(args.inputs)
    else:
        input_paths = []
        for d in default_dirs:
            dpath = Path(d)
            if dpath.is_dir():
                candidate = dpath / "state_data_adjusted.txt"
                if candidate.exists():
                    input_paths.append(candidate)
                else:
                    print(f"Warning: expected file not found, skipping: {candidate}")

    target_path = Path("hnn/state_data_adjusted.txt").resolve()
    for input_path in input_paths:
        if not input_path.exists():
            print(f"Warning: input file does not exist, skipping: {input_path}")
            continue

        try:
            ts, xs, ys, zs = load_state_data(input_path)
        except ValueError as exc:
            print(f"Warning: could not parse '{input_path}', skipping: {exc}")
            continue

        # keep previous special-case adjustment for hnn/state_data_adjusted.txt
        if input_path.resolve() == target_path:
            zs = [z + 0.01 for z in zs]

        xs = [x + 0.3 for x in xs]
        prefix = input_path.parent.name
        title = f"{prefix}_{input_path.stem}" if prefix else input_path.stem
        datasets.append((title, ts, xs, ys, zs))

    if not datasets:
        print("No trajectory files found in 'hnn' or 'lqr' directories. Nothing to plot.")
        return

    plot_combined_3d(datasets, args.output_dir, show=getattr(args, 'show', False))
    plot_multi_2d(datasets, args.output_dir)
    # Also create a standalone X-Z projection image and separate colorbars image
    colorbar_infos = plot_xz(datasets, args.output_dir)
    # Draw colorbars separately (both will be placed side-by-side in a single image)
    plot_colorbars(colorbar_infos, args.output_dir)


if __name__ == "__main__":
    main()

