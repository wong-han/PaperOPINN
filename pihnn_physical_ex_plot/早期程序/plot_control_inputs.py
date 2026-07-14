"""
Plot control inputs from hnn and lqr u_data files.

Usage:
    python plot_control_inputs.py --hnn hnn/u_data.txt --lqr lqr/u_data.txt --output-dir plots [--show]

The script reads tab- or space-separated numeric files. If the first line is a header, it will be skipped.
It expects each data row to be: time <u1> <u2> <u3> <u4> (but will adapt to any number of u columns).
"""
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Tuple
from io import StringIO
import matplotlib as mpl

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
mpl.rcParams['legend.fontsize'] = 24  # legend默认size
mpl.rcParams['axes.titlesize'] = 24   # 设置title默认size
mpl.rcParams['xtick.labelsize'] = 24  # x坐标默认size
mpl.rcParams['ytick.labelsize'] = 24  # y坐标默认size
mpl.rcParams['axes.labelsize'] = 24  # 轴标题默认size
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.alpha'] = 0.5
mpl.rcParams['grid.color'] = 'gray'

NATURE_GREEN = "#00E28E"  # LQR
NATURE_PURPLE = "#D02618"  # HNN


def read_u_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read u data file robustly.

    Returns (t_values, u_values) where u_values shape = (N, n_controls)
    """
    # Read as binary and decode defensively (some files may contain non-UTF8 bytes)
    raw = path.open('rb').read()
    text = raw.decode('utf-8', errors='replace')
    # normalize common delimiters (commas -> whitespace)
    text = text.replace(',', ' ')
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        raise ValueError(f"No data in {path}")
    # detect header: try parsing first line
    first_cols = lines[0].strip().split()
    def all_float(cols):
        try:
            [float(x) for x in cols]
            return True
        except Exception:
            return False
    if not all_float(first_cols):
        data_lines = lines[1:]
    else:
        data_lines = lines
    if not data_lines:
        raise ValueError(f"No numeric data in {path}")
    s = "\n".join(data_lines)
    arr = np.genfromtxt(StringIO(s), delimiter=None)
    if arr.ndim == 0:
        raise ValueError(f"Unable to parse numeric data in {path}")
    if arr.ndim == 1:
        # single row
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 2:
        raise ValueError(f"Expected at least time and one control column in {path}")
    t = arr[:, 0]
    u = arr[:, 1:]
    return t, u


def plot_controls(hnn_path: Path, lqr_path: Path, output_dir: Path, show: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    t_hnn, u_hnn = read_u_file(hnn_path)
    t_lqr, u_lqr = read_u_file(lqr_path)

    # ensure same number of control channels (use min)
    n_ctrl = min(u_hnn.shape[1], u_lqr.shape[1], 4)

    # align lengths by using min time range
    N = min(len(t_hnn), len(t_lqr))
    t = np.arange(N)
    t_hnn = t_hnn[:N] - t_hnn[0]
    t_lqr = t_lqr[:N] - t_lqr[0]
    u_hnn = u_hnn[:N, :n_ctrl]
    u_lqr = u_lqr[:N, :n_ctrl]

    plt.figure(figsize=(12, 6))
    for i in range(n_ctrl):
        plt.plot(t_lqr, u_lqr[:, i], label=f"LQR u{i+1}", linestyle='--', color=NATURE_GREEN)
        plt.plot(t_hnn, u_hnn[:, i], label=f"Ours u{i+1}", color=NATURE_PURPLE)

    ax = plt.gca()
    # 设置白色边框
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
        # spine.set_linewidth(1.8)
    # white text and ticks for dark backgrounds
    ax.set_facecolor('none')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')

    plt.xlabel("Time step")
    plt.ylabel("u")
    plt.title("Control trajectories (single episode)")

    # place legend to the right of the plot to avoid overlapping the curves
    leg = plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    try:
        for text in leg.get_texts():
            text.set_color('white')
    except Exception:
        pass

    plt.grid(True, linestyle='--', alpha=0.4, color='white')
    out_file = output_dir / "control_traj.svg"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches='tight', transparent=True)
    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close()

    # Also create a 4-panel subplot (one row per control input), similar to
    # the example in the main repo. Use Nature colors and save as an SVG.
    control_labels = ['$T$(N)', '$p$(rad/s)', '$q$(rad/s)', '$r$(rad/s)']
    nature_green = NATURE_GREEN
    nature_purple = NATURE_PURPLE

    # create 4 subplots stacked vertically; make wider to leave room for legend
    fig, axs = plt.subplots(4, 1, figsize=(7.6, 6.3), sharex=True)
    for i, ax in enumerate(axs):
        # 设置白色边框
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
            # spine.set_linewidth(1.8)
        # draw LQR (green)
        ax.plot(t_lqr, u_lqr[:, i], color=nature_green, alpha=0.6, linewidth=3, label='LQR' if i == 0 else '')
        # draw HNN (purple)
        ax.plot(t_hnn, u_hnn[:, i], color=nature_purple, alpha=0.8, linewidth=3, label='Ours' if i == 0 else '')
        ax.set_ylabel(control_labels[i])
        ax.grid(True, linestyle='--', alpha=0.36, color='white')
        # style axis text/ticks for dark background
        ax.tick_params(colors='white')
        ax.yaxis.label.set_color('white')
        ax.set_xlim([-0.5, 10])
        if i == 0:
            # show legend on the right of the first subplot (outside axes)
            leg = ax.legend(loc='upper right', bbox_to_anchor=(1.02, 1))
            try:
                for text in leg.get_texts():
                    text.set_color('white')
            except Exception:
                pass

    axs[-1].set_xlabel('$t$ (s)')
    axs[-1].xaxis.label.set_color('white')
    try:
        fig.tight_layout()
    except Exception:
        pass
    out_svg = output_dir / "control_traj_subplots.svg"
    fig.savefig(out_svg, dpi=600, bbox_inches='tight', pad_inches=0.5, transparent=True)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Plot control inputs from hnn and lqr u_data files")
    p.add_argument('--hnn', type=Path, default=Path('hnn/u_data.txt'))
    p.add_argument('--lqr', type=Path, default=Path('lqr/u_data.txt'))
    p.add_argument('--output-dir', type=Path, default=Path('plots'))
    p.add_argument('--show', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if not args.hnn.exists():
        print(f"Warning: {args.hnn} not found")
    if not args.lqr.exists():
        print(f"Warning: {args.lqr} not found")
    plot_controls(args.hnn, args.lqr, args.output_dir, show=args.show)
