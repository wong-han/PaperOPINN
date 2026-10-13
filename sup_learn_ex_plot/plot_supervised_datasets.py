"""Visualize the quadratic and constrained supervised-learning datasets.

Run this file directly. It reads the four ``.npy`` files in ``data`` and
creates one overview figure for each dataset in ``figures/datasets``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "figures" / "datasets"

# The closed-loop simulation step is env.dt = 0.01 s.
# (MPC_DT = 0.2 s was only the internal prediction interval of the MPC solver).
DT = 0.01

# Plot a representative, evenly spaced subset so that the curves remain clear.
# Set this to None if all trajectories should be drawn.
MAX_TRAJECTORIES_TO_PLOT = 200
SHOW_FIGURES = False
FIGURE_DPI = 300

DATASETS = {
    "quadratic": {
        "title": "Quadratic dataset",
        "state_file": DATA_DIR / "X_trajs.npy",
        "control_file": DATA_DIR / "U_trajs.npy",
        "color": "tab:blue",
    },
    "constrained": {
        "title": "Constrained dataset",
        "state_file": DATA_DIR / "X_trajs_constrained.npy",
        "control_file": DATA_DIR / "U_trajs_constrained.npy",
        "color": "tab:orange",
    },
}

STATE_GROUPS = (
    (slice(0, 3), ("x", "y", "z"), "Position", "Position (m)"),
    (slice(3, 6), ("vx", "vy", "vz"), "Velocity", "Velocity (m/s)"),
)
CONTROL_LABELS = ("thrust", "p rate", "q rate", "r rate")


def load_trajectories(state_path, control_path):
    """Load and validate one state/control trajectory pair."""
    if not state_path.exists() or not control_path.exists():
        raise FileNotFoundError(
            f"Dataset not found. Expected:\n  {state_path}\n  {control_path}\n"
            "Run p4_generate_quadrotor3d_TR_datasets_parallel.py first."
        )

    states = np.load(state_path, allow_pickle=True)
    controls = np.load(control_path, allow_pickle=True)
    if len(states) != len(controls) or len(states) == 0:
        raise ValueError("State and control files contain inconsistent trajectories.")

    states = [np.asarray(item, dtype=float) for item in states]
    controls = [np.asarray(item, dtype=float) for item in controls]
    for index, (state, control) in enumerate(zip(states, controls)):
        if state.ndim != 2 or state.shape[1] != 9:
            raise ValueError(f"Trajectory {index} has invalid state shape {state.shape}.")
        if control.ndim != 2 or control.shape[1] != 4:
            raise ValueError(f"Trajectory {index} has invalid control shape {control.shape}.")
        if len(state) != len(control):
            raise ValueError(
                f"Trajectory {index} has {len(state)} states but {len(control)} controls."
            )
    return states, controls


def select_indices(number_of_trajectories):
    """Select trajectories uniformly throughout the saved dataset."""
    if (
        MAX_TRAJECTORIES_TO_PLOT is None
        or MAX_TRAJECTORIES_TO_PLOT >= number_of_trajectories
    ):
        return np.arange(number_of_trajectories, dtype=int)
    return np.linspace(
        0, number_of_trajectories - 1, MAX_TRAJECTORIES_TO_PLOT, dtype=int
    )


def plot_multivariable_trajectories(
    axis, trajectories, indices, component_slice, labels, title, ylabel
):
    colors = plt.get_cmap("tab10").colors
    component_indices = range(component_slice.start, component_slice.stop)
    for component_index, label, color in zip(component_indices, labels, colors):
        for trajectory_index in indices:
            values = trajectories[trajectory_index][:, component_index]
            time = np.arange(len(values)) * DT
            axis.plot(time, values, color=color, alpha=0.16, linewidth=0.75)
        # Empty labeled line gives one clean legend entry per variable.
        axis.plot([], [], color=color, linewidth=2.0, label=label)

    axis.set_title(title)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel(ylabel)
    axis.legend(ncol=len(labels), loc="best")
    axis.grid(True, linestyle="--", alpha=0.35)


def plot_dataset(dataset_name, settings):
    states, controls = load_trajectories(
        settings["state_file"], settings["control_file"]
    )
    indices = select_indices(len(states))

    figure = plt.figure(figsize=(14, 13), constrained_layout=True)
    grid = figure.add_gridspec(3, 2)
    state_axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
    ]
    control_axis = figure.add_subplot(grid[1, 0])
    duration_axis = figure.add_subplot(grid[1, 1])
    path_axis = figure.add_subplot(grid[2, :], projection="3d")

    for axis, group in zip(state_axes, STATE_GROUPS):
        component_slice, labels, title, ylabel = group
        plot_multivariable_trajectories(
            axis, states, indices, component_slice, labels, title, ylabel
        )

    plot_multivariable_trajectories(
        control_axis,
        controls,
        indices,
        slice(0, 4),
        CONTROL_LABELS,
        "Control inputs",
        "Control value",
    )

    for trajectory_index in indices:
        state = states[trajectory_index]
        path_axis.plot(
            state[:, 0],
            state[:, 1],
            state[:, 2],
            color=settings["color"],
            alpha=0.22,
            linewidth=0.9,
        )
    starts = np.asarray([states[index][0, :3] for index in indices])
    ends = np.asarray([states[index][-1, :3] for index in indices])
    path_axis.scatter(
        starts[:, 0], starts[:, 1], starts[:, 2],
        color="tab:green", s=12, alpha=0.8, label="start",
    )
    path_axis.scatter(
        ends[:, 0], ends[:, 1], ends[:, 2],
        color="tab:red", s=12, alpha=0.8, label="end",
    )
    path_axis.set_title("3D position trajectories")
    path_axis.set_xlabel("x (m)")
    path_axis.set_ylabel("y (m)")
    path_axis.set_zlabel("z (m)")
    path_axis.legend()

    lengths = np.asarray([len(trajectory) for trajectory in states])
    duration_axis.hist(lengths * DT, bins=20, color=settings["color"], alpha=0.8)
    duration_axis.axvline(
        lengths.mean() * DT,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"mean = {lengths.mean() * DT:.1f} s",
    )
    duration_axis.set_title("Trajectory duration distribution")
    duration_axis.set_xlabel("Duration (s)")
    duration_axis.set_ylabel("Number of trajectories")
    duration_axis.grid(True, axis="y", linestyle="--", alpha=0.35)
    duration_axis.legend()

    figure.suptitle(
        f"{settings['title']} — {len(states)} trajectories "
        f"({len(indices)} displayed)",
        fontsize=18,
    )
    output_path = OUTPUT_DIR / f"{dataset_name}_dataset_curves.png"
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")

    all_states = np.vstack(states)
    all_controls = np.vstack(controls)
    print(f"\n[{dataset_name}]")
    print(f"  trajectories: {len(states)}; samples: {len(all_states)}")
    print(
        f"  length range: {lengths.min()}-{lengths.max()} steps "
        f"({lengths.min() * DT:.1f}-{lengths.max() * DT:.1f} s)"
    )
    print(f"  state range:   {all_states.min(axis=0)} to {all_states.max(axis=0)}")
    print(f"  control range: {all_controls.min(axis=0)} to {all_controls.max(axis=0)}")
    print(f"  saved figure:  {output_path}")
    return figure


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figures = [
        plot_dataset(dataset_name, settings)
        for dataset_name, settings in DATASETS.items()
    ]
    if SHOW_FIGURES:
        plt.show()
    else:
        for figure in figures:
            plt.close(figure)


if __name__ == "__main__":
    main()
