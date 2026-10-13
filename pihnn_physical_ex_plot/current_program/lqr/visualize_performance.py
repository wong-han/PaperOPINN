from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "performance_outputs"

X_EQ = np.array([0.55, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
Q_DIAG = np.array([5, 5, 20, 0.1, 0.1, 1, 0.1, 0.1, 0.1], dtype=float)
R_DIAG = np.array([10, 1, 1, 1], dtype=float)
WINDOW_SECONDS = 5.0


def load_txt(path: Path, expected_cols: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        data = np.genfromtxt(path, delimiter="\t", skip_header=1, encoding="utf-8-sig")
    if data.size == 0:
        raise ValueError(f"{path.name}: no numeric rows found")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != expected_cols:
        raise ValueError(f"{path.name}: expected {expected_cols} columns, got {data.shape[1]}")
    data = data[np.all(np.isfinite(data), axis=1)]
    if data.size == 0:
        raise ValueError(f"{path.name}: no numeric rows found")
    return data


def timestamp_from_name(path: Path, prefix: str) -> str:
    return path.stem.removeprefix(prefix)


def pair_files() -> list[tuple[str, Path, Path]]:
    state_files = {
        timestamp_from_name(path, "state_data_"): path
        for path in ROOT.glob("state_data_*.txt")
    }
    u_files = {
        timestamp_from_name(path, "u_data_"): path
        for path in ROOT.glob("u_data_*.txt")
    }
    common = sorted(state_files.keys() & u_files.keys())
    if not common:
        raise FileNotFoundError("No matching state_data_*.txt and u_data_*.txt pairs found")
    return [(key, state_files[key], u_files[key]) for key in common]


def compute_pair(label: str, state_path: Path, u_path: Path) -> dict[str, object]:
    state = load_txt(state_path, expected_cols=10)
    control = load_txt(u_path, expected_cols=5)

    t_state = state[:, 0]
    x = state[:, 1:]
    t_u = control[:, 0]
    u = control[:, 1:]

    mse_mask = t_state >= t_u[0]
    if np.count_nonzero(mse_mask) < 2:
        raise ValueError(f"{label}: fewer than two state samples after the first u timestamp")
    t_mse = t_state[mse_mask]
    x_mse = x[mse_mask]
    mse_duration = float(t_mse[-1] - t_mse[0])
    if mse_duration <= 0:
        raise ValueError(f"{label}: MSE duration must be positive")

    position_error_to_target = x_mse[:, :3] - X_EQ[:3]
    target_position_mse_series = np.mean(position_error_to_target**2, axis=1)
    target_position_mse_integral = float(np.trapz(target_position_mse_series, t_mse))
    target_position_time_mse = target_position_mse_integral / mse_duration
    target_position_sample_mse = float(np.mean(target_position_mse_series))

    start = max(t_state[0], t_u[0])
    end = min(t_state[-1], t_u[-1])
    window_end = min(end, start + WINDOW_SECONDS)
    mask = (t_state >= start) & (t_state <= window_end)
    if np.count_nonzero(mask) < 2:
        raise ValueError(f"{label}: fewer than two samples in the first {WINDOW_SECONDS:g} seconds")

    t = t_state[mask]
    x_overlap = x[mask]
    u_interp = np.column_stack([np.interp(t, t_u, u[:, i]) for i in range(u.shape[1])])

    x_error = x_overlap - X_EQ
    state_cost = np.sum((x_error**2) * Q_DIAG, axis=1)
    control_cost = np.sum((u_interp**2) * R_DIAG, axis=1)
    total_cost = state_cost + control_cost
    relative_time = t - t[0]

    integrated_cost = float(np.trapz(total_cost, t))
    duration = float(t[-1] - t[0])
    mean_cost = float(np.mean(total_cost))

    detail = np.column_stack(
        [relative_time, t, state_cost, control_cost, total_cost, x_error, u_interp]
    )
    header = (
        "relative_time_s,absolute_time_s,state_cost,control_cost,total_cost,"
        "x_err,y_err,z_err,roll_err,pitch_err,yaw_err,vx_err,vy_err,vz_err,"
        "u_roll_rate,u_pitch_rate,u_yaw_rate,u_thrust"
    )
    detail_path = OUT_DIR / f"performance_{label}.csv"
    np.savetxt(detail_path, detail, delimiter=",", header=header, comments="", fmt="%.10g")

    return {
        "label": label,
        "relative_time": relative_time,
        "state_cost": state_cost,
        "control_cost": control_cost,
        "total_cost": total_cost,
        "duration": duration,
        "integrated_cost": integrated_cost,
        "mean_cost": mean_cost,
        "max_cost": float(np.max(total_cost)),
        "final_cost": float(total_cost[-1]),
        "mse_duration": mse_duration,
        "target_position_mse_integral": target_position_mse_integral,
        "target_position_time_mse": target_position_time_mse,
        "target_position_sample_mse": target_position_sample_mse,
        "samples": int(t.size),
        "detail_path": detail_path,
    }


def save_summary(results: list[dict[str, object]]) -> Path:
    summary_path = OUT_DIR / "performance_summary.csv"
    summary = np.array(
        [
            [
                result["label"],
                result["samples"],
                result["duration"],
                result["integrated_cost"],
                result["mean_cost"],
                result["max_cost"],
                result["final_cost"],
                result["mse_duration"],
                result["target_position_mse_integral"],
                result["target_position_time_mse"],
                result["target_position_sample_mse"],
                result["detail_path"],
            ]
            for result in results
        ],
        dtype=object,
    )
    header = (
        "label,samples,duration_s,integrated_cost,mean_cost,max_cost,final_cost,"
        "mse_duration_s,target_position_mse_integral,target_position_time_mse,"
        "target_position_sample_mse,detail_csv"
    )
    np.savetxt(summary_path, summary, delimiter=",", header=header, comments="", fmt="%s")
    return summary_path


def plot_summary(results: list[dict[str, object]]) -> Path:
    labels = [str(result["label"])[-6:] for result in results]
    integrated = np.array([float(result["integrated_cost"]) for result in results])
    target_sample_mse = np.array(
        [float(result["target_position_sample_mse"]) for result in results]
    )

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    axes[0].bar(labels, integrated, color="#3377a8")
    axes[0].set_ylabel("Integrated cost")
    axes[0].set_title(
        f"Integrated cost, first {WINDOW_SECONDS:g} s "
        f"(mean={np.mean(integrated):.4f}, var={np.var(integrated):.4f})"
    )

    axes[1].bar(labels, target_sample_mse, color="#8a6bb8")
    axes[1].set_ylabel("Sample MSE")
    axes[1].set_title(
        "position MSE to target [0.55, 0, 0.9], from first u timestamp "
        f"(mean={np.mean(target_sample_mse):.4f}, var={np.var(target_sample_mse):.4f})"
    )
    axes[1].set_xlabel("Log time (HHMMSS)")

    for result in results:
        label = str(result["label"])[-6:]
        axes[2].plot(result["relative_time"], result["total_cost"], linewidth=1.4, label=label)
    axes[2].set_title(r"Total performance metric over time")
    axes[2].set_ylabel(r"$x_e^TQx_e + u^TRu$")
    axes[2].set_xlabel("Time since overlap start (s)")
    axes[2].legend(ncol=5, fontsize=8, title="HHMMSS")

    for axis in axes[:2]:
        axis.grid(True, axis="y", alpha=0.3)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = OUT_DIR / "performance_summary.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    results = []
    skipped = []
    for label, state_path, u_path in pair_files():
        try:
            results.append(compute_pair(label, state_path, u_path))
        except ValueError as exc:
            skipped.append((label, str(exc)))

    if not results:
        raise RuntimeError("No valid file pairs were processed")

    summary_path = save_summary(results)
    summary_plot_path = plot_summary(results)

    print(f"Processed {len(results)} file pairs")
    if skipped:
        print(f"Skipped {len(skipped)} file pairs")
        for label, reason in skipped:
            print(f"  {label}: {reason}")
    print(f"Summary CSV: {summary_path}")
    print(f"Summary plot: {summary_plot_path}")
    print()
    print(
        "label,samples,duration_s,integrated_cost,mean_cost,max_cost,final_cost,"
        "target_position_time_mse,target_position_sample_mse"
    )
    for result in results:
        print(
            f"{result['label']},{result['samples']},{result['duration']:.6f},"
            f"{result['integrated_cost']:.6f},{result['mean_cost']:.6f},"
            f"{result['max_cost']:.6f},{result['final_cost']:.6f},"
            f"{result['target_position_time_mse']:.6f},"
            f"{result['target_position_sample_mse']:.6f}"
        )


if __name__ == "__main__":
    main()
