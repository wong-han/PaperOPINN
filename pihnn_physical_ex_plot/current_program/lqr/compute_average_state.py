from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "performance_outputs"

STATE_COLUMNS = [
    "x_m",
    "y_m",
    "z_m",
    "roll_rad",
    "pitch_rad",
    "yaw_rad",
    "vx_m_s",
    "vy_m_s",
    "vz_m_s",
]


def load_state(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter="\t", skip_header=1, encoding="utf-8-sig")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 10:
        raise ValueError(f"{path.name}: expected 10 columns, got {data.shape[1]}")
    data = data[np.all(np.isfinite(data), axis=1)]
    if data.shape[0] < 2:
        raise ValueError(f"{path.name}: expected at least two numeric rows")
    return data


def label_from_path(path: Path) -> str:
    return path.stem.removeprefix("state_data_")


def compute_average_state(path: Path) -> dict[str, object]:
    data = load_state(path)
    t = data[:, 0]
    states = data[:, 1:]
    duration = float(t[-1] - t[0])
    if duration <= 0:
        raise ValueError(f"{path.name}: duration must be positive")

    sample_mean = np.mean(states, axis=0)
    time_mean = np.trapz(states, t, axis=0) / duration

    return {
        "label": label_from_path(path),
        "samples": int(states.shape[0]),
        "duration": duration,
        "sample_mean": sample_mean,
        "time_mean": time_mean,
    }


def save_summary(results: list[dict[str, object]]) -> Path:
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "average_state_summary.csv"

    header_parts = ["label", "samples", "duration_s"]
    header_parts.extend([f"sample_mean_{name}" for name in STATE_COLUMNS])
    header_parts.extend([f"time_mean_{name}" for name in STATE_COLUMNS])

    rows = []
    for result in results:
        rows.append(
            [result["label"], result["samples"], result["duration"]]
            + list(result["sample_mean"])
            + list(result["time_mean"])
        )

    np.savetxt(
        out_path,
        np.array(rows, dtype=object),
        delimiter=",",
        header=",".join(header_parts),
        comments="",
        fmt="%s",
    )
    return out_path


def print_table(results: list[dict[str, object]]) -> None:
    print("label,samples,duration_s")
    for result in results:
        print(f"{result['label']},{result['samples']},{result['duration']:.6f}")

    print()
    print("Time-mean state values:")
    print("label," + ",".join(STATE_COLUMNS))
    for result in results:
        values = ",".join(f"{value:.6f}" for value in result["time_mean"])
        print(f"{result['label']},{values}")


def main() -> None:
    state_files = sorted(ROOT.glob("state_data_*.txt"))
    if not state_files:
        raise FileNotFoundError("No state_data_*.txt files found")

    results = [compute_average_state(path) for path in state_files]
    summary_path = save_summary(results)

    print(f"Processed {len(results)} state files")
    print(f"Summary CSV: {summary_path}")
    print()
    print_table(results)


if __name__ == "__main__":
    main()
