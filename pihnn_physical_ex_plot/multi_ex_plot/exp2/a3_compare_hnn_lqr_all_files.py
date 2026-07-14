from pathlib import Path
import re

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


# ================== 图像设置 =======================
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['text.usetex'] = True
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 24
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['axes.labelsize'] = 24
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.alpha'] = 0.5
mpl.rcParams['grid.color'] = 'gray'


TARGET_POS = np.array([0.0, 0.0, 0.9])
Q = np.diag([5.0, 5.0, 20.0, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1])
R = np.diag([10.0, 1.0, 1.0, 1.0])
U0_OLD = np.array([0.27, 0.0, 0.21, 0.0])
U0_NEW = np.array([0.27, 0.0, 0.35, 0.0])
J_WINDOW_SECONDS = 4.0


def state_suffix(path):
    return path.stem.removeprefix("state_data_")


def base_time_key(path, controller):
    suffix = state_suffix(path)
    controller_suffix = f"_{controller}"
    if suffix.endswith(controller_suffix):
        suffix = suffix[: -len(controller_suffix)]
    return suffix


def sort_key(path, controller):
    base = base_time_key(path, controller)
    match = re.search(r"(\d{8})_(\d{6})", base)
    if match:
        return match.group(1), match.group(2), path.name
    return base, "", path.name


def has_controller_suffix(path, controller):
    return state_suffix(path).endswith(f"_{controller}")


def split_state_files(data_dir, controller):
    files = sorted(data_dir.glob("state_data_*.txt"))
    with_suffix = [path for path in files if has_controller_suffix(path, controller)]
    without_suffix = [path for path in files if not has_controller_suffix(path, controller)]
    return {
        "without_suffix": sorted(without_suffix, key=lambda path: sort_key(path, controller)),
        "with_suffix": sorted(with_suffix, key=lambda path: sort_key(path, controller)),
    }


def matching_u_file(state_file, data_dir, controller):
    suffix = state_suffix(state_file)
    candidates = [data_dir / f"u_data_{suffix}.txt"]
    controller_suffix = f"_{controller}"
    if suffix.endswith(controller_suffix):
        candidates.append(data_dir / f"u_data_{suffix[: -len(controller_suffix)]}.txt")
    else:
        candidates.append(data_dir / f"u_data_{suffix}_{controller}.txt")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_txt(path, expected_cols):
    rows = []
    skipped = 0
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        next(handle, None)
        for line in handle:
            if "\0" in line:
                line = line.replace("\0", "")
            parts = line.strip().split("\t")
            if len(parts) != expected_cols:
                skipped += 1
                continue
            try:
                row = [float(part) for part in parts]
            except ValueError:
                skipped += 1
                continue
            if all(np.isfinite(row)):
                rows.append(row)
            else:
                skipped += 1

    if skipped:
        print(f"  Skipped {skipped} invalid row(s) in {path.name}")
    if not rows:
        raise ValueError(f"{path.name}: no valid numeric rows")
    return np.array(rows, dtype=float)


def compute_performance(state_data, u_data, q, r, target_pos, u0, end_time_offset=J_WINDOW_SECONDS):
    t_state = state_data[:, 0]
    x_state = state_data[:, 1:]
    t_u = u_data[:, 0]
    u_vals = u_data[:, 1:]
    t_start = t_u[0]

    if end_time_offset is not None:
        mask = (t_state >= t_start) & (t_state <= t_start + end_time_offset)
    else:
        mask = t_state >= t_start

    t_filtered = t_state[mask]
    x_filtered = x_state[mask, :]
    if len(t_filtered) < 2:
        return np.nan, 0.0

    u_interp = np.column_stack(
        [np.interp(t_filtered, t_u, u_vals[:, idx]) for idx in range(u_vals.shape[1])]
    )

    x_eq = np.zeros(9)
    x_eq[:3] = target_pos
    x_error = x_filtered - x_eq
    u_error = u_interp - u0.reshape(1, -1) if u0 is not None else u_interp

    x_q_x = np.einsum("ij,jk,ik->i", x_error, q, x_error)
    u_r_u = np.einsum("ij,jk,ik->i", u_error, r, u_error)
    integrand = x_q_x + u_r_u
    dt = np.diff(t_filtered)
    j_value = np.sum(0.5 * (integrand[:-1] + integrand[1:]) * dt)
    return float(j_value), float(t_filtered[-1] - t_filtered[0])


def compute_pos_mse(state_data, u_data, target_pos):
    t_state = state_data[:, 0]
    pos = state_data[:, 1:4]
    t_start = u_data[0, 0]
    pos_filtered = pos[t_state >= t_start, :]
    if len(pos_filtered) == 0:
        return np.nan
    return float(np.mean(np.sum((pos_filtered - target_pos) ** 2, axis=1)))


def pct_improvement(lqr_value, hnn_value):
    if not np.isfinite(lqr_value) or not np.isfinite(hnn_value) or lqr_value == 0:
        return np.nan
    return float((lqr_value - hnn_value) / lqr_value * 100.0)


def make_pairs(root):
    hnn_dir = root / "hnn"
    lqr_dir = root / "lqr"
    hnn_groups = split_state_files(hnn_dir, "hnn")
    lqr_groups = split_state_files(lqr_dir, "lqr")

    pairs = []
    skipped = []
    group_names = [
        ("without_suffix", "no_suffix"),
        ("with_suffix", "with_suffix"),
    ]
    for group_key, group_label in group_names:
        hnn_files = hnn_groups[group_key]
        lqr_files = lqr_groups[group_key]
        pair_count = min(len(hnn_files), len(lqr_files))
        if len(hnn_files) != len(lqr_files):
            skipped.append(
                (
                    group_label,
                    f"HNN has {len(hnn_files)} file(s), LQR has {len(lqr_files)} file(s); "
                    f"using first {pair_count} time-ordered pair(s)",
                )
            )
        for idx in range(pair_count):
            pairs.append(
                {
                    "label": f"{group_label}_{idx + 1}",
                    "group": group_label,
                    "pair_index": idx + 1,
                    "hnn_state_file": hnn_files[idx],
                    "lqr_state_file": lqr_files[idx],
                    "hnn_u_file": matching_u_file(hnn_files[idx], hnn_dir, "hnn"),
                    "lqr_u_file": matching_u_file(lqr_files[idx], lqr_dir, "lqr"),
                }
            )
    return pairs, skipped


def compute_pair(pair):
    if pair["hnn_u_file"] is None:
        raise FileNotFoundError(f"No matching u file for {pair['hnn_state_file'].name}")
    if pair["lqr_u_file"] is None:
        raise FileNotFoundError(f"No matching u file for {pair['lqr_state_file'].name}")

    hnn_state = load_txt(pair["hnn_state_file"], expected_cols=10)
    lqr_state = load_txt(pair["lqr_state_file"], expected_cols=10)
    hnn_u = load_txt(pair["hnn_u_file"], expected_cols=5)
    lqr_u = load_txt(pair["lqr_u_file"], expected_cols=5)

    hnn_mse = compute_pos_mse(hnn_state, hnn_u, TARGET_POS)
    lqr_mse = compute_pos_mse(lqr_state, lqr_u, TARGET_POS)
    hnn_j4_old, hnn_j4_duration = compute_performance(hnn_state, hnn_u, Q, R, TARGET_POS, U0_OLD)
    lqr_j4_old, lqr_j4_duration = compute_performance(lqr_state, lqr_u, Q, R, TARGET_POS, U0_OLD)
    hnn_j4_new, _ = compute_performance(hnn_state, hnn_u, Q, R, TARGET_POS, U0_NEW)
    lqr_j4_new, _ = compute_performance(lqr_state, lqr_u, Q, R, TARGET_POS, U0_NEW)
    hnn_j4_no_u0, _ = compute_performance(hnn_state, hnn_u, Q, R, TARGET_POS, None)
    lqr_j4_no_u0, _ = compute_performance(lqr_state, lqr_u, Q, R, TARGET_POS, None)
    hnn_j_full_old, hnn_full_duration = compute_performance(
        hnn_state, hnn_u, Q, R, TARGET_POS, U0_OLD, end_time_offset=None
    )
    lqr_j_full_old, lqr_full_duration = compute_performance(
        lqr_state, lqr_u, Q, R, TARGET_POS, U0_OLD, end_time_offset=None
    )

    return {
        "label": pair["label"],
        "group": pair["group"],
        "pair_index": pair["pair_index"],
        "hnn_state_file": pair["hnn_state_file"].name,
        "hnn_u_file": pair["hnn_u_file"].name,
        "lqr_state_file": pair["lqr_state_file"].name,
        "lqr_u_file": pair["lqr_u_file"].name,
        "hnn_mse": hnn_mse,
        "lqr_mse": lqr_mse,
        "mse_improvement_pct": pct_improvement(lqr_mse, hnn_mse),
        "hnn_j_4s_u0_old": hnn_j4_old,
        "lqr_j_4s_u0_old": lqr_j4_old,
        "j_4s_u0_old_improvement_pct": pct_improvement(lqr_j4_old, hnn_j4_old),
        "hnn_j_4s_u0_new": hnn_j4_new,
        "lqr_j_4s_u0_new": lqr_j4_new,
        "j_4s_u0_new_improvement_pct": pct_improvement(lqr_j4_new, hnn_j4_new),
        "hnn_j_4s_no_u0": hnn_j4_no_u0,
        "lqr_j_4s_no_u0": lqr_j4_no_u0,
        "j_4s_no_u0_improvement_pct": pct_improvement(lqr_j4_no_u0, hnn_j4_no_u0),
        "hnn_j_full_u0_old": hnn_j_full_old,
        "lqr_j_full_u0_old": lqr_j_full_old,
        "j_full_u0_old_improvement_pct": pct_improvement(lqr_j_full_old, hnn_j_full_old),
        "hnn_j_4s_duration": hnn_j4_duration,
        "lqr_j_4s_duration": lqr_j4_duration,
        "hnn_full_duration": hnn_full_duration,
        "lqr_full_duration": lqr_full_duration,
    }


def save_csv(results, output_dir):
    path = output_dir / "hnn_lqr_all_files_summary.csv"
    headers = list(results[0].keys())
    rows = [[result[header] for header in headers] for result in results]
    np.savetxt(path, np.array(rows, dtype=object), delimiter=",", header=",".join(headers), comments="", fmt="%s")
    return path


def numeric_metric_keys(results):
    excluded = {
        "pair_index",
    }
    keys = []
    for key, value in results[0].items():
        if key in excluded:
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            keys.append(key)
    return keys


def stats_for_results(results, scope):
    rows = []
    for key in numeric_metric_keys(results):
        values = np.array([float(result[key]) for result in results], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            rows.append([scope, key, 0, np.nan, np.nan])
        else:
            rows.append([scope, key, int(values.size), float(np.mean(values)), float(np.var(values))])
    return rows


def save_statistics(results, output_dir):
    rows = stats_for_results(results, "all")
    for group in sorted({result["group"] for result in results}):
        group_results = [result for result in results if result["group"] == group]
        rows.extend(stats_for_results(group_results, group))

    path = output_dir / "hnn_lqr_all_files_statistics.csv"
    np.savetxt(
        path,
        np.array(rows, dtype=object),
        delimiter=",",
        header="scope,metric,count,mean,variance",
        comments="",
        fmt="%s",
    )
    return path, rows


HNN_COLOR = "#9558B2"
LQR_COLOR = "#389826"
IMPROVEMENT_COLOR = "#1F77B4"


def plot_metric_with_improvement(
    axis,
    labels,
    hnn_values,
    lqr_values,
    improvement_values,
    ylabel,
    title,
):
    x = np.arange(len(labels))
    width = 0.38
    bars_hnn = axis.bar(x - width / 2, hnn_values, width, color=HNN_COLOR, label="OPINN")
    bars_lqr = axis.bar(x + width / 2, lqr_values, width, color=LQR_COLOR, label="LQR")
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylabel(ylabel)
    # axis.set_title(title)
    axis.grid(True, axis="y", linestyle=":", alpha=0.5)

    axis_right = axis.twinx()
    (line_improvement,) = axis_right.plot(
        x,
        improvement_values,
        color=IMPROVEMENT_COLOR,
        marker="o",
        linewidth=2.0,
        label="Improvement",
    )
    axis_right.axhline(0.0, color=IMPROVEMENT_COLOR, linewidth=1.0, alpha=0.35)
    axis_right.set_ylabel(r"Improvement (\%)", color=IMPROVEMENT_COLOR)
    axis_right.tick_params(axis="y", colors=IMPROVEMENT_COLOR)
    axis_right.spines["right"].set_color(IMPROVEMENT_COLOR)

    handles = [bars_hnn, bars_lqr, line_improvement]
    labels_legend = [handle.get_label() for handle in handles]
    axis.legend(handles, labels_legend, loc="best")


def save_plot(results, output_dir):
    labels = [f"case{idx}" for idx in range(1, len(results) + 1)]
    hnn_mse = np.array([result["hnn_mse"] for result in results], dtype=float)
    lqr_mse = np.array([result["lqr_mse"] for result in results], dtype=float)
    hnn_j_full = np.array([result["hnn_j_full_u0_old"] for result in results], dtype=float)
    lqr_j_full = np.array([result["lqr_j_full_u0_old"] for result in results], dtype=float)
    mse_imp = np.array([result["mse_improvement_pct"] for result in results], dtype=float)
    j_full_imp = np.array([result["j_full_u0_old_improvement_pct"] for result in results], dtype=float)

    fig1, ax1 = plt.subplots(figsize=(9, 6))
    plot_metric_with_improvement(
        ax1,
        labels,
        hnn_mse,
        lqr_mse,
        mse_imp,
        "Position MSE",
        "Position MSE",
    )
    fig1.tight_layout()
    path_mse = output_dir / "hnn_lqr_all_files_mse.svg"
    fig1.savefig(path_mse, dpi=300, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(9, 6))
    plot_metric_with_improvement(
        ax2,
        labels,
        hnn_j_full,
        lqr_j_full,
        j_full_imp,
        "$J$",
        f"J from start to end",
    )
    fig2.tight_layout()
    path_j = output_dir / "hnn_lqr_all_files_j_full.svg"
    fig2.savefig(path_j, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    
    return path_mse, path_j


def main():
    root = Path(__file__).resolve().parent
    output_dir = root / "compare_hnn_lqr_all_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs, pairing_notes = make_pairs(root)
    if not pairs:
        raise RuntimeError("No HNN/LQR state file pairs found")

    results = []
    skipped = []
    for pair in pairs:
        try:
            print(
                f"\nProcessing {pair['label']}: "
                f"{pair['hnn_state_file'].name} <-> {pair['lqr_state_file'].name}"
            )
            results.append(compute_pair(pair))
        except (FileNotFoundError, ValueError) as exc:
            skipped.append((pair["label"], str(exc)))
            print(f"Skipped {pair['label']}: {exc}")

    if not results:
        raise RuntimeError("No valid HNN/LQR pairs were processed")

    csv_path = save_csv(results, output_dir)
    stats_path, stats_rows = save_statistics(results, output_dir)
    plot_path_mse, plot_path_j = save_plot(results, output_dir)

    print("\nlabel,group,HNN_MSE,LQR_MSE,MSE_imp_pct,HNN_J4,LQR_J4,J4_imp_pct,HNN_Jfull,LQR_Jfull,Jfull_imp_pct")
    for result in results:
        print(
            f"{result['label']},{result['group']},"
            f"{result['hnn_mse']:.6f},{result['lqr_mse']:.6f},{result['mse_improvement_pct']:.2f},"
            f"{result['hnn_j_4s_u0_old']:.6f},{result['lqr_j_4s_u0_old']:.6f},"
            f"{result['j_4s_u0_old_improvement_pct']:.2f},"
            f"{result['hnn_j_full_u0_old']:.6f},{result['lqr_j_full_u0_old']:.6f},"
            f"{result['j_full_u0_old_improvement_pct']:.2f}"
        )

    if pairing_notes:
        print("\nPairing notes:")
        for label, note in pairing_notes:
            print(f"  {label}: {note}")
    if skipped:
        print("\nSkipped:")
        for label, reason in skipped:
            print(f"  {label}: {reason}")

    print("\nOverall mean/variance:")
    for scope, metric, count, mean, variance in stats_rows:
        if scope == "all":
            print(f"  {metric}: count={count}, mean={mean:.6f}, variance={variance:.6f}")

    print(f"\nCSV summary: {csv_path}")
    print(f"Statistics CSV: {stats_path}")
    print(f"MSE plot: {plot_path_mse}")
    print(f"J_full plot: {plot_path_j}")


if __name__ == "__main__":
    main()
