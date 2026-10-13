"""Quadrotor PPO seed experiments using the original parallel training loop."""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from p1_linear2d_opinn_test import DEFAULT_SEEDS, METHOD_COLORS, set_seed

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def numeric(values):
    return np.asarray([np.nan if v is None else v for v in values], dtype=float)


def save_curves(record, path):
    curve = record["learning_curve"]
    np.savez_compressed(
        path, environment_interactions=curve["environment_interactions"],
        rollout_mean_returns=numeric(curve["rollout_mean_returns"]),
        rolling_mean_returns=numeric(curve["rolling_mean_returns"]),
        completed_episode_counts=curve["completed_episode_counts"],
    )


def smooth_returns(values, alpha):
    """Causal EMA per seed; leave gaps missing and restart after each gap."""
    smoothed = np.full_like(values, np.nan, dtype=float)
    for i, row in enumerate(values):
        previous = np.nan
        for j, value in enumerate(row):
            if not np.isfinite(value):
                previous = np.nan
                continue
            previous = value if not np.isfinite(previous) else alpha * value + (1 - alpha) * previous
            smoothed[i, j] = previous
    return smoothed


def plot_results(records, output, smooth_alpha=0.25):
    # Do not interpolate missing returns or invent episodes that have not ended.
    steps = np.unique(np.concatenate([
        r["learning_curve"]["environment_interactions"] for r in records
    ]))
    values = np.full((len(records), len(steps)), np.nan)
    for i, record in enumerate(records):
        curve = record["learning_curve"]
        values[i, np.searchsorted(steps, curve["environment_interactions"])] = numeric(curve["rolling_mean_returns"])
    counts = np.sum(np.isfinite(values), axis=0)
    mean, sd = np.full(len(steps), np.nan), np.full(len(steps), np.nan)
    for j in np.flatnonzero(counts):
        finite = values[np.isfinite(values[:, j]), j]
        mean[j], sd[j] = finite.mean(), finite.std(ddof=0)
    display_values = smooth_returns(values, smooth_alpha)
    display_mean, display_sd = np.full(len(steps), np.nan), np.full(len(steps), np.nan)
    for j in np.flatnonzero(counts):
        finite = display_values[np.isfinite(display_values[:, j]), j]
        display_mean[j], display_sd[j] = finite.mean(), finite.std(ddof=0)
    np.savez_compressed(output / "aggregate_curves.npz", environment_interactions=steps,
                        mean_return=mean, sd_return=sd, per_seed_returns=values,
                        contributing_seeds=counts, seeds=[r["seed"] for r in records],
                        smoothed_mean_return=display_mean, smoothed_sd_return=display_sd,
                        smoothed_per_seed_returns=display_values, smooth_alpha=smooth_alpha)
    save_json(output / "plot_settings.json", {
        "return_mean_smoothing": "causal per-seed exponential moving average",
        "alpha": smooth_alpha, "sd": "population SD across smoothed seed curves",
        "raw_data_and_summary_unchanged": True,
        "stop_points": "smoothed final return in return_mean; raw final return in return_individual",
    })
    color = METHOD_COLORS["RL"]
    with plt.rc_context({"font.family": "Times New Roman", "font.size": 14,
                         "axes.labelsize": 16, "legend.fontsize": 14,
                         "xtick.labelsize": 14, "ytick.labelsize": 14,
                         "legend.frameon": False, "text.usetex": False}):
        for kind in ("return_individual", "return_mean", "contributing_seeds"):
            fig, ax = plt.subplots(figsize=(7, 3))
            if kind == "return_individual":
                for record in records:
                    curve = record["learning_curve"]
                    ax.plot(curve["environment_interactions"], numeric(curve["rolling_mean_returns"]),
                            color=color, alpha=0.5, linewidth=1, label=f"Seed {record['seed']}")
            elif kind == "return_mean":
                ax.plot(steps, display_mean, color=color, label="Mean Return")
                ax.fill_between(steps, display_mean - display_sd, display_mean + display_sd,
                                color=color, alpha=0.2, label="±SD")
            else:
                ax.step(steps, counts, where="post", color=color)
                ax.set_ylim(0, len(records) + 0.5)
                ax.set_yticks(range(len(records) + 1))
            if kind != "contributing_seeds":
                ends = [(r["learning_curve"]["environment_interactions"][-1],
                         display_values[i, np.searchsorted(steps, r["learning_curve"]["environment_interactions"][-1])]
                         if kind == "return_mean" else r["learning_curve"]["rolling_mean_returns"][-1])
                        for i, r in enumerate(records)
                        if r["learning_curve"]["rolling_mean_returns"][-1] is not None]
                if ends:
                    ax.scatter(*zip(*ends), color=color, edgecolors="white", s=40,
                               zorder=4, label="Stop Points")
                ax.legend()
            ax.set_xlabel("Environment Interactions")
            ax.set_ylabel("Contributing Seeds" if kind == "contributing_seeds" else "Mean Episodic Return")
            ax.set_xscale("linear")
            ax.set_yscale("linear")
            ax.grid(True, linestyle="--", alpha=0.3)
            fig.tight_layout()
            for extension in ("pdf", "png", "svg"):
                fig.savefig(output / f"{kind}.{extension}", dpi=300, bbox_inches="tight")
            plt.close(fig)
    if not np.any(counts):
        print("No episodes finished yet; return curves are empty (not zero returns).")


def main(source_name, script_file):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--steps", type=int, help="Requested environment interactions, rounded up to full rollouts")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--steps-per-env", type=int)
    parser.add_argument("--minibatch-size", type=int)
    parser.add_argument("--updates-per-rollout", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--smooth-alpha", type=float, default=0.25,
                        help="Display-only EMA for return_mean (0 < alpha <= 1; 1 disables smoothing)")
    parser.add_argument("--retrain", action="store_true")
    options = parser.parse_args()
    if not 0 < options.smooth_alpha <= 1:
        parser.error("--smooth-alpha must satisfy 0 < alpha <= 1.")
    if len(set(options.seeds)) != len(options.seeds) or any(s < 0 for s in options.seeds):
        parser.error("Seeds must be distinct nonnegative integers.")
    if options.plot_only and options.retrain:
        parser.error("--plot-only and --retrain cannot be combined.")
    for field in ("steps", "num_envs", "steps_per_env", "minibatch_size", "updates_per_rollout"):
        if getattr(options, field) is not None and getattr(options, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive.")
    output = (options.output_dir or SCRIPT_DIR / "results" / Path(script_file).stem).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if options.plot_only:
        records = []
        for seed in options.seeds:
            with open(output / f"seed_{seed}" / "results" / "training.json", encoding="utf-8") as file:
                records.append(json.load(file))
        plot_results(records, output, options.smooth_alpha)
        print(f"Plots saved: {output}")
        return

    sys.path.insert(0, str(PROJECT_DIR))
    previous_dir = os.getcwd()
    try:
        source = importlib.import_module(f"reinforce_learn_ex_plot.{source_name}")
    finally:
        os.chdir(previous_dir)
    settings = {key: value for key, value in vars(source.Args).items()
                if not key.startswith("_") and not callable(value) and key != "seed"}
    for option, setting in (("steps", "time_steps"), ("num_envs", "num_envs"),
                            ("steps_per_env", "num_steps"), ("minibatch_size", "minibatch_size"),
                            ("updates_per_rollout", "gradient_updates_per_rollout"), ("device", "device")):
        value = getattr(options, option)
        if value is not None:
            settings[setting] = value
    if not 2 <= settings["minibatch_size"] <= settings["num_envs"] * settings["num_steps"]:
        parser.error("Minibatch size must be >=2 and <= num_envs * steps_per_env.")
    source_dir = Path(source.__file__).resolve().parent
    env_dir = PROJECT_DIR / "environment" / (
        "quadrotor3d_TR_constrained" if "constrained" in source_name else "quadrotor3d_TR")
    dependency_paths = [Path(source.__file__), source_dir / "parallel_ppo_training_common.py",
                        PROJECT_DIR / "environment" / "__init__.py", *sorted(env_dir.rglob("*.py"))]
    config = {
        "source_file": str(Path(source.__file__).resolve()), "settings": settings,
        "source_hashes": {str(p.relative_to(PROJECT_DIR)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in dependency_paths},
        "reward_mode": {"theoretic_mode": False, "rl_mode": True},
        "curve_metric": "rolling mean of completed training episode returns (shaped reward)",
        "seed_control": "Python, NumPy, PyTorch; vector environments reset with seed + env index",
        "timing_scope": "Original helper: training loop through environment close; excludes initialization, model saving and plots",
    }
    records = []
    for seed in options.seeds:
        run_dir = output / f"seed_{seed}"
        config_path = run_dir / "configuration.json"
        record_path = run_dir / "results" / "training.json"
        if record_path.exists() and not options.retrain:
            saved_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else None
            if saved_config != config:
                raise RuntimeError(f"Cached configuration differs for seed {seed}; use a new --output-dir or --retrain.")
            record = json.loads(record_path.read_text(encoding="utf-8"))
            print(f"Reusing {source_name}, seed={seed}")
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            set_seed(seed)
            args = SimpleNamespace(**settings, seed=seed)
            print(f"\n[{source_name}/seed={seed}]")
            record = source.train_parallel_ppo(
                args=args, agent_class=source.RLNet, script_dir=str(run_dir),
                model_filename="model.pth", result_filename="training.json",
                learning_curve_filename="training_curve.svg",
            )
            save_json(config_path, config)
        save_curves(record, run_dir / "return_curve.npz")
        records.append(record)
    timing = {}
    for key in ("data_collection_seconds", "training_seconds", "total_seconds"):
        times = np.asarray([r[key] for r in records])
        timing[key] = {"mean": float(times.mean()), "sd": float(times.std(ddof=0))}
    save_json(output / "summary.json", {
        "configuration": config, "seeds": options.seeds, "timing": timing,
        "sd_convention": "population standard deviation (ddof=0)",
        "runs": [{key: r[key] for key in (
            "seed", "requested_total_timesteps", "actual_total_timesteps",
            "data_collection_seconds", "training_seconds", "total_seconds",
            "final_stability_summary", "model_file")} for r in records],
    })
    plot_results(records, output, options.smooth_alpha)
    print(f"\nResults saved: {output}")
