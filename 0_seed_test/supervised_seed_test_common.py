"""Shared SL seed experiments: reuse source training and exclude full-MSE timing."""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from p1_linear2d_opinn_test import DEFAULT_SEEDS, METHOD_COLORS, set_seed

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def digest(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_results(records, output):
    """Align actual evaluation steps; never extend a stopped seed's curve."""
    steps = np.unique(np.concatenate([r["monitor_steps"] for r in records]))
    if not len(steps):
        print("No periodic evaluations recorded; use --eval-interval <= --steps.")
        return
    values = np.full((len(records), len(steps)), np.nan)
    for i, record in enumerate(records):
        values[i, np.searchsorted(steps, record["monitor_steps"])] = record["monitored_training_mse"]
    mean, sd = np.nanmean(values, axis=0), np.nanstd(values, axis=0, ddof=0)
    counts = np.sum(np.isfinite(values), axis=0)
    np.savez_compressed(output / "aggregate_curves.npz", steps=steps, mean=mean,
                        sd=sd, contributing_seeds=counts, per_seed_mse=values,
                        seeds=np.asarray([r["seed"] for r in records]))
    color = METHOD_COLORS["SPNet"]
    with plt.rc_context({"font.family": "Times New Roman", "font.size": 14,
                         "axes.labelsize": 16, "legend.fontsize": 14,
                         "xtick.labelsize": 14, "ytick.labelsize": 14,
                         "legend.frameon": False, "text.usetex": False}):
        for kind in ("loss_individual", "loss_mean", "contributing_seeds"):
            fig, ax = plt.subplots(figsize=(7, 3))
            if kind == "loss_individual":
                for record in records:
                    ax.plot(record["monitor_steps"], record["monitored_training_mse"],
                            color=color, alpha=0.45, linewidth=1,
                            label=f"Seed {record['seed']}")
            elif kind == "loss_mean":
                ax.plot(steps, mean, color=color, label="Mean Loss")
                ax.fill_between(steps, mean - sd, mean + sd, color=color,
                                alpha=0.2, label="±SD")
            else:
                ax.step(steps, counts, where="post", color=color)
                ax.set_ylim(0, len(records) + 0.5)
                ax.set_yticks(range(len(records) + 1))
            if kind != "contributing_seeds":
                ends = [r for r in records if r["monitor_steps"]]
                ax.scatter([r["monitor_steps"][-1] for r in ends],
                           [r["monitored_training_mse"][-1] for r in ends],
                           color=color, edgecolors="white", s=40, zorder=4,
                           label="Stop Points")
                ax.legend()
            ax.set_xlabel("Gradient Updates")
            ax.set_ylabel("Contributing Seeds" if kind == "contributing_seeds" else "Training MSE")
            ax.set_xscale("linear")
            ax.set_yscale("linear")
            ax.grid(True, linestyle="--", alpha=0.3)
            fig.tight_layout()
            for extension in ("pdf", "png", "svg"):
                fig.savefig(output / f"{kind}.{extension}", dpi=300, bbox_inches="tight")
            plt.close(fig)


def main(source_name, problem, script_file):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--device")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    options = parser.parse_args()
    if len(set(options.seeds)) != len(options.seeds):
        parser.error("Seeds must be unique.")
    if any(v is not None and v <= 0 for v in
           (options.steps, options.batch_size, options.eval_interval)):
        parser.error("Steps, batch size and evaluation interval must be positive.")
    if options.plot_only and options.retrain:
        parser.error("--plot-only and --retrain cannot be combined.")
    output = (options.output_dir or SCRIPT_DIR / "results" / Path(script_file).stem).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if options.plot_only:
        # Redrawing saved data does not require the original dataset or GPU.
        records = []
        for seed in options.seeds:
            with open(output / f"seed_{seed}" / "training.json", encoding="utf-8") as file:
                records.append(json.load(file))
        plot_results(records, output)
        print(f"Plots saved to {output}")
        return

    sys.path.insert(0, str(PROJECT_DIR))
    previous_dir = os.getcwd()
    try:
        source = importlib.import_module(f"sup_learn_ex_plot.{source_name}")
    finally:
        os.chdir(previous_dir)
    data_dir = (options.data_dir or Path(source.SCRIPT_DIR) / "data").resolve()
    suffix = "_constrained" if problem == "constrained" else ""
    x_path, u_path = data_dir / f"X_trajs{suffix}.npy", data_dir / f"U_trajs{suffix}.npy"
    for path in (x_path, u_path):
        if not path.is_file():
            raise FileNotFoundError(f"Shared dataset not found: {path}")
    device = options.device or source.Args.device
    env = source.gym.make(source.Args.env_id).unwrapped
    try:
        set_seed(options.seeds[0])
        probe = source.SPNet(env)
        config = {
            "experiment": source.EXPERIMENT_NAME,
            "source_file": str(Path(source.__file__).resolve()),
            "source_sha256": digest(source.__file__),
            "training_helper_sha256": digest(Path(source.SCRIPT_DIR) / "supervised_training_common.py"),
            "siren_sha256": digest(PROJECT_DIR / "utils_folder" / "SIREN.py") if "siren" in source_name else None,
            "dataset_state_file": str(x_path), "dataset_action_file": str(u_path),
            "dataset_state_sha256": digest(x_path), "dataset_action_sha256": digest(u_path),
            "architecture": str(probe), "parameters": sum(p.numel() for p in probe.parameters()),
            "environment": source.Args.env_id, "device": str(device),
            "learning_rate": source.Args.learning_rate, "optimizer": "Adam",
            "batch_size": options.batch_size or source.Args.batch_size,
            "maximum_steps": options.steps or source.MAX_TRAIN_STEPS,
            "evaluation_interval": options.eval_interval or source.EVAL_INTERVAL,
            "target_training_mse": source.TARGET_TRAINING_MSE,
            "required_target_checks": source.REQUIRED_TARGET_CHECKS,
            "metric": "full_training_mse", "dataset_split": "none",
            "training_time_excludes_full_dataset_mse": True,
        }
        del probe
        records = []
        for seed in options.seeds:
            run_dir = output / f"seed_{seed}"
            record_path = run_dir / "training.json"
            config_path = run_dir / "configuration.json"
            if record_path.exists() and not options.retrain:
                saved_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else None
                if saved_config != config:
                    raise RuntimeError(f"Cached configuration differs for seed {seed}. Use a new --output-dir or --retrain.")
                record = json.loads(record_path.read_text(encoding="utf-8"))
                print(f"Reusing {source.EXPERIMENT_NAME}, seed={seed}")
            else:
                run_dir.mkdir(parents=True, exist_ok=True)
                set_seed(seed)
                model = source.SPNet(env).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=source.Args.learning_rate)
                print(f"\n[{source.EXPERIMENT_NAME}/seed={seed}]")
                record = source.train_from_saved_trajectories(
                    model=model, optimizer=optimizer, device=device,
                    experiment_name=source.EXPERIMENT_NAME,
                    x_path=str(x_path), u_path=str(u_path),
                    model_path=str(run_dir / "model.pth"), result_path=str(record_path),
                    training_seed=seed, requested_batch_size=config["batch_size"],
                    evaluation_interval=config["evaluation_interval"],
                    maximum_training_steps=config["maximum_steps"],
                    target_training_mse=config["target_training_mse"],
                    required_target_checks=config["required_target_checks"],
                )
                save_json(config_path, config)
                del model, optimizer
            np.savez_compressed(run_dir / "loss_curve.npz",
                                steps=record["monitor_steps"],
                                training_mse=record["monitored_training_mse"])
            records.append(record)
        times = np.asarray([r["training_seconds"] for r in records])
        save_json(output / "summary.json", {
            "configuration": config, "seeds": options.seeds,
            "training_seconds_mean": float(times.mean()),
            "training_seconds_sd": float(times.std(ddof=0)),
            "sd_convention": "population standard deviation (ddof=0)",
            "runs": [{key: r[key] for key in (
                "seed", "steps", "training_seconds", "training_wall_seconds",
                "full_dataset_mse_evaluation_seconds", "stop_reason", "target_reached",
                "selected_model_training_mse")} for r in records],
        })
        plot_results(records, output)
        print(f"\nSaved results: {output}\nTraining time: {times.mean():.2f} ± {times.std():.2f} s (excluding full MSE).")
    finally:
        env.close()

