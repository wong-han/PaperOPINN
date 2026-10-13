"""Linear OPINN seed experiments; reruns reuse compatible saved loss histories.

Run this file directly for seeds 0--4. Use --plot-only to redraw saved results.
The training network and loss functions are imported from p1_hnn_linear2d.py.
"""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SOURCE_PATH = PROJECT_DIR / "pihnn_experiments_plot" / "p1_hnn_linear2d.py"
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
METHOD_COLORS = {
    "HNN": "#9558B2",
    "OPINN": "#9558B2",
    "RL": "#4B8BBE",
    "SPNet": "#389826",
    "MPC": "#E64B35",
    "LQR": "#F39B7F",
}


def import_source():
    # The original module changes cwd on import; restore the caller's directory.
    sys.path.insert(0, str(PROJECT_DIR))
    previous_dir = os.getcwd()
    try:
        return importlib.import_module("pihnn_experiments_plot.p1_hnn_linear2d")
    finally:
        os.chdir(previous_dir)


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configuration(source, options):
    return {
        "experiment": "p1_linear2d_opinn",
        "source_file": str(SOURCE_PATH),
        "source_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
        "environment": source.Args.env_id,
        "steps": options.steps or source.Args.time_steps,
        "batch_size": options.batch_size or source.Args.batch_size,
        "learning_rate": source.Args.learning_rate,
        "device": options.device or source.Args.device,
        "sample_low": -4.0,
        "sample_high": 4.0,
        "loss": "get_all_loss + lambda_zero_loss (pre-update minibatch loss)",
    }


def train_seed(source, config, seed, output_dir):
    set_seed(seed)
    env = source.gym.make(config["environment"])
    try:
        env.reset(seed=seed)
        model = source.HNN(env).to(config["device"])
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        losses = np.empty(config["steps"], dtype=np.float64)
        elapsed = np.empty_like(losses)
        model.train()
        synchronize(config["device"])
        started = time.perf_counter()
        for index in range(config["steps"]):
            states = np.random.uniform(
                config["sample_low"], config["sample_high"],
                (config["batch_size"], env.observation_space.shape[0]),
            )
            inputs = torch.from_numpy(states).float().to(config["device"])
            inputs.requires_grad_(True)
            loss = source.get_all_loss(model, inputs)
            if config.get("include_equilibrium_loss", True):
                loss = loss + source.lambda_zero_loss(model)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # Record the actual objective every step, without smoothing stored data.
            losses[index] = loss.item()
            elapsed[index] = time.perf_counter() - started
            if not np.isfinite(losses[index]):
                raise FloatingPointError(f"Seed {seed}: non-finite loss at step {index + 1}")
            if index == 0 or (index + 1) % 100 == 0 or index + 1 == config["steps"]:
                system_name = config["environment"].split(":")[-1].removesuffix("-v0")
                print(f"[{system_name}/OPINN/seed={seed}] step={index + 1}/{config['steps']}, "
                      f"loss={losses[index]:.6g}", flush=True)
        synchronize(config["device"])
        duration = time.perf_counter() - started
        steps = np.arange(1, config["steps"] + 1)
        record = dict(config, seed=seed, completed=True, training_seconds=duration)
        # Each seed has independent artifacts; the source model files are untouched.
        torch.save(model.state_dict(), output_dir / f"seed_{seed}_model.pth")
        np.savez_compressed(
            output_dir / f"seed_{seed}_loss.npz", steps=steps, loss=losses,
            elapsed_seconds=elapsed, metadata=np.asarray(json.dumps(record)),
        )
        (output_dir / f"seed_{seed}_summary.json").write_text(
            json.dumps(dict(record, final_loss=float(losses[-1]),
                            minimum_batch_loss=float(losses.min())), indent=2),
            encoding="utf-8",
        )
        return steps, losses, record
    finally:
        env.close()


def load_seed(path, config):
    with np.load(path, allow_pickle=False) as data:
        record = json.loads(data["metadata"].item())
        if not record.get("completed") or any(record.get(k) != v for k, v in config.items()):
            raise ValueError(
                f"Saved run does not match the current source/configuration: {path}\n"
                "Use --output-dir for a new experiment, or --retrain to replace this run."
            )
        steps, losses = data["steps"].copy(), data["loss"].copy()
        if len(losses) != config["steps"] or not np.array_equal(
            steps, np.arange(1, config["steps"] + 1)
        ) or not np.isfinite(losses).all():
            raise ValueError(f"Incomplete or invalid loss history: {path}")
    return steps, losses, record


def plot_results(runs, seeds, output_dir):
    steps = runs[0][0]
    curves = np.stack([run[1] for run in runs])
    mean = curves.mean(axis=0)
    # Population SD describes the selected seeds; this band is not a confidence interval.
    std = curves.std(axis=0, ddof=0)
    np.savez_compressed(output_dir / "loss_aggregate.npz", steps=steps,
                        seeds=np.asarray(seeds), losses=curves, mean=mean, std=std)
    color = METHOD_COLORS["OPINN"]
    with plt.rc_context({"text.usetex": False, "font.size": 11,
                         "axes.labelsize": 12, "legend.fontsize": 10,
                         "xtick.labelsize": 10, "ytick.labelsize": 10}):
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for seed, curve in zip(seeds, curves):
            axes[0].plot(steps, curve, color=color,
                         alpha=0.55, linewidth=0.8, label=f"Seed {seed}")
        axes[1].plot(steps, mean, color=color,
                     linewidth=1.5, label="OPINN mean")
        axes[1].fill_between(steps, mean - std,
                             mean + std, color=color,
                             alpha=0.2, label="Mean ± SD")
        for axis in axes:
            axis.set_yscale("linear")
            axis.set_xlabel("Gradient updates")
            axis.set_ylabel("Training total loss")
            axis.grid(True, which="major", linestyle="--", alpha=0.3)
            axis.legend()
        axes[0].set_title("Individual seeds")
        axes[1].set_title(f"OPINN across {len(seeds)} seeds")
        figure.tight_layout()
        for extension in ("png", "pdf", "svg"):
            figure.savefig(output_dir / f"loss_curves.{extension}", dpi=300,
                           bbox_inches="tight")
        plt.close(figure)
    print(f"Loss histories and figures saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--plot-only", action="store_true", help="Load results without training")
    parser.add_argument("--retrain", action="store_true", help="Replace results for requested seeds")
    parser.add_argument("--steps", type=int, help="Default: source Args.time_steps")
    parser.add_argument("--batch-size", type=int, help="Default: source Args.batch_size")
    parser.add_argument("--device", help="Default: source Args.device")
    parser.add_argument("--output-dir", type=Path,
                        default=SCRIPT_DIR / "results" / "p1_linear2d_opinn")
    options = parser.parse_args()
    if options.plot_only and options.retrain:
        parser.error("--plot-only and --retrain cannot be used together")
    if any(seed < 0 or seed >= 2**32 for seed in options.seeds):
        parser.error("Seeds must be in [0, 2**32)")
    if len(set(options.seeds)) != len(options.seeds):
        parser.error("Seeds must be distinct")
    if (options.steps is not None and options.steps <= 0) or (
        options.batch_size is not None and options.batch_size <= 0
    ):
        parser.error("Steps and batch size must be positive")
    source = import_source()
    config = configuration(source, options)
    output_dir = options.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in options.seeds:
        path = output_dir / f"seed_{seed}_loss.npz"
        if path.exists() and not options.retrain:
            run = load_seed(path, config)
            print(f"Loaded saved seed {seed}; training skipped.")
        elif options.plot_only:
            raise FileNotFoundError(f"No saved history for seed {seed}: {path}")
        else:
            run = train_seed(source, config, seed, output_dir)
        runs.append(run)
    plot_results(runs, options.seeds, output_dir)


if __name__ == "__main__":
    main()
