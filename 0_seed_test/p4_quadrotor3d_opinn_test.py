"""Seed experiments for quadratic quadrotor OPINN, matching the source early stop.

Run directly for seeds 0--4. Saved matching runs are reused; --plot-only redraws.
Loss data include the raw objective, EMA, and early-stopping checks.
"""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from p1_linear2d_opinn_test import DEFAULT_SEEDS, METHOD_COLORS, set_seed, synchronize


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SOURCE_PATH = PROJECT_DIR / "pihnn_experiments_plot" / "p4_hnn_quadrotor3d_TR.py"


def import_source():
    sys.path.insert(0, str(PROJECT_DIR))
    previous_dir = os.getcwd()
    try:
        source = importlib.import_module("pihnn_experiments_plot.p4_hnn_quadrotor3d_TR")
        source._env_tmp.close()
        return source
    finally:
        os.chdir(previous_dir)


def configuration(source, options):
    return {
        "experiment": "p4_quadrotor3d_opinn",
        "source_file": str(SOURCE_PATH),
        "source_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
        "environment": source.Args.env_id,
        "maximum_steps": options.steps or source.Args.time_steps,
        "batch_size": options.batch_size or source.Args.batch_size,
        "device": options.device or source.Args.device,
        "learning_rate": source.Args.learning_rate,
        "weight_decay": 0.01,
        "train_range": source.Args.train_range,
        "forward_horizon": source.Args.T,
        "forward_steps": source.Args.forward_steps,
        "ema_alpha": 0.95,
        "improvement_threshold": 1.0,
        "check_interval": 100,
        "patience": 500,
        "min_steps_zero_based": 1000,
        "loss": "get_all_loss + lambda_zero_loss + forward running cost + terminal cost",
    }


def initialize_globals(source, env, device):
    # These globals are normally created by the source's __main__ block.
    source.m = env.metadata["m"]
    source.g = env.metadata["g"]
    source.P_ref = env.P
    for name, values in (
        ("u0_tensor", env.u0), ("u_min_tensor", env.u_min),
        ("u_max_tensor", env.u_max), ("x_min_tensor", env.state_low),
        ("x_max_tensor", env.state_high),
    ):
        setattr(source, name, torch.tensor(values, dtype=torch.float32, device=device))


def total_loss(source, model, inputs, config, q_matrix, r_matrix, p_matrix):
    loss = source.get_all_loss(model, inputs) + source.lambda_zero_loss(model)
    state = inputs.detach().clone()
    dt = config["forward_horizon"] / config["forward_steps"]
    # Preserve the source's forward rollout (including its state clipping).
    for _ in range(config["forward_steps"]):
        control_matrix = source.get_pfpu(state)
        costate = model.get_lambda(state).unsqueeze(2)
        r_inverse = torch.tensor(np.linalg.inv(model.env.metadata["R"]),
                                 dtype=state.dtype, device=state.device)
        delta_action = -0.5 * torch.matmul(control_matrix.transpose(1, 2), costate)
        delta_action = torch.matmul(r_inverse, delta_action.squeeze(-1).T).T
        action = delta_action + source.u0_tensor
        state = state + source.dynamics(state, action) * dt
        state = torch.clamp(state, source.x_min_tensor, source.x_max_tensor)
        state_cost = torch.einsum("bi,ij,bj->b", state, q_matrix, state)
        action_cost = torch.einsum("bi,ij,bj->b", action - source.u0_tensor,
                                   r_matrix, action - source.u0_tensor)
        running_cost = state_cost + action_cost
        if config.get("include_constraint_penalty", False):
            running_cost = running_cost + source.get_penalty(state)
        loss = loss + running_cost.mean() * dt
    return loss + torch.einsum("bi,ij,bj->b", state, p_matrix, state).mean()


def train_seed(source, config, seed, output_dir):
    set_seed(seed)
    env = source.gym.make(config["environment"]).unwrapped
    try:
        env.reset(seed=seed)
        initialize_globals(source, env, config["device"])
        model = source.HNN(env).to(config["device"])
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                     weight_decay=config["weight_decay"])
        q_matrix = torch.tensor(env.metadata["Q"], dtype=torch.float32, device=config["device"])
        r_matrix = torch.tensor(env.metadata["R"], dtype=torch.float32, device=config["device"])
        p_matrix = torch.tensor(env.P, dtype=torch.float32, device=config["device"])
        losses, ema_history, elapsed = [], [], []
        check_steps, references, counters, check_ema = [], [], [], []
        best_reference, patience_counter, ema = float("inf"), 0, None
        stop_reason = "maximum_steps"
        synchronize(config["device"])
        started = time.perf_counter()
        for index in range(config["maximum_steps"]):
            states = np.random.uniform(-config["train_range"], config["train_range"],
                                       (config["batch_size"], env.observation_space.shape[0]))
            inputs = torch.from_numpy(states).float().to(config["device"])
            inputs.requires_grad_(True)
            loss = total_loss(source, model, inputs, config, q_matrix, r_matrix, p_matrix)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            current = loss.item()
            if not np.isfinite(current):
                raise FloatingPointError(f"Seed {seed}: non-finite loss at step {index + 1}")
            ema = current if ema is None else config["ema_alpha"] * ema + (1 - config["ema_alpha"]) * current
            losses.append(current)
            ema_history.append(ema)
            elapsed.append(time.perf_counter() - started)
            # Source checks zero-based step % 100 == 0, i.e. updates 1, 101, ...
            if index % config["check_interval"] == 0:
                if ema < best_reference - config["improvement_threshold"]:
                    best_reference, patience_counter = ema, 0
                else:
                    patience_counter += config["check_interval"]
                check_steps.append(index + 1)
                references.append(best_reference)
                counters.append(patience_counter)
                check_ema.append(ema)
                print(f"[{config['experiment']}/seed={seed}] step={index + 1}, "
                      f"loss={current:.6g}, EMA={ema:.6g}, "
                      f"patience={patience_counter}/{config['patience']}", flush=True)
            if index > config["min_steps_zero_based"] and patience_counter >= config["patience"]:
                stop_reason = "ema_plateau"
                break
        synchronize(config["device"])
        duration = time.perf_counter() - started
        record = dict(config, seed=seed, completed=True, actual_steps=len(losses),
                      stop_reason=stop_reason, training_seconds=duration,
                      final_loss=losses[-1], final_ema=ema_history[-1],
                      best_ema_reference=best_reference, final_patience=patience_counter)
        torch.save(model.state_dict(), output_dir / f"seed_{seed}_model.pth")
        np.savez_compressed(output_dir / f"seed_{seed}_loss.npz",
                            steps=np.arange(1, len(losses) + 1), loss=losses,
                            ema=ema_history, elapsed_seconds=elapsed,
                            check_steps=check_steps, check_ema=check_ema,
                            best_ema_reference=references, patience=counters,
                            metadata=np.asarray(json.dumps(record)))
        (output_dir / f"seed_{seed}_summary.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8")
        print(f"Seed {seed}: {stop_reason}, {len(losses)} updates, {duration:.2f} s")
    finally:
        env.close()


def load_seed(path, config):
    with np.load(path, allow_pickle=False) as saved:
        record = json.loads(saved["metadata"].item())
        if not record.get("completed") or any(record.get(k) != v for k, v in config.items()):
            raise ValueError(f"Saved configuration differs: {path}. "
                             "Use --output-dir for a new experiment or --retrain to replace it.")
        arrays = {key: saved[key].copy() for key in ("steps", "loss", "ema")}
        n = record["actual_steps"]
        if not 0 < n <= config["maximum_steps"] or any(
            len(value) != n or not np.isfinite(value).all() for value in arrays.values()
        ) or not np.array_equal(arrays["steps"], np.arange(1, n + 1)):
            raise ValueError(f"Invalid saved history: {path}")
    return arrays, record


def plot_results(runs, seeds, output_dir):
    # NaN padding retains unequal stopping lengths without inventing later values.
    lengths = np.asarray([len(arrays["loss"]) for arrays, _ in runs])
    max_steps = int(lengths.max())
    raw = np.full((len(runs), max_steps), np.nan)
    ema = raw.copy()
    for row, (arrays, _) in enumerate(runs):
        raw[row, :lengths[row]] = arrays["loss"]
        ema[row, :lengths[row]] = arrays["ema"]
    steps = np.arange(1, max_steps + 1)
    contributors = np.sum(np.isfinite(ema), axis=0)
    mean, std = np.nanmean(ema, axis=0), np.nanstd(ema, axis=0)
    np.savez_compressed(output_dir / "loss_aggregate.npz", steps=steps, seeds=seeds,
                        actual_steps=lengths, loss=raw, ema=ema,
                        raw_mean=np.nanmean(raw, axis=0), raw_std=np.nanstd(raw, axis=0),
                        ema_mean=mean, ema_std=std, contributing_seeds=contributors)
    color = METHOD_COLORS["OPINN"]
    with plt.rc_context({"text.usetex": False, "font.size": 11, "axes.labelsize": 12,
                         "legend.fontsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14}):
        figures_and_axes = [plt.subplots(figsize=(7.0, 3.0)) for _ in range(3)]
        axes = [axis for _, axis in figures_and_axes]
        for seed, (arrays, record) in zip(seeds, runs):
            axes[0].plot(arrays["steps"], arrays["loss"], color=color, alpha=0.45,
                         linewidth=0.7, label=f"Seed {seed}")
            axes[1].plot(arrays["steps"], arrays["ema"], color=color, alpha=0.25, linewidth=0.7)
        axes[1].plot(steps, mean, color=color, linewidth=1.6, label="Mean Loss")
        axes[1].fill_between(steps, mean - std, mean + std, color=color, alpha=0.2,
                             label="±SD")
        axes[1].scatter(
            [arrays["steps"][-1] for arrays, _ in runs],
            [arrays["ema"][-1] for arrays, _ in runs],
            color=color, s=40, edgecolors="white", linewidths=0.6,
            zorder=5, label="Stop Points",
        )
        axes[2].step(steps, contributors, where="post", color=color)
        axes[2].set_ylim(0, len(seeds) + 0.5)
        axes[2].set_yticks(np.arange(len(seeds) + 1))
        axes[2].set_ylabel("Seeds Still Contributing")
        for axis in axes:
            axis.set_yscale("linear")
            axis.set_xlabel("Gradient Updates")
            axis.grid(True, linestyle="--", alpha=0.3)
        axes[0].set_ylabel("Training Total Loss")
        axes[1].set_ylabel("EMA Total Loss")
        axes[0].legend()
        axes[1].legend()
        names = ("loss_raw", "loss_ema", "contributing_seeds")
        for name, (figure, _) in zip(names, figures_and_axes):
            figure.tight_layout()
            for extension in ("png", "pdf", "svg"):
                figure.savefig(output_dir / f"{name}.{extension}", dpi=300,
                               bbox_inches="tight")
            plt.close(figure)
    (output_dir / "seed_summary.json").write_text(
        json.dumps({"seeds": seeds, "runs": [record for _, record in runs],
                    "aggregation": "Mean and population SD over available seeds at each step"},
                   indent=2), encoding="utf-8")
    print(f"Saved histories, stopping records and figures: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=SCRIPT_DIR / "results" / "p4_quadrotor3d_opinn")
    options = parser.parse_args()
    if options.plot_only and options.retrain:
        parser.error("--plot-only and --retrain cannot be used together")
    if len(set(options.seeds)) != len(options.seeds) or any(s < 0 or s >= 2**32 for s in options.seeds):
        parser.error("Seeds must be distinct integers in [0, 2**32)")
    if any(value is not None and value <= 0 for value in (options.steps, options.batch_size)):
        parser.error("Steps and batch size must be positive")
    source = import_source()
    config = configuration(source, options)
    output_dir = options.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Check existing caches before launching expensive runs.
    for seed in options.seeds:
        path = output_dir / f"seed_{seed}_loss.npz"
        if path.exists() and not options.retrain:
            load_seed(path, config)
        elif options.plot_only:
            raise FileNotFoundError(f"Missing saved history: {path}")
    runs = []
    for seed in options.seeds:
        path = output_dir / f"seed_{seed}_loss.npz"
        if path.exists() and not options.retrain:
            print(f"Loaded saved seed {seed}; training skipped.")
        else:
            train_seed(source, config, seed, output_dir)
        runs.append(load_seed(path, config))
    plot_results(runs, options.seeds, output_dir)


if __name__ == "__main__":
    main()
