"""Utilities for timing and recording single-environment PPO training."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def synchronize_device(device):
    """Make CUDA timing comparable with CPU wall-clock timing."""
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def save_learning_curve(path, steps, rollout_means, rolling_means, window_size):
    finite_rollout = [
        (step, value)
        for step, value in zip(steps, rollout_means)
        if value is not None
    ]
    finite_rolling = [
        (step, value)
        for step, value in zip(steps, rolling_means)
        if value is not None
    ]

    # Do not require a local LaTeX installation just to save the training curve.
    with plt.rc_context({"text.usetex": False}):
        figure, axis = plt.subplots(figsize=(8, 5))
        if finite_rollout:
            x, y = zip(*finite_rollout)
            axis.plot(
                x,
                y,
                color="tab:blue",
                alpha=0.25,
                linewidth=1.0,
                label="Mean return in rollout",
            )
        if finite_rolling:
            x, y = zip(*finite_rolling)
            axis.plot(
                x,
                y,
                color="tab:blue",
                linewidth=2.2,
                label=f"Rolling mean ({window_size} episodes)",
            )
        axis.set_xlabel("Environment interactions")
        axis.set_ylabel("Mean episodic return")
        axis.grid(True, linestyle="--", alpha=0.4)
        if finite_rollout or finite_rolling:
            axis.legend()
        figure.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)


def save_training_record(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)


def stability_summary(steps, rolling_means, maximum_points=10):
    finite_indices = [
        index for index, value in enumerate(rolling_means) if value is not None
    ]
    selected_indices = finite_indices[-maximum_points:]
    if len(selected_indices) < 2:
        return {
            "points_used": len(selected_indices),
            "rolling_return_mean": None,
            "rolling_return_std": None,
            "return_slope_per_100k_interactions": None,
        }

    selected_steps = np.asarray([steps[index] for index in selected_indices], dtype=float)
    selected_returns = np.asarray(
        [rolling_means[index] for index in selected_indices], dtype=float
    )
    return {
        "points_used": len(selected_indices),
        "rolling_return_mean": float(np.mean(selected_returns)),
        "rolling_return_std": float(np.std(selected_returns)),
        "return_slope_per_100k_interactions": float(
            np.polyfit(selected_steps / 100000.0, selected_returns, 1)[0]
        ),
    }
