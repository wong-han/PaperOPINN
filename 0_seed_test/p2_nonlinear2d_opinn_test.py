"""Nonlinear OPINN seed experiments using p2_hnn_nonlinear2d.py.

Run directly for seeds 0--4; completed matching runs are loaded automatically.
Use --plot-only to redraw saved loss curves without running training.
"""

import argparse
import hashlib
import importlib
import os
from pathlib import Path
import sys

from p1_linear2d_opinn_test import (
    DEFAULT_SEEDS, METHOD_COLORS, load_seed, plot_results, train_seed,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SOURCE_PATH = PROJECT_DIR / "pihnn_experiments_plot" / "p2_hnn_nonlinear2d.py"


def import_source():
    sys.path.insert(0, str(PROJECT_DIR))
    previous_dir = os.getcwd()
    try:
        return importlib.import_module("pihnn_experiments_plot.p2_hnn_nonlinear2d")
    finally:
        os.chdir(previous_dir)


def configuration(source, options):
    return {
        "experiment": "p2_nonlinear2d_opinn",
        "source_file": str(SOURCE_PATH),
        "source_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
        "environment": source.Args.env_id,
        "steps": options.steps or source.Args.time_steps,
        "batch_size": options.batch_size or source.Args.batch_size,
        "learning_rate": source.Args.learning_rate,
        "device": options.device or source.Args.device,
        "sample_low": -4.0,
        "sample_high": 4.0,
        # Match train_hnn(): the equilibrium-Jacobian regularizer is disabled.
        "include_equilibrium_loss": False,
        "loss": "get_all_loss (pre-update minibatch loss)",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--plot-only", action="store_true", help="Load results without training")
    parser.add_argument("--retrain", action="store_true", help="Replace results for requested seeds")
    parser.add_argument("--steps", type=int, help="Default: source Args.time_steps")
    parser.add_argument("--batch-size", type=int, help="Default: source Args.batch_size")
    parser.add_argument("--device", help="Default: source Args.device")
    parser.add_argument("--output-dir", type=Path,
                        default=SCRIPT_DIR / "results" / "p2_nonlinear2d_opinn")
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
