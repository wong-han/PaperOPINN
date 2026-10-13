"""Five-seed experiments for constrained quadrotor OPINN.

Imports p4_chnn_quadrotor3d_TR.py and preserves its training objective and EMA
stopping rule. Run directly; reruns reuse matching completed histories.
Use --plot-only to redraw the three separate linear-scale purple figures.
"""

import argparse
import hashlib
import importlib
import os
from pathlib import Path
import sys

from p1_linear2d_opinn_test import DEFAULT_SEEDS
from p4_quadrotor3d_opinn_test import (
    configuration as quadrotor_configuration,
    load_seed, plot_results, train_seed,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SOURCE_PATH = PROJECT_DIR / "pihnn_constrained_ex_plot" / "p4_chnn_quadrotor3d_TR.py"


def import_source():
    sys.path.insert(0, str(PROJECT_DIR))
    previous_dir = os.getcwd()
    try:
        source = importlib.import_module("pihnn_constrained_ex_plot.p4_chnn_quadrotor3d_TR")
        source._env_tmp.close()
        return source
    finally:
        os.chdir(previous_dir)


def configuration(source, options):
    config = quadrotor_configuration(source, options)
    config.update({
        "experiment": "p4_quadrotor3d_constrained_opinn",
        "source_file": str(SOURCE_PATH),
        "source_sha256": hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
        "weight_decay": 0.0001,
        "include_constraint_penalty": True,
        "loss": "get_all_loss + lambda_zero_loss + forward running cost "
                "(including constraint penalty) + terminal cost",
    })
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=SCRIPT_DIR / "results" / "p4_quadrotor3d_constrained_opinn")
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
