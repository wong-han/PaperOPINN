# Multi-Seed RL Tests for the Quadrotor Tasks

Run `p4_quadrotor3d_rl_test.py` and
`p4_quadrotor3d_constrained_rl_test.py` directly; no command-line parameters
are required. By default, each script trains seeds 0, 1, 2, 3, and 4
sequentially so that concurrent jobs do not distort the timing measurements.
Parallel environment sampling within each seed is unchanged.

## Default configuration

The test scripts import `Args`, `RLNet`, and `train_parallel_ppo` from the
corresponding training program and do not alter the original training logic.
The current configuration uses eight environments, 2,000 steps per environment
and rollout, a minibatch size of 2,048, and 256 optimization updates per
rollout. The unconstrained task requests 1,500,000 interactions and executes
1,504,000 interactions because only complete rollouts are collected. The
constrained task requests and executes 2,000,000 interactions. Training uses a
fixed interaction budget rather than return-based early stopping.

The training environments use `theoretic_mode=False` and `rl_mode=True`, so
the reported training return includes reward shaping and is not the true
control cost used in Monte Carlo evaluation. These entry points do not run the
Monte Carlo evaluation.

## Saved outputs

Results are written to `results/<entry-point-name>/` in this directory, without
overwriting the original models or experiment results.

- `seed_N/model/model.pth`: final policy and value networks.
- `seed_N/results/training.json`: return history, actual interaction count,
  three timing measurements, and end-of-training stability statistics.
- `seed_N/return_curve.npz`: interactions per rollout, mean return of episodes
  completed in that rollout, mean return over the latest 50 completed episodes,
  and cumulative episode count.
- `seed_N/configuration.json`: hyperparameters and fingerprints of the source
  program, shared training code, and environment implementation.
- `summary.json`: per-seed results and the mean and standard deviation of each
  timing measurement.
- `aggregate_curves.npz`: cross-seed curves, means, standard deviations, and
  the number of contributing seeds.
- `return_individual`, `return_mean`, and `contributing_seeds`: figures saved in
  PDF, PNG, and SVG formats.

The mean and standard-deviation curves use the moving average over the latest
50 completed episodes. The shaded standard deviation is the population
standard deviation across seeds (`ddof=0`), not a confidence interval. For the
`return_mean` figure, a causal EMA is applied to each seed before computing the
cross-seed mean and standard deviation. The default smoothing factor is
`alpha=0.25`, and the displayed stopping points use the smoothed final values.
Raw returns, raw cross-seed statistics, timing data, and stability statistics
remain unchanged; `return_individual` shows curves without this additional
smoothing.

`plot_settings.json` records the plotting configuration, while
`aggregate_curves.npz` contains both raw and smoothed curves. Use
`--smooth-alpha 0.15` for stronger smoothing or `--smooth-alpha 1` to disable
the additional smoothing. Records with no completed episode are stored as
missing values; they are neither replaced by zero nor interpolated. The
`Stop Points` annotation marks the end of the fixed interaction budget and does
not imply that a convergence condition was satisfied.

The three timing measurements follow the convention of the shared training
code. Data-collection time includes policy sampling and environment
interaction. Training time is total time minus data-collection time and thus
includes network updates and all other intermediate calculations. Total time
runs from the start of the training loop until the environments are closed; it
excludes environment and network initialization, model serialization, and
plotting.

## Reusing completed results

When a script is run again, completed seeds with a matching configuration are
reused instead of retrained. To regenerate figures only, run:

```powershell
python p4_quadrotor3d_rl_test.py --plot-only
python p4_quadrotor3d_constrained_rl_test.py --plot-only
```

Plot-only mode does not import the original environment or create parallel
sampling processes. If the configuration or source code changes, the script
refuses to mix incompatible results. Use `--output-dir <new-directory>` to
start a separate experiment or `--retrain` to overwrite the saved result for a
selected seed.
