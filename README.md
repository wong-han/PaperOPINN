# PaperOPINN

This repository contains the experimental code for the
Optimality-Principles-Informed Neural Network (OPINN). It includes the OPINN,
supervised-learning, reinforcement-learning, and conventional PINN baselines
used in the demonstration, quadrotor-simulation, and physical-flight studies.

## Repository structure

| Directory | Contents |
| --- | --- |
| `environment/` | Dynamical-system and quadrotor environments shared by the experiments. |
| `pihnn_experiments_plot/` | Unconstrained OPINN experiments for the linear, nonlinear, attitude, and quadrotor systems. |
| `pihnn_constrained_ex_plot/` | Constrained OPINN experiments. |
| `normal_pinn_ex_plot/` | Conventional PINN and X-TFC comparison experiments. |
| `sup_learn_ex_plot/` | Demonstration generation and MLP/SIREN supervised-learning baselines. |
| `reinforce_learn_ex_plot/` | PPO reinforcement-learning baselines and parallel rollout utilities. |
| `0_seed_test/` | Multi-seed training entry points and aggregation utilities. |
| `1_compare_all/` | Paired Monte Carlo comparisons of LQR, MLP, SIREN, PPO, and OPINN. |
| `pihnn_physical_ex_plot/` | Physical-flight data-processing and plotting scripts. |
| `utils_folder/` | Shared neural-network components. |

## Running the experiments

The numbered filenames follow the order of the experiments in the paper. Run a
script from its own directory so that its relative `model/`, `data/`, `results/`,
and `image/` paths resolve consistently. The quadrotor supervised-learning
workflow is split into dataset generation and method-specific training scripts;
the generated dataset is shared by MLP and SIREN. The PPO scripts use parallel
environment rollout, while the OPINN scripts sample collocation states
and require no offline trajectory dataset.

For the multi-seed and five-method evaluation workflows, see:

- [`0_seed_test/RL_SEED_TEST_README.md`](0_seed_test/RL_SEED_TEST_README.md)
- [`1_compare_all/FIVE_METHOD_COMPARISON_README.md`](1_compare_all/FIVE_METHOD_COMPARISON_README.md)

Saved models, datasets, figures, and numerical result files are retained to
support reproducibility. Configuration files may contain the absolute paths of
the machines on which the original experiments were executed; these fields are
provenance records and are not required when running the scripts on a new
machine.
