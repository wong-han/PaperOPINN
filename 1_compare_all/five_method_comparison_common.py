"""Paired quadrotor evaluation, preserving the legacy fixed-horizon metrics."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.linalg import solve_continuous_are
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ORDER = ["LQR", "SIREN", "MLP", "RL", "OPINN"]
COLORS = {"LQR": "#F39B7F", "SIREN": "#E64B35", "MLP": "#389826",
          "RL": "#4B8BBE", "OPINN": "#9558B2"}


def import_source(name):
    sys.path.insert(0, str(PROJECT))
    previous = os.getcwd()
    try:
        return importlib.import_module(name)
    finally:
        os.chdir(previous)


def average_sd(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {"count": len(values), "mean": float(values.mean()) if len(values) else None,
            "sd": float(values.std()) if len(values) else None}


def draw(data_dir, image_dir, constrained):
    palette = COLORS
    with plt.rc_context({"font.family": "Times New Roman", "text.usetex": False,
                         "axes.unicode_minus": False, "font.size": 24,
                         "axes.labelsize": 24, "xtick.labelsize": 24, "ytick.labelsize": 24,
                         "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.top": True, "ytick.right": True}):
        plots = [("return_boxplot_data.csv", "Return", "$J$", "return_boxplot")]
        if constrained:
            plots.append(("logy_boxplot_data.csv", "LogAbsYMin", "$\\log_{10}|y_{\\mathrm{min}}|$", "min_y_boxplot"))
        for filename, column, label, stem in plots:
            frame = pd.read_csv(data_dir / filename)
            frame = frame[np.isfinite(frame[column])]
            fig, ax = plt.subplots(figsize=(12, 7))
            if len(frame):
                common = dict(data=frame, x="Controller", y=column, order=ORDER,
                              hue="Controller", hue_order=ORDER, palette=palette,
                              dodge=False, legend=False, ax=ax)
                sns.violinplot(**common, inner=None, cut=0, linewidth=1.2, alpha=0.22)
                sns.boxplot(**common, width=0.25, showcaps=True, showfliers=False,
                            medianprops=dict(color="#6D2D2B", linewidth=2),
                            boxprops=dict(alpha=0.7, edgecolor="k", linewidth=2),
                            whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2))
                np.random.seed(2222)  # Reproducible plotting jitter only.
                sns.stripplot(**common, size=7, alpha=0.4, jitter=0.22,
                              linewidth=0.2, edgecolor="#444")
            else:
                ax.set_xticks(range(len(ORDER)), ORDER)
                ax.text(0.5, 0.5, "No successful trajectories", transform=ax.transAxes,
                        ha="center", va="center", fontsize=18)
            ax.set_xlabel("")
            ax.set_ylabel(label, usetex=True)
            ax.grid(True, linestyle="--", alpha=0.45)
            sns.despine(top=False, right=False, left=False, bottom=False, ax=ax)
            fig.tight_layout()
            for extension in ("pdf", "svg", "png"):
                fig.savefig(image_dir / f"{stem}.{extension}", dpi=300,
                            bbox_inches="tight", pad_inches=0.5)
            plt.close(fig)


def main(constrained=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2222)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--horizon-steps", type=int, help="Short-test override; default uses environment T/dt")
    for method in ("mlp", "siren", "rl", "opinn"):
        parser.add_argument(f"--{method}-model", type=Path)
    options = parser.parse_args()
    if options.episodes <= 0 or options.seed < 0 or (options.horizon_steps is not None and options.horizon_steps <= 0):
        parser.error("Episodes/horizon must be positive and seed nonnegative.")
    experiment = "nonquadratic_five_methods" if constrained else "quadratic_five_methods"
    root = options.output_dir.resolve() if options.output_dir else HERE / "results" / experiment
    data_dir, image_dir = root / "data", root / "image"
    if options.plot_only:
        image_dir.mkdir(parents=True, exist_ok=True)
        draw(data_dir, image_dir, constrained)
        print(f"Redrawn saved results: {image_dir}")
        return
    suffix = "_constrained" if constrained else ""
    hnn_package = "pihnn_constrained_ex_plot" if constrained else "pihnn_experiments_plot"
    hnn_name = "p4_chnn_quadrotor3d_TR" if constrained else "p4_hnn_quadrotor3d_TR"
    sources = {
        "MLP": import_source(f"sup_learn_ex_plot.p4_mlp_quadrotor3d_TR{suffix}_parallel"),
        "SIREN": import_source(f"sup_learn_ex_plot.p4_siren_quadrotor3d_TR{suffix}_parallel"),
        "RL": import_source(f"reinforce_learn_ex_plot.p4_rl_quadrotor3d_TR{suffix}"),
        "OPINN": import_source(f"{hnn_package}.{hnn_name}"),
    }
    paths = {
        "MLP": PROJECT / "sup_learn_ex_plot" / "model" / f"mlp_quadrotor3d{suffix}.pth",
        "SIREN": PROJECT / "sup_learn_ex_plot" / "model" / f"siren_quadrotor3d{suffix}.pth",
        "RL": PROJECT / "reinforce_learn_ex_plot" / "model" / f"rl_quadrotor3d{suffix}.pth",
        "OPINN": PROJECT / hnn_package / "model" / sources["OPINN"].Args.model_name,
    }
    for method in paths:
        override = getattr(options, f"{method.lower()}_model")
        if override is not None:
            paths[method] = override.resolve()
        if not paths[method].is_file():
            raise FileNotFoundError(f"{method} model missing: {paths[method]}; no random-model fallback.")
    env_id = "environment:quadrotor3d_TR_constrained" if constrained else "environment:quadrotor3d_TR-v0"
    envs = {}
    try:
        for method in ORDER:
            envs[method] = gym.make(env_id, theoretic_mode=True, rl_mode=False).unwrapped
        env = envs["LQR"]
        horizon = options.horizon_steps or int(env.T / env.dt)
        models, provenance = {}, {}
        for method, source in sources.items():
            model = (source.RLNet(9, 4) if method == "RL" else
                     source.HNN(envs[method]) if method == "OPINN" else source.SPNet(envs[method]))
            model = model.to(options.device)
            model.load_state_dict(torch.load(paths[method], map_location=options.device, weights_only=True), strict=True)
            models[method] = model.eval()
            provenance[method] = {"file": str(paths[method]),
                                  "sha256": hashlib.sha256(paths[method].read_bytes()).hexdigest(),
                                  "modified": paths[method].stat().st_mtime,
                                  "architecture_source": str(Path(source.__file__).resolve())}
            print(f"Loaded {method}: {paths[method]}")
        sources["OPINN"].m, sources["OPINN"].g = env.metadata["m"], env.metadata["g"]
        P = solve_continuous_are(env.A, env.B, env.Q, env.R)
        K = np.linalg.solve(env.R, env.B.T @ P)
        R_inv = torch.linalg.inv(torch.as_tensor(env.R, dtype=torch.float32, device=options.device))

        def action(method, state):
            if method == "LQR":
                return -K @ state + env.u0
            x = torch.as_tensor(state, dtype=torch.float32, device=options.device).unsqueeze(0)
            with torch.no_grad():
                if method == "OPINN":
                    costate = models[method].get_lambda(x)
                    jacobian = sources[method].get_pfpu(x)
                    term = -0.5 * torch.bmm(jacobian.transpose(1, 2), costate.unsqueeze(2)).squeeze(-1)
                    return (term @ R_inv.T).squeeze(0).cpu().numpy() + env.u0
                if method == "RL":
                    value = models[method].get_action(x).squeeze(0).cpu().numpy()
                    low, high = envs[method].action_space.low, envs[method].action_space.high
                    return value * (high - low) / 2 + (high + low) / 2
                return models[method](x).squeeze(0).cpu().numpy()

        seeds = np.random.RandomState(options.seed).randint(0, 1000000, size=options.episodes)
        all_data, rows, cost_rows, constraint_rows, summary = {}, [], [], [], {}
        for method in ORDER:
            states = np.full((options.episodes, horizon + 1, 9), np.nan)
            controls = np.full((options.episodes, horizon, 4), np.nan)
            returns = np.full(options.episodes, np.nan)
            terminal_flags = np.zeros(options.episodes, dtype=int)
            truncation_flags = np.zeros(options.episodes, dtype=int)
            for ep, seed in enumerate(seeds):
                current, _ = envs[method].reset(seed=int(seed), theoretic_mode=True, rl_mode=False)
                states[ep, 0] = current
                total = 0.0
                for t in range(horizon):
                    u = np.clip(action(method, current), envs[method].action_space.low, envs[method].action_space.high)
                    if not np.all(np.isfinite(u)):
                        total = np.nan
                        break
                    current, reward, terminated, truncated, _ = envs[method].step(u)
                    states[ep, t + 1], controls[ep, t] = current, u
                    terminal_flags[ep] += int(terminated)
                    truncation_flags[ep] += int(truncated)
                    total += float(reward)
                    if not np.all(np.isfinite(current)) or not np.isfinite(total):
                        total = np.nan
                        break
                    # Legacy comparison evaluates the full fixed horizon, even after flags.
                returns[ep] = total
                print(f"{method}: episode {ep + 1}/{options.episodes}")
            distance = np.linalg.norm(states[:, -1, :3], axis=1)
            finite = np.all(np.isfinite(states), axis=(1, 2)) & np.isfinite(returns)
            success = finite & (distance <= 0.1)
            ymin = np.min(states[:, :, 1], axis=1)
            logy = np.log10(np.abs(ymin) + 1e-8)
            cost = -returns
            for ep in range(options.episodes):
                rows.append(dict(Controller=method, Episode=ep, Seed=int(seeds[ep]),
                                 Return=float(returns[ep]), J=float(cost[ep]),
                                 FinalPositionError=float(distance[ep]), Success=bool(success[ep]),
                                 MinY=float(ymin[ep]), LogAbsYMin=float(logy[ep]),
                                 TerminatedFlags=int(terminal_flags[ep]), TruncatedFlags=int(truncation_flags[ep])))
                if success[ep]:
                    cost_rows.append(dict(Controller=method, Episode=ep, Seed=int(seeds[ep]), Return=float(cost[ep])))
                    constraint_rows.append(dict(Controller=method, Episode=ep, Seed=int(seeds[ep]), LogAbsYMin=float(logy[ep])))
            all_data.update({f"{method}_{key}": value for key, value in {
                "states": states, "actions": controls, "returns": returns, "success_mask": success,
                "min_y": ymin, "log_abs_ymin": logy}.items()})
            summary[method] = {"success_rate": float(success.mean()), "successful_episodes": int(success.sum()),
                               "nonfinite_episodes": int((~finite).sum()),
                               "J_successful_only": average_sd(cost[success]),
                               "log_abs_ymin_successful_only": average_sd(logy[success]) if constrained else None}
            print(f"{method}: success={success.mean():.1%}, J(successful)={summary[method]['J_successful_only']}")
        data_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(data_dir / "all_trajectories_and_metrics.npz", episode_seeds=seeds, **all_data)
        pd.DataFrame(rows).to_csv(data_dir / "all_episode_metrics.csv", index=False)
        pd.DataFrame(cost_rows, columns=["Controller", "Episode", "Seed", "Return"]).to_csv(data_dir / "return_boxplot_data.csv", index=False)
        if constrained:
            pd.DataFrame(constraint_rows, columns=["Controller", "Episode", "Seed", "LogAbsYMin"]).to_csv(data_dir / "logy_boxplot_data.csv", index=False)
        record = {"environment": env_id, "episodes": options.episodes, "seed": options.seed,
                  "dt": env.dt, "horizon_steps": horizon, "success_position_tolerance": 0.1,
                  "evaluation_modes": {"theoretic_mode": True, "rl_mode": False},
                  "termination_policy": "fixed horizon; flags recorded, not used for early stopping; nonfinite rollouts aborted",
                  "cost_definition": "J = -sum(environment reward); constrained cost includes environment penalty",
                  "plot_population": "successful episodes only, separately for each method",
                  "constraint_metric": "log10(abs(min_y) + 1e-8), legacy convention; not violation depth",
                  "models": provenance, "methods": summary}
        (data_dir / "summary.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    finally:
        for environment in envs.values():
            environment.close()
        temporary = getattr(sources["OPINN"], "_env_tmp", None)
        if temporary is not None:
            temporary.close()
    draw(data_dir, image_dir, constrained)
    print(f"Saved new comparison: {root}")
