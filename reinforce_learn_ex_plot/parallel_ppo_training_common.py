"""Shared parallel PPO training loop for the two quadrotor problems."""

import json
import math
import os
import sys
import time
from functools import partial

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


COMMON_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(COMMON_DIR)
if COMMON_DIR not in sys.path:
    sys.path.append(COMMON_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)


def _make_environment(env_id):
    # PPO training uses shaped rewards and the RL-specific success threshold.
    # Monte Carlo evaluation creates separate environments in theoretic mode.
    return gym.make(env_id, theoretic_mode=False, rl_mode=True)


def _save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _save_learning_curve(path, steps, rollout_means, rolling_means, window_size):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 5))
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
    if finite_rollout:
        x, y = zip(*finite_rollout)
        axis.plot(x, y, color="tab:blue", alpha=0.25, linewidth=1.0,
                  label="Mean return in rollout")
    if finite_rolling:
        x, y = zip(*finite_rolling)
        axis.plot(x, y, color="tab:blue", linewidth=2.2,
                  label=f"Rolling mean ({window_size} episodes)")
    axis.set_xlabel("Environment interactions")
    axis.set_ylabel("Mean episodic return")
    axis.grid(True, linestyle="--", alpha=0.4)
    if finite_rollout or finite_rolling:
        axis.legend()
    figure.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


class _CyclicShuffledSampler:
    """Return fixed-size minibatches while cycling uniformly through a rollout."""

    def __init__(self, sample_count, rng):
        self.sample_count = sample_count
        self.rng = rng
        self.order = self.rng.permutation(sample_count)
        self.position = 0

    def next(self, minibatch_size):
        pieces = []
        remaining = minibatch_size
        while remaining > 0:
            available = self.sample_count - self.position
            take = min(remaining, available)
            pieces.append(self.order[self.position:self.position + take])
            self.position += take
            remaining -= take
            if self.position == self.sample_count:
                self.order = self.rng.permutation(self.sample_count)
                self.position = 0
        return np.concatenate(pieces)


def train_parallel_ppo(
    *, args, agent_class, script_dir, model_filename, result_filename,
    learning_curve_filename
):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    env_fns = [partial(_make_environment, args.env_id) for _ in range(args.num_envs)]
    envs = gym.vector.AsyncVectorEnv(env_fns, shared_memory=True)
    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]
    action_low = envs.single_action_space.low
    action_high = envs.single_action_space.high

    agent = agent_class(state_dim, action_dim).to(args.device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    rollout_size = args.num_envs * args.num_steps
    if args.minibatch_size > rollout_size:
        envs.close()
        raise ValueError(
            f"minibatch_size={args.minibatch_size} exceeds rollout_size={rollout_size}."
        )
    rollout_count = math.ceil(args.time_steps / rollout_size)
    actual_total_timesteps = rollout_count * rollout_size

    obs = torch.zeros(
        (args.num_steps, args.num_envs, state_dim), device=args.device
    )
    actions = torch.zeros(
        (args.num_steps, args.num_envs, action_dim), device=args.device
    )
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=args.device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=args.device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=args.device)
    values = torch.zeros((args.num_steps, args.num_envs), device=args.device)

    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=args.device)
    next_done = torch.zeros(args.num_envs, device=args.device)

    print(
        f"Starting parallel PPO: envs={args.num_envs}, steps_per_env={args.num_steps}, "
        f"rollout_size={rollout_size}, minibatch={args.minibatch_size}, "
        f"gradient_updates_per_rollout={args.gradient_updates_per_rollout}, "
        f"requested_timesteps={args.time_steps}, actual_timesteps={actual_total_timesteps}."
    )

    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    training_start = time.perf_counter()
    sampling_seconds = 0.0
    global_step = 0
    final_policy_loss = float("nan")
    final_value_loss = float("nan")
    running_episode_returns = np.zeros(args.num_envs, dtype=np.float64)
    completed_episode_returns = []
    learning_curve_steps = []
    rollout_mean_returns = []
    rolling_mean_returns = []
    completed_episode_counts = []

    try:
        for rollout in range(1, rollout_count + 1):
            sampling_start = time.perf_counter()
            returns_completed_this_rollout = []
            for step in range(args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                action_np = action.cpu().numpy()
                scaled_action = (
                    action_np * (action_high - action_low) / 2.0
                    + (action_high + action_low) / 2.0
                )
                next_obs_np, reward, terminated, truncated, _ = envs.step(
                    scaled_action
                )
                reward_array = np.asarray(reward, dtype=np.float64).reshape(
                    args.num_envs
                )
                done_array = np.logical_or(terminated, truncated)
                running_episode_returns += reward_array
                for environment_index in np.flatnonzero(done_array):
                    episode_return = float(
                        running_episode_returns[environment_index]
                    )
                    completed_episode_returns.append(episode_return)
                    returns_completed_this_rollout.append(episode_return)
                    running_episode_returns[environment_index] = 0.0
                rewards[step] = torch.as_tensor(
                    reward, dtype=torch.float32, device=args.device
                )
                next_obs = torch.as_tensor(
                    next_obs_np, dtype=torch.float32, device=args.device
                )
                next_done = torch.as_tensor(
                    done_array,
                    dtype=torch.float32,
                    device=args.device,
                )
            sampling_seconds += time.perf_counter() - sampling_start

            rollout_mean = (
                float(np.mean(returns_completed_this_rollout))
                if returns_completed_this_rollout else None
            )
            rolling_mean = (
                float(np.mean(completed_episode_returns[-args.return_window_episodes:]))
                if completed_episode_returns else None
            )
            learning_curve_steps.append(global_step)
            rollout_mean_returns.append(rollout_mean)
            rolling_mean_returns.append(rolling_mean)
            completed_episode_counts.append(len(completed_episode_returns))

            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(-1)
                advantages = torch.zeros_like(rewards)
                last_gae = torch.zeros(args.num_envs, device=args.device)
                for step in reversed(range(args.num_steps)):
                    if step == args.num_steps - 1:
                        next_nonterminal = 1.0 - next_done
                        next_values = next_value
                    else:
                        next_nonterminal = 1.0 - dones[step + 1]
                        next_values = values[step + 1]
                    delta = (
                        rewards[step]
                        + args.gamma * next_values * next_nonterminal
                        - values[step]
                    )
                    last_gae = (
                        delta
                        + args.gamma
                        * args.gae_lambda
                        * next_nonterminal
                        * last_gae
                    )
                    advantages[step] = last_gae
                returns = advantages + values

            batch_obs = obs.reshape((-1, state_dim))
            batch_logprobs = logprobs.reshape(-1)
            batch_actions = actions.reshape((-1, action_dim))
            batch_advantages = advantages.reshape(-1)
            batch_returns = returns.reshape(-1)

            sampler = _CyclicShuffledSampler(
                rollout_size, np.random.default_rng(args.seed + rollout)
            )
            for _ in range(args.gradient_updates_per_rollout):
                minibatch_indices = sampler.next(args.minibatch_size)
                _, new_logprob, entropy, new_value = agent.get_action_and_value(
                    batch_obs[minibatch_indices], batch_actions[minibatch_indices]
                )
                log_ratio = new_logprob - batch_logprobs[minibatch_indices]
                ratio = log_ratio.exp()

                minibatch_advantages = batch_advantages[minibatch_indices]
                minibatch_advantages = (
                    minibatch_advantages - minibatch_advantages.mean()
                ) / (minibatch_advantages.std() + 1e-8)

                policy_loss_unclipped = -minibatch_advantages * ratio
                policy_loss_clipped = -minibatch_advantages * torch.clamp(
                    ratio, 1 - args.clip_param, 1 + args.clip_param
                )
                policy_loss = torch.max(
                    policy_loss_unclipped, policy_loss_clipped
                ).mean()
                value_loss = 0.5 * (
                    (new_value.view(-1) - batch_returns[minibatch_indices]) ** 2
                ).mean()
                entropy_loss = entropy.mean()
                loss = policy_loss - 0.01 * entropy_loss + 0.5 * value_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

            if str(args.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            final_policy_loss = float(policy_loss.item())
            final_value_loss = float(value_loss.item())
            print(
                f"Rollout {rollout}/{rollout_count}, global_step={global_step}, "
                f"policy_loss={final_policy_loss:.6g}, "
                f"value_loss={final_value_loss:.6g}, "
                f"rolling_mean_return={rolling_mean if rolling_mean is not None else 'n/a'}"
            )
    finally:
        envs.close()

    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    total_seconds = time.perf_counter() - training_start
    training_seconds = max(0.0, total_seconds - sampling_seconds)

    model_path = os.path.join(script_dir, "model", model_filename)
    result_path = os.path.join(script_dir, "results", result_filename)
    learning_curve_path = os.path.join(
        script_dir, "image", learning_curve_filename
    )
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(agent.state_dict(), model_path)
    _save_learning_curve(
        learning_curve_path,
        learning_curve_steps,
        rollout_mean_returns,
        rolling_mean_returns,
        args.return_window_episodes,
    )

    finite_curve_indices = [
        index for index, value in enumerate(rolling_mean_returns)
        if value is not None
    ]
    stability_window_size = min(10, len(finite_curve_indices))
    stability_indices = finite_curve_indices[-stability_window_size:]
    if stability_window_size >= 2:
        stability_steps = np.asarray(
            [learning_curve_steps[index] for index in stability_indices], dtype=float
        )
        stability_returns = np.asarray(
            [rolling_mean_returns[index] for index in stability_indices], dtype=float
        )
        return_slope_per_100k_steps = float(
            np.polyfit(stability_steps / 100000.0, stability_returns, 1)[0]
        )
        final_return_mean = float(np.mean(stability_returns))
        final_return_std = float(np.std(stability_returns))
    else:
        return_slope_per_100k_steps = None
        final_return_mean = None
        final_return_std = None

    record = {
        "environment": args.env_id,
        "seed": args.seed,
        "device": str(args.device),
        "parallel_backend": "gymnasium.AsyncVectorEnv",
        "num_envs": args.num_envs,
        "steps_per_environment": args.num_steps,
        "rollout_size": rollout_size,
        "requested_total_timesteps": args.time_steps,
        "actual_total_timesteps": actual_total_timesteps,
        "rollouts": rollout_count,
        "minibatch_size": args.minibatch_size,
        "gradient_updates_per_rollout": args.gradient_updates_per_rollout,
        "average_sample_uses_per_rollout": (
            args.gradient_updates_per_rollout * args.minibatch_size / rollout_size
        ),
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_param": args.clip_param,
        "data_collection_seconds": sampling_seconds,
        "training_seconds": training_seconds,
        "total_seconds": total_seconds,
        "return_window_episodes": args.return_window_episodes,
        "learning_curve": {
            "environment_interactions": learning_curve_steps,
            "rollout_mean_returns": rollout_mean_returns,
            "rolling_mean_returns": rolling_mean_returns,
            "completed_episode_counts": completed_episode_counts,
        },
        "final_stability_summary": {
            "rollouts_used": stability_window_size,
            "rolling_return_mean": final_return_mean,
            "rolling_return_std": final_return_std,
            "return_slope_per_100k_interactions": return_slope_per_100k_steps,
        },
        "final_policy_loss": final_policy_loss,
        "final_value_loss": final_value_loss,
        "model_file": model_path,
        "learning_curve_file": learning_curve_path,
    }
    _save_json(result_path, record)
    print(f"RL PPO model saved to {model_path}")
    print(f"Training record saved to {result_path}")
    print(
        f"Data collection: {sampling_seconds:.2f} s; "
        f"training: {training_seconds:.2f} s; "
        f"total: {total_seconds:.2f} s."
    )
    return record
