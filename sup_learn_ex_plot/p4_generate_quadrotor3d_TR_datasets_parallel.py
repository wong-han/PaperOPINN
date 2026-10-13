"""Generate both supervised-learning datasets with one command."""

import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count

import casadi as ca
import gymnasium as gym
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)
sys.path.append(PARENT_DIR)

EXPERIMENT_SEED = 42
TARGET_TRAJECTORIES = 200
MAX_GENERATION_ATTEMPTS = 1000
TERMINAL_STATE_TOL = 1e-2
MAX_TRAJECTORY_SECONDS = 5.0  # Discard and resample any trajectory taking longer than 5.0 s (500 steps)
MPC_HORIZON = 20
MAX_WORKERS = 24

PROBLEMS = {
    "quadratic": {
        "env_id": "environment:quadrotor3d_TR-v0",
        "constrained": False,
        "state_file": "X_trajs.npy",
        "action_file": "U_trajs.npy",
    },
    "constrained": {
        "env_id": "environment:quadrotor3d_TR_constrained",
        "constrained": True,
        "state_file": "X_trajs_constrained.npy",
        "action_file": "U_trajs_constrained.npy",
    },
}


def _path(*parts):
    return os.path.join(SCRIPT_DIR, *parts)


def _save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _setup_casadi_mpc(env, constrained):
    env = env.unwrapped if hasattr(env, "unwrapped") else env
    opti = ca.Opti()
    dt = 0.2
    x = opti.variable(9, MPC_HORIZON + 1)
    u = opti.variable(4, MPC_HORIZON)
    x0 = opti.parameter(9, 1)
    opti.subject_to(x[:, 0] == x0)

    q_matrix = env.Q
    r_matrix = env.R
    equilibrium_action = env.u0
    mass = env.metadata["m"]
    gravity = env.metadata["g"]
    cost = 0

    for step in range(MPC_HORIZON):
        state = x[:, step]
        action = u[:, step]
        x_pos, y_pos, z_pos = state[0], state[1], state[2]
        vx, vy, vz = state[3], state[4], state[5]
        phi, theta, psi = state[6], state[7], state[8]
        thrust, p_rate, q_rate, r_rate = action[0], action[1], action[2], action[3]

        cos_phi, sin_phi = ca.cos(phi), ca.sin(phi)
        cos_theta, sin_theta = ca.cos(theta), ca.sin(theta)
        cos_theta_safe = ca.if_else(ca.fabs(cos_theta) < 1e-6, 1e-6, cos_theta)
        tan_theta = sin_theta / cos_theta_safe
        cos_psi, sin_psi = ca.cos(psi), ca.sin(psi)

        derivative = ca.vertcat(
            vx,
            vy,
            vz,
            thrust * (cos_phi * sin_theta * cos_psi + sin_phi * sin_psi) / mass,
            thrust * (cos_phi * sin_theta * sin_psi - sin_phi * cos_psi) / mass,
            thrust * cos_phi * cos_theta / mass - gravity,
            p_rate + tan_theta * sin_phi * q_rate + tan_theta * cos_phi * r_rate,
            cos_phi * q_rate - sin_phi * r_rate,
            sin_phi / cos_theta_safe * q_rate + cos_phi / cos_theta_safe * r_rate,
        )
        opti.subject_to(x[:, step + 1] == state + derivative * dt)
        action_error = action - equilibrium_action
        cost += (
            ca.mtimes([state.T, q_matrix, state]) * dt
            + ca.mtimes([action_error.T, r_matrix, action_error]) * dt
        )

    terminal_state = x[:, MPC_HORIZON]
    cost += ca.mtimes([terminal_state.T, env.P, terminal_state])
    opti.minimize(cost)

    if constrained:
        opti.subject_to(x[1, :] >= 0.0)
    for index in range(9):
        opti.subject_to(opti.bounded(env.state_low[index], x[index, :], env.state_high[index]))
    for index in range(4):
        opti.subject_to(
            opti.bounded(env.action_space.low[index], u[index, :], env.action_space.high[index])
        )

    solver_options = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.sb": "yes",
        "ipopt.max_iter": 500,
    }
    opti.solver("ipopt", solver_options)
    return opti, x0, u


def _generate_single_trajectory(task):
    problem_name, trajectory_index = task
    config = PROBLEMS[problem_name]
    rng = np.random.default_rng(EXPERIMENT_SEED + trajectory_index)
    env = gym.make(config["env_id"], theoretic_mode=True).unwrapped
    opti, initial_state_parameter, action_variable = _setup_casadi_mpc(
        env, config["constrained"]
    )

    initial_state = rng.uniform(-1.0, 1.0, 9)
    if config["constrained"]:
        initial_state[1] = abs(initial_state[1])
    state, _ = env.reset(theoretic_mode=True, options={"init_state": initial_state})
    action_guess = np.repeat(
        np.asarray(env.u0, dtype=float)[:, None], MPC_HORIZON, axis=1
    )

    state_trajectory = []
    action_trajectory = []
    valid = True
    terminal_reached = False

    max_steps = int(min(env.T, MAX_TRAJECTORY_SECONDS) / env.dt)
    for _ in range(max_steps):
        opti.set_value(initial_state_parameter, state)
        opti.set_initial(action_variable, action_guess)
        try:
            solution = opti.solve()
            optimal_action = solution.value(action_variable[:, 0])
            state_trajectory.append(state.copy())
            action_trajectory.append(optimal_action.copy())
            action_sequence = solution.value(action_variable)
            if action_sequence.ndim == 1:
                action_sequence = action_sequence.reshape(-1, 1)
            action_guess[:, :-1] = action_sequence[:, 1:]
            action_guess[:, -1] = action_sequence[:, -1]
        except Exception:
            valid = False
            break

        state, _, _, truncated, info = env.step(optimal_action)
        if info.get("data", {}).get("crashed", [False])[-1]:
            valid = False
            break
        if np.linalg.norm(state[:6]) <= TERMINAL_STATE_TOL:
            terminal_reached = True
            break
        if truncated:
            break

    env.close()
    if not valid or not state_trajectory or not terminal_reached:
        return None
    if len(state_trajectory) * env.dt > MAX_TRAJECTORY_SECONDS:
        return None
    return (
        np.asarray(state_trajectory, dtype=np.float32),
        np.asarray(action_trajectory, dtype=np.float32),
        terminal_reached,
        float(np.linalg.norm(state[:6])),
    )


def generate_problem_dataset(problem_name):
    config = PROBLEMS[problem_name]
    workers = min(cpu_count(), MAX_WORKERS)
    print(
        f"--- Generating {problem_name} dataset: "
        f"{TARGET_TRAJECTORIES} trajectories on {workers} workers ---"
    )
    generation_start = time.perf_counter()
    valid_results = []
    attempts = 0

    with Pool(processes=workers) as pool:
        while len(valid_results) < TARGET_TRAJECTORIES and attempts < MAX_GENERATION_ATTEMPTS:
            batch_count = min(
                TARGET_TRAJECTORIES - len(valid_results),
                MAX_GENERATION_ATTEMPTS - attempts,
            )
            tasks = [
                (problem_name, index)
                for index in range(attempts, attempts + batch_count)
            ]
            attempts += batch_count
            results = pool.map(_generate_single_trajectory, tasks)
            valid_results.extend(result for result in results if result is not None)
            print(
                f"[{problem_name}] valid={len(valid_results)}/{TARGET_TRAJECTORIES}, "
                f"attempts={attempts}"
            )

    if len(valid_results) < TARGET_TRAJECTORIES:
        raise RuntimeError(
            f"Only {len(valid_results)} valid {problem_name} trajectories were generated "
            f"after {attempts} attempts."
        )

    valid_results = valid_results[:TARGET_TRAJECTORIES]
    generation_seconds = time.perf_counter() - generation_start
    state_trajectories = [result[0] for result in valid_results]
    action_trajectories = [result[1] for result in valid_results]
    terminal_reached = np.asarray([result[2] for result in valid_results], dtype=bool)
    terminal_errors = np.asarray([result[3] for result in valid_results], dtype=float)

    data_dir = _path("data")
    results_dir = _path("results")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    state_path = os.path.join(data_dir, config["state_file"])
    action_path = os.path.join(data_dir, config["action_file"])
    np.save(state_path, np.asarray(state_trajectories, dtype=object), allow_pickle=True)
    np.save(action_path, np.asarray(action_trajectories, dtype=object), allow_pickle=True)

    record = {
        "problem": problem_name,
        "seed": EXPERIMENT_SEED,
        "target_trajectories": TARGET_TRAJECTORIES,
        "valid_trajectories": len(state_trajectories),
        "generation_attempts": attempts,
        "samples": int(sum(len(trajectory) for trajectory in state_trajectories)),
        "workers": workers,
        "max_trajectory_seconds": MAX_TRAJECTORY_SECONDS,
        "terminal_position_velocity_l2_tolerance": TERMINAL_STATE_TOL,
        "trajectories_reaching_terminal_tolerance": int(np.sum(terminal_reached)),
        "terminal_error_mean": float(np.mean(terminal_errors)),
        "terminal_error_std": float(np.std(terminal_errors)),
        "data_generation_seconds": generation_seconds,
        "state_file": state_path,
        "action_file": action_path,
    }
    _save_json(os.path.join(results_dir, f"{problem_name}_data_generation.json"), record)
    print(
        f"[{problem_name}] saved {len(state_trajectories)} trajectories/"
        f"{record['samples']} samples in {generation_seconds:.2f} seconds."
    )
    return record


def main():
    total_start = time.perf_counter()
    records = {
        problem_name: generate_problem_dataset(problem_name)
        for problem_name in ("quadratic", "constrained")
    }
    summary = {
        "seed": EXPERIMENT_SEED,
        "total_generation_seconds": time.perf_counter() - total_start,
        "problems": records,
    }
    _save_json(_path("results", "dataset_generation_summary.json"), summary)
    print("Both datasets and all generation-time records have been saved.")


if __name__ == "__main__":
    main()
