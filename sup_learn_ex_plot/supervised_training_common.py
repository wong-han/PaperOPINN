import json
import os
import time

import numpy as np
import torch


def _save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _dataset_loss(model, x_tensor, u_tensor, batch_size):
    model.eval()
    squared_error_sum = 0.0
    element_count = 0
    with torch.no_grad():
        for start in range(0, len(x_tensor), batch_size):
            prediction = model(x_tensor[start:start + batch_size])
            target = u_tensor[start:start + batch_size]
            squared_error_sum += torch.sum((prediction - target) ** 2).item()
            element_count += target.numel()
    model.train()
    return squared_error_sum / element_count


def _timed_dataset_loss(model, x_tensor, u_tensor, batch_size, device):
    # Finish pending optimizer work before timing evaluation on CUDA.
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    evaluation_start = time.perf_counter()
    loss = _dataset_loss(model, x_tensor, u_tensor, batch_size)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    return loss, time.perf_counter() - evaluation_start


def train_from_saved_trajectories(
    *,
    model,
    optimizer,
    device,
    experiment_name,
    x_path,
    u_path,
    model_path,
    result_path,
    training_seed,
    requested_batch_size,
    evaluation_interval,
    maximum_training_steps,
    target_training_mse,
    required_target_checks,
):
    if not os.path.exists(x_path) or not os.path.exists(u_path):
        raise FileNotFoundError(
            "Shared dataset not found. Run the corresponding problem-specific "
            "dataset-generation script first.\n"
            f"Expected:\n  {x_path}\n  {u_path}"
        )

    x_trajectories = np.load(x_path, allow_pickle=True)
    u_trajectories = np.load(u_path, allow_pickle=True)
    if len(x_trajectories) != len(u_trajectories) or len(x_trajectories) < 2:
        raise ValueError("The shared state/action trajectory files are inconsistent or too small.")

    training_indices = np.arange(len(x_trajectories), dtype=int)
    x_train = np.vstack(x_trajectories).astype(np.float32)
    u_train = np.vstack(u_trajectories).astype(np.float32)

    x_train_tensor = torch.from_numpy(x_train).to(device)
    u_train_tensor = torch.from_numpy(u_train).to(device)
    batch_size = min(requested_batch_size, len(x_train))

    print(
        f"Loaded shared dataset: all {len(training_indices)} trajectories/"
        f"{len(x_train)} samples are used for training."
    )
    print(
        f"Stopping target: full training MSE <= {target_training_mse:.6g} "
        f"for {required_target_checks} consecutive evaluations"
    )
    print("--- Training Network ---")

    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    training_start = time.perf_counter()

    step = 0
    training_loss_history = []
    best_training_loss = float("inf")
    best_state = None
    stop_reason = "max_steps"
    target_check_count = 0
    periodic_mse_evaluation_seconds = 0.0

    while step < maximum_training_steps:
        step += 1
        batch_indices = np.random.choice(len(x_train), batch_size)
        prediction = model(x_train_tensor[batch_indices])
        loss = torch.nn.functional.mse_loss(prediction, u_train_tensor[batch_indices])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_batch_loss = loss.item()

        if step % evaluation_interval != 0:
            continue

        training_full_loss, evaluation_seconds = _timed_dataset_loss(
            model, x_train_tensor, u_train_tensor, batch_size, device
        )
        periodic_mse_evaluation_seconds += evaluation_seconds
        training_loss_history.append(training_full_loss)

        if training_full_loss < best_training_loss:
            best_training_loss = training_full_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

        if training_full_loss <= target_training_mse:
            target_check_count += 1
        else:
            target_check_count = 0
        print(
            f"Step {step}, train_batch={training_batch_loss:.6g}, "
            f"train_full={training_full_loss:.6g}, "
            f"target_checks={target_check_count}/{required_target_checks}"
        )

        if target_check_count >= required_target_checks:
            stop_reason = "training_loss_target_sustained"
            print(
                "Training stopped because the full training MSE remained below "
                "the target for the required number of evaluations."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    training_wall_seconds = time.perf_counter() - training_start
    training_seconds = training_wall_seconds - periodic_mse_evaluation_seconds

    selected_training_loss, final_mse_evaluation_seconds = _timed_dataset_loss(
        model, x_train_tensor, u_train_tensor, batch_size, device
    )

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)

    record = {
        "experiment": experiment_name,
        "seed": training_seed,
        "device": str(device),
        "dataset_state_file": os.path.abspath(x_path),
        "dataset_action_file": os.path.abspath(u_path),
        "trajectories": len(x_trajectories),
        "training_trajectories": len(training_indices),
        "training_samples": len(x_train),
        "training_trajectory_indices": training_indices.tolist(),
        "batch_size": batch_size,
        "steps": step,
        "target_training_mse": target_training_mse,
        "required_target_checks": required_target_checks,
        "final_target_checks": target_check_count,
        "target_reached": stop_reason == "training_loss_target_sustained",
        "selected_model_training_mse": selected_training_loss,
        "best_training_mse": best_training_loss,
        "monitored_training_mse": training_loss_history,
        "monitor_steps": [
            evaluation_interval * (index + 1)
            for index in range(len(training_loss_history))
        ],
        "stop_reason": stop_reason,
        "training_seconds": training_seconds,
        "training_wall_seconds": training_wall_seconds,
        "periodic_mse_evaluation_seconds": periodic_mse_evaluation_seconds,
        "final_mse_evaluation_seconds": final_mse_evaluation_seconds,
        "full_dataset_mse_evaluation_seconds": (
            periodic_mse_evaluation_seconds + final_mse_evaluation_seconds
        ),
        "training_time_excludes_full_dataset_mse": True,
        "stopping_rule": {
            "evaluation_interval_steps": evaluation_interval,
            "metric": "full_training_mse",
            "target": target_training_mse,
            "required_consecutive_evaluations": required_target_checks,
            "required_consecutive_steps": required_target_checks * evaluation_interval,
            "maximum_training_steps": maximum_training_steps,
        },
    }
    _save_json(result_path, record)
    print(
        f"Model saved to {model_path}. Training time (excluding full-dataset MSE): "
        f"{training_seconds:.2f} seconds. "
        f"Full-dataset MSE evaluation time: "
        f"{periodic_mse_evaluation_seconds + final_mse_evaluation_seconds:.2f} seconds. "
        f"Selected model training MSE: {selected_training_loss:.6g}."
    )
    return record
