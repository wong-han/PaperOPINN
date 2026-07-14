import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_data(file_path):
    print(f"Loading data from: {file_path.name}")
    try:
        data = np.genfromtxt(file_path, delimiter="\t", skip_header=1, encoding="utf-8-sig")
    except Exception as e:
        print(f"Error loading {file_path.name} with utf-8-sig: {e}")
        data = np.genfromtxt(file_path, delimiter="\t", skip_header=1)
    
    # Clean nan or incomplete lines
    data = data[np.all(np.isfinite(data), axis=1)]
    return data

def compute_performance(state_data, u_data, Q, R, target_pos, u0, start_time_offset=6.0, end_time_offset=10.0):
    t_state = state_data[:, 0]
    x_state = state_data[:, 1:] # columns: x, y, z, roll, pitch, yaw, vx, vy, vz
    
    t_u = u_data[:, 0]
    u_vals = u_data[:, 1:]
    
    # The control starts at t_u[0]. 
    # To remove the first ~6s waiting time from state_data, we align state_data start to t_u[0].
    t_start = t_u[0]
    
    # Filter state data based on relative time window
    if end_time_offset is not None:
        window_duration = end_time_offset - start_time_offset  # e.g., 10.0 - 6.0 = 4.0 seconds
        mask = (t_state >= t_start) & (t_state <= t_start + window_duration)
    else:
        mask = (t_state >= t_start)
        
    t_filtered = t_state[mask]
    x_filtered = x_state[mask, :]
    
    if len(t_filtered) < 2:
        return np.nan, 0.0
        
    duration = t_filtered[-1] - t_filtered[0]
    
    # Interpolate control data to match state timestamps
    u_interp = np.column_stack([np.interp(t_filtered, t_u, u_vals[:, i]) for i in range(u_vals.shape[1])])
    
    # Target state: first 3 components are target_pos, rest are 0
    X_EQ = np.zeros(9)
    X_EQ[:3] = target_pos
    
    # State error
    x_error = x_filtered - X_EQ
    
    # Control error
    if u0 is not None:
        u_error = u_interp - u0.reshape(1, -1)
    else:
        u_error = u_interp
        
    # Cost integrand: x_err^T * Q * x_err + u_err^T * R * u_err
    xQx = np.einsum('ij,jk,ik->i', x_error, Q, x_error)
    uRu = np.einsum('ij,jk,ik->i', u_error, R, u_error)
    integrand = xQx + uRu
    
    # Trapezoidal integration
    dt = np.diff(t_filtered)
    J = np.sum(0.5 * (integrand[:-1] + integrand[1:]) * dt)
    
    return J, duration

def compute_pos_mse(state_data, u_data, target_pos):
    t_state = state_data[:, 0]
    pos = state_data[:, 1:4] # x, y, z
    
    t_u = u_data[:, 0]
    t_start = t_u[0]  # Start evaluating MSE exactly when control begins
    
    mask = (t_state >= t_start)
    pos_filtered = pos[mask, :]
    
    if len(pos_filtered) == 0:
        return np.nan
        
    mse = np.mean(np.sum((pos_filtered - target_pos) ** 2, axis=1))
    return mse

def compute_pos_mse_curve(state_data, u_data, target_pos):
    t_state = state_data[:, 0]
    pos = state_data[:, 1:4] # x, y, z

    t_u = u_data[:, 0]
    t_start = t_u[0]  # Start evaluating MSE exactly when control begins

    mask = (t_state >= t_start)
    t_filtered = t_state[mask]
    pos_filtered = pos[mask, :]

    if len(pos_filtered) == 0:
        return np.array([]), np.array([]), np.array([])

    squared_position_error = np.sum((pos_filtered - target_pos) ** 2, axis=1)
    cumulative_mse = np.cumsum(squared_position_error) / np.arange(1, len(squared_position_error) + 1)
    relative_time = t_filtered - t_filtered[0]
    return relative_time, squared_position_error, cumulative_mse

def save_mse_time_curve(
    hnn_state_data,
    hnn_u_data,
    lqr_state_data,
    lqr_u_data,
    target_pos,
    output_dir,
):
    hnn_t, hnn_squared_error, hnn_cumulative_mse = compute_pos_mse_curve(
        hnn_state_data, hnn_u_data, target_pos
    )
    lqr_t, lqr_squared_error, lqr_cumulative_mse = compute_pos_mse_curve(
        lqr_state_data, lqr_u_data, target_pos
    )

    if len(hnn_t) == 0 or len(lqr_t) == 0:
        print("Warning: no valid MSE curve samples, skipping MSE time curve plot.")
        return None, None

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(hnn_t, hnn_cumulative_mse, color="#d62728", linewidth=2.0, label="HNN (Ours)")
    ax.plot(lqr_t, lqr_cumulative_mse, color="#2ca02c", linestyle="--", linewidth=2.0, label="LQR")
    ax.set_xlabel("Time since control start (s)")
    ax.set_ylabel("Cumulative position MSE")
    ax.set_title("Position MSE over Time")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()

    plot_path = output_dir / "mse_time_curve.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    max_len = max(len(hnn_t), len(lqr_t))
    curve_data = np.full((max_len, 6), np.nan)
    curve_data[: len(hnn_t), 0] = hnn_t
    curve_data[: len(hnn_t), 1] = hnn_squared_error
    curve_data[: len(hnn_t), 2] = hnn_cumulative_mse
    curve_data[: len(lqr_t), 3] = lqr_t
    curve_data[: len(lqr_t), 4] = lqr_squared_error
    curve_data[: len(lqr_t), 5] = lqr_cumulative_mse

    csv_path = output_dir / "mse_time_curve.csv"
    np.savetxt(
        csv_path,
        curve_data,
        delimiter=",",
        header=(
            "hnn_time_s,hnn_squared_position_error,hnn_cumulative_mse,"
            "lqr_time_s,lqr_squared_position_error,lqr_cumulative_mse"
        ),
        comments="",
        fmt="%.10g",
    )

    return plot_path, csv_path

def main():
    script_dir = Path(__file__).resolve().parent
    hnn_dir = script_dir / "hnn"
    lqr_dir = script_dir / "lqr"
    
    # Parameters
    # Target position (X_EQ[:3] from visualize_performance.py is [0.55, 0.0, 0.9])
    target_pos = np.array([0.0, 0.0, 0.9])
    
    # Cost matrices (from visualize_performance.py)
    Q = np.diag([5.0, 5.0, 20.0, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1])
    R = np.diag([10.0, 1.0, 1.0, 1.0])
    
    # Equilibrium control (u0 from old script)
    u0_old = np.array([0.27, 0.0, 0.21, 0.0])
    u0_new = np.array([0.27, 0.0, 0.35, 0.0]) # aligned with new hover values
    
    # Locate files
    hnn_files = sorted(list(hnn_dir.glob("state_data_*.txt")))
    lqr_files = sorted(list(lqr_dir.glob("state_data_*.txt")))
    
    if not hnn_files or not lqr_files:
        print("Error: state data files missing.")
        return
        
    # Load files at index 0 (as updated by user in compare_hnn_lqr.py)
    hnn_state_file = hnn_files[3] if len(hnn_files) > 0 else hnn_files[-1]
    lqr_state_file = lqr_files[3] if len(lqr_files) > 0 else lqr_files[-1]
    
    hnn_suffix = hnn_state_file.name.replace("state_data_", "").replace(".txt", "")
    hnn_u_file = hnn_dir / f"u_data_{hnn_suffix}.txt"
    
    lqr_suffix = lqr_state_file.name.replace("state_data_", "").replace(".txt", "")
    lqr_u_file = lqr_dir / f"u_data_{lqr_suffix}.txt"
    
    # Load HNN & LQR data
    hnn_state_data = load_data(hnn_state_file)
    hnn_u_data = load_data(hnn_u_file)
    lqr_state_data = load_data(lqr_state_file)
    lqr_u_data = load_data(lqr_u_file)
    
    print("\n" + "="*50)
    print("      QUANTITATIVE PERFORMANCE COMPARISON")
    print("      (Aligned exactly to control start time)")
    print("="*50)
    print(f"HNN files: {hnn_state_file.name} / {hnn_u_file.name}")
    print(f"LQR files: {lqr_state_file.name} / {lqr_u_file.name}")
    print(f"Target Position: {target_pos}")
    print(f"Q (diag): {np.diag(Q)}")
    print(f"R (diag): {np.diag(R)}")
    print("-"*50)
    
    # --- 1. Calculate Trajectory Position MSE (t >= t_u[0]) ---
    mse_hnn = compute_pos_mse(hnn_state_data, hnn_u_data, target_pos)
    mse_lqr = compute_pos_mse(lqr_state_data, lqr_u_data, target_pos)
    
    print(f"HNN Trajectory Position MSE (from control start): {mse_hnn:.6f}")
    print(f"LQR Trajectory Position MSE (from control start): {mse_lqr:.6f}")
    print(f"Improvement (MSE): {((mse_lqr - mse_hnn) / mse_lqr * 100.0):.2f}%")
    mse_plot_path, mse_csv_path = save_mse_time_curve(
        hnn_state_data,
        hnn_u_data,
        lqr_state_data,
        lqr_u_data,
        target_pos,
        script_dir / "images",
    )
    if mse_plot_path is not None and mse_csv_path is not None:
        print(f"MSE time curve plot: {mse_plot_path}")
        print(f"MSE time curve CSV: {mse_csv_path}")
    print("-"*50)
    
    # --- 2. Calculate J (first 4.0s of control) with different u0 definitions ---
    print("Integration Window: [0.0s, 4.0s] of control (Fair Comparison Window)")
    
    # u0 = [0.27, 0, 0.21, 0]
    J_hnn_old, _ = compute_performance(hnn_state_data, hnn_u_data, Q, R, target_pos, u0_old, 6.0, 10.0)
    J_lqr_old, _ = compute_performance(lqr_state_data, lqr_u_data, Q, R, target_pos, u0_old, 6.0, 10.0)
    print(f"  J (u0={u0_old}) -> HNN: {J_hnn_old:.6f} | LQR: {J_lqr_old:.6f} | Imp: {((J_lqr_old - J_hnn_old) / J_lqr_old * 100.0):.2f}%")
    
    # u0 = [0.27, 0, 0.35, 0]
    J_hnn_new, _ = compute_performance(hnn_state_data, hnn_u_data, Q, R, target_pos, u0_new, 6.0, 10.0)
    J_lqr_new, _ = compute_performance(lqr_state_data, lqr_u_data, Q, R, target_pos, u0_new, 6.0, 10.0)
    print(f"  J (u0={u0_new}) -> HNN: {J_hnn_new:.6f} | LQR: {J_lqr_new:.6f} | Imp: {((J_lqr_new - J_hnn_new) / J_lqr_new * 100.0):.2f}%")
    
    # No u0 (direct control cost)
    J_hnn_no, _ = compute_performance(hnn_state_data, hnn_u_data, Q, R, target_pos, None, 6.0, 10.0)
    J_lqr_no, _ = compute_performance(lqr_state_data, lqr_u_data, Q, R, target_pos, None, 6.0, 10.0)
    print(f"  J (no u0 subtraction)     -> HNN: {J_hnn_no:.6f} | LQR: {J_lqr_no:.6f} | Imp: {((J_lqr_no - J_hnn_no) / J_lqr_no * 100.0):.2f}%")
    print("-"*50)
    
    # --- 3. Calculate J (from control start to end of state data) ---
    print("Integration Window: [0.0s, End] of control (Full Steady State Window)")
    J_hnn_end, hnn_dur = compute_performance(hnn_state_data, hnn_u_data, Q, R, target_pos, u0_old, 6.0, None)
    J_lqr_end, lqr_dur = compute_performance(lqr_state_data, lqr_u_data, Q, R, target_pos, u0_old, 6.0, None)
    print(f"  J (u0={u0_old}) -> HNN ({hnn_dur:.1f}s): {J_hnn_end:.6f} | LQR ({lqr_dur:.1f}s): {J_lqr_end:.6f} | Imp: {((J_lqr_end - J_hnn_end) / J_lqr_end * 100.0):.2f}%")
    
    J_hnn_end_no, _ = compute_performance(hnn_state_data, hnn_u_data, Q, R, target_pos, None, 6.0, None)
    J_lqr_end_no, _ = compute_performance(lqr_state_data, lqr_u_data, Q, R, target_pos, None, 6.0, None)
    print(f"  J (no u0)     -> HNN ({hnn_dur:.1f}s): {J_hnn_end_no:.6f} | LQR ({lqr_dur:.1f}s): {J_lqr_end_no:.6f} | Imp: {((J_lqr_end_no - J_hnn_end_no) / J_lqr_end_no * 100.0):.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
