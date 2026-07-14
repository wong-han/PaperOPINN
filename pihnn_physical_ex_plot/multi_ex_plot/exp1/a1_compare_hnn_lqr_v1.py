import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set style for professional look
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

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

def main():
    script_dir = Path(__file__).resolve().parent
    
    # Define directories
    hnn_dir = script_dir / "hnn"
    lqr_dir = script_dir / "lqr"
    images_dir = script_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Locate HNN state data files
    hnn_files = sorted(list(hnn_dir.glob("state_data_*.txt")))
    if not hnn_files:
        print(f"No state_data_*.txt files found in {hnn_dir}")
        return
    # Use index 8 as per HNN's a1_plot_trajectory.py
    hnn_state_file = hnn_files[5] if len(hnn_files) > 8 else hnn_files[-1]
    
    # Locate LQR state data files
    lqr_files = sorted(list(lqr_dir.glob("state_data_*.txt")))
    if not lqr_files:
        print(f"No state_data_*.txt files found in {lqr_dir}")
        return
    # Use index 6 as per LQR's a1_plot_trajectory.py
    lqr_state_file = lqr_files[5] if len(lqr_files) > 6 else lqr_files[-1]
    
    # Load state data
    hnn_state_data = load_data(hnn_state_file)
    lqr_state_data = load_data(lqr_state_file)
    
    # Process HNN states
    hnn_t = hnn_state_data[:, 0]
    hnn_t_rel = hnn_t - hnn_t[0]
    hnn_x, hnn_y, hnn_z = hnn_state_data[:, 1], hnn_state_data[:, 2], hnn_state_data[:, 3]
    hnn_vx, hnn_vy, hnn_vz = hnn_state_data[:, 4], hnn_state_data[:, 5], hnn_state_data[:, 6]
    hnn_phi, hnn_theta, hnn_psi = hnn_state_data[:, 7], hnn_state_data[:, 8], hnn_state_data[:, 9]
    
    # Process LQR states
    lqr_t = lqr_state_data[:, 0]
    lqr_t_rel = lqr_t - lqr_t[0]
    lqr_x, lqr_y, lqr_z = lqr_state_data[:, 1], lqr_state_data[:, 2], lqr_state_data[:, 3]
    lqr_vx, lqr_vy, lqr_vz = lqr_state_data[:, 4], lqr_state_data[:, 5], lqr_state_data[:, 6]
    lqr_phi, lqr_theta, lqr_psi = lqr_state_data[:, 7], lqr_state_data[:, 8], lqr_state_data[:, 9]
    
    # Color definition (Ours = Red, LQR = Green)
    color_hnn = '#d62728' # Red
    color_lqr = '#2ca02c' # Green
    
    # --- Figure 1: 3D Trajectory ---
    print("Generating 3D Trajectory comparison...")
    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    # Plot trajectories
    ax_3d.plot(hnn_x, hnn_y, hnn_z, color=color_hnn, linestyle='-', linewidth=2.0, label='HNN (Ours)')
    ax_3d.plot(lqr_x, lqr_y, lqr_z, color=color_lqr, linestyle='--', linewidth=2.0, label='LQR')
    
    # Highlight starts and ends
    # HNN
    ax_3d.scatter(hnn_x[0], hnn_y[0], hnn_z[0], color=color_hnn, s=100, marker='o', edgecolors='black', label='HNN Start', zorder=5)
    ax_3d.scatter(hnn_x[-1], hnn_y[-1], hnn_z[-1], color=color_hnn, s=100, marker='X', edgecolors='black', label='HNN End', zorder=5)
    # LQR
    ax_3d.scatter(lqr_x[0], lqr_y[0], lqr_z[0], color=color_lqr, s=100, marker='o', edgecolors='black', label='LQR Start', zorder=5)
    ax_3d.scatter(lqr_x[-1], lqr_y[-1], lqr_z[-1], color=color_lqr, s=100, marker='X', edgecolors='black', label='LQR End', zorder=5)
    
    # Target state
    ax_3d.scatter(0.55, 0.0, 0.9, color='blue', s=150, marker='*', edgecolors='black', label='Target (0.55, 0.0, 0.9)', zorder=6)
    
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Z (m)')
    ax_3d.set_title('3D Trajectory Comparison: HNN (Ours) vs. LQR', fontsize=14, pad=20)
    ax_3d.legend()
    plt.tight_layout()
    
    fig_3d_path = images_dir / "compare_trajectory_3d.png"
    plt.savefig(fig_3d_path, dpi=300, bbox_inches='tight')
    print(f"Saved 3D trajectory comparison to: {fig_3d_path}")
    
    # --- Figure 2: States Comparison (9 subplots) ---
    print("Generating state comparison plot...")
    fig_states, axs = plt.subplots(3, 3, figsize=(16, 12), sharex='col')
    
    # Row 1: Positions (x, y, z)
    # X
    axs[0, 0].plot(hnn_t_rel, hnn_x, color=color_hnn, linestyle='-', linewidth=2.0, label='HNN (Ours)')
    axs[0, 0].plot(lqr_t_rel, lqr_x, color=color_lqr, linestyle='--', linewidth=2.0, label='LQR')
    axs[0, 0].axhline(y=0.55, color='black', linestyle=':', alpha=0.7, label='Target')
    axs[0, 0].set_ylabel('Position X (m)', fontsize=12)
    axs[0, 0].set_title('Position X', fontsize=12)
    axs[0, 0].grid(True, linestyle=':', alpha=0.6)
    axs[0, 0].legend()
    # Y
    axs[0, 1].plot(hnn_t_rel, hnn_y, color=color_hnn, linestyle='-', linewidth=2.0)
    axs[0, 1].plot(lqr_t_rel, lqr_y, color=color_lqr, linestyle='--', linewidth=2.0)
    axs[0, 1].axhline(y=0.0, color='black', linestyle=':', alpha=0.7)
    axs[0, 1].set_ylabel('Position Y (m)', fontsize=12)
    axs[0, 1].set_title('Position Y', fontsize=12)
    axs[0, 1].grid(True, linestyle=':', alpha=0.6)
    # Z
    axs[0, 2].plot(hnn_t_rel, hnn_z, color=color_hnn, linestyle='-', linewidth=2.0)
    axs[0, 2].plot(lqr_t_rel, lqr_z, color=color_lqr, linestyle='--', linewidth=2.0)
    axs[0, 2].axhline(y=0.9, color='black', linestyle=':', alpha=0.7)
    axs[0, 2].set_ylabel('Position Z (m)', fontsize=12)
    axs[0, 2].set_title('Position Z', fontsize=12)
    axs[0, 2].grid(True, linestyle=':', alpha=0.6)
    
    # Row 2: Velocities (vx, vy, vz)
    # Vx
    axs[1, 0].plot(hnn_t_rel, hnn_vx, color=color_hnn, linestyle='-', linewidth=2.0)
    axs[1, 0].plot(lqr_t_rel, lqr_vx, color=color_lqr, linestyle='--', linewidth=2.0)
    axs[1, 0].set_ylabel('Velocity Vx (m/s)', fontsize=12)
    axs[1, 0].set_title('Velocity X', fontsize=12)
    axs[1, 0].grid(True, linestyle=':', alpha=0.6)
    # Vy
    axs[1, 1].plot(hnn_t_rel, hnn_vy, color=color_hnn, linestyle='-', linewidth=2.0)
    axs[1, 1].plot(lqr_t_rel, lqr_vy, color=color_lqr, linestyle='--', linewidth=2.0)
    axs[1, 1].set_ylabel('Velocity Vy (m/s)', fontsize=12)
    axs[1, 1].set_title('Velocity Y', fontsize=12)
    axs[1, 1].grid(True, linestyle=':', alpha=0.6)
    # Vz
    axs[1, 2].plot(hnn_t_rel, hnn_vz, color=color_hnn, linestyle='-', linewidth=2.0)
    axs[1, 2].plot(lqr_t_rel, lqr_vz, color=color_lqr, linestyle='--', linewidth=2.0)
    axs[1, 2].set_ylabel('Velocity Vz (m/s)', fontsize=12)
    axs[1, 2].set_title('Velocity Z', fontsize=12)
    axs[1, 2].grid(True, linestyle=':', alpha=0.6)
    
    # Row 3: Attitudes (phi, theta, psi)
    # Roll phi
    axs[2, 0].plot(hnn_t_rel, hnn_phi, color=color_hnn, linestyle='-', linewidth=2.0)
    axs[2, 0].plot(lqr_t_rel, lqr_phi, color=color_lqr, linestyle='--', linewidth=2.0)
    axs[2, 0].set_xlabel('Time (s)', fontsize=12)
    axs[2, 0].set_ylabel('Roll $\phi$ (rad)', fontsize=12)
    axs[2, 0].set_title('Attitude Roll', fontsize=12)
    axs[2, 0].grid(True, linestyle=':', alpha=0.6)
    # Pitch theta
    axs[2, 1].plot(hnn_t_rel, hnn_theta, color=color_hnn, linestyle='-', linewidth=2.0)
    axs[2, 1].plot(lqr_t_rel, lqr_theta, color=color_lqr, linestyle='--', linewidth=2.0)
    axs[2, 1].set_xlabel('Time (s)', fontsize=12)
    axs[2, 1].set_ylabel('Pitch $\\theta$ (rad)', fontsize=12)
    axs[2, 1].set_title('Attitude Pitch', fontsize=12)
    axs[2, 1].grid(True, linestyle=':', alpha=0.6)
    # Yaw psi
    axs[2, 2].plot(hnn_t_rel, hnn_psi, color=color_hnn, linestyle='-', linewidth=2.0)
    axs[2, 2].plot(lqr_t_rel, lqr_psi, color=color_lqr, linestyle='--', linewidth=2.0)
    axs[2, 2].set_xlabel('Time (s)', fontsize=12)
    axs[2, 2].set_ylabel('Yaw $\\psi$ (rad)', fontsize=12)
    axs[2, 2].set_title('Attitude Yaw', fontsize=12)
    axs[2, 2].grid(True, linestyle=':', alpha=0.6)
    
    fig_states.suptitle('State Comparison: HNN (Ours) vs. LQR', fontsize=16, y=0.98)
    plt.tight_layout()
    
    fig_states_path = images_dir / "compare_states.png"
    plt.savefig(fig_states_path, dpi=300, bbox_inches='tight')
    print(f"Saved states comparison to: {fig_states_path}")
    
    # --- Figure 3: Control Inputs Comparison ---
    # Find matching control data files
    hnn_suffix = hnn_state_file.name.replace("state_data_", "").replace(".txt", "")
    hnn_u_file = hnn_dir / f"u_data_{hnn_suffix}.txt"
    
    lqr_suffix = lqr_state_file.name.replace("state_data_", "").replace(".txt", "")
    lqr_u_file = lqr_dir / f"u_data_{lqr_suffix}.txt"
    
    if hnn_u_file.exists() and lqr_u_file.exists():
        print("Generating control inputs comparison plot...")
        hnn_u_data = load_data(hnn_u_file)
        lqr_u_data = load_data(lqr_u_file)
        
        # HNN control inputs
        hnn_tu = hnn_u_data[:, 0]
        hnn_tu_rel = hnn_tu - hnn_t[0] # Align time with state start
        hnn_u1, hnn_u2, hnn_u3, hnn_u4 = hnn_u_data[:, 1], hnn_u_data[:, 2], hnn_u_data[:, 3], hnn_u_data[:, 4]
        
        # LQR control inputs
        lqr_tu = lqr_u_data[:, 0]
        lqr_tu_rel = lqr_tu - lqr_t[0] # Align time with state start
        lqr_u1, lqr_u2, lqr_u3, lqr_u4 = lqr_u_data[:, 1], lqr_u_data[:, 2], lqr_u_data[:, 3], lqr_u_data[:, 4]
        
        fig_control, axs_ctrl = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
        
        # u1 (Roll Rate Control)
        axs_ctrl[0, 0].plot(hnn_tu_rel, hnn_u1, color=color_hnn, linestyle='-', linewidth=2.0, label='HNN (Ours)')
        axs_ctrl[0, 0].plot(lqr_tu_rel, lqr_u1, color=color_lqr, linestyle='--', linewidth=2.0, label='LQR')
        axs_ctrl[0, 0].set_ylabel('Control $u_1$ (Roll)', fontsize=12)
        axs_ctrl[0, 0].set_title('Control Input $u_1$', fontsize=12)
        axs_ctrl[0, 0].grid(True, linestyle=':', alpha=0.6)
        axs_ctrl[0, 0].legend()
        
        # u2 (Pitch Rate Control)
        axs_ctrl[0, 1].plot(hnn_tu_rel, hnn_u2, color=color_hnn, linestyle='-', linewidth=2.0)
        axs_ctrl[0, 1].plot(lqr_tu_rel, lqr_u2, color=color_lqr, linestyle='--', linewidth=2.0)
        axs_ctrl[0, 1].set_ylabel('Control $u_2$ (Pitch)', fontsize=12)
        axs_ctrl[0, 1].set_title('Control Input $u_2$', fontsize=12)
        axs_ctrl[0, 1].grid(True, linestyle=':', alpha=0.6)
        
        # u3 (Yaw Rate Control)
        axs_ctrl[1, 0].plot(hnn_tu_rel, hnn_u3, color=color_hnn, linestyle='-', linewidth=2.0)
        axs_ctrl[1, 0].plot(lqr_tu_rel, lqr_u3, color=color_lqr, linestyle='--', linewidth=2.0)
        axs_ctrl[1, 0].set_xlabel('Time (s)', fontsize=12)
        axs_ctrl[1, 0].set_ylabel('Control $u_3$ (Yaw)', fontsize=12)
        axs_ctrl[1, 0].set_title('Control Input $u_3$', fontsize=12)
        axs_ctrl[1, 0].grid(True, linestyle=':', alpha=0.6)
        
        # u4 (Thrust Control)
        axs_ctrl[1, 1].plot(hnn_tu_rel, hnn_u4, color=color_hnn, linestyle='-', linewidth=2.0)
        axs_ctrl[1, 1].plot(lqr_tu_rel, lqr_u4, color=color_lqr, linestyle='--', linewidth=2.0)
        axs_ctrl[1, 1].set_xlabel('Time (s)', fontsize=12)
        axs_ctrl[1, 1].set_ylabel('Control $u_4$ (Thrust)', fontsize=12)
        axs_ctrl[1, 1].set_title('Control Input $u_4$', fontsize=12)
        axs_ctrl[1, 1].grid(True, linestyle=':', alpha=0.6)
        
        fig_control.suptitle('Control Inputs Comparison: HNN (Ours) vs. LQR', fontsize=16, y=0.98)
        plt.tight_layout()
        
        fig_ctrl_path = images_dir / "compare_controls.png"
        plt.savefig(fig_ctrl_path, dpi=300, bbox_inches='tight')
        print(f"Saved control inputs comparison to: {fig_ctrl_path}")
    else:
        print("Warning: Control files could not be found, skipping control comparison plot.")
        
    plt.show()

if __name__ == "__main__":
    main()
