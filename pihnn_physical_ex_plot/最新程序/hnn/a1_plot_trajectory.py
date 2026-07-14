import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set style for professional look
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def main():
    # Locate state data files in the script directory
    script_dir = Path(__file__).resolve().parent
    txt_files = sorted(list(script_dir.glob("state_data_*.txt")))
    
    if not txt_files:
        print(f"No state_data_*.txt files found in {script_dir}")
        return
    
    # Pick the first one
    data_file = txt_files[8]
    print(f"Loading data from: {data_file.name}")
    
    # Load data, skipping the header line
    # The columns are: t, x, y, z, vx, vy, vz, phi, theta, psi
    try:
        data = np.genfromtxt(data_file, delimiter="\t", skip_header=1, encoding="utf-8-sig")
    except Exception as e:
        print(f"Error loading {data_file} with utf-8-sig encoding: {e}")
        data = np.genfromtxt(data_file, delimiter="\t", skip_header=1)
        
    # Clean nan or incomplete lines
    data = data[np.all(np.isfinite(data), axis=1)]
    
    t = data[:, 0]
    # Subtract start time to get relative time
    t_rel = t - t[0]
    
    x, y, z = data[:, 1], data[:, 2], data[:, 3]
    vx, vy, vz = data[:, 4], data[:, 5], data[:, 6]
    phi, theta, psi = data[:, 7], data[:, 8], data[:, 9]
    
    # --- Figure 1: 3D Trajectory ---
    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    # Plot path
    sc = ax_3d.scatter(x, y, z, c=t_rel, cmap='viridis', s=10, alpha=0.8, label='Trajectory')
    ax_3d.plot(x, y, z, color='gray', alpha=0.5, linestyle='--')
    
    # Highlight start and end
    ax_3d.scatter(x[0], y[0], z[0], color='red', s=100, marker='o', edgecolors='black', label='Start', zorder=5)
    ax_3d.scatter(x[-1], y[-1], z[-1], color='green', s=100, marker='X', edgecolors='black', label='End', zorder=5)
    
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Z (m)')
    ax_3d.set_title(f'3D Trajectory - {data_file.name}', fontsize=14, pad=20)
    
    # Colorbar for time
    cbar = fig_3d.colorbar(sc, ax=ax_3d, shrink=0.6, pad=0.1)
    cbar.set_label('Time (s)')
    
    ax_3d.legend()
    plt.tight_layout()
    
    # Save 3D trajectory
    fig_3d_path = script_dir / "trajectory_3d.png"
    plt.savefig(fig_3d_path, dpi=300, bbox_inches='tight')
    print(f"Saved 3D trajectory plot to: {fig_3d_path}")
    
    # --- Figure 2: States vs Time ---
    fig_states, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot positions
    axs[0].plot(t_rel, x, label='x', color='#1f77b4', linewidth=1.5)
    axs[0].plot(t_rel, y, label='y', color='#ff7f0e', linewidth=1.5)
    axs[0].plot(t_rel, z, label='z', color='#2ca02c', linewidth=1.5)
    axs[0].set_ylabel('Position (m)', fontsize=12)
    axs[0].set_title('States vs. Time', fontsize=14)
    axs[0].legend(loc='upper right')
    axs[0].grid(True, linestyle=':', alpha=0.6)
    
    # Plot velocities
    axs[1].plot(t_rel, vx, label='$v_x$', color='#d62728', linewidth=1.5)
    axs[1].plot(t_rel, vy, label='$v_y$', color='#9467bd', linewidth=1.5)
    axs[1].plot(t_rel, vz, label='$v_z$', color='#8c564b', linewidth=1.5)
    axs[1].set_ylabel('Velocity (m/s)', fontsize=12)
    axs[1].legend(loc='upper right')
    axs[1].grid(True, linestyle=':', alpha=0.6)
    
    # Plot Euler angles
    axs[2].plot(t_rel, phi, label='$\\phi$ (roll)', color='#e377c2', linewidth=1.5)
    axs[2].plot(t_rel, theta, label='$\\theta$ (pitch)', color='#7f7f7f', linewidth=1.5)
    axs[2].plot(t_rel, psi, label='$\\psi$ (yaw)', color='#bcbd22', linewidth=1.5)
    axs[2].set_xlabel('Time (s)', fontsize=12)
    axs[2].set_ylabel('Attitude (rad)', fontsize=12)
    axs[2].legend(loc='upper right')
    axs[2].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    # Save states vs time
    fig_states_path = script_dir / "state_vs_time.png"
    plt.savefig(fig_states_path, dpi=300, bbox_inches='tight')
    print(f"Saved state vs time plot to: {fig_states_path}")
    
    # --- Figure 3: Control Inputs vs Time ---
    timestamp_suffix = data_file.name.replace("state_data_", "").replace(".txt", "")
    u_file = script_dir / f"u_data_{timestamp_suffix}.txt"
    if u_file.exists():
        print(f"Loading control data from: {u_file.name}")
        try:
            u_data = np.genfromtxt(u_file, delimiter="\t", skip_header=1, encoding="utf-8-sig")
        except Exception as e:
            print(f"Error loading {u_file} with utf-8-sig encoding: {e}")
            u_data = np.genfromtxt(u_file, delimiter="\t", skip_header=1)
            
        u_data = u_data[np.all(np.isfinite(u_data), axis=1)]
        t_u = u_data[:, 0]
        t_u_rel = t_u - t[0]  # Aligned with state time
        
        u1, u2, u3, u4 = u_data[:, 1], u_data[:, 2], u_data[:, 3], u_data[:, 4]
        
        fig_control, ax_ctrl = plt.subplots(figsize=(10, 4))
        ax_ctrl.plot(t_u_rel, u1, label='$u_1$', color='#1f77b4', linewidth=1.5)
        ax_ctrl.plot(t_u_rel, u2, label='$u_2$', color='#ff7f0e', linewidth=1.5)
        ax_ctrl.plot(t_u_rel, u3, label='$u_3$', color='#2ca02c', linewidth=1.5)
        ax_ctrl.plot(t_u_rel, u4, label='$u_4$', color='#d62728', linewidth=1.5)
        ax_ctrl.set_xlabel('Time (s)', fontsize=12)
        ax_ctrl.set_ylabel('Control Inputs', fontsize=12)
        ax_ctrl.set_title(f'Control Inputs vs. Time - {u_file.name}', fontsize=14)
        ax_ctrl.legend(loc='upper right')
        ax_ctrl.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        fig_ctrl_path = script_dir / "control_vs_time.png"
        plt.savefig(fig_ctrl_path, dpi=300, bbox_inches='tight')
        print(f"Saved control vs time plot to: {fig_ctrl_path}")
    else:
        print(f"No matching control file found: {u_file.name}")
        
    # Show all figures
    plt.show()

if __name__ == "__main__":
    main()
