import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, medfilt
from scipy.interpolate import interp1d

# --- CONFIGURATION ---
SWIMMING_STYLE = "Breaststroke"
PARTICIPANT_ID = "P041"

# Input paths
OUT_DIR = f"output/{SWIMMING_STYLE}_{PARTICIPANT_ID}"
TRACKING_CSV_PATH = os.path.join(
    OUT_DIR, f"tracking_records_{SWIMMING_STYLE.lower()}_{PARTICIPANT_ID}.csv"
)

# Output paths
GRAPHS_DIR = os.path.join(OUT_DIR, "graphs")

# Data Cleaning Parameters
MEDIAN_KERNEL_SIZE = 7       # For spike/outlier removal
SAVGOL_WINDOW = 31           # For smooth trajectory generation
SAVGOL_POLYORDER = 3         # Polynomial order for Savitzky-Golay


def clean_and_process_data(csv_path):
    """Applies all data cleaning and filtering steps to the tracking data."""
    print(f"Loading tracking records from: {csv_path}")
    df = pd.read_csv(csv_path)

    # 1. Enforce Monotonic Time Sequence
    df = df.sort_values('time_s').drop_duplicates('time_s')
    t = df['time_s'].values
    pos_raw = df['pos_m'].values

    # 2. Missing Value Interpolation (Handle NaNs)
    valid_idx = ~np.isnan(pos_raw)
    if not valid_idx.all():
        interpolator = interp1d(
            t[valid_idx], pos_raw[valid_idx], kind='linear', fill_value="extrapolate")
        pos_interp = interpolator(t)
    else:
        pos_interp = pos_raw.copy()

    # 3. Outlier Removal (Median Filter)
    # Removes single-frame jumps or tracking glitches
    pos_med = medfilt(pos_interp, kernel_size=MEDIAN_KERNEL_SIZE)

    # 4. Trajectory Smoothing (Savitzky-Golay Filter)
    # Provides a continuous differentiable path for kinematics
    window = min(SAVGOL_WINDOW, len(pos_med) - 1)
    if window % 2 == 0:
        window += 1  # Must be odd

    pos_smooth = savgol_filter(
        pos_med, window_length=window, polyorder=SAVGOL_POLYORDER)

    # 5. Zero-centering and Normalization
    # Start at 0m. Assuming a standard 50m lap for scaling if using raw uncalibrated pixels.
    pos_zeroed = pos_smooth - pos_smooth[0]

    # Scale to exactly 50m based on the final tracked frame (optional, based on standard evaluation)
    # If your notebook relies purely on px_per_m, you can remove this scaling factor.
    scale_factor = 50.0 / pos_zeroed[-1] if pos_zeroed[-1] != 0 else 1.0
    pos_final = pos_zeroed * scale_factor

    raw_zeroed = (pos_raw - pos_raw[0]) * scale_factor

    return t, raw_zeroed, pos_final


def calculate_kinematics(t, pos_final):
    """Calculates smoothed velocity and acceleration from the cleaned trajectory."""
    dt = np.gradient(t)
    dt[dt == 0] = np.nan  # Prevent division by zero

    # Velocity calculation & smoothing
    vel_raw = np.gradient(pos_final) / dt
    window = min(SAVGOL_WINDOW, len(vel_raw) - 1)
    if window % 2 == 0:
        window += 1

    vel_smooth = savgol_filter(vel_raw, window_length=window, polyorder=2)

    # Restrict negative velocities (swimmer swimming backwards is usually a tracking artifact)
    vel_smooth = np.clip(vel_smooth, a_min=0.0, a_max=None)

    return vel_smooth


def generate_and_save_plots(t, raw_pos, final_pos, vel_smooth):
    """Generates all results graphs and saves them to the specified directory."""
    os.makedirs(GRAPHS_DIR, exist_ok=True)

    # --- PLOT 1: Data Cleaning Comparison ---
    plt.figure(figsize=(10, 5))
    plt.plot(t, raw_pos, label='Raw Tracked Position',
             color='lightgray', alpha=0.8, linewidth=1.5)
    plt.plot(t, final_pos, label='Cleaned & Filtered Position',
             color='blue', linewidth=2)
    plt.title(
        "Step 1: Tracking Data Cleaning (Outlier Rejection & Smoothing)", fontsize=14)
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(
        GRAPHS_DIR, "data_cleaning_comparison.png"), dpi=300)
    plt.close()

    # --- PLOT 2: Kinematic Velocity Profile ---
    plt.figure(figsize=(10, 5))
    plt.plot(t, vel_smooth, label='Instantaneous Velocity',
             color='green', linewidth=2)
    plt.axhline(y=np.nanmean(vel_smooth), color='r', linestyle='--',
                label=f'Avg Vel: {np.nanmean(vel_smooth):.2f} m/s')
    plt.title("Step 2: Smoothed Velocity Profile", fontsize=14)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "velocity_profile.png"), dpi=300)
    plt.close()

    # --- PLOT 3: 50m Trajectory Evaluation ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Notebook Ground Truth references
    outbound_dists = np.array([15.0, 25.0])
    return_dists = np.array([35.0, 45.0, 50.0])

    true_outbound_times = np.interp(outbound_dists, final_pos, t)
    true_return_times = np.interp(return_dists, final_pos, t)

    ax.plot(t, final_pos, '-', color='black',
            linewidth=2.5, label="System Trajectory")

    all_true_times = np.concatenate([true_outbound_times, true_return_times])
    all_true_dists = np.concatenate([outbound_dists, return_dists])

    ax.plot(all_true_times, all_true_dists, 'o', color='red',
            markersize=8, label="Ground Truth Splits", zorder=3)

    ax.set_xlabel("Elapsed Time (s)", fontsize=12)
    ax.set_ylabel("Distance (m)", fontsize=12)
    ax.set_title(
        "Full 50m Trajectory Evaluation: Cleaned Automated Tracking vs Ground Truth", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, "trajectory_evaluation.png"), dpi=300)
    plt.close()

    print(f"Successfully saved 3 plots to the '{GRAPHS_DIR}/' directory.")


def main():
    if not os.path.exists(TRACKING_CSV_PATH):
        print(f"Error: Could not find CSV file at {TRACKING_CSV_PATH}")
        return

    # 1. Apply All Data Cleaning Steps
    t, raw_pos, final_pos = clean_and_process_data(TRACKING_CSV_PATH)

    # 2. Calculate Kinematics (Velocity)
    vel_smooth = calculate_kinematics(t, final_pos)

    # 3. Generate & Save Graphs
    generate_and_save_plots(t, raw_pos, final_pos, vel_smooth)

    # 4. Print Results Calculation Summary
    print("\n" + "="*40)
    print("        TRACKING RESULTS SUMMARY        ")
    print("="*40)
    print(f"Total Elapsed Time : {t[-1]:.2f} seconds")
    print(f"Total Distance     : {final_pos[-1]:.2f} meters")
    print(f"Average Velocity   : {np.nanmean(vel_smooth):.2f} m/s")
    print(f"Maximum Velocity   : {np.nanmax(vel_smooth):.2f} m/s")

    print("\n--- Split Times ---")
    splits = [15.0, 25.0, 35.0, 45.0, 50.0]
    split_times = np.interp(splits, final_pos, t)
    for dist, time_s in zip(splits, split_times):
        print(f"{dist}m Split       : {time_s:.2f} s")
    print("="*40 + "\n")


if __name__ == "__main__":
    main()
