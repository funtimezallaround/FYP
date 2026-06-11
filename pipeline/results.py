"""
results.py – Post-processing and visualisation for swimmer tracking CSV.

Reproduces every plot from the analysis notebook using the CSV saved by main.py.

Usage:
    python results.py <tracking_csv_path>
    python results.py                         # falls back to DEFAULT_CSV below
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter, medfilt


# --- CONFIGURATION ---
SWIMMING_STYLE = "Breaststroke"
PARTICIPANT_ID = "P035"

# Input paths
OUT_DIR = f"output/{SWIMMING_STYLE}_{PARTICIPANT_ID}"
DEFAULT_CSV = os.path.join(
    OUT_DIR, f"tracking_records_{SWIMMING_STYLE.lower()}_{PARTICIPANT_ID}.csv"
)

# Output paths
GRAPHS_DIR = os.path.join(OUT_DIR, "graphs")


MARKER_REAL_DIST_M = 2.5    # physical gap between lane-rope markers (m)
POOL_LENGTH_M = 25.0        # distance from wall to turn (m)
VEL_MAX = 5.0               # velocity outlier ceiling (m/s)
# drop data before this timestamp (s); set e.g. 17.0
START_TIME_S = 0.0

# to match the notebook's time_s >= 17.0 filter
MAX_ARM_REACH_CM = 150    # sanity cap on wrist offset from shoulder (cm)
WRIST_SMOOTH_WIN = 5      # rolling-median window for wrist spatial plots

# Kalman hyper-parameters (verbatim from notebook)
KF_ACCEL_VAR = 3.0
KF_MEAS_VAR = 0.015
KFS_PROC_VAR = 0.20
KFS_MEAS_VAR = 9.0

# ── Ground-truth 50 m evaluation (Plot 8) ─────────────────────────────────────
# Set GT_ENABLED = True and fill in your own timestamps to activate this plot.
# Times are absolute video timestamps (seconds); START_TIME_VIDEO_S is the
# video time that corresponds to "0 s elapsed" in the trajectory.
GT_ENABLED = False
GT_START_VIDEO_S = 17.0
GT_OUTBOUND_TIMES_S = np.array(
    [19, 21, 24, 26, 29, 31, 34, 37, 39, 43], dtype=float)
GT_RETURN_TIMES_S = np.array(
    [44, 46, 49, 52, 55, 58, 60, 63, 66, 68], dtype=float)

# ══════════════════════════════════════════════════════════════════════════════


# ── noise-scale map (only "auto" mode used in main.py) ───────────────────────
_MODE_NOISE_SCALE = {"auto": 1.0}


# ─────────────────────────────────────────────────────────────────────────────
#  Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _nominal_fps(times: np.ndarray) -> float:
    """Estimate frame rate from median inter-frame interval."""
    dts = np.diff(times)
    dts = dts[dts > 0]
    return 1.0 / float(np.median(dts)) if len(dts) > 0 else 60.0


def _mad_inlier_mask(x: np.ndarray, z_thresh: float = 3.5) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad < 1e-12:
        return np.ones_like(x, dtype=bool)
    return np.abs(0.6745 * (x - med) / mad) <= z_thresh


def _fill_nan_interp(arr: np.ndarray) -> np.ndarray:
    """Linearly interpolate across NaN gaps."""
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    inds = np.arange(n)
    mask = np.isfinite(arr)
    if mask.sum() == 0:
        return np.full(n, np.nan)
    if mask.sum() == 1:
        return np.where(mask, arr, np.nan)
    return np.interp(inds, inds[mask], arr[mask])


def _save(fig: plt.Figure, out_dir: str, name: str) -> None:
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  → saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Kalman filters  (verbatim from notebook)
# ─────────────────────────────────────────────────────────────────────────────

def kalman_filter_pos_vel(
    times_s:   np.ndarray,
    pos_meas:  np.ndarray,
    mode_names: list,
    accel_var: float = KF_ACCEL_VAR,
    meas_var:  float = KF_MEAS_VAR,
) -> tuple[np.ndarray, np.ndarray]:
    fps = _nominal_fps(times_s)
    n = len(pos_meas)
    pos_est = np.zeros(n, dtype=float)
    vel_est = np.zeros(n, dtype=float)
    if n == 0:
        return pos_est, vel_est

    x = np.array([pos_meas[0], 0.0], dtype=float)
    P = np.diag([0.10 ** 2, 1.00 ** 2]).astype(float)
    pos_est[0], vel_est[0] = x

    for i in range(1, n):
        dt = max(times_s[i] - times_s[i - 1], 1.0 / fps)
        F = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
        Q = accel_var * np.array(
            [[dt ** 4 / 4.0, dt ** 3 / 2.0],
             [dt ** 3 / 2.0, dt ** 2]], dtype=float)
        x = F @ x
        P = F @ P @ F.T + Q

        H = np.array([[1.0, 0.0]], dtype=float)
        R = np.array([[meas_var * _MODE_NOISE_SCALE.get(mode_names[i], 4.0)]])
        z = np.array([[pos_meas[i]]])
        y = z - H @ x.reshape(-1, 1)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + (K @ y).ravel()
        P = (np.eye(2) - K @ H) @ P
        pos_est[i], vel_est[i] = x

    return pos_est, vel_est


def kalman_filter_scale(
    scale_meas:  np.ndarray,
    mode_names:  list = None,
    sg_window:   int = 31,
    p_order:     int = 3,
) -> np.ndarray:
    """
    Offline scale extraction: interpolates missing/zero values and applies 
    non-causal Savitzky-Golay smoothing instead of carrying forward the state.
    """
    n = len(scale_meas)
    if n == 0:
        return np.zeros(0, dtype=float)

    s = np.copy(scale_meas)

    # Treat 0 or negative values as invalid/missing so they can be interpolated
    s[s <= 0] = np.nan

    # Bidirectional linear interpolation for any gaps
    s = pd.Series(s).interpolate(method="linear",
                                 limit_direction="both").to_numpy()

    # Apply non-causal Savitzky-Golay smoothing over the sequence
    w = min(sg_window, n if n % 2 != 0 else n - 1)
    w = max(w, 3)
    if n >= w:
        s = savgol_filter(s, w, min(p_order, w - 1))

    return s


# ─────────────────────────────────────────────────────────────────────────────
#  Wrist velocity cleaner  (verbatim from notebook)
# ─────────────────────────────────────────────────────────────────────────────

def clean_wrist_velocity(
    vel_raw:      np.ndarray,
    speed_thresh: float = 15.0,
    med_kernel:   int = 5,
    sg_window:    int = 15,
    p_order:      int = 3,
) -> np.ndarray:
    v = np.copy(vel_raw)
    v[np.abs(v) > speed_thresh] = np.nan
    v = pd.Series(v).interpolate(method="linear",
                                 limit_direction="both").to_numpy()
    v = medfilt(v, kernel_size=med_kernel)
    w = min(sg_window, len(v) if len(v) % 2 != 0 else len(v) - 1)
    w = max(w, 3)
    return savgol_filter(v, w, min(p_order, w - 1))


# ─────────────────────────────────────────────────────────────────────────────
#  Wrist pixel→cm conversion helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wrist_cm(df: pd.DataFrame,
              wx: str, wy: str,
              ax: str, ay: str) -> tuple[pd.Series, pd.Series]:
    """Convert wrist pixel offsets from shoulder to centimetres."""
    rel_x = ((df[wx] - df[ax]) / df["px_per_m"]) * 100
    rel_y = ((df[wy] - df[ay]) / df["px_per_m"]) * 100
    return rel_x, rel_y


def _apply_wrist_cm(df: pd.DataFrame,
                    lshoulder_col: str = "lshoulder",
                    rshoulder_col: str = "rshoulder") -> pd.DataFrame:
    """Add lwrist/rwrist relative-cm columns anchored to their own shoulders."""
    df = df.copy()
    df["lwrist_rel_x_cm"], df["lwrist_rel_y_cm"] = _wrist_cm(
        df, "lwrist_x", "lwrist_y", f"{lshoulder_col}_x", f"{lshoulder_col}_y")
    df["rwrist_rel_x_cm"], df["rwrist_rel_y_cm"] = _wrist_cm(
        df, "rwrist_x", "rwrist_y", f"{rshoulder_col}_x", f"{rshoulder_col}_y")
    return df


def _filter_and_smooth_wrist(df: pd.DataFrame,
                             window: int = WRIST_SMOOTH_WIN) -> pd.DataFrame:
    rel_cols = ["lwrist_rel_x_cm", "lwrist_rel_y_cm",
                "rwrist_rel_x_cm", "rwrist_rel_y_cm"]
    df = df[
        (df["lwrist_rel_x_cm"].abs() < MAX_ARM_REACH_CM) &
        (df["lwrist_rel_y_cm"].abs() < MAX_ARM_REACH_CM) &
        (df["rwrist_rel_x_cm"].abs() < MAX_ARM_REACH_CM) &
        (df["rwrist_rel_y_cm"].abs() < MAX_ARM_REACH_CM)
    ].copy()
    for col in rel_cols:
        df[col] = df[col].rolling(window=window, center=True).median()
    return df.dropna(subset=rel_cols)


# ─────────────────────────────────────────────────────────────────────────────
#  4-panel KF overview (reused for Plot 1 and Plot 2)
# ─────────────────────────────────────────────────────────────────────────────

def _plot_kf_overview(times, pos_raw, pos_kf, vel_kf,
                      scales_raw, scale_kf,
                      title_pos_suffix: str = "") -> plt.Figure:
    spd_kf = np.abs(vel_kf)
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

    axes[0].plot(times, pos_raw, ".", ms=2, color="deepskyblue",
                 alpha=0.60, label="Raw position")
    axes[0].plot(times, pos_kf,  "-", lw=1.8,
                 color="black",       label="Kalman position")
    axes[0].axhline(0,                  color="forestgreen", lw=0.7, ls=":")
    axes[0].axhline(MARKER_REAL_DIST_M, color="forestgreen", lw=0.7, ls=":",
                    label=f"Marker gap ({MARKER_REAL_DIST_M} m)")
    axes[0].set_ylabel("Position (m)")
    axes[0].set_title(f"Swimmer position along lane rope{title_pos_suffix}")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.6)

    axes[1].plot(times, scales_raw, ".", ms=2,
                 color="forestgreen", alpha=0.60, label="Raw scale")
    axes[1].plot(times, scale_kf,   "-", lw=1.6,
                 color="black",       label="Kalman scale")
    axes[1].axhline(np.mean(scale_kf), color="black", lw=1.0, ls="--", alpha=0.7,
                    label=f"KF mean {np.mean(scale_kf):.1f} px/m")
    axes[1].set_ylabel("px / m")
    axes[1].set_title("Per-frame pixel-to-metre scale")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.6)

    axes[2].plot(times, vel_kf, "-", lw=1.5, color="orange",
                 label="Kalman velocity (m/s)")
    axes[2].axhline(0, color="black", lw=0.8, ls=":")
    axes[2].set_ylabel("Velocity (m/s)")
    axes[2].set_title(
        "Swimmer velocity (+ = toward right marker,  − = toward left marker)")
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.6)

    axes[3].fill_between(times, spd_kf, alpha=0.25, color="deepskyblue")
    axes[3].plot(times, spd_kf, "-", lw=1.8, color="deepskyblue",
                 label="Kalman |speed| (m/s)")
    axes[3].set_ylabel("Speed (m/s)")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_title("Swimmer Speed")
    axes[3].legend(fontsize=9)
    axes[3].grid(alpha=0.6)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main(csv_path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Load CSV ───────────────────────────────────────────────────────────
    print(f"\nLoading: {csv_path}")
    df_all = pd.read_csv(csv_path)
    print(f"  Total rows: {len(df_all)}")

    if START_TIME_S > 0.0:
        df_all = df_all[df_all["time_s"] >=
                        START_TIME_S].copy().reset_index(drop=True)
        print(f"  After time trim (>= {START_TIME_S}s): {len(df_all)} rows")

    # ── 2. MAD outlier removal ────────────────────────────────────────────────
    df_valid = df_all.dropna(subset=["pos_m"]).copy().reset_index(drop=True)
    print(f"  Rows with valid pos_m: {len(df_valid)}")

    if len(df_valid) >= 5:
        pos_vals = df_valid["pos_m"].to_numpy(dtype=float)
        scale_vals = df_valid["px_per_m"].to_numpy(dtype=float)

        pos_mask = _mad_inlier_mask(pos_vals)
        scale_mask = _mad_inlier_mask(scale_vals)

        dpos = np.r_[0.0, np.abs(np.diff(pos_vals))]
        jump_thr = np.median(dpos) + 4.0 * np.std(dpos)
        jump_mask = (dpos <= jump_thr
                     if np.isfinite(jump_thr) and jump_thr > 0
                     else np.ones(len(dpos), dtype=bool))

        keep_mask = pos_mask & scale_mask & jump_mask
        removed = int((~keep_mask).sum())
        df_valid = df_valid[keep_mask].reset_index(drop=True)
        print(f"  MAD removal: removed {removed}, kept {len(df_valid)}")

    times = df_valid["time_s"].to_numpy(dtype=float)
    pos_raw = df_valid["pos_m"].to_numpy(dtype=float)
    scales_raw = df_valid["px_per_m"].to_numpy(dtype=float)
    mode_seq = df_valid["selection_mode"].tolist()

    # ── 3. First Kalman pass ──────────────────────────────────────────────────
    print("\n── Kalman pass 1 (raw data) ──")
    pos_kf, vel_kf = kalman_filter_pos_vel(times, pos_raw, mode_seq)
    scale_kf = kalman_filter_scale(scales_raw)
    spd_kf = np.abs(vel_kf)
    print(
        f"  pos std raw/KF   : {np.std(pos_raw):.4f} / {np.std(pos_kf):.4f} m")
    print(
        f"  scale std raw/KF : {np.std(scales_raw):.4f} / {np.std(scale_kf):.4f} px/m")
    print(
        f"  peak |speed|     : {spd_kf.max():.4f} m/s  ({spd_kf.max()*3.6:.3f} km/h)")
    print(
        f"  mean |speed|     : {spd_kf.mean():.4f} m/s  ({spd_kf.mean()*3.6:.3f} km/h)")

    # ── Plot 1: raw KF overview ───────────────────────────────────────────────
    print("\nPlot 1  –  Kalman overview (pre-velocity filter)")
    with plt.style.context("default"):
        fig = _plot_kf_overview(times, pos_raw, pos_kf, vel_kf,
                                scales_raw, scale_kf)
        _save(fig, out_dir, "plot1_kalman_raw.png")
        plt.close(fig)

    # ── 4. Velocity-threshold filter ──────────────────────────────────────────
    base_mask = np.isfinite(vel_kf) & (np.abs(vel_kf) <= VEL_MAX)
    kept, dropped = int(base_mask.sum()), int((~base_mask).sum())
    print(
        f"\n── Velocity filter (|v| ≤ {VEL_MAX} m/s): kept {kept}, dropped {dropped} ──")

    times_f = times[base_mask]
    pos_raw_f = pos_raw[base_mask]
    scales_raw_f = scales_raw[base_mask]
    mode_seq_f = [m for m, k in zip(mode_seq, base_mask) if k]
    df_valid_f = df_valid[base_mask].reset_index(drop=True)

    # ── 5. Second Kalman pass on filtered data ────────────────────────────────
    print("\n── Kalman pass 2 (velocity-filtered data) ──")
    pos_kf_f, vel_kf_f = kalman_filter_pos_vel(times_f, pos_raw_f, mode_seq_f)
    scale_kf_f = kalman_filter_scale(scales_raw_f)
    spd_kf_f = np.abs(vel_kf_f)
    print(
        f"  peak |speed| : {spd_kf_f.max():.4f} m/s  ({spd_kf_f.max()*3.6:.3f} km/h)")
    print(
        f"  mean |speed| : {spd_kf_f.mean():.4f} m/s  ({spd_kf_f.mean()*3.6:.3f} km/h)")

    # ── Plot 2: filtered KF overview ──────────────────────────────────────────
    print("\nPlot 2  –  Kalman overview (post-velocity filter)")
    with plt.style.context("default"):
        fig = _plot_kf_overview(
            times_f, pos_raw_f, pos_kf_f, vel_kf_f,
            scales_raw_f, scale_kf_f,
            title_pos_suffix=f"  (|velocity| ≤ {VEL_MAX} m/s)")
        _save(fig, out_dir, "plot2_kalman_filtered.png")
        plt.close(fig)

    # ── 6. Loop-closure / 1-D trajectory optimisation ────────────────────────
    print("\n── Loop-closure optimisation ──")

    # Pull wrist global positions from the velocity-filtered rows
    lwrist_raw = df_valid_f["lwrist_pos_m"].to_numpy(dtype=float)
    rwrist_raw = df_valid_f["rwrist_pos_m"].to_numpy(dtype=float)
    lwrist_interp = pd.Series(lwrist_raw).interpolate(
        method="linear", limit_direction="both").to_numpy()
    rwrist_interp = pd.Series(rwrist_raw).interpolate(
        method="linear", limit_direction="both").to_numpy()

    turn_idx = int(np.argmax(pos_kf_f))
    raw_max_pos = float(pos_kf_f[turn_idx])
    scale_out = POOL_LENGTH_M / raw_max_pos

    corrected_pos = np.copy(pos_kf_f)
    corrected_lwrist_pos = np.copy(lwrist_interp)
    corrected_rwrist_pos = np.copy(rwrist_interp)

    # Outbound lap: linear rescale
    corrected_pos[:turn_idx + 1] = pos_kf_f[:turn_idx + 1] * scale_out
    corrected_lwrist_pos[:turn_idx +
                         1] = lwrist_interp[:turn_idx + 1] * scale_out
    corrected_rwrist_pos[:turn_idx +
                         1] = rwrist_interp[:turn_idx + 1] * scale_out

    # Return lap: warp so endpoints map to [25 m … expected_end]
    if turn_idx < len(pos_kf_f) - 1:
        expected_end = float(pos_kf_f[-1] * scale_out)
        xp = [pos_kf_f[-1], pos_kf_f[turn_idx]]
        fp = [expected_end, POOL_LENGTH_M]
        corrected_pos[turn_idx:] = np.interp(pos_kf_f[turn_idx:], xp, fp)

        for i in range(turn_idx, len(corrected_pos)):
            denom = pos_kf_f[turn_idx] - pos_kf_f[-1]
            ratio = (pos_kf_f[i] - pos_kf_f[-1]) / denom if denom != 0 else 0.0
            base = expected_end + ratio * (POOL_LENGTH_M - expected_end)
            corrected_lwrist_pos[i] = base + \
                (lwrist_interp[i] - pos_kf_f[i]) * scale_out
            corrected_rwrist_pos[i] = base + \
                (rwrist_interp[i] - pos_kf_f[i]) * scale_out

    # Derivatives
    corrected_vel_raw = np.gradient(corrected_pos,        times_f)
    lwrist_vel_raw = np.gradient(corrected_lwrist_pos, times_f)
    rwrist_vel_raw = np.gradient(corrected_rwrist_pos, times_f)

    # Smooth centroid velocity with Savitzky-Golay
    wl = min(61, len(corrected_vel_raw) if len(corrected_vel_raw) %
             2 != 0 else len(corrected_vel_raw) - 1)
    wl = max(wl, 3)
    po = min(3, wl - 1)
    corrected_vel = savgol_filter(corrected_vel_raw, wl, po)
    corrected_spd = np.abs(corrected_vel)

    lwrist_vel = clean_wrist_velocity(lwrist_vel_raw)
    rwrist_vel = clean_wrist_velocity(rwrist_vel_raw)
    lwrist_spd = np.abs(lwrist_vel)
    rwrist_spd = np.abs(rwrist_vel)

    rel_lwrist_vel = lwrist_vel - corrected_vel
    rel_rwrist_vel = rwrist_vel - corrected_vel

    print(f"  Turn index {turn_idx}  (t = {times_f[turn_idx]:.2f} s)")
    print(
        f"  Raw max position : {raw_max_pos:.2f} m → scale_out = {scale_out:.4f}")
    print(
        f"  Peak |speed|     : {corrected_spd.max():.4f} m/s  ({corrected_spd.max()*3.6:.3f} km/h)")
    print(
        f"  Mean |speed|     : {corrected_spd.mean():.4f} m/s  ({corrected_spd.mean()*3.6:.3f} km/h)")

    # ── Plot 3: loop-closure corrected 4-panel overview ───────────────────────
    print("\nPlot 3  –  Loop-closure corrected overview")
    with plt.style.context("default"):
        fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)

        axes[0].plot(times_f, corrected_pos,        "-",
                     lw=2.0, color="black",   label="Centroid")
        axes[0].plot(times_f, corrected_lwrist_pos, "--", lw=1.5,
                     color="teal",    label="Left Wrist",  alpha=0.8)
        axes[0].plot(times_f, corrected_rwrist_pos, ":",  lw=1.5,
                     color="magenta", label="Right Wrist", alpha=0.8)
        axes[0].axhline(0,             color="forestgreen", lw=0.7, ls=":")
        axes[0].axhline(POOL_LENGTH_M, color="red",         lw=1.0, ls=":",
                        label=f"Turn ({POOL_LENGTH_M:.0f} m)")
        axes[0].set_ylabel("Position (m)")
        axes[0].set_title("Swimmer Trajectory Optimized (Loop Closure)")
        axes[0].legend(fontsize=9)
        axes[0].grid(alpha=0.6)

        axes[1].plot(times_f, corrected_vel, "-",  lw=2.0,
                     color="orange",  label="Centroid Velocity")
        axes[1].plot(times_f, lwrist_vel,    "--", lw=1.5,
                     color="teal",    label="Left Wrist Velocity",  alpha=0.8)
        axes[1].plot(times_f, rwrist_vel,    ":",  lw=1.5,
                     color="magenta", label="Right Wrist Velocity", alpha=0.8)
        axes[1].axhline(0, color="black", lw=0.8, ls=":")
        axes[1].set_ylabel("Velocity (m/s)")
        axes[1].set_title("Corrected and Smoothed Velocity")
        axes[1].legend(fontsize=9)
        axes[1].grid(alpha=0.6)

        axes[2].plot(times_f, rel_lwrist_vel, "--", lw=1.5,
                     color="teal",    label="Left Wrist (Relative)",  alpha=0.9)
        axes[2].plot(times_f, rel_rwrist_vel, ":",  lw=1.5,
                     color="magenta", label="Right Wrist (Relative)", alpha=0.9)
        axes[2].axhline(0, color="orange", lw=2.0, ls="-",
                        label="Centroid Baseline (0 m/s)", alpha=0.8)
        axes[2].set_ylabel("Rel. Velocity (m/s)")
        axes[2].set_title("Wrist Velocity Relative to Swimmer Centroid")
        axes[2].legend(fontsize=9)
        axes[2].grid(alpha=0.6)

        axes[3].fill_between(times_f, corrected_spd,
                             alpha=0.25, color="deepskyblue")
        axes[3].plot(times_f, corrected_spd, "-",  lw=2.0,
                     color="deepskyblue", label="Centroid |speed|")
        axes[3].plot(times_f, lwrist_spd,    "--", lw=1.5,
                     color="teal",        label="Left Wrist |speed|",  alpha=0.8)
        axes[3].plot(times_f, rwrist_spd,    ":",  lw=1.5,
                     color="magenta",     label="Right Wrist |speed|", alpha=0.8)
        axes[3].set_ylabel("Speed (m/s)")
        axes[3].set_xlabel("Time (s)")
        axes[3].set_title("Corrected and Smoothed Swimmer Speed")
        axes[3].legend(fontsize=9)
        axes[3].grid(alpha=0.6)

        plt.tight_layout()
        _save(fig, out_dir, "plot3_loop_closure.png")
        plt.close(fig)

    # ── Plot 4: wrist position relative to centroid (time-series) ─────────────
    print("\nPlot 4  –  Wrist movements relative to centroid (time-series)")
    n = len(times_f)
    lw = _fill_nan_interp(corrected_lwrist_pos)
    rw = _fill_nan_interp(corrected_rwrist_pos)

    lw_rel = lw - corrected_pos
    rw_rel = rw - corrected_pos
    lw_rel_f = _fill_nan_interp(lw_rel)
    rw_rel_f = _fill_nan_interp(rw_rel)

    # Savitzky-Golay smoothing
    w_sg = min(51, max(3, n // 8))
    if w_sg % 2 == 0:
        w_sg -= 1
    if w_sg < 3:
        w_sg = 3
    p_sg = 3 if w_sg > 3 else 2
    lw_s = savgol_filter(lw_rel_f, w_sg, p_sg)
    rw_s = savgol_filter(rw_rel_f, w_sg, p_sg)

    with plt.style.context("default"):
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(times_f, lw_rel, ".", ms=3, alpha=0.5,
                label="Left wrist (raw rel)")
        ax.plot(times_f, rw_rel, ".", ms=3, alpha=0.5,
                label="Right wrist (raw rel)")
        ax.plot(times_f, lw_s,   "-", lw=1.5, color="teal",
                label="Left wrist (smoothed)")
        ax.plot(times_f, rw_s,   "-", lw=1.5, color="magenta",
                label="Right wrist (smoothed)")
        ax.axhline(0.0, color="black", ls="--", lw=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Relative distance along lane (m)")
        ax.set_title(
            "Wrist movements anchored to swimmer centroid (centroid subtracted)")
        ax.legend(fontsize=9, ncol=2)
        ax.grid(alpha=0.35)
        plt.tight_layout()
        _save(fig, out_dir, "plot4_wrist_relative_timeseries.png")
        plt.close(fig)

    # ── Shared wrist pixel data for spatial plots (Plots 5, 6, 7) ─────────────
    pix_cols = ["lwrist_x", "lwrist_y", "rwrist_x", "rwrist_y",
                "lshoulder_x", "lshoulder_y", "rshoulder_x", "rshoulder_y", "pos_m"]

    # Check all required columns are present
    missing = [c for c in pix_cols if c not in df_all.columns]
    if missing:
        print(
            f"\n  ⚠  Skipping spatial wrist plots – columns missing: {missing}")
    else:
        df_pix = df_all.dropna(subset=pix_cols).copy()
        df_pix = df_pix[(df_pix["lwrist_x"] != 0) & (df_pix["rwrist_x"] != 0)]
        print(f"\n  Wrist-pixel rows available: {len(df_pix)}")

        if len(df_pix) < 10:
            print("  ⚠  Too few wrist-pixel rows; skipping Plots 5–7.")
        else:
            # ── Plot 5: spatial wrist path anchored to own shoulders ───────────
            print("Plot 5  –  Spatial wrist path (own-shoulder anchor)")
            df5 = _apply_wrist_cm(df_pix)
            df5 = _filter_and_smooth_wrist(df5)
            print(f"  Rows after filter+smooth: {len(df5)}")

            if len(df5) > 0:
                with plt.style.context("default"):
                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.plot(df5["lwrist_rel_x_cm"], df5["lwrist_rel_y_cm"],
                            color="teal",    alpha=0.7, lw=2, label="Left Wrist Path")
                    ax.plot(df5["rwrist_rel_x_cm"], df5["rwrist_rel_y_cm"],
                            color="magenta", alpha=0.7, lw=2, label="Right Wrist Path")
                    ax.plot(0, 0, marker="X", color="black", ms=12,
                            label="Shoulder Origin (0, 0)")
                    ax.set_aspect("equal", adjustable="box")
                    ax.invert_yaxis()
                    ax.set_title("Spatial Wrist Symmetry (Anchored at Shoulders)\nCentimetres",
                                 fontsize=16)
                    ax.set_xlabel("Horizontal Position (cm)", fontsize=12)
                    ax.set_ylabel("Vertical Position (cm)",  fontsize=12)
                    ax.axhline(0, color="gray", ls="--", alpha=0.6)
                    ax.axvline(0, color="gray", ls="--", alpha=0.6)
                    ax.legend(loc="upper right")
                    ax.grid(True, ls=":", alpha=0.6)
                    plt.tight_layout()
                    _save(fig, out_dir, "plot5_wrist_spatial_path.png")
                    plt.close(fig)

            # ── Plot 6: unified KDE, outbound + flipped return ─────────────────
            print("Plot 6  –  Unified wrist KDE (outbound + flipped return)")
            df6_base = _apply_wrist_cm(df_pix)
            df6_base = _filter_and_smooth_wrist(df6_base)

            if len(df6_base) > 10:
                turn_row6 = df6_base["pos_m"].idxmax()
                df6_out = df6_base.loc[:turn_row6].copy()
                df6_ret = df6_base.loc[turn_row6:].copy()
                df6_ret["lwrist_rel_x_cm"] = df6_ret["lwrist_rel_x_cm"] * -1
                df6_ret["rwrist_rel_x_cm"] = df6_ret["rwrist_rel_x_cm"] * -1
                df6_comb = pd.concat([df6_out, df6_ret])

                with plt.style.context("default"):
                    fig, ax = plt.subplots(figsize=(10, 10))
                    sns.kdeplot(data=df6_comb, x="lwrist_rel_x_cm", y="lwrist_rel_y_cm",
                                ax=ax, color="teal",    fill=True,  alpha=0.3, levels=6, thresh=0.1)
                    sns.kdeplot(data=df6_comb, x="lwrist_rel_x_cm", y="lwrist_rel_y_cm",
                                ax=ax, color="teal",    levels=6, thresh=0.1, linewidths=1.5)
                    sns.kdeplot(data=df6_comb, x="rwrist_rel_x_cm", y="rwrist_rel_y_cm",
                                ax=ax, color="magenta", fill=True,  alpha=0.3, levels=6, thresh=0.1)
                    sns.kdeplot(data=df6_comb, x="rwrist_rel_x_cm", y="rwrist_rel_y_cm",
                                ax=ax, color="magenta", levels=6, thresh=0.1, linewidths=1.5)
                    ax.plot(0, 0, marker="X", color="black",
                            ms=12, label="Shoulder (0, 0)")
                    ax.set_aspect("equal", adjustable="box")
                    ax.invert_yaxis()
                    ax.set_title("Unified Spatial Wrist Symmetry\n(Outbound + Flipped Return)",
                                 fontsize=16)
                    ax.set_xlabel("Forward Reach (cm)", fontsize=12)
                    ax.set_ylabel("Vertical Depth (cm)",  fontsize=12)
                    ax.axhline(0, color="gray", ls="--", alpha=0.6)
                    ax.axvline(0, color="gray", ls="--", alpha=0.6)
                    ax.grid(True, ls=":", alpha=0.6)
                    custom = [
                        Line2D([0], [0], color="teal",    lw=3, alpha=0.6),
                        Line2D([0], [0], color="magenta", lw=3, alpha=0.6),
                        Line2D([0], [0], marker="X", color="black",
                               linestyle="None", ms=10),
                    ]
                    ax.legend(custom, ["Left Wrist Path", "Right Wrist Path",
                                       "Shoulder Anchor"], loc="upper right")
                    plt.tight_layout()
                    _save(fig, out_dir, "plot6_kde_unified.png")
                    plt.close(fig)

            # ── Plot 7: unified KDE, camera-facing shoulder anchor ─────────────
            print("Plot 7  –  Unified KDE (camera-facing shoulder anchor)")

            # Split BEFORE computing offsets (outbound → right shoulder anchor,
            # return → left shoulder anchor; then flip return X)
            turn_row7 = df_pix["pos_m"].idxmax()
            df7_out_raw = df_pix.loc[:turn_row7].copy()
            df7_ret_raw = df_pix.loc[turn_row7:].copy()

            # Outbound: both wrists relative to RIGHT shoulder
            for wx, wy, col in [("lwrist_x", "lwrist_y", "lw"),
                                ("rwrist_x", "rwrist_y", "rw")]:
                df7_out_raw[f"{col}rist_rel_x_cm"] = \
                    ((df7_out_raw[wx] - df7_out_raw["rshoulder_x"]
                      ) / df7_out_raw["px_per_m"]) * 100
                df7_out_raw[f"{col}rist_rel_y_cm"] = \
                    ((df7_out_raw[wy] - df7_out_raw["rshoulder_y"]
                      ) / df7_out_raw["px_per_m"]) * 100

            # Return: both wrists relative to LEFT shoulder, then flip X
            for wx, wy, col in [("lwrist_x", "lwrist_y", "lw"),
                                ("rwrist_x", "rwrist_y", "rw")]:
                df7_ret_raw[f"{col}rist_rel_x_cm"] = \
                    ((df7_ret_raw[wx] - df7_ret_raw["lshoulder_x"]
                      ) / df7_ret_raw["px_per_m"]) * 100 * -1
                df7_ret_raw[f"{col}rist_rel_y_cm"] = \
                    ((df7_ret_raw[wy] - df7_ret_raw["lshoulder_y"]
                      ) / df7_ret_raw["px_per_m"]) * 100

            df7_comb = pd.concat([df7_out_raw, df7_ret_raw]
                                 ).reset_index(drop=True)
            df7_comb = _filter_and_smooth_wrist(df7_comb)
            print(f"  Camera-facing KDE rows: {len(df7_comb)}")

            if len(df7_comb) > 10:
                with plt.style.context("default"):
                    fig, ax = plt.subplots(figsize=(12, 10))
                    sns.kdeplot(data=df7_comb, x="lwrist_rel_x_cm", y="lwrist_rel_y_cm",
                                ax=ax, color="teal",    fill=True,  alpha=0.3, levels=6, thresh=0.1)
                    sns.kdeplot(data=df7_comb, x="lwrist_rel_x_cm", y="lwrist_rel_y_cm",
                                ax=ax, color="teal",    levels=6, thresh=0.1, linewidths=1.5)
                    sns.kdeplot(data=df7_comb, x="rwrist_rel_x_cm", y="rwrist_rel_y_cm",
                                ax=ax, color="magenta", fill=True,  alpha=0.3, levels=6, thresh=0.1)
                    sns.kdeplot(data=df7_comb, x="rwrist_rel_x_cm", y="rwrist_rel_y_cm",
                                ax=ax, color="magenta", levels=6, thresh=0.1, linewidths=1.5)
                    ax.plot(0, 0, marker="X", color="black", ms=12,
                            label="Camera-Facing Shoulder")
                    ax.set_aspect("equal", adjustable="box")
                    ax.invert_yaxis()
                    ax.set_title("Unified Density Heatmap\n"
                                 "(Anchored to Camera-Facing Shoulder)", fontsize=16)
                    ax.set_xlabel("Forward Reach (cm)", fontsize=12)
                    ax.set_ylabel("Vertical Depth (cm)",  fontsize=12)
                    ax.axhline(0, color="gray", ls="--", alpha=0.6)
                    ax.axvline(0, color="gray", ls="--", alpha=0.6)
                    ax.grid(True, ls=":", alpha=0.6)
                    custom = [
                        Line2D([0], [0], color="teal",    lw=3, alpha=0.6),
                        Line2D([0], [0], color="magenta", lw=3, alpha=0.6),
                        Line2D([0], [0], marker="X", color="black",
                               linestyle="None", ms=10),
                    ]
                    ax.legend(custom,
                              ["Left Wrist Path", "Right Wrist Path",
                               "Camera-Facing Shoulder Anchor"],
                              loc="center left", bbox_to_anchor=(1.02, 0.5))
                    plt.tight_layout()
                    _save(fig, out_dir, "plot7_kde_camera_facing.png")
                    plt.close(fig)

    # ── Plot 8: 50 m trajectory vs ground truth (optional) ────────────────────
    if GT_ENABLED:
        print("\nPlot 8  –  50 m trajectory evaluation vs ground truth")

        start_idx_gt = int(np.argmin(np.abs(times_f - GT_START_VIDEO_S)))
        synced_times = times_f - times_f[start_idx_gt]
        shifted_pos = corrected_pos - corrected_pos[start_idx_gt]

        turn_gt = int(np.argmax(shifted_pos))
        final_pos = shifted_pos * (POOL_LENGTH_M / np.max(shifted_pos))

        out_t = GT_OUTBOUND_TIMES_S - GT_START_VIDEO_S
        ret_t = GT_RETURN_TIMES_S - GT_START_VIDEO_S
        out_d = np.arange(2.5, POOL_LENGTH_M + 0.1, MARKER_REAL_DIST_M)
        ret_d = np.arange(
            POOL_LENGTH_M - MARKER_REAL_DIST_M, -0.1, -MARKER_REAL_DIST_M)

        def _find_splits(dists, pos_seg, t_seg, phase_start_idx):
            auto_t = []
            for dist in dists:
                if len(pos_seg) > 0:
                    cond = (pos_seg >= dist) if dist > 0 else (pos_seg <= dist)
                    if np.any(cond):
                        auto_t.append(t_seg[np.argmax(cond)])
                    else:
                        auto_t.append(np.nan)
                else:
                    auto_t.append(np.nan)
            return auto_t

        auto_out_t = _find_splits(out_d, final_pos[start_idx_gt:turn_gt],
                                  synced_times[start_idx_gt:turn_gt], start_idx_gt)
        auto_ret_t = _find_splits(ret_d, final_pos[turn_gt:],
                                  synced_times[turn_gt:], turn_gt)

        all_errs = []
        print(f"\n{'Phase':<9} | {'Dist (m)':>8} | {'True (s)':>10} | "
              f"{'Auto (s)':>10} | {'Err (s)':>8}")
        print("-" * 57)
        for dist, tt, at in zip(out_d, out_t, auto_out_t):
            if not np.isnan(at):
                e = at - tt
                all_errs.append(e)
                print(
                    f"{'Outbound':<9} | {dist:>8.1f} | {tt:>10.2f} | {at:>10.2f} | {e:>8.2f}")
            else:
                print(
                    f"{'Outbound':<9} | {dist:>8.1f} | {tt:>10.2f} | {'N/A':>10} | {'N/A':>8}")
        for dist, tt, at in zip(ret_d, ret_t, auto_ret_t):
            if not np.isnan(at):
                e = at - tt
                all_errs.append(e)
                print(
                    f"{'Return':<9} | {dist:>8.1f} | {tt:>10.2f} | {at:>10.2f} | {e:>8.2f}")
            else:
                print(
                    f"{'Return':<9} | {dist:>8.1f} | {tt:>10.2f} | {'N/A':>10} | {'N/A':>8}")
        print("-" * 57)
        if all_errs:
            ae = np.array(all_errs)
            print(f"MAE  : {np.mean(np.abs(ae)):.3f} s")
            print(f"RMSE : {np.sqrt(np.mean(ae**2)):.3f} s")

        with plt.style.context("default"):
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(synced_times[start_idx_gt:], final_pos[start_idx_gt:],
                    "-", color="black", lw=2.5, label="System Trajectory")
            ax.plot(np.concatenate([out_t, ret_t]),
                    np.concatenate([out_d,  ret_d]),
                    "o", color="red", ms=8, label="Ground Truth Splits", zorder=3)
            ax.set_xlabel("Elapsed Time (s)", fontsize=12)
            ax.set_ylabel("Distance (m)",     fontsize=12)
            ax.set_title("Full 50 m Trajectory Evaluation: "
                         "Automated Tracking vs Ground Truth", fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(alpha=0.6)
            plt.tight_layout()
            _save(fig, out_dir, "plot8_trajectory_evaluation.png")
            plt.close(fig)
    else:
        print("\n  (Plot 8 skipped – set GT_ENABLED = True to activate ground-truth evaluation)")

    print(f"\n✓  All plots saved to: {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise swimmer tracking results from a CSV produced by main.py")
    parser.add_argument(
        "csv_path", nargs="?", default=DEFAULT_CSV,
        help=f"Path to the tracking CSV (default: {DEFAULT_CSV})")
    args = parser.parse_args()

    if not os.path.isfile(args.csv_path):
        print(f"Error: CSV not found: {args.csv_path}", file=sys.stderr)
        sys.exit(1)

    main(args.csv_path)
