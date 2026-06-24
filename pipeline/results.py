"""
results.py - Post-processing and visualisation for swimmer tracking CSV.

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
PARTICIPANT_ID = "P049"
POSE_ENGINE = "all"  # "vitpose" | "yolo226l-pose" | "mediapipe" | "all"

# Input paths
OUT_DIR = f"output/{SWIMMING_STYLE}_{PARTICIPANT_ID}"
# All engines (vit/yolo/mp) live as columns in a single tracking CSV.
DEFAULT_CSV = os.path.join(OUT_DIR, f"tracking_results_{POSE_ENGINE}.csv")

# Output paths
GRAPHS_DIR = os.path.join(OUT_DIR, "graphs")


MARKER_REAL_DIST_M = 2.5    # physical gap between lane-rope markers (m)
POOL_LENGTH_M = 25.0        # distance from wall to turn (m)
VEL_MAX = 5.0               # velocity outlier ceiling (m/s)
# drop data before this timestamp (s); matches notebook's time_s >= 17.0 filter
START_TIME_S = 17.0

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

# ── keypoint index maps for the new raw-CSV format ────────────────────────────
# COCO-17 ordering (vitpose / yolo226l-pose): 5=L_Shoulder,6=R_Shoulder,
# 9=L_Wrist,10=R_Wrist, 11=L_Hip,12=R_Hip
# MediaPipe-33 ordering: 11=L_Shoulder,12=R_Shoulder,15=L_Wrist,16=R_Wrist,
# 23=L_Hip,24=R_Hip
_ENGINE_PREFIX = {"vitpose": "vit", "yolo226l-pose": "yolo", "mediapipe": "mp"}

_JOINT_IDX = {
    "vitpose":       {"lshoulder": 5,  "rshoulder": 6,
                      "lwrist": 9,    "rwrist": 10,
                      "lhip": 11,     "rhip": 12},
    "yolo226l-pose": {"lshoulder": 5,  "rshoulder": 6,
                      "lwrist": 9,    "rwrist": 10,
                      "lhip": 11,     "rhip": 12},
    "mediapipe":     {"lshoulder": 11, "rshoulder": 12,
                      "lwrist": 15,   "rwrist": 16,
                      "lhip": 23,     "rhip": 24},
}

CONF_THRESH = 0.3  # below this, joint coordinates are treated as missing

# COCO-17 joint names (index → label)
COCO17_NAMES = [
    "Nose", "L-Eye", "R-Eye", "L-Ear", "R-Ear",
    "L-Shoulder", "R-Shoulder", "L-Elbow", "R-Elbow",
    "L-Wrist", "R-Wrist", "L-Hip", "R-Hip",
    "L-Knee", "R-Knee", "L-Ankle", "R-Ankle",
]


MARKER_MATCH_THRESH_PX = 50.0  # max pixel jump to match a marker frame-to-frame


def _adapt_raw_csv(df_all: pd.DataFrame, engine: str) -> pd.DataFrame:
    """Convert the raw per-frame keypoint CSV (from main.py) into the
    column layout expected by the rest of this script.

    Reproduces the notebook's ego-motion pipeline:
      - tracks a running global camera offset (global_camera_x) by matching
        lane-rope markers frame-to-frame and accumulating their median
        pixel shift
      - derives px_per_m from the OUTERMOST pair of sorted markers, using
        (n_markers - 1) * MARKER_REAL_DIST_M as the physical span
      - projects the swimmer (hip centroid) and both wrists onto the
        marker baseline when >= 2 markers are present, applying the
        ego-motion offset before converting to metres
      - falls back to raw (x + global_camera_x) / px_per_m for wrists when
        fewer than 2 markers are detected
    """
    if engine not in _ENGINE_PREFIX:
        raise ValueError(
            f"Unknown POSE_ENGINE '{engine}'. "
            f"Expected one of {list(_ENGINE_PREFIX)}.")

    prefix = _ENGINE_PREFIX[engine]
    jm = _JOINT_IDX[engine]

    marker_cols = []
    for m_idx in range(5):  # main.py tracks up to 5 markers (marker_0..4)
        xc, yc = f"marker_{m_idx}_x", f"marker_{m_idx}_y"
        if xc in df_all.columns and yc in df_all.columns:
            marker_cols.append(m_idx)

    required = ["timestamp"]
    for name, idx in jm.items():
        required += [f"{prefix}_joint_{idx}_x", f"{prefix}_joint_{idx}_y",
                     f"{prefix}_joint_{idx}_conf"]
    missing = [c for c in required if c not in df_all.columns]
    if missing:
        raise ValueError(
            "Input CSV is missing required columns for the new format: "
            f"{missing}")
    if len(marker_cols) < 2:
        raise ValueError(
            "Input CSV must contain at least marker_0_{x,y} and "
            "marker_1_{x,y} columns for scale/ego-motion computation.")

    n = len(df_all)
    out = pd.DataFrame(index=df_all.index)
    out["time_s"] = df_all["timestamp"].astype(float)

    # ── joint extraction with confidence gating ────────────────────────────
    for name, idx in jm.items():
        x = df_all[f"{prefix}_joint_{idx}_x"].astype(float)
        y = df_all[f"{prefix}_joint_{idx}_y"].astype(float)
        conf = df_all[f"{prefix}_joint_{idx}_conf"].astype(float)
        low_conf = conf < CONF_THRESH
        out[f"{name}_x"] = x.where(~low_conf, np.nan)
        out[f"{name}_y"] = y.where(~low_conf, np.nan)

    hip_x_arr = ((out["lhip_x"] + out["rhip_x"]) / 2.0).to_numpy()
    hip_y_arr = ((out["lhip_y"] + out["rhip_y"]) / 2.0).to_numpy()
    lwrist_x_arr = out["lwrist_x"].to_numpy()
    lwrist_y_arr = out["lwrist_y"].to_numpy()
    rwrist_x_arr = out["rwrist_x"].to_numpy()
    rwrist_y_arr = out["rwrist_y"].to_numpy()

    # ── per-frame marker positions (as lists of (x, y), NaNs dropped) ──────
    frame_markers = []
    for i in range(n):
        pts = []
        for m_idx in marker_cols:
            mx = df_all.at[i, f"marker_{m_idx}_x"]
            my = df_all.at[i, f"marker_{m_idx}_y"]
            if pd.notna(mx) and pd.notna(my):
                pts.append((float(mx), float(my)))
        frame_markers.append(pts)

    # ── sequential ego-motion + scale + projection ──────────────────────────
    pos_m = np.full(n, np.nan)
    lwrist_pos_m = np.full(n, np.nan)
    rwrist_pos_m = np.full(n, np.nan)
    px_per_m_arr = np.full(n, np.nan)

    global_camera_x = 0.0
    prev_markers_x: list = []
    prev_px_per_m = None
    prev_delta_x = 0.0

    for i in range(n):
        current_markers_x = [p[0] for p in frame_markers[i]]

        # 2. Frame-to-frame displacement of markers
        delta_x = 0.0
        if prev_markers_x and current_markers_x:
            valid_deltas = []
            for cx in current_markers_x:
                diffs = [cx - px for px in prev_markers_x]
                abs_diffs = [abs(d) for d in diffs]
                min_idx = int(np.argmin(abs_diffs))
                if abs_diffs[min_idx] < MARKER_MATCH_THRESH_PX:
                    valid_deltas.append(diffs[min_idx])
            if valid_deltas:
                delta_x = float(np.median(valid_deltas))
            else:
                delta_x = prev_delta_x
        elif prev_markers_x and not current_markers_x:
            # Linear projection if markers lost
            delta_x = prev_delta_x

        # 3. Update global camera offset
        global_camera_x -= delta_x
        prev_delta_x = delta_x

        # 4 & 5. Scale from outermost sorted-marker pair
        sorted_markers = sorted(frame_markers[i], key=lambda p: p[0])
        px_per_m = prev_px_per_m if prev_px_per_m is not None else 1.0

        if len(sorted_markers) >= 2:
            A_pt = np.array(sorted_markers[0])
            B_pt = np.array(sorted_markers[-1])
            dist_px = np.linalg.norm(B_pt - A_pt)
            num_intervals = len(sorted_markers) - 1
            physical_dist = num_intervals * MARKER_REAL_DIST_M
            if physical_dist > 0:
                px_per_m = dist_px / physical_dist
                prev_px_per_m = px_per_m

        px_per_m_arr[i] = px_per_m

        # ── Swimmer (hip centroid) and wrist global positions ──────────────
        hx, hy = hip_x_arr[i], hip_y_arr[i]
        if np.isfinite(hx) and np.isfinite(hy):
            virtual_x = hx + global_camera_x
            global_pos_m = virtual_x / px_per_m if px_per_m > 0 else 0.0
            pos_m[i] = global_pos_m

            if len(sorted_markers) >= 2:
                AB = B_pt - A_pt
                AB_len = np.linalg.norm(AB)
                if AB_len > 0:
                    AB_unit = AB / AB_len

                    if np.isfinite(lwrist_x_arr[i]) and np.isfinite(lwrist_y_arr[i]):
                        LW_pt = np.array([lwrist_x_arr[i], lwrist_y_arr[i]])
                        virtual_lw_x = LW_pt[0] + global_camera_x
                        lwrist_pos_m[i] = (virtual_lw_x / px_per_m
                                           if px_per_m > 0 else 0.0)

                    if np.isfinite(rwrist_x_arr[i]) and np.isfinite(rwrist_y_arr[i]):
                        RW_pt = np.array([rwrist_x_arr[i], rwrist_y_arr[i]])
                        virtual_rw_x = RW_pt[0] + global_camera_x
                        rwrist_pos_m[i] = (virtual_rw_x / px_per_m
                                           if px_per_m > 0 else 0.0)
            else:
                # Fallback: no baseline available, use raw X scaling
                if np.isfinite(lwrist_x_arr[i]):
                    lwrist_pos_m[i] = ((lwrist_x_arr[i] + global_camera_x) / px_per_m
                                       if px_per_m > 0 else 0.0)
                if np.isfinite(rwrist_x_arr[i]):
                    rwrist_pos_m[i] = ((rwrist_x_arr[i] + global_camera_x) / px_per_m
                                       if px_per_m > 0 else 0.0)

        # 6. State update
        prev_markers_x = current_markers_x

    out["px_per_m"] = px_per_m_arr
    out["pos_m"] = pos_m
    out["lwrist_pos_m"] = lwrist_pos_m
    out["rwrist_pos_m"] = rwrist_pos_m

    # selection_mode: not present in the new format; only "auto" is defined
    # in _MODE_NOISE_SCALE, so use it for every row.
    out["selection_mode"] = "auto"

    return out


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

def process_engine(df_raw: pd.DataFrame, engine: str, base_out_dir: str) -> dict:
    """Run the full processing/plotting pipeline for a single pose engine.

    Plots are saved under base_out_dir/<engine>/. Returns a dict of the
    key time-series (for cross-engine comparison plots).
    """
    out_dir = os.path.join(base_out_dir, engine)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*70}\nProcessing engine: {engine}\n{'='*70}")
    print(f"  Total rows: {len(df_raw)}")

    # Adapt the new raw-keypoint CSV (from main.py) into the column layout
    # used throughout this script.
    df_all = _adapt_raw_csv(df_raw, engine)

    # NOTE: in the notebook, the time_s >= 17.0 trim is applied ONLY to the
    # spatial wrist plots (5-7), not to the Kalman/loop-closure plots (1-4),
    # which use the full tracking_records. df_all (full) feeds plots 1-4;
    # df_spatial (trimmed) feeds plots 5-7.
    if START_TIME_S > 0.0:
        df_spatial = df_all[df_all["time_s"] >=
                            START_TIME_S].copy().reset_index(drop=True)
        print(f"  Rows for spatial plots (time_s >= {START_TIME_S}s): "
              f"{len(df_spatial)}")
    else:
        df_spatial = df_all

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
    missing = [c for c in pix_cols if c not in df_spatial.columns]
    if missing:
        print(
            f"\n  ⚠  Skipping spatial wrist plots – columns missing: {missing}")
    else:
        df_pix = df_spatial.dropna(subset=pix_cols).copy()
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

    # ── Plot 9: per-joint confidence distributions (box plot) ─────────────────
    print("\nPlot 9  –  Per-joint confidence distributions")
    prefix = _ENGINE_PREFIX[engine]
    jm = _JOINT_IDX[engine]
    n_joints = max(jm.values()) + 1

    conf_data = []
    joint_labels = []
    for j in range(n_joints):
        col = f"{prefix}_joint_{j}_conf"
        if col in df_raw.columns:
            vals = df_raw[col].dropna().to_numpy(dtype=float)
            conf_data.append(vals)
            # Use COCO17 names if available, else numeric
            if engine in ("vitpose", "yolo226l-pose") and j < len(COCO17_NAMES):
                joint_labels.append(COCO17_NAMES[j])
            else:
                joint_labels.append(str(j))

    if conf_data:
        with plt.style.context("default"):
            fig, ax = plt.subplots(figsize=(14, 5))
            bp = ax.boxplot(conf_data, patch_artist=True, notch=False,
                            medianprops=dict(color="black", lw=2))
            colors = plt.cm.RdYlGn(
                [np.median(d) for d in conf_data])
            for patch, col in zip(bp["boxes"], colors):
                patch.set_facecolor(col)
                patch.set_alpha(0.75)
            ax.axhline(CONF_THRESH, color="red", ls="--", lw=1.2,
                       label=f"Confidence threshold ({CONF_THRESH})")
            ax.set_xticks(range(1, len(joint_labels) + 1))
            ax.set_xticklabels(joint_labels, rotation=45,
                               ha="right", fontsize=9)
            ax.set_ylabel("Confidence Score")
            ax.set_title(f"[{engine}]  Per-Joint Confidence Distribution\n"
                         "(coloured green=high, red=low by median)")
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.4)
            plt.tight_layout()
            _save(fig, out_dir, "plot9_joint_confidence_boxplot.png")
            plt.close(fig)

    # ── Plot 10: per-joint detection rate (above CONF_THRESH) ─────────────────
    print("\nPlot 10  –  Per-joint detection rate above confidence threshold")
    det_rates = []
    for j in range(n_joints):
        col = f"{prefix}_joint_{j}_conf"
        if col in df_raw.columns:
            vals = df_raw[col].dropna()
            det_rates.append(float((vals >= CONF_THRESH).mean()) * 100)
        else:
            det_rates.append(0.0)

    if det_rates:
        bar_colors = plt.cm.RdYlGn(
            [r / 100.0 for r in det_rates])
        with plt.style.context("default"):
            fig, ax = plt.subplots(figsize=(14, 4.5))
            bars = ax.bar(range(len(det_rates)), det_rates,
                          color=bar_colors, edgecolor="white", linewidth=0.5)
            ax.axhline(90, color="forestgreen", ls="--", lw=1,
                       label="90% target", alpha=0.8)
            ax.axhline(70, color="orange", ls=":",  lw=1,
                       label="70% warning", alpha=0.8)
            ax.set_xticks(range(len(joint_labels)))
            ax.set_xticklabels(joint_labels, rotation=45,
                               ha="right", fontsize=9)
            ax.set_ylabel(f"Frames detected ≥ {CONF_THRESH} (%)")
            ax.set_ylim(0, 105)
            ax.set_title(f"[{engine}]  Per-Joint Detection Rate  "
                         f"(threshold = {CONF_THRESH})")
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.35)
            plt.tight_layout()
            _save(fig, out_dir, "plot10_joint_detection_rate.png")
            plt.close(fig)

    # ── Plot 11: temporal confidence heatmap (joint × frame) ──────────────────
    print("\nPlot 11  –  Temporal confidence heatmap (joint × time)")
    conf_matrix_rows = []
    for j in range(n_joints):
        col = f"{prefix}_joint_{j}_conf"
        if col in df_raw.columns:
            conf_matrix_rows.append(
                df_raw[col].fillna(0).to_numpy(dtype=float))

    if conf_matrix_rows:
        conf_mat = np.array(conf_matrix_rows)   # shape: (n_joints, n_frames)
        # Downsample frames for legibility
        stride = max(1, conf_mat.shape[1] // 400)
        conf_mat_ds = conf_mat[:, ::stride]
        times_all = df_raw["timestamp"].to_numpy(dtype=float)
        t_ticks = times_all[::stride]

        with plt.style.context("default"):
            fig, ax = plt.subplots(figsize=(16, 6))
            im = ax.imshow(conf_mat_ds, aspect="auto", origin="lower",
                           vmin=0, vmax=1, cmap="RdYlGn",
                           extent=[t_ticks[0], t_ticks[-1],
                                   -0.5, conf_mat_ds.shape[0] - 0.5])
            plt.colorbar(im, ax=ax, label="Confidence",
                         fraction=0.025, pad=0.02)
            ax.set_yticks(range(len(joint_labels)))
            ax.set_yticklabels(joint_labels, fontsize=8)
            ax.set_xlabel("Time (s)")
            ax.set_title(f"[{engine}]  Keypoint Confidence Over Time\n"
                         "(green = high confidence, red = low / missing)")
            plt.tight_layout()
            _save(fig, out_dir, "plot11_confidence_heatmap.png")
            plt.close(fig)

    # ── Plot 12: joint trajectory smoothness (frame-to-frame jitter) ──────────
    print("\nPlot 12  –  Joint trajectory smoothness (pixel jitter per joint)")
    jitter_per_joint = []
    for j in range(n_joints):
        xc = f"{prefix}_joint_{j}_x"
        yc = f"{prefix}_joint_{j}_y"
        cc = f"{prefix}_joint_{j}_conf"
        if xc in df_raw.columns and yc in df_raw.columns:
            conf_v = df_raw[cc].to_numpy(dtype=float)
            x_v = df_raw[xc].where(conf_v >= CONF_THRESH).to_numpy(dtype=float)
            y_v = df_raw[yc].where(conf_v >= CONF_THRESH).to_numpy(dtype=float)
            dx = np.diff(x_v)
            dy = np.diff(y_v)
            step = np.sqrt(dx**2 + dy**2)
            valid = step[np.isfinite(step)]
            # Median frame-to-frame pixel displacement (lower = smoother)
            jitter_per_joint.append(
                float(np.median(valid)) if len(valid) > 0 else np.nan)
        else:
            jitter_per_joint.append(np.nan)

    if any(np.isfinite(j) for j in jitter_per_joint):
        jitter_arr = np.array(jitter_per_joint)
        norm_j = np.where(np.isfinite(jitter_arr), jitter_arr, 0)
        max_j = norm_j.max() if norm_j.max() > 0 else 1.0
        bar_colors_j = plt.cm.RdYlGn_r([v / max_j for v in norm_j])
        with plt.style.context("default"):
            fig, ax = plt.subplots(figsize=(14, 4.5))
            ax.bar(range(len(jitter_arr)), jitter_arr,
                   color=bar_colors_j, edgecolor="white", linewidth=0.5)
            ax.set_xticks(range(len(joint_labels)))
            ax.set_xticklabels(joint_labels, rotation=45,
                               ha="right", fontsize=9)
            ax.set_ylabel("Median frame-to-frame displacement (px)")
            ax.set_title(f"[{engine}]  Joint Trajectory Smoothness\n"
                         "(lower = smoother tracking; green = stable, red = jittery)")
            ax.grid(axis="y", alpha=0.35)
            plt.tight_layout()
            _save(fig, out_dir, "plot12_joint_jitter.png")
            plt.close(fig)

    # Collect per-engine metrics for comparison plots
    # Mean confidence across key joints (shoulders, wrists, hips)
    key_joints = list(jm.values())
    mean_conf_key = float(np.nanmean([
        df_raw[f"{prefix}_joint_{j}_conf"].mean()
        for j in key_joints if f"{prefix}_joint_{j}_conf" in df_raw.columns
    ]))
    det_rate_key = float(np.nanmean([
        det_rates[j] for j in key_joints if j < len(det_rates)
    ]))
    jitter_key = float(np.nanmean([
        jitter_per_joint[j] for j in key_joints
        if j < len(jitter_per_joint) and np.isfinite(jitter_per_joint[j])
    ]))

    print(f"\n✓  Plots for '{engine}' saved to: {out_dir}")

    return {
        "engine": engine,
        "times_f": times_f,
        "pos_kf_f": pos_kf_f,
        "vel_kf_f": vel_kf_f,
        "spd_kf_f": spd_kf_f,
        "corrected_pos": corrected_pos,
        "corrected_vel": corrected_vel,
        "corrected_spd": corrected_spd,
        "corrected_lwrist_pos": corrected_lwrist_pos,
        "corrected_rwrist_pos": corrected_rwrist_pos,
        "lwrist_vel": lwrist_vel,
        "rwrist_vel": rwrist_vel,
        "lwrist_spd": lwrist_spd,
        "rwrist_spd": rwrist_spd,
        "rel_lwrist_vel": rel_lwrist_vel,
        "rel_rwrist_vel": rel_rwrist_vel,
        "lw_rel_smoothed": lw_s,
        "rw_rel_smoothed": rw_s,
        # new metrics for comparison plots
        "mean_conf_key": mean_conf_key,
        "det_rate_key": det_rate_key,
        "jitter_key": jitter_key,
        "det_rates": det_rates,
        "jitter_per_joint": jitter_per_joint,
        "prefix": prefix,
        "n_joints": n_joints,
        "joint_labels": joint_labels,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-engine comparison plots
# ─────────────────────────────────────────────────────────────────────────────

_ENGINE_COLORS = {
    "vitpose":       "tab:blue",
    "yolo226l-pose": "tab:orange",
    "mediapipe":     "tab:green",
}


def _engine_color(engine: str, fallback_idx: int = 0) -> str:
    palette = ["tab:blue", "tab:orange", "tab:green",
               "tab:red", "tab:purple", "tab:brown"]
    return _ENGINE_COLORS.get(engine, palette[fallback_idx % len(palette)])


def run_comparison_plots(results_by_engine: dict, base_out_dir: str) -> None:
    """Overlay key time-series across engines, saved to base_out_dir/comparison/."""
    out_dir = os.path.join(base_out_dir, "comparison")
    os.makedirs(out_dir, exist_ok=True)

    engines = list(results_by_engine.keys())
    if len(engines) < 2:
        print("\n  ⚠  Need >= 2 successful engines for comparison plots; "
              f"got {len(engines)}. Skipping.")
        return

    print(f"\n{'='*70}\nComparison plots across: {', '.join(engines)}\n{'='*70}")

    # ── Comparison Plot A: centroid position / velocity / speed ────────────
    print("Comparison A  –  Centroid trajectory, velocity, speed")
    with plt.style.context("default"):
        fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

        for i, eng in enumerate(engines):
            r = results_by_engine[eng]
            c = _engine_color(eng, i)
            axes[0].plot(r["times_f"], r["corrected_pos"], "-",
                         lw=1.6, color=c, label=eng, alpha=0.85)
            axes[1].plot(r["times_f"], r["corrected_vel"], "-",
                         lw=1.4, color=c, label=eng, alpha=0.85)
            axes[2].plot(r["times_f"], r["corrected_spd"], "-",
                         lw=1.4, color=c, label=eng, alpha=0.85)

        axes[0].axhline(0,             color="forestgreen", lw=0.7, ls=":")
        axes[0].axhline(POOL_LENGTH_M, color="red",         lw=1.0, ls=":",
                        label=f"Turn ({POOL_LENGTH_M:.0f} m)")
        axes[0].set_ylabel("Position (m)")
        axes[0].set_title(
            "Centroid Trajectory (Loop-Closure Corrected) – Engine Comparison")
        axes[0].legend(fontsize=9)
        axes[0].grid(alpha=0.6)

        axes[1].axhline(0, color="black", lw=0.8, ls=":")
        axes[1].set_ylabel("Velocity (m/s)")
        axes[1].set_title("Centroid Velocity – Engine Comparison")
        axes[1].legend(fontsize=9)
        axes[1].grid(alpha=0.6)

        axes[2].set_ylabel("Speed (m/s)")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_title("Centroid |Speed| – Engine Comparison")
        axes[2].legend(fontsize=9)
        axes[2].grid(alpha=0.6)

        plt.tight_layout()
        _save(fig, out_dir, "compA_centroid_trajectory.png")
        plt.close(fig)

    # ── Comparison Plot B: wrist speeds ─────────────────────────────────────
    print("Comparison B  –  Wrist speeds")
    with plt.style.context("default"):
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        for i, eng in enumerate(engines):
            r = results_by_engine[eng]
            c = _engine_color(eng, i)
            axes[0].plot(r["times_f"], r["lwrist_spd"], "-",
                         lw=1.3, color=c, label=eng, alpha=0.85)
            axes[1].plot(r["times_f"], r["rwrist_spd"], "-",
                         lw=1.3, color=c, label=eng, alpha=0.85)

        axes[0].set_ylabel("Speed (m/s)")
        axes[0].set_title("Left Wrist |Speed| – Engine Comparison")
        axes[0].legend(fontsize=9)
        axes[0].grid(alpha=0.6)

        axes[1].set_ylabel("Speed (m/s)")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_title("Right Wrist |Speed| – Engine Comparison")
        axes[1].legend(fontsize=9)
        axes[1].grid(alpha=0.6)

        plt.tight_layout()
        _save(fig, out_dir, "compB_wrist_speed.png")
        plt.close(fig)

    # ── Comparison Plot C: wrist velocity relative to centroid ──────────────
    print("Comparison C  –  Wrist velocity relative to centroid")
    with plt.style.context("default"):
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        for i, eng in enumerate(engines):
            r = results_by_engine[eng]
            c = _engine_color(eng, i)
            axes[0].plot(r["times_f"], r["rel_lwrist_vel"], "-",
                         lw=1.3, color=c, label=eng, alpha=0.85)
            axes[1].plot(r["times_f"], r["rel_rwrist_vel"], "-",
                         lw=1.3, color=c, label=eng, alpha=0.85)

        axes[0].axhline(0, color="black", lw=0.8, ls=":")
        axes[0].set_ylabel("Rel. Velocity (m/s)")
        axes[0].set_title(
            "Left Wrist Velocity Relative to Centroid – Engine Comparison")
        axes[0].legend(fontsize=9)
        axes[0].grid(alpha=0.6)

        axes[1].axhline(0, color="black", lw=0.8, ls=":")
        axes[1].set_ylabel("Rel. Velocity (m/s)")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_title(
            "Right Wrist Velocity Relative to Centroid – Engine Comparison")
        axes[1].legend(fontsize=9)
        axes[1].grid(alpha=0.6)

        plt.tight_layout()
        _save(fig, out_dir, "compC_wrist_relative_velocity.png")
        plt.close(fig)

    # ── Comparison Plot D: wrist position relative to centroid (time-series) ─
    print("Comparison D  –  Wrist position relative to centroid (smoothed)")
    with plt.style.context("default"):
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        for i, eng in enumerate(engines):
            r = results_by_engine[eng]
            c = _engine_color(eng, i)
            axes[0].plot(r["times_f"], r["lw_rel_smoothed"], "-",
                         lw=1.3, color=c, label=eng, alpha=0.85)
            axes[1].plot(r["times_f"], r["rw_rel_smoothed"], "-",
                         lw=1.3, color=c, label=eng, alpha=0.85)

        axes[0].axhline(0.0, color="black", ls="--", lw=0.8)
        axes[0].set_ylabel("Rel. distance (m)")
        axes[0].set_title(
            "Left Wrist Position Relative to Centroid – Engine Comparison")
        axes[0].legend(fontsize=9)
        axes[0].grid(alpha=0.35)

        axes[1].axhline(0.0, color="black", ls="--", lw=0.8)
        axes[1].set_ylabel("Rel. distance (m)")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_title(
            "Right Wrist Position Relative to Centroid – Engine Comparison")
        axes[1].legend(fontsize=9)
        axes[1].grid(alpha=0.35)

        plt.tight_layout()
        _save(fig, out_dir, "compD_wrist_relative_position.png")
        plt.close(fig)

    print(f"\n✓  Comparison plots saved to: {out_dir}")

    # ── Comparison Plot E: per-joint detection rate comparison ─────────────
    print("Comparison E  –  Per-joint detection rate by engine")
    # Use the joint labels from the first engine as the reference
    ref_engine = engines[0]
    ref_labels = results_by_engine[ref_engine].get("joint_labels", [])
    n_j = len(ref_labels)

    if n_j > 0:
        x = np.arange(n_j)
        width = 0.8 / len(engines)
        with plt.style.context("default"):
            fig, ax = plt.subplots(figsize=(16, 5))
            for i, eng in enumerate(engines):
                r = results_by_engine[eng]
                dr = r.get("det_rates", [])
                # Pad / trim to match reference length
                dr_aligned = [dr[j] if j < len(
                    dr) else 0.0 for j in range(n_j)]
                ax.bar(x + i * width - (len(engines) - 1) * width / 2,
                       dr_aligned, width, label=eng,
                       color=_engine_color(eng, i), alpha=0.80, edgecolor="white")
            ax.axhline(90, color="forestgreen", ls="--", lw=1, alpha=0.8,
                       label="90% target")
            ax.axhline(70, color="orange",      ls=":",  lw=1, alpha=0.8,
                       label="70% warning")
            ax.set_xticks(x)
            ax.set_xticklabels(ref_labels, rotation=45, ha="right", fontsize=9)
            ax.set_ylabel(f"Detection rate (%)")
            ax.set_ylim(0, 110)
            ax.set_title("Per-Joint Detection Rate Comparison\n"
                         f"(frames with confidence ≥ {CONF_THRESH})")
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.35)
            plt.tight_layout()
            _save(fig, out_dir, "compE_joint_detection_rate.png")
            plt.close(fig)

    # ── Comparison Plot F: per-joint jitter comparison ─────────────────────
    print("Comparison F  –  Per-joint trajectory jitter by engine")
    if n_j > 0:
        with plt.style.context("default"):
            fig, ax = plt.subplots(figsize=(16, 5))
            for i, eng in enumerate(engines):
                r = results_by_engine[eng]
                jt = r.get("jitter_per_joint", [])
                jt_aligned = [jt[j] if j < len(
                    jt) else np.nan for j in range(n_j)]
                ax.plot(range(n_j), jt_aligned, "o-",
                        color=_engine_color(eng, i), lw=1.8, ms=5,
                        label=eng, alpha=0.85)
            ax.set_xticks(range(n_j))
            ax.set_xticklabels(ref_labels, rotation=45, ha="right", fontsize=9)
            ax.set_ylabel("Median frame-to-frame displacement (px)")
            ax.set_title("Per-Joint Trajectory Smoothness Comparison\n"
                         "(lower = more stable tracking)")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.35)
            plt.tight_layout()
            _save(fig, out_dir, "compF_joint_jitter.png")
            plt.close(fig)

    # ── Comparison Plot G: speed histogram per engine ──────────────────────
    print("Comparison G  –  Swimmer speed distribution by engine")
    with plt.style.context("default"):
        fig, ax = plt.subplots(figsize=(10, 5))
        bins = np.linspace(0, VEL_MAX, 40)
        for i, eng in enumerate(engines):
            r = results_by_engine[eng]
            spd = r["corrected_spd"]
            spd = spd[np.isfinite(spd) & (spd >= 0)]
            ax.hist(spd, bins=bins, density=True, histtype="stepfilled",
                    color=_engine_color(eng, i), alpha=0.35, label=f"{eng}")
            ax.hist(spd, bins=bins, density=True, histtype="step",
                    color=_engine_color(eng, i), lw=1.8)
            ax.axvline(float(np.median(spd)), color=_engine_color(eng, i),
                       ls="--", lw=1.5,
                       label=f"{eng} median {np.median(spd):.2f} m/s")
        ax.set_xlabel("Swimmer Speed (m/s)")
        ax.set_ylabel("Density")
        ax.set_title("Swimmer Speed Distribution – Engine Comparison\n"
                     "(tighter, more realistic distribution = better)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.35)
        plt.tight_layout()
        _save(fig, out_dir, "compG_speed_histogram.png")
        plt.close(fig)

    # ── Comparison Plot H: key-metric bar summary ──────────────────────────
    print("Comparison H  –  Key-metric summary bar chart")
    metrics = {
        "Mean key-joint\nconfidence (↑)": [
            results_by_engine[e].get("mean_conf_key", np.nan) for e in engines],
        "Key-joint\ndetection rate % (↑)": [
            results_by_engine[e].get("det_rate_key", np.nan) / 100.0 for e in engines],
        "Mean centroid\nspeed m/s": [
            float(np.nanmean(results_by_engine[e]["corrected_spd"])) for e in engines],
        "Speed std m/s\n(lower=stable, ↓)": [
            float(np.nanstd(results_by_engine[e]["corrected_spd"])) for e in engines],
    }
    norm_metrics: dict[str, list] = {}
    for k, vals in metrics.items():
        arr = np.array(vals, dtype=float)
        rng = np.nanmax(arr) - np.nanmin(arr)
        if rng > 0:
            norm_metrics[k] = list((arr - np.nanmin(arr)) / rng)
        else:
            norm_metrics[k] = [0.5] * len(vals)

    n_metrics = len(metrics)
    x_m = np.arange(n_metrics)
    width_m = 0.7 / len(engines)

    with plt.style.context("default"):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left: raw values
        for i, eng in enumerate(engines):
            raw_vals = [metrics[k][i] for k in metrics]
            axes[0].bar(x_m + i * width_m - (len(engines) - 1) * width_m / 2,
                        raw_vals, width_m, label=eng,
                        color=_engine_color(eng, i), alpha=0.80, edgecolor="white")
        axes[0].set_xticks(x_m)
        axes[0].set_xticklabels(list(metrics.keys()), fontsize=10)
        axes[0].set_title("Raw Key Metrics by Engine")
        axes[0].legend(fontsize=9)
        axes[0].grid(axis="y", alpha=0.35)

        # Right: normalised 0-1
        for i, eng in enumerate(engines):
            norm_vals = [norm_metrics[k][i] for k in norm_metrics]
            axes[1].bar(x_m + i * width_m - (len(engines) - 1) * width_m / 2,
                        norm_vals, width_m, label=eng,
                        color=_engine_color(eng, i), alpha=0.80, edgecolor="white")
        axes[1].set_xticks(x_m)
        axes[1].set_xticklabels(list(metrics.keys()), fontsize=10)
        axes[1].set_title(
            "Normalised Key Metrics (0 = worst, 1 = best\nwithin shown engines)")
        axes[1].set_ylim(0, 1.15)
        axes[1].legend(fontsize=9)
        axes[1].grid(axis="y", alpha=0.35)

        fig.suptitle("Engine Performance Summary",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        _save(fig, out_dir, "compH_metric_summary.png")
        plt.close(fig)

    # ── Comparison Plot I: radar / spider chart ────────────────────────────
    print("Comparison I  –  Radar chart (performance spider)")
    # Five axes: Confidence, Detection Rate, Smoothness (inv jitter), Speed Consistency (inv std), Mean Speed
    spider_labels = [
        "Key-Joint\nConfidence",
        "Detection\nRate",
        "Tracking\nSmoothness",
        "Speed\nConsistency",
        "Mean Speed\n(m/s)",
    ]
    N_sp = len(spider_labels)
    angles = np.linspace(0, 2 * np.pi, N_sp, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    raw_spider: dict[str, list] = {}
    for eng in engines:
        r = results_by_engine[eng]
        conf_v = r.get("mean_conf_key", 0.5)
        det_v = r.get("det_rate_key", 50.0) / 100.0
        jit_v = r.get("jitter_key", 1.0)
        spd_arr = np.array(r["corrected_spd"])
        spd_std = float(np.nanstd(spd_arr))
        spd_mean = float(np.nanmean(spd_arr))
        raw_spider[eng] = [conf_v, det_v, jit_v, spd_std, spd_mean]

    # Normalise each axis 0-1 (for jitter and speed std, lower is better → invert)
    raw_arr = np.array([raw_spider[e] for e in engines], dtype=float)
    norm_arr = np.zeros_like(raw_arr)
    invert_axes = {2, 3}   # smoothness (jitter), speed std: lower=better
    for col in range(N_sp):
        col_vals = raw_arr[:, col]
        lo, hi = np.nanmin(col_vals), np.nanmax(col_vals)
        rng = hi - lo if (hi - lo) > 0 else 1.0
        normalised = (col_vals - lo) / rng
        if col in invert_axes:
            normalised = 1.0 - normalised
        norm_arr[:, col] = normalised

    with plt.style.context("default"):
        fig, ax = plt.subplots(figsize=(8, 8),
                               subplot_kw=dict(polar=True))
        for i, eng in enumerate(engines):
            vals = norm_arr[i].tolist()
            vals += vals[:1]
            ax.plot(angles, vals, "-o", lw=2,
                    color=_engine_color(eng, i), label=eng)
            ax.fill(angles, vals, alpha=0.12, color=_engine_color(eng, i))

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(spider_labels, fontsize=10)
        ax.set_yticks([0.25, 0.50, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"],
                           fontsize=7, color="grey")
        ax.set_ylim(0, 1)
        ax.set_title("Engine Performance Radar\n"
                     "(all axes: higher = better, normalised within shown engines)",
                     fontsize=12, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=10)
        plt.tight_layout()
        _save(fig, out_dir, "compI_radar_chart.png")
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Top-level dispatcher
# ─────────────────────────────────────────────────────────────────────────────

ALL_ENGINES = ["vitpose", "yolo226l-pose", "mediapipe"]


def main(csv_path: str) -> None:

    print(f"\nLoading: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    print(f"  Total rows: {len(df_raw)}")

    if POSE_ENGINE == "all":
        results_by_engine = {}
        for engine in ALL_ENGINES:
            try:
                results_by_engine[engine] = process_engine(
                    df_raw, engine, GRAPHS_DIR)
            except Exception as exc:
                print(f"\n  ⚠  Skipping engine '{engine}': {exc}")

        if not results_by_engine:
            print("\n✗  No engines processed successfully.")
            return

        run_comparison_plots(results_by_engine, GRAPHS_DIR)

        print(f"\n✓  All done. Per-engine plots in "
              f"{GRAPHS_DIR}/<engine>/, comparisons in "
              f"{GRAPHS_DIR}/comparison/")
    else:
        process_engine(df_raw, POSE_ENGINE, GRAPHS_DIR)
        print(
            f"\n✓  All plots saved to: {os.path.join(GRAPHS_DIR, POSE_ENGINE)}")


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
