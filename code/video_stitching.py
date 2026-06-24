# =========================================================
# AUTO-DEPENDENCY INSTALL (SAFE BOOTSTRAP)
# =========================================================

from scipy.signal import butter, sosfilt, find_peaks
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import librosa
import numpy as np
import ffmpeg
import logging
import json
import subprocess
import sys
import os

from pipeline.main import SWIMMING_STYLE


def install_if_missing(pkg, import_name=None):
    import_name = import_name or pkg
    try:
        __import__(import_name)
    except ImportError:
        print(f"[SETUP] Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


install_if_missing("opencv-python", "cv2")
install_if_missing("numpy")
install_if_missing("ffmpeg-python", "ffmpeg")
install_if_missing("librosa")
install_if_missing("scipy")
install_if_missing("tqdm")


# =========================================================
# IMPORTS
# =========================================================


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

log = logging.getLogger("swim_pipeline")


# =========================================================
# INPUT CONFIG
# =========================================================

PARTICIPANT_IDS = ["P049"]
SWIMMING_STYLES = ["Breaststroke"]


PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)

for PARTICIPANT_ID, SWIMMING_STYLE in zip(PARTICIPANT_IDS, SWIMMING_STYLES):
    INPUT_SESSIONS: dict[str, dict[str, str]] = {
        f"{PARTICIPANT_ID}_{SWIMMING_STYLE}": {
            "ABOVE_WATER": os.path.join(PROJECT_ROOT, "videos", "above", f"Top_{SWIMMING_STYLE}_{PARTICIPANT_ID}.MP4"),
            "UNDER_WATER": os.path.join(PROJECT_ROOT, "videos", "under", f"Bottom_{SWIMMING_STYLE}_{PARTICIPANT_ID}.MP4")
        }
    }

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
TARGET_FPS = 120
SHOW_TIMESTAMPS = False


# =========================================================
# UTILITIES
# =========================================================

def parse_time(t):
    return datetime.fromisoformat(t.replace("Z", ""))


def probe(video_path):
    return ffmpeg.probe(video_path)


def get_meta(video_path):
    data = probe(video_path)

    fmt = data["format"]
    stream = next(s for s in data["streams"] if s["codec_type"] == "video")

    def fps(fr):
        if "/" in fr:
            a, b = fr.split("/")
            return float(a) / float(b)
        return float(fr)

    return {
        "VIDEO_PATH": video_path,
        "FPS": fps(stream.get("r_frame_rate", "0")),
        "DURATION": float(fmt.get("duration", 0)),
        "CREATED": fmt.get("tags", {}).get("creation_time")
    }


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# =========================================================
# ALIGNMENT
# =========================================================


def compute_alignment(meta):
    out = {}

    for session, cams in tqdm(meta.items(), desc="Computing alignment"):
        a = cams["ABOVE_WATER"]
        u = cams["UNDER_WATER"]

        ta = parse_time(a["CREATED"])
        tu = parse_time(u["CREATED"])

        C = (tu - ta).total_seconds()

        out[session] = {"C_i": C}

        log.info(f"{session} sync offset: {C:.2f}s")

    return out


# =========================================================
# EXPORT PIPELINE
# =========================================================

def export_session(above_path, under_path, lag, out_dir):

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    log.info("Probing videos...")

    pa = probe(above_path)
    pu = probe(under_path)

    dur_a = float(pa["format"]["duration"])
    dur_u = float(pu["format"]["duration"])

    width = min(
        int(next(s for s in pa["streams"]
            if s["codec_type"] == "video")["width"]),
        int(next(s for s in pu["streams"]
            if s["codec_type"] == "video")["width"])
    )

    # =====================================================
    # AUDIO EXTRACTION
    # =====================================================

    audio_tmp = Path(out_dir) / "audio.wav"

    (
        ffmpeg
        .input(above_path)
        .output(str(audio_tmp), ac=1, ar=16000)
        .overwrite_output()
        .run(quiet=True)
    )

    y, sr = librosa.load(audio_tmp, sr=16000)

    sos = butter(6, [2000, 5000], btype="bandpass", fs=sr, output="sos")
    y = sosfilt(sos, y)

    rms = librosa.feature.rms(y=y)[0] # type: ignore
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

    thresh = np.mean(rms) + 3.5 * np.std(rms)
    peaks, _ = find_peaks(rms, height=thresh)

    if len(peaks) < 2:
        raise RuntimeError("Whistle detection failed")

    t0, t1 = float(times[peaks[0]]), float(times[peaks[-1]])

    log.info(f"Whistles: {t0:.2f}s → {t1:.2f}s")

    # =====================================================
    # SYNC SHIFT
    # =====================================================

    if lag >= 0:
        a0, a1 = t0, t1
        u0, u1 = t0 - lag, t1 - lag
    else:
        a0, a1 = t0 + lag, t1 + lag
        u0, u1 = t0, t1

    def clamp(s, e, m):
        s, e = max(0, s), min(m, e)
        if e <= s:
            raise RuntimeError("Invalid trim window")
        return s, e

    a0, a1 = clamp(a0, a1, dur_a)
    u0, u1 = clamp(u0, u1, dur_u)

    # =====================================================
    # STREAM BUILDER
    # =====================================================

    def build(path, s, e):
        stream = (
            ffmpeg
            .input(path)
            .filter("trim", start=s, end=e)
            .filter("setpts", "PTS-STARTPTS")
            .filter("fps", fps=TARGET_FPS)
            .filter("scale", width, -2)
            .filter("format", "yuv420p")
        )

        if SHOW_TIMESTAMPS:
            stream = stream.filter(
                "drawtext",
                text="%{pts\\:hms}",
                x=20,
                y=20,
                fontsize=24,
                fontcolor="white"
            )

        return stream

    above = build(above_path, a0, a1)
    under = build(under_path, u0, u1)

    stacked = ffmpeg.filter([above, under], "vstack")

    # =====================================================
    # OUTPUT FILES
    # =====================================================

    out_a = Path(out_dir) / "above.mp4"
    out_u = Path(out_dir) / "under.mp4"
    out_s = Path(out_dir) / f"Stacked_{SWIMMING_STYLE}_{PARTICIPANT_ID}.mp4"

    def write(stream, path):
        (
            ffmpeg
            .output(stream, str(path), vcodec="libx264", crf=18, preset="fast")
            .overwrite_output()
            .run(quiet=True)
        )

    write(above, out_a)
    write(under, out_u)
    write(stacked, out_s)

    log.info(f"Export complete → {out_dir}")

    return {
        "above": str(out_a),
        "under": str(out_u),
        "stacked": str(out_s),
        "whistles": {
            "start": float(t0),
            "end": float(t1),
            "duration": float(t1 - t0)
        },
        "trim": {
            "above": [float(a0), float(a1)],
            "under": [float(u0), float(u1)]
        }
    }


# =========================================================
# MAIN PIPELINE
# =========================================================

def run():

    log.info("Starting swim pipeline")

    # =====================================================
    # INPUT DISPLAY
    # =====================================================

    print("\nINPUT CONFIG")
    print("=" * 60)
    print(json.dumps(INPUT_SESSIONS, indent=2))
    print("=" * 60)

    # =====================================================
    # METADATA EXTRACTION (TQDM)
    # =====================================================

    meta = {}

    for session, cams in tqdm(INPUT_SESSIONS.items(), desc="Extracting metadata"):
        meta[session] = {
            cam: get_meta(path)
            for cam, path in cams.items()
        }

    # =====================================================
    # ALIGNMENT
    # =====================================================

    alignment = compute_alignment(meta)

    # =====================================================
    # EXPORT (TQDM)
    # =====================================================

    outputs = {}

    for session, cams in tqdm(INPUT_SESSIONS.items(), desc="Exporting sessions"):
        out_dir = f"{OUTPUT_DIR}/{session.replace(' ', '_')}"

        session_output = export_session(
            cams["ABOVE_WATER"],
            cams["UNDER_WATER"],
            alignment[session]["C_i"],
            out_dir
        )

        outputs[session] = session_output

    # =====================================================
    # FINAL OUTPUT (SINGLE JSON FILE)
    # =====================================================

    report = {
        "input_sessions": INPUT_SESSIONS,
        "metadata": meta,
        "alignment": alignment,
        "outputs": outputs,
    }

    session_summary = {}

    for session in outputs:
        session_summary[session] = {
            "whistle_duration": outputs[session]["whistles"]["duration"],
            "sync_offset": alignment[session]["C_i"],
            "trim_above": outputs[session]["trim"]["above"],
            "trim_under": outputs[session]["trim"]["under"]
        }

    report["summary"] = session_summary

    output_json_path = Path(OUTPUT_DIR) / \
        session.replace(' ', '_') / "session_report.json"

    save_json(report, output_json_path)

    print("\n✔ REPORT SAVED")
    print("=" * 60)
    print(f"JSON FILE: {output_json_path}")
    print("=" * 60)


if __name__ == "__main__":
    run()
