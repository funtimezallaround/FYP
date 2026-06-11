import os
import cv2
import numpy as np
import torch
import PIL.Image
import subprocess
import pandas as pd
import threading
import queue
from huggingface_hub import snapshot_download
from ultralytics import YOLO
from transformers import AutoProcessor, VitPoseForPoseEstimation
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


# ============ SETUP =====================================================================================

# Configuration
PARTICIPANT_ID = "P041"
SWIMMING_STYLE = "Breaststroke"

START_FRAME_IDX = 0
NUM_FRAMES = -1  # Set to -1 to process all frames
FPS = 120
BATCH_SIZE = 64  # Increased to feed GPU more regularly and maximize throughput
MARKER_REAL_DIST_M = 2.5
YOLO_CONF = 0.3

# Paths
PROJECT_ROOT = os.path.dirname(os.path.join(os.path.abspath(__file__), ".."))

OUT_DIR = f"output/{SWIMMING_STYLE}_{PARTICIPANT_ID}"
YOLO_MARKER_MODEL_PATH = "models/marker_detector.pt"
YOLO_PERSON_MODEL_PATH = "models/yolov8s.pt"
POSE_MODEL_DIR = "models/vitpose-plus-large"
FRAMES_DIR = f"../frames/under/Bottom_{SWIMMING_STYLE}_{PARTICIPANT_ID}/"
VIDEO_DIR = "../videos/under/"
SMOOTH_VIDEO_PATH = os.path.join(
    OUT_DIR, f"ego_motion_{SWIMMING_STYLE.lower()}_{PARTICIPANT_ID}.mp4")
TRACKING_CSV_PATH = os.path.join(
    OUT_DIR, f"tracking_records_{SWIMMING_STYLE.lower()}_{PARTICIPANT_ID}.csv")

# Standard COCO Pose Keypoint Names
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# =========================================================================================================


def get_centroid(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def put_text_bg(img, text, pos, color, scale=0.7, alpha=0.5):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    x1, y1 = int(max(0, x - 5)), int(max(0, y - th - 5))
    x2, y2 = int(min(img.shape[1], x + tw + 5)
                 ), int(min(img.shape[0], y + baseline + 5))

    if x2 > x1 and y2 > y1:
        overlay = img[y1:y2, x1:x2].copy()
        cv2.rectangle(overlay, (0, 0), (x2 - x1, y2 - y1), (0, 0, 0), -1)
        cv2.addWeighted(
            overlay, alpha, img[y1:y2, x1:x2], 1 - alpha, 0, img[y1:y2, x1:x2])

    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def extract_frames_from_video(frames_dir):
    os.makedirs(frames_dir, exist_ok=True)

    base_name = os.path.basename(os.path.normpath(frames_dir))
    candidate_videos = [
        os.path.join(VIDEO_DIR, f"{base_name}.mp4"),
        os.path.join(os.path.dirname(frames_dir), f"{base_name}.mp4"),
    ]

    video_path = next((p for p in candidate_videos if os.path.exists(p)), None)
    if video_path is None:
        print(f"Warning: No corresponding video found for {frames_dir}")
        return False

    print(f"Extracting frames from {video_path} to {frames_dir}...")
    output_dir = frames_dir
    ffmpeg_cmd = f'ffmpeg -i "{video_path}" -q:v 2 -start_number 0 "{output_dir}/frame_%05d.jpg"'
    result = os.system(ffmpeg_cmd)
    if result != 0:
        print(f"Warning: Failed to extract frames from {video_path}")
        return False

    return True


def load_pose_model(device):
    os.makedirs(POSE_MODEL_DIR, exist_ok=True)

    if not os.path.exists(os.path.join(POSE_MODEL_DIR, "config.json")):
        print(f"Downloading ViTPose model to {POSE_MODEL_DIR}...")
        snapshot_download(
            repo_id="usyd-community/vitpose-plus-large",
            local_dir=POSE_MODEL_DIR,
        )

    pose_image_processor = AutoProcessor.from_pretrained(
        POSE_MODEL_DIR,
        local_files_only=True,
    )
    pose_model = VitPoseForPoseEstimation.from_pretrained(
        POSE_MODEL_DIR,
        device_map=device,
        local_files_only=True,
    )
    return pose_image_processor, pose_model


def batch_producer(frame_paths, batch_size, out_queue):
    def read_img(path):
        f = cv2.imread(path)
        return f

    # Use a ThreadPool to speed up the disk I/O of multiple images
    with ThreadPoolExecutor(max_workers=12) as executor:
        for b in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[b: b + batch_size]

            # Read frames concurrently (cv2 releases the GIL during I/O)
            frames = list(executor.map(read_img, batch_paths))
            batch_frames = [f for f in frames if f is not None]

            if batch_frames:
                # This will block if the queue is full, pausing preparation
                out_queue.put(batch_frames)

    # Send a sentinel value to tell the main thread we are done
    out_queue.put(None)


def viz_writer(proc, q, error_holder=None):
    """Background thread to handle pushing frames into FFmpeg"""
    try:
        while True:
            item = q.get()
            if item is None:
                break

            if proc.poll() is not None:
                err = ""
                if proc.stderr is not None:
                    err = proc.stderr.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"ffmpeg exited early:\n{err}")

            frame = np.ascontiguousarray(item)
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)

            try:
                if proc.stdin is not None:
                    proc.stdin.write(frame.tobytes())
            except BrokenPipeError as exc:
                err = ""
                if proc.stderr is not None:
                    err = proc.stderr.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"Video writer pipe closed unexpectedly.\nffmpeg stderr:\n{err}"
                ) from exc
    except Exception as exc:
        if error_holder is not None:
            error_holder.append(exc)
    finally:
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass

        try:
            if proc.stderr is not None:
                proc.stderr.close()
        except Exception:
            pass

        try:
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass


def main():
    # --- Configuration ---
    os.makedirs(OUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load Models ---
    marker_model = YOLO(YOLO_MARKER_MODEL_PATH)
    person_model = YOLO(YOLO_PERSON_MODEL_PATH)

    try:
        pose_image_processor, pose_model = load_pose_model(device)
    except Exception as e:
        print(f"Error loading ViTPose: {e}")
        return

    # --- Setup Video Writer and File Paths ---
    ffmpeg_process = None
    viz_queue = None
    viz_thread = None
    viz_errors = []
    target_h, target_w = 0, 0
    tracking_records = []

    if not os.path.exists(FRAMES_DIR):
        print(
            f"Warning: Frames directory {FRAMES_DIR} does not exist. Trying to extract frames from video...")
        if not extract_frames_from_video(FRAMES_DIR):
            return

    frame_range = range(NUM_FRAMES) if NUM_FRAMES > 0 else range(
        START_FRAME_IDX, START_FRAME_IDX + len(os.listdir(FRAMES_DIR)))
    affix = "frame_" if os.path.exists(os.path.join(
        FRAMES_DIR, f"frame_{START_FRAME_IDX:05d}.jpg")) else ""
    frame_paths = [os.path.join(
        FRAMES_DIR, f"{affix}{START_FRAME_IDX + i:05d}.jpg") for i in frame_range]

    global_camera_x = 0.0
    prev_markers_x = []
    prev_px_per_m = None
    prev_delta_x = 0.0

    # --- Setup the Queue and Producer Thread ---
    MAX_QUEUED_BATCHES = 4  # Queue will hold a max of 4 batches in RAM
    frame_queue = queue.Queue(maxsize=MAX_QUEUED_BATCHES)

    producer = threading.Thread(
        target=batch_producer,
        args=(frame_paths, BATCH_SIZE, frame_queue),
        daemon=True
    )
    producer.start()

    total_batches = (len(frame_paths) + BATCH_SIZE - 1) // BATCH_SIZE

    # --- Processing Loop ---
    batch_idx = 0
    with tqdm(total=total_batches, desc="Processing video", unit="batch") as pbar:
        while True:
            # This blocks until a batch is ready
            batch_frames = frame_queue.get()

            # Check for the sentinel value marking the end of the video
            if batch_frames is None:
                break

            img_h, img_w = batch_frames[0].shape[:2]

            # 1. Run YOLO detectors on the full batch concurrently
            marker_results = marker_model.predict(
                batch_frames, conf=YOLO_CONF, verbose=False)
            person_results = person_model.predict(
                batch_frames, conf=0.15, verbose=False)

            # 2. Prepare lists for batched ViTPose execution
            vitpose_pil_images = []
            vitpose_boxes_list = []
            # Store original YOLO boxes mapped to frame index
            swimmer_hboxes = [None] * len(batch_frames)
            # Keep track of which frames actually have a swimmer
            valid_frame_indices = set()

            # Process Swimmer (largest person box)
            for idx_in_batch, (frame, p_res) in enumerate(zip(batch_frames, person_results)):
                largest_area = -1.0
                person_box = None

                # Find the largest person (swimmer)
                for box in p_res.boxes:
                    if int(box.cls) == 0 and float(box.conf) > 0.15:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        area = (x2 - x1) * (y2 - y1)
                        if area > largest_area:
                            largest_area = area
                            person_box = [x1, y1, x2, y2]

                # If we found a swimmer, crop it directly using OpenCV and prep only the crop for ViTPose
                if person_box is not None:
                    x1, y1, x2, y2 = [int(v) for v in person_box]
                    swimmer_hboxes[idx_in_batch] = (x1, y1, x2, y2)

                    # Constrain crop coordinates to frame boundaries
                    x1_c, y1_c = max(0, x1), max(0, y1)
                    x2_c, y2_c = min(img_w, x2), min(img_h, y2)

                    crop = frame[y1_c:y2_c, x1_c:x2_c]
                    if crop.size > 0:
                        crop_h, crop_w = crop.shape[:2]
                        # ViTPose expects bounding box coordinates relative to the input image passed to it.
                        v_box = np.array([[0, 0, crop_w, crop_h]])
                        vitpose_boxes_list.append(v_box)

                        # Convert only the small crop to PIL (massively faster than converting full BGR frame)
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        vitpose_pil_images.append(
                            PIL.Image.fromarray(crop_rgb))
                        valid_frame_indices.add(idx_in_batch)

            # 3. Run ViTPose on ALL valid crops simultaneously
            batched_pose_results = []

            if vitpose_pil_images:  # Only execute if at least one swimmer was found in the batch
                inputs = pose_image_processor(
                    images=vitpose_pil_images,
                    boxes=vitpose_boxes_list,
                    return_tensors="pt"
                ).to(device)

                if pose_model.config.backbone_config.num_experts > 1:
                    inputs["dataset_index"] = torch.tensor(
                        [0] * len(inputs["pixel_values"])).to(device)

                with torch.no_grad():
                    outputs = pose_model(**inputs)

                batched_pose_results = pose_image_processor.post_process_pose_estimation(
                    outputs, boxes=vitpose_boxes_list
                )

            # 4. Map results back, complete calculations, and pipe to FFmpeg (Properly Indented Frame Loop)
            pose_results_by_frame = {
                frame_i: batched_pose_results[local_i]
                for local_i, frame_i in enumerate(sorted(valid_frame_indices))
            }

            for idx_in_batch, (frame, m_res) in enumerate(zip(batch_frames, marker_results)):
                frame_idx = START_FRAME_IDX + \
                    (batch_idx * BATCH_SIZE) + idx_in_batch

                swimmer_centroid, swimmer_lwrist, swimmer_rwrist = None, None, None
                swimmer_lshoulder, swimmer_rshoulder = None, None
                swimmer_kpts, swimmer_kpt_scores = None, None
                swimmer_hbox = swimmer_hboxes[idx_in_batch]
                current_marker_detections = []

                # Process Markers
                for box in m_res.boxes:
                    if int(box.cls) == 0:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cx, cy = (x1 + x2) / 2, y2
                        current_marker_detections.append(
                            {"pt": (cx, cy), "box": (x1, y1, x2, y2, cx, cy)})

                # Process Keypoints ONLY if this frame was sent to ViTPose
                if idx_in_batch in pose_results_by_frame:
                    frame_pose = pose_results_by_frame[idx_in_batch]

                    if len(frame_pose) > 0:
                        person_pose = frame_pose[0]
                        kpts = person_pose["keypoints"].cpu().numpy()
                        scores = person_pose["scores"].cpu().numpy()

                        # --- CRITICAL MAP BACK STEP ---
                        # Translate keypoint coordinates back from Crop-Space to Original Frame-Space
                        sx1, sy1, _, _ = swimmer_hbox
                        kpts[:, 0] += sx1
                        kpts[:, 1] += sy1

                        swimmer_kpts, swimmer_kpt_scores = kpts, scores

                        # Map joints
                        if len(kpts) > 12 and scores[11] > 0.1 and scores[12] > 0.1:
                            lh, rh = kpts[11], kpts[12]
                            swimmer_centroid = (
                                int((lh[0] + rh[0]) / 2), int((lh[1] + rh[1]) / 2))

                        if len(kpts) > 10:
                            if scores[9] > 0.3:
                                swimmer_lwrist = tuple(kpts[9])
                            if scores[10] > 0.3:
                                swimmer_rwrist = tuple(kpts[10])

                        if len(kpts) > 6:
                            if scores[5] > 0.3:
                                swimmer_lshoulder = tuple(kpts[5])
                            if scores[6] > 0.3:
                                swimmer_rshoulder = tuple(kpts[6])

                # Visualizing the Tracker Frame
                out_frame = frame.copy()
                if swimmer_hbox is not None:
                    sx1, sy1, sx2, sy2 = swimmer_hbox
                    cv2.rectangle(out_frame, (sx1, sy1),
                                  (sx2, sy2), (255, 255, 0), 5)

                if swimmer_kpts is not None and swimmer_kpt_scores is not None:
                    for i, (kp, score) in enumerate(zip(swimmer_kpts, swimmer_kpt_scores)):
                        if score > 0.1:
                            color = (0, 255, 0) if i in [9, 10] and score > 0.3 else (
                                (0, 100, 0) if i in [9, 10] else (0, 0, 255))
                            cv2.circle(
                                out_frame, (int(kp[0]), int(kp[1])), 6, color, -1)

                # Ego-Motion Displacement Calculation
                current_markers_x = [m["pt"][0]
                                     for m in current_marker_detections]
                delta_x = 0.0

                if prev_markers_x and current_markers_x:
                    valid_deltas = [diff for cx in current_markers_x
                                    for diff in [[cx - px for px in prev_markers_x][np.argmin([abs(cx - px) for px in prev_markers_x])]]
                                    if abs(diff) < 50.0]
                    delta_x = float(np.median(valid_deltas)
                                    ) if valid_deltas else prev_delta_x
                elif prev_markers_x and not current_markers_x:
                    delta_x = prev_delta_x

                global_camera_x -= delta_x
                prev_delta_x = delta_x

                sorted_markers = sorted(
                    current_marker_detections, key=lambda m: m["pt"][0])
                px_per_m = prev_px_per_m if prev_px_per_m is not None else 1.0

                if len(sorted_markers) >= 2:
                    A_pt = np.array(sorted_markers[0]["pt"])
                    B_pt = np.array(sorted_markers[-1]["pt"])
                    dist_px = np.linalg.norm(B_pt - A_pt)
                    physical_dist = (len(sorted_markers) -
                                     1) * MARKER_REAL_DIST_M

                    if physical_dist > 0:
                        px_per_m = dist_px / physical_dist
                        prev_px_per_m = px_per_m

                    cv2.line(out_frame, tuple(A_pt.astype(int)), tuple(
                        B_pt.astype(int)), (255, 0, 255), 2, cv2.LINE_AA)

                for md in current_marker_detections:
                    cv2.circle(out_frame, tuple(
                        np.array(md["pt"]).astype(int)), 6, (0, 255, 0), -1)

                # Calculations & Graphing
                if swimmer_centroid is not None:
                    virtual_x = swimmer_centroid[0] + global_camera_x
                    global_pos_m = virtual_x / px_per_m if px_per_m > 0 else 0.0
                    lwrist_pos_m, rwrist_pos_m = np.nan, np.nan

                    if len(sorted_markers) >= 2:
                        AB = B_pt - A_pt
                        AB_len = np.linalg.norm(AB)
                        if AB_len > 0:
                            AB_unit = AB / AB_len
                            S_pt = np.array(swimmer_centroid)
                            proj_S = A_pt + \
                                np.dot(S_pt - A_pt, AB_unit) * AB_unit

                            if swimmer_lwrist is not None:
                                LW_pt = np.array(swimmer_lwrist)
                                lwrist_pos_m = (
                                    LW_pt[0] + global_camera_x) / px_per_m if px_per_m > 0 else 0.0
                                proj_LW = A_pt + \
                                    np.dot(LW_pt - A_pt, AB_unit) * AB_unit
                                cv2.line(out_frame, tuple(LW_pt.astype(int)), tuple(
                                    proj_LW.astype(int)), (0, 255, 0), 1, cv2.LINE_AA)

                            if swimmer_rwrist is not None:
                                RW_pt = np.array(swimmer_rwrist)
                                rwrist_pos_m = (
                                    RW_pt[0] + global_camera_x) / px_per_m if px_per_m > 0 else 0.0
                                proj_RW = A_pt + \
                                    np.dot(RW_pt - A_pt, AB_unit) * AB_unit
                                cv2.line(out_frame, tuple(RW_pt.astype(int)), tuple(
                                    proj_RW.astype(int)), (0, 255, 0), 1, cv2.LINE_AA)
                    else:
                        if swimmer_lwrist is not None:
                            lwrist_pos_m = (
                                swimmer_lwrist[0] + global_camera_x) / px_per_m if px_per_m > 0 else 0.0
                        if swimmer_rwrist is not None:
                            rwrist_pos_m = (
                                swimmer_rwrist[0] + global_camera_x) / px_per_m if px_per_m > 0 else 0.0

                    S_px = tuple(np.array(swimmer_centroid).astype(int))
                    cv2.circle(out_frame, S_px, 8,
                               (255, 191, 0), -1, cv2.LINE_AA)
                    put_text_bg(out_frame, "Swimmer",
                                (S_px[0] + 8, S_px[1] - 15), (255, 191, 0))

                    bar_len = int(px_per_m)
                    bar_y = img_h - 40
                    if bar_len > 0:
                        cv2.arrowedLine(out_frame, (40, bar_y), (40 + bar_len,
                                        bar_y), (0, 255, 255), 3, tipLength=15.0/bar_len)
                        cv2.arrowedLine(out_frame, (40 + bar_len, bar_y),
                                        (40, bar_y), (0, 255, 255), 3, tipLength=15.0/bar_len)
                        put_text_bg(out_frame, "1 m", (40 + bar_len //
                                    2 - 20, bar_y - 15), (0, 255, 255))

                    put_text_bg(
                        out_frame, f"Scale: {px_per_m:.1f} px/m", (40, 40), (0, 255, 255), scale=0.9)
                    put_text_bg(
                        out_frame, f"Global Pos: {global_pos_m:.2f} m", (40, 80), (0, 165, 255), scale=0.9)

                # Create the core tracking record dictionary
                record = {
                    "frame_idx": frame_idx,
                    "time_s": ((batch_idx * BATCH_SIZE) + idx_in_batch) / FPS,
                    "pos_m": float(global_pos_m) if swimmer_centroid is not None else None,
                    "lwrist_pos_m": float(lwrist_pos_m) if (swimmer_centroid is not None and not np.isnan(lwrist_pos_m)) else None,
                    "rwrist_pos_m": float(rwrist_pos_m) if (swimmer_centroid is not None and not np.isnan(rwrist_pos_m)) else None,
                    "px_per_m": float(px_per_m),
                    "selection_mode": "auto",
                    # Legacy coordinate mapping (for backward compatibility)
                    "lwrist_x": swimmer_lwrist[0] if swimmer_lwrist else None,
                    "lwrist_y": swimmer_lwrist[1] if swimmer_lwrist else None,
                    "rwrist_x": swimmer_rwrist[0] if swimmer_rwrist else None,
                    "rwrist_y": swimmer_rwrist[1] if swimmer_rwrist else None,
                    "lshoulder_x": swimmer_lshoulder[0] if swimmer_lshoulder else None,
                    "lshoulder_y": swimmer_lshoulder[1] if swimmer_lshoulder else None,
                    "rshoulder_x": swimmer_rshoulder[0] if swimmer_rshoulder else None,
                    "rshoulder_y": swimmer_rshoulder[1] if swimmer_rshoulder else None
                }

                # Pre-populate all COCO keypoint columns with None to maintain clean column structures
                for name in COCO_KEYPOINTS:
                    record[f"{name}_x"] = None
                    record[f"{name}_y"] = None
                    record[f"{name}_score"] = None

                # Populate dynamic COCO keypoint columns with actual tracking data if available
                if swimmer_kpts is not None and swimmer_kpt_scores is not None:
                    for i, (kp, score) in enumerate(zip(swimmer_kpts, swimmer_kpt_scores)):
                        name = COCO_KEYPOINTS[i] if i < len(
                            COCO_KEYPOINTS) else f"keypoint_{i}"
                        record[f"{name}_x"] = float(kp[0])
                        record[f"{name}_y"] = float(kp[1])
                        record[f"{name}_score"] = float(score)

                tracking_records.append(record)

                prev_markers_x = current_markers_x

                if ffmpeg_process is None:
                    target_h, target_w, _ = out_frame.shape
                    ffmpeg_cmd = [
                        "ffmpeg", "-y",
                        "-f", "rawvideo",
                        "-pix_fmt", "bgr24",
                        "-s", f"{target_w}x{target_h}",
                        "-r", str(FPS),
                        "-i", "-",
                        "-an",
                        "-c:v", "libx264",
                        "-crf", "23",
                        "-preset", "fast",
                        "-pix_fmt", "yuv420p",
                        SMOOTH_VIDEO_PATH,
                    ]
                    ffmpeg_process = subprocess.Popen(
                        ffmpeg_cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                    )
                    # Initialize queue and start background thread here
                    viz_queue = queue.Queue(maxsize=128)
                    viz_thread = threading.Thread(
                        target=viz_writer,
                        args=(ffmpeg_process, viz_queue, viz_errors),
                        daemon=True
                    )
                    viz_thread.start()

                h, w, _ = out_frame.shape
                if w != target_w or h != target_h:
                    out_frame = cv2.resize(out_frame, (target_w, target_h))

                # Push to background thread instead of writing directly
                if viz_queue is not None:
                    while True:
                        if viz_errors:
                            raise RuntimeError(
                                f"Video writer failed: {viz_errors[0]}")
                        try:
                            viz_queue.put(out_frame, timeout=0.5)
                            break
                        except queue.Full:
                            continue

            pbar.update(1)
            batch_idx += 1

    producer.join()

    # Gracefully shut down the background thread before closing ffmpeg
    if viz_queue is not None and not viz_errors:
        viz_queue.put(None)
    if viz_thread is not None:
        viz_thread.join()

    if viz_errors:
        raise RuntimeError(f"Video writer failed: {viz_errors[0]}")

    if ffmpeg_process is not None:
        try:
            if ffmpeg_process.stdin is not None:
                ffmpeg_process.stdin.close()
        except Exception:
            pass
        try:
            ffmpeg_process.wait()
        except Exception:
            pass

    # --- Convert and Save Outputs to CSV ---
    if tracking_records:
        df = pd.DataFrame(tracking_records)
        df.to_csv(TRACKING_CSV_PATH, index=False)
        print(f"Tracking records saved to {TRACKING_CSV_PATH}")

    if os.path.exists(SMOOTH_VIDEO_PATH):
        print(f"Smoothed video successfully saved to: {SMOOTH_VIDEO_PATH}")


if __name__ == "__main__":
    main()
