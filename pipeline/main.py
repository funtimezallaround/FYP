import os
import cv2
import numpy as np
import torch
import PIL.Image
import subprocess
import pandas as pd
import threading
import queue
import urllib.request
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from huggingface_hub import snapshot_download
from ultralytics import YOLO
from transformers import AutoProcessor, VitPoseForPoseEstimation
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


# ============ SETUP =====================================================================================

# Configuration
PARTICIPANT_ID = "P035"
SWIMMING_STYLE = "Breaststroke"

# Pose Engine Options: "vitpose" | "yolo226l-pose" | "mediapipe" | "all"
POSE_ENGINE = "all"

START_FRAME_IDX = 0
NUM_FRAMES = -1  # Set to -1 to process all frames
FPS = 60
BATCH_SIZE = 64
MARKER_REAL_DIST_M = 2.5
YOLO_CONF = 0.3

# Paths
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
OUT_DIR = f"output/{SWIMMING_STYLE}_{PARTICIPANT_ID}"

# Model-specific paths
YOLO_MARKER_MODEL_PATH = "models/marker_detector.pt"
YOLO_PERSON_MODEL_PATH = "models/yolov8s.pt"

# Pose estimation model paths
YOLO_POSE_MODEL_PATH = "models/yolo26l-pose.pt"
MP_POSE_MODEL_PATH = "models/pose_landmarker_heavy.task"
VITPOSE_MODEL_NAME = "usyd-community/vitpose-plus-huge"

# Output files
TRACKING_CSV_PATH = f"{OUT_DIR}/tracking_results_{POSE_ENGINE}.csv"
OUT_VIDEO_PATH = f"{OUT_DIR}/tracking_visualization_{POSE_ENGINE}.mp4"

# Ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)


# ============ UTILITY FUNCTIONS ==========================================================================

def download_mediapipe_model(dest_path=MP_POSE_MODEL_PATH):
    """Downloads the MediaPipe Pose Landmarker model if it doesn't already exist in models/."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if not os.path.exists(dest_path):
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
        print(
            f"Downloading MediaPipe Pose Landmarker model from {url} to {dest_path}...")
        try:
            urllib.request.urlretrieve(url, dest_path)
            print("Download complete!")
        except Exception as e:
            # Fallback to direct numbered version URL if latest alias fails
            fallback_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
            print(
                f"Failed to download from primary URL. Trying fallback: {fallback_url}...")
            urllib.request.urlretrieve(fallback_url, dest_path)
            print("Download complete via fallback!")
    else:
        print(f"Cached MediaPipe model found at {dest_path}.")


def download_huggingface_model(repo_id, base_dest_dir="models"):
    """Downloads a HuggingFace model to the local models/ directory if it doesn't already exist."""
    os.makedirs(base_dest_dir, exist_ok=True)

    # Create a unique local folder name based on the repo id
    model_dir_name = repo_id.replace("/", "_")
    local_dir = os.path.join(base_dest_dir, model_dir_name)

    # Check if the model directory exists and has files
    if not os.path.exists(local_dir) or not os.listdir(local_dir):
        print(f"Downloading HuggingFace model '{repo_id}' to '{local_dir}'...")
        snapshot_download(repo_id=repo_id, local_dir=local_dir)
        print("Download complete!")
    else:
        print(f"Cached HuggingFace model found at '{local_dir}'.")

    return local_dir


# Standard pose connections mapping (indexes of 33 keypoints)
POSE_CONNECTIONS = [
    # Face/Head connections
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Shoulder/Torso connections
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left Arm connections
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Right Arm connections
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Left Leg connections
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Right Leg connections
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
]


# ============ POSE ESTIMATION CLASSES ====================================================================

class MediaPipePoseEstimator:
    def __init__(self, model_path=MP_POSE_MODEL_PATH):
        # Guarantee local model file availability
        download_mediapipe_model(model_path)

        # Initialize modern MediaPipe Tasks Pose Landmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_segmentation_masks=False
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def process_frame(self, frame_bgr):
        # MediaPipe Tasks expects RGB format
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Wrap numpy frame into modern MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Run detection
        results = self.detector.detect(mp_image)

        landmarks_dict = {}
        pose_landmarks = None

        # Parse the output safely
        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            pose_landmarks = results.pose_landmarks[0]
            h, w, _ = frame_bgr.shape
            for idx, landmark in enumerate(pose_landmarks):
                # Scale landmarks to absolute pixel coordinates
                landmarks_dict[idx] = (
                    landmark.x * w,
                    landmark.y * h,
                    landmark.visibility
                )
        return landmarks_dict, pose_landmarks

    def draw_landmarks(self, frame_bgr, pose_landmarks):
        if not pose_landmarks:
            return frame_bgr

        h, w, _ = frame_bgr.shape
        coords = {}

        # Map out detected joints which have high enough confidence
        for idx, landmark in enumerate(pose_landmarks):
            if landmark.visibility > 0.5:
                coords[idx] = (int(landmark.x * w), int(landmark.y * h))

        # Draw skeleton limbs (lines)
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx in coords and end_idx in coords:
                cv2.line(
                    frame_bgr, coords[start_idx], coords[end_idx], (0, 255, 0), 2, cv2.LINE_AA)

        # Draw joints (circles)
        for idx, pt in coords.items():
            # Color split: Green for torso, Red/Blue for left/right sides
            if idx in [11, 12, 23, 24]:
                color = (0, 255, 255)  # Yellow torso bounds
            else:
                color = (255, 0, 0) if idx % 2 == 0 else (0, 0, 255)
            cv2.circle(frame_bgr, pt, 4, color, -1, cv2.LINE_AA)
            cv2.circle(frame_bgr, pt, 5, (255, 255, 255), 1, cv2.LINE_AA)

        return frame_bgr


class Yolov8PoseEstimator:
    def __init__(self, model_path=YOLO_POSE_MODEL_PATH):
        # Ultralytics natively handles caching the downloaded file
        # locally to the provided `model_path` (e.g., models/yolov8x-pose.pt)
        self.model = YOLO(model_path)

    def process_frame(self, frame_bgr):
        # YOLO expects BGR, batch inference of 1 image
        results = self.model(frame_bgr, verbose=False, conf=0.25)[0]
        landmarks_dict = {}

        if len(results.boxes) > 0:
            # We take the first detected person
            # [33, 3] or [17, 3]
            keypoints = results.keypoints.data[0].cpu().numpy()
            # YOLO keypoints format: [num_keypoints, 3] where elements are [x, y, conf]
            for idx, kp in enumerate(keypoints):
                x, y, conf = kp
                landmarks_dict[idx] = (x, y, conf)
        return landmarks_dict, results

    def draw_landmarks(self, frame_bgr, yolo_results):
        # YOLOv8 has built-in plot method, but we can draw keypoints manually on frame
        if len(yolo_results.boxes) > 0:
            keypoints = yolo_results.keypoints.data[0].cpu().numpy()
            for kp in keypoints:
                x, y, conf = kp
                if conf > 0.25:
                    cv2.circle(frame_bgr, (int(x), int(y)),
                               4, (0, 255, 255), -1)
        return frame_bgr


class VitPoseEstimator:
    def __init__(self, model_name=VITPOSE_MODEL_NAME):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Download the model explicitly to the models/ directory and cache it
        local_model_dir = download_huggingface_model(model_name)

        self.processor = AutoProcessor.from_pretrained(local_model_dir)
        self.model = VitPoseForPoseEstimation.from_pretrained(
            local_model_dir).to(self.device)
        self.model.eval()

    def process_frame(self, frame_bgr, person_box=None):
        # VitPose expects person crops
        h, w, _ = frame_bgr.shape
        if person_box is not None:
            x1, y1, x2, y2 = map(int, person_box)
            # Clip values
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                crop = frame_bgr
                x1, y1 = 0, 0
        else:
            crop = frame_bgr
            x1, y1 = 0, 0

        # Convert to PIL Image in RGB
        image_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = PIL.Image.fromarray(image_rgb)

        img_w, img_h = pil_img.size
        box = [[[0, 0, img_w, img_h]]]

        inputs = self.processor(
            images=pil_img, boxes=box, return_tensors="pt").to(self.device)

        inputs["dataset_index"] = torch.tensor(
            [0], device=self.device)  # 0 = COCO

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process to get absolute landmarks
        results = self.processor.post_process_pose_estimation(
            outputs, boxes=box)[0][0]

        landmarks_dict = {}
        for idx, (kp, score) in enumerate(zip(results["keypoints"], results["scores"])):
            x, y = kp
            # Shift back to global frame coords
            global_x = x.item() + x1
            global_y = y.item() + y1
            landmarks_dict[idx] = (global_x, global_y, score.item())

        return landmarks_dict, results

    def draw_landmarks(self, frame_bgr, vitpose_results, person_box=None):
        # Draw VitPose 17 keypoints
        if person_box is not None:
            x1, y1, x2, y2 = map(int, person_box)
            x1, y1 = max(0, x1), max(0, y1)
        else:
            x1, y1 = 0, 0

        for kp, score in zip(vitpose_results["keypoints"], vitpose_results["scores"]):
            if score.item() > 0.3:
                x = int(kp[0].item() + x1)
                y = int(kp[1].item() + y1)
                cv2.circle(frame_bgr, (x, y), 4, (255, 0, 255), -1)
        return frame_bgr


# ============ VIDEO DECODING / ENCODING ==================================================================

def get_video_properties(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, fps, total_frames


def find_input_video():
    # Look for video matching SWIMMING_STYLE and PARTICIPANT_ID inside videos/ folder
    video_dir = os.path.join(PROJECT_ROOT, "videos", "under")

    if not os.path.exists(video_dir):
        return None

    for f in os.listdir(video_dir):
        if PARTICIPANT_ID in f and SWIMMING_STYLE in f and f.lower().endswith(".mp4"):
            return os.path.join(video_dir, f)
    return None


# ============ FRAME EXTRACTION AND CACHING ===============================================================

def extract_frames_if_needed(video_path, frames_dir):
    """Extracts frames from the video to frames_dir if they have not been extracted yet."""
    os.makedirs(frames_dir, exist_ok=True)

    # Simple check: do we have any JPEG/PNG frames already here?
    existing_frames = [f for f in os.listdir(
        frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(existing_frames) > 0:
        print(
            f"Frames already extracted in '{frames_dir}' (found {len(existing_frames)} frames). Skipping extraction step.")
        return len(existing_frames)

    print(
        f"No existing frames found in '{frames_dir}'. Starting one-time extraction from video...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(
            f"Failed to open video {video_path} for frame extraction.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    with tqdm(total=total_frames, desc="Extracting video frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Format filename to maintain sorting order
            frame_name = f"frame_{frame_idx:06d}.jpg"
            frame_path = os.path.join(frames_dir, frame_name)

            # Save frame with high quality
            cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            frame_idx += 1
            pbar.update(1)

    cap.release()
    print(f"Extraction complete. Saved {frame_idx} frames to '{frames_dir}'.")
    return frame_idx


def get_frames_properties(frames_dir, video_path=None, fallback_fps=120):
    """Retrieves spatial properties from the first cached image and grabs the video properties."""
    frame_files = sorted([
        f for f in os.listdir(frames_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if not frame_files:
        raise FileNotFoundError(
            f"No frames available in {frames_dir} to retrieve properties.")

    # Read first image to obtain resolution properties
    first_frame_path = os.path.join(frames_dir, frame_files[0])
    img = cv2.imread(first_frame_path)
    if img is None:
        raise IOError(
            f"Failed to read frame {first_frame_path} for size extraction.")

    h, w, _ = img.shape
    total_frames = len(frame_files)
    fps = fallback_fps

    # Try to grab original FPS if video path is reachable
    if video_path and os.path.exists(video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                orig_fps = cap.get(cv2.CAP_PROP_FPS)
                if orig_fps > 0:
                    fps = orig_fps
                cap.release()
        except Exception:
            pass

    return w, h, fps, total_frames


# ============ BATCHED DECODER (UPDATED FOR DIRECTORY READS) =============================================

def frame_producer(frames_dir, start_idx, num_frames, batch_queue, error_list):
    try:
        # Collect all frame paths and sort them
        frame_files = sorted([
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        total_available = len(frame_files)
        if total_available == 0:
            raise FileNotFoundError(
                f"No frames available to process in {frames_dir}")

        frames_read = 0
        current_idx = start_idx

        while num_frames == -1 or frames_read < num_frames:
            batch = []
            for _ in range(BATCH_SIZE):
                if current_idx >= total_available:
                    break

                frame_path = frame_files[current_idx]
                frame = cv2.imread(frame_path)
                if frame is None:
                    raise IOError(f"Failed to read image file: {frame_path}")

                batch.append((current_idx, frame))
                frames_read += 1
                current_idx += 1
                if num_frames != -1 and frames_read >= num_frames:
                    break

            if not batch:
                break

            batch_queue.put(batch)

        batch_queue.put(None)  # Sentinel
    except Exception as e:
        error_list.append(str(e))
        batch_queue.put(None)


# ============ FFMPEG WRITER ==============================================================================

def video_writer_thread(viz_queue, ffmpeg_process, error_list):
    try:
        while True:
            frame = viz_queue.get()
            if frame is None:
                break
            if ffmpeg_process.poll() is not None:
                raise RuntimeError(
                    f"FFmpeg process exited prematurely with code {ffmpeg_process.returncode}")
            ffmpeg_process.stdin.write(frame.tobytes())
            viz_queue.task_done()
    except Exception as e:
        error_list.append(str(e))


# ============ MAIN PIPELINE ==============================================================================

def main():
    video_path = find_input_video()
    if not video_path:
        raise FileNotFoundError(
            f"Could not locate an input video containing participant id '{PARTICIPANT_ID}' "
            f"and style '{SWIMMING_STYLE}' inside videos/under/"
        )

    # Establish localized frames directory structure for the current sequence
    frames_dir = os.path.join(PROJECT_ROOT, "frames",
                              "under", f"Bottom_{SWIMMING_STYLE}_{PARTICIPANT_ID}")

    # 1. One-time extraction routine
    extract_frames_if_needed(video_path, frames_dir)

    # 2. Extract properties directly from frames directory (with optional video properties support)
    w, h, orig_fps, total_frames_count = get_frames_properties(
        frames_dir, video_path, fallback_fps=FPS)
    print(
        f"Sequence Properties: Size={w}x{h}, FPS={orig_fps:.2f}, Total Frames Available={total_frames_count}")

    frames_to_process = NUM_FRAMES if NUM_FRAMES != - \
        1 else (total_frames_count - START_FRAME_IDX)
    print(
        f"Processing {frames_to_process} frames starting from index {START_FRAME_IDX}...")

    # Initialize models
    print("Loading Detection Models...")
    yolo_person = YOLO(YOLO_PERSON_MODEL_PATH)
    yolo_marker = YOLO(YOLO_MARKER_MODEL_PATH)

    pose_mp_estimator = None
    pose_yolo_estimator = None
    pose_vit_estimator = None

    if POSE_ENGINE in ["mediapipe", "all"]:
        print("Initializing MediaPipe Pose...")
        pose_mp_estimator = MediaPipePoseEstimator()
    if POSE_ENGINE in ["yolov8-pose", "all"]:
        print("Initializing YOLOv8 Pose...")
        pose_yolo_estimator = Yolov8PoseEstimator()
    if POSE_ENGINE in ["vitpose", "all"]:
        print("Initializing VitPose...")
        pose_vit_estimator = VitPoseEstimator()

    # Setup multithreaded queues using disk frame directory
    batch_queue = queue.Queue(maxsize=4)
    producer_errors = []
    producer = threading.Thread(
        target=frame_producer,
        args=(frames_dir, START_FRAME_IDX, frames_to_process,
              batch_queue, producer_errors)
    )
    producer.start()

    # Video output stream configuration (FFmpeg pipeline)
    target_w, target_h = w, h
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f"{target_w}x{target_h}",
        '-r', str(FPS),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast',
        '-nostats',
        '-v', 'error',
        OUT_VIDEO_PATH
    ]

    ffmpeg_process = None
    viz_queue = None
    viz_thread = None
    viz_errors = []

    try:
        ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        viz_queue = queue.Queue(maxsize=32)
        viz_thread = threading.Thread(
            target=video_writer_thread,
            args=(viz_queue, ffmpeg_process, viz_errors)
        )
        viz_thread.start()
        print("FFmpeg process and writer thread started.")
    except Exception as e:
        print(
            f"Warning: Failed to start FFmpeg. Video output will be skipped. Error: {e}")

    tracking_records = []

    with tqdm(total=frames_to_process, desc="Tracking/Pose Pipeline") as pbar:
        batch_idx = 0
        while True:
            if producer_errors:
                raise RuntimeError(
                    f"Decoder thread failed: {producer_errors[0]}")

            batch = batch_queue.get()
            if batch is None:
                break

            for frame_idx, frame in batch:
                out_frame = frame.copy() if ffmpeg_process is not None else None

                # 1. Detect Person with YOLOv8s
                person_results = yolo_person(frame, verbose=False, conf=0.5)[0]
                person_boxes = person_results.boxes.xyxy.cpu().numpy()
                person_confs = person_results.boxes.conf.cpu().numpy()
                person_classes = person_results.boxes.cls.cpu().numpy()

                # Filter class 0 (person)
                person_indices = np.where(person_classes == 0)[0]
                person_boxes = person_boxes[person_indices]
                person_confs = person_confs[person_indices]

                # Draw person box
                best_person_box = None
                if len(person_boxes) > 0:
                    # Choose the person with highest confidence
                    best_person_idx = np.argmax(person_confs)
                    best_person_box = person_boxes[best_person_idx]

                    if out_frame is not None:
                        x1, y1, x2, y2 = map(int, best_person_box)
                        cv2.rectangle(out_frame, (x1, y1),
                                      (x2, y2), (255, 0, 0), 2)
                        cv2.putText(out_frame, f"Person {person_confs[best_person_idx]:.2f}",
                                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # 2. Detect Swimming Cap/Markers (Underwater tracking)
                marker_results = yolo_marker(
                    frame, verbose=False, conf=YOLO_CONF)[0]
                marker_boxes = marker_results.boxes.xyxy.cpu().numpy()
                marker_confs = marker_results.boxes.conf.cpu().numpy()
                marker_classes = marker_results.boxes.cls.cpu().numpy()

                # Save base details
                record = {
                    "frame_idx": frame_idx,
                    "timestamp": frame_idx / FPS,
                    "person_detected": len(person_boxes) > 0,
                    "person_conf": person_confs[best_person_idx] if len(person_boxes) > 0 else 0.0,
                    "markers_detected": len(marker_boxes)
                }

                # Add markers coordinates
                for m_idx in range(min(len(marker_boxes), 5)):  # track up to 5 markers
                    x1, y1, x2, y2 = marker_boxes[m_idx]
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    record[f"marker_{m_idx}_x"] = cx
                    record[f"marker_{m_idx}_y"] = cy
                    record[f"marker_{m_idx}_conf"] = marker_confs[m_idx]

                    if out_frame is not None:
                        cv2.rectangle(out_frame, (int(x1), int(y1)),
                                      (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(out_frame, f"Marker {m_idx}", (int(x1), int(y1)-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                # 3. Running Pose Estimation Engines
                # A. MediaPipe (Migrated to Tasks API)
                mp_landmarks = None
                if pose_mp_estimator is not None:
                    mp_landmarks, mp_raw = pose_mp_estimator.process_frame(
                        frame)
                    if mp_landmarks:
                        if out_frame is not None:
                            out_frame = pose_mp_estimator.draw_landmarks(
                                out_frame, mp_raw)
                        for idx, (x, y, conf) in mp_landmarks.items():
                            record[f"mp_joint_{idx}_x"] = x
                            record[f"mp_joint_{idx}_y"] = y
                            record[f"mp_joint_{idx}_conf"] = conf

                # B. YOLOv8 Pose
                yolo_landmarks = None
                if pose_yolo_estimator is not None:
                    yolo_landmarks, yolo_raw = pose_yolo_estimator.process_frame(
                        frame)
                    if yolo_landmarks:
                        if out_frame is not None:
                            out_frame = pose_yolo_estimator.draw_landmarks(
                                out_frame, yolo_raw)
                        for idx, (x, y, conf) in yolo_landmarks.items():
                            record[f"yolo_joint_{idx}_x"] = x
                            record[f"yolo_joint_{idx}_y"] = y
                            record[f"yolo_joint_{idx}_conf"] = conf

                # C. VitPose
                vit_landmarks = None
                if pose_vit_estimator is not None and best_person_box is not None:
                    vit_landmarks, vit_raw = pose_vit_estimator.process_frame(
                        frame, best_person_box)
                    if vit_landmarks:
                        if out_frame is not None:
                            out_frame = pose_vit_estimator.draw_landmarks(
                                out_frame, vit_raw, best_person_box)
                        for idx, (x, y, conf) in vit_landmarks.items():
                            record[f"vit_joint_{idx}_x"] = x
                            record[f"vit_joint_{idx}_y"] = y
                            record[f"vit_joint_{idx}_conf"] = conf

                tracking_records.append(record)

                # Push to visualization queue if available
                if out_frame is not None:
                    if out_frame.shape[1] != target_w or out_frame.shape[0] != target_h:
                        out_frame = cv2.resize(out_frame, (target_w, target_h))

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

            pbar.update(len(batch))
            batch_idx += 1

    producer.join()

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

    # Save complete outputs to CSV
    if tracking_records:
        df = pd.DataFrame(tracking_records)
        df.to_csv(TRACKING_CSV_PATH, index=False)
        print(f"Tracking records saved to {TRACKING_CSV_PATH}")

    if os.path.exists(OUT_VIDEO_PATH):
        print(f"Tracking visualization video saved to {OUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()
