# /// script
# description = "Run YOLO-tracked SAM 3D Body mesh recovery on sampled video frames"
# requires-python = ">=3.11, <3.13"
# dependencies = [
#   "daft>=0.7.10",
#   "huggingface_hub",
#   "modal",
#   "ultralytics",
# ]
# ///
"""End-to-end video → 3D pose sequence pipeline.

Composes the SAM 3D Body model UDF (``models/sam3d_body``) with frame
sampling, YOLO person detection, and bbox tracking, then writes per-frame
meshes, render overlays, and a ``sequence.json`` viewer manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import modal

# Anchor the repo root so `models.*` imports resolve when this file is loaded
# as a loose script (e.g. `uv run` / `modal run`) instead of an installed package.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.common.modal_infra import APP_DIR, INPUT_DIR, MODEL_CACHE_DIR, OUTPUT_DIR
from models.common.weights import (
    DEFAULT_YOLO_RELEASE,
    normalize_hf_token_env,
    resolve_hf_model_path,
    resolve_yolo_weight_path,
)
from models.sam3d_body.modal_app import (
    GPU_TYPE,
    MODAL_REGION,
    SAM3D_REPO_DIR,
    base_image,
    model_cache,
    outputs,
)
from models.sam3d_body.model import DEFAULT_MODEL, build_dataframe

DEFAULT_VIDEO = os.environ.get("SAM3D_BODY_VIDEO", "")
DEFAULT_YOLO_MODEL = "yolov8n.pt"
DEFAULT_SAMPLE_MODE = "stride"
DEFAULT_SAMPLE_FPS = 3.0
DEFAULT_FRAME_STRIDE = 10
DEFAULT_MAX_FRAMES = 0
UPPER_BODY_KEYPOINTS = [0, 5, 6, 69]
LOWER_BODY_KEYPOINTS = [13, 14, 15, 16, 17, 18, 19, 20]
MID_BODY_KEYPOINTS = [9, 10, 11, 12]
MHR70_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "right_thumb4",
    "right_thumb3",
    "right_thumb2",
    "right_thumb_third_joint",
    "right_forefinger4",
    "right_forefinger3",
    "right_forefinger2",
    "right_forefinger_third_joint",
    "right_middle_finger4",
    "right_middle_finger3",
    "right_middle_finger2",
    "right_middle_finger_third_joint",
    "right_ring_finger4",
    "right_ring_finger3",
    "right_ring_finger2",
    "right_ring_finger_third_joint",
    "right_pinky_finger4",
    "right_pinky_finger3",
    "right_pinky_finger2",
    "right_pinky_finger_third_joint",
    "right_wrist",
    "left_thumb4",
    "left_thumb3",
    "left_thumb2",
    "left_thumb_third_joint",
    "left_forefinger4",
    "left_forefinger3",
    "left_forefinger2",
    "left_forefinger_third_joint",
    "left_middle_finger4",
    "left_middle_finger3",
    "left_middle_finger2",
    "left_middle_finger_third_joint",
    "left_ring_finger4",
    "left_ring_finger3",
    "left_ring_finger2",
    "left_ring_finger_third_joint",
    "left_pinky_finger4",
    "left_pinky_finger3",
    "left_pinky_finger2",
    "left_pinky_finger_third_joint",
    "left_wrist",
    "left_olecranon",
    "right_olecranon",
    "left_cubital_fossa",
    "right_cubital_fossa",
    "left_acromion",
    "right_acromion",
    "neck",
]
CORE_SKELETON_LINKS = [
    [13, 11],
    [11, 9],
    [14, 12],
    [12, 10],
    [9, 10],
    [5, 9],
    [6, 10],
    [5, 6],
    [69, 5],
    [69, 6],
    [5, 7],
    [7, 62],
    [6, 8],
    [8, 41],
    [0, 69],
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [13, 15],
    [13, 16],
    [13, 17],
    [14, 18],
    [14, 19],
    [14, 20],
]


@dataclass
class VideoFrame:
    index: int
    source_frame: int
    timestamp_sec: float
    image_path: str
    bbox: list[float]
    detection_confidence: float
    detection_source: str


def stable_digest(
    path: str,
    sample_fps: float,
    max_frames: int,
    sample_mode: str,
    frame_stride: int,
    model: str,
    model_revision: str,
    yolo_model: str,
    yolo_release: str,
    yolo_confidence: float,
    bbox_padding: float,
) -> str:
    stat = Path(path).stat()
    payload = (
        f"{Path(path).name}:{stat.st_size}:{stat.st_mtime_ns}:{sample_fps}:{max_frames}:{sample_mode}:"
        f"{frame_stride}:{model}:{model_revision}:{yolo_model}:{yolo_release}:{yolo_confidence}:{bbox_padding}"
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def file_digest(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def stage_local_video_input(video_path: Path) -> str:
    digest = file_digest(video_path)
    remote_volume_path = f"/videos/{video_path.stem}-{digest}{video_path.suffix}"
    with input_videos.batch_upload(force=True) as batch:
        batch.put_file(str(video_path), remote_volume_path)
    return str(Path(INPUT_DIR) / remote_volume_path.lstrip("/"))


def clip_bbox(bbox: list[float], width: int, height: int, padding: float = 0.28) -> list[float]:
    x1, y1, x2, y2 = bbox
    box_width = x2 - x1
    box_height = y2 - y1
    cx = x1 + box_width / 2
    cy = y1 + box_height / 2
    box_width *= 1 + padding
    box_height *= 1 + padding
    return [
        max(0.0, cx - box_width / 2),
        max(0.0, cy - box_height / 2),
        min(float(width), cx + box_width / 2),
        min(float(height), cy + box_height / 2),
    ]


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def choose_detection(detections: list[dict], width: int, height: int, previous_bbox: list[float] | None) -> dict | None:
    if not detections:
        return None

    frame_diag = (width**2 + height**2) ** 0.5
    frame_area = width * height
    previous_center = bbox_center(previous_bbox) if previous_bbox else None

    def score_detection(detection: dict) -> float:
        bbox = detection["bbox"]
        x1, y1, x2, y2 = bbox
        area_score = ((x2 - x1) * (y2 - y1)) / frame_area
        center_x, center_y = bbox_center(bbox)
        center_bias = 1 - abs(center_x - width / 2) / (width / 2)
        height_bias = 1 - center_y / height
        score = detection["confidence"] + 5.0 * area_score + 0.25 * center_bias + 0.15 * height_bias
        if previous_center:
            distance = ((center_x - previous_center[0]) ** 2 + (center_y - previous_center[1]) ** 2) ** 0.5
            score += 0.85 * (1 - min(distance / frame_diag, 1))
        return score

    return max(detections, key=score_detection)


def score_track_candidate(detection: dict, width: int, height: int) -> float:
    x1, y1, x2, y2 = detection["bbox"]
    box_width = max(x2 - x1, 1.0)
    box_height = max(y2 - y1, 1.0)
    area = box_width * box_height
    center_x, center_y = bbox_center(detection["bbox"])
    center_bias = max(0.0, 1.0 - abs(center_x - width / 2) / (width / 2))
    height_bias = max(0.0, 1.0 - center_y / height)
    aspect_bias = min(box_width / box_height, 2.0)
    area_score = math.sqrt(area / (width * height))
    return detection["confidence"] + 8.0 * area_score + 0.2 * center_bias + 0.25 * height_bias + 0.15 * aspect_bias


def choose_detection_track(
    candidates_by_frame: list[list[dict]],
    frame_sizes: list[tuple[int, int]],
    transition_penalty: float = 5.0,
) -> list[dict]:
    if not candidates_by_frame:
        return []

    scores: list[list[float]] = []
    backpointers: list[list[int]] = []
    for frame_index, candidates in enumerate(candidates_by_frame):
        width, height = frame_sizes[frame_index]
        if frame_index == 0:
            scores.append([score_track_candidate(candidate, width, height) for candidate in candidates])
            backpointers.append([-1] * len(candidates))
            continue

        previous_candidates = candidates_by_frame[frame_index - 1]
        previous_scores = scores[frame_index - 1]
        frame_diag = (width**2 + height**2) ** 0.5
        frame_scores = []
        frame_backpointers = []
        for candidate in candidates:
            candidate_center = bbox_center(candidate["bbox"])
            best_score = -float("inf")
            best_index = 0
            for previous_index, previous_candidate in enumerate(previous_candidates):
                previous_center = bbox_center(previous_candidate["bbox"])
                distance = (
                    (candidate_center[0] - previous_center[0]) ** 2 + (candidate_center[1] - previous_center[1]) ** 2
                ) ** 0.5
                transition_score = -transition_penalty * min(distance / frame_diag, 1.0)
                score = (
                    previous_scores[previous_index] + score_track_candidate(candidate, width, height) + transition_score
                )
                if score > best_score:
                    best_score = score
                    best_index = previous_index
            frame_scores.append(best_score)
            frame_backpointers.append(best_index)
        scores.append(frame_scores)
        backpointers.append(frame_backpointers)

    selected_index = max(range(len(scores[-1])), key=lambda index: scores[-1][index])
    selected_reversed = []
    for frame_index in range(len(candidates_by_frame) - 1, -1, -1):
        selected_reversed.append(candidates_by_frame[frame_index][selected_index])
        selected_index = backpointers[frame_index][selected_index]
    return list(reversed(selected_reversed))


def video_keyframe_indices(video_path: str, source_fps: float, total_frames: int) -> list[int]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-skip_frame",
            "nokey",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "csv=p=0",
            video_path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    frame_indices = []
    for line in result.stdout.splitlines():
        value = line.split(",", 1)[0].strip()
        if not value:
            continue
        frame_index = round(float(value) * source_fps)
        frame_indices.append(max(0, min(int(frame_index), total_frames - 1)))
    return sorted(dict.fromkeys(frame_indices))


def extract_frames(
    video_path: str,
    output_dir: Path,
    sample_fps: float,
    max_frames: int,
    sample_mode: str = DEFAULT_SAMPLE_MODE,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
) -> list[VideoFrame]:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if sample_mode == "keyframes":
        candidate_frames = video_keyframe_indices(video_path, source_fps, total_frames)
    elif sample_mode == "stride":
        step = max(int(frame_stride), 1)
        candidate_frames = list(range(0, total_frames, step))
    else:
        step = max(int(round(source_fps / sample_fps)), 1)
        candidate_frames = list(range(0, total_frames, step))

    if max_frames and len(candidate_frames) > max_frames:
        stride = max((len(candidate_frames) - 1) / max(max_frames - 1, 1), 1)
        candidate_frames = [candidate_frames[round(i * stride)] for i in range(max_frames)]

    if not candidate_frames:
        cap.release()
        raise ValueError(
            f"No candidate frames selected from {video_path}; total_frames={total_frames}, sample_mode={sample_mode}"
        )

    frames = []
    for index, source_frame in enumerate(candidate_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
        ok, image = cap.read()
        if not ok:
            continue
        image_path = output_dir / f"frame_{index:04d}_src_{source_frame:06d}.jpg"
        cv2.imwrite(str(image_path), image)
        frames.append(
            VideoFrame(
                index=index,
                source_frame=source_frame,
                timestamp_sec=source_frame / source_fps,
                image_path=str(image_path),
                bbox=[],
                detection_confidence=0.0,
                detection_source="",
            )
        )
    cap.release()
    if not frames:
        raise ValueError(f"No frames could be decoded from {video_path}")
    return frames


def detect_athlete_bboxes(
    frames: list[VideoFrame],
    confidence: float,
    padding: float,
    yolo_model: str = DEFAULT_YOLO_MODEL,
    yolo_release: str = DEFAULT_YOLO_RELEASE,
) -> list[VideoFrame]:
    import cv2
    from ultralytics import YOLO

    yolo_weights = resolve_yolo_weight_path(yolo_model, MODEL_CACHE_DIR, release=yolo_release)
    yolo = YOLO(str(yolo_weights))
    candidates_by_frame = []
    frame_sizes = []
    candidate_confidence = min(confidence, 0.03)
    for frame in frames:
        image = cv2.imread(frame.image_path)
        height, width = image.shape[:2]
        frame_sizes.append((width, height))
        result = yolo.predict(frame.image_path, classes=[0], conf=candidate_confidence, device="cpu", verbose=False)[0]
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                detections.append(
                    {
                        "bbox": [float(value) for value in box.xyxy[0].tolist()],
                        "confidence": float(box.conf[0]),
                    }
                )

        if not detections:
            detections = [
                {
                    "bbox": [0.0, 0.0, float(width), float(height)],
                    "confidence": 0.0,
                    "source": "full_frame",
                }
            ]
        for detection in detections:
            detection.setdefault("source", "yolo_track")
        candidates_by_frame.append(detections)

    selected_track = choose_detection_track(candidates_by_frame, frame_sizes)
    for frame, selected, (width, height) in zip(frames, selected_track, frame_sizes):
        frame.bbox = clip_bbox(selected["bbox"], width, height, padding=padding)
        frame.detection_confidence = selected["confidence"]
        frame.detection_source = selected["source"]
    return frames


def wrap_degrees(value: float) -> float:
    return ((value + 180.0) % 360.0) - 180.0


def body_centroid(points, indices: list[int]):
    import numpy as np

    selected = []
    for index in indices:
        if index < len(points) and np.isfinite(points[index]).all():
            selected.append(points[index])
    if not selected:
        return None
    return np.mean(selected, axis=0)


def vector_angle_degrees(vector) -> float:
    return math.degrees(math.atan2(float(vector[1]), float(vector[0])))


def estimate_orientation_from_npz(npz_path: str) -> dict:
    import numpy as np

    if not npz_path:
        return {}

    with np.load(npz_path) as data:
        keypoints_2d = data["pred_keypoints_2d"]
        keypoints_3d = data["pred_keypoints_3d"]

    upper_2d = body_centroid(keypoints_2d, UPPER_BODY_KEYPOINTS)
    lower_2d = body_centroid(keypoints_2d, LOWER_BODY_KEYPOINTS)
    if lower_2d is None:
        lower_2d = body_centroid(keypoints_2d, MID_BODY_KEYPOINTS)

    upper_3d = body_centroid(keypoints_3d, UPPER_BODY_KEYPOINTS)
    lower_3d = body_centroid(keypoints_3d, LOWER_BODY_KEYPOINTS)
    if lower_3d is None:
        lower_3d = body_centroid(keypoints_3d, MID_BODY_KEYPOINTS)

    if upper_2d is None or lower_2d is None or upper_3d is None or lower_3d is None:
        return {}

    image_vector_px = upper_2d[:2] - lower_2d[:2]
    image_vector_viewer = np.asarray([image_vector_px[0], -image_vector_px[1]], dtype=np.float64)

    # The exported PLY goes through Renderer.vertices_to_trimesh, which applies a 180 degree
    # rotation around X. Apply the same basis change before comparing its in-plane body axis.
    mesh_vector = upper_3d[:3] - lower_3d[:3]
    mesh_vector_viewer = np.asarray([mesh_vector[0], -mesh_vector[1]], dtype=np.float64)

    image_axis_length_px = float(np.linalg.norm(image_vector_px))
    mesh_axis_length = float(np.linalg.norm(mesh_vector_viewer))
    if image_axis_length_px < 1e-6 or mesh_axis_length < 1e-6:
        return {}

    image_angle_degrees = vector_angle_degrees(image_vector_viewer)
    mesh_angle_degrees = vector_angle_degrees(mesh_vector_viewer)
    correction_degrees = wrap_degrees(image_angle_degrees - mesh_angle_degrees)

    return {
        "image_body_axis_px": [float(image_vector_px[0]), float(image_vector_px[1])],
        "viewer_body_axis": [float(image_vector_viewer[0]), float(image_vector_viewer[1])],
        "mesh_body_axis": [float(mesh_vector_viewer[0]), float(mesh_vector_viewer[1])],
        "image_body_angle_degrees": image_angle_degrees,
        "mesh_body_angle_degrees": mesh_angle_degrees,
        "mesh_z_correction_degrees": correction_degrees,
        "mesh_z_correction_radians": math.radians(correction_degrees),
        "head_above_feet_in_image": bool(image_vector_px[1] < 0),
        "image_axis_length_px": image_axis_length_px,
        "mesh_axis_length": mesh_axis_length,
    }


def viewer_basis(points, center):
    transformed = points.copy()
    transformed[:, 1] *= -1.0
    transformed[:, 2] *= -1.0
    return transformed - center


def skeleton_from_npz(npz_path: str) -> dict:
    import numpy as np

    if not npz_path:
        return {}

    with np.load(npz_path) as data:
        keypoints_3d = data["pred_keypoints_3d"]
        vertices = data["pred_vertices"]
        cam_t = data["pred_cam_t"]

    mesh_vertices = vertices + cam_t
    mesh_vertices[:, 1] *= -1.0
    mesh_vertices[:, 2] *= -1.0
    mesh_center = (np.max(mesh_vertices, axis=0) + np.min(mesh_vertices, axis=0)) / 2

    keypoints = viewer_basis(keypoints_3d + cam_t, mesh_center)
    finite = np.isfinite(keypoints).all(axis=1)
    if not finite.any():
        return {}

    return {
        "format": "mhr70_viewer_basis_v1",
        "keypoints": [
            [float(value) for value in keypoint] if bool(finite[index]) else None
            for index, keypoint in enumerate(keypoints)
        ],
    }


def write_sequence_outputs(
    sequence_dir: Path,
    frames: list[VideoFrame],
    result: dict,
    video_path: str,
    model: str,
    model_revision: str,
    yolo_model: str,
    yolo_release: str,
    sample_mode: str,
    sample_fps: float,
    frame_stride: int,
    max_frames: int,
    profile: dict | None = None,
) -> dict:
    meshes_dir = sequence_dir / "meshes"
    renders_dir = sequence_dir / "renders"
    meshes_dir.mkdir(parents=True, exist_ok=True)
    renders_dir.mkdir(parents=True, exist_ok=True)

    sequence_frames = []
    for row_index, frame in enumerate(frames):
        mesh_paths = result["mesh_paths"][row_index]
        npz_paths = result["npz_paths"][row_index]
        render_path = result["render_path"][row_index]
        metadata_path = result["metadata_path"][row_index]
        mesh_path = ""
        render_copy = ""
        frame_path = ""
        orientation = estimate_orientation_from_npz(npz_paths[0]) if npz_paths else {}
        skeleton = skeleton_from_npz(npz_paths[0]) if npz_paths else {}
        try:
            frame_path = str(Path(frame.image_path).relative_to(sequence_dir))
        except ValueError:
            frame_path = str(Path("frames") / Path(frame.image_path).name)
        if mesh_paths:
            mesh_copy = meshes_dir / f"frame_{frame.index:04d}.ply"
            shutil.copyfile(mesh_paths[0], mesh_copy)
            mesh_path = str(mesh_copy.relative_to(sequence_dir))
        if render_path:
            render_copy_path = renders_dir / f"frame_{frame.index:04d}.jpg"
            shutil.copyfile(render_path, render_copy_path)
            render_copy = str(render_copy_path.relative_to(sequence_dir))

        sequence_frames.append(
            {
                **asdict(frame),
                "num_people": result["num_people"][row_index],
                "frame_path": frame_path,
                "mesh_path": mesh_path,
                "render_path": render_copy,
                "raw_render_path": render_path,
                "raw_metadata_path": metadata_path,
                "raw_npz_path": npz_paths[0] if npz_paths else "",
                "orientation": orientation,
                "skeleton_3d": skeleton,
                "mesh_rotation": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": orientation.get("mesh_z_correction_radians", 0.0),
                },
            }
        )

    manifest = {
        "video_path": video_path,
        "model": model,
        "model_revision": model_revision,
        "yolo_model": yolo_model,
        "yolo_release": yolo_release,
        "sample_mode": sample_mode,
        "sample_fps": sample_fps,
        "frame_stride": frame_stride,
        "max_frames": max_frames,
        "profile": profile or {},
        "mesh_coordinate_system": "sam3d_renderer_ply",
        "mesh_rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
        "skeleton": {
            "format": "mhr70_viewer_basis_v1",
            "keypoint_names": MHR70_KEYPOINT_NAMES,
            "links": CORE_SKELETON_LINKS,
        },
        "frames": sequence_frames,
    }
    manifest_path = sequence_dir / "sequence.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "sequence_dir": str(sequence_dir),
        "manifest_path": str(manifest_path),
        "frame_count": len(sequence_frames),
        "mesh_count": sum(1 for frame in sequence_frames if frame["mesh_path"]),
        "render_count": sum(1 for frame in sequence_frames if frame["render_path"]),
        "frames": sequence_frames,
    }


app = modal.App("daft-sam3d-body-video")
input_videos = modal.Volume.from_name("sam3d-body-inputs", create_if_missing=True)
video_image = base_image.pip_install("ultralytics").add_local_python_source("models", "pipelines")


@app.function(
    image=video_image,
    cpu=4,
    memory=16384,
    timeout=7200,
    region=MODAL_REGION,
    volumes={MODEL_CACHE_DIR: model_cache},
    secrets=[modal.Secret.from_name("hf-token")],
)
def download_video_model_weights(
    model: str = DEFAULT_MODEL,
    model_revision: str = "",
    yolo_model: str = DEFAULT_YOLO_MODEL,
    yolo_release: str = DEFAULT_YOLO_RELEASE,
) -> dict:
    os.chdir(APP_DIR)
    sam3d_model_path = resolve_hf_model_path(
        model,
        MODEL_CACHE_DIR,
        revision=model_revision or None,
        token=normalize_hf_token_env(),
    )
    yolo_weight_path = resolve_yolo_weight_path(yolo_model, MODEL_CACHE_DIR, release=yolo_release)
    model_cache.commit()
    return {
        "model": model,
        "model_revision": model_revision,
        "model_path": str(sam3d_model_path),
        "yolo_model": yolo_model,
        "yolo_release": yolo_release,
        "yolo_weight_path": str(yolo_weight_path),
    }


@app.function(
    image=video_image,
    gpu=GPU_TYPE,
    cpu=8,
    memory=98304,
    timeout=7200,
    region=MODAL_REGION,
    volumes={MODEL_CACHE_DIR: model_cache, OUTPUT_DIR: outputs, INPUT_DIR: input_videos},
    secrets=[modal.Secret.from_name("hf-token")],
)
def run_video_on_modal(
    video_path: str,
    model: str = DEFAULT_MODEL,
    model_revision: str = "",
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    max_frames: int = DEFAULT_MAX_FRAMES,
    sample_mode: str = DEFAULT_SAMPLE_MODE,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    yolo_model: str = DEFAULT_YOLO_MODEL,
    yolo_release: str = DEFAULT_YOLO_RELEASE,
    yolo_confidence: float = 0.18,
    bbox_padding: float = 0.35,
):
    profile_start = time.perf_counter()
    profile_marks = {}

    def mark(name: str) -> None:
        profile_marks[name] = time.perf_counter()

    os.chdir(APP_DIR)
    input_videos.reload()
    mark("setup")

    digest = stable_digest(
        video_path,
        sample_fps,
        max_frames,
        sample_mode,
        frame_stride,
        model,
        model_revision,
        yolo_model,
        yolo_release,
        yolo_confidence,
        bbox_padding,
    )
    sequence_dir = Path(OUTPUT_DIR) / f"sam3d-body-video-{Path(video_path).stem}-{digest}"
    frames_dir = sequence_dir / "frames"
    frames = extract_frames(
        video_path,
        frames_dir,
        sample_fps=sample_fps,
        max_frames=max_frames,
        sample_mode=sample_mode,
        frame_stride=frame_stride,
    )
    mark("extract_frames")
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    frames = detect_athlete_bboxes(
        frames,
        confidence=yolo_confidence,
        padding=bbox_padding,
        yolo_model=yolo_model,
        yolo_release=yolo_release,
    )
    mark("detect_athlete_bboxes")
    if cuda_visible_devices is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    df = build_dataframe(
        [frame.image_path for frame in frames],
        bbox_jsons=[json.dumps(frame.bbox) for frame in frames],
        model=model,
        model_revision=model_revision,
        output_dir=str(sequence_dir / "sam3d"),
        repo_path=SAM3D_REPO_DIR,
        inference_type="full",
    ).collect()
    mark("sam3d_collect")

    result = df.to_pydict()
    profile_before_write = time.perf_counter()
    profile = {
        "frame_count": len(frames),
        "sample_mode": sample_mode,
        "sample_fps": sample_fps,
        "frame_stride": frame_stride,
        "max_frames": max_frames,
        "seconds": {
            "setup": profile_marks["setup"] - profile_start,
            "extract_frames": profile_marks["extract_frames"] - profile_marks["setup"],
            "detect_athlete_bboxes": profile_marks["detect_athlete_bboxes"] - profile_marks["extract_frames"],
            "sam3d_collect": profile_marks["sam3d_collect"] - profile_marks["detect_athlete_bboxes"],
        },
    }
    sequence_result = write_sequence_outputs(
        sequence_dir,
        frames,
        result,
        video_path=video_path,
        model=model,
        model_revision=model_revision,
        yolo_model=yolo_model,
        yolo_release=yolo_release,
        sample_mode=sample_mode,
        sample_fps=sample_fps,
        frame_stride=frame_stride,
        max_frames=max_frames,
        profile=profile,
    )
    profile["seconds"]["write_sequence_outputs"] = time.perf_counter() - profile_before_write
    commit_start = time.perf_counter()
    model_cache.commit()
    outputs.commit()
    profile["seconds"]["commit_volumes"] = time.perf_counter() - commit_start
    profile["seconds"]["total_remote_function"] = time.perf_counter() - profile_start
    profile["seconds_per_frame"] = {key: value / max(len(frames), 1) for key, value in profile["seconds"].items()}
    manifest_path = Path(sequence_result["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["profile"] = profile
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    outputs.commit()
    sequence_result["profile"] = profile
    sequence_result["volume_sequence_path"] = f"/{sequence_dir.name}"
    return sequence_result


@app.local_entrypoint()
def modal_main(
    video_path: str = DEFAULT_VIDEO,
    model: str = DEFAULT_MODEL,
    model_revision: str = "",
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    max_frames: int = DEFAULT_MAX_FRAMES,
    sample_mode: str = DEFAULT_SAMPLE_MODE,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    yolo_model: str = DEFAULT_YOLO_MODEL,
    yolo_release: str = DEFAULT_YOLO_RELEASE,
    yolo_confidence: float = 0.18,
    bbox_padding: float = 0.35,
    download_only: bool = False,
):
    if download_only:
        print(
            download_video_model_weights.remote(
                model=model,
                model_revision=model_revision,
                yolo_model=yolo_model,
                yolo_release=yolo_release,
            )
        )
        return

    if not video_path:
        raise ValueError("Pass --video-path or set SAM3D_BODY_VIDEO before running video inference.")

    remote_video_path = video_path
    local_video_path = Path(video_path)
    if local_video_path.exists():
        remote_video_path = stage_local_video_input(local_video_path)
    print(
        run_video_on_modal.remote(
            video_path=remote_video_path,
            model=model,
            model_revision=model_revision,
            sample_fps=sample_fps,
            max_frames=max_frames,
            sample_mode=sample_mode,
            frame_stride=frame_stride,
            yolo_model=yolo_model,
            yolo_release=yolo_release,
            yolo_confidence=yolo_confidence,
            bbox_padding=bbox_padding,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", default=DEFAULT_VIDEO)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=os.environ.get("SAM3D_BODY_MODEL_REVISION", ""))
    parser.add_argument("--sample-fps", type=float, default=DEFAULT_SAMPLE_FPS)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--sample-mode", choices=["stride", "uniform", "keyframes"], default=DEFAULT_SAMPLE_MODE)
    parser.add_argument("--frame-stride", type=int, default=DEFAULT_FRAME_STRIDE)
    parser.add_argument("--yolo-model", default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--yolo-release", default=DEFAULT_YOLO_RELEASE)
    parser.add_argument("--yolo-confidence", type=float, default=0.18)
    parser.add_argument("--bbox-padding", type=float, default=0.35)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    if not args.download_only and not args.video_path:
        parser.error("--video-path is required unless --download-only is set")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.download_only:
        sam3d_model_path = resolve_hf_model_path(
            args.model,
            MODEL_CACHE_DIR,
            revision=args.model_revision or None,
            token=normalize_hf_token_env(),
        )
        yolo_weight_path = resolve_yolo_weight_path(args.yolo_model, MODEL_CACHE_DIR, release=args.yolo_release)
        print(
            {
                "model": args.model,
                "model_revision": args.model_revision,
                "model_path": str(sam3d_model_path),
                "yolo_model": args.yolo_model,
                "yolo_release": args.yolo_release,
                "yolo_weight_path": str(yolo_weight_path),
            }
        )
        raise SystemExit(0)

    print(
        run_video_on_modal.local(
            video_path=args.video_path,
            model=args.model,
            model_revision=args.model_revision,
            sample_fps=args.sample_fps,
            max_frames=args.max_frames,
            sample_mode=args.sample_mode,
            frame_stride=args.frame_stride,
            yolo_model=args.yolo_model,
            yolo_release=args.yolo_release,
            yolo_confidence=args.yolo_confidence,
            bbox_padding=args.bbox_padding,
        )
    )
