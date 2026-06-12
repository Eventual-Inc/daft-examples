# /// script
# description = "Run YOLO-tracked SAM 3D Body mesh recovery on sampled video frames"
# requires-python = ">=3.11, <3.13"
# dependencies = [
#   "daft>=0.7.10",
#   "huggingface_hub",
#   "modal",
#   "ultralytics>=8.3.237",
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
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import modal

from models.common.modal_infra import APP_DIR, INPUT_DIR, MODEL_CACHE_DIR, OUTPUT_DIR
from models.common.weights import (
    DEFAULT_YOLO_RELEASE,
    normalize_hf_token_env,
    resolve_hf_file_path,
    resolve_hf_model_path,
    resolve_yolo_weight_path,
)
from models.sam3d_body.modal_app import (
    GPU_TYPE,
    SAM3D_REPO_DIR,
    base_image,
)
from models.sam3d_body.model import DEFAULT_MODEL, build_dataframe
from models.weights import MODEL_CACHE, OUTPUTS, function_kwargs

DEFAULT_VIDEO = os.environ.get("SAM3D_BODY_VIDEO", "")
DEFAULT_YOLO_MODEL = "yolov8n.pt"
DEFAULT_SAMPLE_MODE = "stride"
DEFAULT_SAMPLE_FPS = 3.0
DEFAULT_FRAME_STRIDE = 10
DEFAULT_MAX_FRAMES = 0
DEFAULT_SCENE_DETECTOR = "contact"
DEFAULT_SAM3_MODEL = "facebook/sam3"
DEFAULT_SAM3_FILENAME = "sam3.pt"
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
BODY_SKELETON_LINKS = [
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
RIGHT_HAND_LINKS = [
    [41, 24],
    [24, 23],
    [23, 22],
    [22, 21],
    [41, 28],
    [28, 27],
    [27, 26],
    [26, 25],
    [41, 32],
    [32, 31],
    [31, 30],
    [30, 29],
    [41, 36],
    [36, 35],
    [35, 34],
    [34, 33],
    [41, 40],
    [40, 39],
    [39, 38],
    [38, 37],
]
LEFT_HAND_LINKS = [
    [62, 45],
    [45, 44],
    [44, 43],
    [43, 42],
    [62, 49],
    [49, 48],
    [48, 47],
    [47, 46],
    [62, 53],
    [53, 52],
    [52, 51],
    [51, 50],
    [62, 57],
    [57, 56],
    [56, 55],
    [55, 54],
    [62, 61],
    [61, 60],
    [60, 59],
    [59, 58],
]
HAND_SKELETON_LINKS = RIGHT_HAND_LINKS + LEFT_HAND_LINKS
CORE_SKELETON_LINKS = BODY_SKELETON_LINKS + HAND_SKELETON_LINKS
SCENE_BODY_ANCHOR_KEYPOINTS = [
    0,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    41,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
]
HAND_CONTACT_KEYPOINTS = list(range(21, 63))
LEFT_HAND_KEYPOINTS = list(range(42, 63))
RIGHT_HAND_KEYPOINTS = list(range(21, 42))


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
    target_point: str,
    scene_detector: str,
    sam3_model: str,
    sam3_revision: str,
    scene_prompts: str,
    scene_confidence: float,
    scene_imgsz: int,
) -> str:
    stat = Path(path).stat()
    payload = (
        f"{Path(path).name}:{stat.st_size}:{stat.st_mtime_ns}:{sample_fps}:{max_frames}:{sample_mode}:"
        f"{frame_stride}:{model}:{model_revision}:{yolo_model}:{yolo_release}:{yolo_confidence}:{bbox_padding}:"
        f"{target_point}:{scene_detector}:{sam3_model}:{sam3_revision}:{scene_prompts}:{scene_confidence}:{scene_imgsz}"
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


def bbox_area(bbox: list[float]) -> float:
    return max(bbox[2] - bbox[0], 1.0) * max(bbox[3] - bbox[1], 1.0)


def parse_target_point(value: str, width: int, height: int) -> tuple[float, float] | None:
    if not value:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected --target-point as 'x,y', got {value!r}")
    x, y = (float(parts[0]), float(parts[1]))
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return x * width, y * height
    return x, y


def crop_histogram(image, bbox: list[float]):
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1 = max(0, min(x1, width - 1))
    x2 = max(x1 + 1, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(y1 + 1, min(y2, height))

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop_h, crop_w = crop.shape[:2]
    margin_x = max(0, int(crop_w * 0.18))
    margin_y = max(0, int(crop_h * 0.10))
    core = crop[margin_y : crop_h - margin_y or crop_h, margin_x : crop_w - margin_x or crop_w]
    hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 8, 6], [0, 180, 0, 256, 0, 256]).astype("float32")
    hist = hist.reshape(-1)
    norm = float(np.linalg.norm(hist))
    if norm <= 1e-6:
        return None
    return hist / norm


def histogram_similarity(left, right) -> float:
    if left is None or right is None:
        return 0.0
    return float(max(0.0, min(float(left @ right), 1.0)))


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


def score_track_candidate(
    detection: dict,
    width: int,
    height: int,
    target_point: tuple[float, float] | None = None,
    anchor_histogram=None,
) -> float:
    x1, y1, x2, y2 = detection["bbox"]
    box_width = max(x2 - x1, 1.0)
    box_height = max(y2 - y1, 1.0)
    area = box_width * box_height
    center_x, center_y = bbox_center(detection["bbox"])
    center_bias = max(0.0, 1.0 - abs(center_x - width / 2) / (width / 2))
    height_bias = max(0.0, 1.0 - center_y / height)
    aspect_bias = min(box_width / box_height, 2.0)
    area_score = math.sqrt(area / (width * height))
    score = detection["confidence"] + 2.25 * area_score + 0.12 * center_bias + 0.12 * height_bias + 0.08 * aspect_bias
    if target_point:
        frame_diag = (width**2 + height**2) ** 0.5
        distance = ((center_x - target_point[0]) ** 2 + (center_y - target_point[1]) ** 2) ** 0.5
        score += 5.0 * (1.0 - min(distance / frame_diag, 1.0))
    if anchor_histogram is not None:
        score += 1.1 * histogram_similarity(detection.get("histogram"), anchor_histogram)
    return score


def choose_detection_track(
    candidates_by_frame: list[list[dict]],
    frame_sizes: list[tuple[int, int]],
    target_points: list[tuple[float, float] | None] | None = None,
    transition_penalty: float = 9.0,
) -> list[dict]:
    if not candidates_by_frame:
        return []

    anchor_histogram = None
    if target_points and target_points[0]:
        width, height = frame_sizes[0]
        anchor = max(
            candidates_by_frame[0],
            key=lambda candidate: score_track_candidate(
                candidate,
                width,
                height,
                target_point=target_points[0],
            ),
        )
        anchor_histogram = anchor.get("histogram")

    scores: list[list[float]] = []
    backpointers: list[list[int]] = []
    for frame_index, candidates in enumerate(candidates_by_frame):
        width, height = frame_sizes[frame_index]
        target_point = target_points[frame_index] if target_points else None
        if frame_index == 0:
            scores.append(
                [
                    score_track_candidate(
                        candidate,
                        width,
                        height,
                        target_point=target_point,
                        anchor_histogram=anchor_histogram,
                    )
                    for candidate in candidates
                ]
            )
            backpointers.append([-1] * len(candidates))
            continue

        previous_candidates = candidates_by_frame[frame_index - 1]
        previous_scores = scores[frame_index - 1]
        frame_diag = (width**2 + height**2) ** 0.5
        frame_scores = []
        frame_backpointers = []
        for candidate in candidates:
            candidate_center = bbox_center(candidate["bbox"])
            candidate_area = bbox_area(candidate["bbox"])
            best_score = -float("inf")
            best_index = 0
            for previous_index, previous_candidate in enumerate(previous_candidates):
                previous_center = bbox_center(previous_candidate["bbox"])
                previous_area = bbox_area(previous_candidate["bbox"])
                distance = (
                    (candidate_center[0] - previous_center[0]) ** 2 + (candidate_center[1] - previous_center[1]) ** 2
                ) ** 0.5
                size_change = abs(math.log(candidate_area / previous_area))
                appearance = histogram_similarity(candidate.get("histogram"), previous_candidate.get("histogram"))
                transition_score = (
                    -transition_penalty * min(distance / frame_diag, 1.0)
                    - 0.9 * min(size_change, 2.0)
                    + 1.8 * appearance
                )
                score = (
                    previous_scores[previous_index]
                    + score_track_candidate(candidate, width, height, anchor_histogram=anchor_histogram)
                    + transition_score
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
    target_point: str = "",
) -> list[VideoFrame]:
    import cv2
    from ultralytics import YOLO

    yolo_weights = resolve_yolo_weight_path(yolo_model, MODEL_CACHE_DIR, release=yolo_release)
    yolo = YOLO(str(yolo_weights))
    candidates_by_frame = []
    frame_sizes = []
    target_points = []
    candidate_confidence = min(confidence, 0.03)
    for frame in frames:
        image = cv2.imread(frame.image_path)
        height, width = image.shape[:2]
        frame_sizes.append((width, height))
        target_points.append(parse_target_point(target_point, width, height))
        result = yolo.predict(frame.image_path, classes=[0], conf=candidate_confidence, device="cpu", verbose=False)[0]
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                bbox = [float(value) for value in box.xyxy[0].tolist()]
                detections.append(
                    {
                        "bbox": bbox,
                        "confidence": float(box.conf[0]),
                        "histogram": crop_histogram(image, bbox),
                    }
                )

        if not detections:
            detections = [
                {
                    "bbox": [0.0, 0.0, float(width), float(height)],
                    "confidence": 0.0,
                    "histogram": crop_histogram(image, [0.0, 0.0, float(width), float(height)]),
                    "source": "full_frame",
                }
            ]
        for detection in detections:
            detection.setdefault("source", "yolo_track")
        candidates_by_frame.append(detections)

    selected_track = choose_detection_track(candidates_by_frame, frame_sizes, target_points=target_points)
    for frame, selected, (width, height) in zip(frames, selected_track, frame_sizes):
        frame.bbox = clip_bbox(selected["bbox"], width, height, padding=padding)
        frame.detection_confidence = selected["confidence"]
        frame.detection_source = selected["source"]
    return frames


def clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def line_length(line: dict) -> float:
    x1, y1, x2, y2 = line["points_flat"]
    return float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)


def point_segment_distance(point: tuple[float, float], line: dict) -> float:
    x, y = point
    x1, y1, x2, y2 = line["points_flat"]
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    if denom <= 1e-6:
        return float(((x - x1) ** 2 + (y - y1) ** 2) ** 0.5)
    t = max(0.0, min(((x - x1) * dx + (y - y1) * dy) / denom, 1.0))
    px = x1 + t * dx
    py = y1 + t * dy
    return float(((x - px) ** 2 + (y - py) ** 2) ** 0.5)


def line_payload(kind: str, line: dict, width: int, height: int, confidence: float) -> dict:
    x1, y1, x2, y2 = line["points_flat"]
    return {
        "kind": kind,
        "type": "line",
        "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
        "normalized": [[float(x1 / width), float(y1 / height)], [float(x2 / width), float(y2 / height)]],
        "angle_degrees": float(line["angle"]),
        "length_px": float(line["length"]),
        "confidence": clamp01(confidence),
        "source": "opencv_hough_lines_p",
    }


def select_scene_line(lines: list[dict], predicate, scorer) -> tuple[dict, float] | tuple[None, float]:
    candidates = [line for line in lines if predicate(line)]
    if not candidates:
        return None, 0.0
    scored = [(scorer(line), line) for line in candidates]
    score, line = max(scored, key=lambda item: item[0])
    return line, score


def finite_keypoints_2d(skeleton: dict, indices: list[int]) -> list[list[float]]:
    keypoints = skeleton.get("keypoints_2d") or []
    points = []
    for index in indices:
        if index >= len(keypoints):
            continue
        point = keypoints[index]
        if point and len(point) == 2 and all(math.isfinite(float(value)) for value in point):
            points.append([float(point[0]), float(point[1])])
    return points


def line_with_contact_metrics(line: dict, points: list[list[float]]) -> dict:
    distances = [point_segment_distance((point[0], point[1]), line) for point in points]
    if not distances:
        return {"contact_distance_px": float("inf"), "contact_distance_p20_px": float("inf")}
    distances = sorted(distances)
    percentile_index = min(round((len(distances) - 1) * 0.2), len(distances) - 1)
    return {
        "contact_distance_px": float(sum(distances) / len(distances)),
        "contact_distance_p20_px": float(distances[percentile_index]),
    }


def hand_contact_pole_line(lines: list[dict], skeleton: dict, width: int, height: int) -> tuple[dict | None, float]:
    hand_points = finite_keypoints_2d(skeleton, HAND_CONTACT_KEYPOINTS)
    if len(hand_points) < 8:
        return None, 0.0

    center_x = sum(point[0] for point in hand_points) / len(hand_points)
    center_y = sum(point[1] for point in hand_points) / len(hand_points)
    frame_diag = (width**2 + height**2) ** 0.5
    candidate_scores = []
    for line in lines:
        if not 20.0 <= line["angle"] <= 82.0:
            continue
        if line["length"] < width * 0.035:
            continue
        metrics = line_with_contact_metrics(line, hand_points)
        contact_p20 = metrics["contact_distance_p20_px"]
        center_distance = point_segment_distance((center_x, center_y), line)
        if contact_p20 > height * 0.12 and center_distance > height * 0.18:
            continue
        contact_score = 1.0 - min(contact_p20 / max(height * 0.16, 1.0), 1.0)
        center_score = 1.0 - min(center_distance / max(height * 0.22, 1.0), 1.0)
        length_score = min(line["length"] / max(frame_diag * 0.45, 1.0), 1.0)
        score = 0.58 * contact_score + 0.24 * center_score + 0.18 * length_score
        candidate = {**line, **metrics, "contact_center_distance_px": float(center_distance)}
        candidate_scores.append((score, candidate))

    if not candidate_scores:
        return None, 0.0
    score, line = max(candidate_scores, key=lambda item: item[0])
    return line, score


def append_or_replace_pole_line(scene_lines: list[dict], candidate: dict, width: int, height: int, confidence: float) -> None:
    payload = line_payload("pole", candidate, width, height, confidence)
    payload["source"] = "opencv_hough_hand_contact"
    payload["contact_distance_px"] = candidate.get("contact_distance_px")
    payload["contact_distance_p20_px"] = candidate.get("contact_distance_p20_px")
    payload["contact_center_distance_px"] = candidate.get("contact_center_distance_px")

    existing_index = next((index for index, line in enumerate(scene_lines) if line.get("kind") == "pole"), None)
    if existing_index is None:
        scene_lines.append(payload)
        return

    existing = scene_lines[existing_index]
    existing_confidence = float(existing.get("confidence") or 0.0)
    existing_contact = float(existing.get("contact_distance_p20_px") or float("inf"))
    candidate_contact = float(payload.get("contact_distance_p20_px") or float("inf"))
    if candidate_contact + height * 0.04 < existing_contact or confidence > existing_confidence + 0.12:
        scene_lines[existing_index] = payload


def detect_scene_2d(frame: VideoFrame, skeleton: dict | None = None, contact_aware: bool = True) -> dict:
    import cv2
    import numpy as np

    image = cv2.imread(frame.image_path)
    if image is None:
        return {}

    height, width = image.shape[:2]
    scale = min(1.0, 960.0 / max(width, 1))
    work = cv2.resize(image, (round(width * scale), round(height * scale))) if scale < 1 else image
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 160)
    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=max(60, round(work.shape[1] * 0.08)),
        maxLineGap=18,
    )

    lines = []
    if raw_lines is not None:
        inv_scale = 1.0 / scale
        for raw in raw_lines[:, 0, :]:
            x1, y1, x2, y2 = [float(value) * inv_scale for value in raw]
            length = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
            if length < width * 0.08:
                continue
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            angle = min(angle, 180.0 - angle)
            lines.append(
                {
                    "points_flat": [x1, y1, x2, y2],
                    "length": length,
                    "angle": angle,
                    "mid_x": (x1 + x2) / 2,
                    "mid_y": (y1 + y2) / 2,
                }
            )

    contact_lines = []
    if contact_aware and skeleton:
        contact_edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 15, 60)
        raw_contact_lines = cv2.HoughLinesP(
            contact_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=20,
            minLineLength=max(35, round(work.shape[1] * 0.02)),
            maxLineGap=50,
        )
        if raw_contact_lines is not None:
            inv_scale = 1.0 / scale
            for raw in raw_contact_lines[:, 0, :]:
                x1, y1, x2, y2 = [float(value) * inv_scale for value in raw]
                length = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
                if length < width * 0.035:
                    continue
                angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
                angle = min(angle, 180.0 - angle)
                contact_lines.append(
                    {
                        "points_flat": [x1, y1, x2, y2],
                        "length": length,
                        "angle": angle,
                        "mid_x": (x1 + x2) / 2,
                        "mid_y": (y1 + y2) / 2,
                    }
                )

    if not lines:
        scene_lines = []
        contact_candidate, contact_score = hand_contact_pole_line(contact_lines, skeleton or {}, width, height)
        if contact_candidate:
            append_or_replace_pole_line(scene_lines, contact_candidate, width, height, contact_score)
        return {
            "format": "heuristic_scene_2d_v1",
            "image_size": [width, height],
            "lines": scene_lines,
            "notes": "Hough-line heuristics with optional hand-contact pole scoring; not calibrated 3D reconstruction.",
        }

    frame_diag = (width**2 + height**2) ** 0.5
    bbox = frame.bbox if len(frame.bbox) == 4 else [width * 0.35, height * 0.25, width * 0.65, height * 0.75]
    athlete_center = bbox_center(bbox)

    pole_line, pole_score = select_scene_line(
        lines,
        lambda line: 18.0 <= line["angle"] <= 78.0 and line["length"] > width * 0.16,
        lambda line: (line["length"] / frame_diag)
        + 0.85 * (1.0 - min(point_segment_distance(athlete_center, line) / frame_diag, 1.0)),
    )
    bar_line, bar_score = select_scene_line(
        lines,
        lambda line: line["angle"] <= 10.0 and line["length"] > width * 0.08 and line["mid_y"] < height * 0.68,
        lambda line: (line["length"] / width)
        + 0.45 * (1.0 - min(abs(line["mid_x"] - athlete_center[0]) / max(width / 2, 1), 1.0))
        + 0.25 * (1.0 - line["mid_y"] / height),
    )
    runway_candidates = [
        line
        for line in lines
        if line["angle"] <= 14.0 and line["length"] > width * 0.16 and line["mid_y"] > height * 0.42
    ]
    runway_candidates.sort(key=lambda line: line["length"], reverse=True)

    scene_lines = []
    if pole_line:
        scene_lines.append(line_payload("pole", pole_line, width, height, pole_score))
    if bar_line:
        scene_lines.append(line_payload("bar", bar_line, width, height, bar_score))
    for runway_line in runway_candidates[:3]:
        confidence = runway_line["length"] / width
        scene_lines.append(line_payload("runway", runway_line, width, height, confidence))

    contact_candidate, contact_score = hand_contact_pole_line(contact_lines or lines, skeleton or {}, width, height)
    if contact_candidate:
        append_or_replace_pole_line(scene_lines, contact_candidate, width, height, contact_score)

    return {
        "format": "heuristic_scene_2d_v1",
        "image_size": [width, height],
        "lines": scene_lines,
        "notes": "Hough-line heuristics with optional hand-contact pole scoring; not calibrated 3D reconstruction.",
    }


def resolve_sam3_weight_path(model: str, revision: str = "") -> Path:
    model_path = Path(model).expanduser()
    if model_path.exists() or model_path.suffix == ".pt":
        return model_path
    return resolve_hf_file_path(
        model,
        DEFAULT_SAM3_FILENAME,
        MODEL_CACHE_DIR,
        revision=revision or None,
        token=normalize_hf_token_env(),
    )


def scene_prompt_groups(value: str) -> dict[str, list[str]]:
    defaults = {
        "pole": ["pole vault pole", "vaulting pole"],
        "bar": ["pole vault crossbar", "crossbar"],
        "standard": ["pole vault standard", "upright standard"],
    }
    if not value:
        return defaults

    groups: dict[str, list[str]] = {}
    for chunk in value.split(";"):
        if not chunk.strip():
            continue
        if "=" not in chunk:
            raise ValueError("Expected --scene-prompts as 'kind=prompt one,prompt two;kind=prompt'.")
        kind, prompts = chunk.split("=", 1)
        kind = kind.strip()
        values = [prompt.strip() for prompt in prompts.split(",") if prompt.strip()]
        if kind and values:
            groups[kind] = values
    return groups or defaults


def mask_points(mask, polygon=None):
    import numpy as np

    if polygon is not None and len(polygon):
        points = np.asarray(polygon, dtype="float32")
        if points.ndim == 2 and points.shape[1] == 2:
            return points

    if mask is None:
        return np.empty((0, 2), dtype="float32")
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = mask.squeeze()
    if mask.size == 0:
        return np.empty((0, 2), dtype="float32")
    ys, xs = np.nonzero(mask > 0.5)
    if len(xs) == 0:
        return np.empty((0, 2), dtype="float32")
    return np.column_stack([xs, ys]).astype("float32")


def pca_centerline(points, kind: str) -> tuple[list[list[float]], float]:
    import numpy as np

    if len(points) < 12:
        return [], 0.0
    mean = points.mean(axis=0)
    centered = points - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    projections = centered @ direction
    start_projection = float(np.percentile(projections, 2))
    end_projection = float(np.percentile(projections, 98))
    if end_projection <= start_projection:
        return [], 0.0

    if kind == "pole":
        bins = np.linspace(start_projection, end_projection, 12)
        polyline = []
        for start, end in zip(bins[:-1], bins[1:]):
            mask = (projections >= start) & (projections <= end)
            if int(mask.sum()) < 3:
                continue
            point = np.median(points[mask], axis=0)
            polyline.append([float(point[0]), float(point[1])])
        if len(polyline) >= 2:
            return polyline, float(end_projection - start_projection)

    start = mean + start_projection * direction
    end = mean + end_projection * direction
    return [[float(start[0]), float(start[1])], [float(end[0]), float(end[1])]], float(end_projection - start_projection)


def line_angle_from_points(points: list[list[float]]) -> float:
    if len(points) < 2:
        return 0.0
    start = points[0]
    end = points[-1]
    angle = abs(math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])))
    return min(angle, 180.0 - angle)


def semantic_scene_payload(
    kind: str,
    points: list[list[float]],
    width: int,
    height: int,
    confidence: float,
    source: str,
    mask_area: int,
) -> dict | None:
    if len(points) < 2:
        return None
    length = sum(
        math.hypot(points[index + 1][0] - points[index][0], points[index + 1][1] - points[index][1])
        for index in range(len(points) - 1)
    )
    if length < width * 0.025:
        return None
    return {
        "kind": kind,
        "type": "polyline" if len(points) > 2 else "line",
        "points": [[float(point[0]), float(point[1])] for point in points],
        "normalized": [[float(point[0] / width), float(point[1] / height)] for point in points],
        "angle_degrees": line_angle_from_points(points),
        "length_px": float(length),
        "confidence": clamp01(confidence),
        "mask_area_px": int(mask_area),
        "source": source,
    }


def sam3_result_scene_lines(result, kind: str, width: int, height: int) -> list[dict]:
    import numpy as np

    masks = getattr(result, "masks", None)
    if masks is None:
        return []
    polygons = list(getattr(masks, "xy", []) or [])
    mask_data = getattr(masks, "data", None)
    mask_items = []
    if mask_data is not None:
        if hasattr(mask_data, "detach"):
            mask_data = mask_data.detach().cpu().numpy()
        mask_items = list(np.asarray(mask_data))
    count = max(len(polygons), len(mask_items))
    if count == 0:
        return []

    confidences = []
    boxes = getattr(result, "boxes", None)
    if boxes is not None and getattr(boxes, "conf", None) is not None:
        conf = boxes.conf
        if hasattr(conf, "detach"):
            conf = conf.detach().cpu().numpy()
        confidences = [float(value) for value in np.asarray(conf).reshape(-1)]

    lines = []
    for index in range(count):
        polygon = polygons[index] if index < len(polygons) else None
        mask_item = mask_items[index] if index < len(mask_items) else None
        points = mask_points(mask_item, polygon=polygon)
        if len(points) < 12:
            continue
        centerline, length = pca_centerline(points, kind)
        if not centerline:
            continue
        angle = line_angle_from_points(centerline)
        if kind == "bar" and angle > 16.0:
            continue
        if kind == "standard" and angle < 58.0:
            continue
        if kind == "pole" and not 12.0 <= angle <= 86.0:
            continue
        confidence = confidences[index] if index < len(confidences) else min(length / max(width * 0.4, 1.0), 1.0)
        payload = semantic_scene_payload(
            kind,
            centerline,
            width,
            height,
            confidence,
            "sam3_semantic_mask",
            mask_area=len(points),
        )
        if payload:
            lines.append(payload)
    return sorted(lines, key=lambda line: line.get("confidence", 0.0), reverse=True)


def sam3_inference_device() -> int | str:
    import torch

    if torch.cuda.is_available():
        return 0
    if os.environ.get("SAM3_ALLOW_CPU", "").lower() in {"1", "true", "yes"}:
        return "cpu"
    raise RuntimeError(
        "SAM3 semantic scene detection needs CUDA for practical video use in this pipeline. "
        "Set SAM3_ALLOW_CPU=1 to force the slow CPU path, or run SAM3 in a GPU-visible process."
    )


def detect_scene_2d_sam3(
    frames: list[VideoFrame],
    sam3_model: str,
    sam3_revision: str,
    scene_prompts: str,
    confidence: float,
    imgsz: int,
) -> dict[int, dict]:
    import cv2
    from ultralytics.models.sam import SAM3SemanticPredictor

    if not frames:
        return {}
    model_path = resolve_sam3_weight_path(sam3_model, revision=sam3_revision)
    prompt_groups = scene_prompt_groups(scene_prompts)
    overrides = {
        "conf": confidence,
        "task": "segment",
        "mode": "predict",
        "model": str(model_path),
        "imgsz": imgsz,
        "half": False,
        "device": sam3_inference_device(),
        "save": False,
        "verbose": False,
    }
    predictor = SAM3SemanticPredictor(overrides=overrides)

    scenes: dict[int, dict] = {}
    for frame in frames:
        image = cv2.imread(frame.image_path)
        if image is None:
            continue
        height, width = image.shape[:2]
        frame_lines = []
        predictor.set_image(frame.image_path)
        for kind, prompts in prompt_groups.items():
            for prompt in prompts:
                results = predictor(text=[prompt])
                if not isinstance(results, (list, tuple)):
                    results = [results]
                for result in results:
                    frame_lines.extend(sam3_result_scene_lines(result, kind, width, height)[:2])
        scenes[frame.index] = {
            "format": "sam3_scene_2d_v1",
            "image_size": [width, height],
            "lines": frame_lines,
            "prompts": prompt_groups,
            "notes": "SAM 3 semantic masks converted into pole-vault scene geometry; not calibrated 3D reconstruction.",
        }
    return scenes


def merge_scene_2d(primary: dict | None, fallback: dict | None) -> dict:
    if not primary:
        return fallback or {}
    if not fallback:
        return primary

    merged_lines = []
    used_kinds = set()
    for line in primary.get("lines") or []:
        merged_lines.append(line)
        used_kinds.add(line.get("kind"))
    for line in fallback.get("lines") or []:
        kind = line.get("kind")
        if kind in {"pole", "bar", "standard"} and kind in used_kinds:
            continue
        merged_lines.append(line)
    return {
        **fallback,
        **primary,
        "format": "hybrid_scene_2d_v1",
        "lines": merged_lines,
        "fallback_format": fallback.get("format"),
        "primary_format": primary.get("format"),
    }


def distance_to_polyline(point: tuple[float, float], points: list[list[float]]) -> float:
    if len(points) < 2:
        return float("inf")
    return min(
        point_segment_distance(
            point,
            {
                "points_flat": [
                    float(points[index][0]),
                    float(points[index][1]),
                    float(points[index + 1][0]),
                    float(points[index + 1][1]),
                ]
            },
        )
        for index in range(len(points) - 1)
    )


def scene_contact_from_skeleton(scene_2d: dict, skeleton: dict) -> dict:
    pole_lines = [
        line
        for line in scene_2d.get("lines") or []
        if line.get("kind") == "pole" and isinstance(line.get("points"), list) and len(line["points"]) >= 2
    ]
    if not pole_lines:
        return {"format": "pole_contact_v1", "has_pole": False}

    def hand_summary(indices: list[int]) -> dict:
        points = finite_keypoints_2d(skeleton, indices)
        if not points:
            return {"num_points": 0}
        distances = []
        for point in points:
            distances.append(min(distance_to_polyline((point[0], point[1]), pole["points"]) for pole in pole_lines))
        distances.sort()
        median_distance = distances[len(distances) // 2]
        p20_distance = distances[min(round((len(distances) - 1) * 0.2), len(distances) - 1)]
        return {
            "num_points": len(points),
            "median_distance_px": float(median_distance),
            "p20_distance_px": float(p20_distance),
            "holding_confidence": clamp01(1.0 - p20_distance / 90.0),
        }

    left = hand_summary(LEFT_HAND_KEYPOINTS)
    right = hand_summary(RIGHT_HAND_KEYPOINTS)
    confidences = [
        hand.get("holding_confidence", 0.0)
        for hand in [left, right]
        if hand.get("num_points", 0)
    ]
    return {
        "format": "pole_contact_v1",
        "has_pole": True,
        "left_hand": left,
        "right_hand": right,
        "holding_confidence": max(confidences) if confidences else 0.0,
        "pole_sources": sorted({str(line.get("source", "")) for line in pole_lines if line.get("source")}),
    }


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
        keypoints_2d = data["pred_keypoints_2d"]
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
    finite_2d = np.isfinite(keypoints_2d).all(axis=1)

    return {
        "format": "mhr70_viewer_basis_v1",
        "keypoints_2d": [
            [float(value) for value in keypoint] if bool(finite_2d[index]) else None
            for index, keypoint in enumerate(keypoints_2d)
        ],
        "keypoints": [
            [float(value) for value in keypoint] if bool(finite[index]) else None
            for index, keypoint in enumerate(keypoints)
        ],
    }


def finite_keypoints(skeleton: dict) -> list[list[float]]:
    keypoints = skeleton.get("keypoints") or []
    return [point for point in keypoints if point and all(math.isfinite(float(value)) for value in point)]


def scene_3d_from_frame(scene_2d: dict, skeleton: dict) -> dict:
    points = finite_keypoints(skeleton)
    if not points:
        return {}
    keypoints_3d = skeleton.get("keypoints") or []
    keypoints_2d = skeleton.get("keypoints_2d") or []

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [point[2] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2
    body_height = max(max_y - min_y, 1.0)
    body_width = max(max_x - min_x, 0.45)
    scene_width = max(body_width * 3.0, 1.4)
    scene_depth = max(body_height * 3.2, 3.2)
    floor_y = min_y - body_height * 0.06

    primitives = [
        {
            "kind": "runway",
            "type": "plane",
            "corners": [
                [center_x - scene_width / 2, floor_y, center_z - scene_depth / 2],
                [center_x + scene_width / 2, floor_y, center_z - scene_depth / 2],
                [center_x + scene_width / 2, floor_y, center_z + scene_depth / 2],
                [center_x - scene_width / 2, floor_y, center_z + scene_depth / 2],
            ],
            "confidence": 0.35,
            "source": "skeleton_bounds_reference",
        }
    ]

    image_size = scene_2d.get("image_size") or [1, 1]
    image_width = max(float(image_size[0]), 1.0)
    image_height = max(float(image_size[1]), 1.0)
    x_scale = max(scene_width, 1.4)
    y_scale = max(body_height * 2.2, 2.2)

    def is_finite_vector(point: object, length: int) -> bool:
        return (
            isinstance(point, (list, tuple))
            and len(point) == length
            and all(math.isfinite(float(value)) for value in point)
        )

    paired_body_points = []
    for index in SCENE_BODY_ANCHOR_KEYPOINTS:
        if index >= len(keypoints_2d) or index >= len(keypoints_3d):
            continue
        point_2d = keypoints_2d[index]
        point_3d = keypoints_3d[index]
        if is_finite_vector(point_2d, 2) and is_finite_vector(point_3d, 3):
            paired_body_points.append(
                (
                    [float(point_2d[0]), float(point_2d[1])],
                    [float(point_3d[0]), float(point_3d[1]), float(point_3d[2])],
                )
            )

    def median(values: list[float]) -> float:
        values = sorted(values)
        midpoint = len(values) // 2
        if len(values) % 2:
            return values[midpoint]
        return (values[midpoint - 1] + values[midpoint]) / 2

    def line_y_at_x(line_points: list[list[float]], x_value: float) -> float:
        start, end = line_points
        dx = end[0] - start[0]
        if abs(dx) < 1e-6:
            return (start[1] + end[1]) / 2
        return start[1] + (x_value - start[0]) * (end[1] - start[1]) / dx

    def point_line_distance(point: list[float], start: list[float], end: list[float]) -> float:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        denom = math.hypot(dx, dy)
        if denom < 1e-6:
            return math.hypot(point[0] - start[0], point[1] - start[1])
        return abs(dy * point[0] - dx * point[1] + end[0] * start[1] - end[1] * start[0]) / denom

    def map_image_point(point: list[float]) -> list[float]:
        x, y = point
        return [
            center_x + (float(x) / image_width - 0.5) * x_scale,
            center_y + (0.5 - float(y) / image_height) * y_scale,
            center_z - scene_depth * 0.18,
        ]

    def line_anchor_from_body(line_points: list[list[float]], blend_image_height: bool = False) -> list[float] | None:
        if len(line_points) != 2 or not paired_body_points:
            return None
        start = [float(line_points[0][0]), float(line_points[0][1])]
        end = [float(line_points[1][0]), float(line_points[1][1])]
        anchor_2d_x = median([point_2d[0] for point_2d, _point_3d in paired_body_points])
        anchor_2d_y = line_y_at_x([start, end], anchor_2d_x)
        mapped_anchor_y = map_image_point([anchor_2d_x, anchor_2d_y])[1]
        line_window = max(image_height * 0.075, 60.0)
        x_window = max(image_width * 0.18, 240.0)

        weighted = [0.0, 0.0, 0.0]
        total_weight = 0.0
        for point_2d, point_3d in paired_body_points:
            line_distance = point_line_distance(point_2d, start, end)
            x_distance = abs(point_2d[0] - anchor_2d_x)
            weight = 1.0 / (1.0 + (line_distance / line_window) ** 2)
            weight *= 1.0 / (1.0 + (x_distance / x_window) ** 2)
            total_weight += weight
            weighted[0] += point_3d[0] * weight
            weighted[1] += point_3d[1] * weight
            weighted[2] += point_3d[2] * weight

        if total_weight <= 1e-6:
            return None

        anchor = [value / total_weight for value in weighted]
        if blend_image_height:
            # The 2D line fixes the crossbar height in image space, while nearby
            # body keypoints keep the reference line at the athlete's local depth.
            anchor[1] = anchor[1] * 0.75 + mapped_anchor_y * 0.25
        anchor[1] = min(max(anchor[1], min_y - body_height * 0.04), max_y + body_height * 0.08)
        anchor[2] = min(max(anchor[2], min_z - body_width * 0.12), max_z + body_width * 0.12)
        return anchor

    for line in scene_2d.get("lines") or []:
        kind = line.get("kind")
        if kind not in {"pole", "bar", "standard"}:
            continue
        line_points = line.get("points", [])
        mapped_points = [map_image_point(point) for point in line_points]
        if len(mapped_points) < 2:
            continue
        source = "heuristic_2d_line_reference"
        if kind == "bar":
            anchor = line_anchor_from_body([line_points[0], line_points[-1]], blend_image_height=True)
            if anchor:
                half_span = max(body_width * 0.9, 0.78)
                mapped_points = [
                    [anchor[0] - half_span, anchor[1], anchor[2]],
                    [anchor[0] + half_span, anchor[1], anchor[2]],
                ]
                source = "heuristic_2d_line_body_anchored_reference"
            else:
                bar_y = (mapped_points[0][1] + mapped_points[1][1]) / 2
                bar_y = min(max(bar_y, min_y - body_height * 0.04), max_y + body_height * 0.08)
                mapped_points[0][1] = bar_y
                mapped_points[1][1] = bar_y
                mapped_points[0][2] = center_z
                mapped_points[1][2] = center_z
        elif kind == "pole":
            anchor = line_anchor_from_body([line_points[0], line_points[-1]])
            if anchor:
                for point in mapped_points:
                    point[2] = anchor[2]
                source = "heuristic_2d_line_body_anchored_reference"
        elif kind == "standard":
            for point in mapped_points:
                point[2] = center_z - scene_depth * 0.22
        primitives.append(
            {
                "kind": kind,
                "type": "polyline" if line.get("type") == "polyline" or len(mapped_points) > 2 else "line",
                "points": mapped_points,
                "confidence": line.get("confidence", 0.0),
                "source": source,
            }
        )

    return {
        "format": "heuristic_scene_3d_v1",
        "coordinate_system": "mhr70_viewer_basis_reference",
        "primitives": primitives,
        "notes": "Reference geometry estimated from 2D lines and body bounds; not camera-calibrated.",
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
    target_point: str,
    scene_detector: str,
    sam3_model: str,
    sam3_revision: str,
    scene_prompts: str,
    scene_confidence: float,
    scene_imgsz: int,
    profile: dict | None = None,
) -> dict:
    meshes_dir = sequence_dir / "meshes"
    renders_dir = sequence_dir / "renders"
    meshes_dir.mkdir(parents=True, exist_ok=True)
    renders_dir.mkdir(parents=True, exist_ok=True)

    semantic_scenes = {}
    scene_status = {
        "detector": scene_detector,
        "sam3_model": sam3_model if scene_detector == "sam3" else "",
        "sam3_revision": sam3_revision if scene_detector == "sam3" else "",
    }
    if scene_detector == "sam3":
        try:
            semantic_scenes = detect_scene_2d_sam3(
                frames,
                sam3_model=sam3_model,
                sam3_revision=sam3_revision,
                scene_prompts=scene_prompts,
                confidence=scene_confidence,
                imgsz=scene_imgsz,
            )
            scene_status["sam3_frames"] = len(semantic_scenes)
        except Exception as exc:
            scene_status["sam3_error"] = f"{type(exc).__name__}: {exc}"

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
        fallback_scene_2d = detect_scene_2d(frame, skeleton=skeleton, contact_aware=scene_detector != "heuristic")
        scene_2d = merge_scene_2d(semantic_scenes.get(frame.index), fallback_scene_2d)
        scene_3d = scene_3d_from_frame(scene_2d, skeleton) if skeleton else {}
        contact = scene_contact_from_skeleton(scene_2d, skeleton) if skeleton else {}
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
                "scene_2d": scene_2d,
                "scene_3d": scene_3d,
                "contact": contact,
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
        "target_point": target_point,
        "scene_detector": scene_detector,
        "profile": profile or {},
        "mesh_coordinate_system": "sam3d_renderer_ply",
        "mesh_rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
        "skeleton": {
            "format": "mhr70_viewer_basis_v1",
            "keypoint_names": MHR70_KEYPOINT_NAMES,
            "links": CORE_SKELETON_LINKS,
            "body_links": BODY_SKELETON_LINKS,
            "hand_links": HAND_SKELETON_LINKS,
        },
        "scene": {
            "format": "hybrid_scene_v1",
            "primitives": ["runway", "bar", "pole", "standard"],
            "status": scene_status,
            "prompts": scene_prompt_groups(scene_prompts),
            "notes": (
                "Runway/bar/pole/standards are visual references, not calibrated 3D reconstructions. "
                "The contact detector constrains pole candidates with SAM hand keypoints; SAM 3 semantic masks "
                "are used when --scene-detector sam3 and sam3.pt are available."
            ),
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
video_image = base_image.pip_install(
    "ultralytics>=8.3.237",
    "git+https://github.com/ultralytics/CLIP.git",
).add_local_python_source("models", "pipelines")


@app.function(**function_kwargs(video_image, cpu=4, enable_memory_snapshot=False))
def download_video_model_weights(
    model: str = DEFAULT_MODEL,
    model_revision: str = "",
    yolo_model: str = DEFAULT_YOLO_MODEL,
    yolo_release: str = DEFAULT_YOLO_RELEASE,
    scene_detector: str = DEFAULT_SCENE_DETECTOR,
    sam3_model: str = DEFAULT_SAM3_MODEL,
    sam3_revision: str = "",
) -> dict:
    os.chdir(APP_DIR)
    sam3d_model_path = resolve_hf_model_path(
        model,
        MODEL_CACHE_DIR,
        revision=model_revision or None,
        token=normalize_hf_token_env(),
    )
    yolo_weight_path = resolve_yolo_weight_path(yolo_model, MODEL_CACHE_DIR, release=yolo_release)
    sam3_weight_path = ""
    if scene_detector == "sam3":
        sam3_weight_path = str(resolve_sam3_weight_path(sam3_model, revision=sam3_revision))
    MODEL_CACHE.commit()
    return {
        "model": model,
        "model_revision": model_revision,
        "model_path": str(sam3d_model_path),
        "yolo_model": yolo_model,
        "yolo_release": yolo_release,
        "yolo_weight_path": str(yolo_weight_path),
        "scene_detector": scene_detector,
        "sam3_model": sam3_model,
        "sam3_revision": sam3_revision,
        "sam3_weight_path": sam3_weight_path,
    }


@app.function(
    **function_kwargs(
        video_image,
        gpu=GPU_TYPE,
        memory=98304,
        # Pipeline adds a third volume (uploaded input videos) on top of the
        # canonical weight + output cache.
        volumes={MODEL_CACHE_DIR: MODEL_CACHE, OUTPUT_DIR: OUTPUTS, INPUT_DIR: input_videos},
    )
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
    target_point: str = "",
    scene_detector: str = DEFAULT_SCENE_DETECTOR,
    sam3_model: str = DEFAULT_SAM3_MODEL,
    sam3_revision: str = "",
    scene_prompts: str = "",
    scene_confidence: float = 0.22,
    scene_imgsz: int = 1024,
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
        target_point,
        scene_detector,
        sam3_model,
        sam3_revision,
        scene_prompts,
        scene_confidence,
        scene_imgsz,
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
        target_point=target_point,
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
        "target_point": target_point,
        "scene_detector": scene_detector,
        "sam3_model": sam3_model if scene_detector == "sam3" else "",
        "sam3_revision": sam3_revision if scene_detector == "sam3" else "",
        "scene_confidence": scene_confidence,
        "scene_imgsz": scene_imgsz,
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
        target_point=target_point,
        scene_detector=scene_detector,
        sam3_model=sam3_model,
        sam3_revision=sam3_revision,
        scene_prompts=scene_prompts,
        scene_confidence=scene_confidence,
        scene_imgsz=scene_imgsz,
        profile=profile,
    )
    profile["seconds"]["write_sequence_outputs"] = time.perf_counter() - profile_before_write
    commit_start = time.perf_counter()
    MODEL_CACHE.commit()
    OUTPUTS.commit()
    profile["seconds"]["commit_volumes"] = time.perf_counter() - commit_start
    profile["seconds"]["total_remote_function"] = time.perf_counter() - profile_start
    profile["seconds_per_frame"] = {key: value / max(len(frames), 1) for key, value in profile["seconds"].items()}
    manifest_path = Path(sequence_result["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["profile"] = profile
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    OUTPUTS.commit()
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
    target_point: str = "",
    scene_detector: str = DEFAULT_SCENE_DETECTOR,
    sam3_model: str = DEFAULT_SAM3_MODEL,
    sam3_revision: str = "",
    scene_prompts: str = "",
    scene_confidence: float = 0.22,
    scene_imgsz: int = 1024,
    download_only: bool = False,
):
    if download_only:
        print(
            download_video_model_weights.remote(
                model=model,
                model_revision=model_revision,
                yolo_model=yolo_model,
                yolo_release=yolo_release,
                scene_detector=scene_detector,
                sam3_model=sam3_model,
                sam3_revision=sam3_revision,
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
            target_point=target_point,
            scene_detector=scene_detector,
            sam3_model=sam3_model,
            sam3_revision=sam3_revision,
            scene_prompts=scene_prompts,
            scene_confidence=scene_confidence,
            scene_imgsz=scene_imgsz,
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
    parser.add_argument(
        "--target-point",
        default="",
        help="Optional athlete identity anchor as 'x,y'. Values in [0,1] are normalized image coordinates.",
    )
    parser.add_argument(
        "--scene-detector",
        choices=["heuristic", "contact", "sam3"],
        default=DEFAULT_SCENE_DETECTOR,
        help="Scene-object detector for pole/bar/standards. sam3 falls back to contact-aware heuristics on errors.",
    )
    parser.add_argument("--sam3-model", default=os.environ.get("SAM3_MODEL", DEFAULT_SAM3_MODEL))
    parser.add_argument("--sam3-revision", default=os.environ.get("SAM3_MODEL_REVISION", ""))
    parser.add_argument(
        "--scene-prompts",
        default="",
        help="Optional SAM3 prompts, e.g. 'pole=pole vault pole;bar=pole vault crossbar;standard=upright standard'.",
    )
    parser.add_argument("--scene-confidence", type=float, default=0.22)
    parser.add_argument("--scene-imgsz", type=int, default=1024)
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
        sam3_weight_path = ""
        if args.scene_detector == "sam3":
            sam3_weight_path = str(resolve_sam3_weight_path(args.sam3_model, revision=args.sam3_revision))
        print(
            {
                "model": args.model,
                "model_revision": args.model_revision,
                "model_path": str(sam3d_model_path),
                "yolo_model": args.yolo_model,
                "yolo_release": args.yolo_release,
                "yolo_weight_path": str(yolo_weight_path),
                "scene_detector": args.scene_detector,
                "sam3_model": args.sam3_model,
                "sam3_revision": args.sam3_revision,
                "sam3_weight_path": sam3_weight_path,
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
            target_point=args.target_point,
            scene_detector=args.scene_detector,
            sam3_model=args.sam3_model,
            sam3_revision=args.sam3_revision,
            scene_prompts=args.scene_prompts,
            scene_confidence=args.scene_confidence,
            scene_imgsz=args.scene_imgsz,
        )
    )
