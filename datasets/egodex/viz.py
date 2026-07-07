"""Skeleton overlay rendering for EgoDex episode frames."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .features import FPS
from .schemas import CAMERA, SKELETON_TRANSFORMS

_LEFT_COLOR = (34, 211, 238)  # cyan
_RIGHT_COLOR = (245, 158, 11)  # amber
_LEFT_JOINT_COUNT = 34  # skeleton layout: left side first, then right, then body


def _resolve_episode(root: str | Path, task: str, episode_id: int) -> Path:
    """Find `<root>/**/<task>/<episode_id>.hdf5` (tasks may sit under part dirs)."""
    root = Path(root)
    direct = root / task / f"{episode_id}.hdf5"
    if direct.exists():
        return direct
    matches = sorted(root.glob(f"**/{task}/{episode_id}.hdf5"))
    if not matches:
        raise FileNotFoundError(f"No episode {task}/{episode_id}.hdf5 under {root}")
    return matches[0]


def _read_frame_geometry(h5, frame_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if frame_index < 0:
        raise IndexError("frame_index must be non-negative")

    transforms = [h5[name] for name in SKELETON_TRANSFORMS]
    num_frames = transforms[0].shape[0]
    if frame_index >= num_frames:
        raise IndexError(f"frame_index {frame_index} is outside episode bounds [0, {num_frames - 1}]")

    joints = np.concatenate([np.asarray(dataset[frame_index, :3, 3]) for dataset in transforms]).reshape(
        len(SKELETON_TRANSFORMS), 3
    )
    camera_pose = np.asarray(h5[CAMERA][frame_index], dtype=np.float64)
    intrinsic = np.asarray(h5["camera/intrinsic"][()], dtype=np.float64)
    return joints.astype(np.float64), camera_pose, intrinsic


def _extract_video_frame(mp4_path: Path, frame_index: int):
    from PIL import Image

    if not mp4_path.exists():
        raise FileNotFoundError(f"No sibling video found for overlay: {mp4_path}")

    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "frame.png"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{frame_index / FPS:.5f}",
                "-i",
                str(mp4_path),
                "-frames:v",
                "1",
                str(png),
                "-loglevel",
                "error",
            ],
            check=True,
        )
        return Image.open(png).convert("RGB")


def overlay(root: str | Path, task: str, episode_id: int, frame_index: int, radius: int = 5):
    """Return a PIL image of one EgoDex frame with its 68-joint skeleton drawn on it.

    Reads the frame's skeleton and camera extrinsics from the episode HDF5,
    extracts the matching frame from the sibling ``.mp4`` (snapped to
    ``frame_index / fps``), projects the world joints to pixels with the
    episode's own camera intrinsics, and draws them (left = cyan, right = amber).
    """
    import h5py
    from PIL import ImageDraw

    hdf5_path = _resolve_episode(root, task, episode_id)
    mp4_path = hdf5_path.with_suffix(".mp4")

    with h5py.File(hdf5_path, "r") as h:
        joints, camera_pose, intrinsic = _read_frame_geometry(h, frame_index)

    image = _extract_video_frame(mp4_path, frame_index)
    world_to_camera = np.linalg.inv(camera_pose.reshape(4, 4))

    camera = (world_to_camera @ np.hstack([joints, np.ones((68, 1))]).T).T[:, :3]
    depth = camera[:, 2]
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    pixels_u = fx * camera[:, 0] / depth + cx
    pixels_v = fy * camera[:, 1] / depth + cy
    draw = ImageDraw.Draw(image)
    for joint in range(68):
        if depth[joint] <= 0:
            continue
        color = _LEFT_COLOR if joint < _LEFT_JOINT_COUNT else _RIGHT_COLOR
        u, v = pixels_u[joint], pixels_v[joint]
        draw.ellipse([u - radius, v - radius, u + radius, v + radius], fill=color)
    return image


__all__ = ["overlay"]
