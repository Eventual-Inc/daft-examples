"""Media artifact helpers for model outputs."""

from __future__ import annotations

import subprocess
from pathlib import Path


def save_video(frames, output_path: Path, fps: int) -> None:
    """Encode an array (or list) of RGB frames straight to an MP4 with ffmpeg.

    Accepts float or uint8 frames in NHWC/NCHW (optionally batched) layouts and
    normalizes them to contiguous uint8 RGB before piping raw video into ffmpeg.
    Writing the artifact to disk keeps large videos out of UDF return values.
    """
    import numpy as np

    if not isinstance(frames, np.ndarray):
        frames = np.stack(
            [np.asarray(frame.convert("RGB") if hasattr(frame, "convert") else frame) for frame in frames]
        )
    if frames.ndim == 5:
        frames = frames[0]
    if frames.ndim == 4 and frames.shape[1] in (1, 3, 4):
        frames = np.transpose(frames, (0, 2, 3, 1))
    if np.issubdtype(frames.dtype, np.floating):
        if frames.min() < 0:
            frames = frames * 0.5 + 0.5
        frames = np.clip(frames, 0, 1) * 255
    frames = frames.astype(np.uint8, copy=False)
    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    elif frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)
    frames = np.ascontiguousarray(frames)
    _, height, width, channels = frames.shape
    if channels != 3:
        raise ValueError(f"Expected RGB frames, got shape {frames.shape}")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        input=frames.tobytes(),
        check=True,
        capture_output=True,
    )
