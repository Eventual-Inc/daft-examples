# /// script
# description = "Read video files "
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft", "av", "numpy"]
# ///

import daft
from daft import DataType
import av
import numpy as np
from fractions import Fraction


# Read video metadata
@daft.func(
    return_dtype=DataType.struct(
        {
            "width": DataType.int32(),
            "height": DataType.int32(),
            "fps": DataType.float64(),
            "duration": DataType.float64(),
            "frame_count": DataType.int32(),
            "time_base": DataType.float64(),
        }
    )
)
def read_video_meta(
    file: daft.File,
    *,
    probesize: str = "64k",
    analyzeduration_us: int = 200_000,
):
    """
    Extract basic video metadata from container headers.

    Returns
    -------
    dict
        width, height, fps, frame_count, time_base, keyframe_pts, keyframe_indices
    """
    options = {
        "probesize": str(probesize),
        "analyzeduration": str(analyzeduration_us),
    }

    with av.open(
        file, mode="r", options=options, metadata_encoding="utf-8"
    ) as container:
        video = next(
            (stream for stream in container.streams if stream.type == "video"),
            None,
        )
        if video is None:
            return {
                "width": None,
                "height": None,
                "fps": None,
                "frame_count": None,
                "time_base": None,
            }

        # Basic stream properties ----------
        width = video.width
        height = video.height
        time_base = float(video.time_base) if video.time_base else None

        # Frame rate -----------------------
        fps = None
        if video.average_rate:
            fps = float(video.average_rate)
        elif video.guessed_rate:
            fps = float(video.guessed_rate)

        # Duration -------------------------
        duration = None
        if container.duration and container.duration > 0:
            duration = container.duration / 1_000_000.0
        elif video.duration:
            # Fallback time_base only for duration computation if missing
            tb_for_dur = (
                float(video.time_base) if video.time_base else (1.0 / 1_000_000.0)
            )
            duration = float(video.duration * tb_for_dur)

        # Frame count -----------------------
        frame_count = video.frames
        if not frame_count or frame_count <= 0:
            if duration and fps:
                frame_count = int(round(duration * fps))
            else:
                frame_count = None

        return {
            "width": width,
            "height": height,
            "fps": fps,
            "duration": duration,
            "frame_count": frame_count,
            "time_base": time_base,
        }


@daft.func(
    return_dtype=DataType.struct(
        {
            "keyframe_index": DataType.uint64(),
            "keyframe_pts": DataType.float64(),
        }
    )
)
def read_video_keyframes(
    file: daft.File,
    *,
    probesize: str = "64k",
    analyzeduration_us: int = 200_000,
):
    options = {
        "probesize": str(probesize),
        "analyzeduration": str(analyzeduration_us),
    }
    with file.to_tempfile() as tmp:
        with av.open(
            file, mode="r", options=options, metadata_encoding="utf-8"
        ) as container:
            video = next(
                (stream for stream in container.streams if stream.type == "video"),
                None,
            )
            if video is None:
                return {
                    "index": [],
                    "pts": [],
                }

            fps = None
            if video.average_rate:
                fps = float(video.average_rate)
            elif video.guessed_rate:
                fps = float(video.guessed_rate)

            keyframe_pts = []
            try:
                for packet in container.demux(video):
                    if packet.is_keyframe and packet.pts is not None:
                        pts_seconds = float(packet.pts * float(video.time_base))
                        keyframe_pts.append(pts_seconds)
            except Exception:
                keyframe_pts = []

            keyframe_indices = (
                [int(round(t * fps)) for t in keyframe_pts] if fps else []
            )

            return {
                "index": keyframe_indices,
                "pts": keyframe_pts,
            }


def pts_time_ns(pts: int | None, time_base: Fraction) -> int | None:
    if pts is None:
        return None
    # exact integer nanoseconds without float rounding
    return (pts * time_base.numerator * 1_000_000_000) // time_base.denominator


@daft.func(
    return_dtype=DataType.struct(
        {
            "path": DataType.string(),
            "stream_index": DataType.int32(),
            "frame_time": DataType.float64(),
            "frame_time_base": DataType.string(),
            "frame_time_ns": DataType.int64(),
            "frame_pts": DataType.float64(),
            "frame_dts": DataType.float64(),
            "frame_duration": DataType.float64(),
            "is_key_frame": DataType.bool(),
            "payload": DataType.binary(),  # stores RGB bytes (H x W x 3)
            "payload_size_bytes": DataType.int64(),
        }
    )
)
def read_video_files(
    file: daft.File,
    *,
    start_sec: float = 0.0,
    end_sec: float = float("inf"),
    probesize: str = "64k",
    analyzeduration_us: int = 200_000,
    width: int = 288,
    height: int = 288,
):
    options = {
        "probesize": str(probesize),
        "analyzeduration": str(analyzeduration_us),
    }
    eps = 1e-6
    with av.open(
        file, mode="r", options=options, metadata_encoding="utf-8"
    ) as container:
        # Select video streams (exclude attached thumbnails)
        vs = next((s for s in container.streams if s.type == "video"), None)
        vs.thread_type = "AUTO"

        # Compute seek position and optional end bound in this stream's ticks
        ts_start = int(start_sec / float(vs.time_base)) if start_sec > 0 else 0
        end_pts = (
            None if end_sec == float("inf") else int(end_sec / float(vs.time_base))
        )

        container.seek(ts_start, stream=vs, any_frame=False, backward=True)

        # Decode frames only from this stream
        for frame in container.decode(vs):
            if frame.pts is None:
                continue

            t = frame.pts * float(vs.time_base)
            if t + eps < start_sec:
                continue
            if end_pts is not None and frame.pts > end_pts:
                break

            # Resize & convert
            if width and height:
                frame = frame.reformat(width=width, height=height)

            image_array = frame.to_ndarray()

            yield {
                "path": str(file),
                "stream_index": int(vs.index),
                "frame_time": float(frame.time),
                "frame_time_base": str(frame.time_base),
                "frame_time_ns": pts_time_ns(frame.pts, frame.time_base),
                "frame_pts": float(frame.pts),
                "frame_dts": float(frame.dts)
                if frame.dts is not None
                else float("nan"),
                "frame_duration": float(frame.duration)
                if frame.duration is not None
                else float("nan"),
                "is_key_frame": bool(frame.key_frame),
                "payload": payload,
                "payload_size_bytes": len(payload),
            }


@daft.func(return_dtype=DataType.tensor(DataType.float32()))
def read_video_audio(
    file: daft.File,
    *,
    target_sr: int = 16_000,
    layout: str = "mono",
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, int]:
    """Decode audio from a video file into a NumPy array.

    Parameters
    ----------
    source:
        Path to a video file that contains an audio stream.
    target_sr:
        Sample rate to resample the audio to. Defaults to 16 kHz.
    layout:
        Channel layout passed to PyAV's resampler ("mono", "stereo", ...).
    dtype:
        Floating point dtype for the returned samples.

    Returns
    -------
    audio:
        A 1-D NumPy array containing the decoded audio samples in the
        ``[-1.0, 1.0]`` range.
    sample_rate:
        The sample rate associated with ``audio`` (``target_sr``).
    """

    options = {"probesize": "64k", "analyzeduration": "200000"}
    chunks: list[np.ndarray] = []
    try:
        with av.open(
            file, mode="r", options=options, metadata_encoding="utf-8"
        ) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise ValueError(f"No audio stream found in {source!r}")

            resampler = av.AudioResampler(format="s16", layout=layout, rate=target_sr)

            for frame in container.decode(stream):
                resampled = resampler.resample(frame)
                if resampled is None:
                    continue

                frames = (
                    resampled if isinstance(resampled, (list, tuple)) else [resampled]
                )
                for f in frames:
                    arr = f.to_ndarray()
                    if arr.ndim == 2:
                        if arr.shape[0] == 1:
                            arr = arr[0]
                        elif arr.shape[1] == 1:
                            arr = arr[:, 0]
                        else:
                            arr = arr.mean(axis=0)
                    elif arr.ndim > 2:
                        arr = arr.reshape(-1)

                    if arr.dtype != np.float32:
                        arr = (arr.astype(np.float32) / 32768.0).clip(-1.0, 1.0)

                    chunks.append(arr)

    except Exception as e:
        print(f"Error processing audio: {e}")
        return {"audio_array": [np.zeros((0,), dtype=dtype)], "sample_rate": target_sr}

    audio = np.concatenate(chunks, axis=0).astype(dtype, copy=False)
    return {"audio_array": audio, "sample_rate": target_sr}


if __name__ == "__main__":
    from daft import col
    from daft.functions import file

    SOURCE_URI = "hf://datasets/Eventual-Inc/sample-files/videos/*.mp4"
    DEST_URI = ".data/read_video"

    df = (
        daft.from_glob_path(SOURCE_URI)
        .with_column("file", file(col("path")))
        .with_column("video_meta", read_video_meta(col("file")))
        .with_column("video_keyframes", read_video_keyframes(col("file")))
        .with_column("image_frames", read_video_files(col("file")))
        .with_column("audio", read_video_audio(col("file")))
    )

    df.show()
