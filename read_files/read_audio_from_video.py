# /// script
# description = "Read audio files"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft", "soundfile", "av", "numpy", "matplotlib", "scipy"]
# ///

from pathlib import Path
import daft
from daft import DataType
import soundfile as sf
from typing import Any
import re
import numpy as np
from scipy import signal
import pathlib
import io
import av
from av import AudioResampler


@daft.func(
    return_dtype=DataType.struct(
        {
            "audio_array": DataType.tensor(DataType.float32()),
            "sample_rate": DataType.int32(),
        }
    )
)
def read_audio_from_video(
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


@daft.func()
def write_audio_to_mp3(
    audio: np.ndarray,
    sample_rate: int,
    destination: str,
    *,
    subtype: str = "MPEG_LAYER_III",
) -> str:
    """Persist audio samples to an MP3 file via ``soundfile``.

    ``soundfile`` expects floating point audio in ``[-1.0, 1.0]`` with shape
    ``(n_samples,)`` for mono or ``(n_samples, n_channels)`` for multi-channel
    audio. The helper reshapes typical channel-first buffers.
    """

    arr = np.asarray(audio)
    if arr.ndim == 1:
        data = arr
    elif arr.ndim == 2:
        if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1]:
            data = arr.T
        else:
            data = arr
    else:
        raise ValueError("Audio array must be 1-D or 2-D")

    data = data.astype(np.float32, copy=False)
    data = np.clip(data, -1.0, 1.0)

    # Ensure Directory Exists
    pathlib.Path(destination).parent.mkdir(parents=True, exist_ok=True)

    sf.write(destination, data, sample_rate, format="MP3", subtype=subtype)
    return destination


@daft.func()
def sanitize_filename(filename: str) -> str:
    # Replace problematic characters including Unicode variants
    return re.sub(r"[/\\|｜:<>\"?*\s⧸]+", "_", filename)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    resampled = signal.resample_poly(audio, up, down, padtype="edge")
    return resampled


@daft.func(
    return_dtype=DataType.struct(
        {
            "audio_array": DataType.tensor(DataType.float32()),
            "sample_rate": DataType.float32(),
            "offset": DataType.float32(),
            "duration": DataType.float32(),
        }
    )
)
def read_audio(
    file: daft.File | str,
    frames: int = -1,
    start: float | None = None,
    stop: float | None = None,
    dtype: str = "float32",
    sr: int = 16000,
    fill_value: Any | None = None,
    always_2d: bool = True,
):
    """
    Read audio file into a numpy array
    """
    if isinstance(file, str):
        file = daft.File(file)

    audio, native_sample_rate = sf.read(
        file=file,
        frames=frames,
        start=start,
        stop=stop,
        dtype=dtype,
        fill_value=fill_value,
        always_2d=always_2d,
    )
    if native_sample_rate != sr:
        audio = resample_audio(audio, native_sample_rate, sr)

    # Calculate offset and duration in seconds
    offset_seconds = start / sr if start is not None else 0.0
    duration_seconds = len(audio) / sr

    return {
        "audio_array": audio,
        "sample_rate": sr,
        "offset": offset_seconds,
        "duration": duration_seconds,
    }


if __name__ == "__main__":
    from daft import col, lit
    from daft.functions import file, split, format

    SOURCE_URI = "hf://datasets/Eventual-Inc/sample-files/videos/*.mp4"
    DEST_URI = "../.data/audio/"

    df = (
        daft.from_glob_path(SOURCE_URI)
        .with_column("audio", read_audio_from_video(file(col("path"))))
        .with_column(
            "source_filename",
            sanitize_filename(
                col("path").split("/").list.get(-1).split(".").list.get(0)
            ),
        )
        .with_column(
            "mp3_file_path",
            write_audio_to_mp3(
                audio=col("audio").struct.get("audio_array"),
                sample_rate=16_000,
                destination=format(
                    "{}{}{}", lit(DEST_URI), col("source_filename"), lit(".mp3")
                ),
            ),
        )
        .with_column("audio_array", read_audio(file(col("mp3_file_path"))))
    )

    df.show()
