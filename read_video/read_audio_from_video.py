# /// script
# description = "Read audio files and save as mp3"
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
from .read_video_files import read_video_audio


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
        
    with file.open() as f:
        audio, native_sample_rate = sf.read(
            file=f,
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
        .with_column("audio", read_video_audio(file(col("path"))))
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
