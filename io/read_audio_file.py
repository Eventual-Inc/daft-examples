# /// script
# description = "Read audio files and save as mp3"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft", "soundfile", "numpy", "scipy"]
# ///

import pathlib
from typing import Any

import daft
from daft import DataType

import soundfile as sf
import numpy as np

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    from scipy import signal
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
    file: daft.File,
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


@daft.func()
def write_audio_to_mp3(
    audio: np.ndarray,
    destination: str,
    sample_rate: int = 16000,
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

    sf.write(destination, data, samplerate=sample_rate, format="MP3", subtype=subtype)
    return destination


@daft.func()
def sanitize_filename(filename: str) -> str:
    import re
    # Replace problematic characters including Unicode variants
    return re.sub(r"[/\\|｜:<>\"?*\s⧸]+", "_", filename)


# Copy + Paste this script then `uv run myscript.py` 
if __name__ == "__main__":
    from daft import col, lit
    from daft.functions import file, format

    SOURCE_URI = "hf://datasets/Eventual-Inc/sample-files/audio/*.mp3"
    SOURCE_URI = "/Users/everettkleven/Desktop/*.mp3"
    DEST_URI = "/Users/everettkleven/Desktop/.data/audio/"
    TARGET_SR = 16000

    df = (
        daft.from_glob_path(SOURCE_URI)
        .with_column("audio_file", file(col("path")))
        .with_column("audio", read_audio(col("audio_file")))
        .with_column(
            "filename_sanitized",
            sanitize_filename(
                col("path").split("/").list.get(-1).split(".").list.get(0)
            ),
        )
        .with_column(
            "resampled_path",
            write_audio_to_mp3(
                audio=col("audio").struct.get("audio_array"),
                destination=format(
                    "{}{}.mp3", lit(DEST_URI), col("filename_sanitized")
                ),
                sample_rate=TARGET_SR,
            ),
        )
        
    )

    df.show()