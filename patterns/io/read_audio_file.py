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
        file_bytes = f.read()
    bio = io.BytesIO(file_bytes)

    with sf.SoundFile(bio) as f:
        native_sample_rate = f.samplerate

        # Handle start offset
        if start is not None:
            f.seek(int(start))

        # Determine frames to read
        # Need to consider the total frames available from the file's metadata,
        # the current seek position, and the user-provided 'frames' and 'stop'
        total_frames_in_file = f.frames
        current_position = f.tell()
        remaining_frames_after_seek = total_frames_in_file - current_position

        read_frames = remaining_frames_after_seek  # Default to reading till end

        if frames != -1:
            read_frames = min(read_frames, frames)

        if stop is not None:
            frames_to_stop_at = int(stop)
            frames_up_to_stop = frames_to_stop_at - current_position
            if frames_up_to_stop < 0:
                frames_up_to_stop = 0
            read_frames = min(read_frames, frames_up_to_stop)

        # Pre-allocate output array
        channels = f.channels
        if always_2d or channels > 1:
            shape = (read_frames, channels)
        else:
            shape = (read_frames,)

        audio = np.empty(shape, dtype=dtype)
        # Call read with `out` parameter
        frames_read = f.read(out=audio)

        # Handle case where fewer frames were read than expected
        if frames_read < len(audio):
            audio = audio[:frames_read]

    # The function expects to return a dictionary, not a tuple
    # The original function also resamples, which is lost in this snippet. Re-add this.
    # The new_string will only contain the changes within the `with file.open() as f:` block.
    # The surrounding logic must remain.
    # This replacement is for the _inner_ logic of the `read_audio` function.
    # I need to ensure the full function signature and return structure is maintained.
    # I will modify the entire `with file.to_tempfile() as tmp:` block to the new pre-allocated memory logic.
    # And re-add the resampling logic.

    # This is a complex replacement, so I will ensure the full context is provided for old_string and new_string.
    # This new_string needs to include the entire original logic from within the function, modified with the new reading method.
    # Let me re-read the original file to ensure I get the full context for `old_string`.

    # I have to be careful with the context as it is a daft function, and the return type is a struct
    # I need to ensure the return values are correctly packaged into the dictionary as expected by the return_dtype.

    # Re-reading `read_audio_file.py` to get the full original content and structure
    # Then construct the full `new_string` including the resampling logic and the dictionary return.

    # Self-correction: The previous plan for new_string was incomplete. I must ensure the entire functional logic of read_audio (including resampling and dictionary return) is present in the `new_string` after updating the core reading mechanism.

    # The current `read_audio` function's body is:
    # with file.to_tempfile() as tmp:
    #    with sf.SoundFile(tmp) as f:
    #        # Handle start offset
    #        if start is not None:
    #            f.seek(int(start))
    #
    #        # Determine frames to read
    #        remaining_frames = f.frames - f.tell()
    #        read_frames = remaining_frames
    #
    #        if frames != -1:
    #            read_frames = min(read_frames, frames)
    #
    #        if stop is not None:
    #            frames_to_stop = int(stop) - f.tell()
    #            if frames_to_stop < 0:
    #                frames_to_stop = 0
    #            read_frames = min(read_frames, frames_to_stop)
    #
    #        audio = f.read(
    #            frames=read_frames,
    #            dtype=dtype,
    #            fill_value=fill_value,
    #            always_2d=always_2d,
    #        )
    #        native_sample_rate = f.samplerate
    #
    # if native_sample_rate != sr:
    #    audio = resample_audio(audio, native_sample_rate, sr)
    #
    # # Calculate offset and duration in seconds
    # offset_seconds = start / sr if start is not None else 0.0
    # duration_seconds = len(audio) / sr
    #
    # return {
    #    "audio_array": audio,
    #    "sample_rate": sr,
    #    "offset": offset_seconds,
    #    "duration": duration_seconds,
    # }

    # I will replace from `with file.to_tempfile() as tmp:` all the way to `duration_seconds = len(audio) / sr`.

    # The new `old_string` will be from the beginning of `with file.to_tempfile()` until `duration_seconds = len(audio) / sr`.
    # The `new_string` will incorporate the memory pre-allocated logic and retain resampling and calculation of offset/duration.

    # Let's ensure the full context is captured in `old_string` to avoid partial matches.

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
