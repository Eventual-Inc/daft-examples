"""
Usage patterns for `daft.File` with audio workflows

There isn't a dedicated `AudioFile` class here—audio is handled as `daft.File`.
Typical patterns:
- Create a File column via `daft.functions.file(col("path"))`
- Use `File.open()` to pass a file-like handle into `soundfile`
- Optionally materialize to disk via `File.to_tempfile()` for libraries that require a path

Run:
  uv run usage_patterns/daft_file/daft_audiofile.py
"""

# /// script
# description = "Audio-oriented patterns using daft.File + soundfile (duration/sample_rate)"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[audio]>=0.7.1"]
# ///

import daft
from daft.functions import audio_file, audio_metadata, resample

df = (
	daft.from_glob_path("hf://datasets/Eventual-Inc/sample-files/audio/*.mp3")
	.with_column("file", audio_file(daft.col("path")))
	.with_column("metadata", audio_metadata(daft.col("file")))
	.with_column("resampled", resample(daft.col("file"), sample_rate=16000))
	.select("path", "file", "size", "metadata", "resampled")
)

df.show(3)
