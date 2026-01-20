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
# description = "Video-oriented patterns using daft.VideoFile and PyAV"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[video]>=0.7.1"]
# ///

import daft
from daft.functions import video_file, video_metadata, video_keyframes

df = (
	daft.from_glob_path("hf://datasets/Eventual-Inc/sample-files/videos/*.mp4")
	.with_column("file", video_file(daft.col("path")))
	.with_column("metadata", video_metadata(daft.col("file")))
	.with_column("keyframes", video_keyframes(daft.col("file")))
)

df.collect()


