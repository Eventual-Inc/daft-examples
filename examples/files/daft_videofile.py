# /// script
# description = "Video-oriented patterns using daft.VideoFile and PyAV"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[video]>=0.7.8", "pillow"]
# ///

import daft
from daft.functions import unnest, video_file, video_keyframes, video_metadata

if __name__ == "__main__":
    df = (
        daft.from_glob_path("hf://datasets/Eventual-Inc/sample-files/videos/*.mp4")
        .with_column("file", video_file(daft.col("path")))
        .with_column("metadata", video_metadata(daft.col("file")))
        .with_column("keyframes", video_keyframes(daft.col("file")))
        .select("path", "file", "size", unnest(daft.col("metadata")), "keyframes")
    )

    df.show(3)
