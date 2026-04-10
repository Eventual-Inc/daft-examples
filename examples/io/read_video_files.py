# /// script
# description = "Read video frames from video files"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft>=0.7.6", "av", "numpy"]
# ///

import os
import shutil

import daft
from daft.file import File

if __name__ == "__main__":
    SOURCE_URI = "hf://datasets/Eventual-Inc/sample-files/videos/*.mp4"
    LOCAL_DIR = ".data/videos"

    # Download videos locally (read_video_frames uses PyArrow FS which doesn't support hf://)
    os.makedirs(LOCAL_DIR, exist_ok=True)
    paths = daft.from_glob_path(SOURCE_URI).select("path").collect().to_pydict()["path"]
    for p in paths:
        dest = os.path.join(LOCAL_DIR, os.path.basename(p))
        if not os.path.exists(dest):
            with File(p).to_tempfile() as tmp:
                shutil.copy(tmp.name, dest)

    df = daft.read_video_frames(
        f"{LOCAL_DIR}/*.mp4",
        image_height=480,
        image_width=640,
        sample_interval_seconds=1.0,
    )

    df.show()
