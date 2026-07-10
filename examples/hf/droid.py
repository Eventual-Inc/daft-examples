# /// script
# description = "Build a lazy DROID episode and Hugging Face scene-classification join"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[huggingface]==0.7.19"]
# ///

import os

import daft
from daft.datasets import droid

# DROID scene classifications are a Parquet mirror on the Hugging Face Hub.
kitchen_scenes = droid.scenes().where(daft.col("scene_classification") == "Home kitchen")

# raw() is lazy: it attaches HDF5 and VideoFile references without decoding them.
episodes = droid.raw().where(daft.col("success")).limit(10)
selected = episodes.join(kitchen_scenes, on="scene_id", how="inner").select(
    "uuid",
    "current_task",
    "wrist_cam_video",
    "scene_classification",
)

# The full public DROID catalog is large. Show its lazy schema by default, and
# materialize a small episode sample only when explicitly requested.
print(selected.schema())
if os.environ.get("RUN_DROID") == "1":
    selected.limit(3).show()
