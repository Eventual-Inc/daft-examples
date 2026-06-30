# /// script
# description = "Runnable demo of the egodex facade: raw EgoDex HDF5 -> a queryable hand-pose dataset."
# requires-python = ">=3.12, <3.13"
# dependencies = [
#     "daft[transformers,hdf5,video]",
#     "lerobot",
#     "sentencepiece",
# ]
#
# [tool.uv]
# prerelease = "allow"
# extra-index-url = ["https://nightly.daft.ai"]
# ///
"""Runnable demo of the egodex facade: raw EgoDex HDF5 -> a queryable hand-pose dataset.

The whole pipeline is a handful of lines; each is one stage (see egodex.py for the
Daft details under each). Copy-paste and run:

    uv run egodex_demo.py

Two feature branches come off the same LeRobot read:
  - pose geometry at 30 fps  (add_state_features -> add_skeleton_features)
  - SigLIP-2 embeddings at ~1 fps  (embed_frames)
queried together at the end. Notes:
  - Step 1 (convert) needs a Daft build with the new `Hdf5File` API (nightly / >= next
    release) plus `pip install lerobot` for the write. If you already have the LeRobot
    dataset, skip step 1 and point `lerobot.read` at it directly.
"""

import os

import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image
import numpy as np

import daft
from daft import Series, DataType

from egodex_lib.egodex import convert_egodex_to_lerobot, add_state_features, add_skeleton_features, embed_frames, query


def convert_egodex_to_lerobot(hdf5_glob, repo_id, output_dir):
    """Convert raw EgoDex HDF5 episodes into a LeRobot v3 dataset on disk; returns output_dir.

    Thin wrapper over convert_egodex_to_lerobot.write_lerobot, which reads each episode with
    Daft's new Hdf5File type. Requires a Daft build that has the Hdf5File API.
    """
    from egodex_lib.convert_egodex_to_lerobot import write_lerobot

    files = sorted(glob.glob(hdf5_glob))
    if not files:
        raise FileNotFoundError(f"no HDF5 files matched {hdf5_glob!r}")
    return write_lerobot(files, repo_id=repo_id, output_dir=output_dir)


if __name__ == "__main__":
    DATA_URI = os.path.join(os.path.dirname(__file__), ".data/")

    # -- Build the features ────────────────────────────────────────────────────────
    # 1. HDF5 -> LeRobot from egodex_lib.convert_egodex_to_lerobot
    lerobot_dir = convert_egodex_to_lerobot("egodex/**/*.hdf5", repo_id="egodex", output_dir="egodex_lerobot/")

    # 2. LeRobot -> DataFrame (one row per frame)
    df = daft.datasets.lerobot.read(lerobot_dir, load_video_frames="observation.image")

    # 3. SigLIP-2 image embeddings (~1 fps)
    emb = embed_frames(df)
    emb.select("episode_index", "frame_index", "clip_emb").write_parquet("embeddings/")  #    a separate semantic branch

    # 4. per-frame geometry  (closure, flexion, thumb distances, ...)
    df = add_state_features(df)

    # 5. + action rates over frames (curl_rate, wrist_speed, roll, ...)
    df = add_skeleton_features(df)

    # 6. continuous pose features (30 fps) — compute once
    df.write_parquet("features/")

    # ── Query ────────────────────────────────────────────────────────────────────
    # load the features wherever you like
    features = daft.read_parquet("features/")
    embeddings = daft.read_parquet("embeddings/")

    # 7a. rank by a hand-pose scenario (match count)
    pose_hits = query(features, pose="writing_grip", k=5)

    # 7b. rank by semantic text (facade encodes it with SigLIP)
    text_hits = query(embeddings, text="chopsticks", k=5)

    # 7c. pose AND text: join so each frame carries both pose features and an embedding, then filter by the grip and rank what's left by visual similarity.
    frames = features.join(embeddings, on=["episode_index", "frame_index"])
    combined_hits = query(frames, pose="hammer_grip", text="stapler", k=5)

    # -- Display the results ────────────────────────────────────────────────────────

    for title, hits in [
        ("writing_grip", pose_hits),
        ("text 'chopsticks'", text_hits),
        ("hammer_grip + 'stapler'", combined_hits),
    ]:
        print(f"\n## {title}")
        for hit in hits:
            print(
                f"  episode {hit['episode_index']}: score {hit['score']:.3f}, "
                f"{hit['n_frames']} frames, segments {hit['segments'][:3]}"
            )
