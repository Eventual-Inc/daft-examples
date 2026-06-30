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

from egodex_lib import convert_egodex_to_lerobot, add_state_features, add_skeleton_features, embed_frames, query


# --- SigLIP embedding ------------------------------------------------------

def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

_HAS_CUDA = torch.cuda.is_available()
GPUS = 1 if _HAS_CUDA else 0
DEVICE = _auto_device()
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

MODEL_ID = "google/siglip2-base-patch16-224"
EMB_DIM = 768

def _normalized_embedding(model_output) -> torch.Tensor:
    """Pull the embedding tensor out of a transformers output and L2-normalize it.

    transformers 5.x returns a model-output object from get_image_features /
    get_text_features; older versions returned a bare tensor. Handle both.
    """
    if torch.is_tensor(model_output):
        feats = model_output
    else:
        feats = model_output.pooler_output
    feats = feats.float()
    return feats / feats.norm(dim=-1, keepdim=True)  # unit-norm so cosine == dot product


@daft.cls(gpus=GPUS, max_concurrency=1, use_process=False)
class SiglipEmbedder:
    """Encode each frame into a unit-norm SigLIP image embedding.

    Single-node, in-process: runs on this machine's device (CUDA on the EC2 GPU
    box, CPU/MPS for a local smoke test). The model loads once per instance.
    """

    def __init__(self) -> None:
        self.model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE).eval()
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)

    @daft.method.batch(return_dtype=DataType.embedding(DataType.float32(), EMB_DIM), batch_size=16)
    def embed_image(self, images: Series):
        
        inputs = self.processor(images=images.to_pylist(), return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            model_output = self.model.get_image_features(**inputs)
            embeddings = _normalized_embedding(model_output)
        return list(embeddings.cpu().numpy())


if __name__ == "__main__":
    # 1. HDF5 -> LeRobot
    lerobot_dir = convert_egodex_to_lerobot("egodex/**/*.hdf5", repo_id="egodex", output_dir="egodex_lerobot/")

    df = daft.datasets.lerobot.read(lerobot_dir)  # 2. LeRobot -> DataFrame (one row per frame)
    emb = embed_frames(
        lerobot.read(lerobot_dir, load_video_frames="observation.image")
    )  # 3. SigLIP-2 image embeddings (~1 fps)
    df = add_state_features(df)  # 4. per-frame geometry  (closure, flexion, thumb distances, ...)
    df = add_skeleton_features(df)  # 5. + action rates over frames (curl_rate, wrist_speed, roll, ...)
    df.write_parquet("features/")  # 6. continuous pose features (30 fps) — compute once

    emb.select("episode_index", "frame_index", "clip_emb").write_parquet("embeddings/")  #    a separate semantic branch

    # ── Query ────────────────────────────────────────────────────────────────────
    features = daft.read_parquet("features/")  # load the features wherever you like
    embeddings = daft.read_parquet("embeddings/")

    pose_hits = query(features, pose="writing_grip", k=5)  # 7a. rank by a hand-pose scenario (match count)
    text_hits = query(embeddings, text="chopsticks", k=5)  # 7b. rank by semantic text (facade encodes it with SigLIP)

    # 7c. pose AND text: join so each frame carries both pose features and an embedding,
    #     then filter by the grip and rank what's left by visual similarity.
    frames = features.join(embeddings, on=["episode_index", "frame_index"])
    combined_hits = query(frames, pose="hammer_grip", text="stapler", k=5)

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
