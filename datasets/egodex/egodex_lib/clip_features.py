# /// script
# description = "SigLIP image-embedding UDF for the EgoDex LeRobot pipeline"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[transformers, hdf5, lerobot, video]>=0.7.16", "torch", "transformers>=5.0", "sentencepiece", "pillow", "numpy", "av"]
# ///
"""SigLIP image-embedding UDF for the EgoDex LeRobot pipeline.

Runs as plain single-node Daft on a laptop CPU or a GPU box. Encodes each frame
ONCE into a unit-norm SigLIP image embedding (``clip_emb``) and stores it, so any
scenario query later is just a cheap cosine similarity against a text embedding.
"""

from __future__ import annotations

import os

import daft
import numpy as np
import torch
from daft import DataType, Series
from PIL import Image
from transformers import AutoModel, AutoProcessor

# --- device / config ------------------------------------------------------


def _auto_device() -> str:
    if _HAS_CUDA:
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_HAS_CUDA = torch.cuda.is_available()
GPUS = 1 if _HAS_CUDA else 0

DEVICE = os.environ.get("CLIP_DEVICE", _auto_device())
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

EPISODES: list[int] = [1, 2, 3]
SUBSAMPLE = 30
MODEL_ID = "google/siglip2-base-patch16-224"
EMB_DIM = 768


if __name__ == "__main__":
    DEVICE = os.environ.get("CLIP_DEVICE", _auto_device())
    DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

    # Episodes to embed, identified by episode_index (the dataset's native per-frame
    # key). Fill in the indices for your scenarios — e.g. one per task type.
    EPISODES: list[int] = [1, 2, 3]  # example; list the episode_index values you want to embed
    MODEL_ID = "google/siglip2-base-patch16-224"
    SUBSAMPLE = 30  # keep 1 of every 30 frames (~1 fps); semantic content barely changes between adjacent frames
    EMB_DIM = 768  # SigLIP2-base shared image/text embedding dim (must match the model)

    GPUS = 0

    def _auto_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    df = daft.read_video_frames(SOURCE_URI, image_height=H, image_width=W).limit(ROW_LIMIT)

    embedder = SiglipEmbedder()
    embedder.embed_image(Image.open("test.jpg")).show()
