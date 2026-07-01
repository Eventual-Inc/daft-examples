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

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

import daft
from daft import DataType, Series

# --- device / config ------------------------------------------------------


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

SUBSAMPLE = 30  # keep 1 of every 30 frames (~1 fps); semantic content barely changes frame-to-frame


# --- SigLIP embedding ------------------------------------------------------
def _normalized_embedding(model_output) -> torch.Tensor:
    """Pull the embedding tensor out of a transformers output and L2-normalize it.

    transformers 5.x returns a model-output object from get_image_features /
    get_text_features; older versions returned a bare tensor. Handle both.
    """
    feats = model_output if torch.is_tensor(model_output) else model_output.pooler_output
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

    @daft.method(return_dtype=DataType.embedding(DataType.float32(), EMB_DIM))
    def embed_image_rowwise(self, image: np.ndarray) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            model_output = self.model.get_image_features(**inputs)
            embeddings = _normalized_embedding(model_output)
        return embeddings.cpu().numpy()

    @daft.method.batch(return_dtype=DataType.embedding(DataType.float32(), EMB_DIM), batch_size=16)
    def embed_image(self, images: Series):
        # Upstream filters/subsampling can hand this batched UDF an empty morsel; the HF
        # image processor does images[0].device and errors on []. Return nothing for it.
        if len(images) == 0:
            return []
        # images is an already-materialized batch of <=batch_size rows (not the whole
        # column), so to_pylist() here just hands the HF processor the arrays it needs.
        inputs = self.processor(images=images.to_pylist(), return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            model_output = self.model.get_image_features(**inputs)
            embeddings = _normalized_embedding(model_output)
        return list(embeddings.cpu().numpy())
