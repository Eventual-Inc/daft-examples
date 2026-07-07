"""SigLIP-2 image/text embeddings for EgoDex frames.

Image embeddings run through a stateful ``@daft.cls`` UDF — the model loads
once per worker in ``__init__`` and stays resident in GPU memory, reused
across every batch — so ``clip_emb`` is always a unit-norm 768-d vector.
Text ranking is a cheap dot product against the matching unit-norm SigLIP
text embedding from :func:`encode_text`.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

import daft
from daft import DataType, Series
from daft.expressions import Expression

# --- device / config ---------------------------------------------------------


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

# Default decode cadence: ~1 fps (semantic content barely changes frame-to-frame).
SAMPLE_INTERVAL_SECONDS = 1.0

MODEL_ID = "google/siglip2-base-patch16-224"
EMB_DIM = 768  # SigLIP2-base shared image/text embedding dim (must match the model)


# --- SigLIP image embedder (stateful UDF) --------------------------------------


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
    def __init__(self) -> None:
        self.model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE).eval()
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)

    @daft.method.batch(return_dtype=DataType.embedding(DataType.float32(), EMB_DIM), batch_size=16)
    def embed_image(self, images: Series):
        # images.to_pylist() yields uint8 H×W×C numpy arrays; the SigLIP processor takes them
        # directly (verified identical to the PIL path), so no per-frame Image.fromarray needed.
        inputs = self.processor(images=images.to_pylist(), return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            model_output = self.model.get_image_features(**inputs)
            embeddings = _normalized_embedding(model_output)
        return list(embeddings.cpu().numpy())


def embed_image_normalized(image: Expression) -> Expression:
    """Embed a decoded frame and L2-normalize it in SigLIP image space."""
    return SiglipEmbedder().embed_image(image)


# --- text side ------------------------------------------------------------------


@lru_cache(maxsize=1)
def _text_model():
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def encode_text(text: str) -> np.ndarray:
    """Return a unit-norm SigLIP-2 embedding for `text` (same space as the image embeddings)."""
    model, processor = _text_model()
    with torch.no_grad():
        inputs = processor(text=[text], return_tensors="pt", padding="max_length").to(DEVICE)
        embedding = _normalized_embedding(model.get_text_features(**inputs))
    return embedding.cpu().numpy().astype(np.float32)[0]


__all__ = [
    "DEVICE",
    "DTYPE",
    "EMB_DIM",
    "GPUS",
    "MODEL_ID",
    "SAMPLE_INTERVAL_SECONDS",
    "SiglipEmbedder",
    "encode_text",
    "embed_image_normalized",
]
