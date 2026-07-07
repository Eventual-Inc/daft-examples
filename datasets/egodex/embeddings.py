"""SigLIP-2 image/text embeddings for EgoDex frames.

Image embeddings use Daft's native ``embed_image`` expression. The query path
stores unit-norm ``clip_emb`` vectors, so text ranking is a cheap dot product
against the matching unit-norm SigLIP text embedding from :func:`encode_text`.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

import daft
from daft import DataType
from daft.expressions import Expression
from daft.functions import embed_image

# --- device / config ---------------------------------------------------------


def _auto_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


MODEL_ID = "google/siglip2-base-patch16-224"
EMB_DIM = 768
PROVIDER = "transformers"
IMAGE_BATCH_SIZE = 16

# Default decode cadence: ~1 fps (semantic content barely changes frame-to-frame).
SAMPLE_INTERVAL_SECONDS = 1.0


# --- SigLIP embedding ----------------------------------------------------------


@daft.func(return_dtype=DataType.embedding(DataType.float32(), EMB_DIM))
def normalize_embedding(embedding: np.ndarray | None) -> np.ndarray | None:
    """Return a unit-norm embedding, preserving nulls and zero vectors."""
    if embedding is None:
        return None
    values = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(values)
    if norm == 0 or not np.isfinite(norm):
        return values
    return values / norm


def embed_image_normalized(image: Expression) -> Expression:
    """Embed an image with Daft's transformer provider and L2-normalize it."""
    return normalize_embedding(
        embed_image(
            image,
            provider=PROVIDER,
            model=MODEL_ID,
            batch_size=IMAGE_BATCH_SIZE,
        )
    )


# --- text tower (loaded lazily for semantic queries) -------------------------


@lru_cache(maxsize=1)
def _text_tower():
    from transformers import AutoModel, AutoProcessor

    device = _auto_device()
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor, device


def _normalized_model_output(model_output) -> np.ndarray:
    import torch

    features = model_output if torch.is_tensor(model_output) else model_output.pooler_output
    features = features.float()
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype(np.float32)


def encode_text(text: str) -> np.ndarray:
    """Return a unit-norm SigLIP-2 embedding for `text` (same space as the image embeddings)."""
    import torch

    model, processor, device = _text_tower()
    with torch.no_grad():
        inputs = processor(text=[text], return_tensors="pt", padding="max_length").to(device)
        return _normalized_model_output(model.get_text_features(**inputs))[0]


# --- the pipeline stage --------------------------------------------------------


__all__ = [
    "EMB_DIM",
    "IMAGE_BATCH_SIZE",
    "MODEL_ID",
    "PROVIDER",
    "encode_text",
    "embed_image_normalized",
    "normalize_embedding",
]
