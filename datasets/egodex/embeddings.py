"""SigLIP-2 image/text embeddings for EgoDex frames.

Image embeddings run through a SigLIP UDF so ``clip_emb`` is always a unit-norm
768-d vector. Text ranking is a cheap dot product against the matching unit-norm
SigLIP text embedding from :func:`encode_text`.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

import daft
from daft import DataType, Series
from daft.expressions import Expression

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
IMAGE_BATCH_SIZE = 16

# Default decode cadence: ~1 fps (semantic content barely changes frame-to-frame).
SAMPLE_INTERVAL_SECONDS = 1.0


# --- SigLIP model ------------------------------------------------------------


@lru_cache(maxsize=1)
def _siglip():
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


def _pil_image(image: np.ndarray | None):
    from PIL import Image

    if image is None:
        return None
    return Image.fromarray(np.asarray(image, dtype=np.uint8))


@daft.func.batch(return_dtype=DataType.embedding(DataType.float32(), EMB_DIM))
def _embed_images_batch(images: Series) -> Series:
    import torch

    model, processor, device = _siglip()
    rows = list(images)
    embeddings: list[np.ndarray | None] = [None] * len(rows)
    batch_indices: list[int] = []
    batch_images = []

    for index, image in enumerate(rows):
        pil = _pil_image(image)
        if pil is None:
            continue
        batch_indices.append(index)
        batch_images.append(pil)

    for start in range(0, len(batch_images), IMAGE_BATCH_SIZE):
        chunk_indices = batch_indices[start : start + IMAGE_BATCH_SIZE]
        chunk_images = batch_images[start : start + IMAGE_BATCH_SIZE]
        with torch.no_grad():
            inputs = processor(images=chunk_images, return_tensors="pt").to(device)
            chunk_embeddings = _normalized_model_output(model.get_image_features(**inputs))
        for index, embedding in zip(chunk_indices, chunk_embeddings, strict=True):
            embeddings[index] = embedding

    return Series.from_pylist(embeddings)


def embed_image_normalized(image: Expression) -> Expression:
    """Embed a decoded frame and L2-normalize it in SigLIP image space."""
    return _embed_images_batch(image)


def encode_text(text: str) -> np.ndarray:
    """Return a unit-norm SigLIP-2 embedding for `text` (same space as the image embeddings)."""
    import torch

    model, processor, device = _siglip()
    with torch.no_grad():
        inputs = processor(text=[text], return_tensors="pt", padding="max_length").to(device)
        return _normalized_model_output(model.get_text_features(**inputs))[0]


__all__ = [
    "EMB_DIM",
    "IMAGE_BATCH_SIZE",
    "MODEL_ID",
    "SAMPLE_INTERVAL_SECONDS",
    "encode_text",
    "embed_image_normalized",
]
