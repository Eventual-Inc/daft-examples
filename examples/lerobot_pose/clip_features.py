"""SigLIP + pose feature UDFs for scenario querying over the EgoDex LeRobot dataset.

Runs as plain single-node Daft on a laptop CPU or a GPU box. Two
features per frame:
  - ``hand_curl`` : cheap geometric feature from the 48-D pose (no model).
  - ``clip_emb``  : a unit-norm SigLIP image embedding (GPU/CPU model).

We embed each frame ONCE and store the vector; any scenario query later is a
cheap cosine similarity against a text embedding (see ``clip_text`` and
``query_local.py``).
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
# Resolve the device ONCE. CUDA on the EC2 GPU box; CPU/MPS for a local smoke
# test. gpus=0 when there's no CUDA so @daft.cls doesn't demand a GPU on a CPU
# machine. Force with CLIP_DEVICE=cpu (e.g. to test the CPU path on a Mac).
_HAS_CUDA = torch.cuda.is_available()
GPUS = 1 if _HAS_CUDA else 0


def _auto_device() -> str:
    if _HAS_CUDA:
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = os.environ.get("CLIP_DEVICE", _auto_device())
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Episodes to embed, identified by episode_index (the dataset's native per-frame
# key). Fill in the indices for your scenarios — e.g. one per task type.
EPISODES: list[int] = [0, 1, 2, 3, 4]
SUBSAMPLE = 30  # keep 1 of every 30 frames (~1 fps); semantic content barely changes between adjacent frames
MODEL_ID = "google/siglip-base-patch16-224"
EMB_DIM = 768  # SigLIP-base shared image/text embedding dim (must match the model)


# --- pose feature ----------------------------------------------------------
def _curl_one_hand(state: np.ndarray, base: int) -> float:
    """Mean distance from the 5 fingertips to the wrist for one hand (small = curled).

    Within a hand's 24-dim block: wrist xyz at [base:base+3], rot6d at
    [base+3:base+9], the 5 fingertip xyz at [base+9:base+24].
    """
    wrist = state[base : base + 3]
    fingertips = state[base + 9 : base + 24].reshape(5, 3)
    distances = np.linalg.norm(fingertips - wrist, axis=1)
    return float(distances.mean())


@daft.func(return_dtype=DataType.float32())
def hand_curl(state) -> float:
    """Curl of the more-curled hand. Left hand is dims [0:24], right hand [24:48]."""
    s = np.asarray(state, dtype=np.float32)
    left_curl = _curl_one_hand(s, 0)
    right_curl = _curl_one_hand(s, 24)
    return min(left_curl, right_curl)


# --- SigLIP embedding ------------------------------------------------------
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
        pil_frames = []
        for array in images.to_pylist():
            pil_frames.append(Image.fromarray(np.asarray(array, dtype=np.uint8)))

        inputs = self.processor(images=pil_frames, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            model_output = self.model.get_image_features(**inputs)
            embeddings = _normalized_embedding(model_output)
        return list(embeddings.cpu().numpy())


def clip_text(query: str) -> list[float]:
    """Encode a text scenario (e.g. "a cup") into a unit-norm SigLIP embedding.

    Used at QUERY time (CPU is fine) to score the stored image embeddings.
    """
    model = AutoModel.from_pretrained(MODEL_ID).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    inputs = processor(text=[query], return_tensors="pt", padding="max_length")  # SigLIP needs fixed-length padding
    with torch.no_grad():
        model_output = model.get_text_features(**inputs)
        embedding = _normalized_embedding(model_output)
    return embedding[0].cpu().numpy().tolist()
