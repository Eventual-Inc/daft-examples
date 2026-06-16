"""Materialize SigLIP image embeddings over a list of EgoDex episodes — single-node Daft.

Plain single-node Daft: it runs on whatever machine you launch it on (your
laptop for a CPU smoke test, or an EC2 GPU box). The device
is auto-detected in clip_features. It reads the dataset from HF (or a local copy
via DATASET=), runs CLIP, and writes one parquet of features for cheap querying
later with query_local.py.

  # CPU smoke test against the local converted dataset:
  CLIP_DEVICE=cpu DATASET=/abs/path/to/egodex_lerobot_full OUT=/tmp/clip_smoke \
      python run_clip_features.py

  # real run on an EC2 GPU box, reading the published dataset from HF:
  python run_clip_features.py
"""

import os

import daft
from daft import col
from daft.daft import HuggingFaceConfig
from daft.io import HTTPConfig, IOConfig
from huggingface_hub import get_token

import lerobot  # vendored daft.datasets.lerobot
from clip_features import EPISODES, SUBSAMPLE, SiglipEmbedder

DATASET = os.environ.get("DATASET", "shreyasgarimella/egodex-test-lerobot")
OUT = os.environ.get("OUT", os.path.join(os.path.dirname(__file__), "out", "clip_features"))

# A token (from `hf auth login`) is only needed for the private HF repo; a local
# DATASET path ignores it.
_tok = get_token()
if _tok:
    IO = IOConfig(
        hf=HuggingFaceConfig(token=_tok),
        http=HTTPConfig(num_tries=10, retry_initial_backoff_ms=2000),
    )
else:
    IO = None

df = lerobot.read(DATASET, io_config=IO, load_video_frames="observation.image")
# Cheap filters on episode_index / frame_index push BELOW the video decoder, so
# only ~1 fps of the chosen episodes is decoded and embedded.
df = df.where(col("episode_index").is_in(EPISODES))
df = df.where(col("frame_index") % SUBSAMPLE == 0)
df = df.with_column("clip_emb", SiglipEmbedder().embed_image(col("observation.image")))
# Keep observation.state so curl can be computed locally at query time.
df = df.select("episode_index", "frame_index", "observation.state", "clip_emb")

df.write_parquet(OUT)
print(f"wrote features for episodes {EPISODES} -> {OUT}")
