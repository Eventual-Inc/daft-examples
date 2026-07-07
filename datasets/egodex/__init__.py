"""EgoDex on Daft: class-based dataset pipeline and query helpers.

    from egodex import EgoDexPipeline

    pipeline = EgoDexPipeline(".data")
    episodes = pipeline.raw()
    trajectories = pipeline.trajectory(episodes)
    features = pipeline.calculate_features(trajectories)
    frames = pipeline.camera_frames(episodes, sample_interval_seconds=1.0)
    embeddings = pipeline.embed_frames(frames)

Run ``uv run python -m egodex.pipeline`` for the end-to-end script.
"""

from .features import FPS
from .query import SCENARIOS, calibrate, pose_search, pose_search_many, query, segments_of
from .schemas import (
    FEATURE_TRAJECTORY_FIELDS,
    JOINTS,
    TRAJECTORY_FIELDS,
    TRANSFORM_JOINTS,
)
from .viz import overlay

_LAZY = {
    "EgoDexPipeline": "pipeline",
    "encode_text": "embeddings",
    "embed_image_normalized": "embeddings",
}


def __getattr__(name):
    # Defer torch/transformers imports until an embeddings symbol is touched.
    if name in _LAZY:
        import importlib

        module = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EgoDexPipeline",
    "FPS",
    "FEATURE_TRAJECTORY_FIELDS",
    "JOINTS",
    "SCENARIOS",
    "TRAJECTORY_FIELDS",
    "TRANSFORM_JOINTS",
    "calibrate",
    "encode_text",
    "embed_image_normalized",
    "overlay",
    "pose_search",
    "pose_search_many",
    "query",
    "segments_of",
]
