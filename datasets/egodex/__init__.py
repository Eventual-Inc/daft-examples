"""EgoDex on Daft: class-based dataset pipeline and query helpers.

    from egodex import EgoDexPipeline

    pipeline = EgoDexPipeline(".data")
    episodes = pipeline.raw()
    trajectories = pipeline.trajectory(episodes)
    spatial = pipeline.frame_features(trajectories)   # one row per frame
    features = pipeline.temporal_features(spatial)    # + windowed action rates
    # (or both stages at once: pipeline.calculate_features(trajectories))
    frames = pipeline.camera_frames(episodes, sample_interval_seconds=1.0)
    embeddings = pipeline.embed_frames(frames)

Run ``uv run python -m egodex.pipeline`` for the end-to-end script.
"""

from .features import FPS
from .query import SCENARIOS, TRACKS, calibrate, pose_search, pose_search_many, query, segments_of
from .schemas import (
    FEATURE_TRAJECTORY_FIELDS,
    JOINTS,
    TRAJECTORY_FIELDS,
    TRANSFORM_JOINTS,
)
from .temporal import add_temporal_features
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
    "TRACKS",
    "TRAJECTORY_FIELDS",
    "TRANSFORM_JOINTS",
    "add_temporal_features",
    "calibrate",
    "encode_text",
    "embed_image_normalized",
    "overlay",
    "pose_search",
    "pose_search_many",
    "query",
    "segments_of",
]
