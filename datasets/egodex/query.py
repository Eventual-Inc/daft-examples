"""Scenario queries over episode-level EgoDex pose features.

The pose branch stores one row per episode, with each continuous feature as an
episode-length track. Query-time scenarios turn those tracks into per-frame masks
inside a Daft UDF, then stitch matching frames into contiguous segments. Text
queries rank sampled video-frame embeddings, and combined queries keep only
sampled frames whose pose mask also matches.

    from egodex import EgoDexPipeline, calibrate, query

    pipeline = EgoDexPipeline(".data")
    features = pipeline.calculate_features(pipeline.trajectory(pipeline.raw()))
    thresholds = calibrate(features)
    hits = query(features, pose="writing_grip", k=5, thresholds=thresholds)

For semantic text queries, pass the frame-level embedding table from
:func:`embeddings.embed_frames` as ``clip=`` (with ``text=``). Scenario
callables take ``(tracks, thresholds)`` and return an (N,) boolean mask, where
``tracks`` maps unsuffixed feature names (``closure``, ``flex_nonthumb``, ...)
to one hand's arrays.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

import daft
from daft import DataType, col

from .features import FPS

if TYPE_CHECKING:
    from daft.dataframe import DataFrame

# Action thresholds (units per second), surfaced so callers can see/tune them:
# grasping = curl closing fast enough, lifting = wrist rising fast enough.
GRASP_RATE = 0.20
LIFT_VEL = 0.20
TWIST_ROLL_RATE = 2.0

# Segment stitching: contiguous matching frames become one [start, end] segment;
# gaps shorter than SEG_GAP_MERGE are bridged, runs shorter than SEG_MIN_FRAMES
# dropped, and at most MAX_SEGS (longest) are kept per episode.
SEG_GAP_MERGE = 5
SEG_MIN_FRAMES = 1
MAX_SEGS = 12

# Semantic-only fallback: a window of SEMANTIC_WIN seconds either side of an
# episode's best-matching frame.
SEMANTIC_WIN = 1.5

# Data-driven percentiles for the calibrated thresholds.
REACH_RATE_PERCENTILE = 85
WRIST_STILL_PERCENTILE = 30
ARTICULATION_PERCENTILE = 75

# The per-hand feature tracks the match UDF consumes, in signature order.
_TRACKS = (
    "closure",
    "flex_nonthumb",
    "thumb_min_tip",
    "thumb_min_knuckle",
    "curl_rate",
    "wrist_vert_vel",
    "arm_ext_rate",
    "wrist_speed",
    "articulation",
    "roll",
)


# --- scenarios: (tracks, thresholds) -> (N,) bool mask ------------------------


def writing_grip(t, thr):
    """Tripod: thumb on the index/middle tip, those two not fisted, ring+little more curled."""
    flex = t["flex_nonthumb"]
    return (
        (t["thumb_min_tip"] < thr["thumb_on_tip"])
        & (flex[:, 0] < thr["curled_flexion"])
        & (flex[:, 1] < thr["curled_flexion"])
        & (flex[:, 2] > flex[:, 0] + thr["curl_gap"])
        & (flex[:, 3] > flex[:, 1] + thr["curl_gap"])
    )


def hammer_grip(t, thr):
    """Power: all four fingers curled and the thumb wrapped across a proximal knuckle."""
    return (t["flex_nonthumb"] > thr["curled_flexion"]).all(axis=1) & (t["thumb_min_knuckle"] < thr["thumb_on_knuckle"])


def twisting(t, thr):
    """Forearm roll past a fixed rate (action)."""
    return t["roll"] > TWIST_ROLL_RATE


def reaching(t, thr):
    """Arm extending faster than the calibrated rate (action)."""
    return t["arm_ext_rate"] >= thr["reach"]


def in_hand(t, thr):
    """Wrist still while fingers move, both vs calibrated thresholds (action)."""
    return (t["wrist_speed"] < thr["still"]) & (t["articulation"] > thr["articulation"])


def grasping(t, thr):
    """Fingers closing fast enough (action): curl rate <= -GRASP_RATE."""
    return t["curl_rate"] <= -GRASP_RATE


def lifting(t, thr):
    """Wrist rising fast enough (action): vertical velocity >= LIFT_VEL."""
    return t["wrist_vert_vel"] >= LIFT_VEL


def openness(t, thr, open_lo=0.0, open_hi=1.0):
    """Openness band (1 = fully open palm), mapped onto the closure track via the
    calibrated closure spread [closure_lo, closure_hi]: higher openness -> lower closure."""
    span = (thr["closure_hi"] - thr["closure_lo"]) or 1.0
    clo_lo = thr["closure_hi"] - open_hi * span
    clo_hi = thr["closure_hi"] - open_lo * span
    return (t["closure"] >= clo_lo) & (t["closure"] <= clo_hi)


SCENARIOS: dict[str, Callable[..., np.ndarray]] = {
    "writing_grip": writing_grip,
    "hammer_grip": hammer_grip,
    "twisting": twisting,
    "reaching": reaching,
    "in_hand": in_hand,
    "grasping": grasping,
    "lifting": lifting,
    "openness": openness,
}


# --- calibration ---------------------------------------------------------------

_CALIBRATION_TRACKS = (
    "arm_ext_rate",
    "wrist_speed",
    "articulation",
    "flex_nonthumb",
    "thumb_min_tip",
    "thumb_min_knuckle",
    "closure",
)


def calibrate(features: DataFrame) -> dict[str, float]:
    """Compute the global scenario thresholds once over an episode-level feature table.

    Pools each track over every episode and both hands, then takes np.percentile
    (linear interpolation — identical to the old frame-level Daft percentile aggs).
    Compute once when the features are loaded and pass the dict to :func:`query`.
    """
    columns = [f"{name}_{tag}" for name in _CALIBRATION_TRACKS for tag in ("L", "R")]
    data = features.select(*columns).to_pydict()

    def pooled(name: str, percentile: float, explode: bool = False) -> float:
        arrays = [np.asarray(a) for tag in ("L", "R") for a in data[f"{name}_{tag}"]]
        values = np.concatenate([a.ravel() if explode else a for a in arrays])
        return float(np.percentile(values, percentile))

    return {
        "reach": pooled("arm_ext_rate", REACH_RATE_PERCENTILE),
        "still": pooled("wrist_speed", WRIST_STILL_PERCENTILE),
        "articulation": pooled("articulation", ARTICULATION_PERCENTILE),
        "curled_flexion": pooled("flex_nonthumb", 70, explode=True),
        "thumb_on_tip": pooled("thumb_min_tip", 25),
        "thumb_on_knuckle": pooled("thumb_min_knuckle", 15),
        "closure_lo": pooled("closure", 2),
        "closure_hi": pooled("closure", 98),
        "curl_gap": math.radians(20),
    }


# --- segment stitching ----------------------------------------------------------


def segments_of(frames, gap_merge: int = SEG_GAP_MERGE, min_frames: int = SEG_MIN_FRAMES):
    """Contiguous matching runs (merging gaps < gap_merge) as [(start, end), ...]."""
    frames = sorted(frames)
    if not frames:
        return []
    runs, start, prev = [], frames[0], frames[0]
    for frame in frames[1:]:
        if frame - prev > gap_merge:
            runs.append((start, prev))
            start = frame
        prev = frame
    runs.append((start, prev))
    return [(a, b) for a, b in runs if b - a + 1 >= min_frames] or runs


def _top_segments(frames, max_segments: int = MAX_SEGS):
    """The longest <= max_segments contiguous runs in `frames`, ordered by start time."""
    runs = segments_of(frames)
    return sorted(sorted(runs, key=lambda run: run[1] - run[0], reverse=True)[:max_segments])


# --- the per-episode match UDF ---------------------------------------------------


def _episode_mask(pose, hand, thresholds, open_lo, open_hi, tracks_by_tag) -> np.ndarray:
    scenario = pose if callable(pose) else SCENARIOS[pose]
    extra = (open_lo, open_hi) if scenario is openness else ()
    tags = {"left": ("L",), "right": ("R",)}.get(hand, ("L", "R"))
    mask = None
    for tag in tags:
        hand_mask = np.asarray(scenario(tracks_by_tag[tag], thresholds, *extra), dtype=bool)
        mask = hand_mask if mask is None else (mask | hand_mask)
    return mask


_MATCH_DTYPE = DataType.struct(
    {
        "match_count": DataType.int64(),
        "segments": DataType.list(DataType.list(DataType.int64())),
        "match_mask": DataType.tensor(DataType.bool()),
    }
)


def _build_match_udf(pose, hand, thresholds, open_lo, open_hi, max_segments):
    column_order = [f"{name}_{tag}" for tag in ("L", "R") for name in _TRACKS]

    # daft.func maps columns to parameters by signature, so the UDF needs one
    # named parameter per feature track (20 total), in column_order.
    @daft.func(return_dtype=_MATCH_DTYPE, use_process=False, unnest=True)
    def match_episode(
        closure_L,
        flex_nonthumb_L,
        thumb_min_tip_L,
        thumb_min_knuckle_L,
        curl_rate_L,
        wrist_vert_vel_L,
        arm_ext_rate_L,
        wrist_speed_L,
        articulation_L,
        roll_L,
        closure_R,
        flex_nonthumb_R,
        thumb_min_tip_R,
        thumb_min_knuckle_R,
        curl_rate_R,
        wrist_vert_vel_R,
        arm_ext_rate_R,
        wrist_speed_R,
        articulation_R,
        roll_R,
    ) -> dict[str, object]:
        params = locals()
        tracks_by_tag = {tag: {name: np.asarray(params[f"{name}_{tag}"]) for name in _TRACKS} for tag in ("L", "R")}
        mask = _episode_mask(pose, hand, thresholds, open_lo, open_hi, tracks_by_tag)
        matching = np.flatnonzero(mask)
        segments = _top_segments(matching.tolist(), max_segments)
        return {
            "match_count": int(mask.sum()),
            "segments": [[int(a), int(b)] for a, b in segments],
            "match_mask": mask,
        }

    return match_episode, [col(name) for name in column_order]


# --- the one query API ------------------------------------------------------------


def query(
    features: DataFrame,
    pose=None,
    text: str | None = None,
    clip: DataFrame | None = None,
    k: int = 5,
    hand: str = "either",
    open_lo: float = 0.0,
    open_hi: float = 1.0,
    thresholds: dict[str, float] | None = None,
    encode=None,
    fps: float = FPS,
    semantic_window: float = SEMANTIC_WIN,
    max_segments: int = MAX_SEGS,
):
    """Rank episodes by a pose scenario and/or a semantic text query.

    features:  episode-level DataFrame from calculate_features (or its parquet).
    pose:      scenario name (a SCENARIOS key, incl. "openness"), or a callable
               ``(tracks, thresholds) -> (N,) bool mask``, or None.
    text/clip: semantic query string + the frame-level embedding table from
               embed_frames (or its parquet: task, episode_id, frame_index, clip_emb).
    k:         number of top episodes to return.
    hand:      'left' | 'right' | 'either'.
    thresholds: optional calibrate() dict; computed from `features` if omitted.
    encode:    text encoder override (defaults to embeddings.encode_text).

    Returns a list of ``{task, episode_id, score, n_frames, segments}`` dicts,
    ranked best first. Pose-only ranks by match count; text (with or without
    pose) ranks by max similarity. Text-only falls back to a window around each
    episode's best frame. Combined mode evaluates the pose mask at the sampled
    frames' indices, matching the old joined-frames semantics.
    """
    has_text = text is not None
    if pose is None and not has_text:
        raise ValueError("Pass a pose scenario, a text query, or both.")
    if has_text and clip is None:
        raise ValueError("Text queries need clip= (the embed_frames DataFrame or its parquet).")

    sims = None
    if has_text:
        if encode is None:
            from .embeddings import encode_text as encode
        clip_data = clip.select("task", "episode_id", "frame_index", "clip_emb").to_pydict()
        sims = np.asarray(clip_data["clip_emb"], dtype=np.float32) @ encode(text)

    matches = None
    if pose is not None:
        if thresholds is None:
            thresholds = calibrate(features)
        match_udf, track_columns = _build_match_udf(pose, hand, thresholds, open_lo, open_hi, max_segments)
        matched = (
            features.select("task", "episode_id", match_udf(*track_columns)).where(col("match_count") > 0).to_pydict()
        )
        matches = {
            (task, int(episode)): {
                "match_count": int(count),
                "segments": [tuple(s) for s in segments],
                "mask": np.asarray(mask, dtype=bool),
            }
            for task, episode, count, segments, mask in zip(
                matched["task"],
                matched["episode_id"],
                matched["match_count"],
                matched["segments"],
                matched["match_mask"],
            )
        }

    scored: dict[tuple, dict] = {}
    if pose is not None and not has_text:
        for key, m in matches.items():
            scored[key] = {
                "score": float(m["match_count"]),
                "n_frames": m["match_count"],
                "segments": m["segments"],
            }
    else:
        # one similarity per sampled frame, keyed by (task, episode, frame)
        for task, episode, frame, sim in zip(
            clip_data["task"], clip_data["episode_id"], clip_data["frame_index"], sims
        ):
            key = (task, int(episode))
            frame = int(frame)
            if pose is not None:
                m = matches.get(key)
                if m is None or frame >= len(m["mask"]) or not m["mask"][frame]:
                    continue  # only sampled frames where the pose mask fires count
            entry = scored.setdefault(key, {"sims": [], "frames": []})
            entry["sims"].append(float(sim))
            entry["frames"].append(frame)
        window = int(semantic_window * fps)
        for entry in scored.values():
            best = int(np.argmax(entry["sims"]))
            entry["score"] = entry["sims"][best]
            entry["n_frames"] = len(entry["frames"])
            if pose is not None:
                entry["segments"] = _top_segments(entry["frames"], max_segments)
            else:
                best_frame = entry["frames"][best]
                entry["segments"] = [(max(0, best_frame - window), best_frame + window)]
            del entry["sims"], entry["frames"]

    ranked = sorted(scored.items(), key=lambda item: item[1]["score"], reverse=True)[: int(k)]
    return [
        {
            "task": task,
            "episode_id": episode,
            "score": entry["score"],
            "n_frames": entry["n_frames"],
            "segments": list(entry["segments"]),
        }
        for (task, episode), entry in ranked
    ]


__all__ = [
    "GRASP_RATE",
    "LIFT_VEL",
    "MAX_SEGS",
    "SCENARIOS",
    "SEG_GAP_MERGE",
    "TWIST_ROLL_RATE",
    "calibrate",
    "query",
    "segments_of",
]
