"""egodex - facade API for the EgoDex scenario-search pipeline (blog post #1).

This is the thin, public-facing layer the notebook imports. Each function wraps
heavier logic that lives in the existing modules (clip_features, pose_features,
skeleton_features, query_ui) so a reader runs the whole pipeline in a handful of
lines:

    import daft, lerobot
    from egodex import convert_egodex_to_lerobot, add_state_features, add_skeleton_features, query

    lerobot_dir = convert_egodex_to_lerobot("egodex/**/*.hdf5", repo_id="egodex", output_dir="egodex_lerobot/")
    df = lerobot.read(lerobot_dir)            # one row per frame
    df = add_state_features(df)               # per-frame geometry (closure, flexion, thumb distances, ...)
    df = add_skeleton_features(df)            # + action rates over frames (curl_rate, wrist_speed, roll, ...)
    df.write_parquet("features/")             # continuous features, one parquet (compute once)

    features = daft.read_parquet("features/")          # load wherever you like
    hits = query(features, pose="writing_grip", k=5)   # rank a warm DataFrame by a hand-pose scenario

Features store only continuous geometry; the scenario booleans are computed at
query time (thresholds calibrated once via `calibrate`). For semantic text queries,
also run `embed_frames(df)` (a separate ~1 fps branch) to add `clip_emb`, then pass
`text=`/`text_embeddings=` to query(). `query` is the single ranking core - both this
script and the live UI (query_ui) call it. See egodex_demo.py for the runnable version.
"""
from __future__ import annotations
import torch
from transformers import AutoModel, AutoProcessor
from clip_features import DEVICE, MODEL_ID, _normalized_embedding
import clip_features
import glob
import math
from collections import defaultdict
from typing import Callable
import os
import subprocess
import tempfile
from PIL import Image, ImageDraw
from daft.datasets import lerobot
import daft
import numpy as np
from daft import DataType, col
from daft.functions import euclidean_distance
from daft.window import Window
import egodex_lerobot 
import pose_features          # 48-D state features (curl, wrist, rot6d) - numpy
import skeleton_features      # 204-D skeleton features (closure, grips, arm extension) - numpy

# Action thresholds, surfaced so the blog (and callers) can see/tune them. These
# mirror query_ui: grasping = curl closing fast enough, lifting = wrist rising fast
# enough (48-D state rates, units per second).
GRASP_RATE = 0.20
LIFT_VEL = 0.20

# Segment stitching (matches query_ui): contiguous matching frames become one
# [start, end] segment; gaps shorter than SEG_GAP_MERGE are bridged, runs shorter
# than SEG_MIN_FRAMES dropped, and at most MAX_SEGS (longest) are kept per episode.
SEG_GAP_MERGE = 5
SEG_MIN_FRAMES = 1
MAX_SEGS = 12

# Defaults for the semantic-only (text, no pose) fallback: a window of
# SEMANTIC_WIN seconds either side of an episode's best-matching frame.
SEMANTIC_WIN = 1.5
DEFAULT_FPS = 30.0

# Action-feature thresholds (mirror run_pose_features). Twisting uses a fixed roll
# rate; reaching / in-hand use data-driven percentiles so they fire at a sensible
# fraction of frames.
TWIST_ROLL_RATE = 2.0
REACH_RATE_PERCENTILE = 85
WRIST_STILL_PERCENTILE = 30
ARTICULATION_PERCENTILE = 75


def _rotation_matrix(rot6d):
    """Single-frame rot6d (6,) -> (3, 3) rotation matrix (columns = hand x, y axes, palm normal)."""
    rot6d = np.asarray(rot6d, dtype=np.float64)
    first = rot6d[0:3] / (np.linalg.norm(rot6d[0:3]) + 1e-9)
    second = rot6d[3:6] - np.dot(first, rot6d[3:6]) * first
    second = second / (np.linalg.norm(second) + 1e-9)
    return np.stack([first, second, np.cross(first, second)], axis=1)

# daft udf
@daft.func(return_dtype=DataType.float64())
def forearm_roll(rot6d, rot6d_next, forearm_axis):
    """Wrist roll (rad) about the forearm axis from one frame to the next (0 at an episode's last frame)."""
    if rot6d is None or rot6d_next is None:
        return 0.0
    delta = _rotation_matrix(rot6d_next) @ _rotation_matrix(rot6d).T
    angle = np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1))
    axis = np.array([delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0], delta[1, 0] - delta[0, 1]])
    magnitude = np.linalg.norm(axis)
    if magnitude < 1e-9:
        return 0.0
    return float(abs(angle * np.dot(axis / magnitude, np.asarray(forearm_axis))))



# build a Daft boolean from the continuous columns + thresholds ──
# Each takes (hand 'L'/'R', thr=calibrate() dict). Booleans live here, at query time —
# features store only continuous geometry. The two grips need per-finger logic, so they
# drop into a tiny row UDF; everything else is a native comparison.
# daft udf

@daft.func(return_dtype=DataType.bool())
def _is_writing_grip(flex_nonthumb, thumb_min_tip, curled_flexion, curl_gap, thumb_on_tip):
    """Tripod: thumb on the index/middle tip, those two not fisted, ring+little more curled."""
    flex = np.asarray(flex_nonthumb)
    return bool(thumb_min_tip < thumb_on_tip
                and flex[0] < curled_flexion and flex[1] < curled_flexion
                and flex[2] > flex[0] + curl_gap and flex[3] > flex[1] + curl_gap)


# daft udf
@daft.func(return_dtype=DataType.bool())
def _is_hammer_grip(flex_nonthumb, thumb_min_knuckle, curled_flexion, thumb_on_knuckle):
    """Power: all four fingers curled and the thumb wrapped across a proximal knuckle."""
    flex = np.asarray(flex_nonthumb)
    return bool(bool((flex > curled_flexion).all()) and thumb_min_knuckle < thumb_on_knuckle)

# daft expression
def writing_grip(hand, thr):
    """Tripod/precision grip on `hand` from flexion + thumb-tip distance vs calibrated thresholds."""
    return _is_writing_grip(col(f"flex_nonthumb_{hand}"), col(f"thumb_min_tip_{hand}"),
                            thr["curled_flexion"], thr["curl_gap"], thr["thumb_on_tip"])

# daft expression
def hammer_grip(hand, thr):
    """Power grip on `hand` from flexion + thumb-knuckle distance vs calibrated thresholds."""
    return _is_hammer_grip(col(f"flex_nonthumb_{hand}"), col(f"thumb_min_knuckle_{hand}"),
                           thr["curled_flexion"], thr["thumb_on_knuckle"])

# daft expression
def twisting(hand, thr):
    """Forearm roll past a fixed rate (action) on `hand`."""
    return col(f"roll_{hand}") > TWIST_ROLL_RATE

# daft expression
def reaching(hand, thr):
    """Arm extending faster than the calibrated rate (action) on `hand`."""
    return col(f"arm_ext_rate_{hand}") >= thr["reach"]

# daft expression
def in_hand(hand, thr):
    """Wrist still while fingers move, both vs calibrated thresholds (action) on `hand`."""
    return (col(f"wrist_speed_{hand}") < thr["still"]) & (col(f"articulation_{hand}") > thr["articulation"])

# daft expression
def grasping(hand, thr):
    """Fingers closing fast enough (action): curl rate <= -GRASP_RATE on `hand`."""
    return col(f"curl_rate_{hand}") <= -GRASP_RATE

# daft expression
def lifting(hand, thr):
    """Wrist rising fast enough (action): vertical velocity >= LIFT_VEL on `hand`."""
    return col(f"wrist_vert_vel_{hand}") >= LIFT_VEL

# daft expression
def openness(hand, thr, open_lo=0.0, open_hi=1.0):
    """Openness band on `hand` (1 = fully open palm), mapped onto the closure column via the
    calibrated closure spread [closure_lo, closure_hi]: higher openness -> lower closure."""
    span = (thr["closure_hi"] - thr["closure_lo"]) or 1.0
    clo_lo = thr["closure_hi"] - open_hi * span
    clo_hi = thr["closure_hi"] - open_lo * span
    return (col(f"closure_{hand}") >= clo_lo) & (col(f"closure_{hand}") <= clo_hi)


# The named scenarios query() accepts, keyed by name. Each is a scenario(hand, thr)
# that returns a Daft boolean expression. `openness` also takes open_lo/open_hi, so
# it's handled separately in pose_predicate; the rest are uniform.
SCENARIOS: dict[str, Callable[..., object]] = {
    "writing_grip": writing_grip,
    "hammer_grip": hammer_grip,
    "twisting": twisting,
    "reaching": reaching,
    "in_hand": in_hand,
    "grasping": grasping,
    "lifting": lifting,
    "openness": openness,
}


def _pooled_percentile(frames, name, percentile, explode=False):
    """A feature's percentile pooled over both hands (exploding a list column first if asked).
    Matches np.percentile (linear interpolation). One small 1-row aggregate per call."""
    pooled = frames.select(col(f"{name}_L").alias("v")).concat(frames.select(col(f"{name}_R").alias("v")))
    if explode:
        pooled = pooled.explode("v")
    return float(pooled.agg(col("v").percentile(percentile / 100).alias("p")).to_pydict()["p"][0])


def calibrate(frames):
    """Compute the global scenario thresholds once over `frames`, with Daft percentile aggs.

    These are the data-driven cut points the scenario predicates compare against. Computed once
    when the features are loaded and reused across queries (like the openness bounds already were),
    so per-query cost is just a boolean scan. Returns a dict the predicates read.
    """
    return {
        "reach": _pooled_percentile(frames, "arm_ext_rate", REACH_RATE_PERCENTILE),
        "still": _pooled_percentile(frames, "wrist_speed", WRIST_STILL_PERCENTILE),
        "articulation": _pooled_percentile(frames, "articulation", ARTICULATION_PERCENTILE),
        "curled_flexion": _pooled_percentile(frames, "flex_nonthumb", 70, explode=True),
        "thumb_on_tip": _pooled_percentile(frames, "thumb_min_tip", 25),
        "thumb_on_knuckle": _pooled_percentile(frames, "thumb_min_knuckle", 15),
        "closure_lo": _pooled_percentile(frames, "closure", 2),
        "closure_hi": _pooled_percentile(frames, "closure", 98),
        "curl_gap": math.radians(20),
    }


def pose_predicate(pose, hand="either", thresholds=None, open_lo=0.0, open_hi=1.0):
    """Build the Daft boolean for a pose scenario across the requested hand(s).

    `pose` may be:
      - None or "any"                       -> no pose filter (returns None)
      - a scenario name (a SCENARIOS key,   -> looked up in SCENARIOS, incl. "openness"
        e.g. "writing_grip", "openness")       (which also uses open_lo/open_hi)
      - a callable(hand, thr, ...)->Expr    -> used directly as the scenario
      - a Daft Expression                   -> used as-is (already hand-resolved by the caller)
    `thresholds` is the calibrate() dict; needed for every scenario except the fixed-rate ones
    (grasping/lifting/twisting), which ignore it. `hand` is 'left', 'right', or 'either'.
    Each scenario is called once per requested hand ('L'/'R') and OR'd together for 'either'.
    """
    if pose in (None, "any"):
        return None
    if isinstance(pose, daft.Expression):
        return pose
    scenario = pose if callable(pose) else SCENARIOS[pose]
    extra = (open_lo, open_hi) if scenario is openness else ()
    left, right = scenario("L", thresholds, *extra), scenario("R", thresholds, *extra)
    return left if hand == "left" else right if hand == "right" else (left | right)


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


def _top_segments(frames, max_segments):
    """The longest <= max_segments contiguous runs in `frames`, ordered by start time."""
    runs = segments_of(frames)
    return sorted(sorted(runs, key=lambda run: run[1] - run[0], reverse=True)[:max_segments])


def _best_frame_per_episode(matches):
    """episode -> the frame_index with the highest `sim` (for the semantic-only window)."""
    best_frame, best_sim = {}, {}
    for episode, frame, sim in zip(matches["episode_index"], matches["frame_index"], matches["sim"]):
        episode = int(episode)
        if episode not in best_sim or sim > best_sim[episode]:
            best_sim[episode], best_frame[episode] = sim, int(frame)
    return best_frame


# The SigLIP-2 text tower, loaded once at import from the same MODEL_ID as the image
# embedder. Kept in the facade so callers never touch the model - query(text=...) just works.
_TEXT_MODEL = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
_TEXT_PROC = AutoProcessor.from_pretrained(MODEL_ID)


def _encode_text(text):
    """Return a unit-norm SigLIP-2 embedding for `text` (same space as the image embeddings)."""
    with torch.no_grad():
        inputs = _TEXT_PROC(text=[text], return_tensors="pt", padding="max_length").to(DEVICE)
        return _normalized_embedding(_TEXT_MODEL.get_text_features(**inputs))[0].cpu().numpy().astype(np.float32)


def query(frames, pose=None, text=None, k=5, hand="either",
          open_lo=0.0, open_hi=1.0, thresholds=None, encode=_encode_text, text_sims=None,
          fps=DEFAULT_FPS, semantic_window=SEMANTIC_WIN, max_segments=MAX_SEGS):
    """Rank episodes by a pose scenario and/or a semantic text query - the one query API.

    The single ranking core used by both the notebook and the live UI (query_ui
    calls it). Pure: no globals, no parquet I/O, no media, no UI shape. Loading is
    the caller's job - the notebook does `daft.read_parquet("features/")`, the UI
    passes its warm in-memory frame - so you can source the frames anywhere.

    frames:          a loaded Daft DataFrame, one row per frame, with the pose-feature
                     columns (+ emb_row for text). Ranking covers exactly this frame
                     set's episodes - the embedded subset - so no manual restriction needed.
    pose:            scenario name (a SCENARIOS key, incl. "openness" and "any"),
                     callable(hand, thr)->Expression, a Daft Expression, or None.
    text:            semantic query string. The facade encodes it with SigLIP, reading the
                     `clip_emb` column from `frames` - the caller passes only `text`. Override
                     the encoder with `encode`, or pass precomputed `text_sims` (frames row order).
    k:               number of top episodes to return.
    hand:            'left' | 'right' | 'either'.
    open_lo/open_hi: openness band in [0, 1] for the "openness" scenario.
    thresholds:      optional calibrate() dict; computed from `frames` if omitted. Pass a cached
                     one (calibrated once at load) to avoid recomputing per query.
    fps, semantic_window, max_segments: segment shaping (see module constants).

    Returns a list of {episode_index, score, n_frames, segments} dicts, ranked best
    first. Pose-only ranks by match count; text (with or without pose) ranks by max
    similarity. Text-only falls back to a window around each episode's best frame.
    """
    if pose is not None and thresholds is None:
        thresholds = calibrate(frames)
    predicate = pose_predicate(pose, hand, thresholds, open_lo, open_hi)

    has_text = text is not None or text_sims is not None
    if predicate is None and not has_text:
        raise ValueError("Pass a pose scenario, a text query, or both.")

    scored = frames
    if has_text:
        # one similarity per frame, keyed by (episode, frame). The facade owns the SigLIP text
        # tower and reads the clip_emb column from `frames`, so the caller passes only `text`.
        columns = ["episode_index", "frame_index"] + ([] if text_sims is not None else ["clip_emb"])
        keyed = frames.select(*columns).to_pydict()
        if text_sims is not None:
            sims = np.asarray(text_sims, dtype=np.float32)
        else:
            sims = np.asarray(keyed["clip_emb"], dtype=np.float32) @ encode(text)
        sim_table = daft.from_pydict({"episode_index": keyed["episode_index"],
                                      "frame_index": keyed["frame_index"],
                                      "sim": sims.astype(np.float32)})
        scored = scored.join(sim_table, on=["episode_index", "frame_index"])
    if predicate is not None:
        scored = scored.where(predicate)

    # rank: pose-only by match count, text by max similarity
    score_column = (col("sim").max() if has_text else col("frame_index").count()).alias("score")
    ranked = (scored.groupby("episode_index")
              .agg(col("frame_index").count().alias("n_frames"), score_column)
              .sort("score", desc=True).limit(int(k)).to_pydict())
    top_episodes = [int(e) for e in ranked["episode_index"]]

    # segments for the ranked episodes (computed on the small, top-k-bounded result)
    matches = scored.where(col("episode_index").is_in(top_episodes))
    if predicate is not None:
        grouped = matches.select("episode_index", "frame_index").to_pydict()
        frames_by_episode = defaultdict(list)
        for episode, frame in zip(grouped["episode_index"], grouped["frame_index"]):
            frames_by_episode[int(episode)].append(int(frame))
        segments = {e: _top_segments(frames_by_episode[e], max_segments) for e in top_episodes}
    else:
        # semantic-only: a window around each episode's best-matching frame
        best_frame = _best_frame_per_episode(matches.select("episode_index", "frame_index", "sim").to_pydict())
        window = int(semantic_window * fps)
        segments = {e: [(max(0, best_frame[e] - window), best_frame[e] + window)] for e in top_episodes}

    return [{"episode_index": episode, "score": float(score), "n_frames": int(n_frames),
             "segments": segments[episode]}
            for episode, score, n_frames in zip(top_episodes, ranked["score"], ranked["n_frames"])]


# ── pipeline stages (the rest of the 6-line notebook interface) ──────────────
HANDS = (("L", "left"), ("R", "right"))
WRIST_DIM = 3                  # wrist xyz
LOCAL_JOINTS_DIM = 72          # hand joints in the hand frame, flattened (24 joints x 3)

def convert_egodex_to_lerobot(hdf5_glob, repo_id, output_dir):
    """Convert raw EgoDex HDF5 episodes into a LeRobot v3 dataset on disk; returns output_dir.

    Thin wrapper over egodex_lerobot.write_lerobot, which reads each episode with
    Daft's new Hdf5File type. Requires a Daft build that has the Hdf5File API.
    """
    
    files = sorted(glob.glob(hdf5_glob))
    if not files:
        raise FileNotFoundError(f"no HDF5 files matched {hdf5_glob!r}")
    return egodex_lerobot.write_lerobot(files, repo_id=repo_id, output_dir=output_dir)


def embed_frames(df, subsample=None, image_column="observation.image"):
    """Add a unit-norm SigLIP-2 image embedding column `clip_emb`, subsampled to ~1 fps.

    Wraps clip_features.SiglipEmbedder (a batched @daft.cls UDF; GPU when present, else
    CPU/MPS). `subsample` keeps 1 frame in N (default clip_features.SUBSAMPLE); pass 1 to
    embed every frame. Filtering before the UDF pushes below the video decoder, so only the
    kept frames are decoded and embedded.
    """
    keep = clip_features.SUBSAMPLE if subsample is None else subsample
    sampled = df.where(col("frame_index") % keep == 0)
    return sampled.with_column("clip_emb", clip_features.SiglipEmbedder().embed_image(col(image_column)))


def _geometry_struct():
    """Struct the per-frame geometry UDF emits, both hands: continuous scalars + the vectors
    later stages use (rates differentiate the vectors; query-time grips read flex_nonthumb +
    the thumb mins). No thresholds, no booleans — those are scenarios, computed at query time."""
    scalar, vector = DataType.float64(), DataType.list(DataType.float64())
    fields = {}
    for tag, _ in HANDS:
        for name in ("closure", "curl", "wrist_height", "arm_extension", "thumb_min_tip", "thumb_min_knuckle"):
            fields[f"{name}_{tag}"] = scalar
        for name in ("wrist", "local_joints", "wrist_rot6d", "forearm_axis", "flex_nonthumb"):
            fields[f"{name}_{tag}"] = vector
    return DataType.struct(fields)


@daft.func(return_dtype=_geometry_struct())
def frame_geometry(state, skeleton):
    """One frame's hand geometry. Daft auto-converts the tensor cells to numpy; we reuse the
    vectorized geometry libs at N=1 via [None] and return a struct — no whole-frame to_pydict."""
    state = np.asarray(state, dtype=np.float64)
    raw = pose_features.compute_raw_features(state[None])              # per-frame: curl, wrist
    geo = skeleton_features.compute_state_features(np.asarray(skeleton, dtype=np.float64)[None])
    out = {}
    for tag, side in HANDS:
        thumb_tip, thumb_knuckle = geo[f"thumb_tip_dist_{tag}"][0], geo[f"thumb_knuckle_dist_{tag}"][0]
        out[f"closure_{tag}"] = float(geo[f"closure_{tag}"][0])
        out[f"curl_{tag}"] = float(raw[f"curl_{tag}"][0])
        out[f"wrist_height_{tag}"] = float(raw[f"wrist_{tag}"][0][1])
        out[f"arm_extension_{tag}"] = float(geo[f"arm_extension_{tag}"][0])
        out[f"thumb_min_tip_{tag}"] = float(min(thumb_tip[0], thumb_tip[1]))          # writing grip
        out[f"thumb_min_knuckle_{tag}"] = float(min(thumb_knuckle[0], thumb_knuckle[1]))  # hammer grip
        out[f"wrist_{tag}"] = raw[f"wrist_{tag}"][0].tolist()
        out[f"local_joints_{tag}"] = geo[f"local_joints_{tag}"][0].reshape(-1).tolist()
        out[f"wrist_rot6d_{tag}"] = state[pose_features.rot6d_slice(side)].tolist()
        out[f"forearm_axis_{tag}"] = geo[f"forearm_axis_{tag}"][0].tolist()
        out[f"flex_nonthumb_{tag}"] = geo[f"flex_nonthumb_{tag}"][0].tolist()
    return out


def add_state_features(df):
    """Per-frame continuous hand geometry from observation.state (48) + observation.skeleton (204).

    A single row-wise UDF -> struct -> unnest (no whole-frame to_pydict). Emits only continuous
    quantities — openness (closure), the values add_skeleton_features differentiates into motion,
    and the grip inputs (flex_nonthumb + thumb mins). Scenario booleans are computed at query time.
    """
    df = df.with_column("_geometry", frame_geometry(col("observation.state"), col("observation.skeleton")))
    return df.select("episode_index", "frame_index", col("_geometry").unnest())


def add_skeleton_features(df, fps=DEFAULT_FPS):
    """Per-episode continuous action rates via Daft window functions over the per-frame geometry.

    Differentiates the continuous quantities over time: curl / wrist-height / arm-extension by
    scalar diffs, wrist & finger-joint motion by native euclidean_distance, wrist rotation by
    forearm_roll (then smoothed). All native/in-DAG — no collect, no thresholds, no booleans.
    Keeps the per-frame columns the query-time scenarios read (closure, flex_nonthumb, thumb mins).
    Expects the output of add_state_features.
    """
    dt = 1.0 / fps
    per_episode = Window().partition_by("episode_index").order_by("frame_index")
    smooth = Window().partition_by("episode_index").order_by("frame_index").rows_between(-2, 2)
    for tag, _ in HANDS:
        # euclidean_distance needs fixed-size-list inputs; materialize the casts once per hand
        df = df.with_column(f"_wrist_v_{tag}", col(f"wrist_{tag}").cast(DataType.fixed_size_list(DataType.float64(), WRIST_DIM)))
        df = df.with_column(f"_joints_v_{tag}", col(f"local_joints_{tag}").cast(DataType.fixed_size_list(DataType.float64(), LOCAL_JOINTS_DIM)))
        df = df.with_column(f"curl_rate_{tag}",
            ((col(f"curl_{tag}").lead(1).over(per_episode) - col(f"curl_{tag}")) / dt).fill_null(0.0))
        df = df.with_column(f"wrist_vert_vel_{tag}",
            ((col(f"wrist_height_{tag}").lead(1).over(per_episode) - col(f"wrist_height_{tag}")) / dt).fill_null(0.0))
        df = df.with_column(f"arm_ext_rate_{tag}",
            ((col(f"arm_extension_{tag}").lead(1).over(per_episode) - col(f"arm_extension_{tag}")) / dt).fill_null(0.0))
        df = df.with_column(f"wrist_speed_{tag}",
            (euclidean_distance(col(f"_wrist_v_{tag}"), col(f"_wrist_v_{tag}").lead(1).over(per_episode)) / dt).fill_null(0.0))
        df = df.with_column(f"articulation_{tag}",
            (euclidean_distance(col(f"_joints_v_{tag}"), col(f"_joints_v_{tag}").lead(1).over(per_episode)) / dt).fill_null(0.0))
        df = df.with_column(f"roll_raw_{tag}",
            forearm_roll(col(f"wrist_rot6d_{tag}"), col(f"wrist_rot6d_{tag}").lead(1).over(per_episode),
                         col(f"forearm_axis_{tag}")) / dt)
        df = df.with_column(f"roll_{tag}", col(f"roll_raw_{tag}").mean().over(smooth))

    keep = ["episode_index", "frame_index"]
    for tag, _ in HANDS:
        keep += [f"closure_{tag}", f"flex_nonthumb_{tag}", f"thumb_min_tip_{tag}", f"thumb_min_knuckle_{tag}",
                 f"curl_rate_{tag}", f"wrist_vert_vel_{tag}", f"arm_ext_rate_{tag}",
                 f"wrist_speed_{tag}", f"articulation_{tag}", f"roll_{tag}"]
    return df.select(*keep)


# EgoDex camera intrinsics (apple/ml-egodex, 1920x1080) for projecting joints to pixels.
CAMERA_FX = CAMERA_FY = 736.6339
CAMERA_CX, CAMERA_CY = 960.0, 540.0


def overlay(dataset, episode_index, frame_index, io_config=None):
    """Return a PIL image of one EgoDex frame with its 68-joint skeleton drawn on it.

    Notebook/visual helper: pulls the frame's skeleton + camera extrinsics, extracts that exact
    video frame (frame-accurate, snapped to a + frame/fps), projects the world joints to pixels
    with the EgoDex intrinsics, and draws them (left = cyan, right = amber). Lazy-imports the
    video/render deps so importing egodex stays light.
    """
    

    key = "observation.image"
    meta = (lerobot.read_episodes(dataset, include_video_metadata=True)
            .where(col("episode_index") == episode_index)
            .select(f"videos/{key}/chunk_index", f"videos/{key}/file_index", f"videos/{key}/from_timestamp")
            .to_pydict())
    chunk = int(meta[f"videos/{key}/chunk_index"][0])
    file_index = int(meta[f"videos/{key}/file_index"][0])
    start = float(meta[f"videos/{key}/from_timestamp"][0])
    shard = os.path.join(dataset, "videos", key, f"chunk-{chunk:03d}", f"file-{file_index:03d}.mp4")

    frame = (lerobot.read(dataset, io_config=io_config)
             .where((col("episode_index") == episode_index) & (col("frame_index") == frame_index))
             .select("observation.skeleton", "observation.extrinsics").to_pydict())
    joints = np.asarray(frame["observation.skeleton"][0], dtype=np.float64).reshape(68, 3)
    world_to_camera = np.linalg.inv(np.asarray(frame["observation.extrinsics"][0], dtype=np.float64).reshape(4, 4))

    with tempfile.TemporaryDirectory() as tmp:
        png = os.path.join(tmp, "frame.png")
        subprocess.run(["ffmpeg", "-y", "-ss", f"{start + frame_index / DEFAULT_FPS:.5f}", "-i", shard,
                        "-frames:v", "1", png, "-loglevel", "error"], check=True)
        image = Image.open(png).convert("RGB")

    camera = (world_to_camera @ np.hstack([joints, np.ones((68, 1))]).T).T[:, :3]
    depth = camera[:, 2]
    pixels_u = CAMERA_FX * camera[:, 0] / depth + CAMERA_CX
    pixels_v = CAMERA_FY * camera[:, 1] / depth + CAMERA_CY
    draw = ImageDraw.Draw(image)
    for joint in range(68):
        if depth[joint] <= 0:
            continue
        color = (34, 211, 238) if joint < 34 else (245, 158, 11)   # left hand cyan, right hand amber
        u, v = pixels_u[joint], pixels_v[joint]
        draw.ellipse([u - 5, v - 5, u + 5, v + 5], fill=color)
    return image
