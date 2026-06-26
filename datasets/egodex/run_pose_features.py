# /// script
# description = "Precompute per-frame hand states + Daft-window action rates for a LeRobot dataset"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft>=0.7.15", "numpy"]
# ///
"""Precompute per-frame hand geometry for a LeRobot dataset -> one parquet, with Daft.

STATE features are per-frame (NumPy); ACTION features are per-episode rates
computed with Daft window functions. The UI reads this parquet and just filters
columns at query time instead of recomputing geometry at every launch.

  DATASET=./egodex_lerobot_full python run_pose_features.py
"""
import os
import shutil

import daft
import numpy as np
from daft import DataType, col
from daft.window import Window

import lerobot                 # vendored daft.datasets.lerobot reader
import pose_features
import skeleton_features

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.environ.get("DATASET", "./egodex_lerobot_full")
POSE_OUT = os.environ.get("POSE_OUT", os.path.join(HERE, "out", "pose_features"))

FRAMES_PER_SECOND = 30.0
SECONDS_PER_FRAME = 1.0 / FRAMES_PER_SECOND

# Action thresholds. Twisting uses a fixed roll rate; reaching / in-hand use
# data-driven percentiles so they fire at a sensible fraction of frames.
TWIST_ROLL_RATE = 2.0           # rad/s about the forearm axis
REACH_RATE_PERCENTILE = 85      # arm extending in the fastest 15% of frames
WRIST_STILL_PERCENTILE = 30     # wrist among the stillest 30% of frames
ARTICULATION_PERCENTILE = 75    # fingers moving in the top 25% of frames

HANDS = [{"tag": "L", "side": "left"}, {"tag": "R", "side": "right"}]


def _rotation_matrix(rot6d):
    """Single-frame rot6d (6,) -> (3, 3) rotation matrix (columns = hand x, y axes, palm normal)."""
    rot6d = np.asarray(rot6d, dtype=np.float64)
    first_column = rot6d[0:3] / (np.linalg.norm(rot6d[0:3]) + 1e-9)
    second_column = rot6d[3:6] - np.dot(first_column, rot6d[3:6]) * first_column
    second_column = second_column / (np.linalg.norm(second_column) + 1e-9)
    return np.stack([first_column, second_column, np.cross(first_column, second_column)], axis=1)


@daft.func(return_dtype=DataType.float64())
def step_distance(current, following):
    """Euclidean distance from a vector to its next-frame value (0 at an episode's last frame)."""
    if current is None or following is None:
        return 0.0
    return float(np.linalg.norm(np.asarray(following) - np.asarray(current)))


@daft.func(return_dtype=DataType.float64())
def forearm_roll(rot6d, rot6d_next, forearm_axis):
    """Wrist roll (rad) about the forearm axis from one frame to the next (0 at an episode's last frame)."""
    if rot6d is None or rot6d_next is None:
        return 0.0
    delta = _rotation_matrix(rot6d_next) @ _rotation_matrix(rot6d).T
    angle = np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1))
    rotation_axis = np.array([delta[2, 1] - delta[1, 2],
                              delta[0, 2] - delta[2, 0],
                              delta[1, 0] - delta[0, 1]])
    magnitude = np.linalg.norm(rotation_axis)
    if magnitude < 1e-9:
        return 0.0
    return float(abs(angle * np.dot(rotation_axis / magnitude, np.asarray(forearm_axis))))


# ── read the dataset (one row per frame) ──
print(f"reading {DATASET} ...", flush=True)
frames = lerobot.read(DATASET).select(
    "episode_index", "frame_index", "observation.state", "observation.skeleton",
).to_pydict()
episode_index = np.asarray(frames["episode_index"])
frame_index = np.asarray(frames["frame_index"])
state = np.asarray(frames["observation.state"], dtype=np.float64)        # 48-D
skeleton = np.asarray(frames["observation.skeleton"], dtype=np.float64)  # 204-D
frame_count = len(episode_index)
print(f"  {frame_count} frames, {len(np.unique(episode_index))} episodes", flush=True)

# ── STATE features: per frame, in NumPy ──
raw = pose_features.compute_raw_features(state)                # curl, wrist, ...
geometry = skeleton_features.compute_state_features(skeleton)  # closure, arm extension, ...
grip_thresholds = skeleton_features.calibrate_grip_thresholds(geometry)

# ── one Daft DataFrame: the scalar + vector columns the window step differentiates ──
columns = {"episode_index": episode_index, "frame_index": frame_index}
for hand in HANDS:
    tag, side = hand["tag"], hand["side"]
    columns[f"closure_{tag}"] = geometry[f"closure_{tag}"]                       # openness (output)
    columns[f"curl_{tag}"] = raw[f"curl_{tag}"]                                  # -> curl_rate (grasping)
    columns[f"wrist_height_{tag}"] = raw[f"wrist_{tag}"][:, 1]                   # -> wrist_vert_vel (lifting)
    columns[f"arm_extension_{tag}"] = geometry[f"arm_extension_{tag}"]           # -> arm_ext_rate (reaching)
    columns[f"wrist_{tag}"] = raw[f"wrist_{tag}"].tolist()                       # -> wrist_speed (in-hand)
    columns[f"local_joints_{tag}"] = geometry[f"local_joints_{tag}"].reshape(frame_count, -1).tolist()  # -> articulation
    columns[f"wrist_rot6d_{tag}"] = state[:, pose_features.rot6d_slice(side)].tolist()                   # -> roll (twisting)
    columns[f"forearm_axis_{tag}"] = geometry[f"forearm_axis_{tag}"].tolist()
    columns[f"sc_writing_{tag}"] = skeleton_features.is_writing_grip(geometry, grip_thresholds, tag)     # static grip
    columns[f"sc_hammer_{tag}"] = skeleton_features.is_hammer_grip(geometry, grip_thresholds, tag)       # static grip

frame_features = daft.from_pydict(columns)

# ── ACTION features: per-episode rates via Daft window functions ──
per_episode = Window().partition_by("episode_index").order_by("frame_index")
smooth_window = Window().partition_by("episode_index").order_by("frame_index").rows_between(-2, 2)
for hand in HANDS:
    tag = hand["tag"]
    # scalar rates: (next frame - this frame) / dt; last frame of each episode has no
    # next frame so lead() returns null — fill with 0.0 (no change at a boundary).
    frame_features = frame_features.with_column(
        f"curl_rate_{tag}",
        ((col(f"curl_{tag}").lead(1).over(per_episode) - col(f"curl_{tag}")) / SECONDS_PER_FRAME).fill_null(0.0))
    frame_features = frame_features.with_column(
        f"wrist_vert_vel_{tag}",
        ((col(f"wrist_height_{tag}").lead(1).over(per_episode) - col(f"wrist_height_{tag}")) / SECONDS_PER_FRAME).fill_null(0.0))
    frame_features = frame_features.with_column(
        f"arm_ext_rate_{tag}",
        ((col(f"arm_extension_{tag}").lead(1).over(per_episode) - col(f"arm_extension_{tag}")) / SECONDS_PER_FRAME).fill_null(0.0))
    # vector / matrix rates: lead() brings the next frame into the row, a UDF does the math
    frame_features = frame_features.with_column(
        f"wrist_speed_{tag}",
        step_distance(col(f"wrist_{tag}"), col(f"wrist_{tag}").lead(1).over(per_episode)) / SECONDS_PER_FRAME)
    frame_features = frame_features.with_column(
        f"articulation_{tag}",
        step_distance(col(f"local_joints_{tag}"), col(f"local_joints_{tag}").lead(1).over(per_episode)) / SECONDS_PER_FRAME)
    frame_features = frame_features.with_column(
        f"roll_raw_{tag}",
        forearm_roll(col(f"wrist_rot6d_{tag}"), col(f"wrist_rot6d_{tag}").lead(1).over(per_episode), col(f"forearm_axis_{tag}")) / SECONDS_PER_FRAME)
    frame_features = frame_features.with_column(
        f"roll_{tag}", col(f"roll_raw_{tag}").mean().over(smooth_window))   # smooth brief spikes

frame_features = frame_features.collect()   # materialize the rates once

# ── data-driven action thresholds (pooled over both hands) ──
rates = frame_features.to_pydict()
reach_threshold = float(np.percentile(np.concatenate([rates["arm_ext_rate_L"], rates["arm_ext_rate_R"]]), REACH_RATE_PERCENTILE))
still_threshold = float(np.percentile(np.concatenate([rates["wrist_speed_L"], rates["wrist_speed_R"]]), WRIST_STILL_PERCENTILE))
articulation_threshold = float(np.percentile(np.concatenate([rates["articulation_L"], rates["articulation_R"]]), ARTICULATION_PERCENTILE))

# ── scenario booleans via Daft expressions ──
output_columns = ["episode_index", "frame_index"]
for hand in HANDS:
    tag = hand["tag"]
    frame_features = frame_features.with_column(f"sc_twisting_{tag}", col(f"roll_{tag}") > TWIST_ROLL_RATE)
    frame_features = frame_features.with_column(f"sc_reaching_{tag}", col(f"arm_ext_rate_{tag}") >= reach_threshold)
    frame_features = frame_features.with_column(
        f"sc_inhand_{tag}",
        (col(f"wrist_speed_{tag}") < still_threshold) & (col(f"articulation_{tag}") > articulation_threshold))
    output_columns += [
        f"closure_{tag}", f"curl_rate_{tag}", f"wrist_vert_vel_{tag}",
        f"sc_twisting_{tag}", f"sc_reaching_{tag}", f"sc_inhand_{tag}",
        f"sc_writing_{tag}", f"sc_hammer_{tag}",
    ]

if os.path.exists(POSE_OUT):
    shutil.rmtree(POSE_OUT)   # overwrite: write_parquet appends files, so clear stale runs first
frame_features.select(*output_columns).write_parquet(POSE_OUT)
print(f"DONE: wrote per-frame geometry -> {POSE_OUT}", flush=True)
