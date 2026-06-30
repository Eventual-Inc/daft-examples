"""Precompute per-frame hand geometry for a LeRobot dataset -> one parquet, with Daft.

Thin wrapper over the egodex facade: read -> add_state_features -> add_skeleton_features
-> write_parquet. STATE features are per-frame geometry; ACTION features are per-episode
rates (Daft window functions). The output is *continuous* geometry only — scenario booleans
are computed at query time (egodex.calibrate + egodex.query), not stored here. The UI reads
this parquet and filters at query time instead of recomputing geometry at every launch.

  DATASET=/path/to/egodex_lerobot_full python run_pose_features.py
"""
import os
import shutil

from daft.datasets import lerobot
from egodex import add_state_features, add_skeleton_features

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.environ.get("DATASET", "./egodex_lerobot_full")
POSE_OUT = os.environ.get("POSE_OUT", os.path.join(HERE, "out", "pose_features"))

print(f"reading {DATASET} ...", flush=True)
frames = lerobot.read(DATASET)            # one row per frame (no video decode needed for geometry)
frames = add_state_features(frames)       # per-frame geometry (closure, flexion, thumb distances, ...)
frames = add_skeleton_features(frames)    # + action rates over frames (curl_rate, wrist_speed, roll, ...)

if os.path.exists(POSE_OUT):
    shutil.rmtree(POSE_OUT)               # overwrite: write_parquet appends files, so clear stale runs first
frames.write_parquet(POSE_OUT)
print(f"DONE: wrote continuous pose features -> {POSE_OUT}", flush=True)
