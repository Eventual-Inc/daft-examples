# /// script
# description = "Convert raw EgoDex HDF5 episodes into a LeRobot v3 dataset (48-D state + 204-D skeleton)"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft>=0.7.15", "numpy", "h5py", "lerobot"]
# ///
"""EgoDex (raw HDF5) → LeRobot-format frame features, in Daft.

Reads raw EgoDex HDF5 with daft.datasets.hdf5.read and produces the per-frame
LeRobot feature columns (observation.state[48], observation.extrinsics[16],
action[48], task), then writes an on-disk LeRobot v3 dataset (tabular only — the
observation.image video feature is added separately by egodex_video.py).

"""

from __future__ import annotations
import numpy as np
import daft
from daft import col
from daft.datatype import DataType
from daft.datasets import hdf5
from daft.functions import coalesce, when
from daft.udf import func
from daft.window import Window
import pathlib
import shutil

FPS = 30.0

# observation.state is 48 floats: per hand, wrist xyz + rot6d (first two rotation
# columns) + 5 fingertip xyz (thumb..little); left hand then right hand.
WRIST = {"left": "transforms/leftHand", "right": "transforms/rightHand"}
TIPS = {
    "left": [
        "transforms/leftThumbTip",
        "transforms/leftIndexFingerTip",
        "transforms/leftMiddleFingerTip",
        "transforms/leftRingFingerTip",
        "transforms/leftLittleFingerTip",
    ],
    "right": [
        "transforms/rightThumbTip",
        "transforms/rightIndexFingerTip",
        "transforms/rightMiddleFingerTip",
        "transforms/rightRingFingerTip",
        "transforms/rightLittleFingerTip",
    ],
}
CAMERA = "transforms/camera"
# The 12 transforms feeding build_state, in the order it expects them.
STATE_TRANSFORMS = [
    WRIST["left"],
    *TIPS["left"],
    WRIST["right"],
    *TIPS["right"],
]
ATTRS = ["llm_description", "llm_description2", "which_llm_description"]

# observation.skeleton is 204 floats: the xyz translation of every joint, in
# joint order. Per side: hand, forearm, arm, shoulder, then each finger chain
# (thumb has 4 parts; index/middle/ring/little add a metacarpal); then body
# joints (hip, spine1-7, neck1-4). Camera is excluded (it is observation.extrinsics).
FINGERS = ["Thumb", "Index", "Middle", "Ring", "Little"]

def finger_transforms(side, finger):
    infix = "" if finger == "Thumb" else "Finger"
    parts = (["Metacarpal"] if finger != "Thumb" else []) + ["Knuckle", "IntermediateBase", "IntermediateTip", "Tip"]
    return [f"transforms/{side}{finger}{infix}{part}" for part in parts]

def side_transforms(side):
    arm = [f"transforms/{side}{j}" for j in ("Hand", "Forearm", "Arm", "Shoulder")]
    fingers = [t for finger in FINGERS for t in finger_transforms(side, finger)]
    return arm + fingers

BODY_TRANSFORMS = [f"transforms/{j}" for j in ("hip", *(f"spine{i}" for i in range(1, 8)), *(f"neck{i}" for i in range(1, 5)))]
SKELETON_TRANSFORMS = side_transforms("left") + side_transforms("right") + BODY_TRANSFORMS
SKELETON_DIM = len(SKELETON_TRANSFORMS) * 3

# Transforms every convertible episode must contain (used by egodex_preflight).
REQUIRED_TRANSFORMS = SKELETON_TRANSFORMS + [CAMERA]

def hand_block(wrist, tips):
    # wrist xyz (translation) + rot6d (first two rotation columns) + each fingertip xyz
    w = np.asarray(wrist, dtype=np.float32).reshape(4, 4)
    out = list(w[:3, 3]) + list(w[:3, 0]) + list(w[:3, 1])
    for tip in tips:
        out += list(np.asarray(tip, dtype=np.float32).reshape(4, 4)[:3, 3])
    return out

@func(return_dtype=DataType.tensor(DataType.float32(), shape=(16,)))
def build_extrinsics(camera):
    return np.asarray(camera, dtype=np.float32).reshape(16)  # camera 4x4, row-major

@func(return_dtype=DataType.tensor(DataType.float32(), shape=(48,)))
def build_state(left_hand, left_thumb, left_index, left_middle, left_ring, left_little,
                right_hand, right_thumb, right_index, right_middle, right_ring, right_little):
    block = hand_block(left_hand, [left_thumb, left_index, left_middle, left_ring, left_little])
    block += hand_block(right_hand, [right_thumb, right_index, right_middle, right_ring, right_little])
    return np.asarray(block, dtype=np.float32)

@func(return_dtype=DataType.tensor(DataType.float32(), shape=(SKELETON_DIM,)))
def build_skeleton(*joints):
    # Each joint is a 4x4 transform; take its xyz translation, in SKELETON_TRANSFORMS order.
    return np.concatenate([np.asarray(j, dtype=np.float32).reshape(4, 4)[:3, 3] for j in joints])

def egodex_frames(path):
    """Read raw EgoDex HDF5 → per-frame LeRobot feature columns (no video)."""
    df = hdf5.read(path, datasets=SKELETON_TRANSFORMS + [CAMERA], attrs=ATTRS)
    df = df.with_column("observation.state", build_state(*[col(n) for n in STATE_TRANSFORMS]))
    df = df.with_column("observation.skeleton", build_skeleton(*[col(n) for n in SKELETON_TRANSFORMS]))
    df = df.with_column("observation.extrinsics", build_extrinsics(col(CAMERA)))
    # task = llm_description, or llm_description2 for reversible tasks (which_llm_description="2");
    # fill_null(False) makes the absent-which case fall through to llm_description.
    df = df.with_column(
        "task",
        when((col("which_llm_description") == "2").fill_null(False), col("llm_description2"))
        .otherwise(col("llm_description")),
    )
    # action = next frame's state within the episode; the last frame repeats itself (no wrap).
    window = Window().partition_by("path").order_by("row_index")
    df = df.with_column("action", coalesce(col("observation.state").lead(1).over(window), col("observation.state")))
    return df.select("path", "row_index", "observation.state", "observation.skeleton", "observation.extrinsics", "action", "task")

def state_names():
    # The 48 per-dimension names for observation.state.
    names = []
    for side in ("left", "right"):
        names += [f"{side}_wrist_{a}" for a in "xyz"]
        names += [f"{side}_rot_{i}" for i in range(6)]
        for finger in ("thumb", "index", "middle", "ring", "little"):
            names += [f"{side}_{finger}_{a}" for a in "xyz"]
    return names

def skeleton_names():
    # The 204 per-dimension names for observation.skeleton (joint xyz, in transform order).
    return [f"{t.split('/')[-1]}_{a}" for t in SKELETON_TRANSFORMS for a in "xyz"]

def features():
    return {
        "observation.state": {"dtype": "float32", "shape": (48,), "names": state_names()},
        "observation.skeleton": {"dtype": "float32", "shape": (SKELETON_DIM,), "names": skeleton_names()},
        "observation.extrinsics": {"dtype": "float32", "shape": (16,), "names": [f"extrinsic_{i}" for i in range(16)]},
        "action": {"dtype": "float32", "shape": (48,), "names": [f"action_{i}" for i in range(48)]},
    }

def write_lerobot(files, repo_id, output_dir, batch_size=64):
    """Write EgoDex HDF5 episodes to an on-disk LeRobot v3 dataset (tabular only, no video).

    Daft reads + transforms a batch of files in parallel (one file = one episode); each
    episode's frames are then fed in order to the serial LeRobotDataset writer. The batch
    size bounds driver memory, and the action window partitions by path so episodes in a
    batch never bleed together.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # heavy optional dep; only needed to write

    out = pathlib.Path(output_dir)
    if out.exists():
        shutil.rmtree(out)
    ds = LeRobotDataset.create(
        repo_id=repo_id, fps=int(FPS), features=features(), root=str(out), robot_type="hand", use_videos=False
    )
    episodes = 0
    for start in range(0, len(files), batch_size):
        rows = egodex_frames(files[start : start + batch_size]).sort(["path", "row_index"]).to_pydict()
        paths = rows["path"]
        n = len(paths)
        i = 0
        while i < n:  # group consecutive rows sharing a path into one episode
            p = paths[i]
            while i < n and paths[i] == p:
                ds.add_frame({
                    "observation.state": np.asarray(rows["observation.state"][i], dtype=np.float32),
                    "observation.skeleton": np.asarray(rows["observation.skeleton"][i], dtype=np.float32),
                    "observation.extrinsics": np.asarray(rows["observation.extrinsics"][i], dtype=np.float32),
                    "action": np.asarray(rows["action"][i], dtype=np.float32),
                    "task": rows["task"][i],
                })
                i += 1
            ds.save_episode()
            episodes += 1
    ds.finalize()
    return str(out), episodes
