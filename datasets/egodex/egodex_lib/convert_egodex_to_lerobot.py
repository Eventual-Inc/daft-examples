"""EgoDex (raw HDF5) → LeRobot-format frame features, in Daft.

Reads raw EgoDex HDF5 with Daft's first-class `Hdf5File` type (one row per
episode file) and produces the per-frame LeRobot feature columns
(observation.state[48], observation.skeleton[204], observation.extrinsics[16],
action[48], task), then writes an on-disk LeRobot v3 dataset (tabular only — the
observation.image video feature is added separately by egodex_video.py).

The read is per-episode: `hdf5_file` makes an `Hdf5File` column, the
`read_transforms` UDF pulls every transform dataset as a whole (N, 4, 4) array,
and `write_lerobot` slices those arrays into frames as it feeds the LeRobot
writer. No per-frame explode is needed — the writer consumes one episode at a time.
"""

from __future__ import annotations
import numpy as np
import daft
from daft import col
from daft.datatype import DataType
from daft.functions import hdf5_file, hdf5_attrs
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


BODY_TRANSFORMS = [
    f"transforms/{j}" for j in ("hip", *(f"spine{i}" for i in range(1, 8)), *(f"neck{i}" for i in range(1, 5)))
]
SKELETON_TRANSFORMS = side_transforms("left") + side_transforms("right") + BODY_TRANSFORMS
SKELETON_DIM = len(SKELETON_TRANSFORMS) * 3

# Transforms every convertible episode must contain (used by egodex_preflight),
# and exactly the datasets read_transforms pulls from each HDF5 file.
REQUIRED_TRANSFORMS = SKELETON_TRANSFORMS + [CAMERA]


@daft.func(return_dtype=DataType.struct({name: DataType.tensor(DataType.float32()) for name in REQUIRED_TRANSFORMS}))
def read_transforms(trajectory):
    """Read every required transform dataset from one episode's HDF5 file.

    `trajectory` is an `Hdf5File` (one per episode). A single `read` of all the
    dataset paths opens the file once and returns each as its whole (N, 4, 4)
    array of per-frame 4x4 poses.
    """
    arrays = trajectory.read(REQUIRED_TRANSFORMS)
    return {name: arrays[name].astype(np.float32) for name in REQUIRED_TRANSFORMS}


def hand_block(wrist, tips):
    """One hand's 24 state dims over a whole episode.

    wrist is (N, 4, 4); tips is five (N, 4, 4) arrays. Returns (N, 24):
    wrist xyz (translation) + rot6d (first two rotation columns) + each fingertip xyz.
    """
    translation = wrist[:, :3, 3]
    rotation = np.concatenate([wrist[:, :3, 0], wrist[:, :3, 1]], axis=1)
    fingertips = np.concatenate([tip[:, :3, 3] for tip in tips], axis=1)
    return np.concatenate([translation, rotation, fingertips], axis=1)


def build_state(transforms):
    """observation.state (N, 48): left hand block then right hand block."""
    left = hand_block(transforms[WRIST["left"]], [transforms[name] for name in TIPS["left"]])
    right = hand_block(transforms[WRIST["right"]], [transforms[name] for name in TIPS["right"]])
    return np.concatenate([left, right], axis=1).astype(np.float32)


def build_skeleton(transforms):
    """observation.skeleton (N, 204): the xyz translation of every joint, in SKELETON_TRANSFORMS order."""
    return np.concatenate([transforms[name][:, :3, 3] for name in SKELETON_TRANSFORMS], axis=1).astype(np.float32)


def build_extrinsics(transforms):
    """observation.extrinsics (N, 16): the camera 4x4, row-major, per frame."""
    camera = transforms[CAMERA]
    return camera.reshape(camera.shape[0], 16).astype(np.float32)


def next_frame_action(state):
    """action (N, 48): each frame's target is the next frame's state; the last frame repeats itself (no wrap)."""
    return np.vstack([state[1:], state[-1:]]).astype(np.float32)


def resolve_task(attributes):
    """Task text for one episode: llm_description, or llm_description2 for reversible tasks
    (which_llm_description == "2"); falls back to llm_description when absent."""

    def text(name):
        value = attributes.get(name)
        if isinstance(value, bytes):
            return value.decode("utf-8", "replace")
        return None if value is None else str(value)

    chosen = text("llm_description2") if text("which_llm_description") == "2" else text("llm_description")
    return chosen or ""


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

    Daft reads a batch of files in parallel as an episode-level DataFrame (one row =
    one HDF5 file, transforms held as whole arrays); each episode's frames are then
    sliced in order and fed to the serial LeRobotDataset writer. The batch size bounds
    driver memory.
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
        batch = (
            daft.from_pydict({"path": list(files[start : start + batch_size])})
            .with_column("trajectory", hdf5_file(col("path")))
            .with_column("attributes", hdf5_attrs(col("trajectory")))
            .with_column("transforms", read_transforms(col("trajectory")))
            .sort("path")
        )
        for episode in batch.to_pylist():
            transforms = {name: np.asarray(array, dtype=np.float32) for name, array in episode["transforms"].items()}
            state = build_state(transforms)
            skeleton = build_skeleton(transforms)
            extrinsics = build_extrinsics(transforms)
            action = next_frame_action(state)
            task = resolve_task(episode["attributes"])
            for frame in range(len(state)):
                ds.add_frame(
                    {
                        "observation.state": state[frame],
                        "observation.skeleton": skeleton[frame],
                        "observation.extrinsics": extrinsics[frame],
                        "action": action[frame],
                        "task": task,
                    }
                )
            ds.save_episode()
            episodes += 1
    ds.finalize()
    return str(out), episodes
