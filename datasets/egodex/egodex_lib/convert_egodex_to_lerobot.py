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
import numpy as np

import h5py
import daft
from daft import col, Window
from daft.datatype import DataType
from daft.functions import row_number

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

def hand_block(wrist: np.ndarray, tips: list[np.ndarray]) -> np.ndarray:
    """One hand's 24 state dims over a whole episode.

    wrist is (N, 4, 4); tips is five (N, 4, 4) arrays. Returns (N, 24):
    wrist xyz (translation) + rot6d (first two rotation columns) + each fingertip xyz.
    """
    translation = wrist[:, :3, 3]
    rotation = np.concatenate([wrist[:, :3, 0], wrist[:, :3, 1]], axis=1)
    fingertips = np.concatenate([tip[:, :3, 3] for tip in tips], axis=1)
    return np.concatenate([translation, rotation, fingertips], axis=1)


def build_state(transforms: h5py.File) -> np.ndarray:
    """observation.state (N, 48): left hand block then right hand block."""
    left = hand_block(transforms[WRIST["left"]], [transforms[name] for name in TIPS["left"]])
    right = hand_block(transforms[WRIST["right"]], [transforms[name] for name in TIPS["right"]])
    return np.concatenate([left, right], axis=1).astype(np.float32)


def build_skeleton(transforms: h5py.File) -> np.ndarray:
    """observation.skeleton (N, 204): the xyz translation of every joint, in SKELETON_TRANSFORMS order."""
    return np.concatenate([transforms[name][:, :3, 3] for name in SKELETON_TRANSFORMS], axis=1).astype(np.float32)


def build_extrinsics(transforms: h5py.File) -> np.ndarray:
    """observation.extrinsics (N, 16): the camera 4x4, row-major, per frame."""
    camera = transforms[CAMERA][:]  # read the h5py Dataset into a numpy array before reshaping
    return camera.reshape(camera.shape[0], 16).astype(np.float32)


def next_frame_action(state):
    """action (N, 48): each frame's target is the next frame's state; the last frame repeats itself (no wrap)."""
    return np.vstack([state[1:], state[-1:]]).astype(np.float32)

# One frame's row. observation.* names are applied here so downstream stages
# (add_state_features etc.) need no renaming after the explode.
FRAME_DTYPE = DataType.struct(
    {
        "frame_index": DataType.int64(),
        "observation.state": DataType.tensor(DataType.float32()),       # (48,)
        "observation.skeleton": DataType.tensor(DataType.float32()),    # (204,)
        "observation.extrinsics": DataType.tensor(DataType.float32()),  # (16,)
        "action": DataType.tensor(DataType.float32()),                  # (48,)
    }
)

# One episode's row: the per-episode task string + its N per-frame structs.
# task rides alongside `frames` so explode() broadcasts it onto every frame row.
EPISODE_DTYPE = DataType.struct(
    {
        "task": DataType.string(),
        "frames": DataType.list(FRAME_DTYPE),
    }
)


@daft.func(return_dtype=EPISODE_DTYPE)
def process_egodex_episode(file_: daft.File) -> dict:
    """One EgoDex HDF5 episode -> {task, frames}; frames is a list of N per-frame structs.

    Opens the file through Daft's native Hdf5File type, then hands the byte stream to
    h5py so the build_* helpers can slice datasets by name. The caller explodes `frames`
    into one row per frame.
    """
    h = file_.as_hdf5()

    # Native batched read -> {name: ndarray}. ~100x faster than handing the byte stream
    # to h5py (which seeks per-dataset through the file abstraction). The dict supports the
    # same transforms[name][:, :3, 3] access the build_* helpers use, so they're unchanged.
    transforms = h.read(list(dict.fromkeys(STATE_TRANSFORMS + SKELETON_TRANSFORMS + [CAMERA])))

    state = build_state(transforms)
    skeleton = build_skeleton(transforms)
    extrinsics = build_extrinsics(transforms)
    action = next_frame_action(state)
    task = resolve_task(h.attrs())  # native attrs() returns a dict

    frames = [
        {
            "frame_index": i,
            "observation.state": state[i],
            "observation.skeleton": skeleton[i],
            "observation.extrinsics": extrinsics[i],
            "action": action[i],
        }
        for i in range(len(state))
    ]
    return {"task": task, "frames": frames}



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

def read_egodex(hdf5_glob, with_video: bool = False):
    """Read raw EgoDex HDF5 directly into the per-frame DataFrame the rest of the pipeline
    expects — no LeRobot, no Hugging Face. One row per frame.

    Point Daft at a directory/glob of HDF5 files (one row = one episode), assign a stable
    episode_index over the file-sorted order, decode each episode's frames with the native
    Hdf5File UDF, then explode into per-frame rows. The output columns match what
    daft.datasets.lerobot.read() produces, so add_state_features/add_skeleton_features/
    embed_frames/query run against it unchanged.

    Columns: episode_index, frame_index, task, observation.state/skeleton/extrinsics,
    action, timestamp, index (+ observation.image when with_video=True).
    """
    per_file = Window().order_by(col("file").file_path())
    episodes = (
        daft.from_files(hdf5_glob)  # pass a glob/dir of .hdf5 files; one row per file
        .sort(col("file").file_path())
        # episode_index must be contiguous 0-based in file-sorted order to match the
        # LeRobot dataset; row_number()-1 gives that (monotonically_increasing_id would not).
        .with_column("episode_index", row_number().over(per_file) - 1)
        # carry the HDF5 path so the video decoder can find each episode's sibling .mp4
        .with_column("_src", col("file").file_path())
        .into_batches(8)
        .with_column("_ep", process_egodex_episode(col("file")))
        .with_column("task", col("_ep")["task"])
        .with_column("frames", col("_ep")["frames"])
    )

    frames = (
        episodes.explode("frames")
        .select("episode_index", "task", "_src", col("frames").unnest())
        .with_column("timestamp", (col("frame_index") / FPS).cast(DataType.float32()))
    )

    if with_video:
        # observation.image is a lazy UDF column; embed_frames' frame_index % SUBSAMPLE
        # filter pushes below it, so only the kept (~1 fps) frames are ever decoded.
        frames = frames.with_column("observation.image", _decode_sibling_mp4(col("_src"), col("timestamp")))
    return frames.exclude("_src")


@daft.func(return_dtype=DataType.image("RGB"))
def _decode_sibling_mp4(hdf5_path: str, timestamp: float):
    """Decode the frame nearest `timestamp` (s) from the .mp4 sitting beside the .hdf5.

    Each EgoDex episode `<n>.hdf5` has a `<n>.mp4` next to it. Seek to the preceding
    keyframe, then walk forward to the frame closest in time (mirrors LeRobot's decode).
    """
    import av

    # Daft's file_path() carries a URI scheme (e.g. "file://.data/.../0.hdf5"); av/ffmpeg
    # would try to open that literal string and fail, so strip a leading "file://".
    if hdf5_path.startswith("file://"):
        hdf5_path = hdf5_path[len("file://"):]
    mp4_path = hdf5_path[:-len(".hdf5")] + ".mp4"
    target = float(timestamp)
    with av.open(mp4_path) as container:
        stream = container.streams.video[0]
        container.seek(int(target / stream.time_base), backward=True, stream=stream)
        best = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * stream.time_base)
            if best is None or abs(t - target) < abs(best[0] - target):
                best = (t, frame.to_ndarray(format="rgb24"))
            if t >= target:
                break
    if best is None:
        raise ValueError(f"no frame decoded from {mp4_path} at t={target:.3f}s")
    return best[1]


# potential new 


# @daft.func(return_dtype=DataType.image("RGB"))
# def _decode_sibling_mp4(video_file: daft.VideoFile, timestamp: float):
#     """Decode the frame nearest `timestamp` (s) from the .mp4 sitting beside the .hdf5.

#     Each EgoDex episode `<n>.hdf5` has a `<n>.mp4` next to it. Seek to the preceding
#     keyframe, then walk forward to the frame closest in time (mirrors LeRobot's decode).
#     """
#     import av

#     # Daft's file_path() carries a URI scheme (e.g. "file://.data/.../0.hdf5"); av/ffmpeg
#     # would try to open that literal string and fail, so strip a leading "file://".
#     if hdf5_path.startswith("file://"):
#         hdf5_path = hdf5_path[len("file://"):]
#     mp4_path = hdf5_path[:-len(".hdf5")] + ".mp4"
#     target = float(timestamp)
#     with video_file.open() as vf:
#         stream = container.streams.video[0]
#         container.seek(int(target / stream.time_base), backward=True, stream=stream)
#         best = None
#         for frame in container.decode(stream):
#             if frame.pts is None:
#                 continue
#             t = float(frame.pts * stream.time_base)
#             if best is None or abs(t - target) < abs(best[0] - target):
#                 best = (t, frame.to_ndarray(format="rgb24"))
#             if t >= target:
#                 break
#     if best is None:
#         raise ValueError(f"no frame decoded from {mp4_path} at t={target:.3f}s")
#     return best[1]
