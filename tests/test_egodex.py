from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

import daft
from daft import DataType, col
from egodex import EgoDexPipeline, calibrate, query
from egodex.features import FPS, HANDS, LOCAL_JOINTS_DIM
from egodex.schemas import FEATURE_TRAJECTORY_FIELDS
from egodex.state_geometry import rotation_from_rot6d
from egodex.temporal import add_temporal_features
from egodex.viz import _read_frame_geometry, _resolve_episode


def _episode_transforms(num_frames: int) -> dict[str, np.ndarray]:
    time = np.arange(num_frames, dtype=np.float32)
    transforms = {name: np.tile(np.eye(4, dtype=np.float32), (num_frames, 1, 1)) for name in FEATURE_TRAJECTORY_FIELDS}

    def set_xyz(path: str, xyz: tuple[float, float, float]) -> None:
        values = np.asarray(xyz, dtype=np.float32)[None, :] + np.zeros((num_frames, 3), dtype=np.float32)
        values[:, 1] += 0.002 * time
        transforms[path][:, :3, 3] = values

    for side, x_base in (("left", -0.25), ("right", 0.25)):
        set_xyz(f"transforms/{side}Hand", (x_base, 1.0, 0.0))
        set_xyz(f"transforms/{side}Forearm", (x_base, 0.75, -0.02))
        set_xyz(f"transforms/{side}Arm", (x_base, 0.5, -0.04))
        set_xyz(f"transforms/{side}Shoulder", (x_base, 0.25, -0.04))

        x_sign = -1.0 if side == "left" else 1.0
        finger_offsets = {
            "Index": -0.045,
            "Middle": -0.015,
            "Ring": 0.015,
            "Little": 0.045,
        }
        for finger, offset in finger_offsets.items():
            prefix = f"transforms/{side}{finger}Finger"
            x = x_base + x_sign * offset
            set_xyz(f"{prefix}Metacarpal", (x, 1.025, 0.025))
            set_xyz(f"{prefix}Knuckle", (x, 1.055, 0.04))
            set_xyz(f"{prefix}IntermediateBase", (x, 1.095, 0.055))
            set_xyz(f"{prefix}IntermediateTip", (x, 1.135, 0.07))
            set_xyz(f"{prefix}Tip", (x, 1.175, 0.085))

        thumb_x = x_base - x_sign * 0.055
        set_xyz(f"transforms/{side}ThumbKnuckle", (thumb_x, 1.02, 0.03))
        set_xyz(f"transforms/{side}ThumbIntermediateBase", (thumb_x, 1.055, 0.02))
        set_xyz(f"transforms/{side}ThumbIntermediateTip", (thumb_x, 1.09, 0.01))
        set_xyz(f"transforms/{side}ThumbTip", (thumb_x, 1.125, 0.0))

    for index, path in enumerate(FEATURE_TRAJECTORY_FIELDS):
        if not path.startswith("transforms/"):
            continue
        if np.allclose(transforms[path][:, :3, 3], 0):
            set_xyz(path, (0.0, 0.1 + 0.01 * index, -0.2))

    return transforms


def _write_tiny_egodex(
    root: Path,
    task: str = "toy_task",
    episode_id: int = 0,
    num_frames: int = 8,
) -> None:
    task_dir = root / task
    task_dir.mkdir(parents=True)
    with h5py.File(task_dir / f"{episode_id}.hdf5", "w") as h5:
        h5.attrs["task"] = task
        h5.attrs["llm_verbs"] = np.asarray(["test"], dtype=h5py.string_dtype())
        h5.attrs["llm_objects"] = np.asarray(["hand"], dtype=h5py.string_dtype())
        h5.create_dataset(
            "camera/intrinsic",
            data=np.asarray(
                [[736.6339, 0.0, 960.0], [0.0, 736.6339, 540.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
        )
        for path, values in _episode_transforms(num_frames=num_frames).items():
            h5.create_dataset(path, data=values)


def test_egodex_pipeline_reads_features_and_queries(tmp_path: Path) -> None:
    _write_tiny_egodex(tmp_path)

    pipeline = EgoDexPipeline(str(tmp_path))
    episodes = pipeline.raw(tasks="toy_task", episode_ids=0)
    raw = episodes.select("task", "episode_id", "video").to_pydict()
    assert raw["task"] == ["toy_task"]
    assert raw["episode_id"] == [0]
    assert raw["video"] == [None]

    trajectories = pipeline.trajectory(episodes)
    features = pipeline.calculate_features(trajectories)
    data = features.select("task", "episode_id", "frame_index", "closure_L", "flex_nonthumb_R").to_pydict()

    assert data["task"] == ["toy_task"] * 8
    assert data["episode_id"] == [0] * 8
    assert sorted(data["frame_index"]) == list(range(8))
    assert len(data["flex_nonthumb_R"][0]) == 4
    assert np.isfinite(np.asarray(data["closure_L"])).all()

    thresholds = calibrate(features)
    hits = query(features, pose="openness", k=1, thresholds=thresholds, fps=FPS)
    assert hits == [
        {
            "task": "toy_task",
            "episode_id": 0,
            "score": 8.0,
            "n_frames": 8,
            "segments": [(0, 7)],
        }
    ]


def test_egodex_viz_resolves_episode_and_validates_frame_index(
    tmp_path: Path,
) -> None:
    _write_tiny_egodex(tmp_path, task="toy_task", episode_id=3)

    episode_path = _resolve_episode(tmp_path, "toy_task", 3)
    assert episode_path == tmp_path / "toy_task" / "3.hdf5"

    with h5py.File(episode_path, "r") as h5:
        joints, camera_pose, intrinsic = _read_frame_geometry(h5, frame_index=0)
        assert joints.shape == (68, 3)
        assert camera_pose.shape == (4, 4)
        assert intrinsic.shape == (3, 3)

        try:
            _read_frame_geometry(h5, frame_index=8)
        except IndexError as exc:
            assert "outside episode bounds" in str(exc)
        else:
            raise AssertionError("expected frame bounds validation")


def test_short_episode_roll_tracks_stay_episode_length(tmp_path: Path) -> None:
    _write_tiny_egodex(tmp_path, task="short_task", episode_id=0, num_frames=3)

    pipeline = EgoDexPipeline(str(tmp_path))
    features = pipeline.calculate_features(pipeline.trajectory(pipeline.raw(tasks="short_task", episode_ids=0)))
    data = features.select("frame_index", "roll_L", "roll_R").to_pydict()

    assert sorted(data["frame_index"]) == [0, 1, 2]
    assert np.isfinite(np.asarray(data["roll_L"])).all()
    assert np.isfinite(np.asarray(data["roll_R"])).all()

    hits = query(features, pose="twisting", k=1, fps=FPS)
    for hit in hits:
        for start, end in hit["segments"]:
            assert 0 <= start <= end < 3


def _reference_rate(values: np.ndarray, dt: float) -> np.ndarray:
    rates = np.zeros(len(values))
    rates[:-1] = np.diff(values) / dt
    return rates


def _reference_speed(points: np.ndarray, dt: float) -> np.ndarray:
    speeds = np.zeros(len(points))
    speeds[:-1] = np.linalg.norm(np.diff(points, axis=0), axis=1) / dt
    return speeds


def _reference_smooth(values: np.ndarray, half_width: int = 2) -> np.ndarray:
    return np.asarray(
        [values[max(0, i - half_width) : i + half_width + 1].mean() for i in range(len(values))],
    )


def _reference_roll(rot6d: np.ndarray, forearm_axis: np.ndarray, dt: float) -> np.ndarray:
    rates = np.zeros(len(rot6d))
    rotations = rotation_from_rot6d(rot6d)
    for i in range(len(rot6d) - 1):
        delta = rotations[i + 1] @ rotations[i].T
        angle = np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1))
        axis = np.array([delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0], delta[1, 0] - delta[0, 1]])
        magnitude = np.linalg.norm(axis)
        if magnitude >= 1e-9:
            rates[i] = abs(angle * np.dot(axis / magnitude, forearm_axis[i])) / dt
    return rates


def test_temporal_window_rates_match_numpy_reference() -> None:
    rng = np.random.default_rng(7)
    dt = 1.0 / FPS
    episodes = {0: 6, 1: 3}

    per_episode: dict[int, dict[str, np.ndarray]] = {}
    rows: dict[str, list] = {"task": [], "episode_id": [], "frame_index": []}
    for tag, _ in HANDS:
        for name in ("curl", "wrist_y", "arm_extension", "wrist", "local_joints", "wrist_rot6d", "forearm_axis"):
            rows[f"{name}_{tag}"] = []

    for episode_id, n in episodes.items():
        tracks: dict[str, np.ndarray] = {}
        for tag, _ in HANDS:
            tracks[f"curl_{tag}"] = rng.normal(size=n)
            tracks[f"wrist_y_{tag}"] = rng.normal(size=n)
            tracks[f"arm_extension_{tag}"] = rng.normal(size=n)
            tracks[f"wrist_{tag}"] = rng.normal(size=(n, 3))
            tracks[f"local_joints_{tag}"] = rng.normal(size=(n, LOCAL_JOINTS_DIM))
            tracks[f"wrist_rot6d_{tag}"] = rng.normal(size=(n, 6))
            axes = rng.normal(size=(n, 3))
            tracks[f"forearm_axis_{tag}"] = axes / np.linalg.norm(axes, axis=1, keepdims=True)
        per_episode[episode_id] = tracks
        # scrambled row order: the window's order_by must recover frame order
        for frame in rng.permutation(n):
            rows["task"].append("toy_task")
            rows["episode_id"].append(episode_id)
            rows["frame_index"].append(int(frame))
            for name, values in tracks.items():
                value = values[frame]
                rows[name].append(value.tolist() if isinstance(value, np.ndarray) else float(value))

    frames = daft.from_pydict(rows)
    for tag, _ in HANDS:
        for name, dim in (("wrist", 3), ("local_joints", LOCAL_JOINTS_DIM), ("wrist_rot6d", 6), ("forearm_axis", 3)):
            frames = frames.with_column(
                f"{name}_{tag}",
                col(f"{name}_{tag}").cast(DataType.fixed_size_list(DataType.float64(), dim)),
            )

    out = add_temporal_features(frames).sort(["episode_id", "frame_index"]).to_pydict()

    cursor = 0
    for episode_id, n in episodes.items():
        window = slice(cursor, cursor + n)
        tracks = per_episode[episode_id]
        for tag, _ in HANDS:
            expected = {
                "curl_rate": _reference_rate(tracks[f"curl_{tag}"], dt),
                "wrist_vert_vel": _reference_rate(tracks[f"wrist_y_{tag}"], dt),
                "arm_ext_rate": _reference_rate(tracks[f"arm_extension_{tag}"], dt),
                "wrist_speed": _reference_speed(tracks[f"wrist_{tag}"], dt),
                "articulation": _reference_speed(tracks[f"local_joints_{tag}"], dt),
                "roll": _reference_smooth(
                    _reference_roll(tracks[f"wrist_rot6d_{tag}"], tracks[f"forearm_axis_{tag}"], dt)
                ),
            }
            for name, values in expected.items():
                actual = np.asarray(out[f"{name}_{tag}"][window])
                np.testing.assert_allclose(actual, values, rtol=1e-9, atol=1e-9, err_msg=f"{name}_{tag} ep{episode_id}")
        cursor += n


def test_text_and_combined_queries_use_precomputed_embeddings(tmp_path: Path) -> None:
    _write_tiny_egodex(tmp_path, task="toy_task", episode_id=0)

    pipeline = EgoDexPipeline(str(tmp_path))
    features = pipeline.calculate_features(pipeline.trajectory(pipeline.raw(tasks="toy_task", episode_ids=0)))
    thresholds = calibrate(features)
    clip = daft.from_pydict(
        {
            "task": ["toy_task", "toy_task"],
            "episode_id": [0, 0],
            "frame_index": [0, 4],
            "clip_emb": [
                np.asarray([1.0, 0.0], dtype=np.float32),
                np.asarray([0.0, 1.0], dtype=np.float32),
            ],
        }
    )

    def encode(text: str) -> np.ndarray:
        assert text == "target"
        return np.asarray([0.0, 1.0], dtype=np.float32)

    text_hits = query(
        features,
        text="target",
        clip=clip,
        encode=encode,
        k=1,
        fps=FPS,
        semantic_window=0.1,
    )
    assert text_hits == [
        {
            "task": "toy_task",
            "episode_id": 0,
            "score": 1.0,
            "n_frames": 2,
            "segments": [(1, 7)],
        }
    ]

    combined_hits = query(
        features,
        pose="openness",
        text="target",
        clip=clip,
        encode=encode,
        k=1,
        thresholds=thresholds,
        fps=FPS,
    )
    assert combined_hits == [
        {
            "task": "toy_task",
            "episode_id": 0,
            "score": 1.0,
            "n_frames": 2,
            "segments": [(0, 4)],
        }
    ]
