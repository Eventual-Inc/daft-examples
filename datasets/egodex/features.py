"""Episode-level pose features for EgoDex, computed in one pass per episode.

The feature stage reads the needed HDF5 transform datasets as whole-episode
tensors, builds compact hand-state and skeleton arrays, then derives continuous
per-frame tracks with vectorized NumPy. Keeping one Daft row per episode avoids
exploding every frame while preserving frame-accurate tracks for query-time
scenario matching.

Feature tracks per hand (tag ``L``/``R``):

    closure          mean finger flexion (low = open palm, high = fist)
    flex_nonthumb    (N, 4) per-finger flexion for index..little
    thumb_min_tip    thumb tip -> nearest of index/middle tip (writing grip)
    thumb_min_knuckle thumb tip -> nearest of index/middle knuckle (hammer grip)
    curl_rate        d(curl)/dt        (grasping)
    wrist_vert_vel   d(wrist y)/dt     (lifting)
    arm_ext_rate     d(arm extension)/dt (reaching)
    wrist_speed      |d(wrist)/dt|     (stillness)
    articulation     |d(hand-local joints)/dt| (in-hand manipulation)
    roll             wrist roll rate about the forearm axis, smoothed (twisting)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from daft import DataType

from . import skeleton_geometry, state_geometry
from .schemas import CAMERA, FEATURE_TRAJECTORY_FIELDS, SKELETON_TRANSFORMS, TIPS, WRIST

FPS = 30.0
HANDS = (("L", "left"), ("R", "right"))

# Rolling-mean half-width for the roll track (matches the old
# Window().rows_between(-2, 2) smoothing, including shrunken edge windows).
ROLL_SMOOTH_HALF_WIDTH = 2


@dataclass(frozen=True)
class EgoDexFrameBuilder:
    """Build compact frame arrays from one episode's raw HDF5 transforms."""

    skeleton_transforms: tuple[str, ...] = tuple(SKELETON_TRANSFORMS)
    camera_transform: str = CAMERA

    def hand_block(self, wrist: np.ndarray, tips: list[np.ndarray]) -> np.ndarray:
        """One hand's 24 state dims: wrist xyz + rot6d + fingertip xyz."""
        translation = wrist[:, :3, 3]
        rotation = np.concatenate([wrist[:, :3, 0], wrist[:, :3, 1]], axis=1)
        fingertips = np.concatenate([tip[:, :3, 3] for tip in tips], axis=1)
        return np.concatenate([translation, rotation, fingertips], axis=1)

    def build_state(self, transforms) -> np.ndarray:
        """Hand state (N, 48): left hand block then right hand block."""
        left = self.hand_block(
            transforms[WRIST["left"]], [transforms[name] for name in TIPS["left"]]
        )
        right = self.hand_block(
            transforms[WRIST["right"]], [transforms[name] for name in TIPS["right"]]
        )
        return np.concatenate([left, right], axis=1).astype(np.float32)

    def build_skeleton(self, transforms) -> np.ndarray:
        """Skeleton state (N, 204): joint xyz translations in skeleton order."""
        return np.concatenate(
            [transforms[name][:, :3, 3] for name in self.skeleton_transforms], axis=1
        ).astype(np.float32)

    def build_extrinsics(self, transforms) -> np.ndarray:
        """Camera pose (N, 16): the camera 4x4, row-major, per frame."""
        camera = np.asarray(transforms[self.camera_transform])
        return camera.reshape(camera.shape[0], 16).astype(np.float32)


# --- temporal rates (NumPy replacements for the old window functions) --------


@dataclass(frozen=True)
class TemporalFeatureComputer:
    """Compute per-frame rates from episode-length feature tracks."""

    fps: float = FPS
    roll_smooth_half_width: int = ROLL_SMOOTH_HALF_WIDTH

    @property
    def dt(self) -> float:
        return 1.0 / self.fps

    def forward_rate(self, values: np.ndarray) -> np.ndarray:
        """(next - current) / dt per frame, 0 at the episode's last frame."""
        rates = np.zeros(len(values), dtype=np.float64)
        if len(values) > 1:
            rates[:-1] = np.diff(values, axis=0) / self.dt
        return rates

    def forward_speed(self, points: np.ndarray) -> np.ndarray:
        """|next - current| / dt per frame over (N, d) points, 0 at the last frame."""
        speeds = np.zeros(len(points), dtype=np.float64)
        if len(points) > 1:
            speeds[:-1] = np.linalg.norm(np.diff(points, axis=0), axis=1) / self.dt
        return speeds

    def centered_mean(self, values: np.ndarray) -> np.ndarray:
        """Centered rolling mean with shrinking edge windows."""
        values = np.asarray(values, dtype=np.float64)
        smoothed = np.empty(len(values), dtype=np.float64)
        for index in range(len(values)):
            start = max(0, index - self.roll_smooth_half_width)
            stop = min(len(values), index + self.roll_smooth_half_width + 1)
            smoothed[index] = values[start:stop].mean()
        return smoothed

    def forearm_roll_rates(
        self, rot6d: np.ndarray, forearm_axis: np.ndarray
    ) -> np.ndarray:
        """Wrist roll rate (rad/s) about the forearm axis, per frame."""
        n = len(rot6d)
        rates = np.zeros(n, dtype=np.float64)
        if n < 2:
            return rates
        rotations = state_geometry.rotation_from_rot6d(
            np.asarray(rot6d, dtype=np.float64)
        )
        relative = np.einsum("nij,nkj->nik", rotations[1:], rotations[:-1])
        angles = np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1))
        axes = np.stack(
            [
                relative[:, 2, 1] - relative[:, 1, 2],
                relative[:, 0, 2] - relative[:, 2, 0],
                relative[:, 1, 0] - relative[:, 0, 1],
            ],
            axis=1,
        )
        magnitudes = np.linalg.norm(axes, axis=1)
        safe = magnitudes > 1e-9
        projected = np.zeros(n - 1, dtype=np.float64)
        projected[safe] = np.abs(
            angles[safe]
            * np.einsum(
                "nd,nd->n",
                axes[safe] / magnitudes[safe, None],
                forearm_axis[:-1][safe],
            )
        )
        rates[:-1] = projected / self.dt
        return rates


# --- the episode-level feature UDF -------------------------------------------

_TRACK = DataType.tensor(DataType.float32())

_SCALAR_TRACKS = (
    "closure",
    "thumb_min_tip",
    "thumb_min_knuckle",
    "curl_rate",
    "wrist_vert_vel",
    "arm_ext_rate",
    "wrist_speed",
    "articulation",
    "roll",
)


def _features_dtype() -> DataType:
    fields: dict[str, DataType] = {"num_frames": DataType.int64()}
    for tag, _ in HANDS:
        for name in _SCALAR_TRACKS:
            fields[f"{name}_{tag}"] = _TRACK  # (N,)
        fields[f"flex_nonthumb_{tag}"] = _TRACK  # (N, 4)
    return DataType.struct(fields)


POSE_FEATURES_DTYPE = _features_dtype()


@dataclass(frozen=True)
class EpisodeFeatureComputer:
    """Assemble queryable pose-feature tracks from one episode's raw transforms."""

    frame_builder: EgoDexFrameBuilder = field(default_factory=EgoDexFrameBuilder)
    temporal: TemporalFeatureComputer = field(default_factory=TemporalFeatureComputer)

    def _hand_tracks(
        self,
        *,
        tag: str,
        side: str,
        state: np.ndarray,
        state_features: dict[str, np.ndarray],
        skeleton_features: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        thumb_tip = skeleton_features[f"thumb_tip_dist_{tag}"]
        thumb_knuckle = skeleton_features[f"thumb_knuckle_dist_{tag}"]
        wrist = state_features[f"wrist_{tag}"]
        local_joints = skeleton_features[f"local_joints_{tag}"].reshape(len(state), -1)
        rot6d = state[:, state_geometry.rot6d_slice(side)]

        tracks = {
            "closure": skeleton_features[f"closure_{tag}"],
            "thumb_min_tip": thumb_tip[:, :2].min(axis=1),
            "thumb_min_knuckle": thumb_knuckle[:, :2].min(axis=1),
            "curl_rate": self.temporal.forward_rate(state_features[f"curl_{tag}"]),
            "wrist_vert_vel": self.temporal.forward_rate(wrist[:, 1]),
            "arm_ext_rate": self.temporal.forward_rate(
                skeleton_features[f"arm_extension_{tag}"]
            ),
            "wrist_speed": self.temporal.forward_speed(wrist),
            "articulation": self.temporal.forward_speed(local_joints),
            "roll": self.temporal.centered_mean(
                self.temporal.forearm_roll_rates(
                    rot6d, skeleton_features[f"forearm_axis_{tag}"]
                )
            ),
            "flex_nonthumb": skeleton_features[f"flex_nonthumb_{tag}"],
        }
        return {
            f"{name}_{tag}": values.astype(np.float32) for name, values in tracks.items()
        }

    def compute(self, transforms: dict[str, np.ndarray]) -> dict[str, object]:
        state = self.frame_builder.build_state(transforms).astype(np.float64)
        skeleton = self.frame_builder.build_skeleton(transforms).astype(np.float64)

        state_features = state_geometry.compute_raw_features(state)
        skeleton_features = skeleton_geometry.compute_state_features(skeleton)

        out: dict[str, object] = {"num_frames": len(state)}
        for tag, side in HANDS:
            out.update(
                self._hand_tracks(
                    tag=tag,
                    side=side,
                    state=state,
                    state_features=state_features,
                    skeleton_features=skeleton_features,
                )
            )
        return out


__all__ = [
    "FPS",
    "POSE_FEATURES_DTYPE",
    "FEATURE_TRAJECTORY_FIELDS",
    "SKELETON_TRANSFORMS",
    "EgoDexFrameBuilder",
    "EpisodeFeatureComputer",
    "TemporalFeatureComputer",
]
