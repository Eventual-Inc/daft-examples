"""Per-frame spatial pose features for EgoDex.

The feature stage reads the needed HDF5 transform datasets as whole-episode
tensors, builds compact hand-state and skeleton arrays, then derives the
instantaneous (single-frame) geometry with vectorized NumPy and emits one
struct per frame. Temporal action rates (grasping, lifting, twisting, ...)
are deliberately NOT computed here: they are Daft window expressions over the
exploded per-frame rows — see :mod:`temporal`.

Spatial columns per hand (tag ``L``/``R``):

    closure          mean finger flexion (low = open palm, high = fist)
    flex_nonthumb    (4,) per-finger flexion for index..little
    thumb_min_tip    thumb tip -> nearest of index/middle tip (writing grip)
    thumb_min_knuckle thumb tip -> nearest of index/middle knuckle (hammer grip)
    curl             mean fingertip-to-wrist distance (rate = grasping)
    arm_extension    wrist-to-shoulder reach (rate = reaching)
    wrist_y          wrist height (rate = lifting)
    wrist            (3,) wrist position (rate = wrist speed)
    local_joints     (72,) finger joints in the hand frame (rate = in-hand)
    wrist_rot6d      (6,) wrist rotation (consecutive pairs = forearm roll)
    forearm_axis     (3,) forearm direction (roll projection axis)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from daft import DataType

from . import skeleton_geometry, state_geometry
from .schemas import CAMERA, FEATURE_TRAJECTORY_FIELDS, SKELETON_TRANSFORMS, TIPS, WRIST

FPS = 30.0
HANDS = (("L", "left"), ("R", "right"))


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
        left = self.hand_block(transforms[WRIST["left"]], [transforms[name] for name in TIPS["left"]])
        right = self.hand_block(transforms[WRIST["right"]], [transforms[name] for name in TIPS["right"]])
        return np.concatenate([left, right], axis=1).astype(np.float32)

    def build_skeleton(self, transforms) -> np.ndarray:
        """Skeleton state (N, 204): joint xyz translations in skeleton order."""
        return np.concatenate([transforms[name][:, :3, 3] for name in self.skeleton_transforms], axis=1).astype(
            np.float32
        )

    def build_extrinsics(self, transforms) -> np.ndarray:
        """Camera pose (N, 16): the camera 4x4, row-major, per frame."""
        camera = np.asarray(transforms[self.camera_transform])
        return camera.reshape(camera.shape[0], 16).astype(np.float32)


# --- the per-frame spatial feature dtype --------------------------------------

LOCAL_JOINTS_DIM = 3 * sum(
    len(skeleton_geometry.finger_joint_names("left", finger)) for finger in skeleton_geometry.FINGERS
)

_SPATIAL_SCALARS = (
    "closure",
    "thumb_min_tip",
    "thumb_min_knuckle",
    "curl",
    "arm_extension",
    "wrist_y",
)

_SPATIAL_VECTORS = {
    "flex_nonthumb": 4,
    "wrist": 3,
    "local_joints": LOCAL_JOINTS_DIM,
    "wrist_rot6d": 6,
    "forearm_axis": 3,
}


def _frame_dtype() -> DataType:
    fields: dict[str, DataType] = {"frame_index": DataType.int64()}
    for tag, _ in HANDS:
        for name in _SPATIAL_SCALARS:
            fields[f"{name}_{tag}"] = DataType.float64()
        for name, dim in _SPATIAL_VECTORS.items():
            fields[f"{name}_{tag}"] = DataType.fixed_size_list(DataType.float64(), dim)
    return DataType.struct(fields)


FRAME_FEATURES_DTYPE = DataType.list(_frame_dtype())


# --- the episode-level spatial UDF payload ------------------------------------


@dataclass(frozen=True)
class SpatialFeatureComputer:
    """Assemble per-frame spatial geometry from one episode's raw transforms."""

    frame_builder: EgoDexFrameBuilder = field(default_factory=EgoDexFrameBuilder)

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

        tracks = {
            "closure": skeleton_features[f"closure_{tag}"],
            "flex_nonthumb": skeleton_features[f"flex_nonthumb_{tag}"],
            "thumb_min_tip": thumb_tip[:, :2].min(axis=1),
            "thumb_min_knuckle": thumb_knuckle[:, :2].min(axis=1),
            "curl": state_features[f"curl_{tag}"],
            "arm_extension": skeleton_features[f"arm_extension_{tag}"],
            "wrist_y": wrist[:, 1],
            "wrist": wrist,
            "local_joints": skeleton_features[f"local_joints_{tag}"].reshape(len(state), -1),
            "wrist_rot6d": state[:, state_geometry.rot6d_slice(side)],
            "forearm_axis": skeleton_features[f"forearm_axis_{tag}"],
        }
        return {f"{name}_{tag}": np.asarray(values, dtype=np.float64) for name, values in tracks.items()}

    def compute(self, transforms: dict[str, np.ndarray]) -> list[dict[str, object]]:
        """One spatial-feature dict per frame, ready to explode into rows."""
        state = self.frame_builder.build_state(transforms).astype(np.float64)
        skeleton = self.frame_builder.build_skeleton(transforms).astype(np.float64)

        state_features = state_geometry.compute_raw_features(state)
        skeleton_features = skeleton_geometry.compute_state_features(skeleton)

        tracks: dict[str, np.ndarray] = {}
        for tag, side in HANDS:
            tracks.update(
                self._hand_tracks(
                    tag=tag,
                    side=side,
                    state=state,
                    state_features=state_features,
                    skeleton_features=skeleton_features,
                )
            )

        frames: list[dict[str, object]] = []
        for index in range(len(state)):
            row: dict[str, object] = {"frame_index": index}
            for name, values in tracks.items():
                value = values[index]
                row[name] = value.tolist() if value.ndim else float(value)
            frames.append(row)
        return frames


__all__ = [
    "FPS",
    "FRAME_FEATURES_DTYPE",
    "HANDS",
    "LOCAL_JOINTS_DIM",
    "FEATURE_TRAJECTORY_FIELDS",
    "SKELETON_TRANSFORMS",
    "EgoDexFrameBuilder",
    "SpatialFeatureComputer",
]
