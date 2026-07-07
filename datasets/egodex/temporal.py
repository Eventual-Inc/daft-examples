"""In-DAG temporal action rates over per-frame EgoDex features.

Every rate the scenario queries consume is a Daft window expression over
``Window().partition_by("task", "episode_id").order_by("frame_index")``:
next-frame diffs via ``lead(1)``, point speeds via ``euclidean_distance``,
and smoothing via a centered ``rows_between(-2, 2)`` mean (which shrinks at
episode edges, so short episodes stay episode-length). The one custom UDF,
``forearm_roll``, is fed by the same window — this frame's wrist rotation
plus the next frame's via ``lead(1)`` — and reduces the pair to a roll rate
about the forearm axis.

Because the rates are expressions, the whole computation stays in the query
plan: no collect, and the same code runs whether the per-frame rows come from
``EgoDexPipeline.frame_features`` or from any per-frame table with the same
columns (e.g. LeRobot-style parquet).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import daft
from daft import DataType, col
from daft.functions import euclidean_distance
from daft.window import Window

from . import state_geometry
from .features import FPS, HANDS

if TYPE_CHECKING:
    from daft.dataframe import DataFrame

# Rolling-mean half-width for the roll track; rows_between shrinks the window
# at episode edges, matching a centered mean with shrunken edge windows.
ROLL_SMOOTH_HALF_WIDTH = 2

PER_EPISODE = Window().partition_by("task", "episode_id").order_by("frame_index")
SMOOTH = (
    Window()
    .partition_by("task", "episode_id")
    .order_by("frame_index")
    .rows_between(-ROLL_SMOOTH_HALF_WIDTH, ROLL_SMOOTH_HALF_WIDTH)
)


@daft.func(return_dtype=DataType.float64(), use_process=False)
def forearm_roll(rot6d, rot6d_next, forearm_axis) -> float:
    """Wrist roll (rad) about the forearm axis from one frame to the next.

    ``rot6d_next`` arrives via ``lead(1)`` over the per-episode window, so it
    is null on each episode's last frame — which maps to a roll of 0 there.
    """
    if rot6d is None or rot6d_next is None:
        return 0.0
    rotations = state_geometry.rotation_from_rot6d(np.asarray([rot6d, rot6d_next], dtype=np.float64))
    delta = rotations[1] @ rotations[0].T
    angle = np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1))
    axis = np.array(
        [
            delta[2, 1] - delta[1, 2],
            delta[0, 2] - delta[2, 0],
            delta[1, 0] - delta[0, 1],
        ]
    )
    magnitude = np.linalg.norm(axis)
    if magnitude < 1e-9:
        return 0.0
    return float(abs(angle * np.dot(axis / magnitude, np.asarray(forearm_axis, dtype=np.float64))))


def add_temporal_features(frames: DataFrame, *, fps: float = FPS) -> DataFrame:
    """Add per-hand action-rate columns to a per-frame feature DataFrame.

    Expects the spatial per-frame columns from ``frame_features(...)`` and adds,
    per hand tag ``L``/``R``:

        curl_rate        d(curl)/dt          (grasping)
        wrist_vert_vel   d(wrist y)/dt       (lifting)
        arm_ext_rate     d(arm extension)/dt (reaching)
        wrist_speed      |d(wrist)/dt|       (stillness)
        articulation     |d(hand-local joints)/dt| (in-hand manipulation)
        roll             wrist roll rate about the forearm axis, smoothed (twisting)

    Rates are 0 at each episode's last frame (``lead(1)`` is null there).
    """
    dt = 1.0 / fps
    df = frames
    for tag, _ in HANDS:
        df = (
            df.with_column(
                f"curl_rate_{tag}",
                ((col(f"curl_{tag}").lead(1).over(PER_EPISODE) - col(f"curl_{tag}")) / dt).fill_null(0.0),
            )
            .with_column(
                f"wrist_vert_vel_{tag}",
                ((col(f"wrist_y_{tag}").lead(1).over(PER_EPISODE) - col(f"wrist_y_{tag}")) / dt).fill_null(0.0),
            )
            .with_column(
                f"arm_ext_rate_{tag}",
                ((col(f"arm_extension_{tag}").lead(1).over(PER_EPISODE) - col(f"arm_extension_{tag}")) / dt).fill_null(
                    0.0
                ),
            )
            .with_column(
                f"wrist_speed_{tag}",
                (euclidean_distance(col(f"wrist_{tag}"), col(f"wrist_{tag}").lead(1).over(PER_EPISODE)) / dt).fill_null(
                    0.0
                ),
            )
            .with_column(
                f"articulation_{tag}",
                (
                    euclidean_distance(
                        col(f"local_joints_{tag}"),
                        col(f"local_joints_{tag}").lead(1).over(PER_EPISODE),
                    )
                    / dt
                ).fill_null(0.0),
            )
            .with_column(
                f"roll_raw_{tag}",
                forearm_roll(
                    col(f"wrist_rot6d_{tag}"),
                    col(f"wrist_rot6d_{tag}").lead(1).over(PER_EPISODE),
                    col(f"forearm_axis_{tag}"),
                )
                / dt,
            )
            .with_column(f"roll_{tag}", col(f"roll_raw_{tag}").mean().over(SMOOTH))
        )
    return df


__all__ = [
    "PER_EPISODE",
    "ROLL_SMOOTH_HALF_WIDTH",
    "SMOOTH",
    "add_temporal_features",
    "forearm_roll",
]
