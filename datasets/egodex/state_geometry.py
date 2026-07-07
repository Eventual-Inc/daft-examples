"""Per-frame hand-pose features from the 48-D EgoDex hand-state array.

Layout per hand (24 dims): wrist xyz [0:3], wrist rot6d [3:9], 5 fingertips
xyz [9:24] (order [thumb, index, middle, ring, pinky]). Left hand is [0:24],
right hand [24:48]. World +y is up (verified: wrists sit ~0.93 up, left hand -x /
right hand +x, units ~meters). Everything is vectorized over N frames and model-free.
"""

from __future__ import annotations

import numpy as np

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

# 48-D hand-state layout: per hand, 24 dims = wrist xyz (3) + wrist rot6d (6)
# + 5 fingertips xyz (15). Left hand occupies [0:24], right hand [24:48].
HAND_BLOCK_DIM = 24
ROT6D_OFFSET = 3  # rot6d sits at [3:9] within a hand block
ROT6D_LEN = 6
FINGERTIP_OFFSET = 9  # 5 fingertips xyz sit at [9:24]
HAND_BASE = {"left": 0, "right": HAND_BLOCK_DIM}
HAND_TAGS = [("left", "L"), ("right", "R")]


def rot6d_slice(side):
    """Column slice of a hand's wrist rot6d (6 values) within the 48-D state ('left' / 'right')."""
    start = HAND_BASE[side] + ROT6D_OFFSET
    return slice(start, start + ROT6D_LEN)


def _split_hand(state, base):
    wrist = state[:, base : base + 3]
    rot6d = state[:, base + ROT6D_OFFSET : base + ROT6D_OFFSET + ROT6D_LEN]
    fingertips = state[:, base + FINGERTIP_OFFSET : base + HAND_BLOCK_DIM].reshape(
        -1, 5, 3
    )
    return wrist, rot6d, fingertips


def palm_normal_from_rot6d(rot6d):
    """rot6d = the first two columns of the rotation matrix; the palm normal is their cross product."""
    first_column = rot6d[:, 0:3]
    second_column = rot6d[:, 3:6]
    first_column = first_column / (
        np.linalg.norm(first_column, axis=1, keepdims=True) + 1e-9
    )
    second_column = (
        second_column
        - (first_column * second_column).sum(1, keepdims=True) * first_column
    )
    second_column = second_column / (
        np.linalg.norm(second_column, axis=1, keepdims=True) + 1e-9
    )
    return np.cross(first_column, second_column)


def rotation_from_rot6d(rot6d):
    """rot6d (N, 6) -> (N, 3, 3) rotation matrices (columns = hand x, y axes + palm normal)."""
    first_column = rot6d[:, 0:3]
    first_column = first_column / (
        np.linalg.norm(first_column, axis=1, keepdims=True) + 1e-9
    )
    second_column = rot6d[:, 3:6]
    second_column = (
        second_column
        - (first_column * second_column).sum(1, keepdims=True) * first_column
    )
    second_column = second_column / (
        np.linalg.norm(second_column, axis=1, keepdims=True) + 1e-9
    )
    palm_normal = np.cross(first_column, second_column)
    return np.stack([first_column, second_column, palm_normal], axis=2)


def compute_raw_features(state: np.ndarray) -> dict:
    """Per-frame raw features for both hands, keyed by feature + hand tag ('L' / 'R')."""
    features = {}
    for side, tag in HAND_TAGS:
        wrist, rot6d, fingertips = _split_hand(state, HAND_BASE[side])
        tip_to_wrist = np.linalg.norm(fingertips - wrist[:, None, :], axis=2)  # (N, 5)
        features[f"fingerdist_{tag}"] = tip_to_wrist
        features[f"curl_{tag}"] = tip_to_wrist.mean(1)  # small = curled
        palm_normal = palm_normal_from_rot6d(rot6d)
        features[f"palmnormal_{tag}"] = palm_normal
        features[f"palm_up_{tag}"] = palm_normal[:, 1]  # +y component
        features[f"pinch_{tag}"] = np.linalg.norm(
            fingertips[:, 0] - fingertips[:, 1], axis=1
        )  # thumb-index
        tip_pairs = fingertips[:, :, None, :] - fingertips[:, None, :, :]
        features[f"aperture_{tag}"] = np.linalg.norm(tip_pairs, axis=3).max(
            (1, 2)
        )  # max tip-tip spread
        features[f"wrist_{tag}"] = wrist
    return features
