"""Per-frame hand geometry from the 204-D EgoDex skeleton array.

The skeleton array is 68 joints x xyz (flat (N, 204)); joint i occupies
[3i : 3i+3], in the order produced by features.SKELETON_TRANSFORMS. Per
side: Hand (wrist), Forearm, Arm, Shoulder, then each finger chain, then body.
Finger chains: Thumb = Knuckle / IntermediateBase / IntermediateTip / Tip; the
other four add a leading Metacarpal. Clinical mapping: Knuckle = MCP,
IntermediateBase = PIP, IntermediateTip = DIP. World +y is up, units ~meters.

This module computes only STATIC, per-frame geometry (one frame in, one value
out). The per-episode ACTION rates (reaching, twisting, in-hand) are computed
vectorized in features.py. Everything here is vectorized over N frames and
model-free.
"""

from __future__ import annotations

import numpy as np

SIDES = ("left", "right")
FINGERS = ("Thumb", "Index", "Middle", "Ring", "Little")

# The best-fit palm-plane normal is sign-ambiguous; flip it to a consistent
# palm-facing convention. Calibrated against the 48-D rot6d palm normal
# (see the skeleton geometry tests in the upstream prototype).
PALM_SIGN = {"left": -1.0, "right": -1.0}


def _build_joint_names():
    joint_names = []
    for side in SIDES:
        joint_names += [side + part for part in ("Hand", "Forearm", "Arm", "Shoulder")]
        for finger in FINGERS:
            joint_names += finger_joint_names(side, finger)
    joint_names += [
        "hip",
        *(f"spine{i}" for i in range(1, 8)),
        *(f"neck{i}" for i in range(1, 5)),
    ]
    return joint_names


def finger_part_names(finger):
    """The joint parts along a finger, wrist-to-tip (thumb has no metacarpal)."""
    leading = ["Metacarpal"] if finger != "Thumb" else []
    return leading + ["Knuckle", "IntermediateBase", "IntermediateTip", "Tip"]


def finger_joint_names(side, finger):
    infix = "" if finger == "Thumb" else "Finger"
    return [f"{side}{finger}{infix}{part}" for part in finger_part_names(finger)]


JOINT_NAMES = _build_joint_names()
JOINT_INDEX = {name: i for i, name in enumerate(JOINT_NAMES)}


def joint_position(skeleton, joint_name):
    """(N, 3) world position of a named joint from the (N, 204) skeleton."""
    start = JOINT_INDEX[joint_name] * 3
    return skeleton[:, start : start + 3]


def _unit(vectors, epsilon=1e-9):
    return vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + epsilon)


def angle_between(first_bone, second_bone):
    """Row-wise angle (rad, [0, pi]) between two vectors. atan2 form is stable at 0/pi."""
    cross_magnitude = np.linalg.norm(np.cross(first_bone, second_bone), axis=-1)
    dot_product = (first_bone * second_bone).sum(-1)
    return np.arctan2(cross_magnitude, dot_product)


def finger_flexion(skeleton, side, finger):
    """(N, n_joints) per-joint flexion angles along a finger chain.

    Flexion = the angle between consecutive bone vectors (0 = straight, larger =
    more curled). Non-thumb fingers yield MCP / PIP / DIP (3); the thumb yields 2.
    """
    positions = [joint_position(skeleton, name) for name in finger_joint_names(side, finger)]
    bones = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    joint_angles = [angle_between(bones[i], bones[i + 1]) for i in range(len(bones) - 1)]
    return np.stack(joint_angles, axis=1)


def palm_normal(skeleton, side):
    """(N, 3) unit palm-plane normal via a best-fit plane through the knuckles.

    Plane fit to {wrist + the four non-thumb knuckles}; the normal is the
    eigenvector of the smallest eigenvalue. Sign-oriented by handedness then
    PALM_SIGN so it points consistently (palm-facing) across both hands.
    """
    wrist = joint_position(skeleton, side + "Hand")
    knuckles = [joint_position(skeleton, f"{side}{f}FingerKnuckle") for f in ("Index", "Middle", "Ring", "Little")]
    points = np.stack([wrist] + knuckles, axis=1)
    centered = points - points.mean(1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered)
    _, eigenvectors = np.linalg.eigh(covariance)  # ascending; [:, :, 0] = smallest
    normal = eigenvectors[:, :, 0]
    # orient deterministically by handedness, then flip to the palm-facing convention
    index_knuckle = joint_position(skeleton, f"{side}IndexFingerKnuckle")
    little_knuckle = joint_position(skeleton, f"{side}LittleFingerKnuckle")
    reference = np.cross(index_knuckle - wrist, little_knuckle - wrist)
    orientation = np.sign((normal * reference).sum(1))
    orientation[orientation == 0] = 1.0
    return normal * orientation[:, None] * PALM_SIGN[side]


def hand_scale(skeleton, side):
    """(N,) characteristic hand size (wrist -> middle knuckle) for normalizing distances."""
    middle_knuckle = joint_position(skeleton, f"{side}MiddleFingerKnuckle")
    wrist = joint_position(skeleton, side + "Hand")
    return np.linalg.norm(middle_knuckle - wrist, axis=1)


def arm_extension(skeleton, side):
    """(N,) wrist-to-shoulder distance / total arm length (0 = tucked, ~1 = straight)."""
    shoulder = joint_position(skeleton, side + "Shoulder")
    upper_arm = joint_position(skeleton, side + "Arm")
    forearm = joint_position(skeleton, side + "Forearm")
    wrist = joint_position(skeleton, side + "Hand")
    arm_length = (
        np.linalg.norm(upper_arm - shoulder, axis=1)
        + np.linalg.norm(forearm - upper_arm, axis=1)
        + np.linalg.norm(wrist - forearm, axis=1)
    )
    return np.linalg.norm(wrist - shoulder, axis=1) / (arm_length + 1e-9)


def forearm_axis(skeleton, side):
    """(N, 3) unit vector along the forearm (Forearm -> Hand) — the roll axis for twisting."""
    return _unit(joint_position(skeleton, side + "Hand") - joint_position(skeleton, side + "Forearm"))


def hand_local_joints(skeleton, side):
    """(N, K, 3) the side's finger joints expressed in the hand's own frame.

    The frame is z = palm normal, x = across-palm, origin = wrist; expressing the
    joints in it removes gross hand translation/rotation, leaving only the
    fingers' own motion (used to detect in-hand manipulation).
    """
    wrist = joint_position(skeleton, side + "Hand")
    across_palm = joint_position(skeleton, f"{side}LittleFingerKnuckle") - joint_position(
        skeleton, f"{side}IndexFingerKnuckle"
    )
    z_axis = _unit(palm_normal(skeleton, side))
    x_axis = _unit(across_palm - (across_palm * z_axis).sum(1, keepdims=True) * z_axis)
    y_axis = np.cross(z_axis, x_axis)
    frame = np.stack([x_axis, y_axis, z_axis], axis=1)  # (N, 3, 3), rows = axes
    joint_names = [name for finger in FINGERS for name in finger_joint_names(side, finger)]
    positions = np.stack([joint_position(skeleton, name) for name in joint_names], axis=1)  # (N, K, 3)
    return np.einsum("nij,nkj->nki", frame, positions - wrist[:, None, :])


def compute_state_features(skeleton: np.ndarray) -> dict:
    """Per-frame STATE geometry for both hands (the inputs to every scenario).

    Returns one dict of (N, ...) arrays keyed by feature + hand tag ('L'/'R'):
      closure          mean finger flexion (low = open palm, high = fist)
      flex_nonthumb    (N, 4) per-finger flexion for index..little (grip taxonomy)
      thumb_tip_dist   (N, 4) thumb tip -> each fingertip   (tripod / precision)
      thumb_knuckle_dist (N, 4) thumb tip -> each knuckle   (power wrap)
      arm_extension    wrist-to-shoulder reach (for reaching)
      wrist            (N, 3) wrist position (for wrist speed)
      forearm_axis     (N, 3) forearm direction (for forearm roll / twisting)
      local_joints     (N, K, 3) finger joints in the hand frame (for in-hand)
    """
    skeleton = np.asarray(skeleton, dtype=np.float64)
    features = {}
    for side, tag in (("left", "L"), ("right", "R")):
        per_finger_flexion = np.stack([finger_flexion(skeleton, side, f).sum(1) for f in FINGERS], axis=1)  # (N, 5)
        features[f"flex_nonthumb_{tag}"] = per_finger_flexion[:, 1:]  # index..little
        features[f"closure_{tag}"] = per_finger_flexion[:, 1:].mean(1)

        scale = hand_scale(skeleton, side)
        thumb_tip = joint_position(skeleton, f"{side}ThumbTip")
        fingertips = {f: joint_position(skeleton, f"{side}{f}FingerTip") for f in FINGERS[1:]}
        knuckles = {f: joint_position(skeleton, f"{side}{f}FingerKnuckle") for f in FINGERS[1:]}
        features[f"thumb_tip_dist_{tag}"] = np.stack(
            [np.linalg.norm(thumb_tip - fingertips[f], axis=1) / (scale + 1e-9) for f in FINGERS[1:]],
            axis=1,
        )
        features[f"thumb_knuckle_dist_{tag}"] = np.stack(
            [np.linalg.norm(thumb_tip - knuckles[f], axis=1) / (scale + 1e-9) for f in FINGERS[1:]],
            axis=1,
        )

        features[f"arm_extension_{tag}"] = arm_extension(skeleton, side)
        features[f"wrist_{tag}"] = joint_position(skeleton, side + "Hand")
        features[f"forearm_axis_{tag}"] = forearm_axis(skeleton, side)
        features[f"local_joints_{tag}"] = hand_local_joints(skeleton, side)
    return features


# The grip predicates (writing/hammer) and their threshold calibration live in the
# query layer (calibrated thresholds at query time). This module stays pure NumPy —
# the continuous geometry compute_state_features emits.
