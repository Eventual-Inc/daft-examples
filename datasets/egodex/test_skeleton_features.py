# /// script
# description = "Verify the 204-D skeleton geometry features against synthetic and real EgoDex data"
# requires-python = ">=3.10, <3.13"
# dependencies = ["numpy", "pyarrow"]
# ///
"""Verify skeleton_features (the static geometry behind the wired scenarios)
against synthetic ground truth + the real 204-D data.

Run:  PYTHONPATH=. lr_venv/bin/python test_skeleton_features.py
"""
import glob
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import pose_features as PF       # 48-D reference, for cross-checks
import skeleton_features as SK

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA", os.path.join(_HERE, "egodex_lerobot_full"))


def col_to_np(table, name):
    column = table.column(name).combine_chunks()
    return column.values.to_numpy(zero_copy_only=False).reshape(table.num_rows, -1)


def set_joint(skeleton, name, xyz):
    start = SK.JOINT_INDEX[name] * 3
    skeleton[:, start:start + 3] = xyz


# ---------------------------------------------------------------- synthetic
def test_angle_primitive():
    first = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1.0, 0, 0]])
    second = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [1.0, 1, 0]])
    degrees = np.degrees(SK.angle_between(first, second))
    assert np.allclose(degrees, [0, 90, 180, 45.0], atol=1e-4), degrees
    print(f"[synthetic] angle primitive: {degrees.round(2)}  OK")


def test_finger_flexion_synthetic():
    skeleton = np.zeros((2, 204))
    parts = ["Metacarpal", "Knuckle", "IntermediateBase", "IntermediateTip", "Tip"]
    for i, part in enumerate(parts):                                # frame 0: straight finger -> ~0 flexion
        set_joint(skeleton[0:1], f"leftIndexFinger{part}", [0.02 * i, 0, 0])
    curled = {"Metacarpal": [0, 0, 0], "Knuckle": [0.02, 0, 0], "IntermediateBase": [0.04, 0, 0],
              "IntermediateTip": [0.05, -0.01, 0], "Tip": [0.05, -0.03, 0]}    # frame 1: distal bent
    for part, xyz in curled.items():
        set_joint(skeleton[1:2], f"leftIndexFinger{part}", xyz)
    flexion = SK.finger_flexion(skeleton, "left", "Index")          # (2, 3) MCP/PIP/DIP
    assert flexion[0].max() < np.radians(2), flexion[0]
    assert flexion[1].sum() > flexion[0].sum() + np.radians(20), flexion
    print(f"[synthetic] finger flexion straight={np.degrees(flexion[0]).round(1)} curled={np.degrees(flexion[1]).round(1)}  OK")


def test_grip_predicates_synthetic():
    """Verify the grip classifiers on crafted frames: tripod, hammer, open, fist."""
    flexion = np.array([[0.9, 0.9, 1.8, 1.8],    # tripod: idx/mid moderate, ring/little more curled
                        [2.6, 2.6, 2.6, 2.6],     # hammer: all curled
                        [0.3, 0.3, 0.3, 0.3],     # open
                        [2.6, 2.6, 2.6, 2.6]])    # fist (no thumb wrap)
    thumb_tip = np.array([[0.15, 0.2, 0.6, 0.8], [0.7, 0.7, 0.8, 0.9], [0.9, 0.9, 1.0, 1.1], [0.5, 0.5, 0.6, 0.7]])
    thumb_knuckle = np.array([[0.6, 0.6, 0.7, 0.8], [0.15, 0.2, 0.7, 0.8], [0.9, 0.9, 1.0, 1.1], [0.5, 0.5, 0.6, 0.7]])
    features = {"flex_nonthumb_L": flexion, "thumb_tip_dist_L": thumb_tip, "thumb_knuckle_dist_L": thumb_knuckle}
    thresholds = {"curled_flexion": 2.3, "thumb_on_tip": 0.3, "thumb_on_knuckle": 0.3, "curl_gap": np.radians(20)}
    writing = SK.is_writing_grip(features, thresholds, "L").tolist()
    hammer = SK.is_hammer_grip(features, thresholds, "L").tolist()
    assert writing == [True, False, False, False], writing
    assert hammer == [False, True, False, False], hammer
    print(f"[synthetic] grip predicates: writing={writing} hammer={hammer}  OK")


# ---------------------------------------------------------------- real data
def load_real(n_shards=2):
    files = sorted(glob.glob(f"{DATA}/data/**/*.parquet", recursive=True))[:n_shards]
    table = pa.concat_tables([pq.read_table(f) for f in files])
    skeleton = col_to_np(table, "observation.skeleton").astype(np.float64)
    state = col_to_np(table, "observation.state").astype(np.float64)
    return skeleton, state


def rot6d_palm_normal(state, side):
    """The TRUE palm normal from 48-D rot6d = column 1 of the orthonormalized rotation matrix."""
    return PF.rotation_from_rot6d(state[:, PF.rot6d_slice(side)])[:, :, 1]


def calibrate_palm_sign(skeleton, state):
    """Pick PALM_SIGN[side] so the best-fit palm normal aligns (+) with rot6d column 1."""
    signs = {}
    for side in ("left", "right"):
        SK.PALM_SIGN[side] = 1.0
        normal = SK.palm_normal(skeleton, side)
        signs[side] = 1.0 if (normal * rot6d_palm_normal(state, side)).sum(1).mean() >= 0 else -1.0
    return signs


def main():
    test_angle_primitive()
    test_finger_flexion_synthetic()
    test_grip_predicates_synthetic()

    print("\nloading real 204-D data ...")
    skeleton, state = load_real()
    print(f"  frames={len(skeleton)}")

    SK.PALM_SIGN.update(calibrate_palm_sign(skeleton, state))
    print(f"  calibrated PALM_SIGN = {SK.PALM_SIGN}")

    features = SK.compute_state_features(skeleton)
    grip_thresholds = SK.calibrate_grip_thresholds(features)

    checks = []
    closure = np.concatenate([features["closure_L"], features["closure_R"]])
    checks.append(("closure finite & in range",
                   np.isfinite(closure).all() and 0 <= closure.min() and closure.max() <= 5 * np.pi,
                   f"min={closure.min():.2f} max={closure.max():.2f}"))

    alignment = [(SK.palm_normal(skeleton, side) * rot6d_palm_normal(state, side)).sum(1).mean()
                 for side in ("left", "right")]
    checks.append(("palm normal aligns with rot6d col1", min(alignment) > 0.9, f"mean dot L/R={np.round(alignment, 3)}"))

    # openness orientation: closure low = open. Cross-check vs 48-D fingertip->wrist distance (large = open).
    wrist = state[:, 0:3]
    fingertips = state[:, 9:24].reshape(-1, 5, 3)
    tip_distance = np.linalg.norm(fingertips - wrist[:, None, :], axis=2).mean(1)
    correlation = np.corrcoef(features["closure_L"], tip_distance)[0, 1]
    checks.append(("closure low = open (neg corr w/ tip-dist)", correlation < -0.5, f"corr={correlation:.3f}"))

    arm_extension = np.concatenate([features["arm_extension_L"], features["arm_extension_R"]])
    checks.append(("arm extension in (0, 1.2]",
                   arm_extension.min() > 0 and arm_extension.max() <= 1.2,
                   f"min={arm_extension.min():.2f} max={arm_extension.max():.2f}"))

    print("\n=== correctness checks ===")
    all_passed = True
    for name, passed, detail in checks:
        all_passed &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:40} {detail}")

    grips = {
        "writing grip": SK.is_writing_grip(features, grip_thresholds, "L") | SK.is_writing_grip(features, grip_thresholds, "R"),
        "hammer grip": SK.is_hammer_grip(features, grip_thresholds, "L") | SK.is_hammer_grip(features, grip_thresholds, "R"),
    }
    print(f"\n=== grip hits (over {len(skeleton)} frames) ===")
    for name, mask in grips.items():
        print(f"  {name:14} {int(mask.sum()):6d} frames ({100 * mask.mean():.2f}%)")

    print("\nRESULT:", "ALL CHECKS PASS" if all_passed else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
