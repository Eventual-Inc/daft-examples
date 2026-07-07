"""Schemas, HDF5 paths, and field catalogs for EgoDex."""

from __future__ import annotations

import daft

_F32 = daft.DataType.float32()
_TENSOR = daft.DataType.tensor(_F32)

# The 68 upper-body and hand joints tracked with a per-frame confidence score.
JOINTS: tuple[str, ...] = (
    "hip",
    "leftArm",
    "leftForearm",
    "leftHand",
    "leftIndexFingerIntermediateBase",
    "leftIndexFingerIntermediateTip",
    "leftIndexFingerKnuckle",
    "leftIndexFingerMetacarpal",
    "leftIndexFingerTip",
    "leftLittleFingerIntermediateBase",
    "leftLittleFingerIntermediateTip",
    "leftLittleFingerKnuckle",
    "leftLittleFingerMetacarpal",
    "leftLittleFingerTip",
    "leftMiddleFingerIntermediateBase",
    "leftMiddleFingerIntermediateTip",
    "leftMiddleFingerKnuckle",
    "leftMiddleFingerMetacarpal",
    "leftMiddleFingerTip",
    "leftRingFingerIntermediateBase",
    "leftRingFingerIntermediateTip",
    "leftRingFingerKnuckle",
    "leftRingFingerMetacarpal",
    "leftRingFingerTip",
    "leftShoulder",
    "leftThumbIntermediateBase",
    "leftThumbIntermediateTip",
    "leftThumbKnuckle",
    "leftThumbTip",
    "neck1",
    "neck2",
    "neck3",
    "neck4",
    "rightArm",
    "rightForearm",
    "rightHand",
    "rightIndexFingerIntermediateBase",
    "rightIndexFingerIntermediateTip",
    "rightIndexFingerKnuckle",
    "rightIndexFingerMetacarpal",
    "rightIndexFingerTip",
    "rightLittleFingerIntermediateBase",
    "rightLittleFingerIntermediateTip",
    "rightLittleFingerKnuckle",
    "rightLittleFingerMetacarpal",
    "rightLittleFingerTip",
    "rightMiddleFingerIntermediateBase",
    "rightMiddleFingerIntermediateTip",
    "rightMiddleFingerKnuckle",
    "rightMiddleFingerMetacarpal",
    "rightMiddleFingerTip",
    "rightRingFingerIntermediateBase",
    "rightRingFingerIntermediateTip",
    "rightRingFingerKnuckle",
    "rightRingFingerMetacarpal",
    "rightRingFingerTip",
    "rightShoulder",
    "rightThumbIntermediateBase",
    "rightThumbIntermediateTip",
    "rightThumbKnuckle",
    "rightThumbTip",
    "spine1",
    "spine2",
    "spine3",
    "spine4",
    "spine5",
    "spine6",
    "spine7",
)

# The 69 SE(3) transform datasets: every joint plus the egocentric camera pose.
TRANSFORM_JOINTS: tuple[str, ...] = ("camera", *JOINTS)

# All 138 HDF5 dataset paths present in every EgoDex episode file, in file order.
TRAJECTORY_FIELDS: tuple[str, ...] = (
    "camera/intrinsic",
    *(f"confidences/{joint}" for joint in JOINTS),
    *(f"transforms/{joint}" for joint in TRANSFORM_JOINTS),
)

TRAJECTORY_DTYPES: dict[str, daft.DataType] = {
    field: _TENSOR for field in TRAJECTORY_FIELDS
}

METADATA_FIELD_DTYPES: dict[str, daft.DataType] = {
    "task": daft.DataType.string(),
    "llm_description": daft.DataType.string(),
    "llm_description2": daft.DataType.string(),
    "which_llm_description": daft.DataType.string(),
    "llm_type": daft.DataType.string(),
    "llm_verbs": daft.DataType.list(daft.DataType.string()),
    "llm_objects": daft.DataType.list(daft.DataType.string()),
    "environment": daft.DataType.string(),
    "object": daft.DataType.string(),
    "session_name": daft.DataType.string(),
    "annotated": daft.DataType.bool(),
    "annotator_version": daft.DataType.string(),
    "extra": daft.DataType.string(),
    "description": daft.DataType.string(),
    "description2": daft.DataType.string(),
    "type": daft.DataType.string(),
}

METADATA_DTYPE = daft.DataType.struct(METADATA_FIELD_DTYPES)
METADATA_FIELDS: tuple[str, ...] = tuple(METADATA_FIELD_DTYPES)
LIST_METADATA_FIELDS: frozenset[str] = frozenset(("llm_verbs", "llm_objects"))

WRIST = {"left": "transforms/leftHand", "right": "transforms/rightHand"}
FINGERS = ["Thumb", "Index", "Middle", "Ring", "Little"]
TIPS = {
    side: [
        f"transforms/{side}{finger}{'' if finger == 'Thumb' else 'Finger'}Tip"
        for finger in FINGERS
    ]
    for side in ("left", "right")
}
CAMERA = "transforms/camera"


def finger_transforms(side: str, finger: str) -> list[str]:
    infix = "" if finger == "Thumb" else "Finger"
    parts = (["Metacarpal"] if finger != "Thumb" else []) + [
        "Knuckle",
        "IntermediateBase",
        "IntermediateTip",
        "Tip",
    ]
    return [f"transforms/{side}{finger}{infix}{part}" for part in parts]


def side_transforms(side: str) -> list[str]:
    arm = [f"transforms/{side}{j}" for j in ("Hand", "Forearm", "Arm", "Shoulder")]
    fingers = [t for finger in FINGERS for t in finger_transforms(side, finger)]
    return arm + fingers


BODY_TRANSFORMS = [
    f"transforms/{j}"
    for j in (
        "hip",
        *(f"spine{i}" for i in range(1, 8)),
        *(f"neck{i}" for i in range(1, 5)),
    )
]
SKELETON_TRANSFORMS = side_transforms("left") + side_transforms("right") + BODY_TRANSFORMS
SKELETON_DIM = len(SKELETON_TRANSFORMS) * 3
FEATURE_TRAJECTORY_FIELDS = tuple(dict.fromkeys(SKELETON_TRANSFORMS + [CAMERA]))


__all__ = [
    "BODY_TRANSFORMS",
    "CAMERA",
    "FEATURE_TRAJECTORY_FIELDS",
    "FINGERS",
    "JOINTS",
    "LIST_METADATA_FIELDS",
    "METADATA_DTYPE",
    "METADATA_FIELD_DTYPES",
    "METADATA_FIELDS",
    "SKELETON_DIM",
    "SKELETON_TRANSFORMS",
    "TIPS",
    "TRAJECTORY_DTYPES",
    "TRAJECTORY_FIELDS",
    "TRANSFORM_JOINTS",
    "WRIST",
    "side_transforms",
]
