# Pose Sequence Pipeline

End-to-end video → tracked 3D pose sequence. Composes the
[`models/sam3d_body`](../../models/sam3d_body/) Daft UDF with frame sampling and
YOLO person tracking:

1. Sample frames from the input video.
2. Run YOLO person detection on each frame.
3. Track one athlete bbox across sampled frames (Viterbi-style smoothing).
4. Pass the bbox prompts into the SAM 3D Body UDF.
5. Write per-frame PLY meshes, render overlays, and `sequence.json`.

## Run on Modal

Prewarm both SAM and YOLO weights:

```bash
uv run --extra models modal run pipelines/pose_sequence/pose_sequence.py \
  --download-only \
  --model facebook/sam-3d-body-vith \
  --yolo-model yolov8n.pt
```

YOLO release assets are stored under `/models/ultralytics`, for example `/models/ultralytics/yolov8n.pt`.

Dense run, one pose every 10 encoded source frames with no cap:

```bash
uv run --extra models modal run pipelines/pose_sequence/pose_sequence.py \
  --video-path /path/to/input-video.mov \
  --sample-mode stride \
  --frame-stride 10 \
  --max-frames 0 \
  --yolo-confidence 0.18
```

When `--video-path` points to a local file, the entrypoint uploads it to the `sam3d-body-inputs` Modal
Volume and passes the mounted `/inputs/...` path to the remote GPU function. The source video is not
copied into the Modal image.

Run every encoded video keyframe for a denser sequence:

```bash
uv run --extra models modal run pipelines/pose_sequence/pose_sequence.py \
  --video-path /path/to/input-video.mov \
  --sample-mode keyframes \
  --max-frames 0 \
  --yolo-confidence 0.18
```

## Viewing the sequence

The Modal run prints `volume_sequence_path`, which points to the sequence directory inside the
`sam3d-body-outputs` Volume. Download that directory before opening the browser viewer:

```bash
uv run --extra models modal volume get \
  sam3d-body-outputs \
  /<sequence-dir-name> \
  .context/sam3d-body-video
```

Open the local sequence viewer from a workspace HTTP server:

```bash
uv run python -m http.server 8766 --bind 127.0.0.1
```

Then open:

```text
http://127.0.0.1:8766/pipelines/pose_sequence/sequence_viewer.html?manifest=../../.context/sam3d-body-video/<sequence-dir-name>/sequence.json
```

The sequence manifest includes deterministic per-frame orientation metadata derived from SAM 3D Body
geometry, not from an LLM. The pipeline loads `pred_keypoints_2d` and `pred_keypoints_3d` from each NPZ,
estimates an upper-body to lower-body axis in image space and mesh space, then writes the resulting
`mesh_rotation.z` correction. The viewer applies that frame's `mesh_rotation`; exported PLY files already
use SAM 3D Body's renderer coordinate system. The viewer also shows the sampled source frame with the
selected bbox in the bottom-right corner so pose orientation can be checked while scrubbing or playing.
Successful video runs include a `profile` block in `sequence.json` with remote stage timings for frame
extraction, YOLO tracking, SAM 3D Body inference, artifact writing, and volume commits.
