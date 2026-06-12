# Pose Sequence Pipeline

End-to-end video → tracked 3D pose sequence. Composes the
[`models/sam3d_body`](../../models/sam3d_body/) Daft UDF with frame sampling and
YOLO person tracking:

1. Sample frames from the input video.
2. Run YOLO person detection on each frame.
3. Track one athlete bbox across sampled frames (Viterbi-style smoothing).
4. Pass the bbox prompts into the SAM 3D Body UDF.
5. Write per-frame PLY meshes, render overlays, skeleton/hand keypoints, heuristic scene lines, and `sequence.json`.

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

If there are multiple people in the frame, pass a starting identity anchor for the athlete. Use normalized
image coordinates in the first sampled frame:

```bash
uv run --extra models modal run pipelines/pose_sequence/pose_sequence.py \
  --video-path /path/to/input-video.mov \
  --sample-mode stride \
  --frame-stride 3 \
  --max-frames 0 \
  --target-point 0.815,0.52 \
  --yolo-confidence 0.18
```

The target point is used to choose the initial YOLO candidate, then the tracker follows that person with
motion, size, and color-continuity penalties to avoid jumping to bystanders.

Scene geometry defaults to `--scene-detector contact`, which keeps the OpenCV line fallback but also scores
pole candidates by proximity to SAM hand keypoints. This catches frames where the pole is visible only near
the hands or where the brightest/longest image line is not the pole:

```bash
uv run --extra models modal run pipelines/pose_sequence/pose_sequence.py \
  --video-path /path/to/input-video.mov \
  --sample-mode stride \
  --frame-stride 3 \
  --max-frames 0 \
  --target-point 0.815,0.52 \
  --scene-detector contact
```

For open-vocabulary scene segmentation, use SAM 3 when `facebook/sam3` access is approved and `sam3.pt`
can be downloaded into the model cache:

```bash
uv run --extra models modal run pipelines/pose_sequence/pose_sequence.py \
  --download-only \
  --scene-detector sam3

uv run --extra models modal run pipelines/pose_sequence/pose_sequence.py \
  --video-path /path/to/input-video.mov \
  --sample-mode stride \
  --frame-stride 3 \
  --max-frames 0 \
  --target-point 0.815,0.52 \
  --scene-detector sam3 \
  --scene-prompts "pole=pole vault pole,vaulting pole;bar=pole vault crossbar,crossbar;standard=pole vault standard,upright standard"
```

The SAM 3 scene path converts semantic masks into pole/crossbar/standard lines or polylines, then merges
them with the contact-aware fallback. If SAM 3 weights are missing, gated, or CUDA is not visible to the
semantic predictor, the manifest records the error and falls back to contact-aware scene geometry. Set
`SAM3_ALLOW_CPU=1` only for short debugging runs where the slow CPU path is acceptable.

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
The stick figure includes SAM 3D Body's MHR70 body and hand keypoints. The viewer can toggle the body
skeleton, hand joints, mesh wireframe, and scene references independently, and the source-frame preview
draws the same 2D body/hand landmarks for comparison. These are sparse hand landmarks rather than a
separate high-resolution hand mesh, so closed fingers may appear as tight landmark clusters.

Pole, bar, and runway overlays are heuristic visual references. The pipeline uses OpenCV Hough lines on
each sampled source frame, then maps confident 2D line candidates into approximate 3D reference geometry
near the body mesh. Crossbar depth is anchored from the SAM 2D/3D keypoint correspondence when available
so occluded bar segments can be visualized near the athlete instead of at an arbitrary background depth.
Pole detections also produce a `contact` block per frame with wrist/hand distance to the pole, which helps
flag frames where the body model violates the expected pole-holding geometry.
These are not camera-calibrated reconstructions, but they are useful for judging pose against the vault
scene while scrubbing the sequence.
Successful video runs include a `profile` block in `sequence.json` with remote stage timings for frame
extraction, YOLO tracking, SAM 3D Body inference, artifact writing, and volume commits.
