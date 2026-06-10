# SAM 3D Body Daft cls

Backend: **PyTorch** (the upstream `sam-3d-body` estimator; vLLM does not support this model).

- `model.py` — the `Sam3DBody` `@daft.cls(gpus=1.0, max_concurrency=1)` UDF. It loads the model once per worker, writes rendered previews and mesh artifacts to disk, and returns a lightweight metadata struct. Also a runnable local CLI (needs Python 3.11 plus the deps in its header).
- `modal_app.py` — the Modal deployment shell (image, Volumes, entrypoints).
- `viewer.html` — browser viewer for single-image mesh outputs.

For video → tracked 3D pose sequences, see the end-to-end pipeline in
[`pipelines/pose_sequence/`](../../pipelines/pose_sequence/), which composes this
model with frame sampling and YOLO tracking.

## Run on Modal

The SAM 3D Body Hugging Face checkpoints are gated. The Modal function uses the existing `hf-token` secret and caches weights in the `sam3d-body-model-cache` volume.

Prewarm SAM weights into the Modal Volume:

```bash
uv run --extra models modal run models/sam3d_body/modal_app.py \
  --download-only \
  --model facebook/sam-3d-body-vith
```

Smoke test with the upstream sample image:

```bash
uv run --extra models modal run models/sam3d_body/modal_app.py
```

Run a specific image:

```bash
uv run --extra models modal run models/sam3d_body/modal_app.py \
  --image-path /workspace/path/inside/this/repo/person.jpg
```

Use a bbox prompt and skip detector dependencies:

```bash
uv run --extra models modal run models/sam3d_body/modal_app.py \
  --image-path /workspace/path/inside/this/repo/person.jpg \
  --bbox "[120,40,520,760]"
```

The default model is `facebook/sam-3d-body-vith` because it avoids the extra DINOv3 `torch.hub` path. You can switch to the DINOv3 checkpoint after access is approved:

```bash
uv run --extra models modal run models/sam3d_body/modal_app.py \
  --model facebook/sam-3d-body-dinov3
```

Use `--model-revision` to pin a Hugging Face revision. Repo IDs are downloaded into the Modal Volume at:

```text
/models/huggingface/repos/<repo-id>/<revision-or-main>
```

If you have already staged checkpoint files in the Modal volume or image, bypass Hugging Face repo downloads:

```bash
uv run --extra models modal run models/sam3d_body/modal_app.py \
  --checkpoint-path /models/sam-3d-body-vith/model.ckpt \
  --mhr-path /models/sam-3d-body-vith/assets/mhr_model.pt
```

## Output Shape

The UDF writes one output directory per input image and returns:

- `render_path`: side-by-side source, mesh overlay, and side-view render
- `mesh_paths`: per-person PLY mesh files
- `npz_paths`: compact per-person arrays for vertices, camera, bbox, and keypoints
- `metadata_path`: JSON metadata with per-person artifact paths
- `num_people`, model path and revision, detector settings, and prompt settings

By default the example does not load ViTDet, SAM2, or MoGe2. That keeps the first PyTorch path lean and runnable. Add those optional components after the baseline works if detector, mask, or FOV quality becomes the bottleneck.

## References

- SAM 3D Body GitHub repository: https://github.com/facebookresearch/sam-3d-body
- SAM 3D Body Hugging Face checkpoints: https://huggingface.co/facebook/sam-3d-body-vith and https://huggingface.co/facebook/sam-3d-body-dinov3
- Upstream installation notes: https://github.com/facebookresearch/sam-3d-body/blob/main/INSTALL.md
