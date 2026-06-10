# NVIDIA Cosmos 3 Daft cls

Backend: **offline vLLM** via vLLM-Omni (`Omni` engine, `Cosmos3OmniDiffusersPipeline`).

- `model.py` — the `Cosmos3Omni` `@daft.cls(gpus=1.0, max_concurrency=1)` UDF. It loads the engine once per worker, writes each generated image or video to disk, and returns plain metadata plus the artifact path.
- `modal_app.py` — the Modal deployment shell (image, Volumes, entrypoints).

## Run on Modal

Prewarm model weights into the `cosmos3-model-cache` Volume:

```bash
uv run --extra models modal run models/cosmos3/modal_app.py \
  --download-only \
  --model nvidia/Cosmos3-Nano
```

Image smoke test:

```bash
uv run --extra models modal run models/cosmos3/modal_app.py \
  --prompt "A photorealistic red sports car at golden hour, cinematic lighting." \
  --steps 2
```

Smaller video test:

```bash
uv run --extra models modal run models/cosmos3/modal_app.py \
  --modality video \
  --width 832 --height 480 --num-frames 49 --steps 20 \
  --prompt "A mobile warehouse robot approaches a shelf, pauses for a passing worker, then continues safely."
```

Recipe-grade text-to-video run:

```bash
uv run --extra models modal run models/cosmos3/modal_app.py \
  --modality video \
  --width 1280 --height 720 --num-frames 189 --fps 24 \
  --steps 35 --guidance-scale 6.0 --seed 123 \
  --negative-prompt "blurry, distorted, low quality, jittery, deformed" \
  --prompt "A robot arm is cleaning a plate in the kitchen"
```

The vLLM-Omni recipe's polished video defaults are 1280x720, 189 frames, 24 FPS, and 35 denoising steps. The recipe reports peak video memory around 46 GiB and roughly 90-100 seconds on high-end GPUs.

## Weight Loading

Hugging Face model IDs are downloaded through `snapshot_download` with `HF_HOME` and `HF_HUB_CACHE` pointed at the Modal model Volume:

```text
/models/huggingface
```

Inference passes the downloaded snapshot path to vLLM-Omni. Use `--model-revision` to pin a checkpoint revision and avoid surprise upstream changes.

## Output Shape

The UDF writes artifacts to the `cosmos3-outputs` Modal Volume and returns:

- `output_path`
- `model`
- `model_revision`
- `model_path`
- `modality`
- `prompt`
- `negative_prompt`
- `seed`
- `width`, `height`, `num_frames`, `fps`
- `num_inference_steps`, `guidance_scale`

Videos are written directly as MP4s with an `ffmpeg` subprocess (`models/common/media.py`). This keeps generated video as a Volume artifact instead of pushing a large `list[Image]` through the UDF or Modal return boundary.

## References

- vLLM-Omni Cosmos3-Nano recipe: https://github.com/vllm-project/vllm-omni/blob/main/recipes/nvidia/Cosmos3-Nano.md
- NVIDIA Newsroom: https://nvidianews.nvidia.com/news/nvidia-launches-cosmos-3-the-open-frontier-foundation-model-for-physical-ai
- NVIDIA technical blog: https://developer.nvidia.com/blog/develop-physical-ai-reasoning-world-and-action-models-with-nvidia-cosmos-3/
- NVIDIA Cosmos product page: https://www.nvidia.com/en-us/ai/cosmos/
- NVIDIA Cosmos GitHub repository: https://github.com/NVIDIA/cosmos
