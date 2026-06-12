# Google DiffusionGemma Daft cls

Backend: **offline vLLM** (`LLM` engine via the block-diffusion data path).

- `model.py` — the `DiffusionGemma` `@daft.cls(gpus=1.0, max_concurrency=1)` UDF. It loads the engine once per worker, applies the Gemma 4 chat template per prompt, and returns the generated text plus generation metadata.
- `modal_app.py` — the Modal deployment shell (image, Volume, entrypoints).

DiffusionGemma ([google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it)) is a block-diffusion language model on the Gemma 4 MoE backbone (26B total / 4B active). Instead of decoding left-to-right, it iteratively denoises fixed 256-token canvases in parallel — trading higher time-to-first-token for much higher per-request generation throughput (~1,288 gen tok/s on H200, ~5-6× the autoregressive baseline at low batch sizes).

## Where the usage example actually lives

The [vLLM announcement blog post](https://vllm-project.github.io/2026/06/10/diffusion-gemma) contains **no code**. The canonical usage examples are in:

- [vllm-project/recipes#520](https://github.com/vllm-project/recipes/pull/520) — the official recipe (serve command, offline `LLM()` usage, flag rationale). Not yet merged, so it does not appear on the recipes site.
- [DiffusionGemma: The Developer Guide](https://developers.googleblog.com/en/diffusiongemma-the-developer-guide/) — the `vllm serve` command.
- The [model card](https://huggingface.co/google/diffusiongemma-26B-A4B-it) — Transformers usage and sampler defaults.

## vLLM version requirement

vLLM support is still an open PR ([vllm-project/vllm#45163](https://github.com/vllm-project/vllm/pull/45163), `dgemma` branch) and **requires nightly wheels** — it is not in any stable release:

```bash
uv pip install -U vllm --pre \
  --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match
```

The Modal image in `modal_app.py` installs exactly this. A `vllm-openai:gemma-cu130` Docker image is also planned per the PR.

## Required engine flags

These are non-negotiable for DiffusionGemma and are baked into the UDF's `__init__`:

| Engine arg | Value | Why |
| --- | --- | --- |
| `max_num_seqs` | `4` | Diffusion state buffers pre-allocate `max_seqs × canvas_length × vocab_size` tensors. With Gemma's 262K vocab, higher values OOM an 80 GB GPU. |
| `generation_config` | `"vllm"` | The checkpoint's `generation_config.json` caps `max_tokens` at one canvas (256); this ignores it so per-request limits win. |
| `gpu_memory_utilization` | `0.85` | Headroom for activation memory during denoising. |
| `hf_overrides` | `{"diffusion_sampler": "entropy_bound", "diffusion_entropy_bound": 0.1}` | Configures the entropy-bound denoising sampler. |
| `diffusion_config` | `{"canvas_length": 256}` | Canvas block size for generation. |
| `enable_chunked_prefill` | `True` | Per the official serve command. |

The equivalent server form, from the recipe:

```bash
vllm serve google/diffusiongemma-26B-A4B-it \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.85 \
  --generation-config vllm \
  --hf-overrides '{"diffusion_sampler": "entropy_bound", "diffusion_entropy_bound": 0.1}' \
  --diffusion-config '{"canvas_length": 256}' \
  --enable-chunked-prefill
```

## Run on Modal

Prewarm model weights into the `diffusion-gemma-model-cache` Volume:

```bash
uv run --extra models modal run models/diffusion_gemma/modal_app.py --download-only
```

Single-prompt smoke test (H100):

```bash
uv run --extra models modal run models/diffusion_gemma/modal_app.py \
  --prompt "Why is the sky blue?" \
  --max-tokens 512
```

Thinking mode (structured reasoning via the Gemma 4 chat template):

```bash
uv run --extra models modal run models/diffusion_gemma/modal_app.py \
  --prompt "What is the derivative of x^3 * ln(x)?" \
  --max-tokens 4096 \
  --enable-thinking
```

## Run Locally

`model.py` also has a local CLI that calls the same `build_dataframe()` helper:

```bash
uv run python models/diffusion_gemma/model.py \
  --prompt "Why is the sky blue?" \
  --max-tokens 512
```

This path assumes your local environment already has the nightly vLLM stack from
the version requirement above and enough GPU memory for the checkpoint. Modal is
the recommended smoke-test path for most users because the image pins those
inference dependencies.

## Weight Loading

The Hugging Face model ID is downloaded through `snapshot_download` with `HF_HOME` / `HF_HUB_CACHE` pointed at the Modal model Volume (`/models/huggingface`), and the snapshot path is passed to vLLM. Use `--model-revision` to pin a checkpoint revision. The repo is gated — the `hf-token` Modal secret must hold a token with access.

## Output Shape

One row per prompt:

- `text` — the generated completion
- `prompt`
- `model`, `model_revision`, `model_path`
- `canvas_length`, `entropy_bound`
- `max_tokens`, `temperature`, `seed`, `enable_thinking`
- `num_prompt_tokens`, `num_generated_tokens`, `num_cached_tokens` — token accounting straight off the `RequestOutput`
- `finish_reason` — `"length"` means the output was truncated by `max_tokens`
- `generation_time_s` — wall-clock seconds per request, so per-row gen TPS (`num_generated_tokens / generation_time_s`) is a one-line projection. Useful for reproducing the diffusion-vs-autoregressive throughput comparison from the recipe.

The UDF also records Daft token metrics with `record_token_metrics(protocol="prompt", provider="vllm", ...)`,
matching the pattern used by Daft's OpenAI prompter. Use those aggregate metrics for job-level token accounting and
the row fields above for per-prompt throughput analysis.

Deliberately not captured: `logprobs` / `cumulative_logprob` (always `None` unless requested in `SamplingParams`, and per-token logprobs bloat rows — request them ad hoc if you need sampler-confidence analysis), `stop_reason` (only meaningful with stop strings, which this UDF doesn't set), `RequestOutput.metrics` (can be `None` depending on engine config — fragile as a fixed schema field), and `n > 1` completions (the dataframe-idiomatic way to get multiple samples is multiple rows with different seeds, which `build_dataframe` already does).

## Known Limitations

- **TTFT** is ~10× the autoregressive baseline — a full canvas must denoise before the first token emits.
- **Audio is not supported** — the diffusion checkpoints ship no audio encoder. (This UDF is text-only; the checkpoint does support images via the Gemma 4 vision encoder if you extend it.)
- Quantized FP8 / NVFP4 checkpoints exist for higher throughput; swap `--model` once their repo IDs are published.

## References

- vLLM blog announcement: https://vllm-project.github.io/2026/06/10/diffusion-gemma
- vLLM model support PR: https://github.com/vllm-project/vllm/pull/45163
- Official recipe PR: https://github.com/vllm-project/recipes/pull/520
- Google developer guide: https://developers.googleblog.com/en/diffusiongemma-the-developer-guide/
- Model card: https://huggingface.co/google/diffusiongemma-26B-A4B-it
- Fine-tuning (Hackable Diffusion): https://github.com/google-deepmind/gemma/tree/main/gemma/diffusion
