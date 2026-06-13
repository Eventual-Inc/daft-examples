"""Modal deployment shell for the DiffusionGemma model UDF.

The model wrapper itself lives in ``model.py``; this file only owns the
container image, Volumes, and entrypoints.

DiffusionGemma support is not in a stable vLLM release yet
(vllm-project/vllm#45163), so the image installs vLLM nightly wheels per the
official recipe (vllm-project/recipes#520).
"""

from __future__ import annotations

import modal

import daft
from daft import col
from daft.functions import file, unnest
from models.diffusion_gemma.model import (
    DEFAULT_CANVAS_LENGTH,
    DEFAULT_ENTROPY_BOUND,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    DiffusionGemma,
)
from models.weights import MODEL_CACHE, function_kwargs, with_model_cache

GPU_TYPE = "H100"

app = modal.App("daft-diffusion-gemma")

image = with_model_cache(
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    .run_commands(
        # Nightly wheels are the official install path until vLLM's diffusion
        # support ships in a stable release (vllm-project/recipes#520).
        "uv pip install --system --pre -U vllm 'daft>=0.7.14' huggingface_hub pillow"
        " --extra-index-url https://wheels.vllm.ai/nightly/cu129"
        " --extra-index-url https://download.pytorch.org/whl/cu129"
        " --index-strategy unsafe-best-match"
    ),
    extra_env={"LD_LIBRARY_PATH": "/usr/local/lib/python3.12/site-packages/nvidia/cu13/lib"},
)


@app.function(**function_kwargs(image, cpu=4, enable_memory_snapshot=False))
def download_model_weights(model: str = DEFAULT_MODEL, model_revision: str = "") -> dict:
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=model, revision=model_revision or None)
    MODEL_CACHE.commit()
    return {"model": model, "model_revision": model_revision, "model_path": path}


@app.function(**function_kwargs(image, gpu=GPU_TYPE, memory=65536))
def run_on_modal(
    prompts: list[str],
    image_path: str = "",
    model: str = DEFAULT_MODEL,
    model_revision: str = "",
    canvas_length: int = DEFAULT_CANVAS_LENGTH,
    entropy_bound: float = DEFAULT_ENTROPY_BOUND,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    seed: int = 42,
    enable_thinking: bool = False,
):
    gemma = DiffusionGemma(
        model=model,
        model_revision=model_revision,
        canvas_length=canvas_length,
        entropy_bound=entropy_bound,
        max_model_len=max_model_len,
    )
    data = {
        "prompt": prompts,
        "seed": [seed + index for index in range(len(prompts))],
        "max_tokens": [max_tokens] * len(prompts),
        "temperature": [temperature] * len(prompts),
        "enable_thinking": [enable_thinking] * len(prompts),
    }
    if image_path:
        data["image_path"] = [image_path] * len(prompts)
    df = daft.from_pydict(data)
    if image_path:
        df = df.with_column("image_file", file(col("image_path"))).with_column(
            "result",
            gemma.generate_from_image(
                col("prompt"),
                col("image_file"),
                max_tokens=col("max_tokens"),
                temperature=col("temperature"),
                seed=col("seed"),
                enable_thinking=col("enable_thinking"),
            ),
        )
    else:
        df = df.with_column(
            "result",
            gemma.generate(
                col("prompt"),
                max_tokens=col("max_tokens"),
                temperature=col("temperature"),
                seed=col("seed"),
                enable_thinking=col("enable_thinking"),
            ),
        )
    df = df.select(unnest(col("result"))).collect()

    MODEL_CACHE.commit()
    return df.to_pydict()


@app.local_entrypoint()
def modal_main(
    prompt: str = DEFAULT_PROMPT,
    image_path: str = "",
    model: str = DEFAULT_MODEL,
    model_revision: str = "",
    canvas_length: int = DEFAULT_CANVAS_LENGTH,
    entropy_bound: float = DEFAULT_ENTROPY_BOUND,
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    seed: int = 42,
    enable_thinking: bool = False,
    download_only: bool = False,
):
    if download_only:
        print(download_model_weights.remote(model=model, model_revision=model_revision))
        return

    result = run_on_modal.remote(
        prompts=[prompt],
        image_path=image_path,
        model=model,
        model_revision=model_revision,
        canvas_length=canvas_length,
        entropy_bound=entropy_bound,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
        enable_thinking=enable_thinking,
    )
    for text in result["text"]:
        print(text)
