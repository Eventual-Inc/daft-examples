# /// script
# description = "Deploy the Cosmos 3 Daft UDF on Modal (driver deps only; inference deps live in the image)"
# requires-python = ">=3.12, <3.13"
# dependencies = [
#   "daft>=0.7.14",
#   "modal",
# ]
# ///
"""Modal deployment shell for the Cosmos 3 model UDF.

The model wrapper itself lives in ``model.py``; this file only owns the
container image, Volumes, and entrypoints.
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.common.modal_infra import MODEL_CACHE_DIR, OUTPUT_DIR, hf_cache_env
from models.cosmos3.model import DEFAULT_MODEL, DEFAULT_NEGATIVE_PROMPT, DEFAULT_PROMPT, build_dataframe

GPU_TYPE = "A100-80GB"
MODAL_REGION = ["us-west"]

app = modal.App("daft-cosmos3-omni")
model_cache = modal.Volume.from_name("cosmos3-model-cache", create_if_missing=True)
outputs = modal.Volume.from_name("cosmos3-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0", "libxcb1")
    .pip_install(
        "av",
        "daft>=0.7.14",
        "huggingface_hub",
        "vllm==0.22.0",
        "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git",
    )
    .env(
        {
            **hf_cache_env(MODEL_CACHE_DIR),
            "COSMOS3_OUTPUT_DIR": OUTPUT_DIR,
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_python_source("models")
)


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=7200,
    region=MODAL_REGION,
    volumes={MODEL_CACHE_DIR: model_cache},
    secrets=[modal.Secret.from_name("hf-token")],
)
def download_model_weights(model: str = DEFAULT_MODEL, model_revision: str = "") -> dict:
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=model, revision=model_revision or None)
    model_cache.commit()
    return {"model": model, "model_revision": model_revision, "model_path": path}


@app.function(
    image=image,
    gpu=GPU_TYPE,
    cpu=8,
    memory=98304,
    timeout=7200,
    region=MODAL_REGION,
    volumes={MODEL_CACHE_DIR: model_cache, OUTPUT_DIR: outputs},
    secrets=[modal.Secret.from_name("hf-token")],
)
def run_on_modal(
    prompts: list[str],
    modality: str = "image",
    model: str = DEFAULT_MODEL,
    model_revision: str = "",
    width: int = 1024,
    height: int = 1024,
    num_frames: int = 1,
    fps: int = 24,
    steps: int = 10,
    guidance_scale: float = 7.0,
    flow_shift: float = 10.0,
    seed: int = 42,
    guardrails: bool = False,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
):
    df = build_dataframe(
        prompts,
        output_dir=OUTPUT_DIR,
        modality=modality,
        model=model,
        model_revision=model_revision,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
        steps=steps,
        guidance_scale=guidance_scale,
        flow_shift=flow_shift,
        seed=seed,
        guardrails=guardrails,
        negative_prompt=negative_prompt,
    ).collect()

    model_cache.commit()
    outputs.commit()
    return df.to_pydict()


@app.local_entrypoint()
def modal_main(
    prompt: str = DEFAULT_PROMPT,
    modality: str = "image",
    model: str = DEFAULT_MODEL,
    model_revision: str = "",
    width: int = 1024,
    height: int = 1024,
    num_frames: int = 1,
    fps: int = 24,
    steps: int = 10,
    guidance_scale: float = 7.0,
    flow_shift: float = 10.0,
    seed: int = 42,
    guardrails: bool = False,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    download_only: bool = False,
):
    if download_only:
        print(download_model_weights.remote(model=model, model_revision=model_revision))
        return

    print(
        run_on_modal.remote(
            prompts=[prompt],
            modality=modality,
            model=model,
            model_revision=model_revision,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            steps=steps,
            guidance_scale=guidance_scale,
            flow_shift=flow_shift,
            seed=seed,
            guardrails=guardrails,
            negative_prompt=negative_prompt,
        )
    )
