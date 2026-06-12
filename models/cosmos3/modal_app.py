"""Modal deployment shell for the Cosmos 3 model UDF.

The model wrapper itself lives in ``model.py``; this file only owns the
container image, Volumes, and entrypoints.
"""

from __future__ import annotations

import modal

from models.common.modal_infra import OUTPUT_DIR
from models.cosmos3.model import DEFAULT_MODEL, DEFAULT_NEGATIVE_PROMPT, DEFAULT_PROMPT, build_dataframe
from models.weights import MODEL_CACHE, OUTPUTS, base_image, function_kwargs

GPU_TYPE = "A100-80GB"

app = modal.App("daft-cosmos3-omni")

image = base_image(
    apt=("ffmpeg", "git", "libgl1", "libglib2.0-0", "libxcb1"),
    pip=(
        "av",
        "daft>=0.7.14",
        "vllm==0.22.0",
        "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git",
    ),
    extra_env={
        "COSMOS3_OUTPUT_DIR": OUTPUT_DIR,
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    },
)


@app.function(**function_kwargs(image, cpu=4, enable_memory_snapshot=False))
def download_model_weights(model: str = DEFAULT_MODEL, model_revision: str = "") -> dict:
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=model, revision=model_revision or None)
    MODEL_CACHE.commit()
    return {"model": model, "model_revision": model_revision, "model_path": path}


@app.function(**function_kwargs(image, gpu=GPU_TYPE, memory=98304, with_outputs=True))
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

    MODEL_CACHE.commit()
    OUTPUTS.commit()
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
