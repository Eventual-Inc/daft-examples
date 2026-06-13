"""NVIDIA Cosmos 3 generation as a Daft class UDF.

Backend: offline vLLM via vLLM-Omni (`Omni` engine with the
``Cosmos3OmniDiffusersPipeline`` model class). The ``@daft.cls`` instance loads
the engine once per worker, writes each generated image or video to disk, and
returns plain metadata plus the artifact path.

This module never imports ``modal`` — see ``modal_app.py`` for deployment.
"""

from __future__ import annotations

from pathlib import Path

import daft
from daft import DataType, col
from daft.functions import hash as daft_hash
from daft.functions import unnest
from models.common.media import save_video

DEFAULT_MODEL = "nvidia/Cosmos3-Nano"
DEFAULT_NEGATIVE_PROMPT = "blurry, distorted, low quality, jittery, deformed"
DEFAULT_PROMPT = "A photorealistic red sports car at golden hour, cinematic lighting."

Cosmos3Result = DataType.struct(
    {
        "output_path": DataType.string(),
        "model": DataType.string(),
        "model_revision": DataType.string(),
        "model_path": DataType.string(),
        "modality": DataType.string(),
        "prompt": DataType.string(),
        "negative_prompt": DataType.string(),
        "seed": DataType.int64(),
        "width": DataType.int64(),
        "height": DataType.int64(),
        "num_frames": DataType.int64(),
        "fps": DataType.int64(),
        "num_inference_steps": DataType.int64(),
        "guidance_scale": DataType.float64(),
    }
)


def output_path_expr(output_dir: str, modality: str):
    """Deterministic artifact path from the full generation parameter set."""
    suffix = "mp4" if modality == "video" else "png"
    output_id = daft_hash(
        col("prompt"),
        col("negative_prompt"),
        col("seed"),
        col("modality"),
        col("width"),
        col("height"),
        col("num_frames"),
        col("fps"),
        col("steps"),
        col("guidance_scale"),
        col("flow_shift"),
    ).cast(DataType.string())
    return daft.lit(f"{output_dir}/cosmos3-{modality}-") + output_id + daft.lit(f".{suffix}")


@daft.cls(gpus=1.0, max_concurrency=1)
class Cosmos3Omni:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        model_revision: str = "",
        guardrails: bool = False,
    ):
        from huggingface_hub import snapshot_download
        from vllm_omni.entrypoints.omni import Omni

        self.model = model
        self.model_revision = model_revision
        self.model_path = snapshot_download(repo_id=model, revision=model_revision or None)
        self.omni = Omni(
            model=self.model_path,
            model_class_name="Cosmos3OmniDiffusersPipeline",
            trust_remote_code=True,
            enforce_eager=True,
            model_config={"guardrails": guardrails},
        )

    @daft.method(return_dtype=Cosmos3Result)
    def generate(
        self,
        prompt: str,
        output_path: str,
        *,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        seed: int = 42,
        modality: str = "image",
        width: int = 1024,
        height: int = 1024,
        num_frames: int = 1,
        fps: int = 24,
        steps: int = 10,
        guidance_scale: float = 7.0,
        flow_shift: float = 10.0,
    ) -> dict:
        import torch
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        output_file = Path(output_path)
        output_frames = num_frames if modality == "video" else 1
        output_fps = fps if modality == "video" else 0
        if not output_file.exists():
            sampling_params = OmniDiffusionSamplingParams(
                height=height,
                width=width,
                generator=torch.Generator(device="cpu").manual_seed(seed),
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                num_outputs_per_prompt=1,
                num_frames=output_frames,
                fps=output_fps or None,
                extra_args={"flow_shift": flow_shift},
            )
            output = self.omni.generate(
                {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "modalities": [modality],
                },
                sampling_params,
                use_tqdm=False,
            )[0]
            frames = output.request_output.images[0]

            if modality == "video":
                save_video(frames, output_file, fps)
            else:
                frames.save(output_file)

        return {
            "output_path": str(output_path),
            "model": self.model,
            "model_revision": self.model_revision,
            "model_path": self.model_path,
            "modality": modality,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "width": width,
            "height": height,
            "num_frames": output_frames,
            "fps": output_fps,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
        }


def build_dataframe(
    prompts: list[str],
    *,
    output_dir: str,
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
) -> daft.DataFrame:
    """One row per prompt, each generated by the Cosmos3Omni UDF."""
    cosmos = Cosmos3Omni(
        model=model,
        model_revision=model_revision,
        guardrails=guardrails,
    )
    return (
        daft.from_pydict(
            {
                "prompt": prompts,
                "negative_prompt": [negative_prompt] * len(prompts),
                "seed": [seed + index for index in range(len(prompts))],
                "modality": [modality] * len(prompts),
                "width": [width] * len(prompts),
                "height": [height] * len(prompts),
                "num_frames": [num_frames] * len(prompts),
                "fps": [fps] * len(prompts),
                "steps": [steps] * len(prompts),
                "guidance_scale": [guidance_scale] * len(prompts),
                "flow_shift": [flow_shift] * len(prompts),
            }
        )
        .with_column("output_path", output_path_expr(output_dir, modality))
        .with_column(
            "result",
            cosmos.generate(
                col("prompt"),
                col("output_path"),
                negative_prompt=col("negative_prompt"),
                seed=col("seed"),
                modality=col("modality"),
                width=col("width"),
                height=col("height"),
                num_frames=col("num_frames"),
                fps=col("fps"),
                steps=col("steps"),
                guidance_scale=col("guidance_scale"),
                flow_shift=col("flow_shift"),
            ),
        )
        .select(unnest(col("result")))
    )
