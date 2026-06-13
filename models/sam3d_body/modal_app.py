"""Modal deployment shell for the SAM 3D Body model UDF.

The model wrapper itself lives in ``model.py``; this file only owns the
container image, Volumes, and entrypoints.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

from models.common.modal_infra import APP_DIR, MODAL_LOCAL_DIR_IGNORE, MODEL_CACHE_DIR, OUTPUT_DIR, hf_cache_env
from models.common.weights import normalize_hf_token_env, resolve_hf_model_path
from models.sam3d_body.model import DEFAULT_MODEL, build_dataframe
from models.weights import MODEL_CACHE, OUTPUTS, function_kwargs

SAM3D_REPO_DIR = "/sam-3d-body"
GPU_TYPE = "A100-80GB"
MODAL_REGION = ["us-west"]

app = modal.App("daft-sam3d-body")

# Base image without local Python sources so downstream apps (e.g.
# pipelines/pose_sequence) can append pip layers before mounting sources.
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "build-essential",
        "ffmpeg",
        "git",
        "libegl1",
        "libgl1",
        "libglib2.0-0",
        "libgles2",
        "libosmesa6",
        "libsm6",
        "libxext6",
    )
    .run_commands(f"git clone --depth=1 https://github.com/facebookresearch/sam-3d-body.git {SAM3D_REPO_DIR}")
    .pip_install(
        "appdirs",
        "braceexpand",
        "cython",
        "daft>=0.7.10",
        "dill",
        "einops",
        "fvcore",
        "huggingface_hub",
        "hf_xet",
        "hydra-colorlog",
        "hydra-core",
        "hydra-submitit-launcher",
        "jsonlines",
        "loguru",
        "networkx==3.2.1",
        "opencv-python-headless",
        "optree",
        "pandas",
        "pycocotools",
        "pyrootutils",
        "pyrender",
        "pytorch-lightning",
        "rich",
        "roma",
        "scikit-image",
        "seaborn",
        "tensorboard",
        "timm",
        "torch",
        "torchvision",
        "trimesh",
        "wandb",
        "webdataset",
        "xtcocotools",
        "yacs",
    )
    .add_local_dir(".", remote_path=APP_DIR, copy=True, ignore=MODAL_LOCAL_DIR_IGNORE)
    .env(
        {
            **hf_cache_env(MODEL_CACHE_DIR),
            "TORCH_HOME": f"{MODEL_CACHE_DIR}/torch",
            "PYOPENGL_PLATFORM": "egl",
            "SAM3D_BODY_REPO": SAM3D_REPO_DIR,
            "SAM3D_BODY_OUTPUT_DIR": OUTPUT_DIR,
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        }
    )
)
image = base_image.add_local_python_source("models")


@app.function(**function_kwargs(image, cpu=4, enable_memory_snapshot=False))
def download_model_weights(model: str = DEFAULT_MODEL, model_revision: str = "") -> dict:
    os.chdir(APP_DIR)
    model_path = resolve_hf_model_path(
        model,
        MODEL_CACHE_DIR,
        revision=model_revision or None,
        token=normalize_hf_token_env(),
    )
    MODEL_CACHE.commit()
    return {
        "model": model,
        "model_revision": model_revision,
        "model_path": str(model_path),
    }


@app.function(**function_kwargs(image, gpu=GPU_TYPE, memory=98304, with_outputs=True))
def run_on_modal(
    image_paths: list[str],
    bbox_jsons: list[str] | None = None,
    model: str = DEFAULT_MODEL,
    model_revision: str = "",
    checkpoint_path: str = "",
    mhr_path: str = "",
    detector_name: str = "",
    fov_name: str = "",
    bbox_thr: float = 0.8,
    use_mask: bool = False,
    inference_type: str = "full",
):
    os.chdir(APP_DIR)

    df = build_dataframe(
        image_paths,
        bbox_jsons=bbox_jsons,
        model=model,
        model_revision=model_revision,
        output_dir=OUTPUT_DIR,
        repo_path=SAM3D_REPO_DIR,
        checkpoint_path=checkpoint_path,
        mhr_path=mhr_path,
        detector_name=detector_name,
        fov_name=fov_name,
        bbox_thr=bbox_thr,
        use_mask=use_mask,
        inference_type=inference_type,
    ).collect()

    MODEL_CACHE.commit()
    OUTPUTS.commit()
    return df.to_pydict()


@app.local_entrypoint()
def modal_main(
    image_path: str = f"{SAM3D_REPO_DIR}/notebook/images/dancing.jpg",
    bbox: str = "",
    model: str = DEFAULT_MODEL,
    model_revision: str = "",
    checkpoint_path: str = "",
    mhr_path: str = "",
    detector_name: str = "",
    fov_name: str = "",
    bbox_thr: float = 0.8,
    use_mask: bool = False,
    inference_type: str = "full",
    download_only: bool = False,
):
    if download_only:
        print(download_model_weights.remote(model=model, model_revision=model_revision))
        return

    remote_image_path = image_path
    local_image_path = Path(image_path)
    if local_image_path.exists():
        try:
            remote_image_path = f"{APP_DIR}/{local_image_path.resolve().relative_to(Path.cwd().resolve())}"
        except ValueError:
            pass

    bbox_jsons = [bbox] if bbox else None
    print(
        run_on_modal.remote(
            image_paths=[remote_image_path],
            bbox_jsons=bbox_jsons,
            model=model,
            model_revision=model_revision,
            checkpoint_path=checkpoint_path,
            mhr_path=mhr_path,
            detector_name=detector_name,
            fov_name=fov_name,
            bbox_thr=bbox_thr,
            use_mask=use_mask,
            inference_type=inference_type,
        )
    )
