"""Canonical model registry for Modal.

One shared weight Volume, one Hugging Face cache layout, and one set of
cold-start optimizations (Xet fast downloads + memory snapshots) defined in a
single place so every model package stays interchangeable.

Two layers live here:

- **Registry** — the shared ``MODEL_CACHE`` Volume, the ``hf-token`` Secret, and
  the canonical Hugging Face cache env. Replaces the per-model
  ``modal.Volume.from_name("<model>-model-cache")`` declarations.
- **Template** — ``base_image`` / ``with_model_cache`` build the container, and
  ``function_kwargs`` stamps out the repeated ``@app.function(...)`` config
  (volumes, secret, region, memory snapshot) so each model only declares what is
  actually different (its pip layer, GPU type, memory).

Weight *resolution* (HF snapshots, YOLO assets) stays in
``models.common.weights`` — pure, modal-free, lazily imported — and is
re-exported here so model packages have a single import surface::

    from models.weights import MODEL_CACHE, base_image, function_kwargs, resolve_hf_model_path
"""

from __future__ import annotations

import modal

from models.common.modal_infra import (
    MODEL_CACHE_DIR,
    OUTPUT_DIR,
    hf_cache_env,
)
from models.common.weights import (
    hf_snapshot_dir,
    normalize_hf_token_env,
    resolve_hf_file_path,
    resolve_hf_model_path,
    resolve_yolo_weight_path,
)

__all__ = [
    "MODEL_CACHE",
    "OUTPUTS",
    "VOLUMES",
    "VOLUMES_WITH_OUTPUTS",
    "HF_SECRET",
    "DEFAULT_REGION",
    "cache_env",
    "base_image",
    "with_model_cache",
    "function_kwargs",
    "cls_kwargs",
    # re-exported resolution helpers
    "hf_snapshot_dir",
    "normalize_hf_token_env",
    "resolve_hf_file_path",
    "resolve_hf_model_path",
    "resolve_yolo_weight_path",
]

# --- Registry: one canonical Volume for every model's weights --------------
# A single shared cache dedupes base models across packages and keeps the layout
# interchangeable (see ``hf_snapshot_dir`` for the on-volume path scheme). Per-model
# isolation is traded for cross-model dedup and one place to warm/commit.
MODEL_CACHE = modal.Volume.from_name("daft-model-cache", create_if_missing=True)
OUTPUTS = modal.Volume.from_name("daft-model-outputs", create_if_missing=True)

VOLUMES = {MODEL_CACHE_DIR: MODEL_CACHE}
VOLUMES_WITH_OUTPUTS = {MODEL_CACHE_DIR: MODEL_CACHE, OUTPUT_DIR: OUTPUTS}

HF_SECRET = modal.Secret.from_name("hf-token")

DEFAULT_REGION = ["us-west"]


def cache_env(model_cache_dir: str = MODEL_CACHE_DIR) -> dict[str, str]:
    """Hugging Face cache env pointed at the shared Volume, Xet acceleration on.

    ``hf_cache_env`` already sets ``HF_XET_HIGH_PERFORMANCE=1``; the Xet backend
    only engages when ``hf_xet`` is installed in the image (see ``base_image`` /
    ``with_model_cache``). Note this is mutually exclusive with the legacy
    ``HF_HUB_ENABLE_HF_TRANSFER`` accelerator — do not set both.
    """
    return hf_cache_env(model_cache_dir)


def with_model_cache(
    image: modal.Image,
    *,
    model_cache_dir: str = MODEL_CACHE_DIR,
    extra_env: dict[str, str] | None = None,
    local_python_source: tuple[str, ...] = ("models",),
) -> modal.Image:
    """Apply the canonical cache layer to any base image.

    Installs ``hf_xet`` (so ``HF_XET_HIGH_PERFORMANCE`` actually engages), sets
    the shared HF cache env, and mounts local Python sources. Use this to wrap a
    CUDA / nightly base that can't use ``base_image`` directly.
    """
    env = {**cache_env(model_cache_dir)}
    if extra_env:
        env.update(extra_env)
    return image.pip_install("hf_xet").env(env).add_local_python_source(*local_python_source)


def base_image(
    python_version: str = "3.12",
    *,
    apt: tuple[str, ...] = (),
    pip: tuple[str, ...] = (),
    extra_env: dict[str, str] | None = None,
    local_python_source: tuple[str, ...] = ("models",),
) -> modal.Image:
    """Debian-slim base with the canonical cache layer pre-applied.

    Convenience for the common case (cosmos3, diffusion_gemma). ``huggingface_hub``
    and ``hf_xet`` are always installed; pass model-specific deps via ``pip``.
    CUDA-base models should build their own ``from_registry`` image and wrap it
    with ``with_model_cache`` instead.
    """
    img = modal.Image.debian_slim(python_version=python_version)
    if apt:
        img = img.apt_install(*apt)
    img = img.pip_install("huggingface_hub", "hf_xet", *pip)
    env = {**cache_env()}
    if extra_env:
        env.update(extra_env)
    return img.env(env).add_local_python_source(*local_python_source)


def function_kwargs(
    image: modal.Image,
    *,
    gpu: str | None = None,
    cpu: float = 8,
    memory: int = 16384,
    timeout: int = 7200,
    region: list[str] | None = None,
    with_outputs: bool = False,
    enable_memory_snapshot: bool = True,
    enable_gpu_snapshot: bool = False,
    secrets: list | None = None,
    **overrides,
) -> dict:
    """Standard ``@app.function(...)`` config for a model entrypoint.

    Spread into the decorator: ``@app.function(**function_kwargs(image, gpu=GPU_TYPE))``.

    CPU memory snapshots are on by default (stable): they checkpoint the container
    after imports / JIT compilation, which is the reliable cold-start win. Pass
    ``enable_memory_snapshot=False`` for pure download/CPU functions.

    GPU memory snapshots (``enable_gpu_snapshot=True``) are **opt-in** because the
    feature is alpha and only helps when the model is loaded into VRAM *during* the
    snapshot window — i.e. inside a ``modal.Cls`` ``@modal.enter(snap=True)``. Our
    UDFs currently load lazily on the worker (after the checkpoint), so flipping
    this on a plain function snapshots an empty GPU. Wire the warm-load into the
    snapshot window first, then enable it per function.
    """
    kwargs: dict = {
        "image": image,
        "cpu": cpu,
        "memory": memory,
        "timeout": timeout,
        "region": region if region is not None else DEFAULT_REGION,
        "volumes": VOLUMES_WITH_OUTPUTS if with_outputs else VOLUMES,
        "secrets": secrets if secrets is not None else [HF_SECRET],
        "enable_memory_snapshot": enable_memory_snapshot,
    }
    if gpu is not None:
        kwargs["gpu"] = gpu

    # GPU snapshots are alpha and only help when weights load during the snapshot
    # window — opt-in only, never auto-enabled (see docstring).
    if enable_gpu_snapshot:
        kwargs["experimental_options"] = {"enable_gpu_snapshot": True}

    kwargs.update(overrides)
    return kwargs


def cls_kwargs(
    image: modal.Image,
    *,
    gpu: str | None = None,
    cpu: float = 8,
    memory: int = 16384,
    timeout: int = 7200,
    region: list[str] | None = None,
    with_outputs: bool = False,
    enable_gpu_snapshot: bool = True,
    secrets: list | None = None,
    **overrides,
) -> dict:
    """Config for a snapshot-enabled ``@app.cls`` model service.

    Unlike ``function_kwargs``, this is for the ``modal.Cls`` form that actually
    has a snapshot window: load the model in ``@modal.enter(snap=True)`` so it is
    captured into the GPU snapshot. ``enable_memory_snapshot`` is always on; GPU
    snapshot defaults on here because a Cls is the only place it can pay off.

    Critical constraint for this repo: our inference runs through Daft class UDFs
    (``@daft.cls``), and Daft instantiates a GPU UDF in its own worker process —
    separate from this container's main process, and spun up *after* the snapshot
    window. So a snapshot taken in ``@modal.enter`` will NOT capture a model that a
    Daft UDF loads in its body. To benefit, the Cls must load the heavy engine in
    ``@modal.enter(snap=True)`` into a process-global the UDF reuses, AND the UDF
    must run in-process (``use_process=False``). vLLM / vLLM-Omni add a second
    multiprocess engine layer that needs explicit release/resume of GPU memory to
    be snapshot-compatible (see Modal's SGLang snapshot example) — treat those as
    advanced. See ``models/cosmos3/snapshot_service.py`` for a worked reference.
    """
    kwargs: dict = {
        "image": image,
        "cpu": cpu,
        "memory": memory,
        "timeout": timeout,
        "region": region if region is not None else DEFAULT_REGION,
        "volumes": VOLUMES_WITH_OUTPUTS if with_outputs else VOLUMES,
        "secrets": secrets if secrets is not None else [HF_SECRET],
        "enable_memory_snapshot": True,
    }
    if gpu is not None:
        kwargs["gpu"] = gpu
    if enable_gpu_snapshot:
        kwargs["experimental_options"] = {"enable_gpu_snapshot": True}
    kwargs.update(overrides)
    return kwargs
