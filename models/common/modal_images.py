"""The two container lanes for speech models.

NeMo and faster-whisper (CTranslate2) cannot share an image: NeMo pins
``numpy<2`` and a specific torch/cuDNN stack, while CT2 needs cuDNN 9 with no
torch. So the transcribe+diarize swap matrix runs in two lanes:

- ``nemo_image``  — Parakeet/Canary ASR + Sortformer/MarbleNet (all NeMo).
- ``whisper_image`` — faster-whisper ASR + pyannote diarization + Silero VAD
                      (the WhisperX-family combo, which *does* coexist).

Both build on a CUDA+cuDNN base so CT2's cuDNN 9 dependency and NeMo's torch
CUDA libs resolve at runtime.
"""

from __future__ import annotations

import modal

from models.common.modal_infra import OUTPUT_DIR, hf_cache_env, nemo_cache_env

GPU_TYPE = "A100-40GB"
MODAL_REGION = ["us-west"]
CUDA_BASE = "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04"


def nemo_image(python_version: str = "3.11") -> modal.Image:
    return (
        modal.Image.from_registry(CUDA_BASE, add_python=python_version)
        .apt_install("libsndfile1", "ffmpeg", "git")
        # Cython + packaging must precede nemo or its build fails.
        .pip_install("Cython", "packaging", "numpy<2")
        .pip_install(
            "daft>=0.7.10",
            "nemo_toolkit[asr]",
            "soundfile",
            "huggingface_hub",
            "hf_xet",
        )
        .env(
            {
                **nemo_cache_env(),
                "OUTPUT_DIR": OUTPUT_DIR,
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            }
        )
        .add_local_python_source("models", "pipelines")
    )


def whisper_image(python_version: str = "3.12") -> modal.Image:
    return (
        modal.Image.from_registry(CUDA_BASE, add_python=python_version)
        .apt_install("libsndfile1", "ffmpeg")
        .pip_install(
            "daft>=0.7.10",
            "faster-whisper",
            "nvidia-cublas-cu12",
            "nvidia-cudnn-cu12==9.*",
            "pyannote.audio",
            "silero-vad",
            "soundfile",
            "librosa",  # Daft AudioFile.resample() needs it; NeMo pulls it in, this lane doesn't
            "huggingface_hub",
            "hf_xet",
            "jiwer",
        )
        .env(
            {
                **hf_cache_env(),
                "PYANNOTE_CACHE": "/models/pyannote",
                "OUTPUT_DIR": OUTPUT_DIR,
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                # Surface the pip-installed cuDNN 9 / cuBLAS to CTranslate2's loader.
                "LD_LIBRARY_PATH": (
                    f"/usr/local/lib/python{python_version}/site-packages/nvidia/cudnn/lib:"
                    f"/usr/local/lib/python{python_version}/site-packages/nvidia/cublas/lib:"
                    "${LD_LIBRARY_PATH}"
                ),
            }
        )
        .add_local_python_source("models", "pipelines")
    )
