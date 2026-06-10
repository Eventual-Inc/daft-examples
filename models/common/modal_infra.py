"""Shared Modal deployment conventions for model examples.

Every model's ``modal_app.py`` uses the same container paths and Hugging Face
cache layout so weight Volumes are interchangeable across models.
"""

from __future__ import annotations

# Canonical mount points inside Modal containers.
APP_DIR = "/workspace"
MODEL_CACHE_DIR = "/models"
OUTPUT_DIR = "/outputs"
INPUT_DIR = "/inputs"

# Workspace files that never need to ship with `Image.add_local_dir(".")`.
MODAL_LOCAL_DIR_IGNORE = (
    ".context/**",
    ".git/**",
    ".ruff_cache/**",
    ".venv/**",
    ".env",
    ".envrc",
    "**/.DS_Store",
    "**/__pycache__/**",
    "**/*.py[cod]",
)


def hf_cache_env(model_cache_dir: str = MODEL_CACHE_DIR) -> dict[str, str]:
    """Environment that points all Hugging Face caches at the model-cache Volume."""
    return {
        "HF_HOME": f"{model_cache_dir}/huggingface",
        "HF_HUB_CACHE": f"{model_cache_dir}/huggingface/hub",
        "TRANSFORMERS_CACHE": f"{model_cache_dir}/huggingface/hub",
        "HF_XET_HIGH_PERFORMANCE": "1",
    }
