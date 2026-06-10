"""Model weight resolution: Hugging Face snapshots and YOLO release assets.

Weights are cached under a model-cache directory (a Modal Volume in remote runs)
so repeat runs skip the download.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

HF_TOKEN_ENV_VARS = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
)
DEFAULT_HF_REVISION_DIR = "main"
DEFAULT_YOLO_RELEASE = "v8.4.0"


def normalize_hf_token_env() -> str | None:
    for key in HF_TOKEN_ENV_VARS:
        token = os.environ.get(key)
        if token:
            os.environ.setdefault("HF_TOKEN", token)
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
            return token
    return None


def safe_weight_dir_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "--", value.strip("/")) or "default"


def hf_snapshot_dir(model_cache_dir: str | Path, repo_id: str, revision: str | None = None) -> Path:
    revision_dir = safe_weight_dir_name(revision or DEFAULT_HF_REVISION_DIR)
    return Path(model_cache_dir) / "huggingface" / "repos" / safe_weight_dir_name(repo_id) / revision_dir


def _explicit_path(value: str) -> bool:
    return value.startswith(("/", ".", "~"))


def resolve_hf_model_path(
    repo_or_path: str,
    model_cache_dir: str | Path,
    *,
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    model_path = Path(repo_or_path).expanduser()
    if model_path.exists():
        return model_path
    if _explicit_path(repo_or_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    from huggingface_hub import snapshot_download

    local_dir = hf_snapshot_dir(model_cache_dir, repo_or_path, revision)
    local_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        snapshot_download(
            repo_id=repo_or_path,
            revision=revision or None,
            token=token or normalize_hf_token_env(),
            local_dir=str(local_dir),
        )
    )


def resolve_yolo_weight_path(
    weight: str,
    model_cache_dir: str | Path,
    *,
    release: str = DEFAULT_YOLO_RELEASE,
    repo: str = "ultralytics/assets",
) -> Path:
    weight_path = Path(weight).expanduser()
    if weight_path.exists():
        return weight_path

    target_path = weight_path if _explicit_path(weight) else Path(model_cache_dir) / "ultralytics" / Path(weight).name

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return target_path

    from ultralytics.utils.downloads import attempt_download_asset

    return Path(attempt_download_asset(str(target_path), repo=repo, release=release))
