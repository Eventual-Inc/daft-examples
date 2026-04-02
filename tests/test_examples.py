"""
Smoke tests for every registered script.

Each script is run via `uv run <script>` in a subprocess.
The only assertion: exit code 0.

Scripts are auto-skipped when their required env vars are missing.
Run a specific tier:  pytest -k quickstart
Run no-credential:    pytest -m "not credentials"
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.registry import SCRIPTS, Script

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _make_id(s: Script) -> str:
    return f"{s.tier}::{s.path}".removesuffix(".py")


def _missing_env(s: Script) -> list[str]:
    return [v for v in s.env if not os.environ.get(v)]


def _make_params():
    params = []
    for s in SCRIPTS:
        marks = []
        if s.env:
            marks.append(pytest.mark.credentials)
        missing = _missing_env(s)
        if missing:
            marks.append(pytest.mark.skip(reason=f"missing env: {', '.join(missing)}"))
        if s.skip:
            marks.append(pytest.mark.skip(reason=s.skip))
        params.append(pytest.param(s, id=_make_id(s), marks=marks))
    return params


@pytest.mark.parametrize("script", _make_params())
def test_script_runs(script: Script):
    """Run a script via uv and assert exit 0."""
    path = PROJECT_ROOT / script.path
    assert path.exists(), f"{script.path} not found"

    uv = shutil.which("uv")
    assert uv, "uv not found on PATH"

    result = subprocess.run(
        [uv, "run", str(path)],
        capture_output=True,
        text=True,
        timeout=script.timeout,
        cwd=str(PROJECT_ROOT),
    )

    assert result.returncode == 0, (
        f"{script.path} failed (exit {result.returncode})\n--- stderr ---\n{result.stderr[-2000:]}"
    )
