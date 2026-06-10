# Models

Canonical Daft usage patterns for individual models. Each directory wraps one model
as a `@daft.cls` UDF so pipelines can compose it like any other Daft expression.
End-to-end pipelines that chain models together live in [`pipelines/`](../pipelines/).

## Layout contract

Every model directory follows the same structure:

| File | Responsibility |
| --- | --- |
| `model.py` | The `@daft.cls` UDF, its result-schema struct, and pure helpers. Importable anywhere; **never imports `modal`**. |
| `schema.py` | (optional) Result `DataType` structs when pipelines need them without the model's deps. |
| `modal_app.py` | Modal deployment shell only: container image, Volumes, `download_model_weights` / `run_on_modal` functions, and the `local_entrypoint`. |
| `README.md` | Backend choice, run commands, weight-loading notes, output shape. |

Shared infrastructure lives in [`common/`](./common/):

- `weights.py` — Hugging Face snapshot + YOLO release-asset resolution with a stable cache layout
- `modal_infra.py` — canonical container paths (`/models`, `/outputs`, …), HF cache env, local-dir ignore list
- `media.py` — artifact helpers (e.g. raw-frame → MP4 via ffmpeg)

## Inference backend selection

Prefer backends in this order, and state the choice in the model's README:

1. **Offline vLLM** — in-process engine inside the UDF (e.g. Cosmos 3 via vLLM-Omni)
2. **Online OpenAI-compatible API** — `prompt()` / OpenAI client against a served endpoint
3. **PyTorch** — only when vLLM doesn't support the model (e.g. SAM 3D Body, Faster Whisper)

## Current models

| Model | Backend | Task |
| --- | --- | --- |
| [`cosmos3/`](./cosmos3/) | offline vLLM (vLLM-Omni) | text → image / video world generation |
| [`sam3d_body/`](./sam3d_body/) | PyTorch | image → 3D human mesh recovery |
| [`faster_whisper/`](./faster_whisper/) | CTranslate2 (PyTorch-family) | audio → transcript + VAD |

## Import conventions

`models` is a plain package anchored at the repo root. Entrypoint scripts insert the
repo root on `sys.path` (a deterministic two-liner) so all of these work without
installing anything:

- `uv run --extra models modal run models/<name>/modal_app.py ...`
- `uv run models/faster_whisper/model.py` (PEP 723 isolated env)
- `from models.<name>.model import ...` from pipelines and notebooks

Modal images mount the package with `add_local_python_source("models")`, so the same
imports resolve inside remote containers.
