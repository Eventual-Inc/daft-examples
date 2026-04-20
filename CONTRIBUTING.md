# Contributing to daft-examples

## Repository Structure

```
daft-examples/
├── quickstart/          # 5 examples, <30s each, minimal dependencies
├── examples/            # Atomic feature demonstrations
│   ├── classify/        # Image and text classification
│   ├── commoncrawl/     # Common Crawl data processing
│   ├── embed/           # Embeddings and similarity search
│   ├── files/           # daft.File abstraction (audio, video, PDF, code)
│   ├── io/              # File I/O operations
│   ├── prompt/          # LLM prompting patterns
│   ├── sql/             # SQL and window functions
│   └── udfs/            # User-defined functions
├── pipelines/           # End-to-end multi-stage pipelines
│   ├── context_engineering/
│   ├── image_understanding_eval/
│   ├── rag/
│   ├── social_recommendation/
│   ├── transcribe_diarize/
│   └── voice_ai_analytics/
├── datasets/            # Dataset-specific processing (Common Crawl, LAION, etc.)
├── models/              # Model integration helpers
└── tests/               # Test infrastructure
```

### Where does my example go?

| Type | Directory | Description |
|------|-----------|-------------|
| **Quickstart** | `quickstart/` | <30s, minimal deps, teaches one core concept |
| **Example** | `examples/<category>/` | Single feature demo (one UDF, one function, one pattern) |
| **Pipeline** | `pipelines/` | Multi-stage, combines multiple Daft features end-to-end |
| **Dataset** | `datasets/<source>/` | Processing patterns for a specific public dataset |

---

## Script Format

Every Python file must follow this format. Use `TEMPLATE/main.py` as a starting point.

### PEP 723 Header (required)

```python
# /// script
# description = "Short description of what this script demonstrates"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft[openai]>=0.7.5", "python-dotenv"]
# ///
```

- **`description`**: One line, under 100 characters. Describe what the script does, not what Daft is.
- **`requires-python`**: Always `">=3.10, <3.13"`.
- **`dependencies`**: Always pin daft as `daft[extras]>=0.7.5`. See [Daft Extras](#daft-extras) below.

### File Structure

```python
# /// script
# description = "..."
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft[openai]>=0.7.5", "python-dotenv"]
# ///

import os
import daft
from daft import col, DataType
from daft.functions import prompt, embed_text, file
from dotenv import load_dotenv


# UDFs and class definitions go here (above __main__)

@daft.func(return_dtype=DataType.string())
def my_udf(text: str) -> str:
    """One-line docstring."""
    return text.upper()


if __name__ == "__main__":

    load_dotenv()

    # Constants
    SOURCE_URI = "hf://datasets/Eventual-Inc/sample-files/papers/*.pdf"
    DEST_URI = ".data/my_example"

    # ==============================================================================
    # Section header (for multi-stage pipelines)
    # ==============================================================================

    df = (
        daft.from_glob_path(SOURCE_URI)
        .with_column("result", my_udf(col("path")))
    )

    print("\n=== Results ===")
    df.show(5)
```

### Rules

1. **Always use `if __name__ == "__main__":`** for executable code. UDF/class definitions go above the guard.
2. **Constants are UPPER_CASE**: `SOURCE_URI`, `DEST_URI`, `MODEL_ID`, `MAX_DOCS`.
3. **Imports from `daft.functions`**, not `daft.functions.ai`. The latter is an internal path.
4. **Use `col()` from `from daft import col`**, not `daft.col()`.
5. **No hardcoded local paths**. Use HuggingFace (`hf://`), S3 (`s3://`), or `.data/` for output.
6. **Use `.data/` for local output**. This directory is gitignored.
7. **Use `load_dotenv()`** when the script needs API keys. Never hardcode keys.
8. **Use `daft.set_provider()`** for API configuration, not manual provider class instantiation.

---

## Daft Extras

Use the appropriate extras in your dependency spec:

| Extra | When to use |
|-------|-------------|
| `daft[openai]` | `prompt()`, `embed_text()` with OpenAI |
| `daft[aws]` | S3 paths (`s3://`) |
| `daft[audio]` | `audio_file()`, `audio_metadata()`, `resample()` |
| `daft[video]` | `video_file()`, `video_metadata()`, `video_keyframes()` |
| `daft[transformers]` | HuggingFace model inference |
| `daft[huggingface]` | HuggingFace dataset access |
| `daft[turbopuffer]` | `write_turbopuffer()` |
| `daft[unity, deltalake]` | Databricks Unity Catalog |
| `daft[pandas]` | Pandas interop |

Combine as needed: `daft[aws, openai]>=0.7.5`. If the script only uses core Daft (DataFrames, UDFs, no providers), use `daft>=0.7.8`.

When a script uses daft's built-in provider (e.g., `prompt()` with `provider="openai"`), use the daft extra instead of listing `openai` as a separate dependency. Only list `openai` separately if you import from the OpenAI SDK directly (`from openai import ...`).

---

## Data Sources

Prefer public, no-auth sources for examples:

1. **HuggingFace datasets** (`hf://datasets/Eventual-Inc/sample-files/...`) — audio, video, PDFs, papers
2. **Inline data** (`daft.from_pydict({...})`) — for small demos
3. **S3 public buckets** (`s3://daft-public-data/...`) — large datasets (requires AWS extra)

Never commit data files. Never reference local paths (`/Users/...`, `~/...`, `Desktop/...`).

---

## Test Registry

Every script must be registered in `tests/registry.py`:

```python
Script("examples/my_category/my_example.py", env=["OPENAI_API_KEY"], tier="example"),
```

- **`path`**: Relative to repo root.
- **`env`**: List of required environment variables. Script is auto-skipped when any are missing.
- **`timeout`**: Max seconds (default 120). Use 300 for heavy pipelines.
- **`tier`**: `"quickstart"` | `"example"` | `"pipeline"` | `"dataset"`.
- **`skip`**: Optional reason string to unconditionally skip (for WIP scripts only).

### Running Tests

```bash
# Collect all tests (dry run)
pytest tests/test_examples.py --collect-only

# Run tests that don't need API keys
pytest tests/test_examples.py -m "not credentials"

# Run a specific tier
pytest tests/test_examples.py -k quickstart

# Run a single script
pytest tests/test_examples.py -k "read_video_files"
```

---

## Before Submitting a PR

1. **Script runs**: `uv run your_script.py` exits 0
2. **Syntax clean**: `python -c "import ast; ast.parse(open('your_script.py').read())"`
3. **No local paths**: `grep -r '/Users\|~/\|Desktop\|/home/' your_script.py` returns nothing
4. **Header complete**: Has `description`, `requires-python`, `dependencies`
5. **Registered in tests**: Added entry to `tests/registry.py`
6. **Runs in <2 minutes**: Or set `timeout=300` in the registry entry

---

## Style Reference

See `pipelines/context_engineering/chunking_strategies.py` for the canonical example of the expected style.
