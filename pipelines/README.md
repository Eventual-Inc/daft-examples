# Pipelines

End-to-end multi-stage pipelines that combine multiple Daft features.
Pipelines that persist intermediate results write to a shared **Iceberg catalog**
so tables are queryable, versioned, and trivially portable between local dev and
cloud warehouses.

## Catalog Architecture

```
pipelines/
├── catalog.py                  # Shared catalog factory (PyIceberg + Daft)
├── README.md                   # ← you are here
│
├── transcribe_diarize/         # Audio → transcription + speaker diarization
│   ├── transcribe_diarize.py
│   ├── diarize_schema.py
│   └── report_template.html
│
├── key_moments_extraction.py   # Transcript → key moments → audio clips
├── voice_ai_analytics/         # Voice analytics with transcription + QA
│
├── context_engineering/        # LLM context patterns (chunking, few-shot, etc.)
├── image_understanding_eval/   # Vision model evaluation on image datasets
├── rag/                        # Retrieval-augmented generation
├── social_recommendation/      # Reddit → image similarity → Unity Catalog
├── code/                       # Code analysis with GitHub integration
│
├── ai_search.py                # PDF → embed → Turbopuffer
├── data_enrichment.py          # Reddit comments → LLM enrichment
├── embed_docs.py               # PDF → embeddings → parquet
├── shot_boundary_detection.py  # Video → frame analysis
└── ...
```

### How It Works

Each pipeline that persists data owns a **namespace** in the catalog with its own tables:

```
transcribe_diarize.transcription   # Faster Whisper output
transcribe_diarize.diarization     # pyannote speaker segments
transcribe_diarize.merged          # Combined result with speaker attribution
```

The catalog is backed by **Apache Iceberg** (via PyIceberg), giving you:
- Versioned tables with schema evolution
- Time travel / snapshot queries
- Seamless migration between local filesystem and cloud object stores

### Configuration

Two environment variables control where data lives:

| Variable | Default | Description |
|----------|---------|-------------|
| `DAFT_WAREHOUSE` | `.data/warehouse` | Root path for table data (local path or `gs://`, `s3://`) |
| `DAFT_CATALOG_URI` | `sqlite:///<warehouse>/catalog.db` | Catalog metastore connection string |

**Local development** needs no configuration — the defaults create a SQLite-backed
Iceberg catalog under `.data/warehouse/` (gitignored).

**Cloud deployment** is a URI change:

```bash
# GCS BigLake
export DAFT_WAREHOUSE=gs://my-bucket/lakehouse
export DAFT_CATALOG_URI=sqlite:///path/to/catalog.db

# AWS S3
export DAFT_WAREHOUSE=s3://my-bucket/lakehouse
export DAFT_CATALOG_URI=sqlite:///path/to/catalog.db

# Or use a managed metastore (Glue, BigQuery, etc.) by configuring
# PyIceberg directly in ~/.pyiceberg.yaml
```

### Using the Catalog in a Pipeline

```python
import sys
from pathlib import Path

# Add pipelines/ to path for the shared catalog module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catalog import ensure_namespace

NAMESPACE = "my_pipeline"

def main():
    catalog = ensure_namespace(NAMESPACE)

    # Write phase results as tables
    catalog.write_table(f"{NAMESPACE}.raw", df_raw, mode="overwrite")

    # Read from previous phases
    df = catalog.read_table(f"{NAMESPACE}.raw")

    # Check if a table exists (for caching)
    if catalog.has_table(f"{NAMESPACE}.processed"):
        df = catalog.read_table(f"{NAMESPACE}.processed")
```

### Querying Tables After a Run

```python
from catalog import get_catalog

catalog = get_catalog()

# List all tables
for ns in catalog.list_namespaces():
    for tbl in catalog.list_tables(ns):
        print(tbl)

# Read any table
df = catalog.read_table("transcribe_diarize.merged")
df.show()
```

---

## Pipeline Reference

### Pipelines with Catalog Integration

| Pipeline | Namespace | Tables | Description |
|----------|-----------|--------|-------------|
| `transcribe_diarize/` | `transcribe_diarize` | `transcription`, `diarization`, `merged` | Audio → transcription + speaker diarization + HTML report |

### Pipelines with External Storage

| Pipeline | Storage | Description |
|----------|---------|-------------|
| `ai_search.py` | Turbopuffer | PDF → embed → vector search |
| `social_recommendation/` | S3 + Delta Lake | Reddit → image similarity → Unity Catalog |
| `image_understanding_eval/` | S3 | Vision model evaluation |
| `voice_ai_analytics/` | `.data/` parquet + Lance | Voice analytics with transcription |
| `key_moments_extraction.py` | `.data/` parquet | Transcript → key moments → audio clips |
| `data_enrichment.py` | Local parquet | Reddit comments → LLM enrichment |
| `embed_docs.py` | `.data/` parquet | PDF → embeddings |

### Pipelines without Persistence (show-only)

| Pipeline | Description |
|----------|-------------|
| `context_engineering/chunking_strategies.py` | Compare chunking approaches for RAG |
| `context_engineering/few_shot_example_selection.py` | Dynamic few-shot example selection |
| `context_engineering/lambda_mapreduce.py` | Map-reduce over documents with LLMs |
| `context_engineering/llm_judge_elo.py` | LLM-as-judge with Elo ratings |
| `rag/rag.py` | Simple RAG pipeline |
| `rag/full_rag.py` | Full RAG with reranking |
| `shot_boundary_detection.py` | Video shot boundary detection |
| `code/cursor.py` | Code analysis with GitHub |

---

## Running Pipelines

```bash
# Most pipelines run with uv (dependencies are declared inline)
uv run pipelines/transcribe_diarize/transcribe_diarize.py --source audio.m4a

# Check what tables exist after a run
uv run -c "from pipelines.catalog import get_catalog; [print(t) for t in get_catalog().list_tables('transcribe_diarize')]"
```

See individual pipeline docstrings for required environment variables and arguments.
