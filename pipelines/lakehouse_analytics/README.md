# Lakehouse Analytics Pipeline

Build a complete analytics lakehouse on BigQuery using Daft + Apache Iceberg + BigLake.

This pipeline demonstrates:
- **Custom `DataSource`** implementations for GitHub and PyPI APIs
- **Iceberg catalog** with BigLake REST catalog (GCS-backed, auto-federated to BigQuery)
- **`daft.read_sql()`** for backfilling SQL-compatible data into the lakehouse
- **Direct Daft table operations** via `read_table()`, `where()`, `select()`, and `sort()`
- **Matplotlib** for visualization

## Architecture

```mermaid
flowchart LR
    github[GitHub API] --> sources[Custom Daft DataSources]
    pypi[PyPI API] --> sources
    sql[SQL-compatible source] --> read_sql[daft.read_sql]
    read_sql --> lakehouse[Iceberg Lakehouse]
    sources --> lakehouse
    lakehouse --> bigquery[BigQuery / BigLake]
    lakehouse --> daft[Daft DataFrames]
    daft --> charts[Matplotlib charts]
```

## Quick Start

```bash
# Run from the repository root

# Local mode (SQLite Iceberg — no GCP needed)
uv run --extra lakehouse -m pipelines.lakehouse_analytics.ingest

# BigLake mode (GCS + BigQuery)
GCP_PROJECT=eventual-analytics uv run --extra lakehouse -m pipelines.lakehouse_analytics.ingest

# Backfill SQL-compatible data into the lakehouse
uv run --extra lakehouse -m pipelines.lakehouse_analytics.backfill \
  --source-url sqlite:///source.db \
  --source-table events \
  --target-table events \
  --key id \
  --target-namespace analytics

# Query and visualize
uv run --extra lakehouse -m pipelines.lakehouse_analytics.analyze
GCP_PROJECT=eventual-analytics uv run --extra lakehouse -m pipelines.lakehouse_analytics.analyze
```

Local mode writes its Iceberg catalog to `./.lakehouse/` at the repository root by default. Override it with `LAKEHOUSE_DIR` if needed. Local lakehouse data is gitignored.

Copy `.env.example` to `.env` for a complete list of supported environment variables:

```bash
cp .env.example .env
```

| Variable | Purpose | Default |
|----------|---------|---------|
| `GCP_PROJECT` | Enables BigLake mode and sets the Google Cloud project. Leave unset for local SQLite mode. | unset |
| `GCS_BUCKET` | GCS warehouse bucket for BigLake mode. | `daft-lakehouse` |
| `LAKEHOUSE_NAMESPACE` | Default Iceberg namespace for ingest and analyze. | `analytics` |
| `LAKEHOUSE_DIR` | Local SQLite Iceberg catalog directory. | `.lakehouse` |
| `BACKFILL_SOURCE_URL` | SQLAlchemy source URL for generic SQL backfill. | unset |
| `BACKFILL_QUERY` | SQL query to read from the source. Mutually exclusive with `BACKFILL_SOURCE_TABLE`. | unset |
| `BACKFILL_SOURCE_TABLE` | Source table for `SELECT *` reads. Mutually exclusive with `BACKFILL_QUERY`. | unset |
| `BACKFILL_TARGET_TABLE` | Target Iceberg table for backfill output. | unset |
| `BACKFILL_KEY` | Comma-separated upsert key columns. | unset |
| `BACKFILL_TARGET_NAMESPACE` | Target Iceberg namespace for backfill. | `analytics` |

## SQL Backfill

`backfill.py` reads from any SQLAlchemy-compatible source with `daft.read_sql()` and upserts into the active Iceberg catalog session.

```bash
# Query-based source
uv run --extra lakehouse -m pipelines.lakehouse_analytics.backfill \
  --source-url bigquery://eventual-analytics \
  --query "SELECT * FROM github_daft_stargazers.stargazers" \
  --target-table github_stargazers \
  --key _dlt_id \
  --target-namespace analytics

# Table-based source
uv run --extra lakehouse -m pipelines.lakehouse_analytics.backfill \
  --source-url sqlite:///source.db \
  --source-table events \
  --target-table events \
  --key id
```

The same values can be supplied through `BACKFILL_SOURCE_URL`, `BACKFILL_QUERY`, `BACKFILL_SOURCE_TABLE`, `BACKFILL_TARGET_TABLE`, `BACKFILL_KEY`, and `BACKFILL_TARGET_NAMESPACE`.

## Setup (BigLake mode)

```bash
# 1. Create GCS bucket
gcloud storage buckets create gs://my-lakehouse --location=us-west1 --uniform-bucket-level-access

# 2. Enable APIs
gcloud services enable biglake.googleapis.com bigquery.googleapis.com bigqueryconnection.googleapis.com

# 3. Create BigLake catalog + namespace
gcloud biglake iceberg catalogs create my-lakehouse --catalog-type=gcs-bucket --credential-mode=end-user
gcloud biglake iceberg namespaces create analytics --catalog=my-lakehouse

# 4. Auth
gcloud auth application-default login

# 5. Run
GCP_PROJECT=my-project GCS_BUCKET=my-lakehouse uv run --extra lakehouse -m pipelines.lakehouse_analytics.ingest
```

## Files

| File | Purpose |
|------|---------|
| `pipelines/catalog.py` | Shared session factory |
| `sources.py` | Custom `DataSource` implementations (GitHub, PyPI) |
| `ingest.py` | Run all sources, upsert into Iceberg tables |
| `backfill.py` | Backfill SQL-compatible data into the lakehouse |
| `analyze.py` | Read Iceberg tables with Daft, visualize with matplotlib |

## Notes

- Run this pipeline as a module with `uv run --extra lakehouse -m ...`
- Shared helpers live under `pipelines/`; no `sys.path` mutation is required
