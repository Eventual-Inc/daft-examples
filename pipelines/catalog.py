# /// script
# description = "Shared catalog configuration for pipeline table management"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft>=0.7.8", "pyiceberg[sql-sqlite]"]
# ///
"""Shared Iceberg catalog for all pipelines.

Every pipeline writes outputs as named tables in a shared catalog, making
results queryable, cacheable, and portable between local dev and cloud
warehouses (GCS BigLake, S3, etc.).

Configuration (environment variables):
  DAFT_WAREHOUSE    — root path for table data   (default: .data/warehouse)
  DAFT_CATALOG_URI  — catalog metastore URI       (auto-derived from warehouse)

Examples:

  # Local development — no config needed, uses SQLite + filesystem
  uv run pipelines/transcribe_diarize/transcribe_diarize.py --source audio.m4a

  # GCS BigLake
  export DAFT_WAREHOUSE=gs://my-bucket/lakehouse
  export DAFT_CATALOG_URI=sqlite:///path/to/catalog.db
  uv run pipelines/transcribe_diarize/transcribe_diarize.py --source audio.m4a

Each pipeline owns a namespace (e.g. ``transcribe_diarize``) and writes
phase outputs as tables within it (``transcribe_diarize.transcription``,
``transcribe_diarize.diarization``, ``transcribe_diarize.merged``).
"""

import os

from daft.catalog import Catalog

_catalog: Catalog | None = None


def get_catalog() -> Catalog:
    """Return the shared Daft catalog, creating it on first call.

    Uses PyIceberg's SqlCatalog backed by SQLite for local development.
    Override via DAFT_WAREHOUSE / DAFT_CATALOG_URI for cloud warehouses.
    """
    global _catalog
    if _catalog is not None:
        return _catalog

    from pyiceberg.catalog.sql import SqlCatalog

    warehouse = os.environ.get("DAFT_WAREHOUSE", os.path.join(".data", "warehouse"))
    abs_warehouse = os.path.abspath(warehouse)
    catalog_uri = os.environ.get(
        "DAFT_CATALOG_URI",
        f"sqlite:///{abs_warehouse}/catalog.db",
    )

    os.makedirs(warehouse, exist_ok=True)

    warehouse_uri = warehouse if "://" in warehouse else f"file://{abs_warehouse}"

    ice_catalog = SqlCatalog("daft_examples", uri=catalog_uri, warehouse=warehouse_uri)
    _catalog = Catalog.from_iceberg(ice_catalog)
    return _catalog


def ensure_namespace(namespace: str) -> Catalog:
    """Get the catalog and ensure the given namespace exists."""
    catalog = get_catalog()
    catalog.create_namespace_if_not_exists(namespace)
    return catalog
