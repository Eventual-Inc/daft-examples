# /// script
# description = "Iceberg lakehouse session factory — BigLake REST, AWS Glue, or local SQLite."
# requires-python = ">=3.12, <3.13"
# dependencies = [
#     "daft[iceberg,sql]>=0.7.10",
#     "pyiceberg[sql-sqlite,gcsfs,bigquery,s3fs,glue]",
#     "python-dotenv",
# ]
# ///
"""Shared Iceberg catalog helpers for pipeline scripts.

Three catalog backends, selected by environment variables:

    SQLite (default)    — no env vars needed, writes to .lakehouse/
    GCS + BigLake       — set GCP_PROJECT
    AWS S3 + Glue       — set AWS_S3_BUCKET

The S3 backend pairs naturally with AWS S3 Files: mount the same bucket
at /mnt/s3files/ for POSIX access, while Daft reads via the Glue catalog
with partition pruning and filter pushdown.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from daft import Catalog
from daft.session import Session

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
LAKEHOUSE_DIR = Path(os.environ.get("LAKEHOUSE_DIR") or REPO_ROOT / ".lakehouse")
GCP_PROJECT = os.environ.get("GCP_PROJECT") or None
GCS_BUCKET = os.environ.get("GCS_BUCKET") or "daft-lakehouse"
AWS_S3_BUCKET = os.environ.get("AWS_S3_BUCKET") or None
AWS_REGION = os.environ.get("AWS_REGION") or "us-east-1"
NAMESPACE = os.environ.get("LAKEHOUSE_NAMESPACE") or "analytics"


def get_session(namespace: str | None = None) -> Session:
    """Return a Session with an Iceberg catalog attached."""
    namespace = namespace or NAMESPACE
    if GCP_PROJECT:
        from pyiceberg.catalog import load_catalog

        iceberg_catalog = load_catalog(
            "lakehouse",
            **{
                "type": "rest",
                "uri": "https://biglake.googleapis.com/iceberg/v1/restcatalog",
                "warehouse": f"gs://{GCS_BUCKET}",
                "auth": {"type": "google"},
                "header.x-goog-user-project": GCP_PROJECT,
                "header.X-Iceberg-Access-Delegation": "",
            },
        )
    elif AWS_S3_BUCKET:
        from pyiceberg.catalog import load_catalog

        iceberg_catalog = load_catalog(
            "lakehouse",
            type="glue",
            warehouse=f"s3://{AWS_S3_BUCKET}",
            region_name=AWS_REGION,
        )
    else:
        from pyiceberg.catalog.sql import SqlCatalog

        LAKEHOUSE_DIR.mkdir(parents=True, exist_ok=True)
        iceberg_catalog = SqlCatalog(
            "lakehouse",
            uri=f"sqlite:///{LAKEHOUSE_DIR}/catalog.db",
            warehouse=f"file://{LAKEHOUSE_DIR}",
        )

    catalog = Catalog.from_iceberg(iceberg_catalog)
    sess = Session()
    sess.attach(catalog)
    sess.create_namespace_if_not_exists(namespace)
    sess.use(f"lakehouse.{namespace}")
    return sess
