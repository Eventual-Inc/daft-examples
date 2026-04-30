# /// script
# description = "Ingest GitHub and PyPI data into an Iceberg lakehouse using custom Daft DataSources."
# requires-python = ">=3.12, <3.13"
# dependencies = [
#     "daft[iceberg,sql]>=0.7.8",
#     "pyiceberg[sql-sqlite,gcsfs,bigquery]",
#     "python-dotenv",
# ]
# ///
"""Ingest pipeline — fetch from APIs via custom DataSources, write to Iceberg.

Runs all sources and upserts into the lakehouse. Works in local mode (SQLite)
or BigLake mode (GCS + BigQuery) depending on GCP_PROJECT env var.

Usage:
    uv run --extra lakehouse -m pipelines.lakehouse_analytics.ingest
    GCP_PROJECT=eventual-analytics uv run --extra lakehouse -m pipelines.lakehouse_analytics.ingest
"""

import daft
from pipelines.catalog import get_session
from pipelines.lakehouse_analytics.sources import (
    GitHubDataSource,
    PyPIDataSource,
)


def write_upsert(sess, table: str, new_df: daft.DataFrame, on: str | list[str]) -> None:
    keys = [on] if isinstance(on, str) else on
    try:
        existing = sess.read_table(table)
    except Exception:
        sess.create_table_if_not_exists(table, new_df)
        return

    kept = existing.join(new_df.select(*keys), on=keys, how="anti")
    sess.write_table(table, kept.concat(new_df), mode="overwrite")


def main():
    sess = get_session()

    write_upsert(
        sess,
        "github_daily",
        GitHubDataSource(repo="Eventual-Inc/Daft").read(),
        on="date",
    )
    print("github_daily")

    write_upsert(
        sess,
        "pypi_downloads",
        PyPIDataSource(package="daft").read(),
        on="date",
    )
    print("pypi_downloads")


if __name__ == "__main__":
    main()
