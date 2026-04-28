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

from pipelines.catalog import get_session, upsert
from pipelines.lakehouse_analytics.sources import GitHubDataSource, PyPIDataSource


def main():
    sess = get_session()

    github_daily = upsert(
        sess,
        "github_daily",
        GitHubDataSource(repo="Eventual-Inc/Daft").read(),
        on="date",
    ).collect()
    github_daily_row = github_daily.to_pydict()
    print(
        "github_daily",
        github_daily_row["stars"][0],
        github_daily_row["forks"][0],
        github_daily_row["open_prs"][0],
    )

    pypi_downloads = upsert(
        sess,
        "pypi_downloads",
        PyPIDataSource(package="daft").read(),
        on="date",
    ).collect()
    print("pypi_downloads", pypi_downloads.count_rows())


if __name__ == "__main__":
    main()
