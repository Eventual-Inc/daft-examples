# /// script
# description = "Custom Daft DataSource implementations for GitHub and PyPI APIs."
# requires-python = ">=3.12, <3.13"
# dependencies = [
#     "daft[iceberg,sql]>=0.7.8",
#     "python-dotenv",
# ]
# ///
"""Custom DataSource implementations for analytics ingestion.

Each source implements the Daft DataSource interface, producing parallelizable
tasks that fetch data from APIs and yield MicroPartitions.

Usage:
    from sources import GitHubDataSource, PyPIDataSource

    github = GitHubDataSource(repo="Eventual-Inc/Daft")
    github.read().show()

    pypi = PyPIDataSource(package="daft")
    pypi.read().show()
"""

import json
import subprocess
import urllib.request
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime

from daft.datatype import DataType
from daft.io import DataSource, DataSourceTask
from daft.recordbatch import MicroPartition
from daft.schema import Schema


class GitHubDataSource(DataSource):
    """Fetch daily repo metrics from the GitHub REST API via `gh` CLI.

    Produces a single-row snapshot: stars, forks, open PRs, CI status,
    latest release, contributor count, and weekly commits.
    """

    def __init__(self, repo: str = "Eventual-Inc/Daft"):
        self.repo = repo

    @property
    def name(self) -> str:
        return "GitHub Repo Stats"

    @property
    def schema(self) -> Schema:
        return Schema._from_field_name_and_types(
            [
                ("date", DataType.string()),
                ("stars", DataType.int64()),
                ("stars_delta", DataType.int64()),
                ("forks", DataType.int64()),
                ("open_issues", DataType.int64()),
                ("open_prs", DataType.int64()),
                ("ci_status", DataType.string()),
                ("latest_release", DataType.string()),
                ("release_date", DataType.string()),
                ("contributors", DataType.int64()),
                ("commits_7d", DataType.int64()),
            ]
        )

    async def get_tasks(self, pushdowns) -> AsyncIterator["GitHubDataSourceTask"]:
        yield GitHubDataSourceTask(self.repo)


class GitHubDataSourceTask(DataSourceTask):
    """Fetch a single daily snapshot from GitHub."""

    def __init__(self, repo: str):
        self.repo = repo

    @property
    def schema(self) -> Schema:
        return Schema._from_field_name_and_types(
            [
                ("date", DataType.string()),
                ("stars", DataType.int64()),
                ("stars_delta", DataType.int64()),
                ("forks", DataType.int64()),
                ("open_issues", DataType.int64()),
                ("open_prs", DataType.int64()),
                ("ci_status", DataType.string()),
                ("latest_release", DataType.string()),
                ("release_date", DataType.string()),
                ("contributors", DataType.int64()),
                ("commits_7d", DataType.int64()),
            ]
        )

    def _gh_api(self, path: str) -> dict | list:
        result = subprocess.run(
            ["gh", "api", path, "--cache", "1h"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"gh api {path} failed: {result.stderr.strip()}")
        return json.loads(result.stdout)

    def get_micro_partitions(self) -> Iterator[MicroPartition]:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        repo = self._gh_api(f"/repos/{self.repo}")
        stars = repo.get("stargazers_count", 0)
        forks = repo.get("forks_count", 0)
        open_issues = repo.get("open_issues_count", 0)

        prs = self._gh_api(f"/repos/{self.repo}/pulls?state=open&per_page=100")
        open_prs = len(prs) if isinstance(prs, list) else 0

        ci_status = "unknown"
        try:
            runs = self._gh_api(f"/repos/{self.repo}/actions/runs?branch=main&per_page=5")
            if wf := runs.get("workflow_runs", []):
                latest = wf[0]
                status = latest.get("status", "")
                ci_status = (
                    "running"
                    if status in ("in_progress", "queued")
                    else (latest.get("conclusion") or status or "unknown")
                )
        except Exception:
            pass

        latest_release, release_date = "", ""
        try:
            rel = self._gh_api(f"/repos/{self.repo}/releases/latest")
            latest_release = (rel.get("tag_name", "") or "").lstrip("v")
            release_date = (rel.get("published_at", "") or "")[:10]
        except Exception:
            pass

        contributors = 0
        try:
            contribs = self._gh_api(f"/repos/{self.repo}/stats/contributors")
            contributors = len(contribs) if isinstance(contribs, list) else 0
        except Exception:
            pass

        commits_7d = 0
        try:
            activity = self._gh_api(f"/repos/{self.repo}/stats/commit_activity")
            if isinstance(activity, list) and activity:
                commits_7d = activity[-1].get("total", 0)
        except Exception:
            pass

        yield MicroPartition.from_pydict(
            {
                "date": [today],
                "stars": [stars],
                "stars_delta": [0],
                "forks": [forks],
                "open_issues": [open_issues],
                "open_prs": [open_prs],
                "ci_status": [ci_status],
                "latest_release": [latest_release],
                "release_date": [release_date],
                "contributors": [contributors],
                "commits_7d": [commits_7d],
            }
        )


class PyPIDataSource(DataSource):
    """Fetch daily download stats from pypistats.org.

    Produces one row per day with download count and 7-day trailing average.
    """

    def __init__(self, package: str = "daft"):
        self.package = package

    @property
    def name(self) -> str:
        return "PyPI Downloads"

    @property
    def schema(self) -> Schema:
        return Schema._from_field_name_and_types(
            [
                ("date", DataType.string()),
                ("package", DataType.string()),
                ("downloads", DataType.int64()),
                ("downloads_7d_avg", DataType.float64()),
            ]
        )

    async def get_tasks(self, pushdowns) -> AsyncIterator["PyPIDataSourceTask"]:
        yield PyPIDataSourceTask(self.package)


class PyPIDataSourceTask(DataSourceTask):
    """Fetch daily download data from pypistats.org."""

    def __init__(self, package: str):
        self.package = package

    @property
    def schema(self) -> Schema:
        return Schema._from_field_name_and_types(
            [
                ("date", DataType.string()),
                ("package", DataType.string()),
                ("downloads", DataType.int64()),
                ("downloads_7d_avg", DataType.float64()),
            ]
        )

    def _get(self, url: str) -> dict:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "daft-lakehouse-example/1.0",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    def get_micro_partitions(self) -> Iterator[MicroPartition]:
        data = self._get(f"https://pypistats.org/api/packages/{self.package}/overall?mirrors=true")
        rows = [r for r in data.get("data", []) if r.get("category") == "with_mirrors"]
        rows.sort(key=lambda r: r.get("date", ""))

        dates, packages, downloads, avgs = [], [], [], []
        for i, row in enumerate(rows):
            dates.append(row.get("date", ""))
            packages.append(self.package)
            dl = row.get("downloads", 0)
            downloads.append(dl)
            window = rows[max(0, i - 6) : i + 1]
            avg = sum(r.get("downloads", 0) for r in window) / len(window)
            avgs.append(round(avg, 1))

        yield MicroPartition.from_pydict(
            {
                "date": dates,
                "package": packages,
                "downloads": downloads,
                "downloads_7d_avg": avgs,
            }
        )
