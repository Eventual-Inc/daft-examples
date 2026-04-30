# /// script
# description = "Read Iceberg tables with Daft and visualize with matplotlib."
# requires-python = ">=3.12, <3.13"
# dependencies = [
#     "daft[iceberg,sql]>=0.7.8",
#     "pyiceberg[sql-sqlite,gcsfs,bigquery]",
#     "python-dotenv",
#     "matplotlib",
# ]
# ///
"""Analyze pipeline — read Iceberg tables with Daft and plot with matplotlib.

Usage:
    uv run --extra lakehouse -m pipelines.lakehouse_analytics.analyze
    GCP_PROJECT=eventual-analytics uv run --extra lakehouse -m pipelines.lakehouse_analytics.analyze
"""

from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import daft
from pipelines.catalog import get_session

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def plot_pypi_downloads(sess):
    df = (
        sess.read_table("pypi_downloads")
        .where(daft.col("date") >= "2025-10-01")
        .select("date", "downloads", "downloads_7d_avg")
        .sort("date")
        .collect()
    )

    data = df.to_pydict()
    dates = [datetime.strptime(d, "%Y-%m-%d") for d in data["date"]]
    downloads = data["downloads"]
    avg_7d = data["downloads_7d_avg"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(dates, downloads, alpha=0.3, color="#4A90D9", label="Daily downloads")
    ax.plot(dates, avg_7d, color="#D94A4A", linewidth=2, label="7-day avg")
    ax.axhline(y=35_000, color="#2ECC71", linestyle="--", linewidth=1.5, label="Target (35K)")

    ax.set_title("Daft PyPI Downloads", fontsize=14, fontweight="bold")
    ax.set_ylabel("Downloads")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    fig.autofmt_xdate()
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(dates[0], dates[-1])

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pypi_downloads.png", dpi=150)
    print(OUTPUT_DIR / "pypi_downloads.png")
    plt.close(fig)


def plot_github_stargazers(sess):
    try:
        df = sess.read_table("github_stargazers").select("starred_at").sort("starred_at").collect()
    except Exception:
        return

    if df.count_rows() == 0:
        return

    data = df.to_pydict()
    timestamps = data["starred_at"]
    dates = []
    for ts in timestamps:
        if isinstance(ts, str):
            dates.append(datetime.fromisoformat(ts.replace("+00:00", "+00:00")))
        else:
            dates.append(ts)

    cumulative = list(range(1, len(dates) + 1))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, cumulative, color="#F5A623", linewidth=2)
    ax.fill_between(dates, cumulative, alpha=0.1, color="#F5A623")

    ax.set_title("Daft GitHub Stars", fontsize=14, fontweight="bold")
    ax.set_ylabel("Cumulative Stars")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate()
    ax.grid(axis="y", alpha=0.3)

    ax.annotate(
        f"{cumulative[-1]:,} stars",
        xy=(dates[-1], cumulative[-1]),
        xytext=(-80, 10),
        textcoords="offset points",
        fontsize=12,
        fontweight="bold",
        color="#F5A623",
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "github_stars.png", dpi=150)
    print(OUTPUT_DIR / "github_stars.png")
    plt.close(fig)


def print_scorecard(sess):
    try:
        latest_pypi = (
            sess.read_table("pypi_downloads")
            .select("date", "downloads_7d_avg")
            .sort("date", desc=True)
            .limit(1)
            .collect()
        )
        avg = latest_pypi.to_pydict()["downloads_7d_avg"][0]
        pct = avg / 35_000 * 100
        print(f"pypi_7d_avg {avg:,.0f} {pct:.0f}%")
    except Exception:
        print("pypi_7d_avg no_data")

    try:
        github_stargazers = sess.read_table("github_stargazers").collect()
        print(f"github_stars {github_stargazers.count_rows():,}")
    except Exception:
        print("github_stars no_data")

    try:
        github_daily = (
            sess.read_table("github_daily")
            .select("date", "stars", "forks", "open_prs", "ci_status")
            .sort("date", desc=True)
            .limit(1)
            .collect()
        )
        row = github_daily.to_pydict()
        print(
            "github_daily",
            row["stars"][0],
            row["forks"][0],
            row["open_prs"][0],
            row["ci_status"][0],
        )
    except Exception:
        print("github_daily no_data")


def main():
    sess = get_session()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        plot_pypi_downloads(sess)
    except Exception as e:
        print(f"pypi_downloads skipped: {e}")

    plot_github_stargazers(sess)

    print_scorecard(sess)


if __name__ == "__main__":
    main()
