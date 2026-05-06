# /// script
# description = "Backfill SQL-compatible data into an Iceberg lakehouse with Daft."
# requires-python = ">=3.12, <3.13"
# dependencies = [
#     "daft[iceberg,sql]>=0.7.10",
#     "pyiceberg[sql-sqlite,gcsfs,bigquery]",
#     "python-dotenv",
#     "sqlalchemy-bigquery",
# ]
# ///
"""Backfill SQL-compatible data into the Iceberg lakehouse.

Usage:
    uv run --extra lakehouse -m pipelines.lakehouse_analytics.backfill \
        --source-url sqlite:///source.db \
        --source-table events \
        --target-table events \
        --key id \
        --target-namespace analytics

    uv run --extra lakehouse -m pipelines.lakehouse_analytics.backfill \
        --source-url bigquery://eventual-analytics \
        --query "SELECT * FROM dataset.table" \
        --target-table table \
        --key id
"""

import argparse
import os
import sys

import sqlalchemy
from dotenv import load_dotenv

import daft
from daft import DataType
from pipelines.catalog import get_session

load_dotenv()


def write_upsert(sess, table: str, new_df: daft.DataFrame, on: str | list[str]) -> None:
    keys = [on] if isinstance(on, str) else on
    try:
        existing = sess.read_table(table)
    except Exception:
        sess.create_table_if_not_exists(table, new_df)
        return

    kept = existing.join(new_df.select(*keys), on=keys, how="anti")
    sess.write_table(table, kept.concat(new_df), mode="overwrite")


def env(name: str) -> str | None:
    return os.environ.get(name) or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill SQL-compatible data into an Iceberg lakehouse.")
    parser.add_argument("--source-url", default=env("BACKFILL_SOURCE_URL"), help="SQLAlchemy source URL.")
    parser.add_argument("--query", default=env("BACKFILL_QUERY"), help="SQL query to read from the source.")
    parser.add_argument("--source-table", default=env("BACKFILL_SOURCE_TABLE"), help="Source table for SELECT * reads.")
    parser.add_argument("--target-table", default=env("BACKFILL_TARGET_TABLE"), help="Target Iceberg table name.")
    parser.add_argument("--key", default=env("BACKFILL_KEY"), help="Comma-separated upsert key column names.")
    parser.add_argument(
        "--target-namespace",
        default=env("BACKFILL_TARGET_NAMESPACE") or env("LAKEHOUSE_NAMESPACE"),
        help="Target Iceberg namespace.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    missing = []
    if not args.source_url:
        missing.append("--source-url")
    if not args.target_table:
        missing.append("--target-table")
    if not args.key:
        missing.append("--key")
    if not args.query and not args.source_table:
        missing.append("--query or --source-table")
    if args.query and args.source_table:
        raise ValueError("Use either --query or --source-table, not both.")
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")


def create_conn(source_url: str):
    engine = sqlalchemy.create_engine(source_url)
    return engine.connect()


def cast_null_columns(df: daft.DataFrame) -> daft.DataFrame:
    for col_name in df.column_names:
        if df.schema()[col_name].dtype == DataType.null():
            df = df.with_column(col_name, daft.col(col_name).cast(DataType.string()))
    return df


def source_query(args: argparse.Namespace) -> str:
    if args.query:
        return args.query
    return f"SELECT * FROM {args.source_table}"


def key_columns(value: str) -> str | list[str]:
    keys = [key.strip() for key in value.split(",") if key.strip()]
    if not keys:
        raise ValueError("--key must include at least one column name.")
    return keys[0] if len(keys) == 1 else keys


def main() -> None:
    args = parse_args()
    try:
        validate_args(args)
        sess = get_session(namespace=args.target_namespace)
        df = daft.read_sql(source_query(args), lambda: create_conn(args.source_url))
        df = cast_null_columns(df)
        write_upsert(sess, args.target_table, df, on=key_columns(args.key))
        print(args.target_table)
    except Exception as error:
        print(f"backfill failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
