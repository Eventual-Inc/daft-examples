# /// script
# description = "Lambda Reduce → Iceberg: persist each reduction level to Iceberg tables"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[openai,iceberg]>=0.7.8", "pyiceberg[sql-sqlite]>=0.11.0", "pymupdf", "python-dotenv"]
# ///

"""
Lambda Reduce → Iceberg
========================

Runs hierarchical lambda reduction on research papers and writes
each level to an Iceberg table via PyIceberg's SQLite catalog.

Tables created:
    lambda.pages        — raw extracted pages (path, page_number, page_text)
    lambda.page_results — level 0: per-page summaries
    lambda.doc_results  — level 1: per-document summaries
    lambda.corpus       — level 2: the fixed point

All writes use Daft's native .write_iceberg() — lazy plan → Parquet files
managed by Iceberg's metadata layer.

Usage:
    uv run examples/lambda_reduce_iceberg.py
    uv run examples/lambda_reduce_iceberg.py --max-papers 5 --model gpt-5-mini
    uv run examples/lambda_reduce_iceberg.py --warehouse ./my_warehouse
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

import pymupdf
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.schema import Schema
from pyiceberg.types import (
    IntegerType,
    LongType,
    NestedField,
    StringType,
)

import daft
from daft import DataFrame, col, lit
from daft.functions import format, prompt, unnest

# ==============================================================================
# PDF extraction (same as lambda_reduce.py)
# ==============================================================================


class PdfPage(TypedDict):
    page_number: int
    page_text: str


@daft.func
def extract_pdf(file: daft.File) -> Iterator[PdfPage]:
    """Extract text per page. Generator UDF: 1 file → N pages."""
    pymupdf.TOOLS.mupdf_display_errors(False)
    with file.to_tempfile() as tmp:
        doc = pymupdf.Document(filename=str(tmp.name), filetype="pdf")
        for pno, page in enumerate(doc):
            text = page.get_text("text")
            if len(text.strip()) > 50:
                yield PdfPage(page_number=pno, page_text=text)


def load_papers(source: str, max_papers: int, max_pages: int | None = None) -> DataFrame:
    """Load PDFs → extract pages → flat DataFrame."""
    df = (
        daft.from_glob_path(source)
        .limit(max_papers)
        .with_column("pdf_file", daft.functions.file(col("path")))
        .with_column("page", extract_pdf(col("pdf_file")))
        .select("path", "size", unnest(col("page")))
    )
    if max_pages is not None:
        df = df.limit(max_pages)
    return df


# ==============================================================================
# Iceberg catalog + table setup
# ==============================================================================

PAGES_SCHEMA = Schema(
    NestedField(1, "path", StringType(), required=False),
    NestedField(2, "size", LongType(), required=False),
    NestedField(3, "page_number", IntegerType(), required=False),
    NestedField(4, "page_text", StringType(), required=False),
)

PAGE_RESULTS_SCHEMA = Schema(
    NestedField(1, "path", StringType(), required=False),
    NestedField(2, "page_number", IntegerType(), required=False),
    NestedField(3, "page_text", StringType(), required=False),
    NestedField(4, "result", StringType(), required=False),
)

DOC_RESULTS_SCHEMA = Schema(
    NestedField(1, "path", StringType(), required=False),
    NestedField(2, "result", StringType(), required=False),
)

CORPUS_SCHEMA = Schema(
    NestedField(1, "n_items", LongType(), required=False),
    NestedField(2, "result", StringType(), required=False),
)


def setup_catalog(warehouse: str, namespace: str = "lambda") -> SqlCatalog:
    """Create a SQLite-backed Iceberg catalog."""
    warehouse_path = Path(warehouse).resolve()
    warehouse_path.mkdir(parents=True, exist_ok=True)
    db_path = warehouse_path / "catalog.db"

    catalog = SqlCatalog(
        "lambda_reduce",
        uri=f"sqlite:///{db_path}",
        warehouse=str(warehouse_path),
    )

    # Ensure namespace exists
    try:
        catalog.create_namespace(namespace)
    except Exception:
        pass  # already exists

    return catalog


def ensure_table(catalog: SqlCatalog, name: str, schema: Schema) -> pyiceberg.table.Table:
    """Create or load an Iceberg table."""
    try:
        return catalog.create_table(name, schema=schema)
    except Exception:
        return catalog.load_table(name)


# ==============================================================================
# Reduction levels (same logic as lambda_reduce.py, but each level is written)
# ==============================================================================


def reduce_level_0(df: DataFrame, model: str) -> DataFrame:
    """Level 0: MAP each page → per-page summary."""
    return df.with_column(
        "result",
        prompt(
            lit("Summarize the following text concisely. Capture the key claims, methods, and findings:\n\n")
            + col("page_text"),
            model=model,
        ),
    )


def reduce_level_1(df: DataFrame, model: str) -> DataFrame:
    """Level 1: AGG per document → MAP → per-document summary."""
    return (
        df.groupby("path")
        .agg(col("result").list_agg().alias("parts"))
        .with_column("context", col("parts").list_join("\n\n---\n\n"))
        .with_column(
            "result",
            prompt(
                format(
                    "Merge these partial summaries into one coherent summary of the full paper. "
                    "Preserve the key contributions, methods, results, and conclusions:\n\n{}",
                    col("context"),
                ),
                model=model,
            ),
        )
        .select("path", "result")
    )


def reduce_level_2(df: DataFrame, model: str) -> DataFrame:
    """Level 2: AGG all → MAP → corpus summary. The fixed point."""
    return (
        df.select(
            col("result").count().alias("n_items"),
            col("result").list_agg().alias("parts"),
        )
        .with_column("context", col("parts").list_join("\n\n===\n\n"))
        .with_column(
            "result",
            prompt(
                format(
                    "You have read summaries of {} research papers. "
                    "Synthesize them into a single comprehensive analysis. "
                    "Identify common themes, contrasting approaches, key findings, "
                    "and the overall direction of the field:\n\n{}",
                    col("n_items"),
                    col("context"),
                ),
                model=model,
            ),
        )
        .select("n_items", "result")
    )


# ==============================================================================
# Main
# ==============================================================================

DEFAULT_SOURCE = "hf://datasets/Eventual-Inc/sample-files/papers/*.pdf"


def main():
    parser = argparse.ArgumentParser(description="Lambda Reduce → Iceberg")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--max-papers", type=int, default=5)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--warehouse", default="./lambda_warehouse")
    parser.add_argument("--namespace", default="lambda")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    daft.set_provider("openai", api_key=os.environ.get("OPENAI_API_KEY"))

    print("=" * 70)
    print("  Lambda Reduce → Iceberg")
    print("=" * 70)

    # ── Setup catalog ────────────────────────────────────────────────────
    catalog = setup_catalog(args.warehouse, args.namespace)
    ns = args.namespace

    pages_tbl = ensure_table(catalog, f"{ns}.pages", PAGES_SCHEMA)
    page_results_tbl = ensure_table(catalog, f"{ns}.page_results", PAGE_RESULTS_SCHEMA)
    doc_results_tbl = ensure_table(catalog, f"{ns}.doc_results", DOC_RESULTS_SCHEMA)
    corpus_tbl = ensure_table(catalog, f"{ns}.corpus", CORPUS_SCHEMA)

    print(f"\nWarehouse: {Path(args.warehouse).resolve()}")
    print(f"Namespace: {ns}")
    print(f"Tables: {ns}.pages, {ns}.page_results, {ns}.doc_results, {ns}.corpus")

    # ── SPLIT: load papers ───────────────────────────────────────────────
    t_total = time.perf_counter()
    print(f"\nLoading papers from: {args.source}")

    pages_df = load_papers(args.source, args.max_papers, args.max_pages)

    # Write raw pages
    print(f"\n── Writing {ns}.pages ──")
    t0 = time.perf_counter()
    pages_df.write_iceberg(pages_tbl, mode="overwrite")
    print(f"  ({time.perf_counter() - t0:.1f}s)")

    # Read back from Iceberg (proves the round-trip)
    pages_df = daft.read_iceberg(pages_tbl)
    print(f"  Rows: {pages_df.count_rows()}")

    # ── LEVEL 0: per-page summaries ──────────────────────────────────────
    print(f"\n── Level 0: MAP pages → writing {ns}.page_results ──")
    t0 = time.perf_counter()

    page_results = reduce_level_0(pages_df, args.model)
    page_results.write_iceberg(page_results_tbl, mode="overwrite")
    print(f"  ({time.perf_counter() - t0:.1f}s)")

    page_results = daft.read_iceberg(page_results_tbl)
    print(f"  Rows: {page_results.count_rows()}")
    page_results.select("path", "page_number", col("result").substr(0, 80).alias("preview")).show(5)

    # ── LEVEL 1: per-document summaries ──────────────────────────────────
    print(f"\n── Level 1: AGG → MAP → writing {ns}.doc_results ──")
    t0 = time.perf_counter()

    doc_results = reduce_level_1(page_results, args.model)
    doc_results.write_iceberg(doc_results_tbl, mode="overwrite")
    print(f"  ({time.perf_counter() - t0:.1f}s)")

    doc_results = daft.read_iceberg(doc_results_tbl)
    print(f"  Rows: {doc_results.count_rows()}")
    doc_results.show()

    # ── LEVEL 2: corpus → fixed point ────────────────────────────────────
    print(f"\n── Level 2: AGG all → MAP → writing {ns}.corpus ──")
    t0 = time.perf_counter()

    corpus = reduce_level_2(doc_results, args.model)
    corpus.write_iceberg(corpus_tbl, mode="overwrite")
    print(f"  ({time.perf_counter() - t0:.1f}s)")

    corpus = daft.read_iceberg(corpus_tbl)

    # ── Result ───────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_total

    print("\n" + "=" * 70)
    print("  FIXED POINT REACHED — persisted to Iceberg")
    print("=" * 70)

    rows = corpus.collect().to_pylist()
    for row in rows:
        n = row.get("n_items", "?")
        result = row.get("result", "(empty)")
        print(f"\n  Papers: {n}")
        print(f"  Result preview: {result[:300]}...")

    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"\n  Tables written to: {Path(args.warehouse).resolve()}")
    print(f"    {ns}.pages         — raw pages")
    print(f"    {ns}.page_results  — per-page summaries")
    print(f"    {ns}.doc_results   — per-document summaries")
    print(f"    {ns}.corpus        — the fixed point")

    # Show you can read them back
    print(f"\n  Quick verification — reading {ns}.doc_results back:")
    daft.read_iceberg(doc_results_tbl).select("path", col("result").substr(0, 60).alias("summary")).show()

    print("=" * 70)


if __name__ == "__main__":
    main()
