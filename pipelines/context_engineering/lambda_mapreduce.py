# /// script
# description = "Lambda MapReduce: 6 long-context reasoning patterns expressed as native Daft query plans"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[openai]>=0.7.8", "pymupdf", "python-dotenv"]
# ///

"""
Lambda MapReduce — Long-Context Reasoning via Daft
===================================================

Implements the core insight from lambda-RLM (Roy et al., 2026):
arbitrary recursive LLM reasoning reduces to MAP + REDUCE over
bounded chunks, with the LLM as a leaf oracle.

The key realization: REDUCE is a Daft aggregation, not a Python loop.

    SPLIT  = from_glob_path → extract_pdf → unnest      (pages are chunks)
    FILTER = where(prompt(...).startswith("Y"))      (LLM as predicate)
    MAP    = with_column("result", prompt(col(...)))     (LLM as leaf oracle)
    REDUCE = groupby().agg() → list_join() [→ prompt()]  (fold via Daft expressions)

Six patterns, one skeleton. Every operation stays in the query plan.

Usage:
    uv run pipelines/context_engineering/lambda_mapreduce.py
    uv run pipelines/context_engineering/lambda_mapreduce.py --pattern summarize
    uv run pipelines/context_engineering/lambda_mapreduce.py --pattern qa --query "What methods are proposed?"
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterator
from typing import TypedDict

import pymupdf

import daft
from daft import DataFrame, col, lit
from daft.functions import format, prompt, unnest

# ==============================================================================
# SPLIT: PDF → pages (document structure IS the split)
# ==============================================================================


class PdfPage(TypedDict):
    page_number: int
    page_text: str


@daft.func
def extract_pdf(file: daft.File) -> Iterator[PdfPage]:
    """Extract text from each page of a PDF. Generator UDF: 1 file → N pages."""
    pymupdf.TOOLS.mupdf_display_errors(False)
    with file.to_tempfile() as tmp:
        doc = pymupdf.Document(filename=str(tmp.name), filetype="pdf")
        for pno, page in enumerate(doc):
            text = page.get_text("text")
            if len(text.strip()) > 50:
                yield PdfPage(page_number=pno, page_text=text)


def load_papers(source: str, max_papers: int = 2, max_pages: int | None = None) -> DataFrame:
    """SPLIT: Load PDFs and extract pages into a flat DataFrame."""
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
# The six lambda-RLM patterns — all MAP + REDUCE via Daft expressions
# ==============================================================================


def pattern_search(df: DataFrame, query: str, model: str) -> DataFrame:
    """
    Search: PEEK → FILTER → MAP → SELECT_BEST

    Find a needle in a haystack. Filter cheap (preview), answer expensive (full page).
    """
    return (
        df
        # PEEK: first 200 chars
        .with_column("preview", col("page_text").substr(0, 200))
        # FILTER: LLM relevance check on preview
        .with_column(
            "is_relevant",
            prompt(
                format(
                    "Question: {}\n\nDoes this excerpt likely contain the answer? Reply YES or NO only.\n\nExcerpt:\n{}",
                    lit(query),
                    col("preview"),
                ),
                model=model,
            ),
        )
        .where(col("is_relevant").upper().startswith("Y"))
        # MAP: answer from relevant pages
        .with_column(
            "result",
            prompt(
                format("Question: {}\n\nAnswer using ONLY this context:\n\n{}", lit(query), col("page_text")),
                model=model,
            ),
        )
        # REDUCE: select best — filter noise, take earliest page
        .where(~col("result").lower().contains("not found"))
        .where(~col("result").lower().contains("no information"))
        .sort("page_number")
        .limit(1)
        .select("path", "page_number", "result")
    )


def pattern_summarize(df: DataFrame, model: str) -> DataFrame:
    """
    Summarize: MAP → AGG → LLM_REDUCE

    Summarize each page, aggregate, then merge summaries via one LLM call.
    """
    return (
        df
        # MAP: summarize each page
        .with_column(
            "result",
            prompt(
                lit("Summarize the following text concisely:\n\n") + col("page_text"),
                model=model,
            ),
        )
        # REDUCE: aggregate all summaries per document, then merge via LLM
        .groupby("path")
        .agg(col("result").list_agg().alias("partial_summaries"))
        .with_column("context", col("partial_summaries").list_join("\n\n---\n\n"))
        .with_column(
            "result",
            prompt(
                format(
                    "Merge these partial summaries into one concise, coherent summary. Preserve all key facts:\n\n{}",
                    col("context"),
                ),
                model=model,
            ),
        )
        .select("path", "result")
    )


def pattern_classify(df: DataFrame, model: str) -> DataFrame:
    """
    Classify: MAP → MAJORITY_VOTE

    Classify each page, then majority vote via groupby + count.
    """
    return (
        df
        # MAP: classify each page
        .with_column(
            "label",
            prompt(
                lit(
                    "Classify this text into exactly one category "
                    "(Methods, Results, Introduction, Discussion, Related Work). "
                    "Reply with ONLY the category name:\n\n"
                )
                + col("page_text"),
                model=model,
            ),
        )
        # REDUCE: majority vote — normalize, count, take most frequent
        .with_column("label_norm", col("label").lower().lstrip().rstrip())
        .groupby("label_norm")
        .agg(col("label_norm").count().alias("votes"))
        .sort("votes", desc=True)
        .limit(1)
        .select(col("label_norm").alias("result"), "votes")
    )


def pattern_extract(df: DataFrame, model: str) -> DataFrame:
    """
    Extract: MAP → SPLIT_LINES → DEDUP

    Extract facts from each page, split into lines, deduplicate.
    """
    return (
        df
        # MAP: extract facts, one per line
        .with_column(
            "result",
            prompt(
                lit("Extract all key facts, entities, numbers, and findings. One fact per line:\n\n")
                + col("page_text"),
                model=model,
            ),
        )
        # REDUCE: split lines, explode, dedup
        .with_column("facts", col("result").split("\n"))
        .explode("facts")
        .with_column("fact", col("facts").lstrip().rstrip())
        .where(col("fact").length() > 0)
        .distinct()
        .select(col("fact").alias("result"))
    )


def pattern_qa(df: DataFrame, query: str, model: str) -> DataFrame:
    """
    QA: FILTER → MAP → AGG → LLM_SYNTHESIZE

    Filter relevant pages, answer from each, aggregate, synthesize.
    """
    return (
        df
        # FILTER: LLM relevance check on preview
        .with_column("preview", col("page_text").substr(0, 200))
        .with_column(
            "is_relevant",
            prompt(
                format(
                    "Question: {}\n\nDoes this excerpt contain relevant information? Reply YES or NO only.\n\nExcerpt:\n{}",
                    lit(query),
                    col("preview"),
                ),
                model=model,
            ),
        )
        .where(col("is_relevant").upper().startswith("Y"))
        # MAP: answer from each relevant page
        .with_column(
            "result",
            prompt(
                format("Question: {}\n\nAnswer based on this context only:\n\n{}", lit(query), col("page_text")),
                model=model,
            ),
        )
        # REDUCE: aggregate partial answers, then synthesize via LLM
        .where(~col("result").lower().contains("not found"))
        .groupby("path")
        .agg(col("result").list_agg().alias("partial_answers"))
        .with_column("context", col("partial_answers").list_join("\n\n---\n\n"))
        .with_column(
            "result",
            prompt(
                format(
                    "Question: {}\n\nSynthesise these partial answers into one complete, accurate answer:\n\n{}",
                    lit(query),
                    col("context"),
                ),
                model=model,
            ),
        )
        .select("path", "result")
    )


def pattern_analyze(df: DataFrame, model: str) -> DataFrame:
    """
    Analyze: MAP → AGG → LLM_COMBINE

    Analyze each page, aggregate, combine insights via LLM.
    """
    return (
        df
        # MAP: analyze each page
        .with_column(
            "result",
            prompt(
                lit("Analyze the following text. Identify key themes, arguments, and implications:\n\n")
                + col("page_text"),
                model=model,
            ),
        )
        # REDUCE: aggregate analyses, then combine via LLM
        .groupby("path")
        .agg(col("result").list_agg().alias("partial_analyses"))
        .with_column("context", col("partial_analyses").list_join("\n\n---\n\n"))
        .with_column(
            "result",
            prompt(
                format(
                    "Combine these partial analyses into one comprehensive, well-structured analysis:\n\n{}",
                    col("context"),
                ),
                model=model,
            ),
        )
        .select("path", "result")
    )


# ==============================================================================
# Pattern registry
# ==============================================================================

PATTERNS = {
    "search": pattern_search,
    "summarize": pattern_summarize,
    "classify": pattern_classify,
    "extract": pattern_extract,
    "qa": pattern_qa,
    "analyze": pattern_analyze,
}

QUERY_PATTERNS = {"search", "qa"}
DEFAULT_SOURCE = "hf://datasets/Eventual-Inc/sample-files/papers/*.pdf"


# ==============================================================================
# Runner
# ==============================================================================


def run_pattern(name: str, pages_df: DataFrame, model: str, query: str | None) -> tuple[DataFrame, float]:
    """Run a single pattern. Returns (result_df, elapsed_seconds)."""
    fn = PATTERNS[name]
    t0 = time.perf_counter()

    if name in QUERY_PATTERNS:
        q = query or "What are the main contributions of this paper?"
        result_df = fn(pages_df, query=q, model=model)
    else:
        result_df = fn(pages_df, model=model)

    # Materialize the query plan
    result_df = result_df.collect()
    elapsed = time.perf_counter() - t0
    return result_df, elapsed


def main():
    parser = argparse.ArgumentParser(description="Lambda MapReduce — Long-Context Reasoning via Daft")
    parser.add_argument("--pattern", choices=list(PATTERNS.keys()) + ["all"], default="all")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--max-papers", type=int, default=1)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--query", default=None)
    args = parser.parse_args()

    import os

    from dotenv import load_dotenv

    load_dotenv()
    daft.set_provider("openai", api_key=os.environ.get("OPENAI_API_KEY"))

    print("=" * 60)
    print("Lambda MapReduce — Long-Context Reasoning via Daft")
    print("=" * 60)

    # SPLIT
    pages_df = load_papers(args.source, args.max_papers, args.max_pages)
    print("\nPages loaded:")
    pages_df.select("path", "page_number", col("page_text").length().alias("chars")).show(10)

    patterns_to_run = list(PATTERNS.keys()) if args.pattern == "all" else [args.pattern]

    for name in patterns_to_run:
        print(f"\n{'─' * 60}")
        print(f"Pattern: {name}")
        if name in QUERY_PATTERNS:
            q = args.query or "What are the main contributions of this paper?"
            print(f"Query:   {q}")
        print(f"{'─' * 60}")

        result_df, elapsed = run_pattern(name, pages_df, args.model, args.query)
        result_df.show()
        print(f"({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
