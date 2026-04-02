# /// script
# description = "Pattern 6: Sliding-window overlap bundles — S + P(overlap)"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft[openai]", "pymupdf", "python-dotenv"]
# ///

"""
Sliding-Window Overlap Bundles
==============================

Split text into fixed-size windows with overlap between consecutive
windows. Each chunk shares content with its neighbors for context continuity.

Desugaring: S + P(overlap)
Status: clean
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TypedDict

import pymupdf

import daft
from daft import col
from daft.functions import unnest


class PdfPage(TypedDict):
    page_number: int
    page_text: str


@daft.func
def extract_pdf(file: daft.File) -> Iterator[PdfPage]:
    pymupdf.TOOLS.mupdf_display_errors(False)
    with file.to_tempfile() as tmp:
        doc = pymupdf.Document(filename=str(tmp.name), filetype="pdf")
        for pno, page in enumerate(doc):
            text = page.get_text("text")
            if len(text.strip()) > 50:
                yield PdfPage(page_number=pno, page_text=text)


class WindowChunk(TypedDict):
    chunk_idx: int
    chunk_text: str
    start_char: int
    end_char: int


@daft.func
def sliding_window(text: str, window_size: int = 500, overlap: int = 100) -> Iterator[WindowChunk]:
    """Split text into overlapping windows."""
    n = len(text)
    step = window_size - overlap
    idx = 0
    start = 0
    while start < n:
        end = min(start + window_size, n)
        yield WindowChunk(chunk_idx=idx, chunk_text=text[start:end], start_char=start, end_char=end)
        idx += 1
        start += step


if __name__ == "__main__":
    SOURCE = "hf://datasets/Eventual-Inc/sample-files/papers/*.pdf"

    # Load and extract pages, concatenate per document
    pages = (
        daft.from_glob_path(SOURCE)
        .limit(1)
        .with_column("pdf_file", daft.functions.file(col("path")))
        .with_column("page", extract_pdf(col("pdf_file")))
        .select("path", unnest(col("page")))
    )

    # Concatenate all pages per document into one text
    doc_text = (
        pages.groupby("path")
        .agg(col("page_text").list_agg().alias("pages"))
        .with_column("full_text", col("pages").list_join("\n\n"))
    )

    # Apply sliding window chunking
    chunks = doc_text.with_column("window", sliding_window(col("full_text"))).select("path", unnest(col("window")))

    print("=== Sliding-Window Overlap Bundles (500 chars, 100 overlap) ===")
    chunks.select(
        "path",
        "chunk_idx",
        col("chunk_text").length().alias("chars"),
        "start_char",
        "end_char",
        col("chunk_text").substr(0, 60).alias("preview"),
    ).show(15)
