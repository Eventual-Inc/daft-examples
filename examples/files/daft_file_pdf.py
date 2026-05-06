# /// script
# description = "Extract the content of a PDF file"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[huggingface]>=0.7.10", "pymupdf"]
# ///
from collections.abc import Iterator
from typing import TypedDict

import pymupdf

import daft


class PdfPage(TypedDict):
    page_number: int
    page_text: str
    page_image_bytes: bytes


@daft.func
def extract_pdf(file: daft.File) -> Iterator[PdfPage]:
    """Extracts the content of a PDF file."""
    pymupdf.TOOLS.mupdf_display_errors(False)  # Suppress non-fatal MuPDF warnings

    with file.to_tempfile() as tmp:
        doc = pymupdf.Document(filename=str(tmp.name), filetype="pdf")
        for pno, page in enumerate(doc):
            row = PdfPage(
                page_number=pno,
                page_text=page.get_text("text"),
                page_image_bytes=page.get_pixmap().tobytes(),
            )
            yield row


if __name__ == "__main__":
    from daft import col

    df = (
        daft.from_glob_path("hf://datasets/Eventual-Inc/sample-files/papers/*.pdf")
        .with_column("pdf_file", daft.functions.file(col("path")))
        .with_column("page", extract_pdf(col("pdf_file")))
        .select("path", "size", daft.functions.unnest(col("page")))
    )

    df.show(3)
