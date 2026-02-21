# /// script
# description = "Extract the content of a PDF file"
# requires-python = ">=3.11, <3.13"
# dependencies = ["daft>=0.7.2", "pymupdf"]
# ///
import daft
from daft.functions import unnest, file as daft_file
import pymupdf


@daft.func(
    return_dtype=daft.DataType.list(
        daft.DataType.struct(
            {
                "page_number": daft.DataType.uint8(),
                "page_text": daft.DataType.string(),
                "page_image_bytes": daft.DataType.binary(),
            }
        )
    )
)
def extract_pdf(file: daft.File):
    """Extracts the content of a PDF file."""
    pymupdf.TOOLS.mupdf_display_errors(False)  # Suppress non-fatal MuPDF warnings
    content = []
    with file.to_tempfile() as tmp:
        doc = pymupdf.Document(filename=str(tmp.name), filetype="pdf")
        for pno, page in enumerate(doc):
            row = {
                "page_number": pno,
                "page_text": page.get_text("text"),
                "page_image_bytes": page.get_pixmap().tobytes(),
            }
            content.append(row)
        return content


if __name__ == "__main__":
    from daft import col

    df = (
        daft.from_glob_path("hf://datasets/Eventual-Inc/sample-files/papers/*.pdf")
        .with_column("pdf_file", daft_file(col("path")))
        .with_column("pages", extract_pdf(col("pdf_file")))
        .explode("pages")
        .select("path", "size", unnest(col("pages")))
    )

    df.show(3)
