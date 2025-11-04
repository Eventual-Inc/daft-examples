# /// script
# description = "MinimalRAG Example"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft>=0.6.8", "pymupdf", "pillow"]
# ///

import pymupdf
import daft
from daft import DataType, col
from daft.functions import unnest, decode_image
from mlx_vlm import load, generate
import mlx.core as mx
from PIL import Image
import io

@daft.func(
    return_dtype=DataType.struct({"page_number": DataType.int32(), "page_text": DataType.string(), "page_image_bytes": DataType.binary()})
    
)
def extract_pdf(doc: bytes):
    try:
        # Create a PDF reader object using BytesIO
        document = pymupdf.Document(stream=io.BytesIO(doc), filetype="pdf")

        # Extract text from all pages, track page number
        for page_number, page in enumerate(document):
            text = page.get_text("text")
            yield {"page_number": page_number, "page_text": text, "page_image_bytes": page.get_pixmap().tobytes()}

    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        yield None

if __name__ == "__main__":
    df = (
        daft.from_glob_path("hf://datasets/Eventual-Inc/sample-files/papers/*.pdf")
        .with_column("pdf_bytes", col("path").url.download())
        .with_column("pdf_pages", extract_pdf(col("pdf_bytes")))
        .select(col("path"), unnest(col("pdf_pages")))
        .with_column("page_image", decode_image(col("page_image_bytes")).convert_image("RGB"))
    ).collect()
    df.show()