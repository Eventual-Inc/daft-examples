# /// script
# description = "This example shows how using LLMs and embedding models, Daft chunks documents, extracts metadata, generates vectors, and writes them to any vector database..."
# dependencies = ["daft[openai]", "pymupdf", "python-dotenv"]
# ///
import os
import daft
from daft import col, lit
from daft.functions import embed_text, prompt, file, unnest
from pydantic import BaseModel
from dotenv import load_dotenv
import pymupdf

load_dotenv()

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
def extract_pdf(file: daft.File, num_pages: int = 10):
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


class Classifer(BaseModel):
    title: str
    author: str
    year: int
    keywords: list[str]
    abstract: str



if __name__ == "__main__":

    daft.set_provider("openai")

    # Load documents and generate vector embeddings
    df = (
        daft.from_glob_path("hf://datasets/Eventual-Inc/sample-files/papers/*.pdf").limit(1)
        .with_column(
            "metadata", 
            prompt(
                messages=file(col("path")),
                system_message="Read the paper and extract the classifer metadata.",
                return_format=Classifer, 
                model="gpt-5-mini", 
                provider="openai",
            )
        )
        .with_column(
            "abstract_embedding", 
            embed_text(
                daft.col("metadata")["abstract"], 
                provider="openai", 
                model="text-embedding-3-large"
            )
        )
    )

    print(df.to_pydict())

    df = df.select("path", unnest(col("metadata")), "abstract_embedding")
    df = df.write_turbopuffer(
        namespace="ai_papers",
        api_key=os.environ.get("TURBOPUFFER_API_KEY"), 
        distance_metric="cosine_distance",
        schema={
            "path": "string",
            "title": "string",
            "author": "string",
            "year": "int",
            "keywords": "list[string]",
            "abstract": "string",
            "abstract_embedding": "vector",
        }
    )
