# /// script
# description = "Minimal RAG Example"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[openai]>=0.7.10", "pymupdf", "python-dotenv"]
# ///

import pymupdf

import daft
from daft import DataType


@daft.func(
    return_dtype=DataType.struct(
        {
            "page_number": DataType.int32(),
            "page_text": DataType.string(),
        }
    )
)
def extract_pdf(doc: daft.File):
    with doc.to_tempfile() as temp_file:
        try:
            document = pymupdf.Document(filename=str(temp_file.name), filetype="pdf")
            for page_number, page in enumerate(document):
                text = page.get_text("text")
                if text and text.strip():
                    yield {
                        "page_number": page_number,
                        "page_text": text,
                    }
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            yield None


if __name__ == "__main__":
    from dotenv import load_dotenv

    from daft import col
    from daft.functions import cosine_distance, embed_text, file, unnest

    load_dotenv()

    TEXT_EMBEDDING_MODEL = "text-embedding-3-small"

    # Discover, Load, and Extract Text from PDFs
    df = (
        daft.from_glob_path("hf://datasets/Eventual-Inc/sample-files/papers/*.pdf")
        .with_column("pdf_file", file(col("path")))
        .with_column("pdf_pages", extract_pdf(col("pdf_file")))
        .select("path", unnest(col("pdf_pages")))
        .with_column(
            "text_embedding",
            embed_text(col("page_text"), provider="openai", model=TEXT_EMBEDDING_MODEL),
        )
    )

    # Create and embed the query
    query = daft.from_pydict({"query_text": ["What is Daft?"]}).with_column(
        "query_embedding",
        embed_text(col("query_text"), provider="openai", model=TEXT_EMBEDDING_MODEL),
    )

    # Cross join and rank by cosine distance
    results = (
        query.join(df, how="cross")
        .with_column(
            "distance",
            cosine_distance(col("query_embedding"), col("text_embedding")),
        )
        .sort("distance")
        .select("query_text", "page_text", "distance")
    )

    results.show()
