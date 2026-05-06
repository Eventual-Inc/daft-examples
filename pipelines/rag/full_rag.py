# /// script
# description = "Full RAG example"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[openai]>=0.7.10", "pymupdf", "python-dotenv"]
# ///

import pymupdf  # type: ignore
from dotenv import load_dotenv  # type: ignore

import daft
from daft import DataType, col
from daft.functions import cosine_distance, embed_text, file, format, prompt, unnest


@daft.func(
    return_dtype=DataType.struct(
        {
            "page_number": DataType.int32(),
            "page_text": DataType.string(),
        }
    )
)
def extract_pdf_pages(pdf_file: daft.File):
    """Extract text from each PDF page."""
    with pdf_file.to_tempfile() as temp_file:
        document = None
        try:
            document = pymupdf.Document(filename=str(temp_file.name), filetype="pdf")
            for page_number, page in enumerate(document):
                text = page.get_text("text")
                if text and text.strip():
                    yield {
                        "page_number": page_number,
                        "page_text": text,
                    }
        except Exception as exc:
            print(f"Failed to extract PDF contents: {exc}")
        finally:
            if document is not None:
                document.close()


if __name__ == "__main__":
    load_dotenv()

    PDF_URI = "hf://datasets/Eventual-Inc/sample-files/papers/*.pdf"
    TEXT_EMBEDDING_MODEL = "text-embedding-3-small"
    GENERATION_MODEL = "gpt-5-nano"
    QUESTION = "What is Daft?"
    TOP_K = 3

    # Discover PDFs, extract their pages, and embed the text.
    documents = (
        daft.from_glob_path(PDF_URI)
        .with_column("pdf_file", file(col("path")))
        .with_column("pdf_pages", extract_pdf_pages(col("pdf_file")))
        .select("path", unnest(col("pdf_pages")))
        .with_column(
            "page_embedding",
            embed_text(col("page_text"), provider="openai", model=TEXT_EMBEDDING_MODEL),
        )
    )

    # Encode the user query once.
    query = daft.from_pydict({"query_text": [QUESTION]}).with_column(
        "query_embedding",
        embed_text(col("query_text"), provider="openai", model=TEXT_EMBEDDING_MODEL),
    )

    # Rank all PDF chunks by cosine distance to the query embedding.
    ranked = (
        query.join(documents, how="cross")
        .with_column(
            "distance",
            cosine_distance(col("query_embedding"), col("page_embedding")),
        )
        .sort("distance")
    )

    top_matches = (
        ranked.select("query_text", "path", "page_number", "page_text", "distance").limit(TOP_K).collect().to_pydict()
    )

    if not top_matches["path"]:
        raise RuntimeError("No PDF text was extracted — check your data source.")

    context_sections = []
    for path_value, page_number_value, page_text_value in zip(
        top_matches["path"],
        top_matches["page_number"],
        top_matches["page_text"],
    ):
        context_sections.append(f"Source: {path_value} (page {page_number_value})\n{page_text_value.strip()}")

    context_blob = "\n\n---\n\n".join(context_sections)

    rag_df = daft.from_pydict(
        {
            "question": [QUESTION],
            "context": [context_blob],
        }
    ).with_column(
        "response",
        prompt(
            messages=format(
                "Using the following context, answer the question.\n\nContext:\n{}\n\nQuestion: {}",
                col("context"),
                col("question"),
            ),
            model=GENERATION_MODEL,
            provider="openai",
        ),
    )

    rag_df.select("question", "response").show(truncate=80)
