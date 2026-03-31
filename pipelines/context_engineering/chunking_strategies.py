# /// script
# description = "Context Engineering: Compare fixed-size, sentence-based, and paragraph chunking strategies for PDF text"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft[openai]>=0.7.5", "python-dotenv", "pymupdf"]
# ///

import os
import daft
from daft import col, lit, DataType
from daft.functions import embed_text, cosine_distance, file, format
from dotenv import load_dotenv


@daft.func(return_dtype=DataType.string())
def extract_text(pdf_file: daft.File) -> str:
    """Extract all text from a PDF using pymupdf."""
    import pymupdf

    with pdf_file.to_tempfile() as tmp:
        with pymupdf.Document(filename=str(tmp.name), filetype="pdf") as doc:
            pages = [page.get_text("text") for page in doc]
            return "\n\n".join(pages)


@daft.func(return_dtype=DataType.list(DataType.string()))
def fixed_size_chunk(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into fixed-size character chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


@daft.func(return_dtype=DataType.list(DataType.string()))
def paragraph_chunk(text: str, min_length: int = 80) -> list[str]:
    """Split on double newlines and filter out short chunks."""
    raw = text.split("\n\n")
    return [p.strip() for p in raw if len(p.strip()) >= min_length]


if __name__ == "__main__":

    load_dotenv()

    daft.set_provider("openai", api_key=os.getenv("OPENAI_API_KEY"))

    EMBEDDING_MODEL = "text-embedding-3-small"

    # ==============================================================================
    # Load PDFs and extract text
    # ==============================================================================

    SOURCE_URI = "hf://datasets/Eventual-Inc/sample-files/papers/*.pdf"
    MAX_DOCS = 2

    df = (
        daft.from_glob_path(SOURCE_URI)
        .limit(MAX_DOCS)
        .with_column("pdf_file", file(col("path")))
        .with_column("text", extract_text(col("pdf_file")))
    )

    # ==============================================================================
    # Strategy 1 - Fixed-size chunking (500 chars, 100-char overlap)
    # ==============================================================================

    df_fixed = (
        df
        .with_column("chunks", fixed_size_chunk(col("text"), lit(500), lit(100)))
        .explode("chunks")
        .select(col("path"), col("chunks").alias("chunk_text"))
    )

    print("\n=== Fixed-Size Chunks (500 chars, 100 overlap) ===")
    df_fixed.show(5)

    # ==============================================================================
    # Strategy 2 - Sentence-based chunking via regexp_split
    # ==============================================================================

    df_sentences = (
        df
        .with_column(
            "chunks",
            col("text").regexp_split(r"(?<=[.!?])(?:\s+)(?=[A-Z])"),
        )
        .explode("chunks")
        .where(col("chunks").str.length() > 20)
        .select(col("path"), col("chunks").alias("chunk_text"))
    )

    print("\n=== Sentence-Based Chunks ===")
    df_sentences.show(5)

    # ==============================================================================
    # Strategy 3 - Paragraph chunking (split on double newlines)
    # ==============================================================================

    df_paragraphs = (
        df
        .with_column("chunks", paragraph_chunk(col("text"), lit(80)))
        .explode("chunks")
        .select(col("path"), col("chunks").alias("chunk_text"))
    )

    print("\n=== Paragraph Chunks (min 80 chars) ===")
    df_paragraphs.show(5)

    # ==============================================================================
    # Embed all three strategies
    # ==============================================================================

    df_fixed_emb = df_fixed.with_column(
        "embedding", embed_text(col("chunk_text"), model=EMBEDDING_MODEL),
    )
    df_sentence_emb = df_sentences.with_column(
        "embedding", embed_text(col("chunk_text"), model=EMBEDDING_MODEL),
    )
    df_paragraph_emb = df_paragraphs.with_column(
        "embedding", embed_text(col("chunk_text"), model=EMBEDDING_MODEL),
    )

    # ==============================================================================
    # Compare: query each strategy and rank by cosine distance
    # ==============================================================================

    QUERY = "What is the main contribution of this paper?"

    query = daft.from_pydict({"query_text": [QUERY]}).with_column(
        "query_embedding", embed_text(col("query_text"), model=EMBEDDING_MODEL),
    )

    TOP_K = 3

    for name, df_emb in [
        ("Fixed-Size", df_fixed_emb),
        ("Sentence", df_sentence_emb),
        ("Paragraph", df_paragraph_emb),
    ]:
        ranked = (
            query.join(df_emb, how="cross")
            .with_column(
                "distance",
                cosine_distance(col("query_embedding"), col("embedding")),
            )
            .sort("distance")
            .select("chunk_text", "distance")
            .limit(TOP_K)
        )

        print(f"\n=== Top {TOP_K} matches — {name} Chunking ===")
        ranked.show()
