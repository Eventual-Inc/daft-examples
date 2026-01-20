# /// script
# description = "MinimalRAG Example"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft>=0.6.8", "pymupdf", "pillow", "spacy", "openai", "python-dotenv"]
# ///

import pymupdf
import daft
from daft import DataType
import spacy
from pydantic import BaseModel, Field


@daft.func(
    return_dtype=DataType.struct(
        {
            "page_number": DataType.int32(),
            "page_text": DataType.string(),
            "page_image_bytes": DataType.binary(),
        }
    )
)
def extract_pdf(doc: daft.File):
    with doc.to_tempfile() as temp_file:
        try:
            document = pymupdf.Document(filename=str(temp_file.name), filetype="pdf")

            for page_number, page in enumerate(document):
                text = page.get_text("text")
                yield {
                    "page_number": page_number,
                    "page_text": text,
                    "page_image_bytes": page.get_pixmap().tobytes(),
                }

        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            yield None

        finally:
            temp_file.close()


@daft.cls()
class SpaCyChunkText:
    def __init__(self, model="en_core_web_trf"):
        self.nlp = spacy.load(model)

    @daft.method(
        return_dtype=DataType.list(
            DataType.struct(
                {
                    "sent_id": DataType.int32(),
                    "sent_start": DataType.int32(),
                    "sent_end": DataType.int32(),
                    "sent_text": DataType.string(),
                    "sent_ents": DataType.list(DataType.string()),
                }
            )
        )
    )
    def chunk_text(self, text: list[str]):
        doc = self.nlp(text)
        return [
            {
                "sent_id": i,
                "sent_start": sent.start,
                "sent_end": sent.end,
                "sent_text": sent.text,
                "sent_ents": [ent.text for ent in sent.ents] if sent.ents else [],
            }
            for i, sent in enumerate(doc.sents)
        ]


if __name__ == "__main__":
    from daft import col
    from daft.functions import (
        unnest,
        decode_image,
        embed_text,
        embed_image,
        cosine_distance,
        file,
        prompt,
        resize,
    )

    from dotenv import load_dotenv

    load_dotenv()

    # Define Models
    TEXT_EMBEDDING_MODEL = "text-embedding-3-small"
    IMAGE_EMBEDDING_MODEL = "image-embedding-3-small"

    # Initialize the spaCy model UDF
    ChunkerUDF = SpaCyChunkText(model="en_core_web_trf")

    # Discover, Load, and Extract Text/Images from PDFs
    df = (
        daft.from_glob_path("hf://datasets/Eventual-Inc/sample-files/papers/*.pdf")
        .with_column("pdf_file", file(col("path")))
        .with_column("pdf_pages", extract_pdf(col("pdf_file")))
        .select("path", "pdf_file", unnest(col("pdf_pages")))
        .with_column(
            "page_image",
            decode_image(col("page_image_bytes")).convert_image("RGB").resize(224, 224),
        )
    )

    # Chunk the text with spaCy
    df = df.with_column("pdf_chunks", ChunkerUDF.chunk_text(col("page_text"))).select(
        "path", "pdf_file", unnest(col("pdf_chunks"))
    )

    # Embed the text
    df = df.with_column(
        "text_embedding",
        embed_text(
            daft.col("text"), provider="lm_studio", model="text-embedding-3-small"
        ),
    )

    # Embed the images
    df = df.with_column(
        "image_embedding",
        embed_image(
            daft.col("page_image_bytes"),
            provider="openai",
            model="image-embedding-3-small",
        ),
    )


# Create a query
query = daft.from_pydict({"query_text": ["What is Daft?"]})

# Embed the query
query = query.with_column(
    "query_embedding",
    embed_text(
        daft.col("query_text"), provider="openai", model="text-embedding-3-small"
    ),
)

# Cross join to compare query against all documents
results = query.join(df, how="cross")

# Calculate cosine distance (lower is more similar)
results = results.with_column(
    "distance", cosine_distance(daft.col("query_embedding"), daft.col("text_embedding"))
)

# Sort by distance and show top results
results = results.sort("distance").select("query_text", "text", "distance")
results.show()
