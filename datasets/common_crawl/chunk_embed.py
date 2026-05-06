# /// script
# description = "Chunk Common Crawl text and generate embeddings for semantic search"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[aws]>=0.7.10", "torch", "sentence-transformers", "spacy", "pip", "python-dotenv"]
# ///

import spacy

import daft
from daft import DataType, col
from daft.functions import unnest
from daft.io import IOConfig, S3Config

# ---------------------------
# Parameters
# ---------------------------
CRAWL = "CC-MAIN-2025-33"
NUM_FILES = 2
SPACY_MODEL = "en_core_web_sm"  # Use "en_core_web_trf" for best accuracy (457.4 MB)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUT_DIR = ".data/common_crawl/chunk_embed"


# ---------------------------
# SpaCy Chunker Class
# ---------------------------
SpacyReturnType = DataType.list(
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


@daft.cls()
class SpacyChunker:
    def __init__(self, model="en_core_web_sm"):
        self.nlp = spacy.load(model)

    @daft.method(return_dtype=SpacyReturnType)
    def chunk_text(self, text: str):
        doc = self.nlp(text)
        return [
            {
                "sent_id": i,
                "sent_start": sent.start,
                "sent_end": sent.end,
                "sent_text": sent.text,
                "sent_ents": [ent.text for ent in sent.ents],
            }
            for i, sent in enumerate(doc.sents)
        ]


if __name__ == "__main__":
    from daft.functions import embed_text

    # ---------------------------
    # Common Crawl access (anonymous, public bucket)
    # ---------------------------
    IN_AWS = False
    IOCONFIG = IOConfig(s3=S3Config(anonymous=True, region_name="us-east-1"))

    # Download spaCy model
    spacy.cli.download(SPACY_MODEL)

    # ---------------------------
    # Load Common Crawl Text
    # ---------------------------
    df_warc = daft.datasets.common_crawl(
        crawl=CRAWL,
        content="text",
        num_files=NUM_FILES,
        in_aws=IN_AWS,
        io_config=IOCONFIG,
    ).limit(NUM_FILES)

    # ---------------------------
    # Chunk with spaCy
    # ---------------------------
    spacy_chunk_text = SpacyChunker(model=SPACY_MODEL)

    df_prep = (
        df_warc.with_column("warc_content", col("warc_content").try_decode("utf-8"))
        .drop_null(col("warc_content"))
        .with_column("spacy_results", spacy_chunk_text.chunk_text(text=col("warc_content")))
        .explode("spacy_results")
    )

    # ---------------------------
    # Generate Embeddings
    # ---------------------------
    df_embed = df_prep.with_column(
        "text_embeddings",
        embed_text(
            col("spacy_results")["sent_text"],
            model=EMBEDDING_MODEL,
            provider="transformers",
        ),
    )

    print("\n=== Chunked & Embedded Text ===")
    df_embed.select(unnest(col("spacy_results")), col("text_embeddings")).show(5)

    df_embed.write_parquet(f"{OUT_DIR}/embeddings")
