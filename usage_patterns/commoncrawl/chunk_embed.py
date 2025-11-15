# /// script
# dependencies = ["daft>=0.6.13", "torch", "sentence-transformers", "spacy", "pip", "python-dotenv"]
# ///

import daft
from daft import DataType
from daft.functions import unnest
import spacy

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

    @daft.method(
        return_dtype=SpacyReturnType
    )
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
    from dotenv import load_dotenv
    import os

    import daft
    from daft import col
    from daft.io import IOConfig, S3Config
    from daft.functions import embed_text, decode

    SOURCE_URI = "s3://commoncrawl/crawl-data/CC-MAIN-2025-33/segments/1754151579063.98/warc/CC-MAIN-20250815204238-20250815234238-00999.warc.gz"
    DEST_URI = ".data/common_crawl/chunk_embed"
    SPACY_MODEL = "en_core_web_sm"  # use "en_core_web_trf" for best accuracy (457.4 MB)
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    NUM_FILES = 2

    # Authenticate with AWS
    load_dotenv()

    if os.environ.get("AWS_ACCESS_KEY_ID"):
        IN_AWS = True
        IOCONFIG = IOConfig(
            s3=S3Config(
                region_name="us-east-1",
                requester_pays=True,
                key_id=os.environ["AWS_ACCESS_KEY_ID"],
                access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                anonymous=False,
            )
        )
    else:
        IN_AWS = False
        IOCONFIG = None

    # Read Preprocessed Text from Common Crawl WET
    df_warc = daft.datasets.common_crawl(
        "CC-MAIN-2025-33",
        content="text",
        num_files=NUM_FILES,
        in_aws=IN_AWS,
        io_config=IOCONFIG,  # Apply Creds
    ).limit(NUM_FILES)

    # Download the spaCy model
    spacy.cli.download(SPACY_MODEL)

    # Initialize the spaCy chunker
    spacy_chunk_text = SpacyChunker(model=SPACY_MODEL)

    # Chunk Text into sentences with spaCy
    df_prep = (
        df_warc.with_column(
            "warc_content",
            decode(col("warc_content"), "utf-8"),
        )
        .with_column(
            "spacy_results",
            spacy_chunk_text.chunk_text(text=col("warc_content")),
        )
        .explode("spacy_results")
    )

    # Embed Sentences
    df_embed = df_prep.with_column(
        f"text_embeddings",
        embed_text(
            col("spacy_results")["sent_text"],
            model="sentence-transformers/all-MiniLM-L6-v2",
            provider="transformers",
        ),
    )

    df_embed.select(unnest(col("spacy_results")), col("text_embeddings")).show(format="fancy", max_width=20)
