# /// script
# description = "Embed text from pdfs"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft", "pymupdf", "sentence_transformers", "spacy","pip", "python-dotenv"]
# ///

import daft
from daft import col, DataType
from daft.functions import embed_text, unnest
import io
import pymupdf
import spacy

@daft.func(
    return_dtype=DataType.list(
        DataType.struct(
            {
                "page_number": DataType.uint8(),
                "page_text": DataType.string(),
                "page_image_bytes": DataType.binary(),
            }
        )
    )
)
def extract_pdf(file: daft.File):
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


@daft.cls()
class SpaCyChunkText:
    def __init__(self, model="en_core_web_sm"):
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
    from dotenv import load_dotenv

    TEXT_EMBED_MODEL = "google/paligemma2-3b-mix-448"
    IMAGE_EMBED_MODEL = "google/embeddinggemma-300m"
    MAX_DOCS = 5


    load_dotenv()

    Chunker = SpaCyChunkText("en_core_web_sm")

    # Config
    uri = "hf://datasets/Eventual-Inc/sample-files/papers/*.pdf"

    # Download the spacy model
    spacy.cli.download("en_core_web_sm")

    # Discover and download pdfs
    df = (
        daft.from_glob_path(uri).limit(MAX_DOCS)
        .with_column("documents", col("path").download())

        # Extract text from pdf pages
        .with_column("pages", extract_pdf(col("documents")))
        .explode("pages").select(col("path"), unnest(col("pages")))
        .with_column("images", col("page_image_bytes").decode_image().convert_image("RGB").resize(256, 256))
    )

    df.show()

    df = (
        df
        # Chunk page text into sentences
        .with_column(
            "text_normalized", col("text").normalize(nfd_unicode=True, white_space=True)
        )
        .with_column("sentences", Chunker.chunk_text(col("text_normalized")))
        .explode("sentences")
        .select(col("path"), col("page_number"), unnest(col("sentences")))
        .where(
            col("sent_end") - col("sent_start") > 1
        ) # remove sentences that are too short
        
        # Embed sentences
        .with_column(
            f"text_embed_{MODEL_ID.split('/')[1]}",
            embed_text(
                col("sent_text"), provider="openai", model="text-embedding-3-small"
            ),
        )
    )

    df.write_parquet(".data/embed_text")

    df.show()
