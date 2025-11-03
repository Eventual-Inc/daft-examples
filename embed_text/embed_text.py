# /// script
# description = "Embed text"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft", "sentence-transformers"]
# ///
import daft
from daft.functions import embed_text

df = daft.read_huggingface("togethercomputer/RedPajama-Data-1T")
# Embed Text with Defaults
df = df.with_column(
    "embeddings",
    embed_text(
        daft.col("text"),
        provider="sentence_transformers",
        model="sentence-transformers/all-MiniLM-L6-v2",
    ),
)
df.limit(3).show()
