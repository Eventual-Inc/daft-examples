# /// script
# description = "Embed text"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft", "openai", "python-dotenv"]
# ///

import daft
from daft.functions import embed_text

# Run this script with `uv run embed_text/embed_text.py`
if __name__ == "__main__":

    df = (
        daft.from_pydict({"text": ["Hello World"]})
        .with_column(
            "embeddings", embed_text(daft.col("text"), provider="sentence_transformers", model="sentence-transformers/all-MiniLM-L6-v2")
        )
    )

    df.show()