# /// script
# description = "Generate local text embeddings with a Hugging Face sentence-transformers model"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[transformers]==0.7.19"]
# ///

import daft
from daft.functions import embed_text

texts = daft.from_pydict({"text": ["Daft reads Hugging Face datasets.", "LeRobot is a robotics dataset format."]})

(
    texts.with_column(
        "embedding",
        embed_text(
            daft.col("text"),
            provider="transformers",
            model="sentence-transformers/all-MiniLM-L6-v2",
        ),
    )
    .select("text", "embedding")
    .show()
)
