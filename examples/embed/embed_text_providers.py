# /// script
# description = "Embed text"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[openai]>=0.7.6", "sentence-transformers", "python-dotenv"]
# ///
from dotenv import load_dotenv

import daft
from daft.functions import embed_text

if __name__ == "__main__":
    load_dotenv()

    df = daft.read_huggingface("togethercomputer/RedPajama-Data-1T")

    df = (
        df.with_column(
            "embeddings_transformers",
            embed_text(
                daft.col("text"),
                provider="transformers",
                model="sentence-transformers/all-MiniLM-L6-v2",
            ),
        )
        .with_column(
            "embeddings_openai",
            embed_text(
                daft.col("text"),
                provider="openai",  # Ensure OpenAI API key is set
                model="text-embedding-3-small",
            ),
        )
        .with_column(
            "embeddings_lm_studio",
            embed_text(
                daft.col("text"),
                provider="lm_studio",  # Ensure LM Studio Server is running with the model loaded
                model="text-embedding-nomic-embed-text-v1.5",
            ),
        )
    )

    df.limit(3).show()
