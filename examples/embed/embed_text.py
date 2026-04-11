# /// script
# description = "Embed text"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[openai]>=0.7.8",  "python-dotenv"]
# ///
from dotenv import load_dotenv

import daft
from daft.functions import embed_text

if __name__ == "__main__":
    load_dotenv()

    df = daft.read_huggingface("togethercomputer/RedPajama-Data-1T")

    df = df.with_column(
        "embeddings_1024",
        embed_text(
            daft.col("text"),
            provider="openai",
            model="text-embedding-3-small",
            dimensions=1024,
        ),
    ).with_column(
        "embeddings_512",
        embed_text(
            daft.col("text"),
            provider="openai",  # Ensure OpenAI API key is set
            model="text-embedding-3-small",
            dimensions=512,
        ),
    )

    df.limit(3).show()
