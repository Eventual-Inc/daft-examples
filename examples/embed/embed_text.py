# /// script
# description = "Embed text"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft[openai]>=0.7.5",  "python-dotenv"]
# ///
import daft
from daft.functions import embed_text
from dotenv import load_dotenv


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
