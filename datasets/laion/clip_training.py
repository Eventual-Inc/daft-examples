# /// script
# description = "Prepare LAION data for CLIP-style training"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[aws,openai]>=0.7.10", "python-dotenv"]
# ///

import os

from dotenv import load_dotenv

import daft
from daft import col
from daft.functions import embed_text
from daft.io import IOConfig, S3Config

if __name__ == "__main__":
    load_dotenv()
    io_config = IOConfig(s3=S3Config(anonymous=True, region_name="us-east-1"))

    # Load LAION metadata
    df = daft.read_parquet(
        "s3://daft-public-data/tutorials/laion-parquet/train-00000-of-00001-*.parquet",
        io_config=io_config,
    )

    # Filter high quality data for training
    df_filtered = df.where(
        (col("AESTHETIC_SCORE") > 6.0)
        & (col("WIDTH") >= 256)
        & (col("HEIGHT") >= 256)
        & (col("WIDTH") <= 2048)
        & (col("HEIGHT") <= 2048)
        & (col("TEXT").length() > 10)
    )

    print(f"Filtered {df_filtered.count_rows()} training examples from {df.count_rows()} total")
    df_filtered = df_filtered.with_column(
        "normalized_caption",
        col("TEXT").normalize(lowercase=True, remove_punct=False),
    )
    df_filtered.select("TEXT", "normalized_caption", "AESTHETIC_SCORE").show(5)

    if os.environ.get("OPENAI_API_KEY"):
        print("Text embeddings:")
        daft.set_provider("openai", api_key=os.environ.get("OPENAI_API_KEY"))
        df_sample = df_filtered.limit(5)
        df_embedded = df_sample.with_column(
            "text_embedding",
            embed_text(
                col("normalized_caption"),
                model="text-embedding-3-small",
            ),
        )
        df_embedded.select("TEXT", "text_embedding").show(5)
