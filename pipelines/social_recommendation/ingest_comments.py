# /// script
# description = "Ingest Reddit comments to S3 as parquet"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[huggingface]>=0.7.6", "python-dotenv"]
# ///
import os

from dotenv import load_dotenv

import daft
from daft.io import IOConfig, S3Config

if __name__ == "__main__":
    load_dotenv()

    # --------------------------------------------------------------
    # Configuration
    HF_DATASET = [
        "https://huggingface.co/api/datasets/SocialGrep/the-reddit-irl-dataset/parquet/comments/train/0.parquet",
        "https://huggingface.co/api/datasets/SocialGrep/the-reddit-irl-dataset/parquet/comments/train/1.parquet",
        "https://huggingface.co/api/datasets/SocialGrep/the-reddit-irl-dataset/parquet/comments/train/2.parquet",
        "https://huggingface.co/api/datasets/SocialGrep/the-reddit-irl-dataset/parquet/comments/train/3.parquet",
        "https://huggingface.co/api/datasets/SocialGrep/the-reddit-irl-dataset/parquet/comments/train/4.parquet",
        "https://huggingface.co/api/datasets/SocialGrep/the-reddit-irl-dataset/parquet/comments/train/5.parquet",
    ]
    COMMENTS_URI = "s3://daft-public-datasets/reddit-irl/comments.parquet"

    daft.set_planning_config(
        default_io_config=IOConfig(
            s3=S3Config(
                region_name="us-west-2",
                key_id=os.environ["AWS_ACCESS_KEY_ID"],
                access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            )
        )
    )

    # --------------------------------------------------------------
    # Pipeline

    df = daft.read_parquet(HF_DATASET)

    df = df.select(
        "id",
        daft.col("id").hash().cast(daft.DataType.uint64()).alias("id_xxhash"),
        daft.col("created_utc").cast(daft.DataType.timestamp("ms", "UTC")),
        daft.col("subreddit.id").alias("subreddit_id"),
        daft.col("subreddit.name").alias("subreddit_name"),
        daft.col("subreddit.nsfw").alias("subreddit_nsfw"),
        "permalink",
        "body",
        "sentiment",
        "score",
    )

    df.write_parquet(COMMENTS_URI, write_mode="overwrite")

    print(f"Wrote {df.count_rows()} rows to {COMMENTS_URI}")
