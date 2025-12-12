"""
Job 1: Image Ingestion

Downloads images from source URLs and writes them to S3.
Idempotent - checks if image already exists in S3 before downloading.

The S3 bucket is the source of truth. URIs encode the id and xxhash:
  s3://bucket/path/reddit-irl_xxhash{hash}_id{id}.png
"""

import os
import daft
from daft import col, lit
from daft.functions import format, download, upload
from daft.io import IOConfig, S3Config
from dotenv import load_dotenv

# --------------------------------------------------------------
# Configuration
load_dotenv()

daft.set_planning_config(
    default_io_config=IOConfig(
        s3=S3Config(
            region_name="us-west-2",
            key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            session_token=os.getenv("AWS_SESSION_TOKEN"),
        )
    )
)

SOURCE_URI = "s3://daft-public-datasets/reddit-irl/source"
DEST_URI = "s3://daft-public-datasets/reddit-irl/all_images"
LIMIT = os.getenv("LIMIT", None)

# --------------------------------------------------------------
# PIPILINE
source_df = daft.read_parquet(f"{SOURCE_URI}/*.parquet")

# Glob existing images from s3
existing_files = (
    daft.from_glob_path(f"{DEST_URI}/*.png")
    .with_column("id", col("path").regexp_extract(r"_id([a-zA-Z0-9]+)\.png$", 1))
)

# Filter out rows that have processed already
df = source_df.join(
    existing_files.select("id"),
    on="id",
    how="anti",
    strategy="hash",
)

# Download images from urls and upload to s3
df = (
    df
    .where(col("url").length() > 0)
    .with_column("bytes", download(daft.col("url"), on_error="null"))
    .where(col("bytes").not_null())
    .with_column("image_xxhash", col("bytes").hash())
    .with_column(
        "image_path",
        format("{}/{}_xxhash{}_id{}.png", lit(DEST_URI), lit("reddit-irl"), col("image_xxhash"), col("id"))
    )
    .with_column("image_written", upload(col("bytes"), location=col("image_path"), max_connections=64))
)


if LIMIT:
    df = df.limit(int(LIMIT))

result_df = df.select("id", "image_written").collect()

print(f"Wrote {result_df.count_rows()} images to {DEST_URI}")
