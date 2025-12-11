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
from daft.functions import format, download
from daft.io import IOConfig, S3Config
import aioboto3


def get_io_config():
    return IOConfig(
        s3=S3Config(
            region_name="us-west-2",
            key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            session_token=os.getenv("AWS_SESSION_TOKEN"),
        )
    )


@daft.cls()
class ImageWriter:
    def __init__(self):
        self.session = aioboto3.Session(
            region_name="us-west-2",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("AWS_SESSION_TOKEN")
        )

    async def write_image(self, image_bytes: bytes, path: str) -> str:
        from io import BytesIO
        from urllib.parse import urlparse

        parsed = urlparse(path)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')

        buffer = BytesIO(image_bytes)
        try:
            async with self.session.client('s3') as s3_client:
                await s3_client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
            return path
        except Exception as e:
            raise ValueError(f"Error writing image to {path}: {e}")


def get_existing_ids_from_s3(dest_uri: str, io_config: IOConfig) -> daft.DataFrame:
    """
    Glob S3 bucket and extract IDs from existing image URIs.
    URI format: reddit-irl_xxhash{hash}_id{id}.png
    """
    try:
        existing_files = daft.from_glob_path(f"{dest_uri}/*.png", io_config=io_config)
        # Extract ID from the path using regex
        existing_ids = existing_files.select(
            col("path").regexp_extract(r"_id(\d+)\.png$", 1).cast(daft.DataType.string()).alias("id")
        ).where(col("id").not_null())
        return existing_ids
    except Exception as e:
        print(f"No existing files found or error globbing: {e}")
        # Return empty dataframe with same schema
        return daft.from_pydict({"id": []})


if __name__ == "__main__":
    SOURCE_URI = "s3://daft-public-datasets/reddit-irl/source"
    DEST_URI = "s3://daft-public-datasets/reddit-irl/all_images"
    LIMIT = os.getenv("LIMIT", None)

    io_config = get_io_config()
    daft.set_planning_config(default_io_config=io_config)

    # Initialize image writer
    image_writer = ImageWriter()

    # Read from the source table
    source_df = daft.read_parquet(f"{SOURCE_URI}/*.parquet")

    # Get existing IDs directly from S3 bucket (source of truth)
    existing_ids = get_existing_ids_from_s3(DEST_URI, io_config)

    # Anti-join: keep only rows from source that DON'T already have images in S3
    source_df = source_df.join(
        existing_ids,
        on="id",
        how="anti",
        strategy="hash",
    )

    # Optionally apply limit early to avoid unnecessary downloads
    if LIMIT:
        source_df = source_df.limit(int(LIMIT))

    df = (
        source_df
        .where(col("url").length() > 0)
        .with_column("bytes", download(daft.col("url"), on_error="null"))
        .where(col("bytes").not_null())
        .with_column("image_xxhash", col("bytes").hash())
        .with_column(
            "image_path",
            format("{}/{}_xxhash{}_id{}.png", lit(DEST_URI), lit("reddit-irl"), col("image_xxhash"), col("id"))
        )
        .with_column("image_written", image_writer.write_image(col("bytes"), col("image_path")))
    )

    # Execute - just write images, no index
    result = df.select("id", "image_written").collect()
    print(f"Wrote {len(result)} images to {DEST_URI}")
