import os
import daft 
from daft import col, lit
from daft.functions import format, download, monotonically_increasing_id
from daft.io import IOConfig, S3Config
import numpy as np
from PIL import Image
import aioboto3

from daft.unity_catalog import UnityCatalog
from daft.catalog import Table
#from dotenv import load_dotenv
#load_dotenv()

@daft.cls()
class ImageWriter:
    def __init__(self):
        self.session = aioboto3.Session(
            region_name="us-west-2",
            aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("S3_SESSION_TOKEN")
        )

    async def write_images(self, image_bytes: bytes, path: str) -> str:
        from io import BytesIO
        from urllib.parse import urlparse
        
        # Parse S3 path
        parsed = urlparse(path)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        
        # Write image to bytes buffer
        buffer = BytesIO(image_bytes)
        try:
            async with self.session.client('s3') as s3_client:
                await s3_client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
            return path
        except Exception as e:
            raise ValueError(f"Error writing image to {path}: {e}")

if __name__ == "__main__":
    SOURCE_URI = "s3://daft-public-datasets/reddit-irl/source"
    DEST_URI = "s3://daft-public-datasets/reddit-irl/all_images"
    LIMIT = os.getenv("LIMIT", None)
    MONO_ID = os.getenv("MONO_ID", None)
    WRITE_MODE = os.getenv("WRITE_MODE", "append")

    columns = [
        "type", 
        "id", 
        "subreddit.id", 
        "subreddit.name", 
        "subreddit.nsfw",  
        "title", 
        "created_utc", 
        "permalink", 
        "domain", 
        "score",
        "image_xxhash",
        col("url").alias("image_url"), 
        col("image_written").alias("image_s3_uri")
    ]

    daft.set_planning_config(
        default_io_config=IOConfig(
            s3=S3Config(
                region_name="us-west-2",
                key_id=os.getenv("S3_ACCESS_KEY_ID"),
                access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
                session_token=os.getenv("S3_SESSION_TOKEN"),
            )
        )
    )

    image_writer = ImageWriter()

    # Pipeline
    df = daft.read_parquet(f"{SOURCE_URI}/*.parquet")

    df = (
        df
        .with_column("created_utc", col("created_utc").cast(daft.DataType.timestamp("ms", "UTC")))
        .where(col("url").length() > 0) 
        .with_column("bytes", download(daft.col("url"), on_error="null"))
        .where(col("bytes").not_null())
    )

    if MONO_ID is not None:
        df = df.with_column("mono_id", monotonically_increasing_id()).where(col("mono_id") > int(MONO_ID))

    df = (
        df
        .with_column("image_xxhash", col("bytes").hash())
        .with_column("image_path", format("{}/{}_xxhash{}_id{}.png", lit(DEST_URI), lit("reddit-irl"), col("image_xxhash"), col("id")))
        .with_column("image_written", image_writer.write_images(col("bytes"), col("image_path")))
    )
    
    # Optionally apply limit
    if LIMIT:
        df = df.limit(int(LIMIT))
    
    # Write
    df.select(*columns).write_parquet(f"{DEST_URI}/_reddit_irl_images_index.parquet", write_mode=WRITE_MODE)