import os
import daft 
from daft import col
from daft.io import IOConfig, S3Config
from dotenv import load_dotenv
load_dotenv()

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

#df = daft.read_deltalake("s3://daft-public-datasets/the_cauldron/all_images/index.deltalake")
df = daft.read_parquet("https://huggingface.co/datasets/SocialGrep/the-reddit-irl-dataset/resolve/refs%2Fconvert%2Fparquet/posts/train/0000.parquet")
df = df.where(col("url").length() > 0)

df = df.with_column("bytes", daft.functions.download(daft.col("url"), on_error="null"))
df = df.where(col("bytes").not_null())

df = df.with_column("image", col("bytes").decode_image())
df.limit(10).collect()