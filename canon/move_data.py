import os
import daft 
from daft.io import S3Config, IOConfig

S3_URI = "s3://daft-public-datasets/the_cauldron/"
CAULDRON_SUBSET = "ai2d"


df = daft.read_parquet(f"hf://datasets/HuggingFaceM4/the_cauldron/{CAULDRON_SUBSET}/**/*.parquet")
df.write_parquet(
    f"{S3_URI}/original/{CAULDRON_SUBSET}", 
    io_config=IOConfig(
        s3 = S3Config(
            region_name="us-west-2",
            aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
        )
    )
)
