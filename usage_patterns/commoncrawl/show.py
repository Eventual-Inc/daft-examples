# /// script
# dependencies = ["daft", "python-dotenv"]
# ///
from dotenv import load_dotenv
import os

import daft
from daft.io import IOConfig, S3Config

load_dotenv()

if os.environ.get("AWS_ACCESS_KEY_ID"):
    IN_AWS = True
    IOCONFIG = IOConfig(
        s3=S3Config(
            region_name="us-east-1",
            requester_pays=True,
            key_id=os.environ["AWS_ACCESS_KEY_ID"],
            access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            anonymous=False,
        )
    )
else:
    IN_AWS = False
    IOCONFIG = None

# Find top MIME types
df = (
    daft.datasets.common_crawl(
        "CC-MAIN-2025-33",
        segment="1754151279521.11",
        num_files=2,
        in_aws=IN_AWS,
        io_config=IOCONFIG,
    )
    .select(daft.col("WARC-Identified-Payload-Type"))
    .groupby("WARC-Identified-Payload-Type")
    .agg(daft.col("WARC-Identified-Payload-Type").count().alias("count"))
    .sort("count", desc=True)
)

df.show()
