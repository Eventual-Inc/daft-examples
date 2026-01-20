# /// script
# description = "Load metadata (WAT) from Common Crawl - page information without content"
# dependencies = ["daft[aws]", "python-dotenv"]
# ///

import os
from dotenv import load_dotenv
import daft
from daft import col
from daft.io import IOConfig, S3Config

load_dotenv()

# Configure AWS access if credentials available
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

# Load metadata (WAT files)
df = daft.datasets.common_crawl(
    crawl="CC-MAIN-2025-33",
    content="metadata",  # or "wat"
    num_files=1,
    in_aws=IN_AWS,
    io_config=IOCONFIG,
)

print("\n=== WAT Schema ===")
df.show(1)

print("\n=== MIME Type Analysis ===")
df.select(col("WARC-Identified-Payload-Type")).groupby(
    "WARC-Identified-Payload-Type"
).agg(col("WARC-Identified-Payload-Type").count().alias("count")).sort(
    "count", desc=True
).show(10)

print("\n=== Metadata Samples ===")
df.select("WARC-Target-URI", "WARC-Date", "WARC-Identified-Payload-Type").show(5)
