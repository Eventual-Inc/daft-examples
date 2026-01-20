# /// script
# description = "Load plain text (WET) from Common Crawl - extracted text from web pages"
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

# Load plain text (WET files)
df = daft.datasets.common_crawl(
    crawl="CC-MAIN-2025-33",
    content="text",  # or "wet"
    num_files=1,
    in_aws=IN_AWS,
    io_config=IOCONFIG,
)

print("\n=== WET Schema ===")
df.show(1)

print("\n=== Decoded Text Samples ===")
df.with_column("text", col("warc_content").try_decode("utf-8")).select(
    "WARC-Target-URI", "WARC-Date", "text"
).show(3)

print("\n=== Text Length Distribution ===")
df.with_column("text_length", col("warc_content").length()).select(
    "text_length"
).describe()
