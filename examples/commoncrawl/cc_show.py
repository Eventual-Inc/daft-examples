# /// script
# description = "Show top MIME types from Common Crawl"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft>=0.7.10", "python-dotenv"]
# ///
import daft
from daft.io import IOConfig, S3Config

if __name__ == "__main__":
    IN_AWS = False
    IOCONFIG = IOConfig(s3=S3Config(anonymous=True, region_name="us-east-1"))

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
