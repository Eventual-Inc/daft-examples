# /// script
# dependencies = ["daft", "python-dotenv"]
# ///

if __name__ == "__main__":  # Create a dataframe over a crawl.
    from dotenv import load_dotenv
    import os

    import daft
    from daft.io import IOConfig, S3Config

    # Authenticate with AWS
    load_dotenv()
    s3_config = S3Config(
        region_name="us-east-1",
        requester_pays=True,
        key_id=os.environ["AWS_ACCESS_KEY_ID"],
        access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        anonymous=False,
    )

    # Find top MIME types.
    (
        daft.datasets.common_crawl(
            "CC-MAIN-2025-33",
            segment="1754151279521.11",
            num_files=10,
            io_config=IOConfig(s3=s3_config),
        )
        .select(daft.col("WARC-Identified-Payload-Type"))
        .groupby("WARC-Identified-Payload-Type")
        .agg(daft.col("WARC-Identified-Payload-Type").count().alias("count"))
        .sort("count", desc=True)
        .show()
    )
