# /// script
# description = "Load raw WARC data from Common Crawl - full HTTP responses with headers"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[aws]>=0.7.8", "python-dotenv"]
# ///

from common import get_common_crawl_io

import daft


def main() -> None:
    in_aws, io_config = get_common_crawl_io()

    df = daft.datasets.common_crawl(
        crawl="CC-MAIN-2025-33",
        content="raw",  # or "warc"
        num_files=1,
        in_aws=in_aws,
        io_config=io_config,
    )

    df.show(1)

    (
        df.select(daft.col("WARC-Identified-Payload-Type"))
        .groupby("WARC-Identified-Payload-Type")
        .agg(daft.col("WARC-Identified-Payload-Type").count().alias("count"))
        .sort("count", desc=True)
        .show(10)
    )

    df.select("WARC-Target-URI", "WARC-Date", "warc_headers").show(3)


if __name__ == "__main__":
    main()
