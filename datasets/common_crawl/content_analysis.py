# /// script
# description = "Analyze Common Crawl content distribution - MIME types, domains, and statistics"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[aws]>=0.7.8", "python-dotenv"]
# ///

from common import get_common_crawl_io

import daft
from daft import col


def main() -> None:
    in_aws, io_config = get_common_crawl_io()

    df = daft.datasets.common_crawl(
        crawl="CC-MAIN-2025-33",
        segment="1754151279521.11",
        num_files=2,
        in_aws=in_aws,
        io_config=io_config,
    )

    print("\n=== Top MIME Types ===")
    mime_dist = (
        df.select(col("WARC-Identified-Payload-Type"))
        .groupby("WARC-Identified-Payload-Type")
        .agg(col("WARC-Identified-Payload-Type").count().alias("count"))
        .sort("count", desc=True)
    )
    mime_dist.show(10)

    print("\n=== Content Size Statistics ===")
    df.with_column("content_size", col("warc_content").length()).select("content_size").describe()

    print("\n=== Sample URLs by Content Type ===")
    (
        df.select("WARC-Target-URI", "WARC-Identified-Payload-Type", "WARC-Date")
        .where(col("WARC-Identified-Payload-Type") == "text/html")
        .show(5)
    )


if __name__ == "__main__":
    main()
