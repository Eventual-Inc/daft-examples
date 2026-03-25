# /// script
# description = "Load metadata (WAT) from Common Crawl - page information without content"
# dependencies = ["daft[aws]", "python-dotenv"]
# ///

import daft
from daft import col

from common import get_common_crawl_io


def main() -> None:
    in_aws, io_config = get_common_crawl_io()

    df = daft.datasets.common_crawl(
        crawl="CC-MAIN-2025-33",
        content="metadata",  # or "wat"
        num_files=1,
        in_aws=in_aws,
        io_config=io_config,
    )

    df.show(1)

    (
        df.select(col("WARC-Identified-Payload-Type"))
        .groupby("WARC-Identified-Payload-Type")
        .agg(col("WARC-Identified-Payload-Type").count().alias("count"))
        .sort("count", desc=True)
        .show(10)
    )

    df.select("WARC-Target-URI", "WARC-Date", "WARC-Identified-Payload-Type").show(5)


if __name__ == "__main__":
    main()
