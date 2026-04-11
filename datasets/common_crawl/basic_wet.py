# /// script
# description = "Load plain text (WET) from Common Crawl - extracted text from web pages"
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
        content="text",  # or "wet"
        num_files=1,
        in_aws=in_aws,
        io_config=io_config,
    )

    print("\n=== WET Schema ===")
    df.show(1)

    print("\n=== Decoded Text Samples ===")
    (
        df.with_column("text", col("warc_content").try_decode("utf-8"))
        .select("WARC-Target-URI", "WARC-Date", "text")
        .show(3)
    )

    print("\n=== Text Length Distribution ===")
    df.with_column("text_length", col("warc_content").length()).select("text_length").describe()


if __name__ == "__main__":
    main()
