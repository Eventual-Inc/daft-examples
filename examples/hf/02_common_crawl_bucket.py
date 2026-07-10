# /// script
# description = "Read a small Common Crawl WET sample through the Hugging Face Storage Bucket"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[huggingface]==0.7.19"]
# ///

import daft
from daft.functions import try_decode


def main() -> None:
    # `source="hf"` selects hf://buckets/commoncrawl/commoncrawl explicitly.
    # A single WET file is still substantial, so keep this example intentionally small.
    crawl = daft.datasets.common_crawl(
        "CC-MAIN-2026-25",
        content="text",
        num_files=1,
        source="hf",
    )

    (
        crawl.with_column("text", try_decode(daft.col("warc_content"), charset="utf-8"))
        .where(daft.col("text").not_null())
        .select("WARC-Target-URI", "text")
        .limit(3)
        .show()
    )


if __name__ == "__main__":
    main()
