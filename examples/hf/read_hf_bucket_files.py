# /// script
# description = "Create lazy Daft File references from a Hugging Face Storage Bucket"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[huggingface]==0.7.19"]
# ///

import daft

BUCKET_FILE = (
    "hf://buckets/commoncrawl/commoncrawl/crawl-data/CC-MAIN-2026-17/segments/"
    "1775805908305.14/warc/CC-MAIN-20260410081153-20260410111153-00000.warc.gz"
)

# Build a lazy File reference without downloading the WARC object.
daft.from_files(BUCKET_FILE).show()
