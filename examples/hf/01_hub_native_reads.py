# /// script
# description = "Read Hugging Face datasets, hf:// Parquet, and Storage Bucket files with Daft"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[huggingface]==0.7.19"]
# ///

import daft


def main() -> None:
    # Read an entire public Hub dataset through Daft's native dataset reader.
    daft.read_huggingface("huggingface/documentation-images").limit(2).show()

    # Read a known Xet-backed Parquet shard directly through the hf:// protocol.
    daft.read_parquet("hf://datasets/google-research-datasets/mbpp/full/train-00000-of-00001.parquet").limit(2).show()

    # Build lazy File references from a public Hugging Face Storage Bucket.
    bucket_file = (
        "hf://buckets/commoncrawl/commoncrawl/crawl-data/CC-MAIN-2026-17/segments/"
        "1775805908305.14/warc/CC-MAIN-20260410081153-20260410111153-00000.warc.gz"
    )
    daft.from_files(bucket_file).show()


if __name__ == "__main__":
    main()
