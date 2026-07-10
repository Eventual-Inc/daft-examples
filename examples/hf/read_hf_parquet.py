# /// script
# description = "Read a Xet-backed Hugging Face Parquet file with Daft"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[huggingface]==0.7.19"]
# ///

import daft

# Read a known Xet-backed Parquet shard directly through the hf:// protocol.
daft.read_parquet("hf://datasets/google-research-datasets/mbpp/full/train-00000-of-00001.parquet").limit(2).show()
