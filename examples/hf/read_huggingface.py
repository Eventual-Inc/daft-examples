# /// script
# description = "Read a public Hugging Face dataset with Daft"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[huggingface]==0.7.19"]
# ///

import daft

# Read an entire public Hub dataset through Daft's native dataset reader.
daft.read_huggingface("huggingface/documentation-images").limit(2).show()
