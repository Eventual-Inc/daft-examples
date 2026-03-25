"""
Job 2: Index Builder

Builds/rebuilds the image index by:
1. Globbing S3 to discover all existing images
2. Extracting metadata from URIs (id, xxhash)
3. Joining with source metadata to enrich the index
4. Writing the complete index to parquet

This job is idempotent and can be run at any time to rebuild the index.
The S3 bucket is the source of truth - the index is a derived view.
"""

import os
import daft
from daft import col
from daft.unity_catalog import UnityCatalog
from daft.io import IOConfig, S3Config, UnityConfig
from dotenv import load_dotenv

load_dotenv()

TABLE = "jaytest-unity.demo.reddit_irl_images_index"

# Configure UnityCatalog
unity = UnityCatalog(
    endpoint=os.getenv("DATABRICKS_ENDPOINT"),
    token=os.getenv("DATABRICKS_TOKEN"),
)


# --------------------------------------------------------------
# Build index from what exists in S3
df = daft.read_deltalake(unity.load_table(TABLE), io_config=unity.to_io_config())

df.write_deltalake(
    unity.load_table(TABLE),
    mode="overwrite",
    configuration={"cache_type": "memory"},
    io_config=unity.to_io_config(),
)
