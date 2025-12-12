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
from daft.io import IOConfig, S3Config
from dotenv import load_dotenv
load_dotenv()

SOURCE_URI = "s3://daft-public-datasets/reddit-irl/source"
IMAGES_URI = "s3://daft-public-datasets/reddit-irl/all_images"
IMAGES_INDEX_URI = f"{IMAGES_URI}/_reddit_irl_images_index.parquet"

daft.set_planning_config(default_io_config=IOConfig(
    s3=S3Config(
        region_name="us-west-2",
        key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        session_token=os.getenv("AWS_SESSION_TOKEN"),
    )
))

# --------------------------------------------------------------
# Build index from what exists in S3
files_df = daft.from_glob_path(f"{IMAGES_URI}/*.png")

# Extract metadata from path and include file info from glob
index_df = (
    files_df
    .with_column("id", col("path").regexp_extract(r"_id([a-zA-Z0-9]+)\.png$", 1))
    .with_column(
        "image_xxhash", 
        col("path").regexp_extract(r"_xxhash([0-9]+)_id", 1).cast(daft.DataType.uint64())
    )
)

# --------------------------------------------------------------
# Read source metadata to enrich the index
source_df = daft.read_parquet(f"{SOURCE_URI}/*.parquet")

# Select source columns we want in the index
source_metadata = source_df.select(
    "id",
    "type",
    col("subreddit.id").alias("subreddit_id"),
    col("subreddit.name").alias("subreddit_name"),
    col("subreddit.nsfw").alias("subreddit_nsfw"),
    "title",
    col("created_utc").cast(daft.DataType.timestamp("ms", "UTC")),
    "permalink",
    "domain",
    "score",
    col("url").alias("image_url"),
)

# --------------------------------------------------------------
# Join S3 file info with source metadata
full_index = index_df.join(
    source_metadata,
    on="id",
    how="inner",  # Only keep rows that exist in both
    strategy="hash",
)

# Reorder columns for readability
final_columns = [
    "id",
    "type",
    "subreddit_id",
    "subreddit_name",
    "subreddit_nsfw",
    "title",
    "created_utc",
    "permalink",
    "domain",
    "score",
    "image_url",
    col("path").alias("image_s3_uri"),
    col("size").alias("image_size_bytes"),
    "image_xxhash",
]

# Write the full index (overwrite mode - this is a full rebuild)
final_df = full_index.select(*final_columns)
final_df.write_parquet(IMAGES_INDEX_URI, write_mode="overwrite")

# --------------------------------------------------------------
# Build and display statistics 
index_df = daft.read_parquet(f"{IMAGES_INDEX_URI}/*.parquet")

# Overall stats
stats = index_df.agg(
    col("id").count().alias("total_images"),
    col("image_size_bytes").sum().alias("total_size_bytes"),
    col("image_size_bytes").mean().alias("avg_size_bytes"),
    col("image_size_bytes").min().alias("min_size_bytes"),
    col("image_size_bytes").max().alias("max_size_bytes"),
    col("score").mean().alias("avg_score"),
    col("score").min().alias("min_score"),
    col("score").max().alias("max_score"),
    daft.functions.count_distinct(col("subreddit_name")).alias("unique_subreddits"),
    daft.functions.count_distinct(col("domain")).alias("unique_domains"),
).collect()

print("\n" + "=" * 60)
print("INDEX STATISTICS")
stats.show()

print("\n" + "=" * 60)
print(f"Number of rows in index: {index_df.select("id").count_rows()}")
