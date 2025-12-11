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


def get_io_config():
    return IOConfig(
        s3=S3Config(
            region_name="us-west-2",
            key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            session_token=os.getenv("AWS_SESSION_TOKEN"),
        )
    )


def build_index_from_s3(dest_uri: str, io_config: IOConfig) -> daft.DataFrame:
    """
    Glob S3 bucket and extract metadata from existing image URIs.
    URI format: reddit-irl_xxhash{hash}_id{id}.png
    """
    files_df = daft.from_glob_path(f"{dest_uri}/*.png", io_config=io_config)

    # Extract metadata from path and include file info from glob
    index_df = files_df.select(
        col("path").alias("image_s3_uri"),
        col("size").alias("image_size_bytes"),
        col("path").regexp_extract(r"_id(\d+)\.png$", 1).cast(daft.DataType.string()).alias("id"),
        col("path").regexp_extract(r"_xxhash(-?\d+)_", 1).cast(daft.DataType.int64()).alias("image_xxhash"),
    ).where(col("id").not_null())

    return index_df


if __name__ == "__main__":
    SOURCE_URI = "s3://daft-public-datasets/reddit-irl/source"
    DEST_URI = "s3://daft-public-datasets/reddit-irl/all_images"
    INDEX_URI = f"{DEST_URI}/_reddit_irl_images_index.parquet"

    io_config = get_io_config()
    daft.set_planning_config(default_io_config=io_config)

    # Build index from what exists in S3
    print(f"Globbing {DEST_URI} to discover existing images...")
    s3_index = build_index_from_s3(DEST_URI, io_config)

    # Read source metadata to enrich the index
    print(f"Reading source metadata from {SOURCE_URI}...")
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

    # Join S3 file info with source metadata
    print("Joining S3 index with source metadata...")
    full_index = s3_index.join(
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
        "image_s3_uri",
        "image_size_bytes",
        "image_xxhash",
    ]

    # Write the full index (overwrite mode - this is a full rebuild)
    print(f"Writing index to {INDEX_URI}...")
    final_df = full_index.select(*final_columns)
    final_df.write_parquet(INDEX_URI, write_mode="overwrite")

    # --------------------------------------------------------------
    # Build and display statistics ---------------------------------
    print("\n" + "=" * 60)
    print("INDEX STATISTICS")
    print("=" * 60)

    # Read back the written index for accurate stats
    index_df = daft.read_parquet(INDEX_URI)

    # Overall stats
    stats = index_df.agg(
        daft.functions.count(col("id")).alias("total_images"),
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

    stats_dict = stats.to_pydict()
    total_images = stats_dict["total_images"][0]
    total_size_bytes = stats_dict["total_size_bytes"][0]
    total_size_gb = total_size_bytes / (1024 ** 3) if total_size_bytes else 0
    avg_size_kb = stats_dict["avg_size_bytes"][0] / 1024 if stats_dict["avg_size_bytes"][0] else 0

    print(f"\nTotal Images:        {total_images:,}")
    print(f"Total Size:          {total_size_gb:.2f} GB")
    print(f"Avg Image Size:      {avg_size_kb:.1f} KB")
    print(f"Min Image Size:      {stats_dict['min_size_bytes'][0] / 1024:.1f} KB")
    print(f"Max Image Size:      {stats_dict['max_size_bytes'][0] / 1024:.1f} KB")
    print(f"\nUnique Subreddits:   {stats_dict['unique_subreddits'][0]:,}")
    print(f"Unique Domains:      {stats_dict['unique_domains'][0]:,}")
    print(f"\nScore Range:         {stats_dict['min_score'][0]:,} - {stats_dict['max_score'][0]:,}")
    print(f"Avg Score:           {stats_dict['avg_score'][0]:.1f}")

    # Top subreddits by image count
    print("\n" + "-" * 60)
    print("TOP 10 SUBREDDITS BY IMAGE COUNT")
    print("-" * 60)
    top_subreddits = (
        index_df
        .groupby("subreddit_name")
        .agg(
            daft.functions.count(col("id")).alias("image_count"),
            col("image_size_bytes").sum().alias("total_size_bytes"),
        )
        .sort("image_count", desc=True)
        .limit(10)
    )
    top_subreddits.show()

    # Top domains
    print("\n" + "-" * 60)
    print("TOP 10 DOMAINS")
    print("-" * 60)
    top_domains = (
        index_df
        .groupby("domain")
        .agg(daft.functions.count(col("id")).alias("image_count"))
        .sort("image_count", desc=True)
        .limit(10)
    )
    top_domains.show()

    print("\n" + "=" * 60)
    print(f"Index built successfully with {total_images:,} images")
    print("=" * 60)
