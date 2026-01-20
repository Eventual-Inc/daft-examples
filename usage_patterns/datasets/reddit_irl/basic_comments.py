# /// script
# description = "Load and analyze Reddit IRL comments dataset"
# dependencies = ["daft[aws]"]
# ///

import daft
from daft import col

# Load Reddit comments from public dataset
df = daft.read_parquet("s3://daft-public-datasets/reddit-irl/comments.parquet")

print("\n=== Schema ===")
df.show(1)

print("\n=== Top Subreddits by Comment Count ===")
subreddit_counts = (
    df.groupby("subreddit")
    .agg(col("id").count().alias("comment_count"))
    .sort("comment_count", desc=True)
)
subreddit_counts.show(10)

print("\n=== Sample Comments ===")
df.select("subreddit", "author", "body", "score").show(5)

print("\n=== Comment Length Distribution ===")
df.with_column("text_length", col("body").length()).select("text_length").describe()
