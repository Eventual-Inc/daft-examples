# /// script
# description = "Load Reddit IRL posts with images"
# dependencies = ["daft[aws]"]
# ///

import daft
from daft import col

# Load Reddit posts from source (includes image URLs)
df = daft.read_parquet("s3://daft-public-datasets/reddit-irl/source/**/*.parquet")

print("\n=== Schema ===")
df.show(1)

print("\n=== Posts with Images ===")
posts_with_images = df.where(~col("image_url").is_null())
posts_with_images.select("title", "subreddit", "image_url", "url").show(
    5, max_col_width=50
)

print("\n=== Image URL Distribution by Subreddit ===")
image_counts = (
    posts_with_images.groupby("subreddit")
    .agg(col("id").count().alias("image_count"))
    .sort("image_count", desc=True)
)
image_counts.show(10)

print("\n=== Sample Post Titles ===")
df.select("title", "subreddit", "url").show(5, max_col_width=50)
