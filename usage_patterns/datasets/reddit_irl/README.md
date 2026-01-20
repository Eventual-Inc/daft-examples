# Reddit IRL Dataset

The Reddit IRL dataset contains social media data from Reddit including comments, images, and pre-computed embeddings.

## Dataset Location

**S3 Bucket:** `s3://daft-public-datasets/reddit-irl/`

**Authentication:** Public bucket (no AWS credentials required)

## Dataset Structure

```
reddit-irl/
├── source/                    # Raw source data with images
├── comments.parquet          # Reddit comments
├── all_images/               # Extracted images from posts
└── embeddings/               # Pre-computed embeddings
    └── Qwen--Qwen3-Embedding-0.6B.parquet/
```

## Schema

### Comments (`comments.parquet`)
```
id: String                    # Comment ID
body: String                  # Comment text
author: String                # Username
subreddit: String            # Subreddit name
created_utc: Int64           # Unix timestamp
score: Int64                 # Upvote score
```

### Source (`source/**/*.parquet`)
```
id: String                    # Post ID
title: String                 # Post title
selftext: String             # Post body text
subreddit: String            # Subreddit name
image_url: String            # Image URL if present
url: String                  # Post URL
```

## Examples

| File | Description | Use Case |
|------|-------------|----------|
| **[basic_comments.py](./basic_comments.py)** | Load and analyze Reddit comments | Text analysis, sentiment |
| **[basic_images.py](./basic_images.py)** | Load Reddit post images | Computer vision, multimodal |
| **[embeddings.py](./embeddings.py)** | Work with pre-computed embeddings | Semantic search, clustering |

## Quick Start

```python
import daft

# Load comments
df = daft.read_parquet("s3://daft-public-datasets/reddit-irl/comments.parquet")
df.show(5)

# Load images
df = daft.read_parquet("s3://daft-public-datasets/reddit-irl/source/**/*.parquet")
df.show(5)
```

## Common Use Cases

### 1. Comment Analysis
```python
# Top subreddits by comment count
df = daft.read_parquet("s3://daft-public-datasets/reddit-irl/comments.parquet")
df.groupby("subreddit").agg(
    daft.col("id").count().alias("comment_count")
).sort("comment_count", desc=True).show(10)
```

### 2. Sentiment Analysis
```python
from daft.functions import prompt

df = daft.read_parquet("s3://daft-public-datasets/reddit-irl/comments.parquet").limit(100)
df = df.with_column(
    "sentiment",
    prompt(
        daft.col("body"),
        system_message="Classify sentiment as positive, negative, or neutral",
        model="gpt-4o-mini",
        provider="openai"
    )
)
df.show(5)
```

### 3. Image Understanding
```python
from daft.functions import file, prompt

df = daft.read_parquet("s3://daft-public-datasets/reddit-irl/source/**/*.parquet")
df = df.where(~daft.col("image_url").is_null())
df = df.with_column("image", file(daft.col("image_url")))
df = df.with_column(
    "description",
    prompt(
        daft.col("image"),
        system_message="Describe this image in one sentence",
        model="gpt-4o-mini",
        provider="openai"
    )
)
df.show(5)
```

## Performance Tips

1. **Use filters early**: Reddit IRL is large - filter by subreddit or date first
2. **Limit during development**: Use `.limit(100)` for testing queries
3. **Partition-aware queries**: Data is partitioned - use partitioning columns in filters
4. **Sample for ML**: Use `.sample_fraction(0.01)` for model development

## Data Statistics

- **Comments:** ~10M+ Reddit comments
- **Subreddits:** Multiple popular subreddits
- **Images:** Thousands of images from Reddit posts
- **Embeddings:** Pre-computed Qwen3 embeddings for semantic search

## Related Use Cases

See full examples in `/use_cases/social_recommendation/`:
- `build_index.py` - Build search index from Reddit data
- `ingest_comments.py` - Ingest and process comments
- `reddit.ipynb` - Complete social recommendation notebook
