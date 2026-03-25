# /// script
# description = "Work with pre-computed Reddit IRL embeddings for semantic search"
# dependencies = ["daft[aws]"]
# ///

import daft
from daft import col

# Load pre-computed Qwen3 embeddings
df = daft.read_parquet(
    "s3://daft-public-datasets/reddit-irl/embeddings/Qwen--Qwen3-Embedding-0.6B.parquet/*.parquet"
)

print("\n=== Schema ===")
df.show(1)

print("\n=== Embedding Statistics ===")
print(f"Total embeddings: {df.count_rows()}")

print("\n=== Sample Embeddings ===")
df.select("id", "embedding").show(5)

print("\n=== Embedding Dimensions ===")
# Embeddings are vectors - check dimensionality
df.with_column("embedding_dim", col("embedding").list.length()).select(
    "embedding_dim"
).describe()
