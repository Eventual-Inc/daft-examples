# /// script
# description = "Load and explore LAION metadata"
# dependencies = ["daft[aws]"]
# ///

import daft
from daft import col

# Load LAION parquet metadata
df = daft.read_parquet(
    "s3://daft-public-data/tutorials/laion-parquet/train-00000-of-00001-*.parquet"
)

print(f"Dataset: {df.count_rows()} rows")
df.select("TEXT", "AESTHETIC_SCORE", "URL").show(10)

print("High quality pairs (aesthetic > 6.0):")
high_quality = df.where(col("AESTHETIC_SCORE") > 6.0)
high_quality.select("TEXT", "AESTHETIC_SCORE").show(5)
