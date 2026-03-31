# /// script
# description = "Load and explore LAION metadata"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[aws]>=0.7.6"]
# ///

import daft
from daft import col

if __name__ == "__main__":
    # Load LAION parquet metadata
    df = daft.read_parquet("s3://daft-public-data/tutorials/laion-parquet/train-00000-of-00001-*.parquet")

    print(f"Dataset: {df.count_rows()} rows")
    df.select("TEXT", "AESTHETIC_SCORE", "URL").show(10)

    print("High quality pairs (aesthetic > 6.0):")
    high_quality = df.where(col("AESTHETIC_SCORE") > 6.0)
    high_quality.select("TEXT", "AESTHETIC_SCORE").show(5)
