# /// script
# description = "Work with LAION image-text pairs"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[aws]>=0.7.10"]
# ///

import daft
from daft import col
from daft.functions import when
from daft.io import IOConfig, S3Config

if __name__ == "__main__":
    io_config = IOConfig(s3=S3Config(anonymous=True, region_name="us-east-1"))

    # Load LAION metadata
    df_metadata = daft.read_parquet(
        "s3://daft-public-data/tutorials/laion-parquet/train-00000-of-00001-*.parquet",
        io_config=io_config,
    )

    print("Square images (512x512+, high quality):")
    square_images = df_metadata.where(
        (col("WIDTH") == col("HEIGHT")) & (col("WIDTH") >= 512) & (col("AESTHETIC_SCORE") > 6.0)
    )
    square_images.select("WIDTH", "HEIGHT", "TEXT", "AESTHETIC_SCORE").show(10)

    print("Aspect ratio distribution:")
    df_ratios = df_metadata.with_column("aspect_ratio", col("WIDTH") / col("HEIGHT"))
    df_ratios = df_ratios.with_column(
        "ratio_category",
        when(col("aspect_ratio") < 0.9, "portrait").when(col("aspect_ratio") > 1.1, "landscape").otherwise("square"),
    )
    ratio_dist = (
        df_ratios.groupby("ratio_category").agg(col("ratio_category").count().alias("count")).sort("count", desc=True)
    )
    ratio_dist.show()
