# /// script
# description = "Generate a report on the image understanding performance of a model"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft[aws]", "python-dotenv"]
# ///
import os
import daft
from daft import col
from daft.functions import when
from daft.io import IOConfig, S3Config
from dotenv import load_dotenv

load_dotenv()

SOURCE_URI = "s3://daft-public-datasets/the_cauldron/evals/image_ablation"

df = daft.read_parquet(
    SOURCE_URI,
    io_config=IOConfig(
        s3=S3Config(
            region_name="us-west-2",
            key_id=os.getenv("S3_ACCESS_KEY_ID"),
            access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
            session_token=os.getenv("S3_SESSION_TOKEN"),
        )
    ),
)

# Cast our boolean columns to float64
df = df.with_columns(
    {
        "is_correct": col("is_correct").cast(daft.DataType.float64()),
        "is_correct_no_image": col("is_correct_no_image").cast(daft.DataType.float64()),
    }
)

# Calculate total accuracy
df_total = (
    df.select("subset", "is_correct", "is_correct_no_image")
    .agg(
        col("is_correct").mean().alias("accuracy_with_image"),
        col("is_correct_no_image").mean().alias("accuracy_without_image"),
        col("is_correct").count().alias("total_count"),
    )
    .with_column("subset", daft.lit("ALL"))
).collect()

# Calculate accuracy with and without images, grouped by subset
df_accuracy = (
    df.select("subset", "is_correct", "is_correct_no_image")
    .groupby("subset")
    .agg(
        col("is_correct").mean().alias("accuracy_with_image"),
        col("is_correct_no_image").mean().alias("accuracy_without_image"),
        col("is_correct").cast(daft.DataType.uint64()).count().alias("total_count"),
    )
    .collect()
)

df_accuracy = df_accuracy.concat(df_total.select("subset", "accuracy_with_image", "accuracy_without_image", "total_count"))
df_accuracy.show(format="fancy", max_width=20)


# Break down accuracy by quadrant and subset
df_classified = df.with_column(
    "classification",
    when(
        (col("is_correct") == 1.0) & (col("is_correct_no_image") == 1.0),
        "Both Correct",
    )
    .when(
        (col("is_correct") == 1.0) & (col("is_correct_no_image") == 0.0),
        "Image Helped",
    )
    .when(
        (col("is_correct") == 0.0) & (col("is_correct_no_image") == 1.0),
        "Image Hurt",
    )
    .otherwise("Both Incorrect")
).with_column("_count", daft.lit(1))


df_pivot = (
    df_classified.pivot(
        group_by=col("subset"),
        pivot_col=col("classification"),
        value_col=col("_count"),
        agg_fn="sum",
    )
).collect()


df_pivot_total = df_pivot.agg(
    col("Both Correct").sum(),
    col("Image Helped").sum(),
    col("Image Hurt").sum(),
    col("Both Incorrect").sum(),
).with_column("subset", daft.lit("ALL")).collect()

schema_order = ["subset", "Image Helped", "Image Hurt", "Both Incorrect", "Both Correct"]
df_pivot = df_pivot.select(*schema_order).concat(df_pivot_total.select(*schema_order))
df_pivot.show(format="fancy", max_width=20)

more_analytics = (
    df_pivot_total
    .with_column("total_count", col("Image Helped") + col("Image Hurt") + col("Both Incorrect") + col("Both Correct"))
    .with_columns({
        
        "image_help_rate": col("Image Helped") / (col("Image Helped") + col("Both Correct")),
        "image_hurt_rate": col("Image Hurt") / (col("Image Hurt") + col("Both Correct")),
        "net_image_helpfulness": (col("Image Helped") - col("Image Hurt")) / col("total_count"),
    })
).collect()
more_analytics.select("image_help_rate", "image_hurt_rate", "net_image_helpfulness").show(format="fancy", max_width=20)