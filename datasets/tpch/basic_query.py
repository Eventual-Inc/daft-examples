# /// script
# description = "Load and query TPC-H lineitem data"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[aws]>=0.7.10"]
# ///

import daft
from daft import col
from daft.io import IOConfig, S3Config

if __name__ == "__main__":
    io_config = IOConfig(s3=S3Config(anonymous=True, region_name="us-east-1"))
    daft.set_planning_config(default_io_config=io_config)

    # Load TPC-H lineitem data (single parquet file for fast execution)
    df = daft.read_parquet(
        "s3://daft-public-datasets/tpch-lineitem/100_0/32/108417bd-5bee-43d9-bf9a-d6faec6afb2d-0.parquet",
        io_config=io_config,
    )

    print("\n=== Schema ===")
    df.show(1)

    print("\n=== Basic Statistics ===")
    print(f"Total rows: {df.count_rows()}")

    print("\n=== TPC-H Query 1: Pricing Summary ===")
    result = (
        df.where(col("L_SHIPDATE") <= "1998-12-01")
        .groupby("L_RETURNFLAG", "L_LINESTATUS")
        .agg(
            col("L_QUANTITY").sum().alias("sum_qty"),
            col("L_EXTENDEDPRICE").sum().alias("sum_base_price"),
            (col("L_EXTENDEDPRICE") * (1 - col("L_DISCOUNT"))).sum().alias("sum_disc_price"),
            col("L_ORDERKEY").count().alias("count_order"),
            col("L_QUANTITY").mean().alias("avg_qty"),
            col("L_EXTENDEDPRICE").mean().alias("avg_price"),
            col("L_DISCOUNT").mean().alias("avg_disc"),
        )
        .sort([col("L_RETURNFLAG"), col("L_LINESTATUS")])
    )
    result.show()

    print("\n=== Revenue by Shipping Mode ===")
    revenue = (
        df.groupby("L_SHIPMODE")
        .agg((col("L_EXTENDEDPRICE") * (1 - col("L_DISCOUNT"))).sum().alias("total_revenue"))
        .sort(col("total_revenue"), desc=True)
    )
    revenue.show()
