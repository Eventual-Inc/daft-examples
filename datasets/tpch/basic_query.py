# /// script
# description = "Load and query TPC-H lineitem data"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[aws]>=0.7.6"]
# ///

import daft
from daft import col

if __name__ == "__main__":
    # Load TPC-H lineitem data (10K CSV files)
    df = daft.read_csv("s3://daft-public-datasets/tpch-lineitem/10k-1mb-csv-files/**/*.csv")

    print("\n=== Schema ===")
    df.show(1)

    print("\n=== Basic Statistics ===")
    print(f"Total rows: {df.count_rows()}")

    print("\n=== TPC-H Query 1: Pricing Summary ===")
    result = (
        df.where(col("l_shipdate") <= "1998-12-01")
        .groupby("l_returnflag", "l_linestatus")
        .agg(
            col("l_quantity").sum().alias("sum_qty"),
            col("l_extendedprice").sum().alias("sum_base_price"),
            (col("l_extendedprice") * (1 - col("l_discount"))).sum().alias("sum_disc_price"),
            col("l_orderkey").count().alias("count_order"),
            col("l_quantity").mean().alias("avg_qty"),
            col("l_extendedprice").mean().alias("avg_price"),
            col("l_discount").mean().alias("avg_disc"),
        )
        .sort("l_returnflag", "l_linestatus")
    )
    result.show()

    print("\n=== Revenue by Shipping Mode ===")
    revenue = (
        df.groupby("l_shipmode")
        .agg((col("l_extendedprice") * (1 - col("l_discount"))).sum().alias("total_revenue"))
        .sort("total_revenue", desc=True)
    )
    revenue.show()
