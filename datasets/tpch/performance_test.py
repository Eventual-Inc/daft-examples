# /// script
# description = "Benchmark Daft performance on TPC-H data"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[aws]>=0.7.10"]
# ///

import time

import daft
from daft import col
from daft.io import IOConfig, S3Config

if __name__ == "__main__":
    # Enable dynamic batching for optimal performance
    daft.set_execution_config(enable_dynamic_batching=True)

    io_config = IOConfig(s3=S3Config(anonymous=True, region_name="us-east-1"))
    daft.set_planning_config(default_io_config=io_config)
    df = daft.read_parquet(
        "s3://daft-public-datasets/tpch-lineitem/100_0/32/108417bd-5bee-43d9-bf9a-d6faec6afb2d-0.parquet",
        io_config=io_config,
    )

    print("\n=== Performance Benchmark ===")

    # Test 1: Count rows
    start = time.time()
    row_count = df.count_rows()
    count_time = time.time() - start
    print(f"Count rows: {row_count} rows in {count_time:.2f}s")

    # Test 2: Simple aggregation
    start = time.time()
    agg_result = df.select(
        col("L_QUANTITY").sum().alias("total_qty"),
        col("L_EXTENDEDPRICE").sum().alias("total_price"),
    ).collect()
    agg_time = time.time() - start
    print(f"Aggregation: {agg_time:.2f}s")

    # Test 3: Group by with multiple aggregations
    start = time.time()
    group_result = (
        df.groupby("L_RETURNFLAG")
        .agg(
            col("L_QUANTITY").sum().alias("sum_qty"),
            col("L_EXTENDEDPRICE").mean().alias("avg_price"),
            col("L_ORDERKEY").count().alias("count_order"),
        )
        .collect()
    )
    group_time = time.time() - start
    print(f"Group by aggregation: {group_time:.2f}s")

    # Test 4: Filter + aggregation
    start = time.time()
    filter_result = (
        df.where(col("L_SHIPDATE") <= "1998-12-01")
        .groupby("L_SHIPMODE")
        .agg(col("L_QUANTITY").sum().alias("sum_qty"))
        .collect()
    )
    filter_time = time.time() - start
    print(f"Filter + aggregation: {filter_time:.2f}s")

    print("\n=== Performance Summary ===")
    print(f"Total execution time: {count_time + agg_time + group_time + filter_time:.2f}s")
    print(f"Throughput: {row_count / (count_time + agg_time + group_time + filter_time):.0f} rows/sec")
