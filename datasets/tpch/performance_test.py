# /// script
# description = "Benchmark Daft performance on TPC-H data"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[aws]>=0.7.6"]
# ///

import time

import daft
from daft import col

if __name__ == "__main__":
    # Enable dynamic batching for optimal performance
    daft.set_execution_config(enable_dynamic_batching=True)

    df = daft.read_csv("s3://daft-public-datasets/tpch-lineitem/10k-1mb-csv-files/**/*.csv")

    print("\n=== Performance Benchmark ===")

    # Test 1: Count rows
    start = time.time()
    row_count = df.count_rows()
    count_time = time.time() - start
    print(f"Count rows: {row_count} rows in {count_time:.2f}s")

    # Test 2: Simple aggregation
    start = time.time()
    agg_result = df.select(
        col("l_quantity").sum().alias("total_qty"),
        col("l_extendedprice").sum().alias("total_price"),
    ).collect()
    agg_time = time.time() - start
    print(f"Aggregation: {agg_time:.2f}s")

    # Test 3: Group by with multiple aggregations
    start = time.time()
    group_result = (
        df.groupby("l_returnflag")
        .agg(
            col("l_quantity").sum().alias("sum_qty"),
            col("l_extendedprice").mean().alias("avg_price"),
            col("l_orderkey").count().alias("count_order"),
        )
        .collect()
    )
    group_time = time.time() - start
    print(f"Group by aggregation: {group_time:.2f}s")

    # Test 4: Filter + aggregation
    start = time.time()
    filter_result = (
        df.where(col("l_shipdate") <= "1998-12-01")
        .groupby("l_shipmode")
        .agg(col("l_quantity").sum().alias("sum_qty"))
        .collect()
    )
    filter_time = time.time() - start
    print(f"Filter + aggregation: {filter_time:.2f}s")

    print("\n=== Performance Summary ===")
    print(f"Total execution time: {count_time + agg_time + group_time + filter_time:.2f}s")
    print(f"Throughput: {row_count / (count_time + agg_time + group_time + filter_time):.0f} rows/sec")
