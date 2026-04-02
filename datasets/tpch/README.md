# TPC-H Lineitem Dataset

The TPC-H benchmark dataset, specifically the `lineitem` table, used for testing SQL query performance and distributed data processing.

## Dataset Location

**S3 Bucket:** `s3://daft-public-datasets/tpch-lineitem/`

**Authentication:** Public bucket (no AWS credentials required)

## Dataset Structure

```
tpch-lineitem/
└── 10k-1mb-csv-files/        # 10,000 CSV files (~1MB each)
    └── *.csv
```

## Schema

```
l_orderkey: Int64             # Order ID
l_partkey: Int64              # Part ID
l_suppkey: Int64              # Supplier ID
l_linenumber: Int32           # Line number
l_quantity: Float64           # Quantity
l_extendedprice: Float64      # Extended price
l_discount: Float64           # Discount
l_tax: Float64                # Tax
l_returnflag: String          # Return flag (A/N/R)
l_linestatus: String          # Line status (O/F)
l_shipdate: Date              # Ship date
l_commitdate: Date            # Commit date
l_receiptdate: Date           # Receipt date
l_shipinstruct: String        # Shipping instructions
l_shipmode: String            # Shipping mode
l_comment: String             # Comment
```

## Examples

| File | Description | Use Case |
|------|-------------|----------|
| **[basic_query.py](./basic_query.py)** | Load and query TPC-H data | SQL operations, aggregations |
| **[performance_test.py](./performance_test.py)** | Benchmark Daft performance | Performance testing |
| **[sql_queries.py](./sql_queries.py)** | TPC-H standard queries with Daft SQL | SQL analytics |

## Quick Start

```python
import daft

# Load TPC-H lineitem data
df = daft.read_csv("s3://daft-public-datasets/tpch-lineitem/10k-1mb-csv-files/**/*.csv")
df.show(5)

# Run aggregation query
result = df.groupby("l_returnflag", "l_linestatus").agg(
    daft.col("l_quantity").sum().alias("sum_qty"),
    daft.col("l_extendedprice").sum().alias("sum_price"),
    daft.col("l_orderkey").count().alias("count_order")
)
result.show()
```

## Common Use Cases

### 1. TPC-H Query 1 (Pricing Summary)
```python
import daft
from daft import col

df = daft.read_csv("s3://daft-public-datasets/tpch-lineitem/10k-1mb-csv-files/**/*.csv")

result = (
    df.where(col("l_shipdate") <= "1998-12-01")
    .groupby("l_returnflag", "l_linestatus")
    .agg(
        col("l_quantity").sum().alias("sum_qty"),
        col("l_extendedprice").sum().alias("sum_base_price"),
        (col("l_extendedprice") * (1 - col("l_discount"))).sum().alias("sum_disc_price"),
        col("l_orderkey").count().alias("count_order"),
    )
    .sort("l_returnflag", "l_linestatus")
)
result.show()
```

### 2. Revenue by Shipping Mode
```python
df = daft.read_csv("s3://daft-public-datasets/tpch-lineitem/10k-1mb-csv-files/**/*.csv")

revenue = (
    df.groupby("l_shipmode")
    .agg(
        (col("l_extendedprice") * (1 - col("l_discount"))).sum().alias("total_revenue")
    )
    .sort("total_revenue", desc=True)
)
revenue.show()
```

### 3. Late Shipments Analysis
```python
from daft.functions import sql

df = daft.read_csv("s3://daft-public-datasets/tpch-lineitem/10k-1mb-csv-files/**/*.csv")

# Using Daft SQL
result = daft.sql("""
    SELECT
        l_shipmode,
        COUNT(*) as late_shipments,
        AVG(l_quantity) as avg_quantity
    FROM df
    WHERE l_shipdate > l_commitdate
    GROUP BY l_shipmode
    ORDER BY late_shipments DESC
""")
result.show()
```

## Performance Benchmarking

The TPC-H dataset is ideal for testing:
- **Parallel CSV reading** - 10,000 files test I/O throughput
- **Aggregation performance** - Complex GROUP BY operations
- **Filter pushdown** - Date range filters
- **Sort operations** - Multi-column sorting
- **Join performance** - Can join with other TPC-H tables

## Dataset Statistics

- **Files:** 10,000 CSV files
- **File size:** ~1MB per file
- **Total size:** ~10GB
- **Rows:** ~60M rows (scale factor 10)
- **Format:** CSV with headers

## TPC-H Query Templates

Standard TPC-H queries adapted for Daft:

### Query 1: Pricing Summary Report
Revenue, discount, and tax analysis by return flag and status

### Query 6: Forecasting Revenue Change
Revenue impact of discount increases

### Query 12: Shipping Modes and Order Priority
Shipment mode effectiveness analysis

## Performance Tips

1. **Enable dynamic batching**: `daft.set_execution_config(enable_dynamic_batching=True)`
2. **Use filters early**: Date filters reduce data scanned
3. **Projection pushdown**: Select only needed columns
4. **Parallel execution**: Daft automatically parallelizes across 10K files
5. **Benchmark correctly**: Run queries multiple times for consistent measurements

## Related Benchmarks

- **TPC-H:** Industry standard for decision support benchmarks
- **Scale factors:** This dataset is scale factor 10 (~10GB)
- **Full benchmark:** 22 standard queries test various SQL operations

## Resources

- [TPC-H Specification](http://www.tpc.org/tpch/)
- [Daft Performance Docs](https://docs.daft.ai/en/stable/performance/)
