# /// script
# description = "Run TPC-H queries using Daft SQL"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[aws]>=0.7.6"]
# ///

import daft
from daft.io import IOConfig, S3Config

if __name__ == "__main__":
    io_config = IOConfig(s3=S3Config(anonymous=True, region_name="us-east-1"))
    daft.set_planning_config(default_io_config=io_config)

    # Load TPC-H data
    df = daft.read_parquet("s3://daft-public-datasets/tpch-lineitem/100_0/32/108417bd-5bee-43d9-bf9a-d6faec6afb2d-0.parquet", io_config=io_config)

    print("\n=== TPC-H Query 1: Pricing Summary (SQL) ===")
    result = daft.sql("""
        SELECT
            L_RETURNFLAG,
            L_LINESTATUS,
            SUM(L_QUANTITY) as sum_qty,
            SUM(L_EXTENDEDPRICE) as sum_base_price,
            SUM(L_EXTENDEDPRICE * (1 - L_DISCOUNT)) as sum_disc_price,
            COUNT(*) as count_order
        FROM df
        WHERE L_SHIPDATE <= '1998-12-01'
        GROUP BY L_RETURNFLAG, L_LINESTATUS
        ORDER BY L_RETURNFLAG, L_LINESTATUS
    """)
    result.show()

    print("\n=== TPC-H Query 6: Forecasting Revenue Change (SQL) ===")
    result = daft.sql("""
        SELECT
            SUM(L_EXTENDEDPRICE * L_DISCOUNT) as revenue
        FROM df
        WHERE
            L_SHIPDATE >= '1994-01-01'
            AND L_SHIPDATE < '1995-01-01'
            AND L_DISCOUNT BETWEEN 0.05 AND 0.07
            AND L_QUANTITY < 24
    """)
    result.show()

    print("\n=== Late Shipments by Mode (SQL) ===")
    result = daft.sql("""
        SELECT
            L_SHIPMODE,
            COUNT(*) as late_shipments,
            AVG(L_QUANTITY) as avg_quantity,
            SUM(L_EXTENDEDPRICE) as total_value
        FROM df
        WHERE L_SHIPDATE > L_COMMITDATE
        GROUP BY L_SHIPMODE
        ORDER BY late_shipments DESC
    """)
    result.show()

    print("\n=== High Value Orders (SQL) ===")
    result = daft.sql("""
        SELECT
            L_ORDERKEY,
            SUM(L_EXTENDEDPRICE * (1 - L_DISCOUNT)) as order_total,
            COUNT(*) as line_count
        FROM df
        GROUP BY L_ORDERKEY
        HAVING order_total > 100000
        ORDER BY order_total DESC
        LIMIT 10
    """)
    result.show()
