# /// script
# description = "Run TPC-H queries using Daft SQL"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft[aws]>=0.7.5"]
# ///

import daft
from daft import col

if __name__ == "__main__":
    # Load TPC-H data
    df = daft.read_csv("s3://daft-public-datasets/tpch-lineitem/10k-1mb-csv-files/**/*.csv")

    print("\n=== TPC-H Query 1: Pricing Summary (SQL) ===")
    result = daft.sql("""
        SELECT
            l_returnflag,
            l_linestatus,
            SUM(l_quantity) as sum_qty,
            SUM(l_extendedprice) as sum_base_price,
            SUM(l_extendedprice * (1 - l_discount)) as sum_disc_price,
            COUNT(*) as count_order
        FROM df
        WHERE l_shipdate <= '1998-12-01'
        GROUP BY l_returnflag, l_linestatus
        ORDER BY l_returnflag, l_linestatus
    """)
    result.show()

    print("\n=== TPC-H Query 6: Forecasting Revenue Change (SQL) ===")
    result = daft.sql("""
        SELECT
            SUM(l_extendedprice * l_discount) as revenue
        FROM df
        WHERE
            l_shipdate >= '1994-01-01'
            AND l_shipdate < '1995-01-01'
            AND l_discount BETWEEN 0.05 AND 0.07
            AND l_quantity < 24
    """)
    result.show()

    print("\n=== Late Shipments by Mode (SQL) ===")
    result = daft.sql("""
        SELECT
            l_shipmode,
            COUNT(*) as late_shipments,
            AVG(l_quantity) as avg_quantity,
            SUM(l_extendedprice) as total_value
        FROM df
        WHERE l_shipdate > l_commitdate
        GROUP BY l_shipmode
        ORDER BY late_shipments DESC
    """)
    result.show()

    print("\n=== High Value Orders (SQL) ===")
    result = daft.sql("""
        SELECT
            l_orderkey,
            SUM(l_extendedprice * (1 - l_discount)) as order_total,
            COUNT(*) as line_count
        FROM df
        GROUP BY l_orderkey
        HAVING order_total > 100000
        ORDER BY order_total DESC
        LIMIT 10
    """)
    result.show()
