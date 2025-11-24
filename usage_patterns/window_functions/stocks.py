# /// script
# description = "Window functions for stocks - demonstrates various window functions on real stock data"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft>=0.6.13", "yfinance>=0.2.0", "pandas>=2.0.0"]
# ///

import daft
from daft import Window, col
from daft.functions import rank, dense_rank
import yfinance as yf
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# Data Loading: Fetch Historical Stock Data
# -----------------------------------------------------------------------------

def load_stock_data(tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"], period="1y"):
    """
    Load historical stock data using yfinance (Yahoo Finance).
    
    Args:
        tickers: List of stock ticker symbols
        period: Time period (e.g., "1y", "6mo", "2y")
    
    Returns:
        Daft DataFrame with columns: ticker, date, open, high, low, close, volume
    """
    print(f"Fetching stock data for {tickers} over period: {period}...")
    
    data = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            for date, row in hist.iterrows():
                data.append({
                    "ticker": ticker,
                    "date": date.strftime("%Y-%m-%d"),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"])
                })
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    
    print(f"Loaded {len(data)} rows of stock data")
    return daft.from_pylist(data)


# -----------------------------------------------------------------------------
# Window Function Examples
# -----------------------------------------------------------------------------

# Load the data
df = load_stock_data()

print("\n" + "="*80)
print("ORIGINAL DATA (first 10 rows)")
print("="*80)
print(df.sort(["ticker", "date"]).limit(10).collect())


# -----------------------------------------------------------------------------
# Example 1: Calculate Daily Returns (Price Change %)
# Uses: partition_by + order_by + lag
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("EXAMPLE 1: Daily Returns")
print("="*80)

by_ticker_date = Window().partition_by("ticker").order_by("date")

df_with_returns = df.with_column(
    "daily_return",
    (
        (col("close") - col("close").lag(1).over(by_ticker_date)) / 
        col("close").lag(1).over(by_ticker_date) * 100
    )
).sort(["ticker", "date"])

print(df_with_returns.select("ticker", "date", "close", "daily_return").limit(15).collect())


# -----------------------------------------------------------------------------
# Example 2: Moving Averages (5-day and 20-day)
# Uses: partition_by + order_by + rows_between
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("EXAMPLE 2: Moving Averages")
print("="*80)

by_ticker_date_5day = (
    Window()
    .partition_by("ticker")
    .order_by("date")
    .rows_between(-4, 0)  # Current row + 4 preceding = 5 days
)

by_ticker_date_20day = (
    Window()
    .partition_by("ticker")
    .order_by("date")
    .rows_between(-19, 0)  # Current row + 19 preceding = 20 days
)

df_with_ma = df_with_returns.with_column(
    "ma_5", col("close").mean().over(by_ticker_date_5day)
).with_column(
    "ma_20", col("close").mean().over(by_ticker_date_20day)
).sort(["ticker", "date"])

print(df_with_ma.select("ticker", "date", "close", "ma_5", "ma_20").limit(25).collect())


# -----------------------------------------------------------------------------
# Example 3: Rank Stocks by Daily Volume
# Uses: partition_by + order_by + rank
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("EXAMPLE 3: Daily Volume Rankings (Top 3 per day)")
print("="*80)

by_date_volume = Window().partition_by("date").order_by("volume", desc=True)

df_with_volume_rank = df.with_column(
    "volume_rank", rank().over(by_date_volume)
).filter(
    col("volume_rank") <= 3
).sort(["date", "volume_rank"])

print(df_with_volume_rank.select("date", "ticker", "volume", "volume_rank").limit(20).collect())


# -----------------------------------------------------------------------------
# Example 4: Cumulative Trading Volume per Stock
# Uses: partition_by + order_by (unbounded frame)
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("EXAMPLE 4: Cumulative Trading Volume")
print("="*80)

df_with_cumulative = df.with_column(
    "cumulative_volume", col("volume").sum().over(by_ticker_date)
).sort(["ticker", "date"])

print(df_with_cumulative.select("ticker", "date", "volume", "cumulative_volume").limit(20).collect())


# -----------------------------------------------------------------------------
# Example 5: Identify Highest Volatility Days (using high-low range)
# Uses: partition_by + order_by + rank
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("EXAMPLE 5: Most Volatile Days per Stock (Top 5)")
print("="*80)

df_with_volatility = df.with_column(
    "daily_range", col("high") - col("low")
).with_column(
    "daily_range_pct", (col("high") - col("low")) / col("low") * 100
)

by_ticker_volatility = (
    Window()
    .partition_by("ticker")
    .order_by("daily_range_pct", desc=True)
)

df_top_volatile = df_with_volatility.with_column(
    "volatility_rank", rank().over(by_ticker_volatility)
).filter(
    col("volatility_rank") <= 5
).sort(["ticker", "volatility_rank"])

print(df_top_volatile.select("ticker", "date", "low", "high", "daily_range_pct", "volatility_rank").collect())


# -----------------------------------------------------------------------------
# Example 6: Stock Performance Rankings (Total Return over Period)
# Uses: window function to get first close, then calculate total return
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("EXAMPLE 6: Stock Performance Rankings (Total Return %)")
print("="*80)

by_ticker_date_first = (
    Window()
    .partition_by("ticker")
    .order_by("date")
    .rows_between(-999999, 0)  # Unbounded - all previous rows
)

df_with_first_close = df.with_column(
    "first_close", col("close").min().over(by_ticker_date_first)
).with_column(
    "total_return_pct", (col("close") - col("first_close")) / col("first_close") * 100
)

# Get the most recent date for each stock
by_ticker_last_date = Window().partition_by("ticker").order_by("date", desc=True)

df_latest_performance = df_with_first_close.with_column(
    "date_rank", rank().over(by_ticker_last_date)
).filter(
    col("date_rank") == 1
)

by_performance = Window().order_by("total_return_pct", desc=True)

df_performance_ranking = df_latest_performance.with_column(
    "performance_rank", rank().over(by_performance)
).sort("performance_rank")

print(df_performance_ranking.select(
    "performance_rank", "ticker", "date", "first_close", "close", "total_return_pct"
).collect())


# -----------------------------------------------------------------------------
# Example 7: Golden Cross / Death Cross Detection
# (When 5-day MA crosses 20-day MA)
# Uses: Moving averages + lag to detect crossovers
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("EXAMPLE 7: Golden Cross / Death Cross Detection")
print("="*80)

df_with_signals = df_with_ma.with_column(
    "ma_5_prev", col("ma_5").lag(1).over(by_ticker_date)
).with_column(
    "ma_20_prev", col("ma_20").lag(1).over(by_ticker_date)
).with_column(
    "golden_cross",
    (col("ma_5_prev") <= col("ma_20_prev")) & (col("ma_5") > col("ma_20"))
).with_column(
    "death_cross",
    (col("ma_5_prev") >= col("ma_20_prev")) & (col("ma_5") < col("ma_20"))
)

df_crossovers = df_with_signals.filter(
    col("golden_cross") | col("death_cross")
).sort(["ticker", "date"])

print(df_crossovers.select(
    "ticker", "date", "close", "ma_5", "ma_20", "golden_cross", "death_cross"
).collect())


# -----------------------------------------------------------------------------
# Example 8: Rolling Max and Min (52-week high/low)
# Uses: partition_by + order_by + rows_between with max/min aggregations
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("EXAMPLE 8: Rolling 52-Week High/Low")
print("="*80)

# Approximate 52 weeks as 252 trading days
by_ticker_52week = (
    Window()
    .partition_by("ticker")
    .order_by("date")
    .rows_between(-251, 0)
)

df_with_52week = df.with_column(
    "week_52_high", col("high").max().over(by_ticker_52week)
).with_column(
    "week_52_low", col("low").min().over(by_ticker_52week)
).with_column(
    "distance_from_high_pct", (col("close") - col("week_52_high")) / col("week_52_high") * 100
)

# Get latest data for each stock
df_52week_latest = df_with_52week.with_column(
    "date_rank", rank().over(by_ticker_last_date)
).filter(
    col("date_rank") == 1
).sort("ticker")

print(df_52week_latest.select(
    "ticker", "date", "close", "week_52_high", "week_52_low", "distance_from_high_pct"
).collect())


print("\n" + "="*80)
print("WINDOW FUNCTIONS DEMONSTRATED:")
print("="*80)
print("""
1. Daily Returns - Using lag() to compare with previous day
2. Moving Averages - Using rows_between() for rolling windows
3. Volume Rankings - Using rank() to rank across different partitions
4. Cumulative Volume - Using unbounded window for running totals
5. Volatility Rankings - Finding top N values per partition
6. Performance Rankings - Using first() to calculate total returns
7. Technical Signals - Combining multiple window functions for trading signals
8. 52-Week High/Low - Using max/min with rolling windows

All done! Try modifying the tickers or period to explore different stocks.
""")

