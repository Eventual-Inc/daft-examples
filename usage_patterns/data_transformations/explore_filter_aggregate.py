# /// script
# description = "Data exploration, filtering, and aggregation with Daft"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft>=0.6.13"]
# ///
# Contributor: GitHub Copilot

"""
Explore, Filter, and Aggregate Data with Daft

This example demonstrates fundamental Daft operations:
1. Loading data from various sources
2. Exploring data schema and basic statistics  
3. Filtering rows based on conditions
4. Selecting and transforming columns
5. Aggregating data with groupby operations

Dataset: Sample emotion classification dataset (created inline or loaded from HuggingFace)
"""

import daft
from daft import col


def load_sample_data():
    """
    Create a sample emotion classification dataset.
    In production, you would load from HuggingFace:
    daft.read_parquet("hf://datasets/dair-ai/emotion/split/train-00000-of-00001.parquet")
    """
    # Sample emotion texts with labels
    # Labels: 0=sadness, 1=joy, 2=love, 3=anger, 4=fear, 5=surprise
    data = {
        "text": [
            "i feel so sad and alone today",
            "i am so happy to see you again after all these years",
            "i love spending time with my family",
            "this makes me so angry i could scream",
            "i am terrified of what might happen next",
            "wow i never expected this surprise party",
            "feeling down and depressed about everything",
            "what a wonderful day to be alive and well",
            "my heart is full of love for everyone",
            "i hate when people are so inconsiderate",
            "the dark alley made me feel scared and anxious",
            "i was shocked when i heard the incredible news",
            "crying because nothing ever goes right for me",
            "so excited about my vacation next week",
            "love is the most beautiful feeling in the world",
            "why do people have to be so mean all the time",
            "i get nervous when i have to speak in public",
            "what an unexpected turn of events today",
            "my heart aches with sorrow and grief",
            "jumping for joy because i got the job",
            "i adore my best friend so much",
            "frustrated beyond belief with this situation",
            "worried about the future and what it holds",
            "completely taken aback by the announcement",
            "feeling miserable and want to stay in bed",
            "this is the happiest moment of my life",
            "love makes everything better and brighter",
            "cannot stand how rude some people can be",
            "scared to try new things sometimes",
            "blown away by this amazing performance",
        ],
        "label": [
            0, 1, 2, 3, 4, 5,  # one of each
            0, 1, 2, 3, 4, 5,  # one of each  
            0, 1, 2, 3, 4, 5,  # one of each
            0, 1, 2, 3, 4, 5,  # one of each
            0, 1, 2, 3, 4, 5,  # one of each
        ],
    }
    return daft.from_pydict(data)


def main():
    print("=" * 80)
    print("DAFT DATA EXPLORATION, FILTERING & AGGREGATION EXAMPLE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Step 1: Load Data
    # -------------------------------------------------------------------------
    print("\n📥 Loading sample emotion classification data...")

    # Load sample data (in production, use: daft.read_parquet("hf://datasets/..."))
    df = load_sample_data()

    # -------------------------------------------------------------------------
    # Step 2: Explore the Schema
    # -------------------------------------------------------------------------
    print("\n📊 SCHEMA EXPLORATION")
    print("-" * 40)
    print(f"Schema: {df.schema()}")

    # -------------------------------------------------------------------------
    # Step 3: Sample and Preview Data
    # -------------------------------------------------------------------------
    print("\n👀 DATA PREVIEW (First 5 rows)")
    print("-" * 40)
    df.limit(5).show()

    # -------------------------------------------------------------------------
    # Step 4: Basic Statistics
    # -------------------------------------------------------------------------
    print("\n📈 BASIC STATISTICS")
    print("-" * 40)

    # Count total rows
    total_count = df.count("text").collect()
    print(f"Total rows: {total_count}")

    # Count unique labels
    label_counts = (
        df.groupby("label")
        .agg(col("text").count().alias("count"))
        .sort("count", desc=True)
    )
    print("\nLabel distribution:")
    label_counts.show()

    # -------------------------------------------------------------------------
    # Step 5: Filtering Data
    # -------------------------------------------------------------------------
    print("\n🔍 FILTERING EXAMPLES")
    print("-" * 40)

    # Filter for specific labels (e.g., label 0 = sadness)
    sadness_df = df.filter(col("label") == 0)
    print(f"Rows with label=0 (sadness): {sadness_df.count('text').collect()}")

    # Filter with string operations - texts containing "love"
    love_texts = df.filter(col("text").str.contains("love"))
    print(f"Rows containing 'love': {love_texts.count('text').collect()}")

    # Preview filtered data
    print("\nSample texts containing 'love':")
    love_texts.limit(3).show()

    # -------------------------------------------------------------------------
    # Step 6: Column Transformations
    # -------------------------------------------------------------------------
    print("\n🔄 COLUMN TRANSFORMATIONS")
    print("-" * 40)

    # Add new computed columns
    from daft.functions import when

    df_transformed = (
        df
        # Add text length column
        .with_column("text_length", col("text").str.length())
        # Add word count (approximate - split by space)
        .with_column("word_count", col("text").str.split(" ").list.length())
        # Create label name mapping using when/otherwise
        .with_column(
            "label_name",
            when(col("label") == 0, "sadness")
            .when(col("label") == 1, "joy")
            .when(col("label") == 2, "love")
            .when(col("label") == 3, "anger")
            .when(col("label") == 4, "fear")
            .otherwise("surprise")
        )
    )

    print("Data with computed columns:")
    df_transformed.select("text", "label", "label_name", "text_length", "word_count").limit(5).show()

    # -------------------------------------------------------------------------
    # Step 7: Aggregations
    # -------------------------------------------------------------------------
    print("\n📊 AGGREGATION EXAMPLES")
    print("-" * 40)

    # Group by label and compute statistics
    stats_by_label = (
        df_transformed
        .groupby("label_name")
        .agg(
            col("text").count().alias("count"),
            col("text_length").mean().alias("avg_length"),
            col("text_length").min().alias("min_length"),
            col("text_length").max().alias("max_length"),
            col("word_count").mean().alias("avg_words"),
        )
        .sort("count", desc=True)
    )

    print("Statistics by emotion label:")
    stats_by_label.show()

    # -------------------------------------------------------------------------
    # Step 8: Sorting and Ranking
    # -------------------------------------------------------------------------
    print("\n📋 SORTING EXAMPLES")
    print("-" * 40)

    # Find longest texts
    print("Top 5 longest texts:")
    df_transformed.sort("text_length", desc=True).select(
        "label_name", "text_length", "text"
    ).limit(5).show()

    # Find shortest texts
    print("\nTop 5 shortest texts:")
    df_transformed.sort("text_length").select(
        "label_name", "text_length", "text"
    ).limit(5).show()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY OF DAFT OPERATIONS DEMONSTRATED:")
    print("=" * 80)
    print("""
1. Data Loading:      daft.read_parquet() - Load data from various sources
2. Schema Inspection: df.schema() - View column names and types
3. Data Preview:      df.limit(n).show() - Preview first n rows
4. Filtering:         df.filter(condition) - Filter rows by conditions
5. Column Selection:  df.select(cols) - Select specific columns
6. Transformations:   df.with_column() - Add computed columns
7. Aggregations:      df.groupby().agg() - Group and aggregate data
8. Sorting:           df.sort(col, desc=True/False) - Sort data

These are the building blocks for data analysis pipelines with Daft!
""")


if __name__ == "__main__":
    main()
