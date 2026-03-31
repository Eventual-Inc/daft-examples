# /// script
# description = "Daily Arxiv Summarization and Indexing Workflow"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft[turbopuffer]>=0.6.13", "openai", "python-dotenv"]
# ///

import os
import daft
from daft import col
from daft.functions import prompt, embed_text
from dotenv import load_dotenv

load_dotenv()

# Config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TURBOPUFFER_API_KEY = os.getenv("TURBOPUFFER_API_KEY")


S3_BUCKET = os.getenv("S3_BUCKET", "daft-arxiv-demo")
S3_PREFIX = os.getenv("S3_PREFIX", "raw/")
TARGET_DATE = os.getenv("TARGET_DATE", "2025-11-19")


def main():
    # 1. Read Data (S3 or Local Fallback)
    s3_path = f"s3://{S3_BUCKET}/{S3_PREFIX}{TARGET_DATE}/*.json"
    local_path = f".data/{S3_PREFIX}{TARGET_DATE}/*.json"

    print(f"Attempting to read from: {s3_path}")

    try:
        # Try reading from S3 first
        df = daft.read_json(s3_path)
        # Trigger a light action to check accessibility and infer schema
        df.limit(1).collect()
    except Exception:
        print("S3 read failed (or empty). Checking local data...")
        try:
            df = daft.read_json(local_path)
        except Exception:
            print("No local data found. Generating synthetic data for demonstration.")
            df = daft.from_pydict(
                {
                    "id": ["2310.0001", "2310.0002", "2310.0003"],
                    "title": [
                        "Attention Is All You Need",
                        "Daft: Distributed Dataframes",
                        "Turbopuffer: Fast Vector Search",
                    ],
                    "summary": [
                        "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
                        "Daft is a unified distributed dataframe for multimodal data processing.",
                        "Turbopuffer is a serverless vector database designed for scale and performance.",
                    ],
                    "authors": [
                        ["Vaswani et al."],
                        ["Eventual Computing"],
                        ["Turbopuffer Team"],
                    ],
                    "category": ["cs.CL", "cs.DB", "cs.IR"],
                    "link": [
                        "https://arxiv.org/abs/1706.03762",
                        "https://github.com/Eventual-Inc/Daft",
                        "https://turbopuffer.com",
                    ],
                    "ingested_at": ["2025-11-19T10:00:00"] * 3,
                }
            )

    print("Source Data Preview:")
    df.show(1)

    # 2. Summarize Papers (LLM)
    # Use a cheaper model for bulk summarization
    summary_instruction = (
        "Summarize the academic paper abstract below into a single concise sentence suitable for a weekly digest.\n"
        "Title: " + col("title") + "\n"
        "Abstract: " + col("summary")
    )

    df = df.with_column(
        "short_summary",
        prompt(messages=summary_instruction, model="gpt-4o-mini", max_tokens=100),
    )

    # 3. Create Embeddings (Vector)
    # Embed: "Title: Summary"
    df = df.with_column("search_text", col("title") + ": " + col("short_summary"))

    # Ensure OPENROUTER or OPENAI provider is set up.
    # Daft defaults to OpenAI if not specified differently in set_provider, but env var must be there.
    df = df.with_column(
        "embedding",
        embed_text(
            col("search_text"), provider="openai", model="text-embedding-3-small"
        ),
    )

    # 4. Execute & Index to Turbopuffer
    # We collect results to the driver to upload.
    # For massive scale, we would partition and use a map_partitions approach or custom sink.
    print("Processing summaries and embeddings and writing to Turbopuffer...")

    df.write_turbopuffer(
        namespace="arxiv-papers",
        id_column="id",
        vector_column="embedding",
        api_key=TURBOPUFFER_API_KEY,
    )
    print("Successfully indexed to Turbopuffer!")


if __name__ == "__main__":
    main()
