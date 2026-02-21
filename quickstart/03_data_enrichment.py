# /// script
# description = "A minimal Daft enrichment pipeline: load rows into a DataFrame, normalize text deterministically, call LLMs to extract typed metadata and redact PII (validated via Pydantic schemas), then flatten and write a clean table for downstream search/analytics."
# dependencies = ["daft[openai]==0.7.1", "pydantic", "python-dotenv"]
# ///
import os
from pathlib import Path
import daft
from daft import col
from daft.functions import prompt, unnest, monotonically_increasing_id
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = Path(
    os.environ.get(
        "DAFT_EXAMPLES_OUTPUT_DIR", ".data/quickstart/03_data_enrichment/enriched-comments"
    )
)


class Meta(BaseModel):
    sentiment: str
    topics: list[str]
    has_pii: bool


class Redacted(BaseModel):
    safe_text: str
    pii_types: list[str]


daft.set_execution_config(enable_dynamic_batching=True)
daft.set_provider("openai", api_key=os.environ.get("OPENAI_API_KEY"))

df = (
    daft.read_parquet(
        "https://huggingface.co/api/datasets/SocialGrep/the-reddit-irl-dataset/parquet/comments/train/0.parquet"
    )
    .limit(5)
    .with_column("id", monotonically_increasing_id())
    .select("id", col("body").alias("text"))
    .with_column("norm", col("text").normalize(lowercase=True, remove_punct=True))
    .with_column(
        "meta",
        prompt(
            col("norm"),
            system_message="Extract sentiment, topics, and whether the text contains PII.",
            return_format=Meta,
            model="gpt-5-mini",
        ),
    )
    .with_column(
        "red",
        prompt(
            col("text"),
            system_message="Redact PII by replacing emails/phones/names/addresses with placeholders like [EMAIL]. Return the redacted text and list PII types found.",
            return_format=Redacted,
            model="gpt-5-mini",
        ),
    )
    .select(
        "id",
        "text",
        "norm",
        col("red")["safe_text"].alias("safe_text"),
        col("red")["pii_types"].alias("pii_types"),
        unnest(col("meta")),
    )
)

print("\n=== Sample Enriched Rows ===")
df.show(5)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df.write_parquet(str(OUTPUT_DIR))
