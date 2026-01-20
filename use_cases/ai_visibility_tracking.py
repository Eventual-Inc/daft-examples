# /// script
# description = "Track AI visibility across multiple LLMs (brand mentions, citations, share of voice)"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft","openai","pydantic","python-dotenv","numpy"]
# ///
import os
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import daft
from daft import col, lit
from daft.functions import prompt, unnest, format as dformat

load_dotenv()

# Configure OpenRouter as default provider to access multiple models through one endpoint
daft.set_provider(
    "openai",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# Optional: also configure Gemini OpenAI-compatible endpoint if desired
# daft.set_provider(
#     "gemini",
#     api_key=os.environ.get("GEMINI_API_KEY"),
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# )

BRAND = os.environ.get("BRAND_NAME", "YourBrand")


class Citation(BaseModel):
    url: Optional[str] = Field(default=None, description="Source URL if provided")
    title: Optional[str] = Field(default=None, description="Source title if provided")
    snippet: Optional[str] = Field(
        default=None, description="Short snippet if provided"
    )


class VisibilityEval(BaseModel):
    mentions_brand: bool = Field(
        description="Whether the answer mentions the target brand explicitly"
    )
    sentiment: str = Field(
        description="Overall tone toward the brand if mentioned: positive|neutral|negative"
    )
    reasoning: str = Field(description="Short reasoning for mentions and sentiment")
    citations: List[Citation] = Field(
        default_factory=list, description="Any citations or links provided"
    )
    answer: str = Field(description="The model's answer text")


# Define prompts/keywords your audience would actually use
prompts = [
    "best CRM for small business",
    "top data pipeline framework for multimodal AI workloads",
    "how to run RAG with video and audio",
    "what is a canonical dataset for recommendations",
]

# Define engines/models to test (all via OpenRouter here)
engines = [
    "gpt-5",
    "google/gemini-2.5-flash",
    "nvidia/nemotron-nano-9b-v2:free",
]

# Build a cross-product of prompts × engines
df_prompts = daft.from_pydict({"prompt": prompts})
df_engines = daft.from_pydict({"engine": engines})
df = df_prompts.join(df_engines, how="cross")

# Instruction template pushes for citations and explicit mention checks
system_msg = dformat(
    "You are evaluating AI visibility for brand: {}. "
    "Answer the user question concisely and include citations when possible.",
    lit(BRAND),
)

# Ask the model to self-report mentions/sentiment and provide the answer + citations as structured output
df = df.with_column(
    "result",
    prompt(
        messages=col("prompt"),
        system_message=system_msg,
        model=col("engine"),
        use_chat_completions=True,
        return_format=VisibilityEval,
        provider="openai",
    ),
)

# Flatten structured output
df = df.select("engine", "prompt", unnest(col("result")))

# Compute basic share-of-voice style metric per engine: mean of mentions_brand
df_metrics = (
    df.with_column("mentions_int", col("mentions_brand").cast(int))
    .groupby("engine")
    .agg({"mentions_int": "mean"})
    .rename({"mentions_int_mean": "share_of_voice"})
)

# Write detailed results and summary
os.makedirs(".data/ai_visibility", exist_ok=True)
df.write_json(".data/ai_visibility/results.json")
df_metrics.write_json(".data/ai_visibility/summary.json")

# Show a quick preview
df.show(5, format="fancy", max_width=90)
df_metrics.show(format="fancy", max_width=60)
