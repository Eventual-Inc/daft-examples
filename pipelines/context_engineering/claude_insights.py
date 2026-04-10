# /// script
# description = "Extract structured insights from Claude skills"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft[openai]", "python-dotenv"]
# ///
"""
Run: uv run daft-file-demo.py

Glob your entire memory system into a DataFrame,
then let an LLM extract structured insights from each file.
"""

import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

import daft

load_dotenv()

daft.set_provider("openai", api_key=os.getenv("OPENAI_API_KEY"))


class FileInsight(BaseModel):
    summary: str = Field(description="One-sentence summary of what this file contains")
    topics: list[str] = Field(description="Key topics covered")
    actionable: bool = Field(description="Whether this file contains actionable tasks or just reference material")


df = (
    daft.from_files("~/.claude/skills/**/*.md")
    .limit(3)
    .with_column(
        "insight",
        daft.functions.prompt(
            messages=["look at this skill: " + daft.col("file")],
            return_format=FileInsight,
            system_message="Analyze these skills and extract structured insights.",
            model="gpt-5-mini",
        ),
    )
)

df.show(format="fancy", max_width=120)
