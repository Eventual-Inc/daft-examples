# /// script
# description = "Web Search with OpenAI"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft","openai","pydantic","python-dotenv","numpy"]
# ///

import os
from dotenv import load_dotenv
import daft
from daft.functions.ai import prompt
from pydantic import BaseModel, Field
import json
from typing import List
from datetime import datetime

load_dotenv()

supabase_connection = os.environ["SUPABASE_CONNECTION"]

class Citation(BaseModel):
    url: str = Field(description="The URL of the source")
    title: str = Field(description="The title of the source")
    snippet: str = Field(description="A snippet of the source text")

class SearchResults(BaseModel):
    summary: str = Field(description="A summary of the search results")
    citations: list[Citation] = Field(description="A list of citations")


df = daft.from_pydict({
    "query": [
        "What are some great use-cases for vectorized agentic web search?",
        "If I were to use daft with web search using it's prompt function on OpenAI's internal web-search tool, what would be the most powerful way to leverage both technologies?",
        "Buy one get one free burritos in SF.", 
    ]
})

df = df.with_column(
    "search_results",
    prompt(
        daft.col("query"),
        model="gpt-5",
        tools=[{"type": "web_search"}],
        return_format=SearchResults,
        provider="openai",
    )
)

catalog = daft.Catalog.from_postgres(supabase_connection)
table = catalog.create_table_if_not_exists("prompt_openai_web_search", df.schema())
table.append(df)

