# /// script
# description = "Perform web searches using OpenAI GPT-5 with web_search tools and store results in Supabase"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft","openai","pydantic","python-dotenv","numpy", "uuid_utils"]
# ///
#from dotenv import load_dotenv
import daft
from daft.functions import prompt, file
from pydantic import BaseModel, Field
import json
from dotenv import load_dotenv

load_dotenv()

class Citation(BaseModel):
    url: str = Field(description="The URL of the source")
    title: str = Field(description="The title of the source")
    snippet: str = Field(description="A snippet of the source text")

class SearchResults(BaseModel):
    summary: str = Field(description="A summary of the search results")
    citations: list[Citation] = Field(description="A list of citations")

df = daft.from_glob_path("hf://datasets/Eventual-Inc/sample-files/papers/*.pdf")
df = df.with_column("file", file(daft.col("path")))
df = df.with_column(
    "search_results",
    prompt(
        messages=[daft.lit("Find 5 closely related papers to the one attached"), daft.col("file")],
        model="gpt-5",
        tools=[{"type": "web_search"}],
        return_format=SearchResults,
        provider="openai",
    )
)

df.show(format="fancy", max_width=60)
