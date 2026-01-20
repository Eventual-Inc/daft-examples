# /// script
# description = "Perform web searches using OpenAI GPT-5 with web_search tools and store results in Supabase"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft","openai","pydantic","python-dotenv","numpy", "uuid_utils"]
# ///
# from dotenv import load_dotenv
import daft
from daft.functions import embed_text, prompt, file, regexp_split, unnest
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class Citation(BaseModel):
    url: str = Field(description="The URL of the source")
    title: str = Field(description="The title of the source")
    snippet: str = Field(description="A snippet of the source text")


class SearchResults(BaseModel):
    summary: str = Field(description="A summary of the search results")
    citations: list[Citation] = Field(description="A list of citations")


df = (
    daft.from_glob_path("hf://datasets/Eventual-Inc/sample-files/papers/*.pdf").limit(1)
    .with_column(
        "results",
        prompt(
            messages=[
                daft.lit("Find 5 closely related papers to the one attached"),
                file(daft.col("path")),
            ],
            model="gpt-4-turbo",
            tools=[{"type": "web_search"}],
            return_format=SearchResults,
            provider="openai",
            unnest=True,
        ),
    )
    .with_column("chunks", daft.col("summary").regexp_split(r"(?<=[.!?])\s+|\n+|(?=^#{1,6}\s)"))
    .with_column("embeddings",embed_text(daft.col("search_results"), model="text-embedding-ada-002", provider="openai"))
    
    #.select("path", "size", unnest(daft.col("search_results")))
)
df.show(format="fancy", max_width=60)
