# /// script
# description = "This example shows how using LLMs and embedding models, Daft chunks documents, extracts metadata, generates vectors, and writes them to any vector database..."
# dependencies = ["daft[openai, turbopuffer]==0.7.1", "pymupdf", "python-dotenv"]
# ///
import os
import daft
from daft import col, lit
from daft.functions import embed_text, prompt, file, unnest, monotonically_increasing_id
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class Classifer(BaseModel):
    title: str
    author: str
    year: int
    keywords: list[str]
    abstract: str

daft.set_execution_config(enable_dynamic_batching=True)
daft.set_provider("openai", api_key=os.environ.get("OPENAI_API_KEY"))

# Load documents and generate vector embeddings
df = (
    daft.from_glob_path("hf://datasets/Eventual-Inc/sample-files/papers/*.pdf").limit(10)
    .with_column(
        "metadata", 
        prompt(
            messages=file(col("path")),
            system_message="Read the paper and extract the classifer metadata.",
            return_format=Classifer, 
            model="gpt-5-mini", 
        )
    )
    .with_column(
        "abstract_embedding", 
        embed_text(
            daft.col("metadata")["abstract"], 
            model="text-embedding-3-large"
        )
    )
    .with_column("id", monotonically_increasing_id())
    .select("id", "path", unnest(col("metadata")), "abstract_embedding")
)

# Show sample results
print("\n=== Sample Papers with Embeddings ===")
df.select("id", "title", "author", "year", "keywords").show(5)

# Write to Turbopuffer (optional - requires TURBOPUFFER_API_KEY)
if os.environ.get("TURBOPUFFER_API_KEY"):
    df.write_turbopuffer(
        namespace="ai_papers",
        api_key=os.environ.get("TURBOPUFFER_API_KEY"),
        distance_metric="cosine_distance",
        region='us-west-2',
        schema={
            "id": "int64",
            "path": "string",
            "title": "string",
            "author": "string",
            "year": "int",
            "keywords": "list[string]",
            "abstract": "string",
            "abstract_embedding": "vector",
        }
    )
