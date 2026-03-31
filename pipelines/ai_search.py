# /// script
# description = "This example shows how using LLMs and embedding models, Daft chunks documents, extracts metadata, generates vectors, and writes them to any vector database..."
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft[openai]>=0.7.5", "pymupdf", "python-dotenv", "pydantic", "turbopuffer"]
# ///
import os
import daft
from daft import col, lit
from daft.functions import embed_text, prompt, file, unnest
from pydantic import BaseModel
from dotenv import load_dotenv


class Classifer(BaseModel):
    title: str
    author: str
    year: int
    keywords: list[str]
    abstract: str


if __name__ == "__main__":

    load_dotenv()

    daft.set_provider("openai", api_key=os.environ.get("OPENAI_API_KEY"))

    # Load documents and generate vector embeddings
    df = (
        daft.from_glob_path("hf://datasets/Eventual-Inc/sample-files/papers/*.pdf")
        .limit(1)
        .with_column(
            "metadata",
            prompt(
                messages=file(col("path")),
                system_message="Read the paper and extract the classifer metadata.",
                return_format=Classifer,
                model="gpt-5-mini",
                provider="openai",
            ),
        )
        .with_column(
            "abstract_embedding",
            embed_text(
                daft.col("metadata")["abstract"],
                provider="openai",
                model="text-embedding-3-large",
            ),
        )
        .select("path", unnest(col("metadata")), "abstract_embedding")
    )

    # Write to Turbopuffer
    df.write_turbopuffer(
        namespace="ai_papers",
        api_key=os.environ.get("TURBOPUFFER_API_KEY"),
        distance_metric="cosine_distance",
        schema={
            "path": "string",
            "title": "string",
            "author": "string",
            "year": "int",
            "keywords": "list[string]",
            "abstract": "string",
            "abstract_embedding": "vector",
        },
    )
