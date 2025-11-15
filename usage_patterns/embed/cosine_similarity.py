# /// script
# description = "Similarity Search"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft>=0.6.13", "openai", "numpy", "python-dotenv"]
# ///
import daft
from daft.functions import embed_text, cosine_distance
from dotenv import load_dotenv

load_dotenv()

# Create a knowledge base with documents
documents = daft.from_pydict({
    "doc_id": [1, 2, 3, 4],
    "text": [
        "Python is a high-level programming language",
        "Machine learning models require training data",
        "Daft is a distributed dataframe library",
        "Embeddings capture semantic meaning of text",
    ]
})

# Embed all documents
documents = documents.with_column(
    "embedding",
    embed_text(daft.col("text"), provider="openai", model="text-embedding-3-small")
)

# Create a query
query = daft.from_pydict({
    "query_text": ["What is Daft?"]
})

# Embed the query
query = query.with_column(
    "query_embedding",
    embed_text(daft.col("query_text"), provider="openai", model="text-embedding-3-small")
)

# Cross join to compare query against all documents
results = query.join(documents, how="cross")

# Calculate cosine distance (lower is more similar)
results = results.with_column(
    "distance",
    cosine_distance(daft.col("query_embedding"), daft.col("embedding"))
)

# Sort by distance and show top results
results = results.sort("distance").select("query_text", "text", "distance")
results.show()