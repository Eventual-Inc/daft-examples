# /// script
# description = "Context DSL prototype: segment, annotate, and pack bounded context bundles"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft>=0.7.6"]
# ///

from __future__ import annotations

import sys
from pathlib import Path

import daft

# Allow `uv run path/to/script.py` to resolve repo-local imports.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.context_engineering.context_kernel import (
    pack_context,
    score_query_overlap,
    segment_sentences,
)

if __name__ == "__main__":
    documents = daft.from_pydict(
        {
            "source_id": ["doc-1", "doc-2"],
            "text": [
                (
                    "Daft keeps reasoning in the query plan. "
                    "Context units carry provenance across transformations. "
                    "Packing assembles bounded prompts for an agent."
                ),
                (
                    "Vector search can improve retrieval quality. "
                    "Embeddings help surface semantically related context. "
                    "Relational operators keep the pipeline inspectable."
                ),
            ],
        }
    )

    queries = daft.from_pydict(
        {
            "query_id": [1, 2, 3],
            "query": [
                "How do context units preserve provenance?",
                "How do embeddings help retrieval?",
                "Which sentence mentions galaxies?",
            ],
        }
    )

    units = segment_sentences(documents, source_id_col="source_id", text_col="text")
    scored = score_query_overlap(queries, units, query_id_col="query_id", query_col="query")
    bundles = pack_context(scored, group_by=["query_id", "query"], max_tokens=10)

    print("\n=== Segmented Units ===")
    units.select("source_id", "unit_id", "order_key", "token_estimate", "payload").show()

    print("\n=== Scored Candidates ===")
    (scored.sort("score", desc=True).select("query_id", "source_id", "score", "token_estimate", "payload").show())

    print("\n=== Packed Context Bundles ===")
    bundles.show()

    print("\n=== Bundle Previews ===")
    for row in bundles.collect().to_pylist():
        print(f"\nQuery {row['query_id']}: {row['query']}")
        print(f"Packed units: {row['packed_unit_ids']}")
        print(f"Estimated tokens: {row['packed_token_estimate']}")
        print(row["packed_text"])
