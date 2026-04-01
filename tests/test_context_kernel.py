import daft
from pipelines.context_engineering.context_kernel import (
    pack_context,
    score_query_overlap,
    segment_sentences,
)


def test_pack_context_respects_budget_and_preserves_source_order():
    candidates = daft.from_pydict(
        {
            "query_id": [1, 1, 1],
            "query": ["context"] * 3,
            "unit_id": [101, 102, 103],
            "payload": [
                "Context units preserve provenance.",
                "Pack selects the highest scoring rows.",
                "This sentence is less relevant noise.",
            ],
            "token_estimate": [4, 6, 5],
            "score": [0.8, 0.95, 0.1],
            "order_key": [0, 1, 2],
        }
    )

    rows = (
        pack_context(
            candidates,
            group_by=["query_id", "query"],
            max_tokens=10,
        )
        .collect()
        .to_pylist()
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["packed_unit_ids"] == [101, 102]
    assert row["packed_token_estimate"] == 10
    assert row["packed_count"] == 2
    assert row["packed_text"] == ("Context units preserve provenance.\n\nPack selects the highest scoring rows.")


def test_segment_annotate_pack_builds_query_specific_context():
    documents = daft.from_pydict(
        {
            "source_id": ["doc-1", "doc-2"],
            "text": [
                (
                    "Daft keeps reasoning in the query plan. "
                    "Context units carry provenance. "
                    "Packing assembles bounded prompts."
                ),
                ("Vector search can improve retrieval. Databases can store embeddings. This line is unrelated."),
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
    scored = score_query_overlap(
        queries,
        units,
        query_id_col="query_id",
        query_col="query",
    )
    bundles = pack_context(scored, group_by=["query_id", "query"], max_tokens=10)

    rows = {row["query_id"]: row for row in bundles.collect().to_pylist()}

    assert "Context units carry provenance." in rows[1]["packed_text"]
    assert "Vector search can improve retrieval." in rows[2]["packed_text"]
    assert "This line is unrelated." not in rows[2]["packed_text"]
    assert rows[3]["packed_text"] == ""
    assert rows[3]["packed_unit_ids"] == []
    assert rows[3]["packed_count"] == 0
