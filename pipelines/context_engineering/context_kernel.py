from __future__ import annotations

import re

import daft
from daft import DataFrame, DataType, col
from daft.functions import monotonically_increasing_id

SentenceSegment = DataType.struct(
    {
        "order_key": DataType.int64(),
        "payload": DataType.string(),
    }
)

PackedContext = DataType.struct(
    {
        "packed_text": DataType.string(),
        "packed_unit_ids": DataType.list(DataType.int64()),
        "packed_token_estimate": DataType.int64(),
        "packed_count": DataType.int64(),
    }
)

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str | None) -> list[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


@daft.func(return_dtype=DataType.list(SentenceSegment))
def split_sentences(text: str) -> list[dict[str, object]]:
    """Split text into sentence-like segments with stable per-document order."""
    if not text:
        return []

    normalized = re.sub(r"\s+", " ", text.strip())
    parts = [part.strip() for part in _SENTENCE_BOUNDARY_RE.split(normalized) if part.strip()]
    return [{"order_key": idx, "payload": part} for idx, part in enumerate(parts)]


@daft.func(return_dtype=DataType.int64())
def estimate_tokens(text: str) -> int:
    """Approximate token count with a simple whitespace/token regex heuristic."""
    return len(_tokenize(text))


@daft.func(return_dtype=DataType.float64())
def lexical_overlap_score(query: str, text: str) -> float:
    """Score a unit by query-token recall."""
    query_tokens = set(_tokenize(query))
    text_tokens = set(_tokenize(text))

    if not query_tokens or not text_tokens:
        return 0.0

    return len(query_tokens & text_tokens) / float(len(query_tokens))


@daft.func(return_dtype=PackedContext)
def greedy_pack(
    unit_ids: list[int],
    payloads: list[str],
    token_estimates: list[int],
    scores: list[float],
    order_keys: list[int],
    max_tokens: int,
    delimiter: str = "\n\n",
) -> dict[str, object]:
    """Select the highest-scoring units within budget, then restore source order."""
    rows: list[dict[str, object]] = []
    for unit_id, payload, token_estimate, score, order_key in zip(
        unit_ids,
        payloads,
        token_estimates,
        scores,
        order_keys,
        strict=True,
    ):
        rows.append(
            {
                "unit_id": unit_id,
                "payload": payload,
                "token_estimate": max(0, int(token_estimate or 0)),
                "score": float(score or 0.0),
                "order_key": int(order_key or 0),
            }
        )

    rows.sort(
        key=lambda row: (
            -float(row["score"]),
            int(row["token_estimate"]),
            int(row["order_key"]),
            int(row["unit_id"]),
        )
    )

    selected: list[dict[str, object]] = []
    used_tokens = 0
    for row in rows:
        if float(row["score"]) <= 0:
            continue
        next_cost = used_tokens + int(row["token_estimate"])
        if next_cost > max_tokens:
            continue
        selected.append(row)
        used_tokens = next_cost

    selected.sort(key=lambda row: (int(row["order_key"]), int(row["unit_id"])))

    return {
        "packed_text": delimiter.join(str(row["payload"]) for row in selected),
        "packed_unit_ids": [int(row["unit_id"]) for row in selected],
        "packed_token_estimate": used_tokens,
        "packed_count": len(selected),
    }


def segment_sentences(df: DataFrame, *, source_id_col: str, text_col: str) -> DataFrame:
    """Context DSL primitive: segment documents into sentence-like units."""
    return (
        df.with_column("segments", split_sentences(col(text_col)))
        .explode("segments")
        .select(
            col(source_id_col).alias("source_id"),
            col(source_id_col).alias("parent_unit_id"),
            col("segments")["order_key"].alias("order_key"),
            col("segments")["payload"].alias("payload"),
        )
        .with_column("unit_id", monotonically_increasing_id())
        .with_column("token_estimate", estimate_tokens(col("payload")))
    )


def score_query_overlap(
    queries: DataFrame,
    units: DataFrame,
    *,
    query_id_col: str = "query_id",
    query_col: str = "query",
) -> DataFrame:
    """Context DSL primitive: annotate candidate units with a query-conditioned score."""
    return (
        queries.join(units, how="cross")
        .with_column("score", lexical_overlap_score(col(query_col), col("payload")))
        .select(
            query_id_col,
            query_col,
            "source_id",
            "parent_unit_id",
            "unit_id",
            "order_key",
            "payload",
            "token_estimate",
            "score",
        )
    )


def pack_context(
    candidates: DataFrame,
    *,
    group_by: list[str],
    max_tokens: int,
    delimiter: str = "\n\n",
) -> DataFrame:
    """Context DSL primitive: pack scored units into a bounded context artifact."""
    return (
        candidates.groupby(*group_by)
        .agg(
            col("unit_id").list_agg().alias("unit_ids"),
            col("payload").list_agg().alias("payloads"),
            col("token_estimate").list_agg().alias("token_estimates"),
            col("score").list_agg().alias("scores"),
            col("order_key").list_agg().alias("order_keys"),
        )
        .with_column(
            "packed",
            greedy_pack(
                col("unit_ids"),
                col("payloads"),
                col("token_estimates"),
                col("scores"),
                col("order_keys"),
                max_tokens=max_tokens,
                delimiter=delimiter,
            ),
        )
        .select(
            *group_by,
            col("packed")["packed_text"].alias("packed_text"),
            col("packed")["packed_unit_ids"].alias("packed_unit_ids"),
            col("packed")["packed_token_estimate"].alias("packed_token_estimate"),
            col("packed")["packed_count"].alias("packed_count"),
        )
    )
