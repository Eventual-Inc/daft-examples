# /// script
# description = "Near-duplicate paragraph-level dedupe over Common Crawl WET using MinHash + LSH banding + connected components."
# dependencies = ["daft[aws,pandas]", "python-dotenv"]
# ///

from __future__ import annotations

import os
import re

from dotenv import load_dotenv

import daft
from daft import DataFrame, col
from daft.functions import monotonically_increasing_id
from daft.io import IOConfig, S3Config


# ---------------------------
# Parameters (tune as needed)
# ---------------------------
CRAWL = "CC-MAIN-2025-33"
NUM_FILES = 1  # fetch a small number of WET shards for prototyping

# Paragraph parsing
MIN_PARA_CHARS = 200

# MinHash / LSH
K = 64
NGRAM_SIZE = 5

# LSH banding: B * R must equal K
R = 8
B = 8
assert B * R == K

# Connected components iteration limits
CC_MAX_ITERS = 30
LABEL_PROP_MAX_ITERS = 100

# Output
OUT_DIR = ".data/common_crawl/wet_paragraph_dedupe"


# ---------------------------
# Common Crawl auth (optional)
# ---------------------------
load_dotenv()

if os.environ.get("AWS_ACCESS_KEY_ID"):
    IN_AWS = True
    IOCONFIG = IOConfig(
        s3=S3Config(
            region_name="us-east-1",
            requester_pays=True,
            key_id=os.environ["AWS_ACCESS_KEY_ID"],
            access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            anonymous=False,
        )
    )
else:
    IN_AWS = False
    IOCONFIG = None


# ---------------------------
# Cheap paragraph splitting
# ---------------------------
@daft.func()
def split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs using blank-line boundaries.

    We keep this intentionally cheap:
    - normalize CRLF/CR -> LF
    - split on 1+ blank lines
    - strip and drop short/empty paragraphs
    """
    if text is None:
        return []
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    # one or more blank lines, allowing whitespace on blank lines
    parts = re.split(r"\n\s*\n+", t)
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if len(p) >= MIN_PARA_CHARS:
            out.append(p)
    return out


@daft.func()
def get_band_idx(bands: list[list[int]], B: int) -> list[int]:
    """Return [0..min(len(bands), B)-1] for aligning band index with exploded bands."""
    if bands is None:
        return []
    return list(range(min(len(bands), B)))


# ---------------------------
# Connected components (star contraction)
# ---------------------------
def _canonicalize_edges(edges: DataFrame) -> DataFrame:
    """Order edges so u < v and deduplicate for canonical representation."""
    return (
        edges.with_column("u_can", (col("u") < col("v")).if_else(col("u"), col("v")))
        .with_column("v_can", (col("u") < col("v")).if_else(col("v"), col("u")))
        .select(col("u_can").alias("u"), col("v_can").alias("v"))
        .distinct()
    )


def _edge_sets_equal(a: DataFrame, b: DataFrame) -> bool:
    """Set equality check for undirected edge lists after canonicalization."""
    a_can = _canonicalize_edges(a)
    b_can = _canonicalize_edges(b)
    left_minus = a_can.join(b_can, on=["u", "v"], how="anti").count_rows()
    right_minus = b_can.join(a_can, on=["u", "v"], how="anti").count_rows()
    return (left_minus == 0) and (right_minus == 0)


def _pairs_equal(a: DataFrame, b: DataFrame) -> bool:
    """Set equality for (u, rep) pairs."""
    left_minus = a.join(b, on=["u", "rep"], how="anti").count_rows()
    right_minus = b.join(a, on=["u", "rep"], how="anti").count_rows()
    return (left_minus == 0) and (right_minus == 0)


def _symmetrize(edges: DataFrame) -> DataFrame:
    """Make edge list undirected by adding reverse edges."""
    return edges.select("u", "v").union_all(edges.select(col("v").alias("u"), col("u").alias("v")))


def large_star(edges: DataFrame) -> DataFrame:
    """Large-star: for each u, connect neighbors v>u to m(u)=min({u}∪N(u))."""
    undirected = _symmetrize(edges)
    neigh = undirected.groupby("u").agg_list("v").with_column("nbrs", col("v"))

    neigh = neigh.with_column("m", col("nbrs").list.min())
    neigh = neigh.with_column(
        "m",
        col("m")
        .is_null()
        .if_else(col("u"), (col("u") < col("m")).if_else(col("u"), col("m"))),
    )

    return (
        neigh.explode("nbrs")
        .where(col("nbrs") > col("u"))
        .select(col("nbrs").alias("u"), col("m").alias("v"))
        .where(col("u") != col("v"))
        .distinct()
    )


@daft.func()
def _edge_struct(u: int, v: int) -> dict[str, int]:
    return {"u": u, "v": v}


def small_star(edges: DataFrame) -> DataFrame:
    """Small-star: orient each edge so u is larger endpoint, then connect all neighbors to min."""
    directed = (
        edges.select(
            (col("u") < col("v"))
            .if_else(_edge_struct(col("v"), col("u")), _edge_struct(col("u"), col("v")))
            .alias("e")
        )
        .select(col("e")["*"])
        .where(col("u") != col("v"))
        .distinct()
    )

    neigh = directed.groupby("u").agg_list("v").with_column("nbrs", col("v"))

    neigh = neigh.with_column("m", col("nbrs").list.min())
    neigh = neigh.with_column(
        "m",
        col("m")
        .is_null()
        .if_else(col("u"), (col("u") < col("m")).if_else(col("u"), col("m"))),
    )

    return (
        neigh.explode("nbrs")
        .select(col("nbrs").alias("u"), col("m").alias("v"))
        .where(col("u") != col("v"))
        .distinct()
    )


def connected_components(edges: DataFrame) -> DataFrame:
    """Compute component representatives using alternating Large-/Small-Star + min-label propagation.

    Returns a DataFrame with schema ["u", "rep"] where rep is the global minimum node id in u's component.
    """
    # Alternate until edge set stabilizes
    b = edges
    for _ in range(CC_MAX_ITERS):
        a = large_star(b)
        b_next = small_star(a)
        if _edge_sets_equal(b, b_next):
            b = b_next
            break
        b = b_next

    b_final = b

    # Build initial representative mapping from stabilized edges (may still have multiple local minima)
    nodes = b_final.select(col("u").alias("u")).union_all(b_final.select(col("v").alias("u"))).distinct()
    rep_map = b_final.groupby("u").agg(col("v").min().alias("rep"))
    assignments = (
        nodes.join(rep_map, on="u", how="left")
        .with_column("rep", col("rep").is_null().if_else(col("u"), col("rep")))
        .select("u", "rep")
        .distinct()
    )

    # Ensure a single global minimum label per component via label propagation on undirected edges
    E = _symmetrize(b_final)
    labels = assignments.select(col("u"), col("rep").alias("label"))

    for _ in range(LABEL_PROP_MAX_ITERS):
        nbr_min = (
            E.join(labels, left_on="v", right_on="u", how="left")
            .select(col("u").alias("node"), col("label"))
            .groupby("node")
            .agg(col("label").min().alias("nbr_min"))
        )

        labels_next = (
            labels.join(nbr_min, left_on="u", right_on="node", how="left")
            .with_column(
                "label",
                col("nbr_min")
                .is_null()
                .if_else(
                    col("label"),
                    (col("label") <= col("nbr_min")).if_else(col("label"), col("nbr_min")),
                ),
            )
            .select(col("u"), col("label"))
            .distinct()
        )

        a_pairs = assignments.select(col("u"), col("rep"))
        b_pairs = labels_next.select(col("u"), col("label").alias("rep"))
        if _pairs_equal(a_pairs, b_pairs):
            assignments = b_pairs
            break

        assignments = b_pairs
        labels = labels_next

    return assignments.select("u", "rep").distinct()


if __name__ == "__main__":
    # 1) Load Common Crawl WET (text extracts)
    df_wet = daft.datasets.common_crawl(
        crawl=CRAWL,
        content="wet",
        num_files=NUM_FILES,
        in_aws=IN_AWS,
        io_config=IOCONFIG,
    )

    # 2) Decode + split into paragraphs
    df_para = (
        df_wet.with_column("text", col("warc_content").try_decode("utf-8"))
        .drop_null(col("text"))
        # Some Common Crawl modes include WARC-Type; keep the filter if present.
        # (If the column doesn't exist in your Daft version/content mode, remove this line.)
        .where(col("WARC-Type") == "conversion")
        .with_column("paragraphs", split_paragraphs(col("text")))
        .explode("paragraphs")
        .with_column("paragraph", col("paragraphs"))
        .select("WARC-Record-ID", "paragraph")
        .with_column("node_id", monotonically_increasing_id())
    )

    # 3) MinHash signatures
    df_mh = (
        df_para.with_column(
            "norm",
            col("paragraph").normalize(
                remove_punct=True, lowercase=True, nfd_unicode=True, white_space=True
            ),
        )
        .with_column(
            "min_hashes",
            col("norm").minhash(num_hashes=K, ngram_size=NGRAM_SIZE, seed=42, hash_function="xxhash"),
        )
        .select("node_id", "WARC-Record-ID", "paragraph", "min_hashes")
    )

    # 4) LSH banding -> candidate edges
    df_bands = df_mh.with_column("bands", col("min_hashes").list.chunk(R))
    df_bands = df_bands.with_column("band_idx", get_band_idx(col("bands"), B)).explode(
        "bands", "band_idx"
    )

    df_grouped = df_bands.groupby(col("band_idx"), col("bands")).agg(
        col("node_id").agg_list().alias("nodes")
    )

    # Edges: connect each node in a bucket to the bucket's minimum node id
    df_edges = (
        df_grouped.with_column("u", col("nodes").list.min())
        .explode("nodes")
        .select("u", v=col("nodes"))
        .where(col("u") != col("v"))
        .where(~col("u").is_null())
        .where(~col("v").is_null())
        .distinct()
    )

    # 5) Connected components -> representative per paragraph
    assignments = connected_components(df_edges)

    # 6) Join assignments back and keep only reps (deduped paragraphs)
    df_labeled = df_mh.join(assignments, left_on="node_id", right_on="u", how="left").select(
        "node_id", "WARC-Record-ID", "paragraph", "rep"
    )

    deduped = df_labeled.where(col("node_id") == col("rep")).select(
        "WARC-Record-ID", "paragraph"
    )
    duplicates = df_labeled.where(col("node_id") != col("rep")).select(
        "WARC-Record-ID", "paragraph", "rep"
    )

    # Materialize + write outputs
    print("Writing outputs to:", OUT_DIR)
    deduped.write_parquet(f"{OUT_DIR}/deduped_paragraphs")
    duplicates.write_parquet(f"{OUT_DIR}/duplicate_paragraphs")

    print("Sample deduped paragraphs:")
    deduped.show(5)

    print("Sample duplicates:")
    duplicates.show(5)

