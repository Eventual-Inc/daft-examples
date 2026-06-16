"""Query the materialized CLIP features — cheap, local, no GPU, no model on frames.

Reads the embeddings parquet that run_clip_features.py wrote, adds the `curl`
pose feature locally (free), encodes the text scenario ONCE, and ranks frames by
similarity to it combined with hand-curl. This is the "embed once, query many"
payoff: the CLIP image pass already happened; here we only do a text encode +
cosine over stored vectors.

  python query_local.py "a cup"
  CURL_MAX=0.12 SIM_MIN=0.0 python query_local.py "an open drawer"
"""

import os
import sys

import daft
from daft import DataType, col
from daft.functions import cosine_similarity

from clip_features import EMB_DIM, clip_text, hand_curl

OUT = os.environ.get("OUT", os.path.join(os.path.dirname(__file__), "out", "clip_features"))
if len(sys.argv) > 1:
    QUERY = sys.argv[1]
else:
    QUERY = "a cup"
CURL_MAX = float(os.environ.get("CURL_MAX", "1e9"))  # keep curled frames (small curl); default = no filter
SIM_MIN = float(os.environ.get("SIM_MIN", "-1e9"))  # similarity cutoff; default = no filter (rank instead)

feats = daft.read_parquet(OUT)
feats = feats.with_column("curl", hand_curl(col("observation.state")))  # pose feature, computed locally
text_emb = clip_text(QUERY)  # one tiny text encode -> list of EMB_DIM floats
# Type the constant text vector as an embedding so it matches clip_emb's dtype
# (cosine_similarity needs embedding/fixed-size-list, not a bare float64 list).
text_lit = daft.lit(text_emb).cast(DataType.embedding(DataType.float32(), EMB_DIM))
feats = feats.with_column("sim", cosine_similarity(col("clip_emb"), text_lit))

hits = feats.where(col("curl") < CURL_MAX)
hits = hits.where(col("sim") > SIM_MIN)

print(f"\nquery={QUERY!r}  (curl < {CURL_MAX}, sim > {SIM_MIN}); small curl = curled hand\n")

ranked = hits.select("episode_index", "frame_index", "curl", "sim")
ranked = ranked.sort("sim", desc=True)
ranked.show(15)
