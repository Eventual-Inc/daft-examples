"""Runnable demo of the egodex facade: raw EgoDex HDF5 -> a queryable hand-pose dataset.

The whole pipeline is a handful of lines; each is one stage (see egodex.py for the
Daft details under each). Copy-paste and run:

    uv run egodex_demo.py

Two feature branches come off the same LeRobot read:
  - pose geometry at 30 fps  (add_state_features -> add_skeleton_features)
  - SigLIP-2 embeddings at ~1 fps  (embed_frames)
queried together at the end. Notes:
  - Step 1 (convert) needs a Daft build with the new `Hdf5File` API (nightly / >= next
    release) plus `pip install lerobot` for the write. If you already have the LeRobot
    dataset, skip step 1 and point `lerobot.read` at it directly.
"""
import daft
from daft.datasets import lerobot
from egodex import (convert_egodex_to_lerobot, add_state_features,
                    add_skeleton_features, embed_frames, query)

# ── Build ────────────────────────────────────────────────────────────────────
lerobot_dir = convert_egodex_to_lerobot("egodex/**/*.hdf5", repo_id="egodex", output_dir="egodex_lerobot/")  # 1. HDF5 -> LeRobot

df = lerobot.read(lerobot_dir)                       # 2. LeRobot -> DataFrame (one row per frame)
df = add_state_features(df)                          # 3. per-frame geometry  (closure, flexion, thumb distances, ...)
df = add_skeleton_features(df)                       # 4. + action rates over frames (curl_rate, wrist_speed, roll, ...)
df.write_parquet("features/")                        # 5. continuous pose features (30 fps) — compute once

emb = embed_frames(lerobot.read(lerobot_dir, load_video_frames="observation.image"))   # 6. SigLIP-2 image embeddings (~1 fps)
emb.select("episode_index", "frame_index", "clip_emb").write_parquet("embeddings/")     #    a separate semantic branch

# ── Query ────────────────────────────────────────────────────────────────────
features = daft.read_parquet("features/")            # load the features wherever you like
embeddings = daft.read_parquet("embeddings/")

pose_hits = query(features, pose="writing_grip", k=5)        # 7a. rank by a hand-pose scenario (match count)
text_hits = query(embeddings, text="chopsticks", k=5)        # 7b. rank by semantic text (facade encodes it with SigLIP)

# 7c. pose AND text: join so each frame carries both pose features and an embedding,
#     then filter by the grip and rank what's left by visual similarity.
frames = features.join(embeddings, on=["episode_index", "frame_index"])
combined_hits = query(frames, pose="hammer_grip", text="stapler", k=5)

for title, hits in [("writing_grip", pose_hits),
                    ("text 'chopsticks'", text_hits),
                    ("hammer_grip + 'stapler'", combined_hits)]:
    print(f"\n## {title}")
    for hit in hits:
        print(f"  episode {hit['episode_index']}: score {hit['score']:.3f}, "
              f"{hit['n_frames']} frames, segments {hit['segments'][:3]}")
