# /// script
# description = "Runnable demo of the egodex facade: raw EgoDex HDF5 -> a queryable hand-pose dataset."
# requires-python = ">=3.12, <3.13"
# dependencies = [
#     "daft[transformers,hdf5,video]",
#     "sentencepiece",
# ]
#
# [tool.uv]
# prerelease = "allow"
# extra-index-url = ["https://nightly.daft.ai"]
# ///
"""Runnable demo of the egodex facade: raw EgoDex HDF5 -> a queryable hand-pose dataset.

The whole pipeline is a handful of lines; each is one stage (see egodex.py for the
Daft details under each). Copy-paste and run:

    uv run egodex_demo.py

Two feature branches come off one read of the raw HDF5 (no LeRobot, no Hugging Face):
  - pose geometry at 30 fps  (add_state_features -> add_skeleton_features)
  - SigLIP-2 embeddings at ~1 fps  (embed_frames)
queried together at the end. `read_egodex` uses Daft's native Hdf5File type and needs a
Daft build that has it (nightly / >= next release); `with_video=True` also needs `av`.
"""

import os

from egodex_lib.egodex import (
    read_egodex,
    add_state_features,
    add_skeleton_features,
    embed_frames,
    calibrate,
    query,
)

# Raw EgoDex HDF5 episodes (each <n>.hdf5 has a sibling <n>.mp4). Point this at your
# download; RAW_HDF5 can be overridden to run the demo on a subset.
RAW_HDF5 = os.environ.get("RAW_HDF5", os.path.join(os.path.dirname(__file__), ".data", "**", "*.hdf5"))


if __name__ == "__main__":
    # ── Build the features ──────────────────────────────────────────────────────
    # 1. Raw EgoDex HDF5 -> one row per frame (native Hdf5File, no LeRobot round-trip).
    #    with_video=True adds a lazily-decoded observation.image column for SigLIP.
    df = read_egodex(RAW_HDF5, with_video=True)

    # 2. SigLIP-2 image embeddings (~1 fps) — a separate semantic branch.
    emb = embed_frames(df)
    emb.select("episode_index", "frame_index", "clip_emb").write_parquet("embeddings/")

    # 3. per-frame geometry (closure, flexion, thumb distances, ...)
    feats = add_state_features(df)

    # 4. + action rates over frames (curl_rate, wrist_speed, roll, ...)
    feats = add_skeleton_features(feats)

    # 5. continuous pose features (30 fps) — compute once
    feats.write_parquet("features/")

    # ── Query ───────────────────────────────────────────────────────────────────
    import daft

    features = daft.read_parquet("features/")
    embeddings = daft.read_parquet("embeddings/")
    thresholds = calibrate(features)

    # 6a. rank by a hand-pose scenario (match count)
    pose_hits = query(features, pose="writing_grip", k=5, thresholds=thresholds)

    # 6b. rank by semantic text (facade encodes it with SigLIP)
    text_hits = query(embeddings, text="chopsticks", k=5)

    # 6c. pose AND text: join so each frame carries both pose features and an embedding,
    #     filter by the grip, then rank what's left by visual similarity.
    frames = features.join(embeddings, on=["episode_index", "frame_index"])
    combined_hits = query(frames, pose="hammer_grip", text="stapler", k=5, thresholds=thresholds)

    # ── Display ─────────────────────────────────────────────────────────────────
    for title, hits in [
        ("writing_grip", pose_hits),
        ("text 'chopsticks'", text_hits),
        ("hammer_grip + 'stapler'", combined_hits),
    ]:
        print(f"\n## {title}")
        for hit in hits:
            print(
                f"  episode {hit['episode_index']}: score {hit['score']:.3f}, "
                f"{hit['n_frames']} frames, segments {hit['segments'][:3]}"
            )
