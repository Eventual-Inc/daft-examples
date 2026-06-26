# EgoDex: scenario search over hand-pose video with Daft

Find specific physical **states** and **actions** in egocentric hand-manipulation video —
"a writing-gripped hand", "the wrist twisting", "fingers closing into a grasp" — that a
text-only search can't reach. This example reads [Apple's EgoDex](https://github.com/apple/ml-egodex)
through Daft's LeRobot v3 reader, precomputes per-frame geometry and SigLIP semantic embeddings,
and serves a query UI that combines a geometric filter with semantic ranking.

```
raw EgoDex HDF5
   │  egodex_lerobot.py            # convert → LeRobot v3 (48-D state + 204-D skeleton + video)
   ▼
LeRobot v3 dataset
   │  run_pose_features.py         # NumPy states + Daft-window action rates → out/pose_features
   │  run_clip_features.py         # SigLIP-2 image embeddings (GPU)        → out/clip_features
   ▼
query_ui.py                        # filter by pose scenario + rank by semantic text
```

## ⚠️ Dataset license

EgoDex is released under **CC-BY-NC-ND**: non-commercial use only, and **no redistribution of
derivatives**. These scripts contain no EgoDex data — you must obtain your own copy from
[apple/ml-egodex](https://github.com/apple/ml-egodex) and run the conversion locally. Do not
redistribute the converted dataset or clips derived from it.

## Files

| File | Purpose |
| --- | --- |
| `hdf5.py` | Vendored `daft.datasets` HDF5 reader. |
| `lerobot.py` | Vendored `daft.datasets.lerobot` v3 reader (one row per frame, lazy video decode). |
| `egodex_lerobot.py` | Convert raw EgoDex HDF5 → LeRobot v3, emitting 48-D `observation.state` and 204-D `observation.skeleton`. |
| `pose_features.py` | Per-frame features from the 48-D state (curl, wrist, palm normal, pinch, aperture). |
| `skeleton_features.py` | Per-frame features from the 204-D skeleton (finger flexion, palm plane, arm extension, grip predicates). |
| `run_pose_features.py` | The pipeline: NumPy per-frame **states** + **action** rates via Daft window functions → one parquet. |
| `clip_features.py` | SigLIP-2 image/text embedding UDF (`@daft.cls`). |
| `run_clip_features.py` | Embed frames once with SigLIP → embeddings parquet (the only GPU step). |
| `query_ui.py` | Gradio demo: filter by pose scenario, rank by semantic text, play matched segments. |

## Run it

```bash
# 0) Convert your licensed EgoDex HDF5 → a local LeRobot v3 dataset (one-time)
python egodex_lerobot.py                      # writes ./egodex_lerobot_full

# 1) Geometry precompute — Daft windows, CPU, a few minutes
DATASET=./egodex_lerobot_full python run_pose_features.py     # → out/pose_features

# 2) Semantic embeddings — SigLIP, needs a GPU
DATASET=./egodex_lerobot_full python run_clip_features.py     # → out/clip_features

# 3) Launch the query UI
DATASET=./egodex_lerobot_full python query_ui.py              # prints a local + share URL

# verify the geometry library
```

## How it works

**States** are per-frame geometry — finger flexion, palm orientation, grip type — computed in
NumPy directly from the sensor data, so a "writing grip" is a writing grip regardless of what the
camera sees. **Actions** are rates of change across frames (grasping = fingers closing, twisting =
forearm rolling), computed with Daft window functions:

```python
per_episode = Window().partition_by("episode_index").order_by("frame_index")
df = df.with_column(
    "curl_rate",
    (col("curl").lead(1).over(per_episode) - col("curl")) / SECONDS_PER_FRAME,  # grasping signal
)
```

Both, plus the SigLIP embeddings, are written to parquet once. At query time a pose predicate is a
boolean column scan (milliseconds, no model) and the text query is one dot product against the
stored embeddings — so the geometry narrows candidates and the semantics rank them, with no GPU.
