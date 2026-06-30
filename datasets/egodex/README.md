# EgoDex: scenario search over hand-pose video with Daft

Find specific physical **states** and **actions** in egocentric hand-manipulation video —
"a writing-gripped hand", "the wrist twisting", "fingers closing into a grasp" — that a
text-only search can't reach. This example reads [Apple's EgoDex](https://github.com/apple/ml-egodex)
through Daft's LeRobot v3 reader, precomputes per-frame geometry and SigLIP semantic embeddings,
and serves a query UI that combines a geometric filter with semantic ranking.

## Layout

```
datasets/egodex/
├── lib/                    # importable modules — import these, don't run directly
│   ├── egodex.py           # facade API (convert, features, query, overlay)
│   ├── egodex_lerobot.py   # HDF5 → LeRobot v3 convert logic
│   ├── clip_features.py    # SigLIP-2 image/text embedding UDF
│   ├── pose_features.py    # 48-D state geometry (NumPy)
│   └── skeleton_features.py  # 204-D skeleton geometry (NumPy)
├── egodex_demo.py          # end-to-end pipeline script
├── egodex_demo.ipynb       # same pipeline, one cell per step
├── query_ui.py             # Gradio search UI
└── README.md
```

**`lib/`** holds the reusable logic. **`egodex.py`** is the public facade — scripts and notebooks
import from it rather than reaching into the lower-level modules directly.

**Scripts** are the entry points you run. Each declares its own dependencies inline (PEP 723),
so `uv run <script>.py` installs what it needs without a separate `requirements.txt`.

| Script | What it does |
| --- | --- |
| `egodex_demo.py` | Full pipeline in one file: convert HDF5 → LeRobot, embed frames, compute pose features, run example queries. |
| `egodex_demo.ipynb` | Same pipeline step-by-step — read, embed, add geometry, write parquet, calibrate, query, overlay, launch the UI. |
| `query_ui.py` | Interactive Gradio app over **precomputed** feature + embedding parquets. Filters by hand-pose scenario, ranks by semantic text, plays matched video segments. |

| Library module | What it does |
| --- | --- |
| `lib/egodex.py` | Thin facade: `convert_egodex_to_lerobot`, `add_state_features`, `add_skeleton_features`, `embed_frames`, `calibrate`, `query`, `overlay`. |
| `lib/egodex_lerobot.py` | Reads raw EgoDex HDF5 via Daft's `Hdf5File` type, writes a LeRobot v3 dataset (48-D `observation.state`, 204-D `observation.skeleton`, video). |
| `lib/clip_features.py` | SigLIP-2 `@daft.cls` embedder — produces unit-norm `clip_emb` vectors (~1 fps). |
| `lib/pose_features.py` | Per-frame geometry from the 48-D state (curl, wrist, palm normal, pinch, aperture). |
| `lib/skeleton_features.py` | Per-frame geometry from the 204-D skeleton (finger flexion, palm plane, arm extension, grip predicates). |

## Pipeline

```
raw EgoDex HDF5
   │  egodex.convert_egodex_to_lerobot()     # lib/egodex_lerobot.py
   ▼
LeRobot v3 dataset
   │  egodex_demo.py  (or the notebook)
   │    embed_frames()              → embeddings/   (SigLIP, ~1 fps, GPU)
   │    add_state_features()        ┐
   │    add_skeleton_features()      ┴→ features/   (geometry + action rates, 30 fps, CPU)
   ▼
query_ui.py                          # filter by pose scenario + rank by semantic text
```

Two feature branches come off the same LeRobot read:

- **Pose geometry (30 fps)** — `add_state_features` then `add_skeleton_features`. Continuous
  values only; scenario booleans (`writing_grip`, `grasping`, …) are computed at query time.
- **Semantic embeddings (~1 fps)** — `embed_frames` runs SigLIP-2 once and stores `clip_emb`
  so text queries are just a dot product later, with no GPU at query time.

## Run it

All commands assume you are in `datasets/egodex/` and can import from `lib/`:

```bash
cd datasets/egodex
export PYTHONPATH=lib
```

### 1. End-to-end demo

```bash
uv run egodex_demo.py
```

This converts raw HDF5 (if present), embeds frames, computes pose features, writes
`features/` and `embeddings/`, then runs a few example queries.

Skip the convert step if you already have a LeRobot dataset — edit the script to point
`lerobot.read` at your on-disk copy.

**Requirements:** Daft ≥ 0.7.16 with the `Hdf5File`, LeRobot, and video extras
(`daft[transformers,hdf5,lerobot,video]`). The convert step also needs the `lerobot`
package for the dataset writer.

### 2. Step-by-step (notebook)

Open `egodex_demo.ipynb`. Each cell covers one stage: convert, read, embed, add geometry,
write parquet, calibrate thresholds, query by pose/text, overlay a match, launch the UI.

Set the notebook kernel's `PYTHONPATH` to include `lib/` (or export it in the setup cell).

### 3. Query UI

The UI reads precomputed parquets at startup — it does not re-run SigLIP or recompute
geometry on launch. Build them first with `egodex_demo.py` (or the notebook), then:

```bash
DATASET=./egodex_lerobot \
POSE_OUT=./features \
OUT=./embeddings \
uv run query_ui.py
```

| Env var | Default | Purpose |
| --- | --- | --- |
| `DATASET` | `./egodex_lerobot_full` | LeRobot v3 dataset root (video + raw pose for playback overlay) |
| `POSE_OUT` | `./out/pose_features` | Precomputed geometry parquet from `add_state_features` / `add_skeleton_features` |
| `OUT` | `./out/clip_features` | Precomputed SigLIP embeddings parquet |

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

Both branches, plus the SigLIP embeddings, are written to parquet once. At query time a pose
predicate is a boolean column scan (milliseconds, no model) and the text query is one dot product
against the stored embeddings — geometry narrows candidates and semantics rank them, with no GPU.

The UI adds segment detection (contiguous matching frames), looping video clips per match, and a
live skeleton overlay for inspection — but the query path itself is just scanning precomputed
parquet.
