# EgoDex: scenario search over hand-pose video with Daft

Find specific physical **states** and **actions** in egocentric hand-manipulation video —
"a writing-gripped hand", "the wrist twisting", "fingers closing into a grasp" — that a
text-only search can't reach. This example reads [Apple's EgoDex](https://github.com/apple/ml-egodex)
**straight from its raw HDF5 files** using Daft's native `Hdf5File` type (no LeRobot round-trip),
precomputes per-frame geometry and SigLIP-2 semantic embeddings, and serves a query UI that
combines a geometric filter with semantic ranking.

## Layout

```
datasets/egodex/
├── egodex_lib/                     # importable modules — import these, don't run directly
│   ├── egodex.py                   # facade API (read, features, embed, query, overlay)
│   ├── convert_egodex_to_lerobot.py  # raw HDF5 → per-frame DataFrame (read_egodex); + LeRobot v3 writer
│   ├── egodex_video.py             # optional: adds the observation.image video feature to a LeRobot dataset
│   ├── clip_features.py            # SigLIP-2 image/text embedding UDF
│   ├── pose_features.py            # 48-D state geometry (NumPy)
│   └── skeleton_features.py        # 204-D skeleton geometry (NumPy)
├── egodex_demo.py                  # end-to-end pipeline script
├── egodex_demo.ipynb               # same pipeline, one cell per step
├── query_ui.py                     # Gradio search UI
└── README.md
```

**`egodex_lib/`** holds the reusable logic. **`egodex.py`** is the public facade — scripts and
notebooks import from it (`from egodex_lib.egodex import ...`) rather than reaching into the
lower-level modules directly.

**Scripts** are the entry points you run. `egodex_demo.py` declares its dependencies inline
(PEP 723), so `uv run egodex_demo.py` installs what it needs without a separate `requirements.txt`.

| Script | What it does |
| --- | --- |
| `egodex_demo.py` | Full pipeline in one file: read raw HDF5, embed frames, compute pose features, write parquet, run example queries. |
| `egodex_demo.ipynb` | Same pipeline step-by-step — read, embed, add geometry, write parquet, calibrate, query, overlay, launch the UI. |
| `query_ui.py` | Interactive Gradio app over **precomputed** feature + embedding parquets. Filters by hand-pose scenario, ranks by semantic text, plays matched video segments. |

| Library module | What it does |
| --- | --- |
| `egodex_lib/egodex.py` | Thin facade: `read_egodex`, `add_state_features`, `add_skeleton_features`, `embed_frames`, `calibrate`, `query`, `overlay`. |
| `egodex_lib/convert_egodex_to_lerobot.py` | `read_egodex(glob)` reads raw EgoDex HDF5 via Daft's `Hdf5File` type into one row per frame (48-D `observation.state`, 204-D `observation.skeleton`, 16-D `observation.extrinsics`, `action`, `task`). Also holds the optional LeRobot v3 dataset writer. |
| `egodex_lib/egodex_video.py` | Optional: adds the `observation.image` video feature to a written LeRobot dataset. Not needed for the raw-HDF5 demo. |
| `egodex_lib/clip_features.py` | SigLIP-2 `@daft.cls` embedder — produces unit-norm `clip_emb` vectors (~1 fps). |
| `egodex_lib/pose_features.py` | Per-frame geometry from the 48-D state (curl, wrist, palm normal, pinch, aperture). |
| `egodex_lib/skeleton_features.py` | Per-frame geometry from the 204-D skeleton (finger flexion, palm plane, arm extension, grip predicates). |

## Pipeline

```
raw EgoDex HDF5  (each <n>.hdf5 has a sibling <n>.mp4)
   │  egodex.read_egodex(glob, with_video=True)     # native Hdf5File, one row per frame
   ▼
per-frame DataFrame
   │  egodex_demo.py  (or the notebook)
   │    embed_frames()              → embeddings/   (SigLIP-2, ~1 fps, GPU if present)
   │    add_state_features()        ┐
   │    add_skeleton_features()      ┴→ features/   (geometry + action rates, 30 fps, CPU)
   ▼
query_ui.py                          # filter by pose scenario + rank by semantic text
```

Two feature branches come off the **same** raw-HDF5 read — no LeRobot, no Hugging Face dataset:

- **Pose geometry (30 fps)** — `add_state_features` then `add_skeleton_features`. Continuous
  values only; scenario booleans (`writing_grip`, `grasping`, …) are computed at query time.
- **Semantic embeddings (~1 fps)** — `embed_frames` runs SigLIP-2 once and stores `clip_emb`
  so text queries are just a dot product later, with no GPU at query time.

`with_video=True` adds a lazily-decoded `observation.image` column (each frame is pulled from the
sibling `.mp4`); only the ~1 fps frames `embed_frames` keeps are ever decoded.

## Run it

Run from `datasets/egodex/` so the `egodex_lib` package is importable:

```bash
cd datasets/egodex
```

### 1. End-to-end demo

```bash
uv run egodex_demo.py
```

This reads raw HDF5, embeds frames, computes pose features, writes `features/` and
`embeddings/`, then runs a few example queries. Point it at your download with the `RAW_HDF5`
env var (a glob; defaults to `.data/**/*.hdf5`), e.g. to run on a subset:

```bash
RAW_HDF5='.data/test/add_remove_lid/*.hdf5' uv run egodex_demo.py
```

**Requirements:** a Daft build with the native `Hdf5File` type (nightly / ≥ next release) plus
the transformers, hdf5, and video extras — `daft[transformers,hdf5,video]` — and `sentencepiece`
(for the SigLIP-2 text tokenizer). `egodex_demo.py` pins these inline via PEP 723, so `uv run`
handles them. `with_video=True` needs `av` (PyAV), pulled in by the `video` extra.

### 2. Step-by-step (notebook)

Open `egodex_demo.ipynb`. Each cell covers one stage: read, embed, add geometry, write parquet,
calibrate thresholds, query by pose/text, overlay a match, launch the UI. Run the notebook with a
kernel that can import `egodex_lib` (launch Jupyter from `datasets/egodex/`).

### 3. Query UI

The UI reads precomputed parquets at startup — it does not re-run SigLIP or recompute geometry on
launch. Build them first with `egodex_demo.py` (or the notebook), then point the UI at them:

```bash
OUT=./embeddings \
POSE_OUT=./features \
DATASET='.data/**/*.hdf5' \
uv run query_ui.py
```

| Env var | Default | Purpose |
| --- | --- | --- |
| `OUT` | `./out/clip_features` | Precomputed SigLIP-2 embeddings parquet (`clip_emb`) |
| `POSE_OUT` | `./out/pose_features` | Precomputed geometry parquet from `add_state_features` / `add_skeleton_features` |
| `DATASET` | `.data/**/*.hdf5` | Raw EgoDex HDF5 glob — the UI resolves `episode_index` against `sorted(glob(..., recursive=True))` (matching `read_egodex`) to load skeletons and play the sibling `.mp4` |

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

> **Note on `read_egodex` and laziness.** Daft is lazy, so `read_egodex(...)` only builds a plan;
> work happens on an action like `.write_parquet()`, `.show()`, or `.collect()`. Because the reader
> ends in an `explode` (episode → one row per frame), a trailing `limit`/`show(1)` or a
> `where(...)` on a per-frame column **cannot** short-circuit — Daft decodes *every* HDF5 file
> before trimming. For fast iteration, narrow the glob to a few files, or materialize once with
> `write_parquet` and read that back.
