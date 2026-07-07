# EgoDex Scenario Search

This example reads raw EgoDex HDF5 episodes directly with Daft, computes
queryable hand-pose tracks, optionally embeds sampled video frames with SigLIP,
and searches for physical scenarios by pose, text, or both.

## Workflow

1. `EgoDexPipeline.raw()` discovers `<task>/<episode>.hdf5` files and sibling
   MP4s. Optional task and episode filters are applied before metadata or tensor
   reads, which keeps notebook iteration cheap.
2. `trajectory()` reads the HDF5 transform datasets needed for pose features.
3. `calculate_features()` keeps one row per episode and stores each pose signal
   as an episode-length track.
4. `query()` evaluates state/action scenarios over those tracks and returns
   ranked hits with contiguous frame segments.
5. `camera_frames()` and `embed_frames()` provide the optional semantic branch
   for text-only and combined pose-plus-text search.
6. `overlay()` renders a matched frame by projecting the raw HDF5 skeleton onto
   the sibling video frame.

## Notebook

From the repository root, install the editable package and notebook kernel once:

```bash
uv sync --extra egodex --extra notebook
```

Select the repo `.venv` as the Jupyter kernel, then open `egodex_demo.ipynb`.
The notebook cwd should be the repository root so `datasets/egodex/.data`
resolves correctly.

It defaults to `EPISODE_LIMIT = 10` so a first run stays bounded. Set
`EPISODE_LIMIT = None` when you want every episode under `DATA_ROOT`.

## Dependencies

Install with `uv sync --extra egodex`. That pulls in Daft `>=0.7.17` with the
`transformers`, `video`, and `hdf5` extras plus SigLIP's PyTorch/Transformers
stack.
