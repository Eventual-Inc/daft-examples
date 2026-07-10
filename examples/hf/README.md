# Daft x Hugging Face Examples

Runnable Daft `0.7.19` patterns for the Daft x Hugging Face social series. Each
script has a [PEP 723](https://peps.python.org/pep-0723/) header, so run it with
`uv run examples/hf/<script>.py` from the repository root. Public inputs do not
require a Hugging Face token.

## Planned Social 1: Hub-native reads, `hf://`, and Storage Buckets

- Scripts: [`read_huggingface.py`](read_huggingface.py), [`read_hf_parquet.py`](read_hf_parquet.py), and [`read_hf_bucket_files.py`](read_hf_bucket_files.py)
- Daft guide: [Hugging Face datasets](https://docs.daft.ai/en/stable/connectors/huggingface/)
- Daft API: [`read_huggingface`](https://docs.daft.ai/en/stable/api/io/#daft.read_huggingface)
- Hugging Face references: [Storage Buckets](https://huggingface.co/docs/hub/storage-buckets) and [bucket access patterns](https://huggingface.co/docs/hub/en/storage-buckets-access)
- Grounded inputs: [`huggingface/documentation-images`](https://huggingface.co/datasets/huggingface/documentation-images), the Xet-backed [`google-research-datasets/mbpp`](https://huggingface.co/datasets/google-research-datasets/mbpp) Parquet shard, and the public [`commoncrawl`](https://huggingface.co/buckets/commoncrawl/commoncrawl/tree/main) bucket.

## Planned Social 2: Common Crawl from a Hugging Face Storage Bucket

- Script: [`common_crawl_bucket.py`](common_crawl_bucket.py)
- Daft guide: [Common Crawl](https://docs.daft.ai/en/stable/datasets/common-crawl/)
- Daft API: [`common_crawl`](https://docs.daft.ai/en/stable/api/datasets/#daft.datasets.common_crawl)
- Hugging Face reference: [Common Crawl bucket](https://huggingface.co/buckets/commoncrawl/commoncrawl/tree/main)
- Follow-on Daft example: [`examples/commoncrawl/cc_wet_paragraph_dedupe.py`](../commoncrawl/cc_wet_paragraph_dedupe.py) for the larger WET-text deduplication workflow.

The script deliberately sets `source="hf"`; `CC-MAIN-2026-25` is available in
the public bucket. It limits the crawl to one WET file, but that file can still
be large, so it is intentionally not part of the default smoke suite.

## Planned Social 3: LeRobot v3 and DROID

- Scripts: [`lerobot_v3.py`](lerobot_v3.py) and [`droid.py`](droid.py)
- Daft guides: [LeRobot v3](https://docs.daft.ai/en/stable/datasets/lerobot/) and [DROID](https://docs.daft.ai/en/stable/datasets/droid/)
- Hugging Face reference: [LeRobotDataset v3.0](https://huggingface.co/docs/lerobot/main/lerobot-dataset-v3)
- Grounded inputs: [`lerobot/aloha_sim_insertion_human`](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human), which has the v3 `meta/episodes`, `data`, and `videos` layout, and Daft's public [`Eventual-Inc/droid-scene-classifications`](https://huggingface.co/datasets/Eventual-Inc/droid-scene-classifications) mirror.

`lerobot_v3.py` is the runnable Hub example. `droid.py` prepares the
lazy DROID plan by default; set `RUN_DROID=1` to materialize raw DROID episode
metadata from the public GCS release.

## Planned Social 4: Local Transformers inference and embeddings

- Scripts: [`transformers_prompt.py`](transformers_prompt.py), [`transformers_embed_text.py`](transformers_embed_text.py), and [`transformers_embed_image.py`](transformers_embed_image.py)
- Daft guides: [prompt](https://docs.daft.ai/en/stable/ai-functions/prompt/) and [embeddings](https://docs.daft.ai/en/stable/ai-functions/embed/)
- Daft API: [`prompt`](https://docs.daft.ai/en/stable/api/functions/prompt/) and [AI functions](https://docs.daft.ai/en/stable/api/ai/)
- Hugging Face references: [Transformers pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines), [`HuggingFaceTB/SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), and [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32).

These run local models through `provider="transformers"`; no hosted inference
key is needed. The first invocation downloads the selected model weights.

## Release Pin

Every script pins Daft to `0.7.19`. This keeps snippets aligned with the
release train rather than silently testing against a newer API.
