# Daft Examples

**Prompt an LLM. Embed a document. Process a video. It's all just a DataFrame.**

[![CI](https://github.com/Eventual-Inc/daft-examples/actions/workflows/test.yml/badge.svg)](https://github.com/Eventual-Inc/daft-examples/actions/workflows/test.yml)

[Daft](https://github.com/Eventual-Inc/Daft) is a data engine that treats AI operations as first-class citizens. Calling an LLM, generating embeddings, reading a PDF — these aren't afterthoughts bolted onto a DataFrame library. They're column expressions, right in the query plan.

This repo is 90+ runnable examples that show you how.

```python
df = daft.from_pydict({"text": ["Einstein was a brilliant scientist.", "Mozart was a brilliant pianist."]})

df = df.with_column("summary", prompt(df["text"], model="openai/gpt-4.1-mini"))
df = df.with_column("embedding", embed_text(df["text"], model="Qwen/Qwen3-Embedding-0.6B"))
```

## Get started

```bash
git clone https://github.com/Eventual-Inc/daft-examples.git
cd daft-examples
make setup    # installs deps, copies .env.example → .env
```

Add your OpenAI key to `.env`, then:

```bash
uv run quickstart/01_hello_world_prompt.py
```

That's it. Every script in this repo is self-contained — dependencies are declared inline via [PEP 723](https://peps.python.org/pep-0723/), so `uv run` handles everything.

---

## Quickstart

Five scripts, each under 30 seconds. Start here.

| # | Script | What it does |
|---|--------|-------------|
| 01 | [Hello World](quickstart/01_hello_world_prompt.py) | Classify text with an LLM prompt |
| 02 | [Semantic Search](quickstart/02_semantic_search.py) | PDF → embeddings → vector search → Turbopuffer |
| 03 | [Data Enrichment](quickstart/03_data_enrichment.py) | ETL pipeline with LLM-based enrichment |
| 04 | [Audio Files](quickstart/04_audio_file.py) | Read audio metadata, resample with `daft.File` |
| 05 | [Video Files](quickstart/05_video_file.py) | Extract video frames and metadata |

---

## Examples

Small, focused scripts. One concept each.

### Prompt — LLM as a column expression

| Script | What it shows |
|--------|---------------|
| [prompt.py](examples/prompt/prompt.py) | Basic classification — one function call |
| [prompt_structured_outputs.py](examples/prompt/prompt_structured_outputs.py) | Pydantic models for type-safe LLM output |
| [prompt_chat_completions.py](examples/prompt/prompt_chat_completions.py) | Chat-style completions with system personas |
| [prompt_files_images.py](examples/prompt/prompt_files_images.py) | Multimodal — send images and PDFs to the model |
| [prompt_pdfs.py](examples/prompt/prompt_pdfs.py) | Feed entire PDFs into the prompt |
| [prompt_openai_web_search.py](examples/prompt/prompt_openai_web_search.py) | Web search tool integration |
| [prompt_qa.py](examples/prompt/prompt_qa.py) | Synthetic Q&A generation with LLM-as-judge |
| [prompt_session.py](examples/prompt/prompt_session.py) | Stateful prompt sessions |
| [prompt_unity_catalog.py](examples/prompt/prompt_unity_catalog.py) | Prompt over Unity Catalog tables |
| [prompt_gemini3_code_review.py](examples/prompt/prompt_gemini3_code_review.py) | Automated code review with Gemini |

### Embed — vectors as a column expression

| Script | What it shows |
|--------|---------------|
| [embed_text.py](examples/embed/embed_text.py) | Text embeddings at multiple dimensions |
| [embed_images.py](examples/embed/embed_images.py) | Image embeddings with Apple AIMv2 |
| [embed_text_providers.py](examples/embed/embed_text_providers.py) | Compare embedding providers side by side |
| [embed_video_frames.py](examples/embed/embed_video_frames.py) | Embed individual video frames |
| [cosine_similarity.py](examples/embed/cosine_similarity.py) | Semantic similarity search |

### Files — audio, video, PDF, code as native types

| Script | What it shows |
|--------|---------------|
| [daft_file.py](examples/files/daft_file.py) | `daft.File` basics |
| [daft_audiofile.py](examples/files/daft_audiofile.py) | Audio metadata, resampling |
| [daft_audiofile_udf.py](examples/files/daft_audiofile_udf.py) | Custom audio processing UDF |
| [daft_videofile.py](examples/files/daft_videofile.py) | Video metadata and keyframes |
| [daft_videofile_stream.py](examples/files/daft_videofile_stream.py) | Streaming video frame extraction |
| [daft_file_pdf.py](examples/files/daft_file_pdf.py) | PDF parsing and page extraction |
| [daft_file_code.py](examples/files/daft_file_code.py) | Source code analysis |

### UDFs — bring your own logic

| Script | What it shows |
|--------|---------------|
| [daft_func.py](examples/udfs/daft_func.py) | Simple `@daft.func` UDF |
| [daft_func_async.py](examples/udfs/daft_func_async.py) | Async UDFs for I/O-bound work |
| [daft_func_batch.py](examples/udfs/daft_func_batch.py) | Batch-mode UDFs |
| [daft_cls_model.py](examples/udfs/daft_cls_model.py) | `@daft.cls` — load a model once, run it on every row |
| [daft_cls_with_types.py](examples/udfs/daft_cls_with_types.py) | Class UDFs with TypedDict and Pydantic |
| [daft_cls_async_client.py](examples/udfs/daft_cls_async_client.py) | Async class UDFs with persistent clients |

### SQL & analytics

| Script | What it shows |
|--------|---------------|
| [stocks.py](examples/sql/stocks.py) | Window functions on real stock data — moving averages, rankings, Golden Cross detection |

### Classify

| Script | What it shows |
|--------|---------------|
| [classify_text.py](examples/classify/classify_text.py) | Text classification |
| [classify_image.py](examples/classify/classify_image.py) | Image classification |

### I/O

| Script | What it shows |
|--------|---------------|
| [read_pdfs.py](examples/io/read_pdfs.py) | Discover and read PDFs from remote storage |
| [read_video_files.py](examples/io/read_video_files.py) | Frame-level video reading with `daft.read_video_frames` |

### Common Crawl

| Script | What it shows |
|--------|---------------|
| [cc_show.py](examples/commoncrawl/cc_show.py) | Browse Common Crawl data |
| [cc_chunk_embed.py](examples/commoncrawl/cc_chunk_embed.py) | Chunk and embed web pages |
| [cc_wet_paragraph_dedupe.py](examples/commoncrawl/cc_wet_paragraph_dedupe.py) | Paragraph-level deduplication at scale |

---

## Pipelines

End-to-end workflows. These are where things get interesting.

### RAG

| Pipeline | What it does |
|----------|-------------|
| [rag.py](pipelines/rag/rag.py) | Minimal RAG — embed, retrieve, generate |
| [full_rag.py](pipelines/rag/full_rag.py) | Full RAG — PDF extraction, PyMuPDF UDF, cross-join ranking, generation |

### Context engineering

| Pipeline | What it does |
|----------|-------------|
| [lambda_mapreduce.py](pipelines/context_engineering/lambda_mapreduce.py) | 6 long-context reasoning patterns as native query plans (search, summarize, classify, extract, QA, analyze) |
| [chunking_strategies.py](pipelines/context_engineering/chunking_strategies.py) | Compare fixed-size, sentence, and paragraph chunking |
| [few_shot_example_selection.py](pipelines/context_engineering/few_shot_example_selection.py) | Embedding-based few-shot selection |
| [llm_judge_elo.py](pipelines/context_engineering/llm_judge_elo.py) | LLM-as-judge with ELO ranking |

### Audio & video

| Pipeline | What it does |
|----------|-------------|
| [voice_ai_analytics.py](pipelines/voice_ai_analytics/voice_ai_analytics.py) | Transcription → summarization → translation → embeddings → RAG over transcripts |
| [key_moments_extraction.py](pipelines/key_moments_extraction.py) | Extract and clip key moments from audio transcripts |
| [shot_boundary_detection.py](pipelines/shot_boundary_detection.py) | Video scene detection with frame embeddings |

### Search & recommendations

| Pipeline | What it does |
|----------|-------------|
| [ai_search.py](pipelines/ai_search.py) | PDF search with Turbopuffer |
| [embed_docs.py](pipelines/embed_docs.py) | Codebase analysis with SpaCy chunking and embeddings |
| [data_enrichment.py](pipelines/data_enrichment.py) | LLM-powered data enrichment pipeline |

### Code

| Pipeline | What it does |
|----------|-------------|
| [prompt_github.py](pipelines/code/prompt_github.py) | Prompt over GitHub repos |
| [cursor.py](pipelines/code/cursor.py) | Code analysis pipeline |

---

## Datasets

Processing patterns for real public datasets — not toy data.

| Dataset | Scripts | What you'll process |
|---------|---------|-------------------|
| [Common Crawl](datasets/common_crawl/) | WARC, WAT, WET parsing, [text deduplication](datasets/common_crawl/text_deduplication.py), [chunk & embed](datasets/common_crawl/chunk_embed.py) | Billions of web pages |
| [LAION](datasets/laion/) | [Image-text pairs](datasets/laion/image_text_pairs.py), [CLIP training data](datasets/laion/clip_training.py), metadata | 5B+ image-text pairs |
| [Open Images](datasets/open_images/) | [Image loading](datasets/open_images/basic_images.py), [processing](datasets/open_images/image_processing.py), [vision models](datasets/open_images/vision_models.py) | 9M annotated images |
| [TPC-H](datasets/tpch/) | [SQL queries](datasets/tpch/sql_queries.py), [performance benchmarks](datasets/tpch/performance_test.py) | Industry-standard analytical benchmark |

---

## Setup

### Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- FFmpeg (for audio/video examples)

### API keys

Most examples need an OpenAI key. Some need more. Copy the example and fill in what you have:

```bash
cp .env.example .env
```

| Key | What uses it |
|-----|-------------|
| `OPENAI_API_KEY` | Most prompt, embed, and RAG examples |
| `OPENROUTER_API_KEY` | Multi-model and structured output examples |
| `TURBOPUFFER_API_KEY` | Vector search pipelines |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | Common Crawl, TPC-H, Open Images |
| `HF_TOKEN` | Private HuggingFace datasets |

### Running any example

```bash
uv run quickstart/01_hello_world_prompt.py
uv run examples/prompt/prompt.py
uv run pipelines/rag/full_rag.py
```

Every script declares its own dependencies. No extras to install.

---

## Development

```bash
make format        # auto-format with ruff
make lint          # lint check
make precommit     # lint + format check (runs on git commit)
make test          # run all tests
make test-no-creds # run tests that don't need API keys
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on adding new examples.

---

## License

Apache 2.0
