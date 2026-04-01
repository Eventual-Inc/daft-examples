# Daft Examples

*The fastest way to get started with [Daft](https://github.com/Eventual-Inc/Daft)*

[![CI](https://github.com/Eventual-Inc/daft-examples/actions/workflows/test.yml/badge.svg)](https://github.com/Eventual-Inc/daft-examples/actions/workflows/test.yml)

---

## Quickstart

```bash
git clone https://github.com/Eventual-Inc/daft-examples.git
cd daft-examples
make setup
uv run quickstart/01_hello_world_prompt.py
```

| # | Example | What you'll learn |
|---|---------|-------------------|
| 01 | [Hello World](quickstart/01_hello_world_prompt.py) | LLM prompts and text classification |
| 02 | [Semantic Search](quickstart/02_semantic_search.py) | PDF → embeddings → vector search |
| 03 | [Data Enrichment](quickstart/03_data_enrichment.py) | ETL with LLM-based enrichment |
| 04 | [Audio Files](quickstart/04_audio_file.py) | Audio processing with `daft.File` |
| 05 | [Video Files](quickstart/05_video_file.py) | Video metadata and frame extraction |

---

## Repository Structure

```
daft-examples/
├── quickstart/          # Start here (5 examples)
├── examples/            # Atomic feature demonstrations
│   ├── classify/        # Image and text classification
│   ├── commoncrawl/     # Common Crawl processing
│   ├── embed/           # Embeddings and similarity search
│   ├── files/           # daft.File (audio, video, PDF, code)
│   ├── io/              # File I/O (read PDFs, video frames)
│   ├── prompt/          # LLM prompting patterns
│   ├── sql/             # SQL and window functions
│   └── udfs/            # User-defined functions
├── pipelines/           # End-to-end multi-stage pipelines
│   ├── context_engineering/
│   ├── image_understanding_eval/
│   ├── rag/
│   ├── social_recommendation/
│   └── voice_ai_analytics/
├── datasets/            # Dataset processing (Common Crawl, LAION, TPC-H, Open Images)
├── models/              # Model integration helpers
├── notebooks/           # Interactive Jupyter tutorials
└── tests/               # Test infrastructure
```

---

## Examples

Small, focused scripts demonstrating individual Daft features.

### Prompt
| Script | Description |
|--------|-------------|
| [prompt.py](examples/prompt/prompt.py) | Basic prompting with classification |
| [prompt_structured_outputs.py](examples/prompt/prompt_structured_outputs.py) | Pydantic models for structured LLM outputs |
| [prompt_chat_completions.py](examples/prompt/prompt_chat_completions.py) | Chat-style completions with personas |
| [prompt_files_images.py](examples/prompt/prompt_files_images.py) | Multimodal prompting (text + images + PDFs) |
| [prompt_openai_web_search.py](examples/prompt/prompt_openai_web_search.py) | Web search tool integration |
| [prompt_qa.py](examples/prompt/prompt_qa.py) | Synthetic Q&A generation and judging |

### Embeddings
| Script | Description |
|--------|-------------|
| [embed_text.py](examples/embed/embed_text.py) | Text embeddings with OpenAI |
| [embed_images.py](examples/embed/embed_images.py) | Image embeddings with Apple AIMv2 |
| [embed_text_providers.py](examples/embed/embed_text_providers.py) | Compare embedding providers |
| [cosine_similarity.py](examples/embed/cosine_similarity.py) | Semantic similarity search |

### Files and I/O
| Script | Description |
|--------|-------------|
| [daft_file.py](examples/files/daft_file.py) | `daft.File` basics |
| [daft_audiofile.py](examples/files/daft_audiofile.py) | Audio file metadata and resampling |
| [daft_videofile.py](examples/files/daft_videofile.py) | Video file metadata and keyframes |
| [read_pdfs.py](examples/io/read_pdfs.py) | PDF discovery and download |
| [read_video_files.py](examples/io/read_video_files.py) | Video frame extraction with `daft.read_video_frames` |

### UDFs
| Script | Description |
|--------|-------------|
| [daft_func.py](examples/udfs/daft_func.py) | Simple function-based UDFs |
| [daft_cls_model.py](examples/udfs/daft_cls_model.py) | Class-based UDFs with model loading |
| [daft_cls_with_types.py](examples/udfs/daft_cls_with_types.py) | Class UDFs with TypedDict/Pydantic |
| [daft_func_async.py](examples/udfs/daft_func_async.py) | Async UDFs |

---

## Pipelines

Complete end-to-end pipelines demonstrating real-world applications.

| Pipeline | Description |
|----------|-------------|
| [chunking_strategies.py](pipelines/context_engineering/chunking_strategies.py) | Compare fixed-size, sentence, and paragraph chunking |
| [few_shot_example_selection.py](pipelines/context_engineering/few_shot_example_selection.py) | Embedding-based few-shot example selection |
| [lambda_mapreduce.py](pipelines/context_engineering/lambda_mapreduce.py) | Map-reduce summarization over PDFs |
| [llm_judge_elo.py](pipelines/context_engineering/llm_judge_elo.py) | LLM-as-judge ELO ranking |
| [full_rag.py](pipelines/rag/full_rag.py) | Full RAG: PDF extraction → embedding → retrieval → generation |
| [rag.py](pipelines/rag/rag.py) | Minimal RAG pipeline |
| [voice_ai_analytics.py](pipelines/voice_ai_analytics/voice_ai_analytics.py) | Transcription → summarization → translation → embeddings |
| [key_moments_extraction.py](pipelines/key_moments_extraction.py) | Extract key moments from audio |
| [shot_boundary_detection.py](pipelines/shot_boundary_detection.py) | Video scene detection with frame embeddings |
| [embed_docs.py](pipelines/embed_docs.py) | Codebase analysis with SpaCy chunking and embeddings |
| [ai_search.py](pipelines/ai_search.py) | PDF search with Turbopuffer |

---

## Datasets

Processing patterns for public datasets.

| Dataset | Scripts |
|---------|---------|
| [Common Crawl](datasets/common_crawl/) | WARC/WAT/WET parsing, text deduplication, chunk & embed |
| [LAION](datasets/laion/) | Image-text pairs, CLIP training, metadata |
| [Open Images](datasets/open_images/) | Image loading, resizing, vision models |
| [TPC-H](datasets/tpch/) | SQL queries, performance benchmarks |

---

## Setup

### Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- FFmpeg (for audio/video examples)

### Installation

```bash
git clone https://github.com/Eventual-Inc/daft-examples.git
cd daft-examples
make setup
```

### API Keys

Some examples require API keys. Create a `.env` file:

```bash
cp .env.example .env
```

| Variable | Used by |
|----------|---------|
| `OPENAI_API_KEY` | Prompt, embedding, and RAG examples |
| `OPENROUTER_API_KEY` | Multi-model LLM examples |
| `TURBOPUFFER_API_KEY` | Vector search pipelines |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | Common Crawl, TPC-H, Open Images |

### Running

```bash
uv run quickstart/01_hello_world_prompt.py
uv run examples/prompt/prompt.py
uv run pipelines/rag/full_rag.py
```

---

## Development

```bash
make format    # Auto-format with ruff
make lint      # Lint check
make check     # Lint + format check (CI runs this)
make test      # Run all tests
make test-no-creds  # Run tests that don't need API keys
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for script format guidelines and style reference.

---

## License

Apache 2.0
