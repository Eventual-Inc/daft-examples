# Daft Examples

*The fastest way to get started with [Daft](https://github.com/Eventual-Inc/Daft)*

[![Test Quickstart](https://github.com/Eventual-Inc/daft-examples/actions/workflows/test-quickstart.yml/badge.svg)](https://github.com/Eventual-Inc/daft-examples/actions/workflows/test-quickstart.yml)
[![Test Patterns](https://github.com/Eventual-Inc/daft-examples/actions/workflows/test-patterns.yml/badge.svg)](https://github.com/Eventual-Inc/daft-examples/actions/workflows/test-patterns.yml)

---

## 🚀 Quickstart (< 5 minutes)

New to Daft? Start here. These examples run in under 30 seconds and demonstrate core capabilities:

| Example | Runtime | What you'll learn |
|---------|---------|-------------------|
| **[01. Hello World](quickstart/01_hello_world_prompt.py)** | <10s | Basic text classification with LLM prompts |
| **[02. Semantic Search](quickstart/02_semantic_search.py)** | ~30s | PDF → embeddings → vector search pipeline |
| **[03. Data Enrichment](quickstart/03_data_enrichment.py)** | ~20s | ETL with LLM-based data enrichment |
| **[04. Audio Files](quickstart/04_audio_file.py)** | ~20s | Audio file processing with `daft.File` |
| **[05. Video Files](quickstart/05_video_file.py)** | ~15s | Video metadata and frame extraction |

```bash
# Clone and setup
git clone https://github.com/Eventual-Inc/daft-examples.git
cd daft-examples
make setup

# Run any example
uv run quickstart/01_hello_world_prompt.py
```

📖 **[Full quickstart guide →](quickstart/README.md)**

---

## 📂 Repository Structure

```
daft-examples/
├── quickstart/          # 🎯 Start here (5 examples, <30s each)
├── patterns/            # 🧩 Atomic feature demonstrations
│   ├── prompt/          # LLM prompting patterns
│   ├── embed/           # Embeddings and similarity search
│   ├── classify/        # Classification tasks
│   ├── io/              # File I/O operations
│   ├── daft_file/       # daft.File abstraction examples
│   ├── udfs/            # User-defined functions
│   └── commoncrawl/     # Common Crawl data processing
├── use_cases/           # 🏗️ Complete end-to-end pipelines
│   ├── voice_ai_analytics/
│   ├── social_recommendation/
│   ├── ai_visibility_tracking.py
│   ├── key_moments_extraction.py
│   ├── shot_boundary_detection.py
│   ├── embed_docs.py
│   ├── rag/
│   ├── code/
│   └── context_engineering/
├── models/              # 🤖 Model integrations
└── notebooks/           # 📓 Interactive tutorials
```

---

## 🧩 Patterns

Small, focused examples demonstrating specific Daft features. Perfect for learning individual capabilities.

### Prompt
- **[prompt.py](patterns/prompt/prompt.py)** - Basic prompting with anime classification
- **[prompt_structured_outputs.py](patterns/prompt/prompt_structured_outputs.py)** - Pydantic models for structured LLM outputs
- **[prompt_chat_completions.py](patterns/prompt/prompt_chat_completions.py)** - Chat-style completions with personas
- **[prompt_files_images.py](patterns/prompt/prompt_files_images.py)** - Multimodal prompting (text + images + PDFs)
- **[prompt_pdfs.py](patterns/prompt/prompt_pdfs.py)** - PDF document analysis
- **[prompt_session.py](patterns/prompt/prompt_session.py)** - Custom provider configuration
- **[prompt_openai_web_search.py](patterns/prompt/prompt_openai_web_search.py)** - Web search integration

### Embeddings
- **[embed_images.py](patterns/embed/embed_images.py)** - Image embeddings with Apple AIMv2
- **[embed_text_providers.py](patterns/embed/embed_text_providers.py)** - Compare embedding providers
- **[cosine_similarity.py](patterns/embed/cosine_similarity.py)** - Semantic similarity search

### Classification
- **[classify_image.py](patterns/classify/classify_image.py)** - Image classification with CLIP
- **[classify_text.py](patterns/classify/classify_text.py)** - Multi-label text classification

### I/O & File Handling
- **[read_audio_file.py](patterns/io/read_audio_file.py)** - Audio file reading and resampling
- **[read_pdfs.py](patterns/io/read_pdfs.py)** - PDF discovery and download
- **[read_video_files.py](patterns/io/read_video_files.py)** - Video metadata and keyframe extraction
- **[daft_file/](patterns/daft_file/)** - Complete `daft.File` examples (audio, video, PDF, code)

### UDFs
- **[daft_func.py](patterns/udfs/daft_func.py)** - Simple function-based UDFs
- **[daft_cls_with_types.py](patterns/udfs/daft_cls_with_types.py)** - Class-based UDFs with TypedDict/Pydantic

### Common Crawl
- **[chunk_embed.py](patterns/commoncrawl/chunk_embed.py)** - Text chunking and embedding
- **[show.py](patterns/commoncrawl/show.py)** - Query and filter MIME types

---

## 🏗️ Use Cases

Complete end-to-end pipelines demonstrating real-world applications.

### 🎤 Voice & Audio
- **[voice_ai_analytics/](use_cases/voice_ai_analytics/)** - Transcription → summarization → translation → RAG Q&A
- **[key_moments_extraction.py](use_cases/key_moments_extraction.py)** - Extract key moments from audio and generate clips

### 🖼️ Vision & Multimodal
- **[shot_boundary_detection.py](use_cases/shot_boundary_detection.py)** - Video scene detection using frame embeddings
- **[image_understanding_eval/](use_cases/image_understanding_eval/)** - Multimodal structured outputs evaluation

### 📚 RAG & Search
- **[rag/](use_cases/rag/)** - Minimal RAG implementation (PDF → embeddings → semantic search)
- **[context_engineering/arxiv_search/](use_cases/context_engineering/arxiv_search/)** - Semantic ArXiv paper search with Turbopuffer

### 💻 Code Analysis
- **[code/cursor.py](use_cases/code/cursor.py)** - Code analysis and IDE integration
- **[embed_docs.py](use_cases/embed_docs.py)** - Python codebase analysis with embeddings

### 🔍 Analytics & Benchmarking
- **[ai_visibility_tracking.py](use_cases/ai_visibility_tracking.py)** - Track brand mentions across multiple LLMs
- **[context_engineering/llm_judge_elo.py](use_cases/context_engineering/llm_judge_elo.py)** - LLM-as-judge ranking with ELO scores

### 🔗 Social & Recommendations
- **[social_recommendation/](use_cases/social_recommendation/)** - Reddit data ingestion and image recommendation pipeline

---

## 📓 Notebooks

Interactive tutorials for learning Daft:

- **[getting_started_with_common_crawl.ipynb](notebooks/getting_started_with_common_crawl.ipynb)** - Common Crawl introduction
- **[voice_ai_analytics.ipynb](notebooks/voice_ai_analytics.ipynb)** - Voice AI analytics walkthrough
- **[window_functions.ipynb](notebooks/window_functions.ipynb)** - Window functions tutorial
- **[mm_structured_outputs.ipynb](notebooks/mm_structured_outputs.ipynb)** - Multimodal structured outputs
- **[minhash_dedupe_common_crawl.ipynb](notebooks/minhash_dedupe_common_crawl.ipynb)** - MinHash deduplication

---

## 🛠️ Setup & Requirements

### Installation

```bash
# Clone repository
git clone https://github.com/Eventual-Inc/daft-examples.git
cd daft-examples

# Setup environment
make setup
```

### Running Examples

This project uses [uv scripts](https://docs.astral.sh/uv/guides/scripts/) for dependency isolation:

```bash
# Run any example
uv run quickstart/01_hello_world_prompt.py
uv run patterns/prompt/prompt.py
uv run use_cases/ai_visibility_tracking.py
```

### System Dependencies

Some examples require:
- **FFmpeg** - For audio/video processing (required by `soundfile`, `PyAV`)
- **API Keys** - Set in `.env` file:
  - `OPENAI_API_KEY` - OpenAI models
  - `OPENROUTER_API_KEY` - OpenRouter multi-model access
  - `TURBOPUFFER_API_KEY` - Vector search
  - `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` - S3 access

Create `.env` from template:
```bash
cp .env.example .env
# Edit .env with your API keys
```

---

## 🎯 Dynamic Batching

Daft includes automatic batch size tuning for optimal throughput:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                           Introducing Dynamic Batching: Auto-Tuning for Daft Pipelines                            ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
   stream in                           auto-tune loop                        work out
┌───────────────┐    rows       ┌──────────────────────────┐     batch    ┌────────────┐              ┌─────────┐
│   Source(s)   │ ────────────▶ │   Buffer + Dispatcher    │ ───────────▶ │  Operator  │ ───────────▶ | Results |
└───────────────┘               │  collects until ready    │              │ (UDF/Model)│              └─────────┘
                                │  lower..upper bounds     │              └─────┬──────┘
                                └──────────────────────────┘                    │
                                              ^  timing / memory / progress     │
                                              │  stats per batch                │
                                              │                                 ▼
                                    ┌──────────────────────────┐       updates   ┌───────────┐
                                    │      Batch Manager       │ ◀────────────── │  Metrics  │
                                    │  hit latency target (W)  │ ──────────────▶ │  + Logs   │
                                    │  grow/shrink batch (N)   │     new bounds  └───────────┘
                                    └──────────────────────────┘

                 small batches → fast first output + frequent progress
                 big batches   → high throughput (without hand tuning)
```

---

## 📚 Resources

- **[Daft Documentation](https://www.getdaft.io/docs/)** - Official docs
- **[GitHub](https://github.com/Eventual-Inc/Daft)** - Main Daft repository
- **[Discord](https://discord.gg/daft)** - Community support

---

## 🧪 Testing & CI

All examples are automatically tested via GitHub Actions:
- **Quickstart examples** - Tested on every push and PR
- **Patterns & use cases** - Tested daily to catch regressions
- **CI status** - See badges above

**Local testing:**
```bash
# Test a single example
uv run quickstart/01_hello_world_prompt.py

# Test all quickstart examples
for example in quickstart/*.py; do uv run "$example"; done
```

**CI Documentation:** See [`.github/CI-SETUP.md`](.github/CI-SETUP.md) for:
- Adding new examples to CI
- Configuring secrets
- Troubleshooting failures
- Future Daft Cloud testing

---

## 🤝 Contributing

Contributions welcome! Please open an issue or PR for:
- New examples
- Bug fixes
- Documentation improvements
- Feature requests

**Before submitting:**
1. Test your example locally: `uv run your_example.py`
2. Ensure it runs in <2 minutes
3. Add to appropriate CI workflow if needed
4. Update README with your example

---

## 📄 License

Apache 2.0
