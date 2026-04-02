# Quickstart Examples

Get started with Daft in under 5 minutes. These examples are designed to run quickly (<30 seconds) and demonstrate core capabilities.

## Examples

### 01. Hello World Prompt
**File:** `01_hello_world_prompt.py`
**Runtime:** <10 seconds
**Description:** Simple text classification using LLM prompts. Classifies anime quotes into genres.

```bash
uv run quickstart/01_hello_world_prompt.py
```

**What you'll learn:**
- Basic `daft.prompt()` usage
- Working with text data
- Simple classification tasks

---

### 02. Semantic Search
**File:** `02_semantic_search.py`
**Runtime:** 30-45 seconds
**Description:** End-to-end PDF processing pipeline: read PDFs → generate embeddings → semantic search → store in vector database (Turbopuffer).

```bash
uv run quickstart/02_semantic_search.py
```

**What you'll learn:**
- PDF file processing
- Text embeddings
- Vector database integration
- Semantic search patterns

---

### 03. Data Enrichment
**File:** `03_data_enrichment.py`
**Runtime:** 20-30 seconds
**Description:** ETL pipeline that reads, extracts, enriches data with LLM, and writes results.

```bash
uv run quickstart/03_data_enrichment.py
```

**What you'll learn:**
- Reading structured data
- LLM-based data enrichment
- Writing processed results
- CRUD loop patterns

---

### 04. Audio File Processing
**File:** `04_audio_file.py`
**Runtime:** ~20 seconds
**Description:** Work with audio files using `daft.File` - read metadata, extract audio properties.

```bash
uv run quickstart/04_audio_file.py
```

**What you'll learn:**
- `daft.File` abstraction
- Audio file handling
- Extracting audio metadata

**Requirements:** FFmpeg installed

---

### 05. Video File Processing
**File:** `05_video_file.py`
**Runtime:** 15-20 seconds
**Description:** Work with video files using `daft.File` - read frames, extract metadata.

```bash
uv run quickstart/05_video_file.py
```

**What you'll learn:**
- Video file handling
- Frame extraction
- Video metadata processing

**Requirements:** FFmpeg, `av` library

---

## Next Steps

After completing these quickstart examples, explore:

- **[patterns/](../patterns/)** - Atomic feature demonstrations (prompt, embed, classify, io, etc.)
- **[use_cases/](../use_cases/)** - Complete end-to-end pipelines
- **[notebooks/](../notebooks/)** - Interactive tutorials

## Requirements

Most examples need:
- Python 3.10+
- `uv` package manager (for dependency isolation)

Some examples require API keys (set in `.env`):
- `OPENAI_API_KEY` - For OpenAI models
- `TURBOPUFFER_API_KEY` - For vector search (example 02)

See main [README.md](../README.md) for full setup instructions.
