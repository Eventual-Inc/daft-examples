<div align="center">

# Daft Examples

<i>An examples hub for running Multimodal AI Workloads on [Daft](https://github.com/Eventual-Inc/Daft)</i>

</div>

## Getting Started

This repository is organized into three sections:

1. **[Usage Patterns](#usage-patterns)** - Small atomic demonstrations of core features. 
2. **[Use Cases](#use-cases)** - Entire Pipelines built in Daft. 
3. **[Notebooks](#notebooks)** - End to End tutorials on working with Daft in an interactive Jupyter Notebook

To get started, run the following in your terminal.

```bash
git clone https://github.com/Eventual-Inc/daft-examples.git
cd daft-examples
make setup
```

# Running Examples

This project leverages [uv scripts](https://docs.astral.sh/uv/guides/scripts/) for dependency management isolation. This is because many of the scripts in this repository have different requirements. The venv that is created by the `setup` make target is just for type hints and notebooks. 

You can run any script like:

```bash
uv run usage_patterns/prompt/prompt.py
```

If you don't have `uv`, check out this [installation guide](https://docs.astral.sh/uv/getting-started/installation/).


### System dependencies
- Some examples leverage libraries like [`soundfile`](https://github.com/bastibe/python-soundfile) or [`PyAv`](https://github.com/PyAV-Org/PyAV) to process audio and video files which requires [`ffmpeg`](https://ffmpeg.org/download.html).


## Usage Patterns 

### Prompt
- [`prompt/prompt.py`](usage_patterns/prompt/prompt.py) - Basic prompt example using OpenRouter with anime quote classification
- [`prompt/prompt_chat_completions.py`](usage_patterns/prompt/prompt_chat_completions.py) - Use chat completions with personas from Nemotron dataset via LM Studio
- [`prompt/prompt_files_images.py`](usage_patterns/prompt/prompt_files_images.py) - Multimodal prompting with images and PDFs using Gemini 2.5 Flash via OpenRouter
- [`prompt/prompt_github.py`](usage_patterns/prompt/prompt_github.py) - Analyze markdown files from GitHub repositories with GPT-5
- [`prompt/prompt_openai_web_search.py`](usage_patterns/prompt/prompt_openai_web_search.py) - Perform web searches using OpenAI GPT-5 with web_search tools and store results in Supabase
- [`prompt/prompt_pdfs.py`](usage_patterns/prompt/prompt_pdfs.py) - Analyze PDF documents with GPT-5-nano reasoning models
- [`prompt/prompt_personas.py`](usage_patterns/prompt/prompt_personas.py) - Multi-persona business strategy analysis with web search and collective intelligence synthesis
- [`prompt/prompt_session.py`](usage_patterns/prompt/prompt_session.py) - Use custom OpenRouter provider through Daft sessions
- [`prompt/prompt_structured_outputs.py`](usage_patterns/prompt/prompt_structured_outputs.py) - Get structured outputs using Pydantic models with OpenRouter
- [`prompt/prompt_synthetic_customer_discovery.py`](usage_patterns/prompt/prompt_synthetic_customer_discovery.py) - Generate synthetic customer discovery responses from diverse personas using structured outputs

### Embeddings
- [`embed/embed_images.py`](usage_patterns/embed/embed_images.py) - Generate image embeddings using Apple's AIMv2 model
- [`embed/embed_pdf.py`](usage_patterns/embed/embed_pdf.py) - Extract text from PDFs, chunk with spaCy, and embed sentences
- [`embed/embed_text_providers.py`](usage_patterns/embed/embed_text_providers.py) - Compare text embeddings across providers (Transformers, OpenAI, LM Studio)
- [`embed/embed_video_frames.py`](usage_patterns/embed/embed_video_frames.py) - Extract and embed video frames from YouTube videos
- [`embed/shot_boundary_detection.py`](usage_patterns/embed/shot_boundary_detection.py) - Detect scene cuts and dissolves in videos using frame embeddings
- [`embed/similarity_Search.py`](usage_patterns/embed/similarity_Search.py) - Semantic search using cosine distance between text embeddings

### Classification
- [`classify/classify_image.py`](usage_patterns/classify/classify_image.py) - Classify images using CLIP model with custom labels
- [`classify/classify_text.py`](usage_patterns/classify/classify_text.py) - Multi-label text classification using BART model

### Common Crawl
- [`commoncrawl/chunk_embed.py`](usage_patterns/commoncrawl/chunk_embed.py) - Process Common Crawl data: chunk text with spaCy and embed sentences
- [`commoncrawl/show.py`](usage_patterns/commoncrawl/show.py) - Query and display MIME types from Common Crawl datasets

### UDFs
- [`udfs/cls_with_types.py`](usage_patterns/udfs/cls_with_types.py) - Class-based UDFs with TypedDict, Pydantic, batch processing, and async functions
- [`udfs/udf.py`](usage_patterns/udfs/udf.py) - Simple UDF example to extract file names from File objects

### I/O
- [`io/read_audio_file.py`](usage_patterns/io/read_audio_file.py) - Read and resample audio files, write to MP3 format
- [`io/read_pdfs.py`](usage_patterns/io/read_pdfs.py) - Discover and download PDF files from URLs
- [`io/read_video_files.py`](usage_patterns/io/read_video_files.py) - Extract video metadata, keyframes, and audio from video files

### Data Transformations
- [`data_transformations/explore_filter_aggregate.py`](usage_patterns/data_transformations/explore_filter_aggregate.py) - Data exploration, filtering, column transformations, and aggregations with Daft (Contributor: GitHub Copilot)

## Use Cases

### Transcription
- [`transcribe/faster_whisper_schema.py`](use_cases/transcription/faster_whisper_schema.py) - Schema definitions for Faster Whisper transcription results
- [`transcribe/key_moments_extraction.py`](use_cases/transcription/key_moments_extraction.py) - Extract key moments from transcripts and clip audio for short-form content
- [`transcribe/transcribe_faster_whisper.py`](use_cases/transcription/transcribe_faster_whisper.py) - Transcribe audio with Faster Whisper and Voice Activity Detection (VAD)

### Voice AI Analytics

- [`voice_ai_analytics/voice_ai_analytics.py`](use_cases/voice_ai_analytics/voice_ai_analytics.py) - Complete pipeline: transcription, summarization, translation to Chinese, segment embeddings, and RAG Q&A
- [`voice_ai_analytics/voice_ai_analytics_openai.py`](use_cases/voice_ai_analytics/voice_ai_analytics_openai.py) - Voice analytics with OpenAI Whisper, GPT-5 summarization, and Spanish translation
- [`voice_ai_analytics/voice_ai_tutorial.py`](use_cases/voice_ai_analytics/voice_ai_tutorial.py) - Voice AI analytics tutorial: transcription, summarization, Chinese translation, and embeddings

### Notebooks
- [`notebooks/getting_started_with_common_crawl.ipynb`](notebooks/getting_started_with_common_crawl.ipynb) - Interactive Common Crawl tutorial
- [`notebooks/voice_ai_analytics.ipynb`](notebooks/voice_ai_analytics.ipynb) - Voice AI analytics walkthrough
- [`notebooks/window_functions.ipynb`](notebooks/window_functions.ipynb) - Window functions examples and usage

