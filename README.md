# daft-examples

This repository is a central hub for multimodal ai workloads including text, image, audio, video, and documents.

## Getting Started

This project leverages [uv scripts](https://docs.astral.sh/uv/guides/scripts/) to make running examples easy. If you don't have `uv`:

```bash
pip install uv
```

Run any script like:

```bash
uv run path/to/script.py
```

### Environment variables
Some examples require credentials. Create a `.env` in the repo root with the keys you need:

```bash
# OpenAI (for speech analytics)
OPENAI_API_KEY=sk-...

# AWS (for Common Crawl access; requester pays)
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

### System dependencies
- Recommended: `ffmpeg` for video/audio handling (required by some `av`/`soundfile` use-cases).
  - macOS (Homebrew): `brew install ffmpeg`

## Examples

- Read PDFs
  - Read PDF files into a Daft dataframe.
  - Run: `uv run read_files/read_pdfs.py`

- Embed text from PDFs
  - Extract text per page, sentence-chunk with spaCy, embed with Sentence-Transformers, and save to `.data/embed_text`.
  - Run: `uv run embed_text/embed_pdf_text.py`

- Embed images (parquet → embeddings)
  - Read images from a Hugging Face parquet dataset and compute image embeddings with Transformers; writes to `.data/embed_images` and displays.
  - Run: `uv run embed_images/embed_images.py`

- Embed video frames from YouTube
  - Download frames from a YouTube video and compute image embeddings for each frame.
  - Run: `uv run embed_images/embed_video_frames.py`

- Video shot boundary detection (embeddings + windows)
  - Compute per-frame image embeddings, derive L2 deltas with window functions, and visualize potential cut/dissolve boundaries.
  - Run: `uv run embed_images/shot_boundary_detection.py`

- Extract audio from video and write MP3
  - Read audio tracks from sample videos and persist MP3s to `.data/audio`.
  - Run: `uv run read_video/read_audio_from_video.py`
  - Note: This script references a helper `read_video_files.read_video_audio`. If not present in your checkout, treat this example as WIP or supply an equivalent reader.

- Common Crawl: explore content types
  - Query Common Crawl and show top MIME types (requester pays).
  - Run: `uv run commoncrawl/show.py`

- Common Crawl: chunk + embed text
  - Load text content, sentence-chunk with spaCy, embed with Sentence-Transformers, and display.
  - Run: `uv run commoncrawl/chunk_embed.py`
  - Requires: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` in `.env`.

- Speech analytics (OpenAI)
  - Transcribe MP3s with Whisper, summarize and translate with LLMs, embed text, and save to Lance at `.data/voice_ai`.
  - Run: `uv run speech/speech_analytics_openai.py`
  - Requires: `OPENAI_API_KEY` in `.env`.

- Template
  - Minimal Daft script you can copy to start new examples.
  - Run: `uv run TEMPLATE/main.py`

### Data locations
- Examples commonly write outputs under `.data/`. Clean or inspect as needed.

### Notes
- Scripts will install and use their declared dependencies via `uv` headers.
- Some scripts download spaCy models at runtime (e.g., `en_core_web_sm`).