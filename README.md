# daft-examples

This repository is an examples hub for Multimodal AI Workloads including text, image, audio, video, and documents.

## Getting Started

This project leverages [uv scripts](https://docs.astral.sh/uv/guides/scripts/) to make running examples easy. 

You can run any script like:

```bash
uv run path/to/script.py
```

If you don't have `uv`:

```bash
pip install uv
```

### Environment variables
Some examples require credentials. Create a `.env` file from in the repo root with the keys you need:

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
