.PHONY: setup test-classify test-commoncrawl test-embed test-io test-prompt test-udfs test-all

# Environment setup
setup:
	uv venv .venv
	uv pip install daft openai pillow numpy ipykernel ipywidgets
	cp .env.example .env

# usage_patterns/classify
test-classify:
	uv run usage_patterns/classify/classify_image.py
	uv run usage_patterns/classify/classify_text.py

# usage_patterns/commoncrawl
test-commoncrawl:
	uv run usage_patterns/commoncrawl/chunk_embed.py
	uv run usage_patterns/commoncrawl/show.py

# usage_patterns/embed
test-embed:
	uv run usage_patterns/embed/cosine_similarity.py
	uv run usage_patterns/embed/embed_images.py
	uv run usage_patterns/embed/embed_pdf.py
	uv run usage_patterns/embed/embed_text_providers.py
	uv run usage_patterns/embed/embed_video_frames.py

# usage_patterns/io
test-io:
	uv run usage_patterns/io/read_audio_file.py
	uv run usage_patterns/io/read_pdfs.py
	uv run usage_patterns/io/read_video_files.py

# usage_patterns/prompt
test-prompt:
	uv run usage_patterns/prompt/prompt.py
	uv run usage_patterns/prompt/prompt_chat_completions.py
	uv run usage_patterns/prompt/prompt_files_images.py
	uv run usage_patterns/prompt/prompt_github.py
	uv run usage_patterns/prompt/prompt_openai_web_search.py
	uv run usage_patterns/prompt/prompt_pdfs.py
	uv run usage_patterns/prompt/prompt_qa.py
	uv run usage_patterns/prompt/prompt_session.py
	uv run usage_patterns/prompt/prompt_structured_outputs.py

# usage_patterns/udfs
test-udfs:
	uv run usage_patterns/udfs/daft_cls_with_types.py
	uv run usage_patterns/udfs/daft_func.py

test-all: test-classify test-commoncrawl test-embed test-io test-prompt test-udfs