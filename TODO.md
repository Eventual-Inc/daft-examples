# Repo Cleanup TODO

## Code Bugs

- ~~`examples/io/read_audio_file.py`~~ — removed; filed [Daft#6532](https://github.com/Eventual-Inc/Daft/issues/6532) for AudioFile write support
- ~~`examples/io/read_video_files.py`~~ — rewrote using `daft.read_video_frames()`; filed [Daft#6533](https://github.com/Eventual-Inc/Daft/issues/6533) for VideoFile gaps
- ~~`examples/prompt/prompt_files_images.py`~~ — replaced local paths with HF datasets; also fixed `prompt.py`
- ~~`examples/prompt/prompt_openai_web_search.py`~~ — fixed: explicit unnest, removed broken embedding step, updated model
- ~~`pipelines/context_engineering/llm_judge_elo.py`~~ — rewritten: fixed cartesian product, concat, f-string conflicts, column names; needs valid OPENROUTER_API_KEY
- ~~`pipelines/image_understanding_eval/eval_image_understanding.py`~~ — removed broken `propose_new_question` stub, fixed truncated `if`, fixed `DEST_URI` → `dest_uri`
- ~~`pipelines/rag/full_rag.py`~~ — cleaned up garbage dict entries, removed unused SpaCy class, fixed invalid prompt syntax
- ~~`pipelines/rag/rag.py`~~ — simplified to text-only RAG, fixed code outside `__main__`, fixed schema mismatch
- ~~`pipelines/voice_ai_analytics/voice_ai_analytics_openai.py`~~ — replaced inline dict return type with explicit `return_dtype`, `llm_generate` → `prompt`
- ~~`pipelines/context_engineering/arxiv_search/daily_workflow.py`~~ — removed `daft.set_config`, pass `api_key` to `write_turbopuffer` directly, fixed param names
- ~~`pipelines/voice_ai_analytics/voice_ai_tutorial.py`~~ — credential/infra issue (needs OpenRouter key + whisper model download), not a code bug

## Credential / Infra Issues

- AWS `InvalidAccessKeyId` — common_crawl/*, social_recommendation/build_index (6 scripts)
- AWS `InvalidToken` — tpch/*, laion/clip_training, open_images/vision_models, social_recommendation/ingest_images (6 scripts)
- OpenRouter 401 — prompt_chat_completions, prompt_session, prompt_structured_outputs (3 scripts)
- Google API key — prompt_gemini3_code_review (verify updated key works)

## Dataset Schema Mismatches

- ~~`datasets/open_images/basic_images.py`~~ — replaced `image_metadata` with `image_width`/`image_height`, `file` with `decode_image(download())`
- ~~`datasets/open_images/image_processing.py`~~ — replaced `image_resize` with `resize`, `image_metadata` with `image_width`/`image_height`
- ~~`datasets/reddit_irl/`*~~ — removed

## Structural

- `models/faster-whisper/` — decide placement (move under `pipelines/`?)
- `shot_boundary_detection.py` — hardcoded local video paths, needs real test data or skip

