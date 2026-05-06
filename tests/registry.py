"""
Script registry — single source of truth for every runnable example.

Each entry declares:
  - path: relative to repo root
  - env: list of required environment variables (script is skipped when any are missing)
  - timeout: max seconds (default 120)
  - tier: "quickstart" | "example" | "pipeline" | "dataset"
  - skip: optional reason string to unconditionally skip (for WIP / broken scripts)

To add a new example, just append an entry here. Tests and the Makefile
target both read from this list.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Script:
    path: str
    env: list[str] = field(default_factory=list)
    timeout: int = 120
    tier: str = "example"
    skip: str | None = None


# fmt: off
SCRIPTS: list[Script] = [
    # ── quickstart ──────────────────────────────────────────────────
    Script("quickstart/01_hello_world_prompt.py",   env=["OPENAI_API_KEY"], tier="quickstart"),
    Script("quickstart/02_semantic_search.py",      env=["OPENAI_API_KEY", "TURBOPUFFER_API_KEY"], tier="quickstart", timeout=300),
    Script("quickstart/03_data_enrichment.py",      env=["OPENAI_API_KEY"], tier="quickstart"),
    Script("quickstart/04_audio_file.py",           tier="quickstart"),
    Script("quickstart/05_video_file.py",           tier="quickstart"),

    # ── examples/classify ───────────────────────────────────────────
    Script("examples/classify/classify_image.py",  skip="daft 0.7.8 bug: TransformersImageClassifierPipeline missing 'framework' attribute"),
    Script("examples/classify/classify_text.py"),

    # ── examples/commoncrawl ────────────────────────────────────────
    Script("examples/commoncrawl/cc_chunk_embed.py",           timeout=300),
    Script("examples/commoncrawl/cc_show.py"),
    Script("examples/commoncrawl/cc_wet_paragraph_dedupe.py",  timeout=300, skip="daft 0.7.8 bug: FixedSizeList field name mismatch in minhash().chunk()"),


    # ── examples/embed ──────────────────────────────────────────────
    Script("examples/embed/cosine_similarity.py",       env=["OPENAI_API_KEY"]),

    Script("examples/embed/embed_images.py"),
    Script("examples/embed/embed_text.py",              env=["OPENAI_API_KEY"]),
    Script("examples/embed/embed_text_providers.py",    env=["OPENAI_API_KEY"], skip="requires LM Studio running locally"),
    Script("examples/embed/embed_video_frames.py",  skip="requires YouTube download via yt-dlp"),

    # ── examples/files ──────────────────────────────────────────────
    Script("examples/files/daft_audiofile.py"),
    Script("examples/files/daft_audiofile_udf.py"),
    Script("examples/files/daft_file.py"),
    Script("examples/files/daft_file_code.py"),
    Script("examples/files/daft_file_markdown.py"),
    Script("examples/files/daft_file_pdf.py"),
    Script("examples/files/daft_videofile.py"),
    Script("examples/files/daft_videofile_stream.py", skip="daft 0.7.8 bug: video_keyframes stream finalization can fail"),

    # ── examples/io ─────────────────────────────────────────────────
    Script("examples/io/read_pdfs.py"),
    Script("examples/io/read_video_files.py",  timeout=300),

    # ── examples/prompt ─────────────────────────────────────────────
    Script("examples/prompt/prompt.py",                     env=["OPENAI_API_KEY"]),
    Script("examples/prompt/prompt_chat_completions.py",    env=["OPENROUTER_API_KEY"]),
    Script("examples/prompt/prompt_files_images.py",        env=["OPENAI_API_KEY"]),
    Script("examples/prompt/prompt_gemini3_code_review.py", env=["GOOGLE_API_KEY"]),
    Script("examples/prompt/prompt_openai_web_search.py",   env=["OPENAI_API_KEY"]),
    Script("examples/prompt/prompt_pdfs.py",                env=["OPENAI_API_KEY"], timeout=300),
    Script("examples/prompt/prompt_qa.py",                  env=["OPENAI_API_KEY"], timeout=300),
    Script("examples/prompt/prompt_session.py",             env=["OPENROUTER_API_KEY"]),
    Script("examples/prompt/prompt_structured_outputs.py",  env=["OPENROUTER_API_KEY"]),
    Script("examples/prompt/prompt_unity_catalog.py",       env=["DATABRICKS_TOKEN", "OPENAI_API_KEY"], skip="daft.unity_catalog module not available in daft 0.7.8"),

    # ── examples/session ────────────────────────────────────────────
    Script("examples/session/session_basics.py"),
    Script("examples/session/session_catalogs.py"),
    Script("examples/session/session_context_config.py"),
    Script("examples/session/session_extension_h3.py"),
    Script("examples/session/session_namespaces.py"),
    Script("examples/session/session_providers.py",     env=["OPENAI_API_KEY"]),
    Script("examples/session/session_sql.py", skip="daft 0.7.8 bug: native shutdown segfault on CI runners"),

    # ── examples/sql ────────────────────────────────────────────────
    Script("examples/sql/stocks.py"),

    # ── examples/udfs ───────────────────────────────────────────────
    Script("examples/udfs/daft_cls_async_client.py",    env=["OPENAI_API_KEY"]),
    Script("examples/udfs/daft_cls_model.py"),
    Script("examples/udfs/daft_cls_with_types.py"),
    Script("examples/udfs/daft_func.py"),
    Script("examples/udfs/daft_func_async.py",          env=["OPENAI_API_KEY"]),
    Script("examples/udfs/daft_func_batch.py"),
    Script("examples/udfs/daft_func_batch_scalars.py"),

    # ── pipelines ───────────────────────────────────────────────────
    Script("pipelines/ai_search.py",        env=["OPENAI_API_KEY", "TURBOPUFFER_API_KEY"], tier="pipeline", timeout=300),
    Script("pipelines/data_enrichment.py",  env=["OPENAI_API_KEY"], tier="pipeline"),
    Script("pipelines/embed_docs.py",       env=["OPENAI_API_KEY"], tier="pipeline"),
    Script("pipelines/key_moments_extraction.py",   tier="pipeline", timeout=300, skip="requires faster-whisper model download"),
    Script("pipelines/shot_boundary_detection.py",  tier="pipeline", timeout=600, skip="requires large ML model download (aimv2-large-patch14-224-lit)"),

    Script("pipelines/code/cursor.py",          env=["OPENAI_API_KEY", "GITHUB_TOKEN"], tier="pipeline"),
    Script("pipelines/code/prompt_github.py",   env=["OPENAI_API_KEY", "GITHUB_TOKEN"], tier="pipeline"),

    Script("pipelines/context_engineering/arxiv_search/daily_workflow.py",   env=["OPENAI_API_KEY", "TURBOPUFFER_API_KEY"], tier="pipeline"),
    Script("pipelines/context_engineering/arxiv_search/ingest_lambda.py",    env=["S3_BUCKET"], tier="pipeline"),
    Script("pipelines/context_engineering/arxiv_search/search.py",           env=["OPENAI_API_KEY", "TURBOPUFFER_API_KEY"], tier="pipeline"),
    Script("pipelines/context_engineering/llm_judge_elo.py",                 env=["OPENROUTER_API_KEY"], tier="pipeline"),
    Script("pipelines/context_engineering/chunking_strategies.py",          env=["OPENAI_API_KEY"], tier="pipeline"),
    Script("pipelines/context_engineering/few_shot_example_selection.py",   env=["OPENAI_API_KEY"], tier="pipeline"),
    Script("pipelines/context_engineering/lambda_mapreduce.py",             env=["OPENAI_API_KEY"], tier="pipeline"),

    Script("pipelines/image_understanding_eval/eval_image_understanding.py",    env=["OPENAI_API_KEY"], tier="pipeline", timeout=300),
    Script("pipelines/image_understanding_eval/image_understanding_report.py",  env=["OPENAI_API_KEY"], tier="pipeline"),

    Script("pipelines/rag/full_rag.py",     env=["OPENAI_API_KEY"], tier="pipeline"),
    Script("pipelines/rag/rag.py",          env=["OPENAI_API_KEY"], tier="pipeline"),

    Script("pipelines/social_recommendation/build_index.py",        env=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"], tier="pipeline"),
    Script("pipelines/social_recommendation/ingest_comments.py",    env=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"], tier="pipeline"),
    Script("pipelines/social_recommendation/ingest_images.py",      env=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"], tier="pipeline"),
    Script("pipelines/social_recommendation/write_index_to_uc.py",  env=["DATABRICKS_TOKEN"], tier="pipeline", skip="daft.unity_catalog module not available in daft 0.7.8"),

    Script("pipelines/voice_ai_analytics/voice_ai_analytics.py",        env=["OPENROUTER_API_KEY"], tier="pipeline", timeout=300, skip="requires faster-whisper model download"),
    Script("pipelines/voice_ai_analytics/voice_ai_analytics_openai.py",  env=["OPENAI_API_KEY"], tier="pipeline"),
    Script("pipelines/voice_ai_analytics/voice_ai_tutorial.py",          tier="pipeline", skip="requires faster-whisper model download"),

    # ── datasets ────────────────────────────────────────────────────
    Script("datasets/common_crawl/basic_warc.py",           tier="dataset"),
    Script("datasets/common_crawl/basic_wat.py",            tier="dataset"),
    Script("datasets/common_crawl/basic_wet.py",            tier="dataset"),
    Script("datasets/common_crawl/chunk_embed.py",          env=["OPENAI_API_KEY"], tier="dataset"),
    Script("datasets/common_crawl/content_analysis.py",     tier="dataset"),
    Script("datasets/common_crawl/text_deduplication.py",   tier="dataset", skip="daft 0.7.8 bug: FixedSizeList field name mismatch in minhash().chunk()"),

    Script("datasets/laion/basic_metadata.py",      tier="dataset"),
    Script("datasets/laion/clip_training.py",       env=["OPENAI_API_KEY"], tier="dataset"),
    Script("datasets/laion/image_text_pairs.py",    tier="dataset"),

    Script("datasets/open_images/basic_images.py",      tier="dataset"),
    Script("datasets/open_images/image_processing.py",  tier="dataset"),
    Script("datasets/open_images/vision_models.py",     env=["OPENAI_API_KEY"], tier="dataset", timeout=300),

    Script("datasets/tpch/basic_query.py",          tier="dataset"),
    Script("datasets/tpch/performance_test.py",     tier="dataset", timeout=300),
    Script("datasets/tpch/sql_queries.py",          tier="dataset", timeout=300),
]
# fmt: on


# Helper to avoid importing the module when not needed (e.g., Makefile generation)
SCRIPTS_BY_TIER: dict[str, list[Script]] = {}
for _s in SCRIPTS:
    SCRIPTS_BY_TIER.setdefault(_s.tier, []).append(_s)
