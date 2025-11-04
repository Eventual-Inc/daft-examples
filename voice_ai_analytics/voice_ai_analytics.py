# /// script
# description = "Summarize podcasts"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft>=0.6.8", "faster-whisper", "soundfile", "sentence-transformers", "python-dotenv", "openai"]
# ///
from dataclasses import asdict

import daft
from daft import DataType
from faster_whisper import WhisperModel, BatchedInferencePipeline

from transcription_schema import TranscriptionResult, InfoStruct

# Define Constants
SAMPLE_RATE = 16000
DTYPE = "float32"
BATCH_SIZE = 16


@daft.cls()
class FasterWhisperTranscriber:
    def __init__(self, model="distil-large-v3", compute_type="float32", device="auto"):
        self.model = WhisperModel(model, compute_type=compute_type, device=device)
        self.pipe = BatchedInferencePipeline(self.model)

    @daft.method(return_dtype=TranscriptionResult)
    def transcribe(self, audio_file: daft.File):
        """Transcribe Audio Files with Voice Activity Detection (VAD) using Faster Whisper"""
        with audio_file.to_tempfile() as tmp:
            segments_iter, info = self.pipe.transcribe(
                str(tmp.name),
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
                word_timestamps=True,
                without_timestamps=False,
                temperature=0,
                batch_size=BATCH_SIZE,
            )
            segments = [asdict(seg) for seg in segments_iter]
            text = " ".join([seg["text"] for seg in segments])

            return {"transcript": text, "segments": segments, "info": asdict(info)}


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    from daft import col, Window
    from daft.functions import format, file, unnest, rank
    from daft.functions.ai import prompt, embed_text
    from daft.ai.openai.provider import OpenAIProvider

    # Define Parameters
    SOURCE_URI = "hf://datasets/Eventual-Inc/sample-files/audio/*.mp3"
    DEST_URI = ".data/transcribe/faster-whisper/"
    LLM_MODEL_ID = "openai/gpt-oss-120b"
    EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    CONTEXT = "Daft: Unified Engine for Data Analytics, Engineering & ML/AI (github.com/Eventual-Inc/Daft) YouTube channel video. Transcriptions can have errors like 'DAF' referring to 'Daft'."
    PRINT_SEGMENTS = True

    # Load environment variables
    load_dotenv()

    # Create an OpenAI provider, attach, and set as the default
    openrouter_provider = OpenAIProvider(
        name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )
    daft.attach_provider(openrouter_provider)
    daft.set_provider("OpenRouter")

    transcripts_path = os.path.join(DEST_URI, "transcripts.parquet")
    segments_path = os.path.join(DEST_URI, "segments.parquet")

    os.makedirs(DEST_URI, exist_ok=True)

    transcripts_exist = os.path.exists(transcripts_path)
    segments_exist = os.path.exists(segments_path)

    if transcripts_exist and segments_exist:
        print("Loading cached transcripts and segments from parquet.")
        df_summaries = daft.read_parquet(transcripts_path)
        df_segments = daft.read_parquet(segments_path)
    else:
        if transcripts_exist or segments_exist:
            print("Found partial cache. Re-running transcription pipeline to refresh outputs.")
        else:
            print("No cached transcripts detected. Running transcription pipeline.")

        # Instantiate Transcription UDF only when needed
        fwt = FasterWhisperTranscriber()

        # ==============================================================================
        # Transcription
        # ==============================================================================

        # Transcribe the audio files
        df_transcript = (
            # Discover the audio files
            daft.from_glob_path(SOURCE_URI)
            # Wrap the path as a daft.File
            .with_column("audio_file", file(col("path")))
            # Transcribe the audio file with Voice Activity Detection (VAD) using Faster Whisper
            .with_column("result", fwt.transcribe(col("audio_file")))
            # Unpack Results
            .select("path", "audio_file", unnest(col("result")))
        )

        # ==============================================================================
        # Summarization
        # ==============================================================================

        # Summarize the transcripts and translate to Chinese.
        df_summaries = (
            df_transcript
            # Summarize the transcripts
            .with_column(
                "summary",
                prompt(
                    format(
                        "Summarize the following transcript from a YouTube video belonging to {}: \n {}",
                        daft.lit(CONTEXT),
                        col("transcript"),
                    ),
                    model=LLM_MODEL_ID,
                ),
            ).with_column(
                "summary_chinese",
                prompt(
                    format(
                        "Translate the following text to Simplified Chinese: <text>{}</text>",
                        col("summary"),
                    ),
                    system_message="You will be provided with a piece of text. Your task is to translate the text to Simplified Chinese exactly as it is written. Return the translated text only, no other text or formatting.",
                    model=LLM_MODEL_ID,
                ),
            )
        )

        # ==============================================================================
        # Subtitles Generation
        # ==============================================================================

        # Explode the segments, embed, and translate to simplified Chinese for subtitles.
        df_segments = (
            df_transcript.select("path", "segments")
            .explode("segments")
            .select("path", unnest(col("segments")))
            .with_column(
                "segment_text_chinese",
                prompt(
                    format(
                        "Translate the following text to Simplified Chinese: <text>{}</text>",
                        col("text"),
                    ),
                    system_message="You will be provided with a transcript segment. Your task is to translate the text to Simplified Chinese exactly as it is written. Return the translated text only, no other text or formatting.",
                    model=LLM_MODEL_ID,
                ),
            )
            .with_column(
                "segment_embeddings",
                embed_text(
                    col("text"),
                    provider="transformers",
                    model=EMBEDDING_MODEL_ID,
                ),
            )
        )

        # ==============================================================================
        # Store Transcripts and Segments
        # ==============================================================================

        # Store Transcripts with Summaries
        df_summaries.select(
            "path", "info", "transcript", "summary", "summary_chinese"
        ).write_parquet(transcripts_path, write_mode="overwrite")
        # Store Segments
        df_segments.select(
            "path",
            "id",
            "start",
            "end",
            "text",
            "tokens",
            "avg_logprob",
            "compression_ratio",
            "no_speech_prob",
            "words",
            "temperature",
            "segment_text_chinese",
            "segment_embeddings",
        ).write_parquet(segments_path, write_mode="overwrite")

    # ==============================================================================
    # RAG QA Section
    # ==============================================================================

    # Define questions to answer
    QUESTIONS = [
        "What is Daft?",
        "What are the main features of Daft?",
        "How does Daft handle data processing?",
    ]
    TOP_K = 5  # Number of most relevant segments to retrieve per question

    # Create a dataframe with questions
    df_questions = daft.from_pydict({"question": QUESTIONS})

    # Embed the questions
    df_questions_embedded = df_questions.with_column(
        "question_embedding",
        embed_text(
            col("question"),
            provider="transformers",
            model=EMBEDDING_MODEL_ID,
        ),
    )

    # Cross join questions with segments to calculate similarity
    # Use the in-memory df_segments to avoid column name issues
    df_cross = df_questions_embedded.join(
        df_segments.select(
            "path",
            "id",
            "start",
            "end",
            "text",
            "segment_text_chinese",
            "segment_embeddings",
        ),
        how="cross",
    )

    # Calculate cosine distance between question and segment embeddings
    df_with_distance = df_cross.with_column(
        "similarity",
        col("question_embedding").cosine_distance(col("segment_embeddings")),
    )

    # Get top-k most similar segments for each question
    # Use window functions to rank and select top K per question
    window = Window().partition_by("question").order_by(col("similarity").asc())

    df_ranked = df_with_distance.with_column("rank", rank().over(window)).where(
        col("rank") <= TOP_K
    )

    # Aggregate retrieved segments into context for each question
    df_context = df_ranked.groupby("question").agg(
        col("text").alias("retrieved_texts"),
    )

    # Join the texts together into a context string
    df_context_str = df_context.with_column(
        "context",
        col("retrieved_texts").str.join(separator="\n\n"),
    ).select("question", "context")

    # Answer questions using retrieved context
    df_qa_results = df_context_str.with_column(
        "answer",
        prompt(
            format(
                "Context from video transcripts:\n{}\n\nQuestion: {}\n\nAnswer the question based only on the provided context. If the context doesn't contain enough information to answer the question, say so.",
                col("context"),
                col("question"),
            ),
            system_message=f"You are a helpful assistant that answers questions based on video transcript context. {CONTEXT}",
            model=LLM_MODEL_ID,
        ),
    )

    # Display results
    print("\n" + "=" * 80)
    print("RAG QA Results")
    print("=" * 80)
    df_qa_results.select("question", "answer").show()

    # Store QA results
    df_qa_results.select("question", "context", "answer").write_parquet(
        f"{DEST_URI}/qa_results.parquet", write_mode="overwrite"
    )
