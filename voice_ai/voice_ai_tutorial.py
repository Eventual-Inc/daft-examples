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

    from daft import col
    from daft.functions import format, file, unnest
    from daft.functions.ai import prompt, embed_text
    from daft.ai.openai.provider import OpenAIProvider

    # Define Parameters
    SOURCE_URI = "hf://datasets/Eventual-Inc/sample-files/audio/*.mp3"
    DEST_URI = ".data/voice_ai_analytics"
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

    # Instantiate Transcription UDF
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
    ).collect()

    print(
        "\n\nRunning Transcription with Voice Activity Detection (VAD) using Faster Whisper..."
    )

    # Show the transcript.
    df_transcript.select(
        "path",
        "info",
        "transcript",
        "segments",
    ).show(format="fancy", max_width=40)

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

    print("\n\nGenerating Summaries...")

    # Show the summaries and the transcript.
    df_summaries.select(
        "path",
        "transcript",
        "summary",
        "summary_chinese",
    ).show(format="fancy", max_width=40)

    # ==============================================================================
    # Subtitles Generation
    # ==============================================================================

    # Explode the segments, embed, and translate to simplified Chinese for subtitles.
    df_segments = (
        df_transcript.explode("segments")
        .select(
            "path",
            unnest(col("segments")),
        )
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
    )

    print("\n\nGenerating Chinese Subtitles...")

    # Show the segments and the transcript.
    df_segments.select(
        "path",
        col("text"),
        "segment_text_chinese",
    ).show(format="fancy", max_width=40)

    # ==============================================================================
    # Embedding Generation
    # ==============================================================================

    # Embed the segments and translate to simplified Chinese for subtitles.
    df_segments = df_segments.with_column(
        "segment_embeddings",
        embed_text(
            col("text"),
            provider="sentence_transformers",
            model=EMBEDDING_MODEL_ID,
        ),
    )

    print("\n\nGenerating Embeddings for Segments...")

    # Show the segments and the transcript.
    df_segments.select(
        "path",
        "text",
        "segment_embeddings",
    ).show(format="fancy", max_width=40)
