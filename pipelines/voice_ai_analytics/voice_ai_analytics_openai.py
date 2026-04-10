# /// script
# description = "Voice analytics with OpenAI Whisper transcription, GPT summarization, and Spanish translation"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft[openai]>=0.7.8", "openai", "python-dotenv", "soundfile", "numpy", "pylance"]
# ///

from openai import AsyncOpenAI

import daft
from daft import DataType


@daft.cls()
class OpenAITranscription:
    def __init__(self):
        self.client = AsyncOpenAI()

    @daft.method(
        return_dtype=DataType.struct(
            {
                "transcript": DataType.string(),
                "segments": DataType.list(
                    DataType.struct(
                        {
                            "seg_text": DataType.string(),
                            "seg_start": DataType.float64(),
                            "seg_end": DataType.float64(),
                        }
                    )
                ),
            }
        ),
    )
    async def transcribe(self, audio_file: daft.File):
        with audio_file.to_tempfile() as tmpfile:
            transcriptions = await self.client.audio.transcriptions.create(
                model="whisper-1",
                file=str(tmpfile.name),
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

        segments = [{"seg_text": t.text, "seg_start": t.start, "seg_end": t.end} for t in transcriptions.segments]
        transcript = " ".join([t.text for t in transcriptions.segments])

        return {"transcript": transcript, "segments": segments}


if __name__ == "__main__":
    # Run this script with `uv run speech/speech_analytics_openai.py`
    from dotenv import load_dotenv

    from daft import col
    from daft.functions import embed_text, file, format, prompt

    SOURCE_URI = "hf://datasets/Eventual-Inc/sample-files/audio/*.mp3"
    DEST_URI = ".data/voice_ai"
    FILE_LIMIT = 10

    # Load environment variables from .env file
    load_dotenv()

    oai_transcriptor = OpenAITranscription()

    df_transcripts = (
        # Read the audio files
        daft.from_glob_path(SOURCE_URI)
        .where(col("path").endswith(".mp3"))
        .limit(FILE_LIMIT)
        # Transcribe the audio files
        .with_column("transcription", oai_transcriptor.transcribe(file(col("path"))))
        .with_column("transcript", col("transcription").get("transcript"))
        # Summarize the transcript
        .with_column(
            "transcript_summary",
            prompt(
                format("Summarize the following podcast transcript: {}", col("transcript")),
                model="gpt-5-nano",
                provider="openai",
            ),
        )
        # Translate Segment Subtitles to Spanish for Localization
        .with_column(
            "transcript_spanish",
            prompt(
                format("Translate the following text to Spanish: {}", col("transcript")),
                model="gpt-5-nano",
                provider="openai",
            ),
        )
        # Embed the transcript and summary
        .with_column(
            "transcript_embeddings",
            embed_text(col("transcript"), model="text-embedding-ada-002", provider="openai"),
        )
        .with_column(
            "transcript_summary_embeddings",
            embed_text(
                daft.col("transcript_summary"),
                model="text-embedding-ada-002",
                provider="openai",
            ),
        )
    )

    # Save and Show
    df_transcripts.write_lance(DEST_URI, mode="overwrite")

    # Query the data and display the results
    daft.read_lance(DEST_URI).show(format="fancy", max_width=40)
