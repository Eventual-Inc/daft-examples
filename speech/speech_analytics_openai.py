# /// script
# description = "Summarize podcasts"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft[lance]", "openai", "python-dotenv", "soundfile", "numpy"]
# ///
import io
import pathlib

import daft
from daft import DataType
from openai import OpenAI
import soundfile as sf


@daft.func(
    return_dtype=DataType.struct(
        {
            "transcript": DataType.string(),
            "segments": DataType.list(
                DataType.struct(
                    {
                        "seg_text": DataType.string(),
                        "seg_start": DataType.float32(),
                        "seg_end": DataType.float32(),
                    }
                )
            ),
        }
    )
)
def transcribe(audio_file: daft.File):
    """Transcribes an audio file using openai whisper."""

    client = OpenAI()

    with file.open() as f:
        file_obj = io.BytesIO(f.read())
        file_obj.name = "audio.mp3"

    transcriptions = client.audio.transcriptions.create(
        model="whisper-1",
        file=file_obj,
        response_format="verbose_json",
        timestamp_granularities=["segment"],
    )

    segments = [
        {"seg_text": t.text, "seg_start": t.start, "seg_end": t.end}
        for t in transcriptions.segments
    ]
    transcript = " ".join([t.text for t in transcriptions.segments])

    return {"transcript": transcript, "segments": segments}


if __name__ == "__main__":
    from daft import col
    from daft.functions import embed_text, llm_generate, format, unnest, file
    from dotenv import load_dotenv

    SOURCE_URI = "hf://datasets/Eventual-Inc/sample-files/audio/*.mp3"
    DEST_URI = ".data/voice_ai"
    FILE_LIMIT = 10

    # Load environment variables from .env file
    load_dotenv()

    df_transcripts = (
        daft.from_glob_path(SOURCE_URI)
        .with_column("transcript_segments", transcribe(file(col("path"))))
        .select(col("path"), unnest(col("transcript_segments")))
        # Summarize the transcript
        .with_column(
            "transcript_summary",
            llm_generate(
                format(
                    "Summarize the following podcast transcript: {}", col("transcript")
                ),
                model="gpt-5-nano",
                provider="openai",
            ),
        )
        # Translate Segment Subtitles to Spanish for Localization
        .with_column(
            "transcript_spanish",
            llm_generate(
                format(
                    "Translate the following text to Spanish: {}", col("transcript")
                ),
                model="gpt-5-nano",
                provider="openai",
            ),
        )
        # Embed the transcript and summary
        .with_column(
            "transcript_embeddings",
            embed_text(
                col("transcript"), model="text-embedding-ada-002", provider="openai"
            ),
        )
        .with_column(
            "transcript_summary_embeddings",
            embed_text(
                col("transcript_summary"),
                model="text-embedding-ada-002",
                provider="openai",
            ),
        )
    )

    # Save and Show
    df_transcripts.write_lance(DEST_URI)

    # Query the data and display the results
    daft.read_lance(DEST_URI).show()
