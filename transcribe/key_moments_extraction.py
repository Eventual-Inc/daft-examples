# /// script
# description = "Summarize podcasts"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft>=0.6.7", "faster-whisper", "soundfile", "pydantic", "python-dotenv", "openai", "sentence-transformers"]
# ///
import daft
from daft import DataType
from pydantic import BaseModel, Field

# Define Constants
SAMPLE_RATE = 16000
DTYPE = "float32"
BATCH_SIZE = 16


# Define Pydantic Models for Extracting Key Moments using Structured Outputs
class KeyMoment(BaseModel):
    moment_reasoning: str = Field(
        ...,
        description="Use this field as a notepad to reason about moments worth clipping for short form content.",
    )
    moment_summary: str = Field(
        ..., description="A concise one-sentence summary of the key moment."
    )
    moment_start_sec: float = Field(
        ...,
        description="The start time of the first segment in the group of segments in seconds. Reference the provided segment start timestamps.",
        strict=True,
    )
    moment_end_sec: float = Field(
        ...,
        description="The end time of the last segment in the group of segments in seconds. Reference the provided segment end timestamps.",
        strict=True,
    )
    moment_text: str = Field(
        ...,
        description="The text of all segments combined. Used for validation. Reference the provided segment text.",
    )


@daft.func()
def print_segments_w_timestamps(
    segments: list[dict], 
    print_segments: bool = True,
) -> str:
    """Print segments with timestamps"""

    segment_string = "\n".join(
        [f"{seg['start']} - {seg['end']}: {seg['text']}" for seg in segments]
    )

    if print_segments:
        print(segment_string)

    return f"<segments>\n{segment_string}\n</segments>"


@daft.func(
    return_dtype=DataType.struct({
        "start_sec_snapped": DataType.float64(), 
        "end_sec_snapped": DataType.float64()
    }),
)
def snap_to_segment_bounds(
    segments: list[dict], 
    start_sec: float, 
    end_sec: float, 
    pad_ms: int = 200,
):
    """Snap to segment bounds"""

    pad = pad_ms / 1000.0
    left_candidates = [seg["start"] for seg in segments if seg["start"] <= start_sec]
    right_candidates = [seg["end"] for seg in segments if seg["end"] >= end_sec]
    left = max(left_candidates) if left_candidates else start_sec
    right = min(right_candidates) if right_candidates else end_sec
    if right < end_sec:
        right = end_sec
    if left > start_sec:
        left = start_sec
    return {"start_sec_snapped": max(0.0, left - pad), "end_sec_snapped": right + pad}


@daft.func()
def clip_audio(
    path: str, 
    audio_file: daft.File, 
    dest: str, 
    start_time: float, 
    end_time: float,
) -> str:
    """Clip audio"""

    import soundfile as sf
    from pathlib import Path

    # Convert seconds -> frame indices using the file's native samplerate
    with audio_file.open() as f:
        with sf.SoundFile(f) as snd:
            sample_rate = snd.samplerate
            total_frames = len(snd)

            # Clamp to valid range and convert to integer frame indices
            start_frame = int(round(max(0.0, start_time) * sample_rate))
            end_time_capped = max(start_time, end_time)
            end_frame = int(
                round(min(end_time_capped, total_frames / sample_rate) * sample_rate)
            )

            if end_frame < start_frame:
                end_frame = start_frame

            # Read the desired window
            snd.seek(start_frame)
            num_frames = end_frame - start_frame
            audio = snd.read(frames=num_frames, dtype=DTYPE, always_2d=True)

    # Extract just the filename (without path) from path
    original_filename = Path(path).stem

    # Build the destination directory and ensure it exists
    dest_dir = Path(dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Create the full destination path with the adjusted filename
    rounded_start = f"{start_time:.2f}"
    rounded_end = f"{end_time:.2f}"
    clip_filename = f"{original_filename}_{rounded_start}_{rounded_end}.mp3"
    full_dest = dest_dir / clip_filename

    # Write the audio to the destination path
    sf.write(
        str(full_dest),
        audio,
        samplerate=sample_rate,
        format="MP3",
        subtype="MPEG_LAYER_III",
    )
    return str(full_dest)


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    from daft import col
    from daft.functions import format, file, unnest
    from daft.functions.ai import prompt, embed_text
    from daft.ai.openai.provider import OpenAIProvider

    # Define Parameters
    SOURCE_URI = "hf://datasets/Eventual-Inc/sample-files/audio/*.mp3"
    DEST_URI = ".data/audio_clips/"
    LLM_MODEL_ID = "openai/gpt-oss-120b"
    EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    CONTEXT = "Daft: Unified Engine for Data Analytics, Engineering & ML/AI (github.com/Eventual-Inc/Daft) YouTube channel video. Transcriptions can have errors like 'DAF' referring to 'Daft'."
    FILE_LIMIT = 5
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


    # ==============================================================================
    # Transcription
    # ==============================================================================

    # Transcribe the audio files
    if not os.path.exists(os.path.join(DEST_URI, "transcripts.parquet")):
        from transcribe_faster_whisper import FasterWhisperTranscriber
        
        # Instantiate Transcription UDF
        fwt = FasterWhisperTranscriber()
        
        # Transcribe the audio files
        df_transcript = (
            # Discover the audio files
            daft.from_glob_path(SOURCE_URI)
            .limit(FILE_LIMIT)
            # Wrap the path as a daft.File
            .with_column("audio_file", file(col("path")))
            # Transcribe the audio file with Voice Activity Detection (VAD) using Faster Whisper
            .with_column("result", fwt.transcribe(col("audio_file")))
            # Unpack Results
            .select("path", "audio_file", unnest(col("result")))
        )
    else:
        df_transcript = daft.read_parquet(os.path.join(DEST_URI, "transcripts.parquet"))

    # ==============================================================================
    # Key Moments Extraction
    # ==============================================================================

    # Extract key moments from the transcript and write the clips to disk.
    df_key_moments = (
        df_transcript.with_column(
            "key_moments",
            prompt(
                messages=format(
                    "Extract 1 key moment from the following transcript + timestamps and identify a group of segments of about 30 seconds total duration to clip for short-form content: \n {}",
                    print_segments_w_timestamps(col("segments")),
                ),
                system_message=f"You are a world-class short-form content creator and Developer Advocate for {CONTEXT}.",
                return_format=KeyMoment,
                model=LLM_MODEL_ID,
            ),
        )
        # Unpack the key moments
        .select("path", "audio_file", "segments", unnest(col("key_moments")))
    ).collect()

    # ==============================================================================
    # Clip Audio
    # ==============================================================================

    df_clips = (
        df_key_moments
        # Snap boundaries to nearby segment edges with padding
        .with_column(
            "snapped",
            snap_to_segment_bounds(
                col("segments"),
                col("moment_start_sec"),
                col("moment_end_sec"),
                pad_ms=200,
            ),
        )
        # Save Key Moment Clips to Disk
        .with_column(
            "short_form_clip_path",
            clip_audio(
                col("path"),
                col("audio_file"),
                dest=DEST_URI,
                start_time=col("snapped")["start_sec_snapped"],
                end_time=col("snapped")["end_sec_snapped"],
            ),
        )
    )