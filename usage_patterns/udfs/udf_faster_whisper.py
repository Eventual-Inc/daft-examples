# /// script
# description = "Run Faster Whisper model on Daft UDFs for audio transcription"
# requires-python = ">=3.10, <3.13"
# dependencies = ["daft>=0.6.8", "faster-whisper"]
# ///
"""
Usage Pattern: Running Models on Daft UDFs
==========================================

This example demonstrates how to run ML models like Faster Whisper using Daft's
class-based UDF (User-Defined Function) system. This pattern is useful for:

1. **Model Loading Once**: Using @daft.cls() allows you to initialize the model once
   and reuse it across all rows, avoiding redundant loading.

2. **Complex Return Types**: Using structured DataTypes to return rich, typed results
   from model inference.

3. **Stateful Processing**: Maintaining model state (like batch processing pipelines)
   across invocations.

Key Daft UDF Concepts Demonstrated:
- @daft.cls() decorator for class-based UDFs
- @daft.method() decorator for methods that operate on columns
- Custom DataType schemas for structured outputs
- Working with daft.File for handling file inputs
"""

from dataclasses import asdict

import daft
from daft import DataType

# ==============================================================================
# Schema Definitions for Faster Whisper Output
# ==============================================================================

# Word-level transcription result
WordStruct = DataType.struct(
    {
        "start": DataType.float64(),
        "end": DataType.float64(),
        "word": DataType.string(),
        "probability": DataType.float64(),
    }
)

# Segment-level transcription result
SegmentStruct = DataType.struct(
    {
        "id": DataType.int64(),
        "seek": DataType.int64(),
        "start": DataType.float64(),
        "end": DataType.float64(),
        "text": DataType.string(),
        "tokens": DataType.list(DataType.int64()),
        "avg_logprob": DataType.float64(),
        "compression_ratio": DataType.float64(),
        "no_speech_prob": DataType.float64(),
        "words": DataType.list(WordStruct),
        "temperature": DataType.float64(),
    }
)

# Transcription options structure
TranscriptionOptionsStruct = DataType.struct(
    {
        "beam_size": DataType.int64(),
        "best_of": DataType.int64(),
        "patience": DataType.float64(),
        "length_penalty": DataType.float64(),
        "repetition_penalty": DataType.float64(),
        "no_repeat_ngram_size": DataType.int64(),
        "log_prob_threshold": DataType.float64(),
        "no_speech_threshold": DataType.float64(),
        "compression_ratio_threshold": DataType.float64(),
        "condition_on_previous_text": DataType.bool(),
        "prompt_reset_on_temperature": DataType.float64(),
        "temperatures": DataType.list(DataType.float64()),
        "initial_prompt": DataType.python(),
        "prefix": DataType.string(),
        "suppress_blank": DataType.bool(),
        "suppress_tokens": DataType.list(DataType.int64()),
        "without_timestamps": DataType.bool(),
        "max_initial_timestamp": DataType.float64(),
        "word_timestamps": DataType.bool(),
        "prepend_punctuations": DataType.string(),
        "append_punctuations": DataType.string(),
        "multilingual": DataType.bool(),
        "max_new_tokens": DataType.float64(),
        "clip_timestamps": DataType.python(),
        "hallucination_silence_threshold": DataType.float64(),
        "hotwords": DataType.string(),
    }
)

# Voice Activity Detection (VAD) options
VadOptionsStruct = DataType.struct(
    {
        "threshold": DataType.float64(),
        "neg_threshold": DataType.float64(),
        "min_speech_duration_ms": DataType.int64(),
        "max_speech_duration_s": DataType.float64(),
        "min_silence_duration_ms": DataType.int64(),
        "speech_pad_ms": DataType.int64(),
    }
)

# Language probability structure
LanguageProbStruct = DataType.struct(
    {
        "language": DataType.string(),
        "probability": DataType.float64(),
    }
)

# Transcription info structure
InfoStruct = DataType.struct(
    {
        "language": DataType.string(),
        "language_probability": DataType.float64(),
        "duration": DataType.float64(),
        "duration_after_vad": DataType.float64(),
        "all_language_probs": DataType.list(LanguageProbStruct),
        "transcription_options": TranscriptionOptionsStruct,
        "vad_options": VadOptionsStruct,
    }
)

# Complete transcription result
TranscriptionResult = DataType.struct(
    {
        "transcript": DataType.string(),
        "segments": DataType.list(SegmentStruct),
        "info": InfoStruct,
    }
)


# ==============================================================================
# Class-Based UDF for Faster Whisper
# ==============================================================================


@daft.cls()
class FasterWhisperTranscriber:
    """
    A class-based UDF for transcribing audio files using Faster Whisper.

    This demonstrates the pattern of:
    1. Loading a model once in __init__
    2. Using the model for inference in decorated methods
    3. Returning structured results using custom DataTypes

    Parameters:
        model: Whisper model name (e.g., "distil-large-v3", "tiny", "base", "small", "medium", "large-v3")
        compute_type: Computation precision ("float32", "float16", "int8")
        device: Device to run on ("auto", "cpu", "cuda")
    """

    def __init__(
        self,
        model: str = "distil-large-v3",
        compute_type: str = "float32",
        device: str = "auto",
    ):
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        self.whisper_model = WhisperModel(model, compute_type=compute_type, device=device)
        self.pipe = BatchedInferencePipeline(self.whisper_model)

    @daft.method(return_dtype=TranscriptionResult)
    def transcribe(self, audio_file: daft.File, batch_size: int = 16):
        """
        Transcribe an audio file with Voice Activity Detection (VAD).

        This method demonstrates:
        - Working with daft.File inputs using to_tempfile()
        - Returning structured results matching the defined schema
        - Using model inference within a UDF

        Args:
            audio_file: A daft.File object containing the audio data
            batch_size: Batch size for inference

        Returns:
            A dict matching TranscriptionResult schema with transcript,
            segments, and info
        """
        with audio_file.to_tempfile() as tmp:
            segments_iter, info = self.pipe.transcribe(
                str(tmp.name),
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                word_timestamps=True,
                batch_size=batch_size,
            )
            segments = [asdict(seg) for seg in segments_iter]
            text = " ".join([seg["text"] for seg in segments])

            return {"transcript": text, "segments": segments, "info": asdict(info)}


if __name__ == "__main__":
    from daft import col
    from daft.functions import file, unnest

    # ==============================================================================
    # Configuration
    # ==============================================================================

    # Audio source - using sample audio files from HuggingFace
    SOURCE_URI = "hf://datasets/Eventual-Inc/sample-files/audio/*.mp3"

    # Batch size for Whisper inference
    BATCH_SIZE = 16

    # ==============================================================================
    # Create the Transcription Pipeline
    # ==============================================================================

    # Instantiate the Faster Whisper transcriber UDF
    # The model is loaded once and reused for all audio files
    transcriber = FasterWhisperTranscriber(
        model="distil-large-v3",  # Use distilled model for faster inference
        compute_type="float32",  # Use float32 for CPU compatibility
        device="auto",  # Auto-detect device (CPU/GPU)
    )

    # Build the transcription pipeline
    df_transcript = (
        # Step 1: Discover audio files from the source URI
        daft.from_glob_path(SOURCE_URI)
        # Step 2: Wrap the path as a daft.File for handling by the UDF
        .with_column("audio_file", file(col("path")))
        # Step 3: Transcribe using the class-based UDF
        .with_column("result", transcriber.transcribe(col("audio_file")))
        # Step 4: Unpack the structured result into separate columns
        .select("path", unnest(col("result")))
    )

    # Collect the base transcript results for reuse
    df_transcript_result = df_transcript.collect()

    # ==============================================================================
    # Execute and Display Detailed Results
    # ==============================================================================

    # Process segments for detailed view
    df_segments = (
        df_transcript_result.to_daft()
        # Explode segments to get one row per segment
        .explode("segments")
        # Unnest segment details for analysis
        .select("path", "info", "transcript", unnest(col("segments")))
    )

    # Collect and display the segment-level results
    result = df_segments.collect()
    result.show(format="fancy", max_width=40)

    # ==============================================================================
    # Example Output Analysis
    # ==============================================================================

    # Show summary statistics
    print("\n" + "=" * 60)
    print("Transcription Summary")
    print("=" * 60)

    # Reuse the collected results for summary (no redundant model inference)
    summary_df = (
        df_transcript_result.to_daft()
        .select("path", "transcript", col("info").struct.get("language").alias("language"))
    )

    summary_df.show()
