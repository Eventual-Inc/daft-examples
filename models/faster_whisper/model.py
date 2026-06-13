# /// script
# description = "Transcribe + VAD with Faster Whisper"
# requires-python = ">=3.12, <3.13"
# dependencies = ["daft>=0.7.8", "faster-whisper"]
# ///
"""Faster Whisper transcription as a Daft class UDF.

Backend: CTranslate2 (faster-whisper) — runs locally on CPU or GPU; Whisper-style
audio models are not supported by vLLM.

The result schema lives in ``schema.py`` so pipelines can import it without
pulling in faster-whisper.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

# Anchor the repo root so `models.*` imports resolve when this file is loaded
# as a loose script (e.g. `uv run` / `modal run`) instead of an installed package.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import daft
from daft import col
from daft.functions import file, unnest
from models.faster_whisper.schema import TranscriptionResult


@daft.cls()
class FasterWhisperTranscriber:
    def __init__(self, model="distil-large-v3", compute_type="float32", device="auto", batch_size=16):
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        self.model = WhisperModel(model, compute_type=compute_type, device=device)
        self.pipe = BatchedInferencePipeline(self.model)
        self.batch_size = batch_size

    @daft.method(return_dtype=TranscriptionResult)
    def transcribe(self, audio_file: daft.File):
        """Transcribe Audio Files with Voice Activity Detection (VAD) using Faster Whisper"""
        with audio_file.to_tempfile() as tmp:
            segments_iter, info = self.pipe.transcribe(
                str(tmp.name),
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                word_timestamps=True,
                batch_size=self.batch_size,
            )
            segments = [asdict(seg) for seg in segments_iter]
            text = " ".join([seg["text"] for seg in segments])

            return {"transcript": text, "segments": segments, "info": asdict(info)}


if __name__ == "__main__":
    # Define Parameters & Constants
    SOURCE_URI = "hf://datasets/Eventual-Inc/sample-files/audio/*.mp3"

    # Instantiate Transcription UDF
    fwt = FasterWhisperTranscriber()

    # Transcribe the audio files
    df_transcript = (
        # Discover the audio files
        daft.from_glob_path(SOURCE_URI)
        # Wrap the path as a daft.File
        .with_column("audio_file", file(col("path")))
        # Transcribe the audio file with Voice Activity Detection (VAD) using Faster Whisper
        .with_column("result", fwt.transcribe(col("audio_file")))
        # Unpack Results
        .select("path", unnest(col("result")))
        .explode("segments")
        .select("path", "info", "transcript", unnest(col("segments")))
    ).collect()

    df_transcript.show(format="fancy", max_width=40)
