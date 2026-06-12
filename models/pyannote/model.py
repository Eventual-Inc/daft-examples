"""pyannote speaker diarization as a Daft class UDF.

Backend: PyTorch. The Whisper-lane diarizer — no 4-speaker cap, but weights are
gated (set HF_TOKEN and accept the model terms). Emits the shared SpeakerSegment
contract so it's swappable with Sortformer.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import daft
from daft import col
from models.common.audio import SAMPLE_RATE, read_waveform_16k_mono
from models.common.speech import SpeakerSegmentStruct

DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"
SpeakerSegmentsResult = daft.DataType.list(SpeakerSegmentStruct)


@daft.cls(gpus=1.0, max_concurrency=1)
class PyannoteDiarizer:
    def __init__(self, *, model: str = DEFAULT_MODEL):
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise OSError("HF_TOKEN required for gated pyannote weights")
        import torch
        from pyannote.audio import Pipeline

        self.model_name = model
        self.pipeline = Pipeline.from_pretrained(model, token=token)
        self.pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    @daft.method(return_dtype=SpeakerSegmentsResult)
    def diarize(self, audio: daft.AudioFile, transcript: str):
        if not transcript:
            return []
        import torch

        waveform = read_waveform_16k_mono(audio)
        result = self.pipeline({"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": SAMPLE_RATE})
        annotation = result.speaker_diarization if hasattr(result, "speaker_diarization") else result
        return [
            {"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)}
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]


def attach_speakers(df: daft.DataFrame, processor: PyannoteDiarizer) -> daft.DataFrame:
    return df.with_column("speaker_segments", processor.diarize(col("audio"), col("transcript")))
