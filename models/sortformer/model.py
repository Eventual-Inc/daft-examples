"""NVIDIA Sortformer speaker diarization as a Daft class UDF.

Backend: PyTorch via NeMo. Sortformer is end-to-end — it emits speaker turns
directly from audio, and its per-frame activity matrix subsumes VAD, so it
replaces both Silero VAD and pyannote in one model. Limits: 4 speakers max.

This module never imports ``modal``; see ``modal_app.py`` for deployment.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

# Keep `models.*` importable when run as a loose script under `modal run`.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import daft
from daft import DataType, col
from models.common.audio import SAMPLE_RATE, read_waveform_16k_mono
from models.common.speech import SpeakerSegmentStruct

SpeakerSegmentsResult = DataType.list(SpeakerSegmentStruct)


def _parse_turns(raw: Any) -> list[dict]:
    """NeMo diarize() yields 'start end speaker' strings (or nested lists)."""
    turns: list[dict] = []
    items = raw[0] if raw and isinstance(raw[0], list) else raw
    for entry in items or []:
        if isinstance(entry, str):
            parts = entry.split()
            if len(parts) < 3:
                continue
            start, end, speaker = parts[0], parts[1], parts[2]
        else:
            start, end, speaker = entry[0], entry[1], entry[2]
        turns.append({"start": float(start), "end": float(end), "speaker": str(speaker)})
    return turns


@daft.cls(gpus=1.0, max_concurrency=1)
class SortformerDiarizer:
    def __init__(self, *, model: str = "nvidia/diar_streaming_sortformer_4spk-v2"):
        from nemo.collections.asr.models import SortformerEncLabelModel

        self.model_name = model
        self.diarizer = SortformerEncLabelModel.from_pretrained(model).eval()

    @daft.method(return_dtype=SpeakerSegmentsResult)
    def diarize(self, audio: daft.AudioFile, transcript: str):
        if not transcript:
            return []
        import soundfile as sf

        # Per-file fault isolation: a single oversized/odd file returns no speaker
        # turns instead of crashing the whole Daft batch (heterogeneous folders).
        try:
            waveform = read_waveform_16k_mono(audio)
            with tempfile.TemporaryDirectory() as tmp:
                wav_path = str(Path(tmp) / "clip.wav")
                sf.write(wav_path, waveform, SAMPLE_RATE)
                raw = self.diarizer.diarize(audio=[wav_path])
            return _parse_turns(raw)
        except Exception as exc:  # noqa: BLE001 — isolate per-file failures
            print(f"sortformer: diarization failed for one file ({type(exc).__name__}: {exc})")
            return []


def attach_speakers(df: daft.DataFrame, processor: SortformerDiarizer) -> daft.DataFrame:
    """Add a `speaker_segments` column. Expects `audio` and `transcript` columns."""
    return df.with_column("speaker_segments", processor.diarize(col("audio"), col("transcript")))
