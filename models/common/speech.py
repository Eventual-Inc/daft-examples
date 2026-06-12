"""Backend-agnostic speech schemas and speaker-merge helpers.

Every ASR backend (Parakeet, faster-whisper, ...) emits ``ASRResult`` and every
diarizer emits ``list[SpeakerSegment]`` so the transcribe+diarize pipeline can
swap one stage without touching the others.
"""

from __future__ import annotations

from typing import Any

from daft import DataType

# One transcript line. `speaker` is filled in after diarization (empty until then).
SegmentStruct = DataType.struct(
    {
        "id": DataType.int64(),
        "start": DataType.float64(),
        "end": DataType.float64(),
        "text": DataType.string(),
        "speaker": DataType.string(),
    }
)

# One diarization turn.
SpeakerSegmentStruct = DataType.struct(
    {
        "start": DataType.float64(),
        "end": DataType.float64(),
        "speaker": DataType.string(),
    }
)

InfoStruct = DataType.struct(
    {
        "language": DataType.string(),
        "duration": DataType.float64(),
    }
)

# The contract every ASR backend returns.
ASRResult = DataType.struct(
    {
        "transcript": DataType.string(),
        "segments": DataType.list(SegmentStruct),
        "info": InfoStruct,
    }
)


def dominant_speaker(start: float, end: float, speaker_segments: list[dict[str, Any]]) -> str | None:
    """Speaker whose diarization turn overlaps [start, end] the most."""
    best, best_overlap = None, 0.0
    for segment in speaker_segments:
        overlap = min(end, segment["end"]) - max(start, segment["start"])
        if overlap > best_overlap:
            best_overlap = overlap
            best = segment["speaker"]
    return best


def merge_speakers(segments: list[dict[str, Any]], speaker_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign each transcript segment the max-overlap diarization speaker."""
    for segment in segments:
        segment["speaker"] = dominant_speaker(segment["start"], segment["end"], speaker_segments) or ""
    return segments
