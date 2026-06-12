"""Backend-agnostic audio helpers: 16 kHz mono reads and VAD silence-compaction.

The compaction trick (lifted from the marketing transcribe+diarize pipeline) is
the highest-leverage throughput lever: strip non-speech *before* the expensive
ASR pass, then restore original timestamps afterward. It is engine-agnostic — it
takes speech windows from any VAD and feeds compacted audio to any ASR.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np

import daft

SAMPLE_RATE = 16000


class SpeechWindow(TypedDict):
    compact_start: float  # seconds, position in the silence-stripped waveform
    compact_end: float
    original_start: float  # seconds, position in the source waveform
    original_end: float


def read_waveform_16k_mono(audio: daft.AudioFile) -> np.ndarray:
    """Resample to 16 kHz, downmix to mono, return a contiguous float32 array."""
    waveform = np.asarray(audio.resample(SAMPLE_RATE), dtype=np.float32)
    if waveform.ndim == 2:
        channel_axis = 0 if waveform.shape[0] <= 8 else 1
        waveform = waveform.mean(axis=channel_axis)
    return np.ascontiguousarray(waveform.reshape(-1))


def compact_speech(
    waveform: np.ndarray,
    speech_timestamps: list[dict[str, int]],
) -> tuple[np.ndarray, list[SpeechWindow], float]:
    """Concatenate speech-only spans and record the original↔compact time map.

    ``speech_timestamps`` are sample-index ``{"start", "end"}`` spans from a VAD.
    Returns the compacted waveform, the windows needed to restore timestamps, and
    the compacted duration in seconds.
    """
    if not speech_timestamps:
        return np.array([], dtype=np.float32), [], 0.0

    chunks: list[np.ndarray] = []
    windows: list[SpeechWindow] = []
    compact_cursor = 0
    for timestamp in speech_timestamps:
        start = int(timestamp["start"])
        end = int(timestamp["end"])
        if end <= start:
            continue
        chunks.append(waveform[start:end])
        chunk_len = end - start
        windows.append(
            {
                "compact_start": compact_cursor / SAMPLE_RATE,
                "compact_end": (compact_cursor + chunk_len) / SAMPLE_RATE,
                "original_start": start / SAMPLE_RATE,
                "original_end": end / SAMPLE_RATE,
            }
        )
        compact_cursor += chunk_len

    if not chunks:
        return np.array([], dtype=np.float32), [], 0.0

    compact_waveform = np.ascontiguousarray(np.concatenate(chunks))
    return compact_waveform, windows, len(compact_waveform) / SAMPLE_RATE


def restore_original_time(compact_time: float, windows: list[SpeechWindow], *, is_end: bool) -> float:
    """Map a timestamp in compacted time back to the original waveform."""
    if not windows:
        return compact_time
    if compact_time <= 0:
        return windows[0]["original_start"]
    for window in windows:
        if compact_time < window["compact_end"] or (is_end and compact_time <= window["compact_end"]):
            offset = max(0.0, compact_time - window["compact_start"])
            return min(window["original_end"], window["original_start"] + offset)
    return windows[-1]["original_end"]


def restore_segment_bounds(
    start: float, end: float, duration: float, windows: list[SpeechWindow] | None
) -> tuple[float, float]:
    """Restore a [start, end] segment from compacted to original time."""
    if windows:
        start = restore_original_time(start, windows, is_end=False)
        end = restore_original_time(end, windows, is_end=True)
    end = max(start, end if end is not None else duration)
    return start, end


def segments_to_result(
    segments: list[dict[str, Any]],
    *,
    duration: float,
    language: str = "",
) -> dict[str, Any]:
    """Wrap normalized segments in the ``ASRResult`` contract from common.speech."""
    transcript = " ".join(segment["text"] for segment in segments).strip()
    return {
        "transcript": transcript,
        "segments": segments,
        "info": {"language": language, "duration": duration},
    }


def empty_result(duration: float, language: str = "") -> dict[str, Any]:
    return {"transcript": "", "segments": [], "info": {"language": language, "duration": duration}}
