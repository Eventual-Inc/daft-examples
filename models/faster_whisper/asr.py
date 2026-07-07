"""Benchmark-grade faster-whisper ASR backend emitting the shared ASRResult.

Backend: CTranslate2. This is the Whisper-lane ASR for the transcribe+diarize
swap matrix — `large-v3` is a near-drop-in for Transformers `large-v3` (same
weights, same WER, ~4× throughput). `turbo` and `distil-large-v3` are opt-in
speed tiers (turbo: small WER regression; distil: English-only).

Distinct from ``model.py``'s ``FasterWhisperTranscriber``, which is the simple
standalone example with its own richer schema. This one conforms to
``models.common.speech.ASRResult`` so it's swappable with Parakeet.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import daft
from daft import Series, col
from daft.functions import audio_file, audio_metadata
from models.common.audio import empty_result, read_waveform_16k_mono, segments_to_result
from models.common.speech import ASRResult

DEFAULT_MODEL = "large-v3"
BATCH_SIZES = (1, 8, 16, 32)


@daft.cls(gpus=1.0, max_concurrency=1)
class FasterWhisperASR:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        compute_type: str = "float16",
        vad: str = "builtin",
        language: str = "",
        asr_chunk_batch_size: int = 16,
    ):
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        self.model_name = model
        self.language = language or None
        # faster-whisper has its own Silero VAD via vad_filter; "builtin" uses it,
        # "none" disables. (MarbleNet/external compaction lives in the NeMo lane.)
        self.vad_filter = vad == "builtin"
        self.asr_chunk_batch_size = asr_chunk_batch_size
        self.whisper = WhisperModel(model, device="cuda", compute_type=compute_type)
        self.pipe = BatchedInferencePipeline(model=self.whisper)

    def _transcribe_one(self, audio, duration) -> dict:
        # Per-file fault isolation: a single undecodable/oversized/odd file returns
        # an empty result instead of crashing the whole Daft batch — essential for
        # heterogeneous real-world folders (multi-hour files, music, junk).
        try:
            waveform = read_waveform_16k_mono(audio)  # float32 mono 16 kHz, [-1, 1]
            if not len(waveform):
                return empty_result(float(duration or 0.0))
            segments_iter, info = self.pipe.transcribe(
                waveform,
                batch_size=self.asr_chunk_batch_size,
                word_timestamps=False,
                vad_filter=self.vad_filter,
                language=self.language,
            )
            segments = []
            for seg in segments_iter:
                text = (seg.text or "").strip()
                if not text:
                    continue
                segments.append(
                    {"id": len(segments), "start": float(seg.start), "end": float(seg.end), "text": text, "speaker": ""}
                )
            return segments_to_result(segments, duration=float(duration or info.duration), language=info.language or "")
        except Exception as exc:  # noqa: BLE001 — isolate per-file failures
            print(f"faster_whisper: transcription failed for one file ({type(exc).__name__}: {exc})")
            return empty_result(float(duration or 0.0))

    @daft.method.batch(return_dtype=ASRResult, batch_size=1)
    def transcribe_1(self, audio: Series, duration: Series):
        return [self._transcribe_one(a, d) for a, d in zip(audio.to_pylist(), duration.to_pylist())]

    @daft.method.batch(return_dtype=ASRResult, batch_size=8)
    def transcribe_8(self, audio: Series, duration: Series):
        return [self._transcribe_one(a, d) for a, d in zip(audio.to_pylist(), duration.to_pylist())]

    @daft.method.batch(return_dtype=ASRResult, batch_size=16)
    def transcribe_16(self, audio: Series, duration: Series):
        return [self._transcribe_one(a, d) for a, d in zip(audio.to_pylist(), duration.to_pylist())]

    @daft.method.batch(return_dtype=ASRResult, batch_size=32)
    def transcribe_32(self, audio: Series, duration: Series):
        return [self._transcribe_one(a, d) for a, d in zip(audio.to_pylist(), duration.to_pylist())]


def attach_transcript(df: daft.DataFrame, processor: FasterWhisperASR, *, row_batch_size: int = 16) -> daft.DataFrame:
    if row_batch_size not in BATCH_SIZES:
        raise ValueError(f"row_batch_size must be one of {BATCH_SIZES}")
    transcribe = getattr(processor, f"transcribe_{row_batch_size}")
    return (
        df.with_column("audio", audio_file(col("path")))
        .with_column("audio_metadata", audio_metadata(col("audio")))
        .with_column("duration", col("audio_metadata")["frames"] / col("audio_metadata")["sample_rate"])
        .with_column("tx", transcribe(col("audio"), col("duration")))
        .with_column("transcript", col("tx")["transcript"])
        .with_column("segments", col("tx")["segments"])
        .with_column("info", col("tx")["info"])
    )
