"""NVIDIA Parakeet (and Canary) ASR as a Daft class UDF.

Backend: PyTorch via NeMo. Parakeet-TDT-0.6B is the throughput champion of the
ASR candidates (RTFx ~3.4k vs Canary ~0.75k), with first-class word/segment
timestamps and a CC-BY-4.0 license.

- ``nvidia/parakeet-tdt-0.6b-v2`` — English, fastest. The "fastest pipeline" default.
- ``nvidia/parakeet-tdt-0.6b-v3`` — 25 European languages, long-audio local attention.
- ``nvidia/canary-1b-v2`` / ``canary-1b-flash`` — multilingual + translation, slower.

VAD silence-compaction is an optional, engine-agnostic front-end (see
``models.common.audio``) — it strips non-speech before the ASR pass and restores
timestamps after. This module never imports ``modal``; see ``modal_app.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Keep `models.*` importable when run as a loose script under `modal run`.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import daft
from daft import Series, col
from daft.functions import audio_file, audio_metadata, unnest
from models.common.audio import (
    SAMPLE_RATE,
    compact_speech,
    empty_result,
    read_waveform_16k_mono,
    restore_segment_bounds,
    segments_to_result,
)
from models.common.speech import ASRResult
from models.common.vad import build_vad

DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v2"
# Daft row-batch sizes the benchmark can sweep. Each needs its own
# @daft.method.batch (the decorator's batch_size is static at class definition).
BATCH_SIZES = (1, 8, 16, 32)


def _normalize_nemo_segments(
    nemo_output: Any,
    *,
    duration: float,
    windows,
    language: str,
) -> dict:
    """Convert a NeMo transcription result to the shared ASRResult contract."""
    timestamp = getattr(nemo_output, "timestamp", None) or {}
    raw_segments = timestamp.get("segment") or []
    segments = []
    for entry in raw_segments:
        text = (entry.get("segment") or entry.get("text") or "").strip()
        if not text:
            continue
        start, end = restore_segment_bounds(
            float(entry.get("start", 0.0)), float(entry.get("end", 0.0)), duration, windows
        )
        segments.append({"id": len(segments), "start": start, "end": end, "text": text, "speaker": ""})

    if not segments:
        text = (getattr(nemo_output, "text", "") or "").strip()
        if text:
            start = windows[0]["original_start"] if windows else 0.0
            end = windows[-1]["original_end"] if windows else duration
            segments.append({"id": 0, "start": start, "end": end, "text": text, "speaker": ""})

    return segments_to_result(segments, duration=duration, language=language)


@daft.cls(gpus=1.0, max_concurrency=1)
class ParakeetASR:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        vad: str = "none",
        language: str = "",
        long_audio: bool = False,
        vad_threshold: float = 0.5,
        vad_min_speech_duration_ms: int = 250,
        vad_min_silence_duration_ms: int = 500,
        vad_speech_pad_ms: int = 200,
    ):
        import nemo.collections.asr as nemo_asr

        self.model_name = model
        self.language = language
        self.vad_name = vad
        self.asr = nemo_asr.models.ASRModel.from_pretrained(model_name=model)
        # Parakeet-TDT's greedy decoder captures a shape-specific CUDA graph; replaying
        # it across Daft batches of differing shapes raises cudaErrorIllegalAddress.
        # Disable the graph decoder for stable batch-to-batch inference.
        self._disable_cuda_graph_decoder()
        if long_audio and hasattr(self.asr, "change_attention_model"):
            # Parakeet v3: switch to local attention to fit multi-hour audio.
            self.asr.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[256, 256])
        self.vad = (
            build_vad(
                vad,
                threshold=vad_threshold,
                min_speech_duration_ms=vad_min_speech_duration_ms,
                min_silence_duration_ms=vad_min_silence_duration_ms,
                speech_pad_ms=vad_speech_pad_ms,
            )
            if vad != "none"
            else None
        )

    def _disable_cuda_graph_decoder(self) -> None:
        try:
            from omegaconf import OmegaConf, open_dict

            decoding_cfg = self.asr.cfg.decoding

            # TDT/RNNT decoders expose the CUDA-graph toggle under different keys
            # across model versions (`use_cuda_graph_decoder` on v2, `allow_cuda_graphs`
            # on v3's TDT greedy config). Replaying a captured graph across Daft
            # batches of differing audio shapes segfaults (cudaErrorIllegalAddress),
            # so clear every variant we find, at any nesting depth.
            def _disable_all(node) -> None:
                if OmegaConf.is_dict(node):
                    for key in ("use_cuda_graph_decoder", "allow_cuda_graphs"):
                        if key in node:
                            node[key] = False
                    for value in node.values():
                        _disable_all(value)
                elif OmegaConf.is_list(node):
                    for value in node:
                        _disable_all(value)

            with open_dict(decoding_cfg):
                _disable_all(decoding_cfg)
            self.asr.change_decoding_strategy(decoding_cfg)
        except Exception as exc:  # noqa: BLE001 — best-effort; fall back to default decoder
            print(f"parakeet: could not disable cuda graph decoder ({exc})")

    def _prepare(self, audio, duration, tmpdir: Path) -> tuple[str | None, dict, list]:
        """Read → optional VAD compaction → temp 16 kHz wav. Returns (path, meta, windows)."""
        import soundfile as sf

        waveform = read_waveform_16k_mono(audio)
        windows: list = []
        compacted_duration = float(duration or len(waveform) / SAMPLE_RATE)
        if self.vad is not None:
            waveform, windows, compacted_duration = compact_speech(waveform, self.vad.speech_timestamps(waveform))
        if not len(waveform):
            return None, {"duration": float(duration or 0.0), "compacted_duration": 0.0}, windows
        path = tmpdir / f"clip_{abs(hash(audio)) & 0xFFFFFFFF:08x}.wav"
        sf.write(str(path), waveform, SAMPLE_RATE)
        return str(path), {"duration": float(duration or 0.0), "compacted_duration": compacted_duration}, windows

    def _transcribe_many(self, audios: list, durations: list) -> list[dict]:
        import tempfile

        with tempfile.TemporaryDirectory() as raw_tmp:
            tmpdir = Path(raw_tmp)
            results: list[dict | None] = [None] * len(audios)
            paths: list[str] = []
            index_meta: list[tuple[int, dict, list]] = []
            for index, (audio, duration) in enumerate(zip(audios, durations, strict=True)):
                path, meta, windows = self._prepare(audio, duration, tmpdir)
                if path is None:
                    results[index] = empty_result(meta["duration"], self.language)
                    continue
                paths.append(path)
                index_meta.append((index, meta, windows))

            if paths:
                outputs = self.asr.transcribe(paths, timestamps=True, batch_size=len(paths))
                for (index, meta, windows), output in zip(index_meta, outputs, strict=True):
                    results[index] = _normalize_nemo_segments(
                        output,
                        duration=meta["compacted_duration"] if windows else meta["duration"],
                        windows=windows,
                        language=self.language,
                    )
                    # Report original duration for the throughput metric.
                    results[index]["info"]["duration"] = meta["duration"]

            return [r if r is not None else empty_result(0.0, self.language) for r in results]

    # Row-batch ladder — pick one via getattr(processor, f"transcribe_{n}").
    @daft.method.batch(return_dtype=ASRResult, batch_size=1)
    def transcribe_1(self, audio: Series, duration: Series):
        return self._transcribe_many(audio.to_pylist(), duration.to_pylist())

    @daft.method.batch(return_dtype=ASRResult, batch_size=8)
    def transcribe_8(self, audio: Series, duration: Series):
        return self._transcribe_many(audio.to_pylist(), duration.to_pylist())

    @daft.method.batch(return_dtype=ASRResult, batch_size=16)
    def transcribe_16(self, audio: Series, duration: Series):
        return self._transcribe_many(audio.to_pylist(), duration.to_pylist())

    @daft.method.batch(return_dtype=ASRResult, batch_size=32)
    def transcribe_32(self, audio: Series, duration: Series):
        return self._transcribe_many(audio.to_pylist(), duration.to_pylist())


def attach_transcript(df: daft.DataFrame, processor: ParakeetASR, *, row_batch_size: int = 16) -> daft.DataFrame:
    """Add audio/duration/transcript columns using the chosen row-batch method."""
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


if __name__ == "__main__":
    # Local smoke test requires NeMo + a GPU; normally run via modal_app.py.
    source = "hf://datasets/Eventual-Inc/sample-files/audio/*.mp3"
    processor = ParakeetASR()
    out = (
        attach_transcript(daft.from_glob_path(source).where(col("size") > 0), processor)
        .select("path", "transcript", unnest(col("info")))
        .collect()
    )
    out.show(format="fancy", max_width=60)
