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
import time
from pathlib import Path
from typing import Any

# Keep `models.*` importable when run as a loose script under `modal run`.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import daft
from daft import DataType, col
from daft.functions import audio_file, audio_metadata, unnest
from models.common.audio import (
    SAMPLE_RATE,
    compact_speech,
    empty_result,
    read_waveform_16k_mono,
    restore_segment_bounds,
    segments_to_result,
)
from models.common.speech import ASRResult, InfoStruct, SegmentStruct, SpeakerSegmentStruct, merge_speakers
from models.common.vad import build_vad

TranscribeDiarizeResult = DataType.struct(
    {
        "transcript": DataType.string(),
        "segments": DataType.list(SegmentStruct),
        "speaker_segments": DataType.list(SpeakerSegmentStruct),
        "info": InfoStruct,
        "vad_speech_seconds": DataType.float64(),
        "vad_seconds_removed": DataType.float64(),
    }
)


def _runtime_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - torch may not be importable in lightweight contexts
        return "cpu"


def _move_to_runtime_device(model: Any) -> Any:
    if model is not None and hasattr(model, "to"):
        return model.to(_runtime_device())
    return model


def _disable_cuda_graph_decoder(asr: Any) -> None:
    try:
        from omegaconf import OmegaConf, open_dict

        decoding_cfg = asr.cfg.decoding

        for path in (
            "greedy.use_cuda_graph_decoder",
            "greedy.allow_cuda_graphs",
            "beam.use_cuda_graph_decoder",
            "beam.allow_cuda_graphs",
        ):
            OmegaConf.update(decoding_cfg, path, False, merge=False, force_add=True)

        # TDT/RNNT decoders expose the CUDA-graph toggle under different keys
        # across model versions (`use_cuda_graph_decoder` on v2, `allow_cuda_graphs`
        # on v3's TDT greedy config). Replaying a captured graph across Daft
        # batches of differing audio shapes segfaults (cudaErrorIllegalAddress),
        # so clear every variant we find, at any nesting depth.
        def _disable_all(node) -> None:
            if OmegaConf.is_dict(node):
                with open_dict(node):
                    for key in ("use_cuda_graph_decoder", "allow_cuda_graphs"):
                        if key in node:
                            node[key] = False
                    for value in node.values():
                        _disable_all(value)
            elif OmegaConf.is_list(node):
                for value in node:
                    _disable_all(value)

        _disable_all(decoding_cfg)
        asr.change_decoding_strategy(decoding_cfg)
    except Exception as exc:  # noqa: BLE001 — best-effort; fall back to default decoder
        print(f"parakeet: could not disable cuda graph decoder ({exc})")


def _timestamp_text(entry: Any, *keys: str) -> str:
    if isinstance(entry, dict):
        for key in keys:
            value = entry.get(key)
            if value is not None:
                return str(value).strip()
    for key in keys:
        value = getattr(entry, key, None)
        if value is not None:
            return str(value).strip()
    return ""


def _timestamp_float(entry: Any, *keys: str, default: float = 0.0) -> float:
    if isinstance(entry, dict):
        for key in keys:
            value = entry.get(key)
            if value is not None:
                return float(value)
    for key in keys:
        value = getattr(entry, key, None)
        if value is not None:
            return float(value)
    return default


def _timestamp_bounds(entry: Any) -> tuple[float, float]:
    start = _timestamp_float(entry, "start", "start_offset", "start_time", default=0.0)
    end = _timestamp_float(entry, "end", "end_offset", "end_time", default=start)
    return start, max(start, end)


def _overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _normalize_nemo_words(raw_words: list[Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for entry in raw_words:
        text = _timestamp_text(entry, "word", "text")
        if not text:
            continue
        start, end = _timestamp_bounds(entry)
        words.append({"text": text, "start": start, "end": end})
    return words


def _assign_words_to_segments(
    words: list[dict[str, Any]],
    segment_bounds: list[tuple[float, float]],
    *,
    duration: float,
    windows,
) -> list[list[dict[str, Any]]]:
    buckets: list[list[dict[str, Any]]] = [[] for _ in segment_bounds]
    if not words or not segment_bounds:
        return buckets

    for word in words:
        best_index = 0
        best_overlap = -1.0
        word_start = float(word["start"])
        word_end = float(word["end"])
        word_mid = (word_start + word_end) / 2
        for index, (seg_start, seg_end) in enumerate(segment_bounds):
            overlap = _overlap_seconds(word_start, word_end, seg_start, seg_end)
            if overlap == 0.0 and seg_start <= word_mid <= seg_end:
                overlap = 1e-9
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = index
        start, end = restore_segment_bounds(word_start, word_end, duration, windows)
        buckets[best_index].append({"text": word["text"], "start": start, "end": end})
    return buckets


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
    words = _normalize_nemo_words(timestamp.get("word") or [])
    raw_bounds: list[tuple[float, float]] = []
    segments = []
    for entry in raw_segments:
        text = _timestamp_text(entry, "segment", "text")
        if not text:
            continue
        raw_start, raw_end = _timestamp_bounds(entry)
        start, end = restore_segment_bounds(raw_start, raw_end, duration, windows)
        raw_bounds.append((raw_start, raw_end))
        segments.append({"id": len(segments), "start": start, "end": end, "text": text, "speaker": "", "words": []})

    word_buckets = _assign_words_to_segments(words, raw_bounds, duration=duration, windows=windows)
    for segment, bucket in zip(segments, word_buckets, strict=True):
        segment["words"] = bucket

    if not segments:
        text = (getattr(nemo_output, "text", "") or "").strip()
        if text:
            start = windows[0]["original_start"] if windows else 0.0
            end = windows[-1]["original_end"] if windows else duration
            restored_words = []
            for word in words:
                word_start, word_end = restore_segment_bounds(word["start"], word["end"], duration, windows)
                restored_words.append({"text": word["text"], "start": word_start, "end": word_end})
            segments.append({"id": 0, "start": start, "end": end, "text": text, "speaker": "", "words": restored_words})

    return segments_to_result(segments, duration=duration, language=language)


@daft.cls(gpus=1.0, max_concurrency=1)
class ParakeetASR:
    def __init__(
        self,
        *,
        model: str = "nvidia/parakeet-tdt-0.6b-v2",
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
        _disable_cuda_graph_decoder(self.asr)

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

    def _transcribe_one(self, audio, duration) -> dict:
        import tempfile

        with tempfile.TemporaryDirectory() as raw_tmp:
            tmpdir = Path(raw_tmp)
            path, meta, windows = self._prepare(audio, duration, tmpdir)
            if path is None:
                return empty_result(meta["duration"], self.language)
            output = self.asr.transcribe([path], timestamps=True)[0]

        result = _normalize_nemo_segments(
            output,
            duration=meta["compacted_duration"] if windows else meta["duration"],
            windows=windows,
            language=self.language,
        )
        result["info"]["duration"] = meta["duration"]
        return result

    @daft.method(return_dtype=ASRResult)
    def transcribe(self, audio: daft.AudioFile, duration: float):
        return self._transcribe_one(audio, duration)


def _as_waveform(waveform: Any):
    import numpy as np

    array = np.asarray(waveform, dtype=np.float32)
    return np.ascontiguousarray(array.reshape(-1))


def _write_tmp_waveform(waveform: Any, tmpdir: Path, name: str) -> str | None:
    import soundfile as sf

    array = _as_waveform(waveform)
    if not len(array):
        return None
    path = tmpdir / name
    sf.write(str(path), array, SAMPLE_RATE)
    return str(path)


@daft.cls(gpus=1.0, max_concurrency=1, use_process=False)
class TranscribeDiarizeVad:
    """Fused Parakeet + MarbleNet + Sortformer pipeline in one Daft actor.

    This is the device-resident path: the model objects live on ``self`` and one
    Daft method owns the full per-file flow, avoiding host round-trips between
    VAD, ASR, and diarization graph nodes.
    """

    def __init__(
        self,
        *,
        asr_model: str = "nvidia/parakeet-tdt-0.6b-v2",
        vad: str = "marblenet",
        vad_model: str = "nvidia/frame_vad_multilingual_marblenet_v2.0",
        diarizer: str = "sortformer",
        diarizer_model: str = "nvidia/diar_streaming_sortformer_4spk-v2",
        language: str = "",
        long_audio: bool = False,
        vad_threshold: float = 0.5,
        vad_min_speech_duration_ms: int = 250,
        vad_min_silence_duration_ms: int = 500,
        vad_speech_pad_ms: int = 200,
    ):
        import nemo.collections.asr as nemo_asr

        self.language = language
        self.vad_name = vad
        self.diarizer_name = diarizer
        self.parakeet = _move_to_runtime_device(nemo_asr.models.ASRModel.from_pretrained(model_name=asr_model))
        _disable_cuda_graph_decoder(self.parakeet)
        if long_audio and hasattr(self.parakeet, "change_attention_model"):
            self.parakeet.change_attention_model(
                self_attention_model="rel_pos_local_attn",
                att_context_size=[256, 256],
            )

        if vad == "none":
            self.vad = build_vad("none")
        elif vad == "marblenet":
            from models.common.vad import MarbleNetVAD

            self.vad = MarbleNetVAD(
                model_id=vad_model,
                threshold=vad_threshold,
                min_speech_duration_ms=vad_min_speech_duration_ms,
                min_silence_duration_ms=vad_min_silence_duration_ms,
                speech_pad_ms=vad_speech_pad_ms,
            )
        else:
            self.vad = build_vad(
                vad,
                threshold=vad_threshold,
                min_speech_duration_ms=vad_min_speech_duration_ms,
                min_silence_duration_ms=vad_min_silence_duration_ms,
                speech_pad_ms=vad_speech_pad_ms,
            )

        self.sortformer = None
        if diarizer == "sortformer":
            from nemo.collections.asr.models import SortformerEncLabelModel

            self.sortformer = _move_to_runtime_device(SortformerEncLabelModel.from_pretrained(diarizer_model).eval())
        elif diarizer != "none":
            raise ValueError(f"unknown NeMo diarizer backend '{diarizer}'")

    def _log_timing(self, stage: str, started: float, **extra: Any) -> None:
        fields = " ".join(f"{key}={value}" for key, value in extra.items())
        print(f"transcribe_diarize_vad.{stage} seconds={time.perf_counter() - started:.4f} {fields}".rstrip())

    def _vad_windows(self, waveform: Any) -> tuple[list[dict[str, Any]], float]:
        array = _as_waveform(waveform)
        spans = self.vad.speech_timestamps(array)
        windows: list[dict[str, Any]] = []
        for span in spans:
            sample_start = max(0, min(len(array), int(span["start"])))
            sample_end = max(sample_start, min(len(array), int(span["end"])))
            if sample_end <= sample_start:
                continue
            windows.append(
                {
                    "sample_start": sample_start,
                    "sample_end": sample_end,
                    "start": sample_start / SAMPLE_RATE,
                    "end": sample_end / SAMPLE_RATE,
                    "duration": (sample_end - sample_start) / SAMPLE_RATE,
                }
            )
        return windows, sum(window["duration"] for window in windows)

    def _transcribe_paths(
        self,
        paths: list[str],
        windows: list[dict[str, Any]],
        *,
        source_duration: float,
    ) -> list[dict[str, Any]]:
        if not paths:
            return []
        outputs = self.parakeet.transcribe(paths, timestamps=True)
        all_segments: list[dict[str, Any]] = []
        for output, window in zip(outputs, windows, strict=True):
            duration = float(window["duration"])
            restore_windows = [
                {
                    "compact_start": 0.0,
                    "compact_end": duration,
                    "original_start": float(window["start"]),
                    "original_end": float(window["end"]),
                }
            ]
            result = _normalize_nemo_segments(
                output, duration=duration, windows=restore_windows, language=self.language
            )
            for segment in result["segments"]:
                if segment.get("text", "").strip():
                    all_segments.append(segment)
        all_segments.sort(key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))))
        for index, segment in enumerate(all_segments):
            segment["id"] = index
            segment.setdefault("speaker", "")
            segment.setdefault("words", [])
            segment["start"] = max(0.0, min(float(segment["start"]), source_duration))
            segment["end"] = max(segment["start"], min(float(segment["end"]), source_duration))
        return all_segments

    def _diarize_paths(
        self, paths: list[str], windows: list[dict[str, Any]], transcripts: list[str]
    ) -> list[dict[str, Any]]:
        if self.sortformer is None:
            return []
        from models.sortformer.model import _parse_turns

        active = [
            (path, window) for path, window, transcript in zip(paths, windows, transcripts, strict=True) if transcript
        ]
        if not active:
            return []
        try:
            raw = self.sortformer.diarize(audio=[path for path, _window in active])
        except Exception as exc:  # noqa: BLE001 - isolate per-file diarization failures
            print(f"sortformer: diarization failed ({type(exc).__name__}: {exc})")
            return []
        if len(active) == 1:
            grouped_turns = [_parse_turns(raw)]
        elif isinstance(raw, list) and len(raw) == len(active):
            grouped_turns = [_parse_turns(item if isinstance(item, list) else [item]) for item in raw]
        else:
            grouped_turns = [_parse_turns(raw), *([] for _path, _window in active[1:])]

        speaker_segments: list[dict[str, Any]] = []
        for (_path, window), turns in zip(active, grouped_turns, strict=True):
            offset = float(window["start"])
            speaker_segments.extend(
                {"start": turn["start"] + offset, "end": turn["end"] + offset, "speaker": turn["speaker"]}
                for turn in turns
            )
        speaker_segments.sort(key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))))
        return speaker_segments

    @daft.method(return_dtype=TranscribeDiarizeResult)
    def process(self, audio: daft.AudioFile, duration: float):
        import tempfile

        started = time.perf_counter()
        waveform = read_waveform_16k_mono(audio)
        source_duration = float(duration or len(waveform) / SAMPLE_RATE)
        windows, vad_speech_seconds = self._vad_windows(waveform)
        if not windows:
            self._log_timing("process", started, windows=0, segments=0)
            return {
                "transcript": "",
                "segments": [],
                "speaker_segments": [],
                "info": {"language": self.language, "duration": source_duration},
                "vad_speech_seconds": 0.0,
                "vad_seconds_removed": source_duration,
            }

        paths: list[str] = []
        path_windows: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmpdir = Path(raw_tmp)
            array = _as_waveform(waveform)
            for index, window in enumerate(windows):
                chunk = array[int(window["sample_start"]) : int(window["sample_end"])]
                path = _write_tmp_waveform(chunk, tmpdir, f"speech_{index:05d}.wav")
                if path is None:
                    continue
                paths.append(path)
                path_windows.append(window)

            asr_started = time.perf_counter()
            segments = self._transcribe_paths(paths, path_windows, source_duration=source_duration)
            self._log_timing("asr", asr_started, windows=len(paths), segments=len(segments))

            text_by_window: list[str] = []
            for window in path_windows:
                start = float(window["start"])
                end = float(window["end"])
                pieces = [
                    segment.get("text", "")
                    for segment in segments
                    if _overlap_seconds(float(segment["start"]), float(segment["end"]), start, end) > 0.0
                ]
                text_by_window.append(" ".join(piece for piece in pieces if piece).strip())

            diar_started = time.perf_counter()
            speaker_segments = self._diarize_paths(paths, path_windows, text_by_window)
            self._log_timing("diarize", diar_started, windows=len(paths), segments=len(speaker_segments))

        if speaker_segments:
            segments = merge_speakers(segments, speaker_segments)
        transcript = " ".join(
            segment.get("text", "").strip() for segment in segments if segment.get("text", "").strip()
        )
        self._log_timing("process", started, windows=len(windows), segments=len(segments))
        return {
            "transcript": transcript,
            "segments": segments,
            "speaker_segments": speaker_segments,
            "info": {"language": self.language, "duration": source_duration},
            "vad_speech_seconds": float(vad_speech_seconds),
            "vad_seconds_removed": max(0.0, source_duration - float(vad_speech_seconds)),
        }


def attach_transcript(df: daft.DataFrame, processor: ParakeetASR, *, row_batch_size: int = 16) -> daft.DataFrame:
    """Add audio/duration/transcript columns using one Daft method call per row."""
    del row_batch_size
    return (
        df.with_column("audio", audio_file(col("path")))
        .with_column("audio_metadata", audio_metadata(col("audio")))
        .with_column("duration", col("audio_metadata")["frames"] / col("audio_metadata")["sample_rate"])
        .with_column("tx", processor.transcribe(col("audio"), col("duration")))
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
