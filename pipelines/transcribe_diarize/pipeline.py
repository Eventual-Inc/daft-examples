"""Compose pluggable ASR / diarizer stages into one transcribe+diarize DataFrame.

``build_dataframe`` dispatches on backend names. Only the backends for the
chosen config are imported, so this module loads cheaply in either container
lane — but a given call must run in a lane whose image has those deps (NeMo
backends in ``nemo_image``, Whisper backends in ``whisper_image``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import daft
from daft import col
from daft.functions import audio_file, audio_metadata
from models.common.speech import SegmentStruct, merge_speakers

AUDIO_EXTENSION_RE = r".*\.(aac|flac|m4a|mp3|ogg|opus|wav)$"

# Which lane each backend belongs to — the benchmark uses this to route configs.
ASR_LANE = {"parakeet": "nemo", "canary": "nemo", "faster_whisper": "whisper"}
DIARIZER_LANE = {"sortformer": "nemo", "pyannote": "whisper"}


@daft.func(return_dtype=daft.DataType.list(SegmentStruct))
def apply_speakers(segments: list[dict[str, Any]], speaker_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return merge_speakers(segments, speaker_segments)


def resolve_source(source: str) -> str:
    if "://" not in source and Path(source).is_dir():
        return str(Path(source) / "*")
    return source


def _build_asr(asr: str, *, model: str = "", vad: str = "none", language: str = "", **kwargs):
    if asr == "parakeet" or asr == "canary":
        from models.parakeet.model import ParakeetASR, attach_transcript

        processor_kwargs = {"vad": vad, "language": language, **kwargs}
        if model:
            processor_kwargs["model"] = model
        processor = ParakeetASR(**processor_kwargs)
        return processor, attach_transcript
    if asr == "faster_whisper":
        from models.faster_whisper.asr import FasterWhisperASR, attach_transcript

        whisper_vad = vad if vad in ("builtin", "none") else "builtin"
        processor_kwargs = {"vad": whisper_vad, "language": language, **kwargs}
        if model:
            processor_kwargs["model"] = model
        processor = FasterWhisperASR(**processor_kwargs)
        return processor, attach_transcript
    raise ValueError(f"unknown ASR backend '{asr}'")


def _build_diarizer(diarizer: str, *, model: str = ""):
    if diarizer == "sortformer":
        from models.sortformer.model import SortformerDiarizer, attach_speakers

        return SortformerDiarizer(**({"model": model} if model else {})), attach_speakers
    if diarizer == "pyannote":
        from models.pyannote.model import PyannoteDiarizer, attach_speakers

        return PyannoteDiarizer(**({"model": model} if model else {})), attach_speakers
    if diarizer == "none":
        return None, None
    raise ValueError(f"unknown diarizer backend '{diarizer}'")


def _build_nemo_dataframe(
    source: str,
    *,
    asr: str,
    diarizer: str,
    vad: str,
    asr_model: str,
    diarizer_model: str,
    language: str,
    asr_kwargs: dict | None,
    include_vad_stats: bool,
) -> daft.DataFrame:
    from models.parakeet.model import TranscribeDiarizeVad

    if asr not in ("parakeet", "canary"):
        raise ValueError(f"unknown NeMo ASR backend '{asr}'")
    if diarizer not in ("sortformer", "none"):
        raise ValueError(f"unknown NeMo diarizer backend '{diarizer}'")

    processor_kwargs = {"vad": vad, "diarizer": diarizer, "language": language, **(asr_kwargs or {})}
    if asr_model:
        processor_kwargs["asr_model"] = asr_model
    if diarizer_model:
        processor_kwargs["diarizer_model"] = diarizer_model
    processor = TranscribeDiarizeVad(**processor_kwargs)
    df = (
        daft.from_glob_path(resolve_source(source))
        .where(col("size") > 0)
        .where(col("path").lower().regexp(AUDIO_EXTENSION_RE))
        .with_column("audio", audio_file(col("path")))
        .with_column("audio_metadata", audio_metadata(col("audio")))
        .with_column("duration", col("audio_metadata")["frames"] / col("audio_metadata")["sample_rate"])
        .with_column("out", processor.process(col("audio"), col("duration")))
        .with_column("transcript", col("out")["transcript"])
        .with_column("segments", col("out")["segments"])
        .with_column("speaker_segments", col("out")["speaker_segments"])
        .with_column("info", col("out")["info"])
        .where(col("transcript").length() > 0)
    )

    select_cols = ["path", "size", "transcript", "segments"]
    if diarizer == "sortformer":
        select_cols.append("speaker_segments")
    select_cols.append("info")
    if include_vad_stats:
        df = df.with_column("vad_speech_seconds", col("out")["vad_speech_seconds"]).with_column(
            "vad_seconds_removed", col("out")["vad_seconds_removed"]
        )
        select_cols.extend(["vad_speech_seconds", "vad_seconds_removed"])
    return df.select(*select_cols)


def build_dataframe(
    source: str,
    *,
    asr: str = "parakeet",
    diarizer: str = "sortformer",
    vad: str = "none",
    asr_model: str = "",
    diarizer_model: str = "",
    language: str = "",
    row_batch_size: int = 16,
    asr_kwargs: dict | None = None,
    include_vad_stats: bool = False,
) -> daft.DataFrame:
    """Build the transcribe(+diarize) DataFrame for one backend configuration."""
    asr_lane = ASR_LANE.get(asr)
    diarizer_lane = DIARIZER_LANE.get(diarizer, asr_lane)
    if asr_lane is None:
        raise ValueError(f"unknown ASR backend '{asr}'")
    if diarizer_lane != asr_lane:
        raise ValueError(
            f"ASR backend '{asr}' and diarizer '{diarizer}' are in different lanes and cannot run in one pass"
        )
    if asr_lane == "nemo":
        return _build_nemo_dataframe(
            source,
            asr=asr,
            diarizer=diarizer,
            vad=vad,
            asr_model=asr_model,
            diarizer_model=diarizer_model,
            language=language,
            asr_kwargs=asr_kwargs,
            include_vad_stats=include_vad_stats,
        )

    asr_processor, attach_transcript = _build_asr(
        asr, model=asr_model, vad=vad, language=language, **(asr_kwargs or {})
    )

    df = (
        daft.from_glob_path(resolve_source(source))
        .where(col("size") > 0)
        .where(col("path").lower().regexp(AUDIO_EXTENSION_RE))
    )
    df = attach_transcript(df, asr_processor, row_batch_size=row_batch_size)
    df = df.where(col("transcript").length() > 0)

    diarizer_processor, attach_speakers = _build_diarizer(diarizer, model=diarizer_model)
    if diarizer_processor is not None:
        df = attach_speakers(df, diarizer_processor)
        df = df.with_column("segments", apply_speakers(col("segments"), col("speaker_segments")))
        select_cols = ["path", "size", "transcript", "segments", "speaker_segments", "info"]
    else:
        select_cols = ["path", "size", "transcript", "segments", "info"]

    return df.select(*select_cols)
