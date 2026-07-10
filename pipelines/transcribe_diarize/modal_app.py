# /// script
# description = "Run the fastest transcribe+diarize pipeline and the swap-matrix benchmark on Modal"
# requires-python = ">=3.11, <3.13"
# dependencies = ["daft>=0.7.10", "modal"]
# ///
"""Modal deployment for the transcribe+diarize pipeline + swap-matrix benchmark.

Two GPU functions, one per container lane (NeMo / Whisper), because the two ASR
stacks can't share an image. The ``benchmark`` entrypoint routes each config to
its lane, measures RTFx + $/audio-hour, and writes a leaderboard. The ``run``
entrypoint executes the fastest config and writes the transcript table.

    # fastest end-to-end (Parakeet + Sortformer)
    uv run --extra models modal run pipelines/transcribe_diarize/modal_app.py::run \\
      --source 'hf://datasets/Eventual-Inc/sample-files/audio/*.mp3'

    # full swap matrix
    uv run --extra models modal run pipelines/transcribe_diarize/modal_app.py::benchmark
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import modal

# Anchor the repo root for local `modal run` (driver-side import resolution).
# Inside the Modal container the entrypoint lives at /root/modal_app.py — too
# shallow for parents[2] — and `models`/`pipelines` arrive via
# add_local_python_source, so the anchor is skipped there.
_here = Path(__file__).resolve()
if len(_here.parents) >= 3:
    _repo_root = str(_here.parents[2])
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

import os

from models.common.modal_images import MODAL_REGION, nemo_image, whisper_image
from models.common.modal_infra import MODEL_CACHE_DIR
from models.weights import HF_SECRET, MODEL_CACHE, OUTPUTS, VOLUMES, VOLUMES_WITH_OUTPUTS, cls_kwargs
from pipelines.transcribe_diarize.benchmark import SWAP_MATRIX, matrix_by_lane

# GPU is fixed per deploy; sweep it by re-running with BENCH_GPU set. Keep this
# lane on the goal's L40S/L4 target class instead of the repo-wide A100 default.
GPU = os.environ.get("BENCH_GPU", "L40S")
NEMO_REGION = [os.environ.get("BENCH_REGION", "us")]
SOURCE = "hf://datasets/Eventual-Inc/sample-files/audio/*.mp3"

AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac")
# When the same recording exists in several formats, keep one (decode-safe order).
DEDUPE_FORMAT_PREF = (".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wav")

app = modal.App("daft-transcribe-diarize")

_VOLUMES = VOLUMES
_SECRETS = [HF_SECRET]

# A scratch Volume for ad-hoc local folders pushed up for transcription.
# Files are transcoded to 16 kHz-mono FLAC client-side and land under FLAC_DIR so
# libsndfile (Daft's audio reader, which can't decode m4a/aac) handles them all.
AUDIO_DIR = "/audio"
FLAC_DIR = f"{AUDIO_DIR}/flac"
audio_volume = modal.Volume.from_name("daft-audio-dump", create_if_missing=True)
_NEMO_SERVICE_VOLUMES = {**VOLUMES_WITH_OUTPUTS, AUDIO_DIR: audio_volume}


def _reload_audio_volume_if_needed(source: str) -> None:
    if source.startswith(f"{AUDIO_DIR}/") or source == AUDIO_DIR:
        audio_volume.reload()


def _measure(config: dict, source: str, gpu: str) -> dict:
    """Run one config in the current image; return a BenchmarkRow dict.

    Warms the model on a 2-row sample (cold load + compile), then times a full
    collect so RTFx reflects steady-state throughput, not weight loading.
    """
    from models.common.metrics import BenchmarkRow
    from pipelines.transcribe_diarize.pipeline import ASR_LANE, build_dataframe

    row = BenchmarkRow(
        config=config["name"],
        lane=ASR_LANE[config["asr"]],
        asr=config["asr"],
        vad=config.get("vad", "none"),
        diarizer=config["diarizer"],
        gpu=gpu,
    )
    try:
        import torch

        df = build_dataframe(
            source,
            asr=config["asr"],
            diarizer=config["diarizer"],
            vad=config.get("vad", "none"),
            asr_model=config.get("asr_model", ""),
            diarizer_model=config.get("diarizer_model", ""),
            row_batch_size=config.get("row_batch_size", 16),
            asr_kwargs=config.get("asr_kwargs") or {},
        )
        df.limit(2).collect()  # warmup: pays model load + cuDNN autotune here

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = df.collect()
        row.wall_seconds = time.perf_counter() - start

        data = result.to_pydict()
        row.audio_seconds = float(sum((info or {}).get("duration", 0.0) for info in data.get("info", [])))
        row.peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
        row.extra = {"rows": len(data.get("path", []))}
    except Exception as exc:  # noqa: BLE001 — record the failure as a benchmark cell
        row.error = f"{type(exc).__name__}: {exc}"
    MODEL_CACHE.commit()
    return row.to_dict()


@app.cls(
    **cls_kwargs(
        nemo_image(),
        gpu=GPU,
        cpu=4,
        memory=32768,
        timeout=7200,
        region=NEMO_REGION,
        with_outputs=True,
        secrets=_SECRETS,
        volumes=_NEMO_SERVICE_VOLUMES,
        scaledown_window=300,
    )
)
class NemoPipelineService:
    def _status(self) -> dict:
        status = {
            "architecture": "fused_daft_cls",
            "model_residency": "TranscribeDiarizeVad.__init__",
        }
        try:
            import torch

            if torch.cuda.is_available():
                status["cuda_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
                status["cuda_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
        except Exception as exc:  # noqa: BLE001 - diagnostic only
            status["cuda_memory_error"] = f"{type(exc).__name__}: {exc}"
        return status

    @modal.method()
    def measure(self, config: dict, source: str, gpu: str) -> dict:
        _reload_audio_volume_if_needed(source)
        return _measure(config, source, gpu)

    @modal.method()
    def transcribe(self, config: dict, source: str, limit: int = 0, output_name: str = "") -> dict:
        from models.common.modal_infra import OUTPUT_DIR
        from pipelines.transcribe_diarize.pipeline import build_dataframe

        _reload_audio_volume_if_needed(source)
        df = build_dataframe(
            source,
            asr=config["asr"],
            diarizer=config["diarizer"],
            vad=config.get("vad", "marblenet"),
            asr_model=config.get("asr_model", ""),
            diarizer_model=config.get("diarizer_model", ""),
            row_batch_size=config.get("row_batch_size", 16),
            asr_kwargs=config.get("asr_kwargs") or {},
            include_vad_stats=True,
        )
        if limit:
            df = df.limit(limit)
        result = df.collect().to_pydict()
        MODEL_CACHE.commit()
        if output_name:
            output_dir = Path(OUTPUT_DIR)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / output_name
            output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
            OUTPUTS.commit()
            return {
                "source": source,
                "output_path": str(output_path),
                "rows": len(result.get("path", [])),
                "audio_seconds": float(sum((item or {}).get("duration", 0.0) for item in result.get("info", []))),
                "segments": sum(len(items or []) for items in result.get("segments", [])),
                "speaker_segments": sum(len(items or []) for items in result.get("speaker_segments", [])),
                "vad_seconds_removed": float(sum(result.get("vad_seconds_removed", []) or [])),
            }
        return result

    @modal.method()
    def status(self) -> dict:
        return self._status()

    @modal.method()
    def audio_files(self, prefix: str = FLAC_DIR) -> dict:
        _reload_audio_volume_if_needed(prefix)
        root = Path(prefix)
        files = sorted(str(path) for path in root.glob("*") if path.is_file())
        return {"prefix": prefix, "exists": root.exists(), "files": files}

    @modal.method()
    def bench_vad(
        self,
        source: str,
        thresholds: list[float],
        asr_model: str = "",
        row_batch_size: int = 16,
        limit: int = 0,
        max_wer_delta: float = 0.05,
        max_deletion_wer: float = 0.02,
    ) -> dict:
        from pipelines.transcribe_diarize.pipeline import build_dataframe

        _reload_audio_volume_if_needed(source)

        def collect(vad: str, threshold: float | None = None) -> dict:
            kwargs = {"vad_threshold": threshold} if threshold is not None else {}
            df = build_dataframe(
                source,
                asr="parakeet",
                diarizer="none",
                vad=vad,
                asr_model=asr_model,
                row_batch_size=row_batch_size,
                asr_kwargs=kwargs,
                include_vad_stats=True,
            )
            return df.collect().to_pydict()

        def filter_paths(data: dict, paths: list[str]) -> dict:
            if not paths:
                return data
            wanted = set(paths)
            indexes = [index for index, path in enumerate(data.get("path", [])) if path in wanted]
            return {
                key: [values[index] for index in indexes] if isinstance(values, list) else values
                for key, values in data.items()
            }

        baseline = collect("none")
        if limit:
            baseline = {key: values[:limit] if isinstance(values, list) else values for key, values in baseline.items()}
        sample_paths = baseline.get("path", [])
        rows = [_vad_row("none", None, baseline, baseline)]
        for threshold in thresholds:
            candidate = filter_paths(collect("marblenet", threshold), sample_paths)
            rows.append(_vad_row("marblenet", threshold, baseline, candidate))
        MODEL_CACHE.commit()
        return {
            "source": source,
            "thresholds": thresholds,
            "max_wer_delta": max_wer_delta,
            "max_deletion_wer": max_deletion_wer,
            "rows": rows,
            "recommended": _recommend_vad_threshold(rows, max_wer_delta, max_deletion_wer),
        }

    @modal.method()
    def profile(
        self,
        config: dict,
        source: str,
        trace_name: str = "transcribe_diarize_profile.json",
        limit: int = 0,
    ) -> dict:
        import os

        import torch

        from models.common.modal_infra import OUTPUT_DIR
        from pipelines.transcribe_diarize.pipeline import build_dataframe

        _reload_audio_volume_if_needed(source)

        df = build_dataframe(
            source,
            asr=config["asr"],
            diarizer=config["diarizer"],
            vad=config.get("vad", "none"),
            asr_model=config.get("asr_model", ""),
            diarizer_model=config.get("diarizer_model", ""),
            row_batch_size=config.get("row_batch_size", 16),
            asr_kwargs=config.get("asr_kwargs") or {},
        )
        if limit:
            df = df.limit(limit)

        output_dir = Path(os.environ.get("OUTPUT_DIR", OUTPUT_DIR))
        output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = output_dir / trace_name
        summary_path = trace_path.with_suffix(".summary.txt")
        activities = [torch.profiler.ProfilerActivity.CPU]
        sort_by = "cpu_time_total"
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
            sort_by = "cuda_time_total"

        started = time.perf_counter()
        with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
            result = df.collect()
        wall_seconds = time.perf_counter() - started
        prof.export_chrome_trace(str(trace_path))
        summary = prof.key_averages().table(sort_by=sort_by, row_limit=40)
        summary_path.write_text(summary, encoding="utf-8")
        MODEL_CACHE.commit()
        OUTPUTS.commit()
        data = result.to_pydict()
        return {
            "trace_path": str(trace_path),
            "summary_path": str(summary_path),
            "wall_seconds": wall_seconds,
            "rows": len(data.get("path", [])),
            "sort_by": sort_by,
        }


def _word_tokens(text: str) -> list[str]:
    import re

    return re.findall(r"\w+", (text or "").lower())


def _edit_counts(reference: list[str], hypothesis: list[str]) -> dict[str, int]:
    prev = [(index, 0, index, 0) for index in range(len(hypothesis) + 1)]
    for ref_index, ref_word in enumerate(reference, start=1):
        row = [(ref_index, ref_index, 0, 0)]
        for hyp_index, hyp_word in enumerate(hypothesis, start=1):
            if ref_word == hyp_word:
                row.append(prev[hyp_index - 1])
                continue
            sub = prev[hyp_index - 1]
            delete = prev[hyp_index]
            insert = row[hyp_index - 1]
            candidates = [
                (sub[0] + 1, sub[1], sub[2], sub[3] + 1),
                (delete[0] + 1, delete[1] + 1, delete[2], delete[3]),
                (insert[0] + 1, insert[1], insert[2] + 1, insert[3]),
            ]
            row.append(min(candidates, key=lambda item: item[0]))
        prev = row
    errors, deletions, insertions, substitutions = prev[-1]
    return {
        "ref_words": len(reference),
        "errors": errors,
        "deletions": deletions,
        "insertions": insertions,
        "substitutions": substitutions,
    }


def _vad_row(vad: str, threshold: float | None, baseline: dict, candidate: dict) -> dict:
    ref_by_path = dict(zip(baseline.get("path", []), baseline.get("transcript", []), strict=False))
    hyp_by_path = dict(zip(candidate.get("path", []), candidate.get("transcript", []), strict=False))
    counts = {"ref_words": 0, "errors": 0, "deletions": 0, "insertions": 0, "substitutions": 0}
    for path, reference in ref_by_path.items():
        path_counts = _edit_counts(_word_tokens(reference), _word_tokens(hyp_by_path.get(path, "")))
        for key, value in path_counts.items():
            counts[key] += value

    ref_words = max(1, counts["ref_words"])
    info = candidate.get("info", [])
    audio_seconds = float(sum((item or {}).get("duration", 0.0) for item in info))
    seconds_removed = float(sum(candidate.get("vad_seconds_removed", []) or []))
    return {
        "vad": vad,
        "threshold": threshold,
        "rows": len(candidate.get("path", [])),
        "audio_seconds": audio_seconds,
        "seconds_removed": seconds_removed,
        "removed_pct": (seconds_removed / audio_seconds * 100.0) if audio_seconds else 0.0,
        "wer_vs_none": counts["errors"] / ref_words,
        "deletion_wer_vs_none": counts["deletions"] / ref_words,
        "insertions": counts["insertions"],
        "substitutions": counts["substitutions"],
        "deletions": counts["deletions"],
        "ref_words": counts["ref_words"],
    }


def _recommend_vad_threshold(rows: list[dict], max_wer_delta: float, max_deletion_wer: float) -> dict:
    candidates = [row for row in rows if row.get("vad") != "none"]
    if not candidates:
        return {}
    eligible = [
        row
        for row in candidates
        if row["wer_vs_none"] <= max_wer_delta and row["deletion_wer_vs_none"] <= max_deletion_wer
    ]
    if eligible:
        chosen = max(
            eligible,
            key=lambda row: (
                row["seconds_removed"],
                -row["deletion_wer_vs_none"],
                -row["wer_vs_none"],
            ),
        )
        reason = f"largest seconds removed while WER <= {max_wer_delta:.3f} and deletion-WER <= {max_deletion_wer:.3f}"
    else:
        chosen = min(
            candidates,
            key=lambda row: (
                row["deletion_wer_vs_none"],
                row["wer_vs_none"],
                -row["seconds_removed"],
            ),
        )
        reason = "no threshold met tolerances; lowest deletion-WER, then WER, then most removed"
    return {**chosen, "reason": reason}


@app.function(
    image=whisper_image(),
    gpu=GPU,
    cpu=4,
    memory=32768,
    timeout=3600,
    region=MODAL_REGION,
    volumes=_VOLUMES,
    secrets=_SECRETS,
)
def run_whisper(config: dict, source: str, gpu: str) -> dict:
    return _measure(config, source, gpu)


def _run_nemo(config: dict, source: str, gpu: str) -> dict:
    return NemoPipelineService().measure.remote(config, source, gpu)


def _run_whisper(config: dict, source: str, gpu: str) -> dict:
    return run_whisper.remote(config, source, gpu)


_LANE_RUNNER = {"nemo": _run_nemo, "whisper": _run_whisper}


@app.function(
    image=whisper_image(),
    gpu="A100-80GB",  # headroom for multi-hour files
    cpu=4,
    memory=65536,
    timeout=10800,
    region=MODAL_REGION,
    volumes={MODEL_CACHE_DIR: MODEL_CACHE, AUDIO_DIR: audio_volume},
    secrets=_SECRETS,
)
def transcribe_folder(
    asr: str = "faster_whisper",
    asr_model: str = "large-v3",
    diarizer: str = "none",
    row_batch_size: int = 16,
    limit: int = 0,
) -> dict:
    """Transcribe every audio file uploaded to the audio Volume.

    Uses faster-whisper (CTranslate2) for robust real-world coverage: per-file
    automatic language detection (the folder is multilingual) and built-in VAD
    chunking of long audio (handles multi-hour files without OOM). Per-file fault
    isolation in the ASR means a single bad file can't sink the run. Diarization
    defaults off — pyannote is too costly on the multi-hour files here. ``limit``
    caps the file count for a quick smoke test.
    """
    from pipelines.transcribe_diarize.pipeline import build_dataframe

    audio_volume.reload()
    df = build_dataframe(
        FLAC_DIR,
        asr=asr,
        asr_model=asr_model,
        diarizer=diarizer,
        vad="builtin",
        row_batch_size=row_batch_size,
    )
    if limit:
        df = df.limit(limit)
    df = df.collect()

    MODEL_CACHE.commit()
    result = df.to_pydict()
    # Strip the volume mount prefix so downstream sees bare filenames.
    result["filename"] = [p.rsplit("/", 1)[-1] for p in result.get("path", [])]
    return result


@app.function(
    image=nemo_image(),
    gpu="A100-80GB",
    cpu=4,
    memory=65536,
    timeout=10800,
    region=MODAL_REGION,
    volumes={MODEL_CACHE_DIR: MODEL_CACHE, AUDIO_DIR: audio_volume},
    secrets=_SECRETS,
)
def diarize_folder(filenames: list[str], diarizer_model: str = "") -> dict:
    """Sortformer speaker diarization for specific FLAC files on the volume.

    Decoupled from transcription so it runs in the NeMo lane (Sortformer) while
    transcription runs in the Whisper lane. Returns one ``speaker_segments`` list
    per file; merge into transcripts client-side. Sortformer is per-row and
    fault-isolated, so a multi-hour file can't sink the batch (4-speaker max).
    """
    import daft
    from daft import col
    from daft.functions import audio_file
    from models.sortformer.model import SortformerDiarizer, attach_speakers

    audio_volume.reload()
    df = daft.from_pydict({"filename": filenames, "path": [f"{FLAC_DIR}/{name}" for name in filenames]})
    df = df.with_column("audio", audio_file(col("path")))
    # diarize() guards on a non-empty transcript; these files already have one.
    df = df.with_column("transcript", daft.lit("x"))
    processor = SortformerDiarizer(**({"model": diarizer_model} if diarizer_model else {}))
    df = attach_speakers(df, processor)
    out = df.select("filename", "speaker_segments").collect()

    MODEL_CACHE.commit()
    return out.to_pydict()


@app.local_entrypoint()
def diarize_transcripts(
    transcripts: str = ".context/audio_report/transcripts.json",
    diarizer_model: str = "",
):
    """Run Sortformer over the already-transcribed files and merge speaker turns
    into ``transcripts.json`` in place (no re-transcription)."""
    import json
    from pathlib import Path

    from models.common.speech import merge_speakers

    payload = json.loads(Path(transcripts).read_text(encoding="utf-8"))
    results = payload["results"]
    filenames = results.get("filename", [])
    if not filenames:
        print("no transcribed files to diarize")
        return

    print(f"diarizing {len(filenames)} transcribed files with Sortformer ...")
    diar = diarize_folder.remote(filenames=filenames, diarizer_model=diarizer_model)
    speakers_by_name = dict(zip(diar["filename"], diar["speaker_segments"], strict=False))

    merged_segments = []
    speaker_segments = []
    for index, name in enumerate(filenames):
        segs = results["segments"][index] or []
        turns = speakers_by_name.get(name) or []
        merged_segments.append(merge_speakers(segs, turns) if turns else segs)
        speaker_segments.append(turns)
    results["segments"] = merged_segments
    results["speaker_segments"] = speaker_segments
    payload["diarizer"] = diarizer_model or "nvidia/diar_streaming_sortformer_4spk-v2"

    Path(transcripts).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    diarized = sum(1 for t in speaker_segments if t)
    print(f"merged speaker turns into {diarized}/{len(filenames)} files -> {transcripts}")


@app.local_entrypoint()
def dump_transcripts(
    folder: str,
    out: str = ".context/audio_report/transcripts.json",
    asr_model: str = "large-v3",
    diarizer: str = "none",
    limit: int = 0,
    skip_upload: bool = False,
):
    """Upload a local audio folder, transcribe+diarize it, write transcripts.json.

    Same-stem duplicate formats (e.g. ``X.wav`` / ``X.mp3`` / ``X.m4a``) are
    collapsed to a single decode-safe copy so we don't transcribe a recording
    several times.
    """
    import json
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    src = Path(folder).expanduser()
    files = [p for p in src.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS]

    by_stem: dict[str, list[Path]] = {}
    for p in files:
        by_stem.setdefault(p.stem, []).append(p)
    chosen: list[Path] = []
    skipped: list[Path] = []
    for group in by_stem.values():
        if len(group) == 1:
            chosen.append(group[0])
            continue
        ranked = sorted(group, key=lambda p: DEDUPE_FORMAT_PREF.index(p.suffix.lower()))
        chosen.append(ranked[0])
        skipped.extend(ranked[1:])

    targets = chosen if not limit else chosen[:limit]

    staging = Path(out).expanduser().parent / "staging_flac"
    staging.mkdir(parents=True, exist_ok=True)

    def transcode(p: Path):
        """Source -> 16 kHz mono FLAC (cached). Returns (flac_path, original_name) or None."""
        target = staging / f"{p.stem}.flac"
        if not (target.exists() and target.stat().st_size > 0):
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-nostdin", "-i", str(p), "-ac", "1", "-ar", "16000", "-c:a", "flac", str(target)],
                    check=True,
                    capture_output=True,
                    timeout=3600,
                )
            except Exception as exc:  # noqa: BLE001 — skip undecodable/corrupt files
                print(f"  transcode FAILED: {p.name} ({type(exc).__name__})")
                return None
        if not (target.exists() and target.stat().st_size > 0):
            return None
        return (target, p.name)

    if skip_upload:
        ready = [(staging / f"{p.stem}.flac", p.name) for p in targets if (staging / f"{p.stem}.flac").exists()]
        print(f"skip_upload: reusing {len(ready)} transcoded files already on the volume")
    else:
        print(f"transcoding {len(targets)} files to 16kHz mono FLAC ({len(skipped)} dup-format skipped)...")
        with ThreadPoolExecutor(max_workers=6) as pool:
            ready = [r for r in pool.map(transcode, targets) if r]
        print(f"transcoded {len(ready)}/{len(chosen)}; uploading to volume {FLAC_DIR} ...")
        with audio_volume.batch_upload(force=True) as batch:
            for flac_path, _orig in ready:
                batch.put_file(str(flac_path), f"/flac/{flac_path.name}")

    result = transcribe_folder.remote(asr_model=asr_model, diarizer=diarizer, limit=limit)

    payload = {
        "asr_model": asr_model,
        "diarizer": diarizer,
        "uploaded": [flac.name for flac, _ in ready],
        "name_map": {flac.name: orig for flac, orig in ready},
        "skipped_dupes": [p.name for p in skipped],
        "results": result,
    }
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    n = len(result.get("filename", []))
    print(f"wrote {n} transcripts ({len(chosen)} uploaded) -> {out_path}")


@app.function(image=nemo_image(), cpu=2, timeout=1800, region=MODAL_REGION, volumes=_VOLUMES, secrets=_SECRETS)
def prewarm_nemo(models: list[str]) -> dict:
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.models import SortformerEncLabelModel

    for model in models:
        if "sortformer" in model or "diar" in model:
            SortformerEncLabelModel.from_pretrained(model)
        else:
            nemo_asr.models.ASRModel.from_pretrained(model_name=model)
    MODEL_CACHE.commit()
    return {"prewarmed": models}


@app.function(image=whisper_image(), cpu=2, timeout=1800, region=MODAL_REGION, volumes=_VOLUMES, secrets=_SECRETS)
def prewarm_whisper(asr_model: str = "large-v3", diarizer: str = "pyannote/speaker-diarization-3.1") -> dict:
    import os

    from faster_whisper import WhisperModel

    WhisperModel(asr_model, device="cpu", compute_type="int8")
    if os.environ.get("HF_TOKEN"):
        from pyannote.audio import Pipeline

        Pipeline.from_pretrained(diarizer, token=os.environ["HF_TOKEN"])
    MODEL_CACHE.commit()
    return {"prewarmed": [asr_model, diarizer]}


@app.local_entrypoint()
def run(source: str = SOURCE, gpu: str = GPU, output: str = ""):
    """Run the single fastest config (Parakeet + Sortformer) and print metrics."""
    fastest = next(c for c in SWAP_MATRIX if c.name == "parakeet+sortformer")
    result = NemoPipelineService().measure.remote(dataclasses.asdict(fastest), source, gpu)
    print(result)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(str(result), encoding="utf-8")


def _config_by_name(name: str) -> dict:
    try:
        return dataclasses.asdict(next(c for c in SWAP_MATRIX if c.name == name))
    except StopIteration as exc:
        choices = ", ".join(c.name for c in SWAP_MATRIX)
        raise ValueError(f"unknown config '{name}'; choices: {choices}") from exc


def _parse_thresholds(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _print_vad_rows(rows: list[dict]) -> None:
    print("| vad | threshold | rows | removed | WER vs none | deletion WER |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        threshold = "-" if row["threshold"] is None else f"{row['threshold']:.2f}"
        print(
            f"| {row['vad']} | {threshold} | {row['rows']} | "
            f"{row['seconds_removed']:.1f}s ({row['removed_pct']:.1f}%) | "
            f"{row['wer_vs_none']:.3f} | {row['deletion_wer_vs_none']:.3f} |"
        )


def _print_vad_recommendation(recommended: dict) -> None:
    if not recommended:
        return
    threshold = recommended.get("threshold")
    threshold_text = "-" if threshold is None else f"{threshold:.2f}"
    print(
        f"\nrecommended MarbleNet threshold: {threshold_text} "
        f"({recommended['seconds_removed']:.1f}s removed, "
        f"WER delta {recommended['wer_vs_none']:.3f}, "
        f"deletion-WER {recommended['deletion_wer_vs_none']:.3f})"
    )
    print(f"reason: {recommended['reason']}")


@app.local_entrypoint()
def bench_vad(
    source: str = SOURCE,
    thresholds: str = "0.5,0.7,0.8",
    asr_model: str = "",
    gpu: str = GPU,
    limit: int = 0,
    max_wer_delta: float = 0.05,
    max_deletion_wer: float = 0.02,
    output: str = ".context/transcribe_diarize/VAD_BENCH.json",
):
    """Compare no VAD against MarbleNet thresholds using transcript deltas."""
    del gpu  # GPU is selected at deploy time via BENCH_GPU, kept for CLI symmetry.
    result = NemoPipelineService().bench_vad.remote(
        source=source,
        thresholds=_parse_thresholds(thresholds),
        asr_model=asr_model,
        limit=limit,
        max_wer_delta=max_wer_delta,
        max_deletion_wer=max_deletion_wer,
    )
    _print_vad_rows(result["rows"])
    _print_vad_recommendation(result.get("recommended") or {})
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {out_path}")


@app.local_entrypoint()
def profile(
    source: str = SOURCE,
    config: str = "parakeet+sortformer",
    trace_name: str = "transcribe_diarize_profile.json",
    limit: int = 0,
):
    """Collect a torch.profiler Chrome trace for one NeMo pipeline config."""
    result = NemoPipelineService().profile.remote(
        config=_config_by_name(config),
        source=source,
        trace_name=trace_name,
        limit=limit,
    )
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def measure(source: str = SOURCE, gpu: str = GPU, config: str = "parakeet+sortformer"):
    """Measure snapshot service cold/warm walls around a NeMo config."""
    cfg = _config_by_name(config)
    service = NemoPipelineService()

    started = time.perf_counter()
    first = service.measure.remote(cfg, source, gpu)
    first_wall = time.perf_counter() - started
    print(f"first service call wall: {first_wall:.1f}s")
    print(f"  measured collect wall: {first.get('wall_seconds', 0.0):.1f}s")
    print(f"  error: {first['error']}" if first.get("error") else f"  rows/audio: {first.get('extra', {})}")

    status = service.status.remote()
    print(f"snapshot load status: {status}")

    started = time.perf_counter()
    second = service.measure.remote(cfg, source, gpu)
    second_wall = time.perf_counter() - started
    print(f"second service call wall: {second_wall:.1f}s")
    print(f"  measured collect wall: {second.get('wall_seconds', 0.0):.1f}s")
    print("\nFor true cold-restore timing, redeploy or force a fresh container between invocations.")


@app.local_entrypoint()
def benchmark(
    source: str = SOURCE,
    gpu: str = GPU,
    configs: str = "",
    output: str = ".context/transcribe_diarize/RESULTS_measured.md",
):
    """Run the swap matrix and write a $/audio-hour leaderboard.

    --configs is an optional comma-separated subset of config names.
    """
    from models.common.metrics import leaderboard_markdown

    selected = SWAP_MATRIX
    if configs:
        wanted = {name.strip() for name in configs.split(",")}
        selected = [c for c in SWAP_MATRIX if c.name in wanted]

    raw_rows: list[dict] = []
    for lane, lane_configs in matrix_by_lane(selected).items():
        runner = _LANE_RUNNER[lane]
        for config in lane_configs:
            print(f"running [{lane}] {config.name} on {gpu} ...")
            raw_rows.append(runner(dataclasses.asdict(config), source, gpu))

    rows = [_row_from_dict(raw) for raw in raw_rows]
    board = leaderboard_markdown(rows)
    print("\n" + board)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(f"# Swap-matrix results ({gpu})\n\nSource: `{source}`\n\n{board}\n", encoding="utf-8")
    print(f"\nwrote {output}")


def _row_from_dict(raw: dict):
    from models.common.metrics import BenchmarkRow

    row = BenchmarkRow(
        config=raw["config"],
        lane=raw["lane"],
        asr=raw["asr"],
        vad=raw["vad"],
        diarizer=raw["diarizer"],
        gpu=raw["gpu"],
        audio_seconds=raw.get("audio_seconds", 0.0),
        wall_seconds=raw.get("wall_seconds", 0.0),
        peak_vram_gb=raw.get("peak_vram_gb"),
        wer=raw.get("wer"),
        der=raw.get("der"),
        vad_seconds_removed=raw.get("vad_seconds_removed"),
        error=raw.get("error", ""),
    )
    return row
