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

from models.common.modal_images import GPU_TYPE, MODAL_REGION, nemo_image, whisper_image
from models.common.modal_infra import MODEL_CACHE_DIR
from models.weights import HF_SECRET, MODEL_CACHE, VOLUMES
from pipelines.transcribe_diarize.benchmark import SWAP_MATRIX, matrix_by_lane

# GPU is fixed per deploy; sweep it by re-running with BENCH_GPU set.
GPU = os.environ.get("BENCH_GPU", GPU_TYPE)
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


@app.function(
    image=nemo_image(),
    gpu=GPU,
    cpu=4,
    memory=32768,
    timeout=3600,
    region=MODAL_REGION,
    volumes=_VOLUMES,
    secrets=_SECRETS,
)
def run_nemo(config: dict, source: str, gpu: str) -> dict:
    return _measure(config, source, gpu)


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


_LANE_FN = {"nemo": run_nemo, "whisper": run_whisper}


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
    row_batch_size: int = 1,
    limit: int = 0,
) -> dict:
    """Transcribe every audio file uploaded to the audio Volume.

    Uses faster-whisper (CTranslate2) for robust real-world coverage: per-file
    automatic language detection (the folder is multilingual) and built-in VAD
    chunking of long audio (handles multi-hour files without OOM). One file per
    Daft batch (``row_batch_size=1``) plus per-file fault isolation in the ASR
    means a single bad file can't sink the run. Diarization defaults off — pyannote
    (the whisper-lane diarizer) is too costly on the multi-hour files here.
    ``limit`` caps the file count for a quick smoke test.
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
    from models.sortformer.model import DEFAULT_MODEL, SortformerDiarizer, attach_speakers

    audio_volume.reload()
    df = daft.from_pydict({"filename": filenames, "path": [f"{FLAC_DIR}/{name}" for name in filenames]})
    df = df.with_column("audio", audio_file(col("path")))
    # diarize() guards on a non-empty transcript; these files already have one.
    df = df.with_column("transcript", daft.lit("x"))
    processor = SortformerDiarizer(model=diarizer_model or DEFAULT_MODEL)
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
    result = run_nemo.remote(dataclasses.asdict(fastest), source, gpu)
    print(result)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(str(result), encoding="utf-8")


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
        fn = _LANE_FN[lane]
        for config in lane_configs:
            print(f"running [{lane}] {config.name} on {gpu} ...")
            raw_rows.append(fn.remote(dataclasses.asdict(config), source, gpu))

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
