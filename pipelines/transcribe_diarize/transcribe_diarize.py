# /// script
# description = "Transcribe + speaker diarization with Faster Whisper and pyannote.audio"
# requires-python = ">=3.12, <3.13"
# dependencies = [
#     "daft>=0.7.8",
#     "faster-whisper",
#     "pyannote.audio",
#     "python-dotenv",
#     "torch",
#     "torchaudio",
# ]
# ///
"""
Transcription + Speaker Diarization Pipeline
=============================================
Transcribes audio with faster-whisper, diarizes speakers with pyannote.audio,
and merges the results so every word and segment is attributed to a speaker.

Architecture:
  Phase 1 — Transcribe (faster-whisper, CPU)  → transcription.parquet
  Phase 2 — Diarize   (pyannote 3.1, MPS/GPU) → diarization.parquet
  Phase 3 — Merge     (timestamp overlap)      → final.parquet
  Phase 4 — Report    (HTML generation)        → report.html

Phases run sequentially so models don't compete for memory.
Intermediate parquets are cached — rerun skips completed phases.

Audio is converted to 16 kHz mono WAV before processing (pyannote
requires exact sample counts that lossy formats like MP3/M4A can't guarantee).

Requirements:
  - HF_TOKEN env var (pyannote requires Hugging Face authentication)
  - ffmpeg on PATH (for non-WAV audio conversion)

Usage:
  uv run transcribe_diarize.py --source /path/to/audio.m4a
  uv run transcribe_diarize.py --source recording.wav --dest .data/output
  uv run transcribe_diarize.py --source meeting.m4a --no-cache
"""

import argparse
import gc
import os
import subprocess
import time
from collections import Counter
from dataclasses import asdict

from dotenv import load_dotenv

DEST_DIR = ".data/transcribe_diarize"
BATCH_SIZE = 16
PALETTE = [
    "#3b82f6",
    "#10b981",
    "#f59e0b",
    "#ef4444",
    "#8b5cf6",
    "#ec4899",
    "#06b6d4",
    "#84cc16",
]


# ── Audio Preprocessing ─────────────────────────────────────────────────────


def ensure_wav(source: str, dest_dir: str) -> str:
    """Convert audio to 16 kHz mono WAV if not already WAV."""
    basename = os.path.splitext(os.path.basename(source))[0]
    wav_path = os.path.join(dest_dir, f"{basename}.wav")

    if os.path.exists(wav_path):
        print(f"  Using cached WAV: {wav_path}")
        return wav_path

    if source.lower().endswith(".wav"):
        return source

    print(f"  Converting to WAV: {source}")
    t0 = time.perf_counter()
    subprocess.run(
        ["ffmpeg", "-i", source, "-ar", "16000", "-ac", "1", "-y", wav_path],
        capture_output=True,
        check=True,
    )
    print(f"  Converted in {time.perf_counter() - t0:.1f}s")
    return wav_path


# ── Phase 1: Transcription ──────────────────────────────────────────────────
#
# Imports are deferred so the Whisper model is only loaded during this phase,
# freeing memory before diarization starts.


def run_transcription(wav_path: str, dest_dir: str, batch_size: int) -> str:
    """Transcribe audio with faster-whisper. Returns path to parquet."""
    from diarize_schema import DiarizedSegmentStruct, InfoStruct
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    import daft
    from daft import DataType, col
    from daft.functions import file, unnest

    parquet_path = os.path.join(dest_dir, "transcription.parquet")

    TranscriptionResult = DataType.struct(
        {
            "transcript": DataType.string(),
            "segments": DataType.list(DiarizedSegmentStruct),
            "info": InfoStruct,
        }
    )

    @daft.cls()
    class Transcriber:
        def __init__(self):
            self.model = WhisperModel("distil-large-v3", compute_type="float32", device="cpu")
            self.pipe = BatchedInferencePipeline(self.model)

        @daft.method(return_dtype=TranscriptionResult)
        def transcribe(self, audio_file: daft.File):
            with audio_file.to_tempfile() as tmp:
                segments_iter, info = self.pipe.transcribe(
                    str(tmp.name),
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                    word_timestamps=True,
                    batch_size=batch_size,
                )
                segments = [asdict(seg) for seg in segments_iter]
                for seg in segments:
                    seg["speaker"] = None
                    for word in seg.get("words") or []:
                        word["speaker"] = None
                return {
                    "transcript": " ".join(seg["text"] for seg in segments),
                    "segments": segments,
                    "info": asdict(info),
                }

    t0 = time.perf_counter()
    transcriber = Transcriber()

    df = (
        daft.from_pydict({"path": [wav_path]})
        .with_column("audio_file", file(col("path")))
        .with_column("result", transcriber.transcribe(col("audio_file")))
        .select("path", unnest(col("result")))
    )
    df.write_parquet(parquet_path, write_mode="overwrite")
    print(f"  Transcription: {time.perf_counter() - t0:.1f}s")

    del transcriber
    gc.collect()

    return parquet_path


# ── Phase 2: Diarization ────────────────────────────────────────────────────


def run_diarization(wav_path: str, dest_dir: str, hf_token: str) -> str:
    """Diarize speakers with pyannote. Returns path to parquet."""
    import torch
    from diarize_schema import SpeakerSegmentStruct
    from pyannote.audio import Pipeline

    import daft
    from daft import DataType, col
    from daft.functions import file

    parquet_path = os.path.join(dest_dir, "diarization.parquet")

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    DiarizationResult = DataType.list(SpeakerSegmentStruct)

    @daft.cls()
    class Diarizer:
        def __init__(self):
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token,
            )
            self.pipeline.to(torch.device(device))

        @daft.method(return_dtype=DiarizationResult)
        def diarize(self, audio_file: daft.File):
            with audio_file.to_tempfile() as tmp:
                result = self.pipeline(str(tmp.name))
                ann = result.speaker_diarization if hasattr(result, "speaker_diarization") else result
                return [
                    {"start": turn.start, "end": turn.end, "speaker": speaker}
                    for turn, _, speaker in ann.itertracks(yield_label=True)
                ]

    t0 = time.perf_counter()
    diarizer = Diarizer()

    df = (
        daft.from_pydict({"path": [wav_path]})
        .with_column("audio_file", file(col("path")))
        .with_column("speaker_segments", diarizer.diarize(col("audio_file")))
    )
    df.write_parquet(parquet_path, write_mode="overwrite")
    print(f"  Diarization ({device}): {time.perf_counter() - t0:.1f}s")

    del diarizer
    gc.collect()

    return parquet_path


# ── Phase 3: Merge ───────────────────────────────────────────────────────────


def find_speaker(start: float, end: float, speaker_segments: list[dict]) -> str | None:
    """Find the speaker with maximum time overlap for the given interval."""
    best, best_ov = None, 0.0
    for ss in speaker_segments:
        ov = min(end, ss["end"]) - max(start, ss["start"])
        if ov > best_ov:
            best_ov = ov
            best = ss["speaker"]
    return best


def run_merge(dest_dir: str) -> str:
    """Merge transcription + diarization by timestamp overlap. Returns path to parquet."""
    import daft
    from daft import col
    from daft.functions import unnest

    tx_path = os.path.join(dest_dir, "transcription.parquet")
    dz_path = os.path.join(dest_dir, "diarization.parquet")
    final_path = os.path.join(dest_dir, "final.parquet")

    t0 = time.perf_counter()

    tx_rows = daft.read_parquet(tx_path).collect().to_pydict()
    dz_rows = daft.read_parquet(dz_path).collect().to_pydict()

    merged_rows: dict[str, list] = {
        "path": [],
        "transcript": [],
        "segments": [],
        "info": [],
        "speaker_segments": [],
    }
    for i in range(len(tx_rows["path"])):
        segments = tx_rows["segments"][i]
        speaker_segs = dz_rows["speaker_segments"][i]

        for seg in segments:
            if seg.get("words"):
                for word in seg["words"]:
                    word["speaker"] = find_speaker(word["start"], word["end"], speaker_segs)
                speakers = [w["speaker"] for w in seg["words"] if w.get("speaker")]
                seg["speaker"] = Counter(speakers).most_common(1)[0][0] if speakers else None
            else:
                seg["speaker"] = find_speaker(seg["start"], seg["end"], speaker_segs)

        merged_rows["path"].append(tx_rows["path"][i])
        merged_rows["transcript"].append(tx_rows["transcript"][i])
        merged_rows["segments"].append(segments)
        merged_rows["info"].append(tx_rows["info"][i])
        merged_rows["speaker_segments"].append(speaker_segs)

    daft.from_pydict(merged_rows).write_parquet(final_path, write_mode="overwrite")
    print(f"  Merge: {time.perf_counter() - t0:.1f}s")

    df_preview = (
        daft.read_parquet(final_path)
        .select("segments")
        .explode("segments")
        .select(unnest(col("segments")))
        .select("speaker", "start", "end", "text")
    )
    df_preview.show()

    return final_path


# ── Phase 4: HTML Report ────────────────────────────────────────────────────


def format_timestamp(seconds: float) -> str:
    """Format seconds as H:MM:SS or M:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def generate_report(dest_dir: str, source_path: str) -> str:
    """Generate HTML report with speaker summaries and interactive renaming."""
    import json as _json

    import daft

    t0 = time.perf_counter()
    final_path = os.path.join(dest_dir, "final.parquet")
    report_path = os.path.join(dest_dir, "report.html")

    data = daft.read_parquet(final_path).collect().to_pydict()
    segments = data["segments"][0]
    info = data["info"][0]

    speaker_durations: Counter[str] = Counter()
    speaker_words: Counter[str] = Counter()
    for seg in segments:
        dur = seg["end"] - seg["start"]
        speaker_durations[seg["speaker"]] += dur
        speaker_words[seg["speaker"]] += len((seg.get("text") or "").split())

    total_speech = sum(speaker_durations.values())
    ranked_speakers = sorted(speaker_durations.items(), key=lambda x: -x[1])

    speaker_colors = {sp: PALETTE[i] if i < len(PALETTE) else "#6b7280" for i, (sp, _) in enumerate(ranked_speakers)}

    turns: list[dict] = []
    for seg in segments:
        if turns and turns[-1]["speaker"] == seg["speaker"]:
            turns[-1]["end"] = seg["end"]
            turns[-1]["text"] += " " + (seg.get("text") or "").strip()
        else:
            turns.append(
                {
                    "speaker": seg["speaker"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": (seg.get("text") or "").strip(),
                }
            )

    source_name = os.path.basename(source_path)

    speaker_table_rows = "\n".join(
        f"      <tr>"
        f'<td><span class="swatch" style="background:{speaker_colors[sp]}"></span>'
        f'<input class="rename" data-speaker-id="{sp}" value="{sp}" '
        f'style="color:{speaker_colors[sp]};font-weight:bold"></td>'
        f"<td>{format_timestamp(dur)}</td>"
        f"<td>{dur / total_speech * 100:.1f}%</td>"
        f"<td>{speaker_words[sp]:,}</td></tr>"
        for sp, dur in ranked_speakers
    )

    turn_blocks = "\n".join(
        f'    <div class="turn">\n'
        f'      <h3><strong class="speaker-label" data-speaker-id="{t["speaker"]}" '
        f'style="color:{speaker_colors[t["speaker"]]}">{t["speaker"]}</strong> '
        f"<code>[{format_timestamp(t['start'])} - {format_timestamp(t['end'])}]</code></h3>\n"
        f"      <p>{t['text']}</p>\n"
        f"    </div>"
        for t in turns
    )

    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_template.html")
    with open(template_path) as f:
        template = f.read()

    substitutions = {
        "source_name": source_name,
        "source_name_json": _json.dumps(source_name),
        "storage_key_json": _json.dumps(f"diar-renames::{source_name}"),
        "duration": format_timestamp(info["duration"]),
        "language": info["language"],
        "segments_count": f"{len(segments):,}",
        "speakers_count": str(len(speaker_durations)),
        "speaker_table_rows": speaker_table_rows,
        "turn_blocks": turn_blocks,
    }
    html = template
    for key, value in substitutions.items():
        html = html.replace(f"{{{{{key}}}}}", str(value))

    with open(report_path, "w") as f:
        f.write(html)

    elapsed = time.perf_counter() - t0
    print(f"  Report: {elapsed:.1f}s -> {report_path}")
    print(f"  Open with: open {report_path}")
    return report_path


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe + diarize audio",
        epilog="Requires HF_TOKEN env var and ffmpeg on PATH.",
    )
    parser.add_argument("--source", required=True, help="Path to audio file (WAV, M4A, MP3, etc.)")
    parser.add_argument("--dest", default=DEST_DIR, help="Output directory (default: %(default)s)")
    parser.add_argument("--no-cache", action="store_true", help="Force re-run all phases")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise OSError("HF_TOKEN not set. Add it to .env or export it.")

    os.makedirs(args.dest, exist_ok=True)
    t_total = time.perf_counter()

    print("\n[1/5] Preprocessing")
    wav_path = ensure_wav(args.source, args.dest)

    tx_parquet = os.path.join(args.dest, "transcription.parquet")
    if not args.no_cache and os.path.exists(tx_parquet):
        print("\n[2/5] Transcription (cached)")
    else:
        print("\n[2/5] Transcription")
        run_transcription(wav_path, args.dest, BATCH_SIZE)

    dz_parquet = os.path.join(args.dest, "diarization.parquet")
    if not args.no_cache and os.path.exists(dz_parquet):
        print("\n[3/5] Diarization (cached)")
    else:
        print("\n[3/5] Diarization")
        run_diarization(wav_path, args.dest, hf_token)

    print("\n[4/5] Merge")
    run_merge(args.dest)

    print("\n[5/5] Report")
    generate_report(args.dest, args.source)

    elapsed = time.perf_counter() - t_total
    print(f"\nDone in {elapsed:.1f}s. Results at {args.dest}/final.parquet")


if __name__ == "__main__":
    load_dotenv()
    main()
