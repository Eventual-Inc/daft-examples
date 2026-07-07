# /// script
# description = "LLM-judge quality pass over transcribe+diarize output, rendered to a self-contained HTML report"
# requires-python = ">=3.11"
# dependencies = ["openai>=1.0", "python-dotenv"]
# ///
"""Turn ``transcripts.json`` (from modal_app.py::dump_transcripts) into an HTML
report with an LLM-judge quality pass.

The judge (OpenRouter, matching the repo's voice-analytics provider) scores each
transcript for content type, language, transcription quality, and diarization
plausibility. Output is one self-contained HTML file — speaker-colored segments,
per-file judge verdicts, and a summary table.

    uv run pipelines/transcribe_diarize/build_report.py \
      --in .context/audio_report/transcripts.json \
      --out .context/audio_report/transcript_report.html
"""

from __future__ import annotations

import argparse
import html
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

JUDGE_MODEL = "openai/gpt-oss-120b"
TRANSCRIPT_CHAR_LIMIT = 6000
SPEAKER_PALETTE = [
    "#2563eb",
    "#dc2626",
    "#059669",
    "#d97706",
    "#7c3aed",
    "#db2777",
    "#0891b2",
    "#65a30d",
    "#ea580c",
    "#4f46e5",
]
QUALITY_COLORS = {5: "#059669", 4: "#65a30d", 3: "#d97706", 2: "#ea580c", 1: "#dc2626", 0: "#6b7280"}

JUDGE_SYSTEM = (
    "You are a meticulous speech-transcription quality auditor. You are given an "
    "automatic transcript (Parakeet ASR + Sortformer diarization) of an audio file, "
    "plus metadata. Judge the transcript itself — fluency, coherence, whether it "
    "plausibly reflects real speech — not the audio you cannot hear. Many files are "
    "not speech (music, singing, ambient noise); flag those. Respond with a single "
    "JSON object and nothing else."
)

JUDGE_SCHEMA_HINT = (
    "Return JSON with exactly these keys:\n"
    '  "content_type": one of "speech","conversation","monologue","music","singing","noise","silence","mixed"\n'
    '  "language": the dominant language name, or "unknown"\n'
    '  "quality_score": integer 1-5 (5=clean fluent transcript, 1=garbled/hallucinated/unusable)\n'
    '  "quality_label": one of "excellent","good","fair","poor","unusable"\n'
    '  "diarization_plausible": true/false/null (null if not applicable, e.g. non-speech)\n'
    '  "summary": one concise sentence describing the content\n'
    '  "issues": array of short strings naming concrete problems (empty array if none)'
)


def fmt_time(seconds: float) -> str:
    seconds = int(seconds or 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fmt_duration(seconds: float) -> str:
    seconds = float(seconds or 0)
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    if seconds >= 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.0f} s"


def rows_from_results(results: dict) -> list[dict]:
    """Columnar pydict -> list of per-file row dicts."""
    n = len(results.get("filename", []))
    keys = ["filename", "transcript", "segments", "speaker_segments", "info", "size"]
    rows = []
    for i in range(n):
        rows.append({k: results.get(k, [None] * n)[i] for k in keys})
    return rows


def judge_one(client: OpenAI, row: dict) -> dict:
    transcript = (row.get("transcript") or "").strip()
    truncated = len(transcript) > TRANSCRIPT_CHAR_LIMIT
    info = row.get("info") or {}
    segments = row.get("segments") or []
    speakers = sorted({(s or {}).get("speaker") or "" for s in segments} - {""})
    user = (
        f"File: {row.get('filename')}\n"
        f"Duration: {fmt_duration(info.get('duration'))}\n"
        f"Transcript segments: {len(segments)} | distinct diarized speakers: {len(speakers)}\n\n"
        f"{JUDGE_SCHEMA_HINT}\n\n"
        f'Transcript{" (truncated)" if truncated else ""}:\n"""\n'
        f'{transcript[:TRANSCRIPT_CHAR_LIMIT]}\n"""'
    )
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        verdict = json.loads(resp.choices[0].message.content)
    except Exception as exc:  # noqa: BLE001 — record judge failure inline
        verdict = {
            "content_type": "unknown",
            "language": "unknown",
            "quality_score": 0,
            "quality_label": "judge-error",
            "diarization_plausible": None,
            "summary": f"LLM judge failed: {type(exc).__name__}: {exc}",
            "issues": [],
        }
    verdict["_speakers"] = speakers
    return verdict


def run_judge(rows: list[dict]) -> list[dict]:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set (expected in .env)")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(lambda r: judge_one(client, r), rows))


# --------------------------------------------------------------------------- HTML


def esc(text) -> str:
    return html.escape(str(text if text is not None else ""))


def speaker_color(speaker: str, mapping: dict) -> str:
    if speaker not in mapping:
        mapping[speaker] = SPEAKER_PALETTE[len(mapping) % len(SPEAKER_PALETTE)]
    return mapping[speaker]


def quality_badge(score: int, label: str) -> str:
    color = QUALITY_COLORS.get(int(score or 0), "#6b7280")
    return f'<span class="badge" style="background:{color}">{esc(label)} · {esc(score)}/5</span>'


def render_segments(segments: list[dict]) -> str:
    if not segments:
        return '<p class="muted">No speech segments.</p>'
    mapping: dict[str, str] = {}
    lines = []
    for seg in segments:
        seg = seg or {}
        spk = seg.get("speaker") or ""
        color = speaker_color(spk, mapping) if spk else "#9ca3af"
        label = esc(spk) if spk else "—"
        lines.append(
            '<div class="seg">'
            f'<span class="ts">{fmt_time(seg.get("start"))}</span>'
            f'<span class="spk" style="color:{color};border-color:{color}">{label}</span>'
            f'<span class="txt">{esc(seg.get("text"))}</span>'
            "</div>"
        )
    return "\n".join(lines)


def render_report(payload: dict, verdicts: list[dict], rows: list[dict]) -> str:
    results = payload.get("results", {})
    uploaded = payload.get("uploaded", [])
    skipped = payload.get("skipped_dupes", [])
    name_map = payload.get("name_map", {})

    def display(fname):
        return name_map.get(fname, fname)

    transcribed = set(results.get("filename", []))
    no_transcript = [display(f) for f in uploaded if f not in transcribed]

    total_dur = sum(float((r.get("info") or {}).get("duration") or 0) for r in rows)
    scored = [v.get("quality_score", 0) for v in verdicts if v.get("quality_score")]
    avg_q = sum(scored) / len(scored) if scored else 0
    type_counts: dict[str, int] = {}
    for v in verdicts:
        type_counts[v.get("content_type", "unknown")] = type_counts.get(v.get("content_type", "unknown"), 0) + 1

    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    # summary cards
    cards = [
        ("Files transcribed", f"{len(rows)}"),
        ("No transcript", f"{len(no_transcript)}"),
        ("Total audio", fmt_duration(total_dur)),
        ("Avg quality", f"{avg_q:.1f}/5"),
        ("Dup formats skipped", f"{len(skipped)}"),
    ]
    cards_html = "".join(
        f'<div class="card"><div class="card-v">{esc(v)}</div><div class="card-k">{esc(k)}</div></div>'
        for k, v in cards
    )
    type_html = " ".join(f'<span class="chip">{esc(t)}: {esc(c)}</span>' for t, c in sorted(type_counts.items()))

    # summary table + detail cards
    table_rows, detail_cards = [], []
    order = sorted(range(len(rows)), key=lambda i: verdicts[i].get("quality_score", 0))
    for rank, i in enumerate(order):
        row, v = rows[i], verdicts[i]
        info = row.get("info") or {}
        anchor = f"file-{rank}"
        table_rows.append(
            "<tr>"
            f'<td><a href="#{anchor}">{esc(display(row.get("filename")))}</a></td>'
            f'<td class="num">{esc(fmt_duration(info.get("duration")))}</td>'
            f"<td>{esc(v.get('language'))}</td>"
            f"<td>{esc(v.get('content_type'))}</td>"
            f"<td>{esc(len(v.get('_speakers', [])))}</td>"
            f"<td>{quality_badge(v.get('quality_score', 0), v.get('quality_label', '?'))}</td>"
            f'<td class="summary">{esc(v.get("summary"))}</td>'
            "</tr>"
        )
        issues = v.get("issues") or []
        issues_html = (
            "<ul class='issues'>" + "".join(f"<li>{esc(x)}</li>" for x in issues) + "</ul>"
            if issues
            else '<span class="muted">none flagged</span>'
        )
        diar = v.get("diarization_plausible")
        diar_txt = {True: "plausible", False: "questionable", None: "n/a"}.get(diar, esc(diar))
        detail_cards.append(
            f'<section class="detail" id="{anchor}">'
            f'<div class="detail-head"><h3>{esc(display(row.get("filename")))}</h3>'
            f"{quality_badge(v.get('quality_score', 0), v.get('quality_label', '?'))}</div>"
            '<div class="meta">'
            f"<span>{esc(fmt_duration(info.get('duration')))}</span>"
            f"<span>lang: {esc(v.get('language'))}</span>"
            f"<span>type: {esc(v.get('content_type'))}</span>"
            f"<span>speakers: {esc(len(v.get('_speakers', [])))}</span>"
            f"<span>diarization: {diar_txt}</span>"
            "</div>"
            f'<div class="verdict"><strong>Judge:</strong> {esc(v.get("summary"))}'
            f'<div class="issues-wrap"><strong>Issues:</strong> {issues_html}</div></div>'
            f'<div class="transcript">{render_segments(row.get("segments"))}</div>'
            "</section>"
        )

    no_tx_html = ""
    if no_transcript:
        items = "".join(f"<li>{esc(f)}</li>" for f in no_transcript)
        no_tx_html = (
            '<section class="notx"><h2>No transcript produced</h2>'
            "<p class='muted'>Uploaded but yielded no speech segments — typically music, "
            "singing, silence, or non-speech audio.</p>"
            f"<ul>{items}</ul></section>"
        )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Transcript Quality Report</title>
<style>
:root {{ color-scheme: light; }}
* {{ box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  margin: 0; background: #f4f5f7; color: #1f2937; line-height: 1.5; }}
header {{ background: #111827; color: #fff; padding: 28px 32px; }}
header h1 {{ margin: 0 0 4px; font-size: 22px; }}
header .sub {{ color: #9ca3af; font-size: 13px; }}
.wrap {{ max-width: 1100px; margin: 0 auto; padding: 24px 32px 80px; }}
.cards {{ display: flex; gap: 14px; flex-wrap: wrap; margin: 8px 0 18px; }}
.card {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px 18px; min-width: 130px; }}
.card-v {{ font-size: 24px; font-weight: 700; }}
.card-k {{ font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: .04em; }}
.chip {{ display: inline-block; background: #eef2ff; color: #3730a3; border-radius: 999px;
  padding: 3px 10px; font-size: 12px; margin: 2px; }}
table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 10px; overflow: hidden;
  box-shadow: 0 1px 2px rgba(0,0,0,.05); font-size: 13px; }}
th, td {{ text-align: left; padding: 9px 12px; border-bottom: 1px solid #f0f1f3; vertical-align: top; }}
th {{ background: #f9fafb; font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: #6b7280; }}
td.num {{ white-space: nowrap; }} td.summary {{ color: #4b5563; }}
a {{ color: #2563eb; text-decoration: none; }} a:hover {{ text-decoration: underline; }}
.badge {{ color: #fff; border-radius: 999px; padding: 2px 9px; font-size: 11px; font-weight: 600; white-space: nowrap; }}
.muted {{ color: #9ca3af; font-size: 13px; }}
h2 {{ margin: 30px 0 12px; font-size: 16px; }}
.detail {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 18px 20px; margin: 16px 0; }}
.detail-head {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; }}
.detail-head h3 {{ margin: 0; font-size: 15px; word-break: break-word; }}
.meta {{ display: flex; flex-wrap: wrap; gap: 14px; margin: 8px 0 12px; font-size: 12px; color: #6b7280; }}
.verdict {{ background: #f9fafb; border-left: 3px solid #6366f1; padding: 10px 14px; border-radius: 6px;
  font-size: 13px; margin-bottom: 12px; }}
.issues-wrap {{ margin-top: 6px; }}
ul.issues {{ margin: 4px 0 0; padding-left: 18px; }} ul.issues li {{ color: #b91c1c; font-size: 12px; }}
.transcript {{ max-height: 360px; overflow-y: auto; border: 1px solid #f0f1f3; border-radius: 8px; padding: 8px 10px; }}
.seg {{ display: flex; gap: 10px; padding: 3px 0; font-size: 13px; align-items: baseline; }}
.seg .ts {{ color: #9ca3af; font-variant-numeric: tabular-nums; font-size: 11px; min-width: 52px; }}
.seg .spk {{ font-size: 11px; font-weight: 600; border: 1px solid; border-radius: 4px; padding: 0 5px; white-space: nowrap; }}
.seg .txt {{ flex: 1; }}
.notx ul {{ columns: 2; font-size: 13px; color: #4b5563; }}
</style></head>
<body>
<header><h1>Transcript Quality Report</h1>
<div class="sub">{esc(payload.get("asr_model"))} + {esc(payload.get("diarizer"))} diarization · LLM judge: {esc(JUDGE_MODEL)} · {generated}</div>
</header>
<div class="wrap">
  <div class="cards">{cards_html}</div>
  <div>{type_html}</div>
  <h2>Summary (lowest quality first)</h2>
  <table><thead><tr><th>File</th><th>Duration</th><th>Language</th><th>Type</th><th>Spk</th><th>Quality</th><th>Judge summary</th></tr></thead>
  <tbody>{"".join(table_rows)}</tbody></table>
  {no_tx_html}
  <h2>Transcripts</h2>
  {"".join(detail_cards)}
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=".context/audio_report/transcripts.json")
    ap.add_argument("--out", dest="out", default=".context/audio_report/transcript_report.html")
    args = ap.parse_args()

    payload = json.loads(Path(args.inp).read_text(encoding="utf-8"))
    rows = rows_from_results(payload.get("results", {}))
    print(f"judging {len(rows)} transcripts with {JUDGE_MODEL} ...")
    verdicts = run_judge(rows)

    out = Path(args.out)
    out.write_text(render_report(payload, verdicts, rows), encoding="utf-8")
    print(f"wrote report -> {out}  ({len(rows)} files)")


if __name__ == "__main__":
    main()
