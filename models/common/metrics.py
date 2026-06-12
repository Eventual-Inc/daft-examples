"""Throughput, cost, and quality metrics for the transcription benchmark.

The headline number is **$ per hour of audio**, which falls straight out of
real-time factor and the GPU's hourly price:

    RTFx       = audio_seconds_processed / wall_seconds   (steady state)
    $/audio-hr = gpu_hourly_usd / RTFx

So throughput and cost are the same question: a faster stack on a given GPU is
proportionally cheaper, and a cheaper GPU can win on $/audio-hr even at lower
RTFx. Quality metrics (WER/CER, DER) are the floor that keeps a fast-but-wrong
stack from "winning".
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Approximate Modal on-demand GPU pricing (USD/hour). VERIFY against
# https://modal.com/pricing before quoting — these drift.
GPU_HOURLY_USD: dict[str, float] = {
    "T4": 0.59,
    "L4": 0.80,
    "A10G": 1.10,
    "L40S": 1.95,
    "A100-40GB": 2.10,
    "A100-80GB": 2.50,
    "H100": 3.95,
    "H200": 4.54,
    "B200": 6.25,
}


def rtfx(audio_seconds: float, wall_seconds: float) -> float:
    """Real-time factor: ×faster than realtime. 40 = one audio-hour per 90 wall-s."""
    return audio_seconds / wall_seconds if wall_seconds > 0 else 0.0


def cost_per_audio_hour(gpu: str, rtfx_value: float) -> float | None:
    """USD to process one hour of audio on ``gpu`` at the given RTFx."""
    hourly = GPU_HOURLY_USD.get(gpu)
    if hourly is None or rtfx_value <= 0:
        return None
    return hourly / rtfx_value


def wall_hours_for_backlog(total_audio_hours: float, rtfx_value: float, n_workers: int = 1) -> float:
    """Wall-clock to clear a backlog with ``n_workers`` parallel GPU containers."""
    if rtfx_value <= 0 or n_workers <= 0:
        return float("inf")
    return total_audio_hours / (rtfx_value * n_workers)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER via jiwer (lazy). Lower is better."""
    import jiwer

    return float(jiwer.wer(reference, hypothesis))


def char_error_rate(reference: str, hypothesis: str) -> float:
    import jiwer

    return float(jiwer.cer(reference, hypothesis))


def diarization_error_rate(reference_rttm: str, hypothesis_turns: list[dict], *, collar: float = 0.25) -> float:
    """DER via pyannote.metrics (lazy). ``reference_rttm`` is an RTTM file path.

    Returns the diarization error rate for one file. Needs reference speaker
    turns — there is no DER without ground truth, so this only runs on a labeled
    eval set, not on unlabeled production audio.
    """
    from pyannote.core import Annotation, Segment
    from pyannote.database.util import load_rttm
    from pyannote.metrics.diarization import DiarizationErrorRate

    reference = next(iter(load_rttm(reference_rttm).values()))
    hypothesis = Annotation()
    for turn in hypothesis_turns:
        hypothesis[Segment(turn["start"], turn["end"])] = turn["speaker"]
    metric = DiarizationErrorRate(collar=collar)
    return float(metric(reference, hypothesis))


@dataclass
class BenchmarkRow:
    """One swap-matrix cell: a (lane, asr, vad, diarizer, gpu) configuration."""

    config: str
    lane: str
    asr: str
    vad: str
    diarizer: str
    gpu: str
    audio_seconds: float = 0.0
    wall_seconds: float = 0.0
    peak_vram_gb: float | None = None
    wer: float | None = None
    der: float | None = None
    vad_seconds_removed: float | None = None
    error: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def rtfx(self) -> float:
        return rtfx(self.audio_seconds, self.wall_seconds)

    @property
    def usd_per_audio_hour(self) -> float | None:
        return cost_per_audio_hour(self.gpu, self.rtfx)

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "lane": self.lane,
            "asr": self.asr,
            "vad": self.vad,
            "diarizer": self.diarizer,
            "gpu": self.gpu,
            "audio_seconds": round(self.audio_seconds, 2),
            "wall_seconds": round(self.wall_seconds, 2),
            "rtfx": round(self.rtfx, 1),
            "usd_per_audio_hour": (round(self.usd_per_audio_hour, 4) if self.usd_per_audio_hour is not None else None),
            "peak_vram_gb": self.peak_vram_gb,
            "wer": self.wer,
            "der": self.der,
            "vad_seconds_removed": self.vad_seconds_removed,
            "error": self.error,
            **self.extra,
        }


def leaderboard_markdown(rows: list[BenchmarkRow]) -> str:
    """Render rows sorted by $/audio-hour (cheapest first) as a markdown table."""
    ok = [r for r in rows if not r.error]
    failed = [r for r in rows if r.error]
    ok.sort(key=lambda r: (r.usd_per_audio_hour is None, r.usd_per_audio_hour or 0.0))

    header = (
        "| Config | GPU | RTFx | $/audio-hr | WER | DER | VRAM (GB) |\n| --- | --- | ---: | ---: | ---: | ---: | ---: |"
    )
    lines = [header]
    for r in ok:
        cost = f"${r.usd_per_audio_hour:.4f}" if r.usd_per_audio_hour is not None else "—"
        wer = f"{r.wer:.3f}" if r.wer is not None else "—"
        der = f"{r.der:.3f}" if r.der is not None else "—"
        vram = f"{r.peak_vram_gb:.1f}" if r.peak_vram_gb is not None else "—"
        lines.append(f"| {r.config} | {r.gpu} | {r.rtfx:.1f} | {cost} | {wer} | {der} | {vram} |")
    if failed:
        lines.append("\n**Failed configs:**")
        lines.extend(f"- `{r.config}` ({r.gpu}): {r.error}" for r in failed)
    return "\n".join(lines)
