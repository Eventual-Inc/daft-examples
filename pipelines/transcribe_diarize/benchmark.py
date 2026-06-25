"""The swap matrix: what we tried, and how each config is scored.

Pure/importable (no modal, no GPU). ``modal_app.py`` runs each config and fills
in the measured columns; ``leaderboard_markdown`` renders the result.

Each config names a (lane, ASR, VAD, diarizer) point. The benchmark holds the
audio set + GPU fixed and flips one stage at a time, so a difference is
attributable to the stage that changed — that's the "should we port it?" signal.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipelines.transcribe_diarize.pipeline import ASR_LANE, DIARIZER_LANE


@dataclass(frozen=True)
class SwapConfig:
    name: str
    asr: str
    diarizer: str
    vad: str = "none"
    asr_model: str = ""
    diarizer_model: str = ""
    row_batch_size: int = 16
    note: str = ""
    asr_kwargs: dict = field(default_factory=dict)

    @property
    def lane(self) -> str:
        lanes = {ASR_LANE[self.asr], DIARIZER_LANE.get(self.diarizer, ASR_LANE[self.asr])}
        if len(lanes) > 1:
            raise ValueError(
                f"config '{self.name}' mixes lanes {lanes}: ASR '{self.asr}' and diarizer "
                f"'{self.diarizer}' need separate images and cannot run in one pass"
            )
        return lanes.pop()


# The default swap matrix. ASR-only rows isolate the engine; +diarizer rows show
# the end-to-end cost. Flip one stage per row vs the row above it.
SWAP_MATRIX: list[SwapConfig] = [
    # ── NeMo lane ────────────────────────────────────────────────────────────
    SwapConfig("parakeet", asr="parakeet", diarizer="none", note="fastest ASR, English (v2)"),
    SwapConfig(
        "parakeet+vad", asr="parakeet", diarizer="none", vad="marblenet", note="MarbleNet silence-compaction front-end"
    ),
    SwapConfig(
        "parakeet+sortformer",
        asr="parakeet",
        diarizer="sortformer",
        vad="marblenet",
        note="★ fastest end-to-end; Sortformer = joint VAD+diar",
    ),
    SwapConfig(
        "canary+sortformer",
        asr="canary",
        diarizer="sortformer",
        asr_model="nvidia/canary-1b-v2",
        note="multilingual ASR swap",
    ),
    # ── Whisper lane ─────────────────────────────────────────────────────────
    SwapConfig(
        "whisper-large-v3",
        asr="faster_whisper",
        diarizer="none",
        vad="builtin",
        note="CT2 large-v3, quality-held baseline",
    ),
    SwapConfig(
        "whisper-turbo",
        asr="faster_whisper",
        diarizer="none",
        vad="builtin",
        asr_model="turbo",
        note="speed tier, small WER regression",
    ),
    SwapConfig(
        "whisper+pyannote",
        asr="faster_whisper",
        diarizer="pyannote",
        vad="builtin",
        note="current-style stack, no speaker cap",
    ),
]


def matrix_by_lane(matrix: list[SwapConfig] | None = None) -> dict[str, list[SwapConfig]]:
    """Group configs by image lane so each lane's image is built once."""
    configs = matrix if matrix is not None else SWAP_MATRIX
    grouped: dict[str, list[SwapConfig]] = {}
    for config in configs:
        grouped.setdefault(config.lane, []).append(config)
    return grouped
