"""Pluggable VAD stage: produce speech timestamps from a 16 kHz mono waveform.

Every VAD returns the same contract — a list of sample-index ``{"start", "end"}``
spans — so ``models.common.audio.compact_speech`` can front any ASR engine with
any VAD. Swap a VAD by swapping the class; nothing downstream changes.

Candidates:
- ``SileroVAD``    — tiny, CPU-friendly, the current default (lazy `silero_vad`).
- ``MarbleNetVAD`` — NVIDIA Frame-VAD MarbleNet v2 (lazy `nemo_toolkit[asr]`),
                     GPU, multilingual. Only usable in the NeMo image lane.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from models.common.audio import SAMPLE_RATE


class VAD(Protocol):
    def speech_timestamps(self, waveform: np.ndarray) -> list[dict[str, int]]:
        """Return speech spans as sample-index ``{"start", "end"}`` dicts."""
        ...


class SileroVAD:
    """Silero VAD. Loaded once per worker; CPU is fine (the model is tiny)."""

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 500,
        speech_pad_ms: int = 200,
        max_speech_duration_s: float = float("inf"),
    ):
        from silero_vad import load_silero_vad

        self.model = load_silero_vad()
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.max_speech_duration_s = max_speech_duration_s

    def speech_timestamps(self, waveform: np.ndarray) -> list[dict[str, int]]:
        import torch
        from silero_vad import get_speech_timestamps

        return get_speech_timestamps(
            torch.from_numpy(waveform),
            self.model,
            threshold=self.threshold,
            sampling_rate=SAMPLE_RATE,
            min_speech_duration_ms=self.min_speech_duration_ms,
            max_speech_duration_s=self.max_speech_duration_s,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
        )


class MarbleNetVAD:
    """NVIDIA Frame-VAD Multilingual MarbleNet v2 (20 ms frame resolution).

    Outputs per-frame speech probability; we threshold + merge into spans. NeMo
    ships ``frame_vad_infer.py`` for production RTTM output, but for a UDF the
    in-process forward pass below avoids subprocess/manifest overhead.

    Requires ``nemo_toolkit[asr]`` — only available in the NeMo image lane.
    """

    FRAME_MS = 20
    MODEL_ID = "nvidia/frame_vad_multilingual_marblenet_v2.0"

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 500,
        speech_pad_ms: int = 200,
        device: str | None = None,
    ):
        import nemo.collections.asr as nemo_asr
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nemo_asr.models.EncDecFrameClassificationModel.from_pretrained(model_id).eval().to(self.device)
        self.threshold = threshold
        self.min_speech_frames = max(1, min_speech_duration_ms // self.FRAME_MS)
        self.min_silence_frames = max(1, min_silence_duration_ms // self.FRAME_MS)
        self.pad_frames = speech_pad_ms // self.FRAME_MS

    def speech_timestamps(self, waveform: np.ndarray) -> list[dict[str, int]]:
        signal = self.torch.from_numpy(waveform).unsqueeze(0).to(self.device)
        length = self.torch.tensor([waveform.shape[0]], device=self.device).long()
        with self.torch.no_grad():
            logits = self.model(input_signal=signal, input_signal_length=length)
        probs = self.torch.softmax(logits, dim=-1)[0, :, 1].cpu().numpy()
        return self._frames_to_spans(probs)

    def _frames_to_spans(self, probs: np.ndarray) -> list[dict[str, int]]:
        frame_samples = int(SAMPLE_RATE * self.FRAME_MS / 1000)
        speech = probs >= self.threshold

        spans: list[list[int]] = []
        run_start: int | None = None
        silence = 0
        for index, is_speech in enumerate(speech):
            if is_speech:
                if run_start is None:
                    run_start = index
                silence = 0
            elif run_start is not None:
                silence += 1
                if silence >= self.min_silence_frames:
                    spans.append([run_start, index - silence + 1])
                    run_start = None
        if run_start is not None:
            spans.append([run_start, len(speech)])

        result: list[dict[str, int]] = []
        for start_frame, end_frame in spans:
            if end_frame - start_frame < self.min_speech_frames:
                continue
            start = max(0, (start_frame - self.pad_frames) * frame_samples)
            end = (end_frame + self.pad_frames) * frame_samples
            result.append({"start": int(start), "end": int(end)})
        return result


VAD_REGISTRY: dict[str, type] = {
    "silero": SileroVAD,
    "marblenet": MarbleNetVAD,
}


def build_vad(name: str, **kwargs) -> VAD:
    if name == "none":
        return _PassthroughVAD()
    try:
        return VAD_REGISTRY[name](**kwargs)
    except KeyError as exc:
        raise ValueError(f"unknown VAD '{name}'; choices: {sorted(VAD_REGISTRY) + ['none']}") from exc


class _PassthroughVAD:
    """No VAD — emit one span over the whole waveform (lets ASR see everything)."""

    def speech_timestamps(self, waveform: np.ndarray) -> list[dict[str, int]]:
        return [{"start": 0, "end": int(waveform.shape[0])}]
