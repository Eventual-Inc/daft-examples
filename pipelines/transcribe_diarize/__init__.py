"""Transcribe + diarize pipeline with pluggable ASR / VAD / diarizer stages.

Two container lanes (NeMo and faster-whisper cannot coexist — see
``models/common/modal_images.py``):

- **NeMo lane (fastest):** Parakeet ASR + Sortformer diarizer (Sortformer's
  activity matrix is intrinsic VAD).
- **Whisper lane:** faster-whisper ASR + pyannote diarizer + Silero VAD.

``benchmark.py`` sweeps configs across both lanes and reports RTFx + $/audio-hour.
"""
