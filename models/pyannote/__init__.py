"""pyannote speaker diarization. Backend: PyTorch. The Whisper-lane diarizer.

No speaker-count cap (unlike Sortformer's 4), gated weights (needs HF_TOKEN +
license acceptance). Emits the shared SpeakerSegment contract.
"""
