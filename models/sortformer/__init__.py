"""NVIDIA Sortformer speaker diarization. Backend: PyTorch (NeMo).

End-to-end neural diarization: the per-frame speaker-activity matrix is intrinsic
VAD, so Sortformer collapses a separate VAD stage and pyannote into one model.
Hard cap of 4 speakers. License varies by variant — streaming-v2 is CC-BY-4.0
(commercial OK); offline v1 is CC-BY-NC.
"""
