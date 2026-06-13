# Transcribe + Diarize (fastest pipeline + swap matrix)

Audio/video → speaker-labeled transcript, optimized for **throughput / $-per-hour-of-audio**,
with every stage pluggable so component swaps are an experiment, not a rewrite.

The pipeline is five stages — `decode → VAD → ASR → diarize → merge` — and the
only engine-locked one is ASR. See [`RESULTS.md`](./RESULTS.md) for what we tried.

## The two container lanes

NeMo and faster-whisper **cannot share an image** (NeMo pins `numpy<2` + a torch
cuDNN stack; CTranslate2 needs cuDNN 9 with no torch). So configs run in two lanes:

| Lane | ASR | Diarizer | VAD | Notes |
| --- | --- | --- | --- | --- |
| **NeMo** (fastest) | Parakeet-TDT-0.6B / Canary | **Sortformer** | MarbleNet or none | Sortformer's activity matrix is intrinsic VAD — it collapses VAD+diarization into one model |
| **Whisper** | faster-whisper CT2 `large-v3` / `turbo` | pyannote | Silero (built-in) | The WhisperX-family combo; no 4-speaker cap |

`models/common/modal_images.py` owns both image builders.

## Fastest config

**Parakeet-TDT-0.6B-v2 + Sortformer** (`parakeet+sortformer`, marked ★). Parakeet is
the throughput champion (RTFx ~3.4k vs Canary ~0.75k, CC-BY-4.0), and Sortformer
gives diarization "for free" off the same GPU pass instead of a separate pyannote stage.

```bash
uv run --extra models modal run pipelines/transcribe_diarize/modal_app.py::run \
  --source 'hf://datasets/Eventual-Inc/sample-files/audio/*.mp3'
```

Point `--source` at your own audio/video (a glob URI or a local dir; video is
decoded for its audio track by Daft's `audio_file`).

## Run the swap matrix

```bash
# prewarm weights into the model-cache Volume first (per lane)
uv run --extra models modal run pipelines/transcribe_diarize/modal_app.py::benchmark

# a subset, on a different GPU
BENCH_GPU=L40S uv run --extra models modal run \
  pipelines/transcribe_diarize/modal_app.py::benchmark \
  --configs 'parakeet+sortformer,whisper+pyannote'
```

It routes each config to its lane, warms the model, times a steady-state collect,
and writes a `$/audio-hour` leaderboard to `.context/transcribe_diarize/RESULTS_measured.md`.

`$/audio-hr = GPU_$/hr / RTFx` (see `models/common/metrics.py`). Sweep `BENCH_GPU`
across `L4 / A10G / L40S / A100-40GB / H100` — a cheaper GPU can win on cost even
at lower RTFx. Total backlog wall-clock = `total_audio_hr / (RTFx × n_workers)`,
so to "get it done quickly" you raise Modal's container count; cost stays ~flat.

## Quality floor

`models/common/metrics.py` provides WER/CER (jiwer) and DER (pyannote.metrics).
DER needs reference speaker turns (RTTM), so it only runs on a **labeled** eval
set — there is no DER on unlabeled production audio. For your own data without
references, compare transcripts side-by-side or with an LLM judge; hand-label a
few files if you want hard WER/DER numbers on your content.

## Swapping a stage

Each stage is a backend behind a stable contract (`models/common/speech.py`):
ASR backends emit `ASRResult`, diarizers emit `list[SpeakerSegment]`. Add a
candidate by writing one `@daft.cls` that conforms, registering it in
`pipeline.py`'s dispatch, and adding a `SwapConfig` row in `benchmark.py`.

- **VAD** lives in `models/common/vad.py` (`SileroVAD`, `MarbleNetVAD`) behind a
  `speech_timestamps()` protocol; silence-compaction (`models/common/audio.py`)
  fronts any ASR with any VAD.
- **ASR**: `models/parakeet/model.py`, `models/faster_whisper/asr.py`.
- **Diarizer**: `models/sortformer/model.py`, `models/pyannote/model.py`.
