# What we tried across swaps

A record of the component candidates evaluated for the transcribe+diarize
pipeline, why each is in (or out), and where the measured numbers land. The
"Published" tables are vendor/leaderboard figures gathered during selection
(early 2026); the "Measured" table is filled in by
`modal_app.py::benchmark` on **your** audio + GPU — published RTFx are
batch-128 leaderboard numbers and will not match single-stream UDF throughput.

## Decision summary

- **Fastest end-to-end:** Parakeet-TDT-0.6B-v2 (ASR) + Sortformer (diarize). All
  NeMo, one image, one GPU pass for diarization+VAD.
- **Port from the other stack:** the **CT2 backend** for Whisper (`large-v3` on
  faster-whisper) — same weights/WER as Transformers large-v3, ~4× throughput.
  This is the single highest-value port if you stay on Whisper.
- **Structural win:** **Sortformer replaces Silero VAD + pyannote with one model**
  (its per-frame activity matrix is intrinsic VAD). Caveats: **4-speaker cap**, and
  license — streaming-v2 is CC-BY-4.0 (commercial), offline v1 is CC-BY-**NC**.
- **VAD is the highest-leverage stage** for throughput: it sits upstream of the
  expensive ASR, so silence it strips never gets transcribed. Judge a VAD swap on
  *seconds-removed AND deletion-WER together*, not either alone.

## ASR candidates (Published, early 2026)

| Model | Lang | WER (OpenASR) | RTFx | Timestamps | License | Verdict |
| --- | --- | ---: | ---: | --- | --- | --- |
| **nvidia/parakeet-tdt-0.6b-v2** | EN | 6.05% | ~3386 | word+segment | CC-BY-4.0 | ★ fastest default |
| nvidia/parakeet-tdt-0.6b-v3 | 25 EU | 6.34% | ~3333 | word+segment | CC-BY-4.0 | multilingual, ≤3h local-attn |
| nvidia/canary-1b-v2 | 25 EU | 7.15% | ~749 | word+segment | CC-BY-4.0 | multilingual + translation |
| nvidia/canary-qwen-2.5b | EN | 5.63% | ~418 | weak | CC-BY-4.0 | best WER, slow, no clean ts |
| faster-whisper `large-v3` | 99 | (= Whisper L3) | ~4× openai | word/segment | MIT | quality-held baseline |
| faster-whisper `turbo` | 99 | small regression | faster | word/segment | MIT | speed tier |
| faster-whisper `distil-large-v3` | EN | — | fastest CT2 | word/segment | MIT | English-only |

## Diarization candidates (Published, early 2026)

| Model | Max spk | DER (CALLHOME 2sp) | Streaming | Intrinsic VAD | License | Verdict |
| --- | ---: | ---: | --- | --- | --- | --- |
| **nvidia/diar_streaming_sortformer_4spk-v2** | 4 | 6.57% | yes | yes | CC-BY-4.0 | ★ commercial-safe |
| nvidia/diar_sortformer_4spk-v1 | 4 | 5.85% | no | yes | CC-BY-**NC** | research only |
| nvidia/diar_streaming_sortformer_4spk-v2.1 | 4 | 6.65% | yes | yes | NVIDIA OML | AMI-tuned |
| pyannote/speaker-diarization-3.1 | unbounded | ~ competitive | no | no (needs VAD) | gated (HF) | no speaker cap |

## VAD candidates

| Model | Where | Cost | Notes |
| --- | --- | --- | --- |
| Silero | both (built into faster-whisper) | tiny, CPU | current default |
| nvidia Frame-VAD MarbleNet v2 | NeMo lane | 91.5K params, GPU | multilingual; front-ends Parakeet via compaction |
| (Sortformer intrinsic) | NeMo lane | free w/ diarization | no separate VAD stage needed |

## Measured leaderboard (fill in via `::benchmark`)

> Run `modal_app.py::benchmark` to populate. Sorted by $/audio-hour.

| Config | GPU | RTFx | $/audio-hr | WER | DER | VRAM (GB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| parakeet+sortformer | _A100-40GB_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| parakeet | _A100-40GB_ | _tbd_ | _tbd_ | _tbd_ | — | _tbd_ |
| whisper-large-v3 | _A100-40GB_ | _tbd_ | _tbd_ | _tbd_ | — | _tbd_ |
| whisper+pyannote | _A100-40GB_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

## Open / not yet wired

- MarbleNet VAD forward pass (`models/common/vad.py`) needs validation against
  NeMo's `frame_vad_infer.py` output thresholds on real audio.
- Sortformer `diarize()` return shape (`_parse_turns`) is defensive across NeMo
  versions — verify against your pinned NeMo build.
- Canary-Qwen-2.5b (best WER) is excluded: needs NeMo-from-git and lacks clean
  timestamps. Revisit if WER floor matters more than speed.
