# Smoke eval — Whisper-tiny on 5 real FLEURS clips

**Status: pipeline proof only. Not a baseline number. Do not cite as a
benchmark result.**

## What ran

| item | value |
|---|---|
| Date | 2026-11-18 |
| Model | openai-whisper `tiny` (39 M params), zero-shot, CPU |
| Clips | **5** — real audio from FLEURS `ne_np` test split (`google/fleurs`, CC BY 4.0), downloaded from the official Hugging Face repo; kept in `/tmp/v0-audio/`, never in the repo |
| References | FLEURS `test.tsv` normalised-transcription column (Devanagari, NFC) |
| Harness | `benchmark/evaluate.py --transcribe --model tiny` |
| Command | `.venv/bin/python benchmark/evaluate.py --manifest /tmp/smoke-manifest.jsonl --transcribe --model tiny --audio-dir /tmp/v0-audio --out /tmp/tiny-smoke-results.json` |

Clip durations 5.1–10.0 s (39.7 s total). All 5 clips are male speakers —
the FLEURS `ne_np` test split's `test.tsv` lists only MALE rows, so no
gender comparison was possible in this sample. Speaker IDs are not
published in the FLEURS TSV, so `speaker` is recorded as `unknown`.

## Result (n = 5)

| metric | value |
|---|---|
| **WER** | **101.43%** (71 errors / 70 reference words; sub 40, del 30, ins 1) |
| **CER** | **95.72%** (447 errors / 467 reference characters) |

Per-clip WER: 1.00, 1.00, 1.00, 1.00, 1.06. Every slice equals the
overall (all 5 clips are male / clean / FLEURS / region unknown), so the
per-slice table adds nothing at this sample size — it is printed by the
harness anyway.

## Why the number is ~100%: the model, not the harness

Whisper-tiny zero-shot on Nepali does not produce Devanagari at all on
these clips — it outputs romanised hallucination. Example (clip
`...15110567688551233602`, reference
"किनमेल गर्न सबैभन्दा कम रोचक ठाउँहरूमध्ये मोरक्कोमा कासाब्लान्का हो"):

> Hypothesis: `¡Kinmost ganas avvvandakvaram – rojakhta varva… …dimo oka dewal adana`

That is a real transcription (the model ran, audio decoded, text came
out) — it is just wrong in essentially every word, which is exactly what
a ~100% WER means. This is consistent in direction with Paudel &
Sayami (2026), who report zero-shot Whisper at ~94% WER out-of-domain;
tiny is the smallest Whisper and Nepali is a low-resource language for
it, so ~101% on 5 clips is a plausible zero-shot tiny result, not a
harness bug. The harness itself is pinned by 24 hand-computed tests
(`benchmark/tests/test_evaluate.py`).

## What this proves

1. The harness transcribes real audio end to end (`--transcribe` path:
   model load → ffmpeg decode → transcribe → score → JSON out).
2. NFC + jiwer scoring, micro-averaged overall, and per-slice reporting
   all run on real (non-synthetic) data.
3. The manifest schema (`clip_id`, `source`, `speaker`, `transcript`,
   `audio_path`, `gender`, `region`, `quality`, `duration_s`) works
   against real FLEURS rows.

## What it does NOT prove

- Nothing about any model's real Nepali accuracy: n = 5, one source,
  one gender, one (tiny) model, zero-shot.
- Nothing about the full v0 set: the v0 manifest
  (`benchmark/manifests/v0.jsonl`) is built by a separate step and did
  not exist when this smoke eval ran.
- The full baseline (Whisper / MMS / XLS-R on the full v0 set) is a
  free-Kaggle-GPU job — see `kaggle-baseline-runbook.md`.

## Reproduce

```bash
# 1. Fetch the 5 clips (FLEURS ne_np test tarball, ~420 MB — the tarball
#    itself is deleted after extraction; only the 5 wavs are kept in /tmp)
curl -L -o test.tar.gz https://huggingface.co/datasets/google/fleurs/resolve/main/data/ne_np/audio/test.tar.gz
mkdir -p /tmp/v0-audio
tar xzf test.tar.gz -C /tmp/v0-audio --strip-components=1 \
  test/10723015018907695854.wav test/15110567688551233602.wav \
  test/1892031230327726981.wav test/2654532460547677228.wav \
  test/5974790822098555955.wav

# 2. Build the manifest from test.tsv (normalised-transcription column)
curl -L -o /tmp/fleurs-ne-test.tsv \
  https://huggingface.co/datasets/google/fleurs/resolve/main/data/ne_np/test.tsv
# (5 rows: first 5 distinct sentence_ids; speaker "unknown" — the TSV
#  publishes no speaker IDs; gender "male" per the TSV)

# 3. Run (CPU torch must be the +cpu build — see requirements-eval.txt)
.venv/bin/python benchmark/evaluate.py --manifest /tmp/smoke-manifest.jsonl \
  --transcribe --model tiny --audio-dir /tmp/v0-audio \
  --out /tmp/tiny-smoke-results.json
```
