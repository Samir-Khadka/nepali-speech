# Nepali ASR Benchmark — Leaderboard

> **Static markdown leaderboard, 2026-11-18.** Updated by hand (or by
> a maintainer script) when a submission's full harness output is
> received. A row is listed only with the **complete sliced output**
> (overall + per-source + per-slice) from `benchmark/evaluate.py` —
> no single-average entries. Every number states its training data.
> **No model has been scored by this project yet** — the only numbers
> below are the 2026 paper's, labelled as theirs.

## How to read this table

- **WER / CER** are micro-averaged percentages (errors ÷ reference
  words/chars), NFC-normalised, scored with `jiwer` — see
  [`README.md`](README.md) §5–§6.
- **Overall** is the micro-average over **all v0 clips**; the
  per-source columns break it out (OpenSLR / FLEURS / Common Voice).
- **Slices** (gender, region/province, quality) are published with
  every submission in its results JSON; the table shows the overall
  and per-source numbers, and links to the full sliced output.
  A submission whose slices are missing is not listed.
- **TBD** means "not yet measured" — never a guess, never an estimate.
- **Reference rows** (marked ◆) are Paudel & Sayami (2026) numbers,
  reproduced here for comparison. They are **their** results, not
  this project's, and they were scored under their protocol (same
  NFC + `jiwer` protocol this benchmark adopts).

## Reference rows — Paudel & Sayami (2026) ◆

Six models fine-tuned on OpenSLR SLR54 under one protocol (NFC +
`jiwer`, speaker-disjoint splits, three test sets). Source:
https://p-sumann.github.io/nepali-asr-benchmark/

Only the numbers stated in the project's source summary of that paper
are reproduced here. Other models' exact WERs are **not** reproduced
because they were not in that summary — they are TBD from the paper,
not estimated.

| Model | Params | Fine-tuned on | WER — OpenSLR (in-domain) | WER — FLEURS | WER — Common Voice | CER (overall) | Date | Link |
|---|---|---|---|---|---|---|---|---|
| Whisper-Large-v3-Turbo ◆ | 809M | SLR54 | **14.76%** | **39.56%** | **48.35%** | TBD | 2026 | https://p-sumann.github.io/nepali-asr-benchmark/ |
| Whisper (zero-shot, no fine-tune) ◆ | — | None | ~94% out-of-domain (as reported) | TBD | TBD | TBD | 2026 | https://p-sumann.github.io/nepali-asr-benchmark/ |
| IndicWav2Vec ◆ | 94M | SLR54 | **14.89%** | **36.34%** | **42.16%** | TBD | 2026 | https://p-sumann.github.io/nepali-asr-benchmark/ |

Qualitative findings from the same paper (not numbers, reported here
as stated in the summary): **CTC decoding is 29× faster** than
autoregressive decoding in their comparison; zero-shot Whisper is
~94% WER out-of-domain, i.e. fine-tuning on SLR54 is what makes the
in-domain number possible.

## Submissions scored on this benchmark (v0)

_None yet. The first row appears when the v0 manifest exists and a
model's full harness output is received._

| Model | Params | Trained / fine-tuned on | WER overall | CER overall | WER — OpenSLR | WER — FLEURS | WER — Common Voice | Date | Full results | Link |
|---|---|---|---|---|---|---|---|---|---|---|
| — | — | — | TBD | TBD | TBD | TBD | TBD | — | — | — |

## Realistic portion (Phase 2 data — pending)

These slices do not exist yet: Phase 2 collection has produced
**zero audio** (no volunteers; live deploy still pending). Rows appear
here only when the audio exists and a model is scored on it.

| Slice | Clips | Models scored | Best WER | Date |
|---|---|---|---|---|
| Phone-quality | 0 — pending | — | TBD | — |
| Noisy | 0 — pending | — | TBD | — |
| Dialect / regional | 0 — pending | — | TBD | — |
| Code-switched | 0 — pending | — | TBD | — |
| Spontaneous | 0 — pending | — | TBD | — |

## How to submit

1. Build or obtain the v0 manifest (`benchmark/manifests/v0.jsonl`).
2. Score with the harness — full output, all slices:
   ```bash
   python benchmark/evaluate.py --manifest benchmark/manifests/v0.jsonl \
       --hypotheses /tmp/run1.jsonl --out /tmp/run1-results.json
   ```
3. Verify speaker-disjointness if you trained on SLR54 or any
   overlapping corpus:
   ```bash
   python benchmark/evaluate.py --check-splits \
       --train-manifest data/train.jsonl \
       --test-manifest benchmark/manifests/v0.jsonl
   ```
4. Send: model name + checkpoint link, parameter count, training /
   fine-tuning data (dataset + split), the full results JSON, the
   date, and a statement that no benchmark clip or speaker was in
   training. The maintainers add the row with the overall, per-source,
   and per-slice numbers and a link to the full results.
5. **Disqualification:** training on any benchmark clip, speaker, or
   transcript; reporting only a single overall number; or
   speaker overlap between training data and the test set.

## Version history

| Version | Date | Change |
|---|---|---|
| v0 (spec) | 2026-11-18 | Leaderboard created. Reference rows from Paudel & Sayami (2026) only. No submissions scored yet; realistic portion empty. |
