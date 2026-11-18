# Model card (DRAFT — nothing published, no model exists yet)

> DRAFT for the future fine-tuned Nepali ASR model from the
> `nepali-speech` project (Phase 4). Fill in the bracketed fields after
> a real training run and a full v0-holdout score. Do not publish this
> card — or any version of it — until the numbers in it are measured.
> Nothing in this draft claims a result: there is no fine-tuned model
> yet, and no WER/CER has been measured for one.

---
language:
- ne
license: TBD (must be compatible with the training-data licenses below)
tags:
- automatic-speech-recognition
- nepali
- lora
- wav2vec2
- whisper
datasets:
- openslr/slr54
- google/fleurs
---

# Nepali ASR — fine-tuned Whisper-LoRA / Wav2Vec2-CTC (DRAFT)

## Model description

Two fine-tuned Nepali speech-recognition models, trained on the same
train split with the `nepali-speech` Phase 4 pipeline
(`training/` in https://github.com/Samir-Khadka/nepali-speech):

- **Whisper-LoRA**: `openai/whisper-large-v3-turbo` + a LoRA adapter
  (r=8, alpha=16, dropout=0.05, targets `q_proj`/`v_proj`).
  [Fill in: base model actually used, adapter revision, steps trained.]
- **CTC**: `facebook/wav2vec2-xls-r-300m` fine-tuned end-to-end
  (feature extractor frozen), or `ai4bharat/indicwav2vec_v1_nepali`
  if that gated model was used instead. [Fill in: which base model,
  steps trained.]

[Fill in after the run: training date, repo commit, `train_config.json`
contents, checkpoint location.]

## Intended use

Transcribing Nepali speech — primarily **read speech**, which is what
the training and evaluation data are. The CTC model is the intended
default for the phone-call assistant use case (see "Limitations" and
`training/serving.md`): CTC decoders run far faster than autoregressive
Whisper at inference (the 2026 benchmark measured CTC ~29× faster in
real-time factor), which is what a live phone call needs.

Out-of-scope uses: anything requiring a diagnosis, a legal transcript,
or a safety-critical decision based on the transcription; use as the
sole evidence in any adjudication; transcription of languages other
than Nepali (the models were not trained for them).

## Training data

| Source | Split used | Clips | License |
| --- | --- | --- | --- |
| OpenSLR SLR54 (Nepali ASR) | train portion only — speakers whose `sha256(user_id) % 10 != 0` | ~144,486 utterances (ESTIMATE — confirm from the run's `train-manifest.summary.json`) | **CC BY-SA 4.0** |
| FLEURS `ne_np` | official train split only | 3,332 clips | **CC BY 4.0** |

- **Holdout discipline:** the v0 test split (SLR54 speakers with
  `sha256(user_id) % 10 == 0`, 13,419 clips; FLEURS official test
  split, 726 clips) was never used in training. Disjointness is
  enforced by `training/prepare_data.py`, which refuses to write a
  train manifest on any speaker overlap (SLR54) or clip overlap
  (FLEURS). Verify with `benchmark/evaluate.py --check-splits`.
- **No synthetic speech.** No TTS, no synthesised audio. Augmentation,
  if any was used, is recorded in `train_config.json` (speed
  perturbation / additive synthetic brown noise / phone-bandwidth
  proxy — all OFF by default, and unvalidated as a robustness claim
  until real noisy/phone-quality Nepali data exists).
- **No Phase 2 collected audio** was used (none existed at training
  time). The realistic portion — phone-quality, noisy, dialect,
  code-switched Nepali — is pending Phase 2 collection.

## Evaluation

- **Protocol:** the v0 benchmark (`benchmark/` in the same repo) —
  NFC normalisation, WER/CER with jiwer 4.0.0, micro-averaged overall
  and per slice (source / gender / region / quality). Scored with
  `training/score_checkpoint.py` against the full v0 holdout
  (14,145 clips). A `--limit` smoke score is a pipeline check, not a
  benchmark result.
- **Results (CPU smoke run, 2026-11-18 — pipeline check, NOT a
  benchmark result):** Whisper-tiny LoRA (r=8/α=16, 300 steps, 1
  epoch over 300 clips, ~50 min audio) scored on a 50-clip slice of
  the v0 holdout (25 FLEURS test + 25 SLR54 holdout, harness-measured
  with `training/score_checkpoint.py`; full numbers in
  `training/results/smoke-whisper-tiny-2026-11-18.json`):
  overall **WER 138.45%, CER 146.57%** — FLEURS 137.67% / 120.44%,
  SLR54 143.24% / 372.70%. The same slice scored with the **base
  Whisper-tiny, zero-shot** (`training/results/base-whisper-tiny-2026-11-18.json`):
  overall **WER 132.01%, CER 137.77%**. The two are indistinguishable
  on n=50: the smoke run neither improved nor degraded accuracy, which
  is the expected outcome — a smoke run proves the pipeline, not
  accuracy. **Why the absolute numbers are this high:** Whisper-tiny
  romanizes Nepali speech into Latin script even when primed with the
  `<|ne|>` language token (verified by direct probe on 2026-11-18:
  `language="ne"` + `task="transcribe"` kwargs and
  `forced_decoder_ids` all give Latin-script output), so nearly every
  word counts as a substitution against the Devanagari references.
  Devanagari output is what the full GPU fine-tune (larger model, full
  data) has to teach; the smoke run does not demonstrate it. Training
  loss did fall (3.69 → 1.70), so the LoRA training itself ran
  correctly. No full-holdout (14,145-clip) number exists yet.

### Comparison bar (theirs, not ours)

Paudel & Sayami (2026), Nepali ASR Benchmark
(https://p-sumann.github.io/nepali-asr-benchmark/) — the numbers our
results are compared against, not our results:

- Whisper-Large-v3-Turbo: **14.76% WER** in-domain (OpenSLR) /
  **39.56%** FLEURS / **48.35%** Common Voice.
- IndicWav2Vec: **14.89%** / **36.34%** / **42.16%**.

**No claim of beating the 2026 baseline is made here, and none may be
added until a real run scores the full v0 holdout and the numbers are
in the results JSON.** The v0 holdout mirrors the 2026 protocol
(including the 80/10/10 speaker-disjoint SLR54 split) but defines its
own SLR54 holdout, because the benchmark's exact test speaker list was
never published — so the numbers are comparable in protocol, not
identical in test set.

## Limitations

- **v0 is read speech only.** Both training and evaluation data are
  read, mostly clean, studio-style utterances (SLR54, FLEURS). The
  models have not been shown to handle phone-quality, noisy, dialect,
  or code-switched Nepali — the realistic slices are defined in the
  benchmark schema but empty until Phase 2 audio exists. Do not report
  these models as robust to real phone audio.
- **Augmentation is not robustness.** If speed/noise/phone-codec
  augmentation was used in training, it makes training harder in a
  plausible direction; it does not prove the model handles real phone
  audio.
- **Devanagari numerals are not converted by default.** The protocol
  normalises with NFC only; Devanagari digits (०-९) and Arabic digits
  (0-9) are different tokens. Training and scoring must use the same
  numeral setting (recorded in `train_config.json`).
- **Speaker and domain coverage:** SLR54 train speakers are the
  non-holdout portion of 527 speakers; FLEURS carries no speaker ids.
  Performance on speakers, accents, and recording conditions outside
  these corpora is unmeasured.
- **Whisper hallucinations:** as with the base model, the Whisper
  variant can hallucinate text during silence or noise; the CTC
  variant is less prone to that failure mode but makes more
  substitution errors on long utterances. [Confirm or correct after
  the run, from the per-clip scores.]

## Ethical considerations

- The training data is public, openly licensed speech (CC BY-SA 4.0 /
  CC BY 4.0) with no personal data beyond voice recordings released
  under those licences. No Phase 2 collected audio — which is gathered
  under explicit CC0 consent — was used.
- Speech recognition can mis-transcribe, and errors fall unevenly
  across accents and recording conditions. The per-slice reporting in
  the evaluation exists so that unevenness is visible rather than
  averaged away; check the slices before deploying to a new population.

## How to reproduce

1. Follow `training/kaggle-finetune-runbook.md` (data prep, both
   training commands, scoring).
2. The reproducibility record is `train_config.json` in the checkpoint
   dir (model, config file, manifest path + SHA-256, steps, hyperparams,
   augmentation flags) and the results JSON from
   `training/score_checkpoint.py` (checkpoint, manifest SHA-256, clip
   count, harness WER/CER overall + slices).
