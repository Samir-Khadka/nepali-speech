# Training pipeline — Phase 4

Fine-tune Whisper (LoRA) and Wav2Vec2-CTC models on Nepali speech, and
score the resulting checkpoints against the v0 holdout with the benchmark
harness. No full training run has happened yet — this package is the
pipeline, not a result. The GPU runbook (a separate worker's deliverable)
is the path to the real run.

## 0. Environment

Training deps live in a SEPARATE venv — do not install torch into the
repo's main `.venv` (it must stay lean for backend/CI):

```bash
python3.12 -m venv .venv-train
.venv-train/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv-train/bin/pip install -r training/requirements.txt
.venv-train/bin/python -c "import torch, transformers, peft"
```

CPU-only torch FIRST, from the PyTorch CPU index — a plain
`pip install` pulls multi-GB CUDA wheels. (Same rule as
`benchmark/requirements-eval.txt`.) The venv used to build this package:
`~/workspace/nepali-speech/.venv-train` (torch 2.15.0+cpu — see
`training/requirements.txt` for the pinned set).

No accounts are needed or created for the pipeline itself (no HF, no
Kaggle, no Colab). The CTC alternative model `ai4bharat/indicwav2vec_v1_nepali`
is gated on the Hub and would need an account that accepted its terms —
that is a decision for the real run, not this package.

## 1. Pipeline

```
prepare_data.py  ->  train-manifest.jsonl (+ .summary.json)
        |                (train split only — see §2)
        v
train_whisper_lora.py  -- or --  train_ctc.py
        |                          (both write train_config.json)
        v
score_checkpoint.py  ->  results JSON (harness WER/CER, overall + slices)
```

### Data prep

```bash
# Full train manifest (SLR54 non-holdout + FLEURS official train split):
.venv-train/bin/python training/prepare_data.py \
    --out training/train-manifest.jsonl --tmp-dir /tmp/nepali-speech-train

# Smoke manifest: 200 clips, at most 2 SLR54 archives, metadata only:
.venv-train/bin/python training/prepare_data.py \
    --out /tmp/train-smoke.jsonl --max-clips 200 --archives 2 \
    --tmp-dir /tmp/nepali-speech-train --dry-run   # print the plan first

# Also download the smoke audio (only the archives the sampled clips need):
.venv-train/bin/python training/prepare_data.py \
    --out /tmp/train-smoke.jsonl --max-clips 200 --archives 2 \
    --audio-dir /tmp/nepali-speech-train/audio
```

`--archives` takes a count (`--archives 2` = first 2 archives, hex order,
among those touched by the sampled clips) or a list
(`--archives asr_nepali_0.zip,asr_nepali_1.zip`). SLR54 archives are
~1.2 GB each — a smoke run must never download all 16. Audio goes to a
caller-chosen directory OUTSIDE the repo (enforced); `data/` in the repo
stays empty of audio. No audio files in the repo, ever.

### Train — Whisper LoRA

```bash
.venv-train/bin/python training/train_whisper_lora.py --dry-run \
    --train-manifest training/train-manifest.jsonl --output-dir /tmp/whisper-lora-v1
.venv-train/bin/python training/train_whisper_lora.py \
    --train-manifest training/train-manifest.jsonl --output-dir /tmp/whisper-lora-v1 \
    --audio-dir /tmp/nepali-speech-train/audio \
    --model openai/whisper-large-v3-turbo --max-steps 1000 --batch-size 4
```

Default model is `openai/whisper-tiny` (smoke); the real run uses
`openai/whisper-large-v3-turbo`. LoRA settings (r=8, alpha=16,
dropout=0.05, targets q_proj/v_proj) live in
`training/configs/whisper_lora.yaml` and are validated before training.

### Train — CTC (Wav2Vec2)

```bash
.venv-train/bin/python training/train_ctc.py --dry-run \
    --train-manifest training/train-manifest.jsonl --output-dir /tmp/ctc-v1
.venv-train/bin/python training/train_ctc.py \
    --train-manifest training/train-manifest.jsonl --output-dir /tmp/ctc-v1 \
    --audio-dir /tmp/nepali-speech-train/audio --max-steps 1000 --batch-size 4
```

Default model `facebook/wav2vec2-xls-r-300m`. This is the
deployment-relevant path: CTC decoding is far faster than autoregressive
Whisper at inference (the 2026 benchmark measured CTC ~29x faster in
real-time factor), which is what the phone-call use case needs.
Alternative: `ai4bharat/indicwav2vec_v1_nepali` — verified to EXIST on
the Hub (2026-11-18, per-language v1 release, 2022-07-25), GATED
("auto"), and the bare id `ai4bharat/indicwav2vec` does NOT exist as a
model. Whether the paper's 94M checkpoint is exactly that id is
UNVERIFIED (the Hub config is unreadable without an account); check it
before the real run. Do not use `ai4bharat/indicwav2vec-hindi` for
Nepali — it is the Hindi model.

Both train scripts share the CLI shape: `--model --train-manifest
--output-dir --max-steps --max-clips --batch-size --lr
--gradient-checkpointing --fp16/--bf16 --num-workers --audio-dir`, plus
`--dry-run` (loads the config, validates the manifest, prints the plan,
exits 0 — no model downloaded, no network). Each writes a
`train_config.json` into its output dir recording the exact config
(model, manifest path + SHA-256, steps, hyperparams, augmentation flags)
so any result is reproducible. Neither script downloads a model at
import time; a missing torch/transformers/peft fails with install
directions pointing at `training/requirements.txt`.

### Score a checkpoint

```bash
.venv-train/bin/python training/score_checkpoint.py \
    --checkpoint /tmp/whisper-lora-v1 --base-model openai/whisper-large-v3-turbo \
    --manifest benchmark/manifests/v0.jsonl --audio-dir /tmp/v0-audio \
    --out /tmp/whisper-lora-v1-results.json --limit 200
```

`--limit N` scores the first N manifest clips. `--model-type`
(whisper|ctc) is auto-detected from the checkpoint
(`adapter_config.json` → Whisper LoRA; Wav2Vec2 in `config.json` →
CTC). The script transcribes, writes a hypotheses JSONL, and scores it
by importing `benchmark/evaluate.py` — the results file is produced by
the harness (overall + per-slice WER/CER), with the exact config
(checkpoint, manifest + SHA-256, steps trained and LoRA/CTC settings
read back from `train_config.json`). No hand-written numbers.

## 2. The split guarantee — train never contains holdout speakers

* SLR54 holdout rule, imported from `benchmark/build.py` (not copied):
  `sha256(user_id) % 10 == 0` → test. Train = every other speaker.
  `prepare_data.py` reuses `build.speaker_bucket`,
  `build.SLR54_TEST_BUCKET`, `build.slr54_archive_for` and
  `build.make_clip_entry`.
* FLEURS: train = the official train split only; the v0 test split is
  never touched. FLEURS carries no speaker ids, so its disjointness is
  enforced at the clip level (no test recording basename in train),
  exactly as `build.fetch_fleurs` does it.

Enforcement: after building, `prepare_data.py` checks the train rows
against `benchmark/manifests/v0.jsonl` and EXITS NON-ZERO WITHOUT
WRITING ANY MANIFEST on any speaker overlap (SLR54) or any clip overlap
(FLEURS / global). Verify independently afterwards:

```bash
.venv/bin/python benchmark/evaluate.py --check-splits \
    --train-manifest training/train-manifest.jsonl \
    --test-manifest benchmark/manifests/v0.jsonl
```

One caveat on that CLI check: it treats speaker `"unknown"` literally,
and every FLEURS row (train and test) carries speaker `"unknown"`, so it
reports an overlap for any manifest containing FLEURS rows. The
clip-level check inside `prepare_data.py` is the authoritative one for
FLEURS; the CLI check is meaningful for the SLR54 speakers. (The
`--check-splits` limitation is a harness property, not a leak — the
train/test FLEURS clip ids are disjoint by construction and verified by
`prepare_data.py`.)

## 3. Normalisation

One code path: `training/normalize.py` re-exports
`benchmark/evaluate.py::normalize_text` — NFC, strip, collapse internal
whitespace to single spaces. Training labels and scoring use the
identical function, so they cannot drift.

Devanagari numerals: the 2026 benchmark's protocol normalises with NFC
only and does NOT convert numerals, so training matches that — Devanagari
digits (०-९) and Arabic digits (0-9) are different tokens, by default.
`normalize_text_with_options(..., convert_devanagari_numerals=True)` is an
opt-in experiment flag (default OFF, on both train scripts and
`score_checkpoint.py`); training and scoring MUST use the same value,
and the value is recorded in `train_config.json` / the results JSON.

## 4. Augmentation (all OFF by default)

`training/augment.py` — pure numpy functions, unit-testable, no TTS, no
synthesised speech:

* `--augment-speed 0.9,1.1` — speed perturbation (resample; pitch shifts
  with speed).
* `--augment-noise <noise.wav> --noise-snr-db 10` — additive noise at a
  configurable SNR. The only noise shipped is a SYNTHETIC brown-noise
  WAV generated locally (`augment.make_synthetic_noise_wav`, labelled
  synthetic in a sidecar, NOT speech).
* `--augment-phone-codec` — phone-bandwidth proxy: downsample to 8 kHz,
  back up to 16 kHz. Not a real codec (no AMR/G.711 step).

WARNING: augmentation quality is UNVALIDATED until real noisy /
phone-quality Nepali data exists. These transforms make training harder
in a plausible direction; they do not prove the model handles real phone
audio, and must never be reported as a robustness result.

## 5. Honest Phase 4 status

No full training run has happened. Nothing in this package claims a WER
or CER — there are no results files, no placeholder numbers. The bar is
the 2026 benchmark (Paudel & Sayami): Whisper-Large-v3-Turbo 14.76% WER
in-domain (OpenSLR) / 39.56% FLEURS / 48.35% Common Voice; IndicWav2Vec
14.89% / 36.34% / 42.16%. A checkpoint is only comparable to those
numbers when scored on the full v0 holdout (14,145 clips) through
`score_checkpoint.py` / the harness — a `--limit` smoke score is a
pipeline check, not a benchmark result. The GPU runbook (separate
worker) is the path to the real run.

## 6. Files

| File | What it does |
| --- | --- |
| `training/normalize.py` | Re-exports the benchmark normaliser; numeral opt-in |
| `training/config.py` | YAML/JSON config load + validation |
| `training/prepare_data.py` | Builds train manifests (SLR54 non-holdout + FLEURS train), enforces disjointness, optional audio materialisation |
| `training/train_whisper_lora.py` | Whisper LoRA fine-tune (Trainer + peft) |
| `training/train_ctc.py` | CTC fine-tune (Wav2Vec2-family, Trainer) |
| `training/augment.py` | Speed / noise / phone-codec augmentation (OFF by default) |
| `training/score_checkpoint.py` | Transcribe + score a checkpoint via the harness |
| `training/common.py` | Manifest loading, 16 kHz audio loading, heavy-dep checks |
| `training/_bootstrap.py` | Puts repo root + benchmark dir on sys.path |
| `training/configs/whisper_lora.yaml` | LoRA + training defaults (Whisper) |
| `training/configs/ctc.yaml` | Training defaults (CTC) |
| `training/requirements.txt` | Pinned training deps (CPU torch first) |
| `training/tests/` | Offline pytest suite (normalise / augment / prep / configs) |

Run the tests with the training venv (it has pytest + yaml + jiwer):

```bash
.venv-train/bin/python -m pytest training/tests/ -q
```
