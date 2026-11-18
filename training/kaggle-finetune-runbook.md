# Kaggle fine-tune runbook — full training run (Phase 4)

How to run the FULL Whisper-LoRA and CTC fine-tunes on free Kaggle GPUs,
then score the checkpoints on the v0 benchmark. No accounts are created
by this document — creating the Kaggle account and accepting the
IndicWav2Vec gated-model terms (if that model is chosen) are the
operator's steps, in their own browser.

Status: **no full training run has happened yet.** Every GPU-hour figure
below is an ESTIMATE with its basis stated in §6. Nothing here is a
measured result. Re-check every quota figure against Kaggle's own docs
at run time — free tiers change without notice.

## 0. Free-tier facts (verified live 2026-11-18)

From Kaggle's own documentation, read 2026-11-18:

- **GPU quota: 30 hours per week**, resetting weekly, "or sometimes
  higher depending on demand and resources"
  (https://www.kaggle.com/docs/efficient-gpu-usage).
- **Session cap: 12 hours** for CPU and GPU notebook sessions (9 hours
  for TPU) (https://www.kaggle.com/docs/notebooks).
- **Hardware: GPU T4 x2** — 2× NVIDIA Tesla T4 — or 1× Tesla P100
  (same docs page, "T4 x2 GPU Specifications").
- **20 GB of output** may be saved to disk in `/kaggle/working` and
  reused by later notebooks (same docs page).
- Interactive sessions idle-timeout after **60 minutes**
  (https://www.kaggle.com/docs/efficient-gpu-usage) — stop the session
  before closing the window or it keeps burning quota.
- GPU access requires a verified Kaggle account (phone verification);
  confirm the exact requirement on the account settings page at setup.
  Your own remaining quota is shown in Kaggle account settings.

**Re-check at run time:** the 30 h/week figure, the 12 h session cap,
and the T4 x2 / P100 choice. If any of them has changed, this runbook's
session plan (§6) must be re-done, not the training commands.

### Alternative: Google Colab (free tier)

- Free tier: usually a single NVIDIA T4 (16 GB) when one is available.
- Sessions run **at most ~12 hours**; idle runtimes time out
  (~90 minutes is the commonly reported figure — Google's FAQ does not
  publish an exact idle duration).
- **Google does not publish a free-tier weekly quota.** Its FAQ says
  limits "fluctuate" and GPU access "varies over time"; heavy sustained
  use triggers temporary blocks (commonly a day or more).
- Colab is therefore the fallback for smoke runs, not the plan: Kaggle
  publishes its allowance, Colab does not. Re-check Colab's FAQ at run
  time (https://research.google.com/colaboratory/faq.html) before
  relying on it.

## 1. What the full run needs (sizes — honest figures)

| Item | Size / count | Basis |
| --- | --- | --- |
| SLR54 full audio (16 FLAC archives) | **~19 GB** | 16 archives × ~1.2 GB each (`training/README.md` §1) |
| SLR54 TSV (`utt_spk_text.tsv`) | ~10 MB | `benchmark/README.md` |
| SLR54 train clips | **~144,486** utterances | 157,905 total − 13,419 holdout (`benchmark/manifests/v0.json`) |
| SLR54 train audio | **~151 h (ESTIMATE)** | Total corpus ~165 h (Phase 3 verification) × train share 144,486/157,905. The TSV carries no durations, so this is a proportion estimate, not a measurement |
| FLEURS ne_np train split | 3,332 clips, **~10.5 h (ESTIMATE)** | Test split is 726 clips = 2.284 h (`benchmark/manifests/v0.json`); train hours scaled by clip count at the same mean clip length (~11.3 s) |
| FLEURS train parquet download | order of a few hundred MB (TO VERIFY at download) | Not measured; the test split is 2.28 h of audio |
| **Total train audio** | **~160 h (ESTIMATE)** | ~151 h SLR54 + ~10.5 h FLEURS |
| Whisper large-v3-turbo weights | ~3.2 GB (TO VERIFY at download) | 809M parameters × 4 bytes (fp32); derived, not measured |
| XLS-R 300M weights | ~1.2 GB (TO VERIFY at download) | 300M parameters × 4 bytes (fp32); derived, not measured |
| Whisper LoRA adapter (r=8, q/v only) | a few MB | LoRA trains a small fraction of the base model |
| Trainer checkpoints | grows without bound unless managed | Config `save_steps: 100` with **no `save_total_limit`** — see §5 |

**Disk warning — plan around this before starting.** `/kaggle/working`
holds at most 20 GB of saved output. The full SLR54 audio alone is
~19 GB, so audio + base model + checkpoints do **not** fit in
`/kaggle/working` together. Options, in order of preference:

1. Attach the audio as a **Kaggle Dataset** (input data, separate from
   the 20 GB working cap) — upload the prepared train manifest + audio
   once from a local machine, then attach it read-only to the training
   notebook. This is the recommended path.
2. Download only the SLR54 archives the run needs (the smoke path in
   §3 uses `--archives 2`), and accept a smaller train set.
3. Train on a subset (`--max-clips`) — honest about it in the results;
   a subset run is not the full run.

## 2. Setup (Kaggle notebook, GPU T4 x2)

1. Create the notebook (New Notebook), set **Accelerator: GPU T4 x2**,
   turn **Internet ON** (the data download needs it).
2. Clone the repo into `/kaggle/working` (or upload it):
   ```bash
   git clone https://github.com/Samir-Khadka/nepali-speech.git
   cd nepali-speech
   git log --oneline -1   # record the commit in the run notes
   ```
3. Check the GPU and the preinstalled torch — Kaggle's image already
   ships CUDA torch; verify it before installing anything:
   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   nvidia-smi
   ```
4. Install the training deps **without the CPU-torch line**.
   `training/requirements.txt` pins `torch==2.15.0+cpu`, which is only
   satisfiable from the PyTorch CPU index — installing that file as-is
   on a GPU would either fail or replace the CUDA torch with a CPU
   build. So install everything **except** the torch line, keeping
   Kaggle's preinstalled CUDA torch:
   ```bash
   grep -v '^torch==' training/requirements.txt > /tmp/requirements-gpu.txt
   pip install -r /tmp/requirements-gpu.txt
   python -c "import torch, transformers, peft; print(torch.__version__, torch.cuda.is_available())"
   ```
   If the preinstalled torch version is incompatible with the pinned
   `transformers==5.23.0` / `peft==0.21.3` / `accelerate==1.16.0`,
   install a matching CUDA torch from the PyTorch CUDA index instead —
   never the CPU index on a GPU machine:
   ```bash
   # ONLY if the version check above fails:
   pip install --index-url https://download.pytorch.org/whl/cu126 torch
   ```
   (Pick the CUDA index matching the installed CUDA toolkit; check
   `nvidia-smi` first. Do not guess the cu* version.)
5. Sanity-check the pipeline before any download:
   ```bash
   python training/train_whisper_lora.py --dry-run \
       --train-manifest training/train-manifest.jsonl \
       --output-dir /kaggle/working/whisper-lora-v1
   ```
   This will fail on the missing manifest — that is expected at this
   point; it proves the config loads and the CLI parses. (A `--dry-run`
   against a real manifest is step §3's last check.)

## 3. Data download (full run)

```bash
python training/prepare_data.py \
    --out training/train-manifest.jsonl --tmp-dir /kaggle/working/tmp-train
```

- Downloads the SLR54 TSV (~10 MB) and the FLEURS train/test Parquet
  metadata, builds the train manifest (SLR54 non-holdout speakers +
  FLEURS official train split), and **fails without writing anything**
  if any train row overlaps the v0 holdout (speaker overlap for SLR54,
  clip overlap for FLEURS). Expected output: ~147,800 clips
  (~144,486 SLR54 + 3,332 FLEURS). Check the printed summary and the
  `.summary.json` next to the manifest.
- Then materialise the audio (only needed if the audio is not already
  attached as a Kaggle Dataset — see §1):
  ```bash
  python training/prepare_data.py \
      --out training/train-manifest.jsonl --tmp-dir /kaggle/working/tmp-train \
      --audio-dir /kaggle/working/train-audio
  ```
  This downloads all 16 SLR54 archives (**~19 GB**) plus the FLEURS
  train audio. Budget download time: at a typical 50–100 MB/s this is
  roughly 5–10 minutes for the archives alone; the FLEURS extraction
  is additional. Watch the 20 GB `/kaggle/working` cap (§1) — if the
  audio is downloaded here, the model + checkpoints will not also fit.
- Verify disjointness independently afterwards:
  ```bash
  python benchmark/evaluate.py --check-splits \
      --train-manifest training/train-manifest.jsonl \
      --test-manifest benchmark/manifests/v0.jsonl
  ```
  (Known harness quirk: this CLI treats speaker `"unknown"` literally,
  so it reports an overlap for any manifest containing FLEURS rows.
  The clip-level check inside `prepare_data.py` is authoritative for
  FLEURS; the CLI check is meaningful for the SLR54 speakers —
  `training/README.md` §2.)

**Smoke first (do this before the full run, in the same notebook):**
```bash
python training/train_whisper_lora.py --dry-run \
    --train-manifest training/train-manifest.jsonl \
    --output-dir /kaggle/working/whisper-lora-v1
python training/train_whisper_lora.py \
    --train-manifest training/train-manifest.jsonl \
    --output-dir /kaggle/working/whisper-lora-smoke \
    --audio-dir /kaggle/working/train-audio \
    --model openai/whisper-tiny --max-steps 20 --batch-size 4 --bf16
```
The 20-step smoke run measures the real seconds-per-step on the T4.
**Record it** — it is the only honest basis for the full-run estimate
(§6). If the smoke run fails, fix it here, not mid-way through a
12-hour session.

## 4. The two training commands (full run)

Both use the shipped configs (`training/configs/whisper_lora.yaml`,
`training/configs/ctc.yaml`): LoRA r=8 / alpha=16 / dropout=0.05 on
`q_proj`+`v_proj`, lr 1e-4, batch 4, 3 epochs, warmup 0.03,
`save_steps: 100`. `--max-steps 0` (the default) means "use
`num_train_epochs` from the config" — so the commands below run the
full 3 epochs unless `--max-steps` caps them.

### 4a. Whisper LoRA (large-v3-turbo) — the benchmark model

```bash
python training/train_whisper_lora.py \
    --train-manifest training/train-manifest.jsonl \
    --output-dir /kaggle/working/whisper-lora-v1 \
    --audio-dir /kaggle/working/train-audio \
    --model openai/whisper-large-v3-turbo \
    --batch-size 4 --bf16 --gradient-checkpointing
```

- `--bf16`: T4 supports bf16; mixed precision roughly halves memory
  and speeds up the matmuls. `--gradient-checkpointing` trades ~30%
  speed for much lower activation memory — needed to fit batch 4 of
  30-second mel frames on a 16 GB T4. (Both figures are the usual
  rules of thumb for this hardware, not measurements — the smoke run
  in §3 measures the real combination.)
- The base model downloads from the Hugging Face Hub (~3.2 GB, §1).
  No account is needed for `openai/whisper-large-v3-turbo` (public).
- Output: the LoRA adapter (`adapter_model.*`), the processor, and
  `train_config.json` (exact config + manifest SHA-256 + steps trained).

### 4b. CTC (XLS-R 300M) — the deployment-relevant model

```bash
python training/train_ctc.py \
    --train-manifest training/train-manifest.jsonl \
    --output-dir /kaggle/working/ctc-v1 \
    --audio-dir /kaggle/working/train-audio \
    --batch-size 4 --bf16 --gradient-checkpointing
```

- Default model is `facebook/wav2vec2-xls-r-300m` (public, no account).
- Alternative: `ai4bharat/indicwav2vec_v1_nepali` — the 2026 paper's
  94M model family — is **gated** on the Hub: whoever runs this must
  have a Hugging Face account that has accepted its terms, and must
  first verify that the Hub config is the 94M checkpoint the paper
  used (unverified as of 2026-11-18; see `training/README.md` §1).
  Do not use `ai4bharat/indicwav2vec-hindi` for Nepali.
- Output: the fine-tuned model + processor + `train_config.json`.

Run the two models in **separate notebooks / separate weeks if needed**
— each is a multi-session job on the free quota (§6). Do not run both
in one 12-hour session.

## 5. Checkpoint saving

- Both scripts save a Trainer checkpoint every 100 steps
  (`save_steps: 100` in the configs) into `--output-dir` as
  `checkpoint-<step>/` (model/adapter + optimizer state, so a resume
  is possible in principle — see §7).
- **No `save_total_limit` is set**, so checkpoints accumulate
  indefinitely. At ~111k steps for the full Whisper run that would be
  ~1,100 checkpoint dirs — far beyond the 20 GB working cap. Mitigate:
  raise `save_steps` for the full run (e.g. edit a copy of the config
  to `save_steps: 1000` and record the change in the run notes), and/or
  delete older `checkpoint-*` dirs once a newer one is verified saved.
  Keep the final adapter/model + `train_config.json` — that is the
  deliverable, not the intermediate optimizer states.
- **After every session, download the output dir to a local machine**
  (Kaggle: Output tab → download, or attach the notebook's output as a
  dataset input to the next notebook). `/kaggle/working` is per-notebook
  versioned output; a new notebook does not automatically see the old
  one's files unless you attach it.

## 6. Expected GPU-hours — ESTIMATES with stated basis

**These are estimates, not measurements.** The honest basis is the
arithmetic below plus one measured number (seconds per step) that only
the §3 smoke run provides. Recompute the estimate from the smoke run's
measured step time before committing a week of quota.

Basis (all from §1, not guessed):
- Train clips: ~147,800 (~144,486 SLR54 + 3,332 FLEURS).
- Batch size 4, 3 epochs → **~110,850 optimizer steps**
  (147,800 ÷ 4 × 3).
- Train audio: ~160 h (estimate, §1).

| Run | Estimate | Arithmetic (basis) |
| --- | --- | --- |
| Whisper LoRA, large-v3-turbo, 3 epochs, batch 4 | **~30–60 GPU-hours** | 110,850 steps × 1 s/step ≈ 31 h; × 2 s/step ≈ 62 h. Step time is the unknown — T4, bf16, grad checkpointing, 30 s mel frames, autoregressive decoder |
| CTC XLS-R 300M, 3 epochs, batch 4 | **~10–25 GPU-hours** | 110,850 steps × 0.3–0.8 s/step. Encoder-only (no autoregressive decoder), so each step is typically faster than Whisper's; the 0.3–0.8 s range is an assumption, not a measurement |

What the estimates mean against the quota:
- Kaggle gives 30 GPU-hours/week, sessions capped at 12 h. The Whisper
  run therefore needs **1–2 quota weeks and at least 3–5 sessions**;
  the CTC run fits in **1 week / 1–2 sessions** if the low end holds.
- If the smoke run measures >2 s/step for Whisper, the full 3-epoch run
  exceeds 60 GPU-hours: the honest options are (a) spread it over more
  weeks, (b) cap with `--max-steps` (e.g. 50,000 steps ≈ 1.35 epochs at
  batch 4) and report the cap, or (c) reduce epochs in a config copy
  and record it. Never silently report a capped run as a 3-epoch run —
  `train_config.json` records the truth.
- The CPU smoke fine-tune running locally (Worker B, Nov 2026) is not a
  basis for GPU estimates: CPU step time does not transfer to a T4.

## 7. If a session dies mid-run

Kaggle sessions end at the 12-hour cap, on idle timeout, or on
preemption. Plan for it; do not hope it won't happen.

1. **Checkpoints exist.** Every 100 steps (config `save_steps`) the
   Trainer wrote `checkpoint-<step>/` into `--output-dir`, including
   optimizer state. The latest checkpoint dir is the resume point.
   Download the output dir first (§5) — a checkpoint left only in a
   dead session's `/kaggle/working` is recoverable only via that
   notebook's versioned output.
2. **Resume is NOT wired to a CLI flag (known gap, 2026-11-18).**
   Neither `train_whisper_lora.py` nor `train_ctc.py` accepts a
   `--resume` / `--resume-from-checkpoint` argument, and both call
   `trainer.train()` with no arguments. To resume you must either:
   - (a) make a small code change — pass
     `resume_from_checkpoint="<path>/checkpoint-<step>"` to
     `trainer.train()` — and record the change, or
   - (b) restart the run from scratch with the remaining budget
     (`--max-steps` set to the steps still needed) and record that the
     run is a restart, not a continuation.
   Do not claim a restarted run "resumed from checkpoint N" unless
   option (a) was actually used.
3. **If the run died before any checkpoint** (first 100 steps), just
   re-run: nothing was lost except the smoke-level time.
4. **After any recovery**, re-run `--dry-run` against the manifest
   before the real command, and confirm the output dir still contains
   `train_config.json` from the earlier attempt (or note its absence).
5. **Two environment gotchas, both found on the CPU smoke run
   (2026-11-18) — no action needed, recorded so a re-occurrence is
   recognised, not re-diagnosed.**
   - Model download dying on the HF xet client (`cas-server.xethub`,
     "File reconstruction error: CAS Client Error") while plain HTTPS
     to huggingface.co works: set `HF_HUB_DISABLE_XET=1` and retry —
     the weights then download over the plain CDN. (Seen behind a
     proxy; Kaggle's direct internet may never hit it.)
   - `TypeError: ... got multiple values for keyword argument
     'input_ids'` (then `'inputs_embeds'`) on the first training
     step: peft 0.21's `PeftModelForSeq2SeqLM.forward` injects
     `input_ids=None` / `inputs_embeds=None` and transformers 5.x
     Whisper forwards unknown `**kwargs` to the decoder. Fixed in
     code — `training/common.py::apply_peft_whisper_input_ids_fix`,
     applied by both `train_whisper_lora.py` and
     `score_checkpoint.py`, pinned by
     `training/tests/test_peft_whisper_forward.py`. If a future
     peft/transformers upgrade changes the symptom, that helper is
     the place to look first. The CTC script does not use peft and is
     unaffected.

## 8. Scoring the result on the v0 benchmark

Score the finished checkpoint against the **full** v0 holdout
(14,145 clips) — a `--limit` smoke score is a pipeline check, not a
benchmark result.

```bash
# Audio for the v0 holdout must be materialised first (same prepare_data
# / benchmark build flow; see benchmark/README.md). Then:
python training/score_checkpoint.py \
    --checkpoint /kaggle/working/whisper-lora-v1 \
    --base-model openai/whisper-large-v3-turbo \
    --manifest benchmark/manifests/v0.jsonl \
    --audio-dir /kaggle/working/v0-audio \
    --out /kaggle/working/whisper-lora-v1-results.json
```

```bash
python training/score_checkpoint.py \
    --checkpoint /kaggle/working/ctc-v1 \
    --manifest benchmark/manifests/v0.jsonl \
    --audio-dir /kaggle/working/v0-audio \
    --out /kaggle/working/ctc-v1-results.json
```

- `--model-type` is auto-detected (`adapter_config.json` → Whisper
  LoRA; Wav2Vec2 `config.json` → CTC). Pass it explicitly only if
  detection fails.
- The results JSON carries the exact config (checkpoint, manifest +
  SHA-256, clip count, steps trained, LoRA/CTC settings) plus the
  harness's overall + per-slice WER/CER. Report those numbers, overall
  AND per slice (source / gender / quality), never a single average.
- **Comparison bar (theirs, not ours):** Paudel & Sayami (2026) —
  Whisper-Large-v3-Turbo 14.76% WER in-domain (OpenSLR) / 39.56%
  FLEURS / 48.35% Common Voice; IndicWav2Vec 14.89% / 36.34% / 42.16%.
  A checkpoint is only comparable to those numbers when scored on the
  full v0 holdout through `score_checkpoint.py`. Do not claim to beat
  the baseline until a real run does, on the full holdout.
- **Numerals must match training.** If training used
  `--convert-devanagari-numerals`, scoring must pass the same flag;
  the default (OFF) on both sides is the protocol. The flag value is
  recorded in `train_config.json` and in the results JSON.

## 9. Run log (fill in during the run)

- Date / Kaggle account quota remaining at start:
- Repo commit:
- Notebook URL (private link is fine; record it, don't publish it):
- Accelerator (T4 x2 / P100):
- Measured seconds/step (smoke, Whisper / CTC):
- Recomputed full-run estimate from the measured step time:
- Sessions used (start/end step each session):
- Checkpoints downloaded (where):
- Final `train_config.json` + results JSON locations:
- Deviations from this runbook (config edits, `--max-steps` caps,
  restarts, resume code change):
