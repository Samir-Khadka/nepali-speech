# Kaggle GPU runbook — full baseline on the v0 held-out test set

**Purpose:** run the full baseline (Whisper / MMS / XLS-R family) on the
complete v0 test set on a **free Kaggle GPU**. This is documentation only —
no Kaggle account was created by the agent, and nothing here spends money.

**Status (2026-11-18):** the v0 manifest (`benchmark/manifests/v0.jsonl`)
does not exist yet — Worker 2's manifest build is a separate step. Do not
run this runbook until the manifest exists and its clip count is known.
The smoke eval (`SMOKE_EVAL.md`) proves the harness + Whisper-tiny
pipeline on 5 real FLEURS clips; it is not a baseline.

## 0. What you need before you start

- A Kaggle account (free; Samir creates it — the agent cannot).
- The repo cloned locally or on Kaggle: `https://github.com/Samir-Khadka/nepali-speech`
- The v0 manifest: `benchmark/manifests/v0.jsonl` (schema in
  `benchmark/evaluate.py` and `benchmark/README.md`).
- The v0 audio: download per the manifest's `source` (see §1). **Do not
  commit audio to the repo** — the harness reads it from a local folder.

## 1. Get the v0 audio onto Kaggle

Audio is **not** in the repo (`.gitignore` blocks `*.wav`/`*.mp3`).
The v0 set is **14,145 clips** (real count from the 2026-11-18 build —
see `benchmark/manifests/README.md`):

- **OpenSLR SLR54: 13,419 clips** (speaker-disjoint holdout, 49 test
  speakers) — the bulk of the set. Fetch with the repo builder:
  `python benchmark/build.py materialise --manifest benchmark/manifests/v0.json --out-dir /path/to/v0-audio`
  (downloads the SLR54 ZIP archives holding the test clips, ~9 GB total
  if you take everything; source https://www.openslr.org/54, CC BY-SA 4.0).
- **FLEURS `ne_np` test: 726 clips, 2.28 h** — same `materialise`
  command (FLEURS test Parquet ~526 MB; CC BY 4.0).
- **Common Voice ne: 0 clips in v0 — gated.** Since October 2025 it is
  distributed only through the Mozilla Data Collective; no account was
  created for the build, so the slot stays empty until a maintainer
  with Data Collective access exports ne-NP. Do not point users at the
  old (removed) Hugging Face mirrors.

Upload the materialised audio as a private Kaggle Dataset, or fetch it
inside the notebook with the `materialise` command above. **Do not
commit audio to the repo** — the harness reads it from a local folder.

## 2. Start a Kaggle notebook (free GPU)

1. kaggle.com → **Create → New Notebook**.
2. Right panel → **Accelerator → GPU T4 x2** (free tier; the default
   free GPU). Do **not** pick P100 unless T4 is unavailable — T4 is
   enough for every model below.
3. **Add data** (right panel): attach the private dataset(s) from §1
   and the repo (upload the repo as a private dataset, or
   `!git clone https://github.com/Samir-Khadka/nepali-speech` in a cell —
   cloning is simpler and always current).
4. Free-tier limits to respect: **30 h GPU per week** (quota resets
   weekly), 9 h max per notebook session, 20 GB disk. The full run
   below is ~17–18 GPU-hours, so it needs **three sessions** (§4) —
   never try it in one.

## 3. Notebook outline (paste cell by cell)

```python
# Cell 1 — install (CPU-free, GPU notebook; ~3–5 min)
!pip install -q openai-whisper jiwer==4.0.0
!pip install -q transformers torchaudio --index-url https://download.pytorch.org/whl/cu121
# (transformers/torch are for the MMS / XLS-R / IndicWav2Vec runs, §4)

# Cell 2 — sanity: GPU present?
!nvidia-smi --query-gpu=name,memory.total --format=csv

# Cell 3 — clone repo + point at data
!git clone https://github.com/Samir-Khadka/nepali-speech
import os
MANIFEST = "nepali-speech/benchmark/manifests/v0.jsonl"
AUDIO_DIR = "/kaggle/input/<your-dataset-name>"   # <-- EDIT THIS

# Cell 4 — transcribe the whole v0 set with one model
# (uses the repo harness; --transcribe runs openai-whisper)
!python nepali-speech/benchmark/evaluate.py \
    --manifest $MANIFEST \
    --transcribe --model tiny \
    --audio-dir $AUDIO_DIR \
    --out /kaggle/working/tiny-results.json
```

The harness prints the overall + per-slice WER/CER and writes the full
JSON (overall, every slice, per-clip scores) to `--out`. **Keep that
JSON — it is the submission artefact** (see §5).

## 4. Which models to run, in which order

Run in this order; stop and record after each one so a session timeout
never loses everything. Times are rough estimates for the full v0 set
(**14,145 clips: 13,419 SLR54 + 726 FLEURS, ~17 h of audio** — the SLR54
part dominates; its exact hours are unknown because the SLR54 TSV has
no durations, so treat these as ±50%):

| # | Model | How | Est. time (T4) |
|---|---|---|---|
| 1 | Whisper-tiny (39 M) | `--transcribe --model tiny` | ~40 min (pipeline check at scale) |
| 2 | Whisper-base (74 M) | `--transcribe --model base` | ~1 h |
| 3 | Whisper-small (244 M) | `--transcribe --model small` | ~2 h |
| 4 | Whisper-medium (769 M) | `--transcribe --model medium` | ~4 h |
| 5 | Whisper-Large-v3-Turbo (809 M) | `--transcribe --model large-v3-turbo` | ~5 h |
| 6 | IndicWav2Vec (94 M, CTC) | transformers script (see below) | ~1 h + setup |
| 7 | MMS-1B (965 M, CTC) | transformers script | ~2 h |
| 8 | XLSR-53 (317 M, CTC) | transformers script | ~1.5 h |

Total: roughly **17–18 GPU-hours** across sessions — inside the free
30 h/week quota, but **not** inside one 9 h session. Split into three
sessions (e.g. models 1–3, then 4–5, then 6–8).

**Reference numbers to compare against** (Paudel & Sayami 2026,
fine-tuned on SLR54 — theirs, not ours): Whisper-Large-v3-Turbo
14.76% WER in-domain (OpenSLR), 39.56% FLEURS, 48.35% Common Voice;
IndicWav2Vec 14.89% / 36.34% / 42.16%. Our runs 1–5 are **zero-shot**
(no fine-tuning), so expect much worse than 14.76% in-domain — the
honest comparison is against their zero-shot figure (~94% WER
out-of-domain) and against each other. The fine-tuned comparison
needs a separate fine-tuning run (out of scope for this runbook).

**CTC models (6–8) — outline, not a drop-in script.** The harness
scores a hypotheses JSONL, so any transcription path works: transcribe
with a transformers pipeline, write `{"clip_id", "hypothesis"}` lines,
then score with `evaluate.py --hypotheses`:

```python
# Sketch — adapt checkpoint names to the HF repo you use
from transformers import pipeline
pipe = pipeline("automatic-speech-recognition",
                model="facebook/mms-1b-fl102",  # or the IndicWav2Vec / XLSR-53 checkpoint
                device=0)
# ... loop over manifest rows, pipe(audio_path)["text"], write JSONL ...
```

Then:

```bash
python nepali-speech/benchmark/evaluate.py \
    --manifest $MANIFEST --hypotheses /kaggle/working/mms-1b.jsonl \
    --out /kaggle/working/mms-1b-results.json
```

## 5. What to save after each run

For every model, download from `/kaggle/working/` and keep:

- `<model>-results.json` — the harness output (overall + slices +
  per-clip). This is the submission artefact.
- The exact command / notebook version + the manifest's git commit.
- Model checkpoint name + parameter count + what it was trained on.

**Report a model only with the full sliced output** (overall + gender +
region + quality + source). A single overall number is incomplete and
is not listed on the leaderboard (`benchmark/leaderboard.md`).

## 6. If something goes wrong

- **Session times out (9 h limit):** re-run from the next model in §4 —
  each model's `--out` JSON is independent, nothing is lost except the
  interrupted model.
- **Out of GPU quota (30 h/week):** wait for the weekly reset; do not
  create a second account to dodge the quota.
- **Out of disk (20 GB):** the SLR54 test audio is the likely cause —
  delete a model's audio after its run, or keep only one source's audio
  on disk at a time and run per-source.
- **A model OOMs on T4 (16 GB):** drop to the next smaller model, or
  for the CTC models pass shorter clips / smaller batches. Whisper
  large-v3-turbo fits on a T4; MMS-1B fits with batch size 1.
- **Hypothesis file has missing clips:** the harness scores them as
  empty and reports `missing_hypotheses` — check that count before
  trusting the number; a high count usually means the audio path in
  the manifest did not match `--audio-dir`.

## 7. After the run

1. Paste each `<model>-results.json` summary into
   `benchmark/leaderboard.md` (overall + every slice, with model,
   checkpoint, training data, date).
2. Commit the leaderboard update to the repo (results JSON only —
   **never the audio**).
3. Note the date + manifest commit in the leaderboard row so the
   number stays reproducible.
