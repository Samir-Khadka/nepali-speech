# Nepali Speech — an open Nepali speech dataset

An open, crowd-sourced Nepali speech dataset, built in the open so anyone can
use it to build better Nepali speech technology.

People record short Nepali sentences (read from prompts) on the web app or
over Telegram. Recordings go through a human review queue, and the approved
recordings become a freely usable dataset. Later phases add a public
benchmark and a fine-tuned Nepali ASR model.

**Status: Phase 1 — collection tool. No dataset has been published yet, and
no audio has been collected yet.**

Repository: https://github.com/Samir-Khadka/nepali-speech

## Why this exists

Nepali ASR today is trained almost entirely on clean, read speech. What is
missing is realistic speech: phone-quality recordings, background noise,
different dialects and accents, and the Nepali–English code-switching people
actually use every day. No existing benchmark covers that. This project
collects exactly that kind of data, in the open, under CC0.

## The verified baseline

Existing Nepali speech resources (verified figures):

| Dataset | Size |
| --- | --- |
| OpenSLR SLR54 | ~165 h, 527 speakers, read speech |
| FLEURS `ne_np` | ~10 h |
| Mozilla Common Voice `ne-NP` | ~1.5–5 h validated, 32–49 speakers |
| IndicVoices | 100 h train + 5 h held-out, 496 speakers |
| Meta Seamless | 153 h |

Almost all of it is read speech recorded in clean conditions.

2026 benchmark (Paudel & Sayami, TU MSc thesis,
https://p-sumann.github.io/nepali-asr-benchmark/):

- Whisper-Large-v3-Turbo fine-tuned: **14.76% WER in-domain**, but
  **39.56% WER on FLEURS** and **48.35% WER on Common Voice**.
- IndicWav2Vec (94M) ties it in-domain.
- CTC decoders are **29× faster** than attention decoders.
- Zero-shot Whisper on Nepali is ~**94% WER out-of-domain**.

In short: models look good on the data they were trained on and fall apart
on anything else. That gap — realistic, noisy, dialect, code-switched
Nepali — is what this dataset is for.

## Roadmap

- **Phase 0 — Research (done).** Surveyed existing datasets and benchmarks;
  the baseline above is the result.
- **Phase 1 — Collection tool (this repo, in progress).** Prompt set, web
  recording app, Telegram bot, review queue, admin stats.
- **Phase 2 — First 50 hours.** Collect and review the first 50 hours of
  approved recordings.
- **Phase 3 — Benchmark.** A public benchmark split covering noisy,
  phone-quality, dialect and code-switched Nepali.
- **Phase 4 — First model.** Fine-tune an open ASR model on the dataset and
  publish the results.
- **Phase 5 — Public release.** Publish the dataset (CC0) and the benchmark.
- **Phase 6 — Expansion.** More hours, more dialects, more domains.

## How it works

1. **Consent first.** A contributor reads the consent text
   (`prompts/consent-ne.md` / `prompts/consent-en.md`) and agrees. Consent
   version: `v1`. No account is needed; the frontend generates a UUID
   contributor id.
2. **Record.** The app (or the Telegram bot) shows one prompt at a time
   from `prompts/prompts.json`. The contributor reads it aloud and submits
   the recording. Audio is at most 5 MB per recording.
3. **Review.** Every recording enters a review queue. A reviewer listens
   and approves or rejects it (with a reason when rejecting). About 20% of
   recordings are flagged for a second, independent review.
4. **Release.** Approved recordings become part of the dataset, released
   under CC0 (see `DATASET_LICENSE.md`).

The full API contract is in [`CONTRACT.md`](CONTRACT.md).

## Prompts

`prompts/prompts.json` holds 200 original Nepali sentences, written for this
project — no text copied from any published source. They cover farming (40),
weather (25), prices (25), numbers (20), dates (15), addresses (10),
voice-assistant commands (30), daily life (25), and Nepali–English
code-switching (10). See [`prompts/README.md`](prompts/README.md) for how to
add more.

## Run locally

Prerequisites: Python 3.11+, Node.js 18+.

### Backend

```bash
cd backend
uvicorn app.main:app --reload   # http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend reads the API base URL from `VITE_API_URL`
(default `http://localhost:8000`).

### Telegram bot

```bash
cd bot
pip install -r requirements.txt
export TELEGRAM_BOT_TOKEN=<your-bot-token>
export API_BASE_URL=http://localhost:8000   # optional, this is the default
python bot/main.py
```

Without `TELEGRAM_BOT_TOKEN` the bot prints a clear error and exits.

## Environment variables

| Variable | Where | Default | Notes |
| --- | --- | --- | --- |
| `DATA_DIR` | backend | `<project root>/data` | Audio files and the SQLite database live here |
| `DATABASE_URL` | backend | SQLite file under `DATA_DIR` | SQLAlchemy URL; Postgres-ready |
| `ADMIN_EMAILS` | backend | — | Comma-separated admin emails; `/api/admin/*` requires an `X-Admin-Email` header matching one of these |
| `VITE_API_URL` | frontend | `http://localhost:8000` | Backend base URL |
| `TELEGRAM_BOT_TOKEN` | bot | — | **Required.** Bot token from BotFather |
| `API_BASE_URL` | bot | `http://localhost:8000` | Backend base URL |

No credentials are stored in this repo. Never commit a bot token or any
secret.

## Licenses

- **Code:** MIT — see [`LICENSE`](LICENSE).
- **Collected dataset:** CC0 1.0 Universal (public domain dedication) — see
  [`DATASET_LICENSE.md`](DATASET_LICENSE.md).
- **Prompt texts:** original sentences written for this project; released
  with the dataset under CC0.

## Contributing

Contributions are welcome — new prompts (see `prompts/README.md`), code,
reviewing, and recordings. Please keep prompts original: write them
yourself, do not copy text from books, news sites, Wikipedia, or any other
published source.
