# Nepali Speech — Telegram Bot (Phase 1: collection only)

Collects Nepali voice recordings over Telegram and uploads them to the
backend API (`POST /api/recordings`). Talks to the API over HTTP only —
no direct database access.

## Flow

1. User sends `/start` → the bot shows a short Nepali consent summary:
   recordings are released CC0, no personal data is attached, and the user
   can stop any time.
2. User taps **“✅ म सहमत छु”** → the bot posts consent
   (`POST /api/consent`, contributor id `tg:<telegram user id>`, version `v1`)
   and sends the first prompt (`GET /api/prompts/next`).
3. User sends a voice message (Telegram voice = Ogg/Opus) → the bot
   downloads it, uploads it (`POST /api/recordings` multipart), confirms,
   and sends the next prompt.
4. A voice message before consent → the bot asks the user to run `/start` first.
5. `/help` → short Nepali/English help.
6. If the API is unreachable, the bot replies in Nepali:
   “सर्भर अहिले उपलब्ध छैन, कृपया पछि प्रयास गर्नुहोस्।”

## Run

```bash
# from the project root (~/workspace/nepali-speech)
python3 -m venv .venv          # only if it does not exist yet
source .venv/bin/activate
pip install -r bot/requirements.txt

export TELEGRAM_BOT_TOKEN=<token from @BotFather>
export API_BASE_URL=http://localhost:8000   # optional, this is the default

python -m main        # run from inside bot/
# or
python bot/main.py    # run from the project root
```

Without `TELEGRAM_BOT_TOKEN` the bot prints a clear error naming the
variable and exits with code 1 — it never starts without a token.

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `TELEGRAM_BOT_TOKEN` | yes | — | Bot token from @BotFather |
| `API_BASE_URL` | no | `http://localhost:8000` | Backend API base URL |

## BotFather steps (Samir — your step)

1. Open Telegram and talk to **@BotFather**.
2. Send `/newbot`, pick a name and a username (must end in `bot`,
   e.g. `nepali_speech_bot`).
3. BotFather replies with a token that looks like `123456:ABC-DEF...`.
4. Copy that token into the `TELEGRAM_BOT_TOKEN` environment variable
   where you run the bot (see above).
5. **Never paste the token in chat, and never put it in any file.**
   If a token ever leaks, revoke it with `/revoke` in @BotFather.

## Tests

```bash
# from bot/ (needs the venv + requirements installed; no token, no network)
python -m pytest test_bot.py -v
```

Tests cover only the pure helpers (contributor-id formatting, consent
payload, prompt message formatting) and the API client with mocked HTTP.
