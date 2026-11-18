# nepali-speech — Phase 1 API contract (binding for backend, frontend, bot)

Base URL (dev): `http://localhost:8000`. Frontend reads it from `VITE_API_URL`, default `http://localhost:8000`. Bot reads it from `API_BASE_URL`, default `http://localhost:8000`.

## Auth
- Contributors: no account. Frontend generates a UUID contributor id, stores it in localStorage under `ns-contributor-id`, sends it on every call.
- Admin: endpoints under `/api/admin/*` require header `X-Admin-Email` whose value is in the `ADMIN_EMAILS` env var (comma-separated). Anything else → 403. Documented as a dev-phase mechanism; production needs real auth.

## Endpoints

### `GET /health` → `{"status": "ok"}`

### `GET /api/prompts/next?contributor_id=<uuid>`
Returns the next prompt this contributor has not recorded yet (any order, deterministic enough for tests):
`{"prompt_id": 12, "text": "...", "domain": "farming", "language": "ne"}`
When the contributor has recorded every prompt: `{"prompt_id": null, "done": true, "total": 200}`.
`language` is `"ne"` or `"ne-en"` (code-switched).

### `GET /api/prompts/count` → `{"total": 200}`

### `POST /api/consent`
JSON: `{"contributor_id": "<uuid>", "version": "v1", "province": "<str|null>", "age_range": "<str|null>", "gender": "<str|null>"}` — metadata fields all optional, may be null/absent.
Upserts the contributor + consent record. → `{"ok": true, "contributor_id": "<uuid>"}`.

### `POST /api/recordings` (multipart/form-data)
Fields: `prompt_id` (int), `contributor_id` (str), `audio` (file).
Validation, in order:
1. Contributor must have a consent record → else 403 `{"detail": "consent required"}`.
2. Prompt must exist → else 404.
3. File size ≤ 5 MB → else 413.
4. Content-Type must be one of `audio/webm`, `audio/ogg`, `audio/mp4`, `audio/mpeg`, `audio/wav`, `video/webm` → else 400.
5. Magic bytes must match the claimed type: WebM `1A 45 DF A3`, Ogg `OggS`, WAV `RIFF`, MP4 `ftyp` at offset 4 → else 400.
On success stores the file at `data/audio/<recording_id>.<ext>` and returns 201:
`{"recording_id": 7, "status": "pending_review"}`.

### `GET /api/recordings/mine?contributor_id=<uuid>`
`{"recorded": 3, "approved": 1, "pending": 2, "rejected": 0, "recordings": [{"recording_id": 7, "prompt_text": "...", "status": "pending_review"}]}`

### `GET /api/review/next?reviewer_id=<str>`
Next pending recording this reviewer has not reviewed yet (recordings with 0 reviews first):
`{"recording_id": 7, "prompt_text": "...", "domain": "farming", "language": "ne", "audio_url": "/api/review/audio/7"}`
When the queue is empty for this reviewer: `{"empty": true, "message": "..."}`.

### `GET /api/review/audio/<recording_id>` → audio bytes, correct Content-Type, 404 if unknown.

### `POST /api/review/<recording_id>`
JSON: `{"reviewer_id": "<str>", "decision": "approve"|"reject", "reason": "<str|null>"}` (reason required when rejecting → 400 otherwise).
Review rules:
- First approval → status `approved`, UNLESS the recording is flagged `double_review` (flag ~20% of recordings at creation) → status `needs_second_review`.
- Second approval on a `needs_second_review` recording → `approved`.
- Any reject → `rejected` (reason stored).
- A reviewer may not review the same recording twice → 400.
→ `{"ok": true, "status": "<new status>"}`.

### `GET /api/admin/stats` (admin header required)
`{"prompts": 200, "contributors": 12, "recordings_total": 45, "by_status": {"pending_review": 20, "approved": 20, "rejected": 3, "needs_second_review": 2}, "reviews_total": 48, "double_reviewed": 5, "agreement_rate": 0.8, "by_domain": {"farming": 15, ...}}`
`agreement_rate` is the fraction of double-reviewed recordings where both reviewers agreed, or `null` when none completed.

## Data model
- **Prompt**: id (int), text, domain, language (`ne`|`ne-en`), source (`original`).
- **Contributor**: id (TEXT uuid, pk), consent_version, consented_at, province, age_range, gender (all nullable except id/consent).
- **Recording**: id, prompt_id (FK), contributor_id (FK), file_path, mime, size_bytes, status (`pending_review`|`approved`|`rejected`|`needs_second_review`), double_review (bool), created_at.
- **Review**: id, recording_id (FK), reviewer_id, decision, reason, created_at.

## Storage
- `backend/app/storage.py` exposes `save_audio(data: bytes, recording_id: int, ext: str) -> str` and `read_audio(recording_id: int) -> tuple[bytes, str]`.
- Data root: `DATA_DIR` env, default `<project root>/data`. Audio under `data/audio/<id>.<ext>`. SQLite file `data/nepali-speech.db` by default; `DATABASE_URL` env overrides (Postgres-ready via SQLAlchemy URL).
- Prompts are seeded from `prompts/prompts.json` on startup **only when the prompts table is empty**. That file is an array of `{"id": 1, "text": "...", "domain": "farming", "language": "ne", "source": "original"}`.

## Frontend routes (HashRouter)
- `/` — consent screen (Nepali) → recording page (one prompt, record/playback/re-record/submit, progress).
- `/review` — review queue UI (reviewer id in localStorage, prompt text, audio player, approve/reject + reason, skip).
- `/stats` — admin stats (asks for admin email, sends `X-Admin-Email`).

## Bot (python-telegram-bot)
Talks to this API over HTTP only (no direct DB access). `/start` → consent text + first prompt. Voice message → download → `POST /api/recordings`. Token from `TELEGRAM_BOT_TOKEN` env; missing → clear error, exit 1.
