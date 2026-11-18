# nepali-speech — backend (Phase 1: collection tool)

FastAPI + SQLAlchemy API implementing `../CONTRACT.md` exactly:
prompt serving, consent, recording upload (with magic-byte validation),
review queue with double review, and admin stats.

## Run

```bash
python -m venv ../.venv
../.venv/bin/pip install -r requirements.txt
cp .env.example .env   # optional; defaults work for local dev
../.venv/bin/uvicorn app.main:app --reload --port 8000
```

Health check: `curl http://localhost:8000/health` → `{"status": "ok"}`.

Prompts are seeded from `../prompts/prompts.json` on startup, only when
the prompts table is empty.

## Environment variables

| Var | Default | Meaning |
| --- | --- | --- |
| `DATABASE_URL` | `sqlite:///<DATA_DIR>/nepali-speech.db` | SQLAlchemy URL (Postgres-ready) |
| `DATA_DIR` | `<project root>/data` | Audio stored under `<DATA_DIR>/audio` |
| `ADMIN_EMAILS` | _(empty)_ | Comma-separated emails allowed on `/api/admin/*` via `X-Admin-Email` |
| `PORT` | `8000` | Dev server port |
| `MAX_UPLOAD_BYTES` | `5242880` (5 MB) | Upload size cap (override in tests) |
| `DOUBLE_REVIEW_RATE` | `0.2` | Fraction of recordings flagged `double_review` at creation |

## Tests

```bash
../.venv/bin/pip install -r requirements.txt -r requirements-test.txt
../.venv/bin/pytest -q
```

Tests use synthetic audio bytes only (correct magic headers, random
payload) — no real audio files. Note `httpx2` in `requirements-test.txt`
is required: starlette >= 1.10's `TestClient` needs it.
