# Deploy — Render (fallback, from 1 Dec 2026)

Fallback for when Render's free tier resets. **Do not deploy this before
1 Dec 2026**: all free web services in the workspace are suspended until
then (750 free instance hours/month exhausted). The live target for Phase 2
is PythonAnywhere — see `docs/deploy-pythonanywhere.md`.

Expected URLs (service names must be unique per Render account — if a name
is taken, rename it in `deploy/render.yaml` and update the URLs):

```
API:      https://nepali-speech-api.onrender.com
Frontend: https://nepali-speech-web.onrender.com
```

## What the blueprint creates

`deploy/render.yaml` (a Render Blueprint) defines two free services:

| Service | Type | Why separate |
| --- | --- | --- |
| `nepali-speech-api` | Web service (Python 3.12) | FastAPI via `uvicorn backend.app.main:app` — the module path from the repo root. No a2wsgi needed: Render runs ASGI natively. |
| `nepali-speech-web` | Static site | The React PWA build (`frontend/dist`). Static sites are free and do **not** consume the 750 free instance hours, so the frontend stays up even when the API sleeps. |

The frontend is a separate static site (not served from the API service)
because the backend is an API-only FastAPI app: serving the SPA from the
same service would require changing `backend/app/main.py`, which Phase 2
deliberately avoids. The trade-off is that the frontend build bakes in the
API URL at build time (`VITE_API_URL` env var on the static site) — if the
API service is renamed, update the value in `render.yaml` and redeploy the
static site.

Environment variables (already in the blueprint):

- API service: `DATA_DIR=/opt/render/project/src/data`,
  `ADMIN_EMAILS=samirshirjan890@gmail.com`. `DATABASE_URL` is intentionally
  unset → SQLite file under `DATA_DIR`.
- Static site: `VITE_API_URL=https://nepali-speech-api.onrender.com`.

## Deploy steps (on or after 1 Dec 2026)

1. Render dashboard → **New +** → **Blueprint** → connect the
   `Samir-Khadka/nepali-speech` repo → select `deploy/render.yaml` →
   **Apply**. (Or: **New + → Web Service** / **Static Site** manually with
   the same build/start commands and env vars from the blueprint.)
2. Wait for both services to build. The API's first start creates the
   SQLite database and seeds the 200 prompts (lifespan startup).
3. Smoke test: `https://nepali-speech-api.onrender.com/health` →
   `{"status": "ok"}`. Then open the frontend URL and record a test
   sentence end-to-end.
4. If the API service name had to change, update `VITE_API_URL` in
   `render.yaml` (static site env vars) and redeploy the static site.

## Two hard limits on the free tier — read before collecting on Render

### 1. No persistent disk: data is ephemeral

Free Render services have **no persistent disk** (disks are a paid
feature). The SQLite database and all audio files live on the ephemeral
filesystem and are **lost on every redeploy and every service restart**.
That makes free Render unsuitable as the durable home for Phase 2
collection data — it is a reachability/demo fallback, and PythonAnywhere
(persistent home directory) stays the live target. If collection ever
moves to Render, add a paid persistent disk and point `DATA_DIR` at its
mount path, or switch `DATABASE_URL` to a hosted Postgres (e.g. Neon free
tier) and move audio to object storage.

### 2. No 5-minute uptime pings — this is what burned the free hours

**Do NOT point UptimeRobot (or any monitor) at these services on a
5-minute interval.** The 750 free instance hours/month are shared across
the whole workspace, and 5-minute pings keep every free web service awake
24/7 — that is the likely cause of the workspace-wide suspension that
runs until 1 Dec 2026. On the free tier, let the API sleep after 15
minutes idle and spin up on the first real request (cold start ~30–60s);
that is the intended trade-off. If uptime monitoring is wanted, check at
most once an hour, or monitor the static site only (static sites don't
consume instance hours).

## Redeploying after a code change

Push to `main` → Render auto-deploys both services (if auto-deploy is on)
or click **Manual Deploy** on each service. Note the data warning above:
a redeploy of the API wipes the database and recordings on the free tier.
