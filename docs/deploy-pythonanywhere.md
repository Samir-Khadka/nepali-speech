# Deploy — PythonAnywhere (Phase 2 live target)

Live target for Phase 2 collection: **PythonAnywhere free tier** ("Beginner",
no card required). One Python web app, WSGI-only, Python 3.12, 512 MB disk.

Expected live URL (if the username below is free):

```
https://samirkhadka.pythonanywhere.com
```

If `samirkhadka` is taken, use the actual username everywhere below instead
of `samirkhadka` — the URL, the paths, and the `VITE_API_URL` build step are
all derived from it.

**What is deployed:** the whole collection tool as ONE web app. The WSGI
entrypoint `deploy/pythonanywhere_wsgi.py` serves the FastAPI backend
(`/health`, `/api/*`, via a2wsgi) and the built React frontend (`/`,
`/assets/*`, with an SPA fallback) from a single process. No outbound
network calls are made by the app, so PythonAnywhere's restricted outbound
internet on the free tier is irrelevant. The Telegram bot is NOT part of
this deploy (it needs a long-running process; it stays on a laptop/server).

**What this guide does NOT do:** create the account. That step needs a
human (or a browser task) — account creation and the email confirmation
click cannot be done from the sandbox.

## 0. Prerequisites

- A GitHub account is not needed (the repo is public).
- The Gmail account `samirshirjan890@gmail.com` (for the PythonAnywhere
  account and for `ADMIN_EMAILS`).
- This repo, already pushed to `main` with `deploy/pythonanywhere_wsgi.py`
  and `a2wsgi` in `backend/requirements.txt`.

## 1. Create the PythonAnywhere account

1. Go to https://www.pythonanywhere.com/ → **Pricing & signup** (or
   **Create a Beginner account**).
2. Choose the **Beginner** (free) plan.
3. Username: `samirkhadka`
4. Email: `samirshirjan890@gmail.com`
5. Set a password (store it in the password manager — never in the repo).
6. Open the confirmation email and click the confirmation link. Until the
   email is confirmed, the account cannot create a web app.

## 2. Clone the repo and build (Bash console)

On the PythonAnywhere dashboard, open **Consoles → Bash** (or "Start a new
console → Bash") and run:

```bash
cd ~
git clone https://github.com/Samir-Khadka/nepali-speech.git
cd nepali-speech

# Python 3.12 virtualenv (PythonAnywhere's virtualenvwrapper)
mkvirtualenv --python=/usr/bin/python3.12 nepali-speech
# If mkvirtualenv is unavailable, use instead:
#   python3.12 -m venv ~/.virtualenvs/nepali-speech
#   source ~/.virtualenvs/nepali-speech/bin/activate

pip install -r backend/requirements.txt
```

Verify the install (should print `1.10.10` and no error):

```bash
python -c "from a2wsgi import ASGIMiddleware; import a2wsgi; print(a2wsgi.__version__)"
```

Build the frontend. **The empty `VITE_API_URL=` matters**: it makes the
frontend call the API on the same origin (`/api/...`) instead of the
development default `http://localhost:8000`. (Verified: a build with
`VITE_API_URL=` contains no `localhost:8000` anywhere in its JS.)

```bash
cd ~/nepali-speech/frontend
npm ci
VITE_API_URL= npm run build
```

Check the build exists:

```bash
ls ~/nepali-speech/frontend/dist/index.html ~/nepali-speech/frontend/dist/assets/
```

## 3. Web tab settings

Dashboard → **Web** → **Add a new web app**:

1. Domain name: accept the default `samirkhadka.pythonanywhere.com`
   (free accounts get exactly one `<username>.pythonanywhere.com` app).
2. Select a Python Web framework: **Manual configuration**.
3. Select a Python version: **Python 3.12**.

Then, in the web app's settings:

### Code

- **WSGI configuration file:** click the file path link (it looks like
  `/var/www/samirkhadka_pythonanywhere_com_wsgi.py`). Delete its entire
  contents and paste in the full contents of
  `deploy/pythonanywhere_wsgi.py` from the repo. Save.
  - The file must expose the WSGI callable as `application` — it does.
  - The entrypoint finds the repo at `~/nepali-speech` automatically. If
    the repo was cloned somewhere else, set `NEPALI_SPEECH_ROOT` to that
    path in the environment variables below.

- **Virtualenv:** enter the full path to the venv created in step 2:
  `/home/samirkhadka/.virtualenvs/nepali-speech`
  (if you used `python3.12 -m venv`, the same path applies).

### Static files (optional but recommended)

The WSGI entrypoint already serves the frontend itself, so the app works
without this mapping — it only makes asset delivery faster (served by the
front-end server instead of Python). Add one entry:

| URL | Directory |
| --- | --- |
| `/` | `/home/samirkhadka/nepali-speech/frontend/dist/` |

This is safe for the API: PythonAnywhere serves a static file only when
one exists at that path, and passes everything else (e.g. `/api/...`,
`/health`) through to the WSGI app.

### Environment variables

Scroll to **Environment variables** on the Web tab and add:

| Variable | Value |
| --- | --- |
| `DATA_DIR` | `/home/samirkhadka/nepali-speech/data` |
| `ADMIN_EMAILS` | `samirshirjan890@gmail.com` |

Do NOT set `DATABASE_URL` — the default SQLite file under `DATA_DIR` is
what this deploy uses. These variables must be set before the first load:
the app creates its database and seeds the 200 prompts at import/startup.

## 4. Reload and smoke test

Click **Reload samirkhadka.pythonanywhere.com** at the top of the Web tab.
Wait a few seconds, then:

1. `https://samirkhadka.pythonanywhere.com/health` → `{"status": "ok"}`
2. `https://samirkhadka.pythonanywhere.com/` → the Nepali Speech Collector
   app loads (record page, "रेकर्ड गर्नुहोस्").
3. In the app: consent → a prompt appears → record a test sentence and
   submit → it shows up under "My recordings". If any step fails, check
   the **error log** link on the Web tab first.
4. Admin check: `GET /api/admin/stats` with header
   `X-Admin-Email: samirshirjan890@gmail.com` → counts (or use the Stats
   page in the app).

## 5. Disk ceiling — read this before collecting

PythonAnywhere free tier gives **512 MB of disk** for the whole account.
The app stores everything under `DATA_DIR`:

- SQLite database `nepali-speech.db` (tiny).
- Audio files in `data/audio/<recording_id>.<ext>`.

Rough budget: 50 hours of Opus audio ≈ **360 MB**. That leaves ~150 MB for
the repo, the venv, the frontend build, and the database. It fits Phase 2,
but only just — there is no room for a second 50-hour batch on the same
account.

**When the disk fills:** new uploads fail (the server returns 500 / the
recording cannot be saved), while the recordings already stored stay
intact. The fix is manual, from a Bash console:

```bash
cd ~/nepali-speech
du -sh data/                       # how full it is
tar -czf /tmp/nepali-speech-export-<date>.tar.gz data/   # export first
# then download the export via the Files tab, and only then:
rm data/audio/<oldest files>       # free space; never delete the .db
```

Export early and often — do not wait for the disk to fill. If Phase 2
outgrows 512 MB, the next step is the Render fallback
(`docs/deploy-render.md`, live from 1 Dec) or a paid PythonAnywhere plan,
not deleting collected data.

## 6. Redeploying after a code change

```bash
cd ~/nepali-speech
git pull
# backend change only:
workon nepali-speech && pip install -r backend/requirements.txt
# frontend change only:
cd frontend && npm ci && VITE_API_URL= npm run build
```

Then **Reload** on the Web tab. The database and audio in `data/` are
untouched by a `git pull` (`data/` is gitignored).

## Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| `500` on `/api/*`, `/health` OK | Lifespan (table creation + prompt seeding) failed at startup — read the **error log** on the Web tab; usually `DATA_DIR` is wrong or not writable. |
| Site shows "Frontend build not found" | `npm run build` was not run, or ran in the wrong directory. Rebuild per step 2 and reload. |
| App loads but API calls go to `localhost:8000` | Frontend was built without `VITE_API_URL=` — rebuild with the empty value and reload. |
| `ImportError: No module named 'a2wsgi'` | The Web tab virtualenv path is wrong, or `pip install` ran in a different venv. Re-check step 2/3. |
| Uploads fail with disk errors | Disk full — see section 5. |
