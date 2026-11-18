"""WSGI entrypoint for deploying nepali-speech on PythonAnywhere (free tier).

PythonAnywhere's free tier serves Python web apps through WSGI only, so the
FastAPI (ASGI) app is wrapped with a2wsgi's ASGI-to-WSGI adapter. This file
also serves the built React frontend (``frontend/dist/``) so the whole
collection tool runs as ONE web app on the single free PythonAnywhere site:

* ``/health`` and ``/api/*``  -> the FastAPI app (via a2wsgi).
* ``/`` and any other path   -> ``frontend/dist/index.html`` (SPA fallback).
  The frontend uses HashRouter, so no server-side routing is needed; the
  fallback only exists so a refresh or a deep link still lands on the app.
* ``/assets/*`` and other real files under ``frontend/dist/`` are served
  directly from disk (with correct MIME types). A missing ``/assets/*``
  file is a real 404, not the SPA fallback, so a broken build fails loudly.

How PythonAnywhere finds this file
-----------------------------------
In the Web tab, click the "WSGI configuration file" link (it points at
``/var/www/<username>_pythonanywhere_com_wsgi.py``) and replace its whole
contents with this file. The file must expose the WSGI callable as
``application`` — that is what PythonAnywhere imports. Because the file then
lives in ``/var/www`` instead of the repo, the repo root is resolved from
``NEPALI_SPEECH_ROOT`` if set, otherwise ``~/nepali-speech`` (the clone
location in the deploy guide). If the repo was cloned elsewhere, set
``NEPALI_SPEECH_ROOT`` to its path in the Web tab's environment variables.

(If the Web tab ever lets you change the WSGI file path itself, pointing it
directly at ``<repo>/deploy/pythonanywhere_wsgi.py`` also works: the file
derives the repo root from its own location in that case.)

Path resolution
---------------
The repo root is derived from this file's location (``deploy/`` sits directly
under the repo root). ``NEPALI_SPEECH_ROOT`` overrides it if the file is ever
copied elsewhere. ``DATA_DIR`` / ``ADMIN_EMAILS`` come from the environment
variables set on the Web tab — they must be set before this module is
imported, because importing ``backend.app.main`` creates the app (and its
SQLite database) at import time.

No outbound network calls are made by this wrapper or by the app.
"""

import mimetypes
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root on sys.path, so `backend.app.main` is importable.
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
_ENV_ROOT = os.environ.get("NEPALI_SPEECH_ROOT")
if _ENV_ROOT:
    REPO_ROOT = Path(_ENV_ROOT).resolve()
elif _HERE.parent.name == "deploy":
    # Normal layout: <repo>/deploy/pythonanywhere_wsgi.py
    REPO_ROOT = _HERE.parent.parent
else:
    # File was copied elsewhere (e.g. /var/www/..._wsgi.py): fall back to the
    # cloned repo location. Adjust NEPALI_SPEECH_ROOT if this is wrong.
    REPO_ROOT = Path.home() / "nepali-speech"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DIST_DIR = REPO_ROOT / "frontend" / "dist"

# ---------------------------------------------------------------------------
# The FastAPI app, wrapped as WSGI. Imported lazily inside a function would
# hide import errors from PythonAnywhere's error log, so import at module
# level: a broken import fails the reload loudly instead of silently.
# ---------------------------------------------------------------------------

from a2wsgi import ASGIMiddleware  # noqa: E402

from backend.app.main import app as fastapi_app  # noqa: E402

# ---------------------------------------------------------------------------
# Run the app's lifespan (create tables + seed prompts) explicitly.
#
# a2wsgi's ASGIMiddleware only ever calls the app with ``http`` scopes — it
# never sends a ``lifespan`` scope. FastAPI's startup work lives in the
# lifespan handler (``Base.metadata.create_all`` + ``seed_prompts`` in
# ``backend/app/main.py``), so without this the SQLite tables would never be
# created and every database endpoint would fail. We therefore drive the
# lifespan protocol ourselves, once, in a daemon thread: send
# ``lifespan.startup``, wait for ``startup.complete``, and leave the task
# parked until process exit (it only wakes on a shutdown message, which a
# WSGI worker never sends). This reuses main.py's lifespan code exactly —
# nothing about table creation or seeding is duplicated here.
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402
import threading  # noqa: E402


def _run_lifespan(app, timeout: float = 30.0) -> None:
    started = threading.Event()
    failure: list[str] = []

    async def _lifespan() -> None:
        queue: asyncio.Queue = asyncio.Queue()

        async def receive():
            return await queue.get()

        async def send(message):
            if message["type"] == "lifespan.startup.complete":
                started.set()
            elif message["type"] == "lifespan.startup.failed":
                failure.append(str(message.get("message", "lifespan startup failed")))
                started.set()

        await queue.put({"type": "lifespan.startup"})
        # Blocks until a shutdown message arrives (i.e. process exit).
        await app(
            {"type": "lifespan", "asgi": {"version": "3.0", "spec_version": "2.0"}},
            receive,
            send,
        )

    def _runner() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_lifespan())

    thread = threading.Thread(
        target=_runner, daemon=True, name="nepali-speech-lifespan"
    )
    thread.start()
    if not started.wait(timeout=timeout):
        raise RuntimeError("FastAPI lifespan startup did not complete in time")
    if failure:
        raise RuntimeError(f"FastAPI lifespan startup failed: {failure[0]}")


_run_lifespan(fastapi_app)

_asgi_app = ASGIMiddleware(fastapi_app)

# ---------------------------------------------------------------------------
# Static file serving for the built frontend.
# ---------------------------------------------------------------------------

# mimetypes knows .js/.css/.html/.svg, but be explicit about the ones Vite
# emits that some systems miss.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("application/manifest+json", ".webmanifest")


def _serve_file(path: Path, start_response, status="200 OK"):
    """Serve one file from disk as a WSGI response."""
    content_type, _ = mimetypes.guess_type(str(path))
    if content_type is None:
        content_type = "application/octet-stream"
    data = path.read_bytes()
    headers = [
        ("Content-Type", content_type),
        ("Content-Length", str(len(data))),
    ]
    start_response(status, headers)
    return [data]


def _serve_index(start_response):
    """Serve frontend/dist/index.html, or a clear 500 if the build is missing."""
    index = DIST_DIR / "index.html"
    if not index.is_file():
        body = (
            b"Frontend build not found: run `npm run build` in frontend/ "
            b"so frontend/dist/index.html exists, then reload the web app."
        )
        start_response(
            "500 Internal Server Error",
            [("Content-Type", "text/plain"), ("Content-Length", str(len(body)))],
        )
        return [body]
    return _serve_file(index, start_response)


def application(environ, start_response):
    """The WSGI callable PythonAnywhere imports."""
    raw_path = environ.get("PATH_INFO", "/") or "/"
    # PATH_INFO is URL-decoded by the server; normalise and block traversal.
    path = "/" + raw_path.lstrip("/")
    if ".." in Path(path).parts:
        body = b"Not Found"
        start_response(
            "404 Not Found",
            [("Content-Type", "text/plain"), ("Content-Length", str(len(body)))],
        )
        return [body]

    # --- API: hand off to FastAPI via a2wsgi. ------------------------------
    if path == "/health" or path.startswith("/api/") or path == "/api":
        return _asgi_app(environ, start_response)

    # --- Static files from the frontend build. ------------------------------
    # Map the request path onto frontend/dist/ ("/" -> "index.html").
    candidate = (DIST_DIR / path.lstrip("/")).resolve()
    if candidate.is_file() and DIST_DIR.resolve() in candidate.parents:
        if environ.get("REQUEST_METHOD", "GET") == "HEAD":
            content_type, _ = mimetypes.guess_type(str(candidate))
            start_response(
                "200 OK",
                [
                    ("Content-Type", content_type or "application/octet-stream"),
                    ("Content-Length", str(candidate.stat().st_size)),
                ],
            )
            return []
        return _serve_file(candidate, start_response)

    # A missing file under /assets/ is a broken build, not a route: 404.
    if path.startswith("/assets/"):
        body = b"Not Found"
        start_response(
            "404 Not Found",
            [("Content-Type", "text/plain"), ("Content-Length", str(len(body)))],
        )
        return [body]

    # --- SPA fallback: everything else serves the app shell. ----------------
    # HashRouter keeps routes after '#', so this only matters for a refresh
    # or a pasted deep link; it is harmless either way.
    return _serve_index(start_response)
