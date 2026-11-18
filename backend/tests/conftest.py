"""Shared fixtures: each client gets an isolated temp DATA_DIR + SQLite DB."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import create_app  # noqa: E402

WEBM = b"\x1a\x45\xdf\xa3" + bytes(range(100))  # synthetic WebM bytes (magic + payload)


@pytest.fixture()
def make_client(tmp_path, monkeypatch):
    """Build a TestClient with its own temp DATA_DIR/DB; env overrides per call."""
    clients = []

    def _make(**overrides):
        data_dir = tmp_path / f"data-{len(clients)}"
        env = {
            "DATA_DIR": str(data_dir),
            "DATABASE_URL": f"sqlite:///{data_dir}/test.db",
            "ADMIN_EMAILS": "admin@example.com",
            "DOUBLE_REVIEW_RATE": "0",
            "MAX_UPLOAD_BYTES": str(5 * 1024 * 1024),
        }
        env.update(overrides)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        app = create_app(database_url=env["DATABASE_URL"], data_dir=data_dir)
        tc = TestClient(app)
        tc.__enter__()  # runs lifespan: create tables + seed prompts
        clients.append(tc)
        return tc

    yield _make
    for tc in clients:
        tc.__exit__(None, None, None)


@pytest.fixture()
def client(make_client):
    return make_client()


def consent(client, contributor_id="11111111-1111-1111-1111-111111111111", **extra):
    body = {"contributor_id": contributor_id, "version": "v1"}
    body.update(extra)
    resp = client.post("/api/consent", json=body)
    assert resp.status_code == 200, resp.text
    return resp


def upload(client, contributor_id, prompt_id=1, audio_bytes=WEBM, mime="audio/webm"):
    return client.post(
        "/api/recordings",
        data={"prompt_id": str(prompt_id), "contributor_id": contributor_id},
        files={"audio": ("rec.webm", audio_bytes, mime)},
    )


def make_recording(client, contributor_id, prompt_id=1):
    consent(client, contributor_id)
    resp = upload(client, contributor_id, prompt_id)
    assert resp.status_code == 201, resp.text
    return resp.json()["recording_id"]
