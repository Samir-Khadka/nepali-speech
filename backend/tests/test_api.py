"""API tests for CONTRACT.md. Synthetic audio bytes only — no real audio."""

import json
from pathlib import Path

from conftest import WEBM, consent, make_recording, upload

CID = "11111111-1111-1111-1111-111111111111"
CID2 = "22222222-2222-2222-2222-222222222222"

PROMPTS = json.loads(
    (Path(__file__).resolve().parents[2] / "prompts" / "prompts.json").read_text()
)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_prompts_count_matches_seed_file(client):
    assert client.get("/api/prompts/count").json() == {"total": len(PROMPTS)}


def test_next_prompt_excludes_recorded_then_done(client):
    total = len(PROMPTS)
    consent(client, CID)
    seen = []
    for _ in range(total):
        resp = client.get("/api/prompts/next", params={"contributor_id": CID})
        assert resp.status_code == 200
        body = resp.json()
        assert "done" not in body
        assert body["prompt_id"] not in seen
        seen.append(body["prompt_id"])
        assert body["text"]
        assert body["domain"]
        assert body["language"] in ("ne", "ne-en")
        assert upload(client, CID, body["prompt_id"]).status_code == 201
    done = client.get("/api/prompts/next", params={"contributor_id": CID}).json()
    assert done == {"prompt_id": None, "done": True, "total": total}
    # A fresh contributor still gets prompts.
    assert client.get("/api/prompts/next", params={"contributor_id": CID2}).json()[
        "prompt_id"
    ] == PROMPTS[0]["id"]


def test_consent_upsert(client):
    r1 = consent(client, CID, province="Koshi")
    assert r1.json() == {"ok": True, "contributor_id": CID}
    r2 = consent(client, CID, province="Bagmati", age_range="18-25")
    assert r2.json() == {"ok": True, "contributor_id": CID}
    stats = client.get("/api/admin/stats", headers={"X-Admin-Email": "admin@example.com"})
    assert stats.json()["contributors"] == 1


def test_upload_happy_path_stores_file_and_serves_audio(client, tmp_path):
    rec_id = make_recording(client, CID)
    # File on disk at data/audio/<id>.webm (find it under the temp DATA_DIR).
    matches = list(tmp_path.rglob(f"{rec_id}.webm"))
    assert len(matches) == 1
    assert matches[0].read_bytes() == WEBM
    audio = client.get(f"/api/review/audio/{rec_id}")
    assert audio.status_code == 200
    assert audio.content == WEBM
    assert audio.headers["content-type"].startswith("audio/webm")


def test_upload_wrong_magic_bytes_400(client):
    consent(client, CID)
    assert upload(client, CID, audio_bytes=b"not-a-webm-file").status_code == 400


def test_upload_wrong_content_type_400(client):
    consent(client, CID)
    assert upload(client, CID, mime="text/plain").status_code == 400


def test_upload_oversize_413_with_small_cap(make_client):
    client = make_client(MAX_UPLOAD_BYTES="100")
    consent(client, CID)
    resp = upload(client, CID, audio_bytes=WEBM + b"\x00" * 200)
    assert resp.status_code == 413


def test_upload_no_consent_403(client):
    assert upload(client, CID).status_code == 403
    assert upload(client, CID).json()["detail"] == "consent required"


def test_upload_unknown_prompt_404(client):
    consent(client, CID)
    assert upload(client, CID, prompt_id=99999).status_code == 404


def test_recordings_mine_counts(client):
    rec_id = make_recording(client, CID)
    body = client.get("/api/recordings/mine", params={"contributor_id": CID}).json()
    assert body["recorded"] == 1
    assert body["approved"] == 0
    assert body["pending"] == 1
    assert body["rejected"] == 0
    assert body["recordings"] == [
        {
            "recording_id": rec_id,
            "prompt_text": PROMPTS[0]["text"],
            "status": "pending_review",
        }
    ]


def test_review_next_empty(client):
    body = client.get("/api/review/next", params={"reviewer_id": "rev1"}).json()
    assert body["empty"] is True
    assert "message" in body


def test_review_next_serves_pending_and_audio_404(client):
    assert client.get("/api/review/audio/99999").status_code == 404
    rec_id = make_recording(client, CID)
    body = client.get("/api/review/next", params={"reviewer_id": "rev1"}).json()
    assert body["recording_id"] == rec_id
    assert body["prompt_text"] == PROMPTS[0]["text"]
    assert body["domain"] == PROMPTS[0]["domain"]
    assert body["language"] == PROMPTS[0]["language"]
    assert body["audio_url"] == f"/api/review/audio/{rec_id}"


def test_review_approve(client):
    rec_id = make_recording(client, CID)
    resp = client.post(
        f"/api/review/{rec_id}",
        json={"reviewer_id": "rev1", "decision": "approve"},
    )
    assert resp.json() == {"ok": True, "status": "approved"}
    mine = client.get("/api/recordings/mine", params={"contributor_id": CID}).json()
    assert mine["approved"] == 1
    assert mine["pending"] == 0


def test_review_reject_requires_reason(client):
    rec_id = make_recording(client, CID)
    assert (
        client.post(
            f"/api/review/{rec_id}",
            json={"reviewer_id": "rev1", "decision": "reject"},
        ).status_code
        == 400
    )
    assert (
        client.post(
            f"/api/review/{rec_id}",
            json={"reviewer_id": "rev1", "decision": "reject", "reason": None},
        ).status_code
        == 400
    )
    resp = client.post(
        f"/api/review/{rec_id}",
        json={"reviewer_id": "rev1", "decision": "reject", "reason": "too noisy"},
    )
    assert resp.json() == {"ok": True, "status": "rejected"}


def test_review_same_reviewer_twice_400(client):
    rec_id = make_recording(client, CID)
    client.post(f"/api/review/{rec_id}", json={"reviewer_id": "rev1", "decision": "approve"})
    resp = client.post(
        f"/api/review/{rec_id}", json={"reviewer_id": "rev1", "decision": "approve"}
    )
    assert resp.status_code == 400


def test_review_unknown_recording_404(client):
    assert (
        client.post(
            "/api/review/99999",
            json={"reviewer_id": "rev1", "decision": "approve"},
        ).status_code
        == 404
    )


def test_double_review_flow(make_client):
    client = make_client(DOUBLE_REVIEW_RATE="1")
    rec_id = make_recording(client, CID)
    first = client.post(
        f"/api/review/{rec_id}",
        json={"reviewer_id": "rev1", "decision": "approve"},
    )
    assert first.json() == {"ok": True, "status": "needs_second_review"}
    second = client.post(
        f"/api/review/{rec_id}",
        json={"reviewer_id": "rev2", "decision": "approve"},
    )
    assert second.json() == {"ok": True, "status": "approved"}
    stats = client.get("/api/admin/stats", headers={"X-Admin-Email": "admin@example.com"})
    body = stats.json()
    assert body["double_reviewed"] == 1
    assert body["agreement_rate"] == 1.0


def test_admin_stats_auth(client):
    assert client.get("/api/admin/stats").status_code == 403
    assert (
        client.get("/api/admin/stats", headers={"X-Admin-Email": "nope@example.com"}).status_code
        == 403
    )
    resp = client.get("/api/admin/stats", headers={"X-Admin-Email": "admin@example.com"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["prompts"] == len(PROMPTS)
    assert body["contributors"] == 0
    assert body["recordings_total"] == 0
    assert body["by_status"] == {
        "pending_review": 0,
        "approved": 0,
        "rejected": 0,
        "needs_second_review": 0,
    }
    assert body["reviews_total"] == 0
    assert body["double_reviewed"] == 0
    assert body["agreement_rate"] is None
    assert body["by_domain"] == {}


def test_admin_stats_counts_after_activity(client):
    rec1 = make_recording(client, CID, prompt_id=1)
    rec2 = make_recording(client, CID, prompt_id=2)
    client.post(f"/api/review/{rec1}", json={"reviewer_id": "rev1", "decision": "approve"})
    client.post(
        f"/api/review/{rec2}",
        json={"reviewer_id": "rev1", "decision": "reject", "reason": "unclear"},
    )
    body = client.get(
        "/api/admin/stats", headers={"X-Admin-Email": "admin@example.com"}
    ).json()
    assert body["contributors"] == 1
    assert body["recordings_total"] == 2
    assert body["by_status"]["approved"] == 1
    assert body["by_status"]["rejected"] == 1
    assert body["reviews_total"] == 2
    # Both recorded prompts are farming in the current seed file; compute
    # the expectation from the file rather than hardcoding domains.
    from collections import Counter

    assert body["by_domain"] == dict(Counter(PROMPTS[i]["domain"] for i in (0, 1)))
