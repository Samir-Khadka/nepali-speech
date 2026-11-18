"""Tests for the pure helpers and API client in bot/main.py.

No network and no Telegram token needed: HTTP is mocked with
httpx.MockTransport.
"""

import httpx
import pytest

import main


def test_contributor_id():
    assert main.contributor_id(12345) == "tg:12345"
    assert main.contributor_id(0) == "tg:0"


def test_build_consent_payload():
    payload = main.build_consent_payload("tg:12345")
    assert payload == {
        "contributor_id": "tg:12345",
        "version": "v1",
        "province": None,
        "age_range": None,
        "gender": None,
    }


def test_build_consent_payload_custom_version():
    payload = main.build_consent_payload("tg:9", version="v2")
    assert payload["version"] == "v2"
    assert payload["contributor_id"] == "tg:9"


def test_format_prompt_message():
    prompt = {"prompt_id": 12, "text": "मलाई चिया मन पर्छ।",
              "domain": "farming", "language": "ne"}
    msg = main.format_prompt_message(prompt)
    assert "मलाई चिया मन पर्छ।" in msg
    assert "farming" in msg
    assert "🎙️" in msg


def test_format_prompt_message_no_domain():
    prompt = {"prompt_id": 1, "text": "नमस्ते", "domain": None, "language": "ne"}
    msg = main.format_prompt_message(prompt)
    assert "नमस्ते" in msg
    assert "विषय" not in msg


def test_format_prompt_message_done():
    msg = main.format_prompt_message(
        {"prompt_id": None, "done": True, "total": 200})
    assert "200" in msg
    assert "बधाई" in msg


def test_format_done_message():
    assert "200" in main.format_done_message(200)
    assert "बधाई" in main.format_done_message()


def test_consent_text_mentions_cc0_no_personal_data_and_stop_anytime():
    assert "CC0" in main.CONSENT_TEXT
    assert "व्यक्तिगत जानकारी" in main.CONSENT_TEXT
    assert "रोक्न सक्नुहुन्छ" in main.CONSENT_TEXT


def _mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler),
                        base_url="http://test")


def test_post_consent_sends_expected_body():
    seen = {}

    def handler(request):
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["json"] = request.read().decode()
        return httpx.Response(200, json={"ok": True, "contributor_id": "tg:7"})

    result = main.post_consent("http://test", "tg:7",
                               client=_mock_client(handler))
    assert result == {"ok": True, "contributor_id": "tg:7"}
    assert seen["method"] == "POST"
    assert seen["url"] == "http://test/api/consent"
    assert '"contributor_id":"tg:7"' in seen["json"]
    assert '"version":"v1"' in seen["json"]


def test_fetch_next_prompt():
    def handler(request):
        assert str(request.url) == (
            "http://test/api/prompts/next?contributor_id=tg%3A7")
        return httpx.Response(200, json={"prompt_id": 12, "text": "नमस्ते",
                                         "domain": "farming", "language": "ne"})

    prompt = main.fetch_next_prompt("http://test", "tg:7",
                                    client=_mock_client(handler))
    assert prompt["prompt_id"] == 12
    assert prompt["text"] == "नमस्ते"


def test_upload_recording_multipart():
    seen = {}

    def handler(request):
        seen["content_type"] = request.headers["content-type"]
        body = request.read()
        seen["body"] = body
        assert b"prompt_id" in body
        assert b"tg:7" in body
        assert b"OggS" in body  # fake ogg magic bytes in the payload
        return httpx.Response(201, json={"recording_id": 7,
                                         "status": "pending_review"})

    result = main.upload_recording("http://test", 12, "tg:7", b"OggS\x00\x01",
                                   client=_mock_client(handler))
    assert result == {"recording_id": 7, "status": "pending_review"}
    assert seen["content_type"].startswith("multipart/form-data")


def test_post_consent_raises_on_http_error():
    def handler(request):
        return httpx.Response(500, json={"detail": "boom"})

    with pytest.raises(httpx.HTTPStatusError):
        main.post_consent("http://test", "tg:7", client=_mock_client(handler))
