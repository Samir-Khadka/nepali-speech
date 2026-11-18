"""Eval-results schema tests — what score_checkpoint.py writes.

The results JSON is produced by ``training/score_checkpoint.py``'s writer:
a ``config`` block (model, checkpoint, manifest + SHA-256, clip count,
steps trained, LoRA/CTC settings read back from ``train_config.json``)
plus the harness's ``evaluate()`` output (overall + per-slice WER/CER).
These tests run the REAL writer end to end, offline, by monkeypatching
only the transcription step (which needs a model) to return fixed
hypotheses — the manifest loading, harness scoring, SHA computation and
JSON writing are all the production code paths.

Required fields are asserted, not assumed: a missing field here would
break every downstream consumer (the model card, the runbook's scoring
step, any comparison against the 2026 baseline).
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate  # noqa: E402
import score_checkpoint  # noqa: E402
from common import manifest_sha256  # noqa: E402

MANIFEST_ROWS = [
    {
        "clip_id": "fleurs-ne_np-test-1",
        "source": "fleurs",
        "speaker": "unknown",
        "transcript": "नमस्ते संसार",
        "audio_path": "fleurs/fleurs-ne_np-test-1.wav",
        "gender": "male",
        "region": "unknown",
        "quality": "clean",
        "duration_s": 2.0,
    },
    {
        "clip_id": "slr54-utt-1",
        "source": "slr54",
        "speaker": "spk-a",
        "transcript": "मेरो नाम समीर हो",
        "audio_path": "slr54/slr54-utt-1.flac",
        "gender": "unknown",
        "region": "unknown",
        "quality": "clean",
        "duration_s": 3.0,
    },
    {
        "clip_id": "slr54-utt-2",
        "source": "slr54",
        "speaker": "spk-b",
        "transcript": "काठमाडौं उपत्यका",
        "audio_path": "slr54/slr54-utt-2.flac",
        "gender": "unknown",
        "region": "unknown",
        "quality": "clean",
        "duration_s": 2.5,
    },
]

HYPS = {
    "fleurs-ne_np-test-1": "नमस्ते संसार",       # perfect
    "slr54-utt-1": "मेरो नाम समीर",              # one word deleted
    "slr54-utt-2": "काठमाडौं उपत्यका",          # perfect
}

REQUIRED_TOP = {"generated_at", "script", "config", "results"}
REQUIRED_CONFIG = {
    "model_type", "checkpoint", "base_model", "manifest", "manifest_sha256",
    "n_clips_scored", "limit", "audio_dir", "hypotheses_file",
    "convert_devanagari_numerals", "steps_trained", "train_config",
}
REQUIRED_RESULTS = {"n_clips", "n_hypotheses", "missing_hypotheses",
                    "overall", "slices", "per_clip"}
REQUIRED_OVERALL = {"n", "wer", "cer", "wer_errors", "wer_ref_words",
                    "cer_errors", "cer_ref_chars", "substitutions",
                    "deletions", "insertions"}


@pytest.fixture()
def manifest_file(tmp_path):
    path = tmp_path / "test-manifest.jsonl"
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in MANIFEST_ROWS),
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def whisper_checkpoint(tmp_path):
    ckpt = tmp_path / "ckpt-whisper"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text(
        json.dumps({"r": 8, "lora_alpha": 16}), encoding="utf-8"
    )
    (ckpt / "train_config.json").write_text(
        json.dumps({
            "script": "training/train_whisper_lora.py",
            "model": "openai/whisper-large-v3-turbo",
            "steps_trained": 1000,
            "lora": {"r": 8, "alpha": 16, "dropout": 0.05,
                     "target_modules": ["q_proj", "v_proj"]},
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    return ckpt


def _run_scoring(monkeypatch, manifest_file, checkpoint, out_path, extra=()):
    """Run score_checkpoint.main with transcription stubbed out."""
    monkeypatch.setattr(
        score_checkpoint, "transcribe_whisper",
        lambda checkpoint_, base_model, rows, audio_dir: dict(HYPS),
    )
    monkeypatch.setattr(
        score_checkpoint, "transcribe_ctc",
        lambda checkpoint_, rows, audio_dir: dict(HYPS),
    )
    rc = score_checkpoint.main([
        "--checkpoint", str(checkpoint),
        "--base-model", "openai/whisper-large-v3-turbo",
        "--manifest", str(manifest_file),
        "--out", str(out_path),
        *extra,
    ])
    assert rc == 0
    return json.loads(out_path.read_text(encoding="utf-8"))


def test_results_json_has_all_required_fields(monkeypatch, manifest_file,
                                              whisper_checkpoint, tmp_path):
    out = tmp_path / "results.json"
    report = _run_scoring(monkeypatch, manifest_file, whisper_checkpoint, out)
    assert REQUIRED_TOP <= set(report)
    assert report["script"] == "training/score_checkpoint.py"
    assert REQUIRED_CONFIG <= set(report["config"])
    assert REQUIRED_RESULTS <= set(report["results"])
    assert REQUIRED_OVERALL <= set(report["results"]["overall"])


def test_results_config_records_model_manifest_and_steps(
        monkeypatch, manifest_file, whisper_checkpoint, tmp_path):
    out = tmp_path / "results.json"
    report = _run_scoring(monkeypatch, manifest_file, whisper_checkpoint, out)
    cfg = report["config"]
    assert cfg["model_type"] == "whisper"
    assert cfg["checkpoint"] == str(whisper_checkpoint)
    assert cfg["base_model"] == "openai/whisper-large-v3-turbo"
    assert cfg["manifest"] == str(manifest_file)
    assert cfg["manifest_sha256"] == manifest_sha256(manifest_file)
    assert len(cfg["manifest_sha256"]) == 64
    assert cfg["n_clips_scored"] == 3
    assert cfg["limit"] is None
    assert cfg["convert_devanagari_numerals"] is False
    assert cfg["steps_trained"] == 1000
    # LoRA settings read back from the checkpoint's train_config.json.
    assert cfg["train_config"]["lora"]["r"] == 8
    assert cfg["train_config"]["model"] == "openai/whisper-large-v3-turbo"


def test_results_metrics_are_harness_numbers(monkeypatch, manifest_file,
                                             whisper_checkpoint, tmp_path):
    out = tmp_path / "results.json"
    report = _run_scoring(monkeypatch, manifest_file, whisper_checkpoint, out)
    res = report["results"]
    assert res["n_clips"] == 3
    assert res["n_hypotheses"] == 3
    assert res["missing_hypotheses"] == 0
    overall = res["overall"]
    assert 0.0 <= overall["wer"] < 1.0
    assert 0.0 <= overall["cer"] < 1.0
    # Cross-check against the harness directly: same manifest + hyps.
    expected = evaluate.evaluate(
        evaluate.load_manifest(manifest_file), dict(HYPS))
    assert overall["wer"] == pytest.approx(expected["overall"]["wer"])
    assert overall["cer"] == pytest.approx(expected["overall"]["cer"])
    # Slices are present for every dimension the manifest carries.
    assert "source" in res["slices"]
    assert set(res["slices"]["source"]) == {"fleurs", "slr54"}
    assert res["slices"]["source"]["slr54"]["n"] == 2
    assert len(res["per_clip"]) == 3
    assert {c["clip_id"] for c in res["per_clip"]} == set(HYPS)


def test_hypotheses_file_written_alongside_results(
        monkeypatch, manifest_file, whisper_checkpoint, tmp_path):
    out = tmp_path / "results.json"
    report = _run_scoring(monkeypatch, manifest_file, whisper_checkpoint, out)
    hyp_path = Path(report["config"]["hypotheses_file"])
    assert hyp_path.exists()
    rows = [json.loads(line) for line in hyp_path.read_text(encoding="utf-8").splitlines()]
    assert {r["clip_id"] for r in rows} == set(HYPS)
    assert all("hypothesis" in r for r in rows)


def test_limit_is_recorded_and_clips_scored_shrinks(
        monkeypatch, manifest_file, whisper_checkpoint, tmp_path):
    out = tmp_path / "results.json"
    report = _run_scoring(
        monkeypatch, manifest_file, whisper_checkpoint, out,
        extra=["--limit", "2"],
    )
    assert report["config"]["limit"] == 2
    assert report["config"]["n_clips_scored"] == 2
    assert report["results"]["n_clips"] == 2


def test_missing_train_config_means_steps_trained_is_null(
        monkeypatch, manifest_file, tmp_path):
    ckpt = tmp_path / "ckpt-no-train-config"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "results.json"
    report = _run_scoring(monkeypatch, manifest_file, ckpt, out)
    assert report["config"]["steps_trained"] is None
    assert report["config"]["train_config"] is None


# --- model-type auto-detection (no model loaded) ------------------------------

def test_detect_model_type_whisper_ctc_and_unknown(tmp_path):
    w = tmp_path / "w"
    w.mkdir()
    (w / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert score_checkpoint.detect_model_type(w) == "whisper"
    c = tmp_path / "c"
    c.mkdir()
    (c / "config.json").write_text(
        json.dumps({"architectures": ["Wav2Vec2ForCTC"]}), encoding="utf-8"
    )
    assert score_checkpoint.detect_model_type(c) == "ctc"
    empty = tmp_path / "e"
    empty.mkdir()
    assert score_checkpoint.detect_model_type(empty) is None


def test_dry_run_validates_without_a_model(manifest_file, whisper_checkpoint,
                                           tmp_path, capsys):
    rc = score_checkpoint.main([
        "--checkpoint", str(whisper_checkpoint),
        "--manifest", str(manifest_file),
        "--out", str(tmp_path / "results.json"),
        "--dry-run",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "numerals:     convert=False" in out
    assert not (tmp_path / "results.json").exists()
