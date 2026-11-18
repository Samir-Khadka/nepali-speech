"""Tests for the v0 test-set builder (benchmark/build.py).

All tests are offline except the builder's own fetchers, which are
monkeypatched here with synthetic data. No real audio, no network, no
accounts. Run from benchmark/: ../.venv/bin/python -m pytest tests/ -q
"""

import json
import sys
import unicodedata
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build  # noqa: E402
from build import (  # noqa: E402
    DisjointnessError,
    check_clip_overlap,
    check_speaker_disjointness,
    make_clip_entry,
    materialise,
    slr54_archive_for,
    speaker_bucket,
    validate_manifest,
    nfc,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _clip(source, clip_id, speaker_id=None, split="test", **extra):
    entry = {
        "clip_id": clip_id,
        "source": source,
        "source_dataset": f"test/{source}",
        "split": split,
        "speaker_id": speaker_id,
        "transcript": "परीक्षण",
        "transcript_normalized": "परीक्षण",
        "duration_seconds": None,
        "gender": None,
        "region": None,
        "quality": "clean",
        "license": "CC-BY-4.0",
        "audio_url": "https://example.com/audio",
        "audio_path": "clip",
        "parquet_url": None,
        "parquet_row": None,
        "archive_url": None,
        "archive_member": None,
        "access": {"kind": "manifest-only"},
    }
    entry.update(extra)
    return entry


def _valid_manifest(clips):
    return {
        "version": "v0",
        "generated_at": "2026-11-18T00:00:00+00:00",
        "sources": [],
        "clips": clips,
        "checks": {"speaker_disjointness": {"status": "pass"}},
    }


# --------------------------------------------------------------------------
# NFC normalisation
# --------------------------------------------------------------------------

def test_nfc_normalises_nfd_input():
    # "ऱ" (U+0931 DEVANAGARI LETTER RRA) decomposes under NFD to
    # "र" + nukta; most Devanagari text has no canonical decomposition,
    # so this is the realistic NFD/NFC pair to pin the behaviour on.
    nfd = unicodedata.normalize("NFD", "ऱ")
    assert nfd != "ऱ"
    assert nfc(nfd) == "ऱ"


def test_manifest_rejects_non_nfc_transcript():
    clip = _clip("fleurs", "fleurs-1")
    clip["transcript"] = unicodedata.normalize("NFD", "ऱ")
    errors = validate_manifest(_valid_manifest([clip]))
    assert any("NFC" in e for e in errors)


# --------------------------------------------------------------------------
# SLR54 archive mapping (rule verified against the real archives 2026-11-18)
# --------------------------------------------------------------------------

def test_slr54_archive_for_first_and_last_buckets():
    name, url, member = slr54_archive_for("00ee752d8d")
    assert name == "asr_nepali_0.zip"
    assert url == "https://www.openslr.org/resources/54/asr_nepali_0.zip"
    assert member == "asr_nepali/data/00/00ee752d8d.flac"
    name, url, member = slr54_archive_for("ff12345678")
    assert name == "asr_nepali_f.zip"
    assert member == "asr_nepali/data/ff/ff12345678.flac"
    name, _, member = slr54_archive_for("1a2b3c4d5e")
    assert name == "asr_nepali_1.zip"
    assert member == "asr_nepali/data/1a/1a2b3c4d5e.flac"


def test_speaker_bucket_is_deterministic_and_in_range():
    assert speaker_bucket("abc12") == speaker_bucket("abc12")
    assert all(0 <= speaker_bucket(f"spk{i}") <= 9 for i in range(200))
    assert len({speaker_bucket(f"spk{i}") for i in range(200)}) > 5


# --------------------------------------------------------------------------
# License gate
# --------------------------------------------------------------------------

def test_non_redistributable_source_produces_manifest_only_entries(monkeypatch):
    monkeypatch.setitem(build.SOURCES, "locked", {
        "dataset_id": "example/locked",
        "split": "test",
        "license": "ALL-RIGHTS-RESERVED",
        "download_url": "https://example.com/locked",
        "redistributable": False,
        "notes": "Download from the vendor after signing the agreement.",
    })
    entry = make_clip_entry(
        "locked", "locked-1", "परीक्षण",
        audio_url="https://example.com/locked/clip.wav",
        audio_path="clip.wav",
        archive_url="https://example.com/locked.zip",
        archive_member="clip.wav",
        parquet_url="https://example.com/locked.parquet",
    )
    assert entry["audio_url"] is None
    assert entry["audio_path"] is None
    assert entry["archive_url"] is None
    assert entry["archive_member"] is None
    assert entry["parquet_url"] is None
    assert entry["access"]["kind"] == "manifest-only"
    assert "example.com/locked" in entry["access"]["download_url"]
    assert entry["clip_id"] == "locked-1"
    assert entry["transcript"] == "परीक्षण"


def test_redistributable_source_keeps_audio_location():
    entry = make_clip_entry(
        "slr54", "slr54-00ee752d8d", "परीक्षण",
        speaker_id="abc12",
        archive_url="https://www.openslr.org/resources/54/asr_nepali_0.zip",
        archive_member="asr_nepali/data/00/00ee752d8d.flac",
        audio_url="https://www.openslr.org/resources/54/asr_nepali_0.zip",
        audio_path="asr_nepali/data/00/00ee752d8d.flac",
    )
    assert entry["audio_url"] is not None
    assert entry["archive_member"] == "asr_nepali/data/00/00ee752d8d.flac"
    assert entry["access"]["kind"] == "manifest-only"  # audio never bundled


# --------------------------------------------------------------------------
# Manifest schema validation
# --------------------------------------------------------------------------

def test_validate_manifest_accepts_valid_manifest():
    assert validate_manifest(_valid_manifest([
        _clip("fleurs", "fleurs-1"),
        _clip("slr54", "slr54-1", speaker_id="abc12"),
    ])) == []


def test_validate_manifest_rejects_duplicate_clip_ids():
    errors = validate_manifest(_valid_manifest([
        _clip("fleurs", "dup"), _clip("slr54", "dup", speaker_id="abc12")]))
    assert any("duplicate clip_id" in e for e in errors)


def test_validate_manifest_rejects_missing_and_wrong_typed_fields():
    bad = _clip("fleurs", "bad")
    del bad["transcript"]
    bad["duration_seconds"] = "three seconds"
    errors = validate_manifest(_valid_manifest([bad]))
    assert any("missing transcript" in e for e in errors)
    assert any("duration_seconds" in e for e in errors)


def test_validate_manifest_rejects_unknown_source_and_version():
    clip = _clip("nope", "x")
    errors = validate_manifest(_valid_manifest([clip]))
    assert any("unknown source" in e for e in errors)
    manifest = _valid_manifest([_clip("fleurs", "ok")])
    manifest["version"] = "v9"
    assert any("version" in e for e in validate_manifest(manifest))


# --------------------------------------------------------------------------
# Speaker-disjointness logic
# --------------------------------------------------------------------------

def test_disjointness_passes_for_clean_synthetic_split():
    clips = [
        _clip("slr54", "s-1", speaker_id="spk-a"),
        _clip("slr54", "s-2", speaker_id="spk-b"),
        _clip("fleurs", "f-1"),  # no speaker id: skipped, not failed
        _clip("fleurs", "f-2"),
    ]
    result = check_speaker_disjointness(clips, {"slr54": {"spk-c", "spk-d"}})
    assert result["status"] == "pass"
    assert result["per_source"]["slr54"]["test_speakers"] == 2
    assert result["per_source"]["slr54"]["overlap"] == []
    assert "note" in result["per_source"]["fleurs"]


def test_disjointness_fails_when_test_speaker_is_also_in_train():
    # Synthetic overlap case: speaker spk-a is in the test set AND in the
    # training portion — the check must FAIL (test set must be test-only).
    clips = [_clip("slr54", "s-1", speaker_id="spk-a")]
    with pytest.raises(DisjointnessError, match="also in train"):
        check_speaker_disjointness(clips, {"slr54": {"spk-a", "spk-z"}})


def test_disjointness_fails_when_speaker_spans_two_splits():
    clips = [
        _clip("slr54", "s-1", speaker_id="spk-a", split="test"),
        _clip("slr54", "s-2", speaker_id="spk-a", split="validation"),
    ]
    with pytest.raises(DisjointnessError, match="more than one split"):
        check_speaker_disjointness(clips, {"slr54": {"spk-z"}})


def test_disjointness_fails_when_speaker_id_shared_across_sources():
    clips = [
        _clip("slr54", "s-1", speaker_id="shared-1"),
        _clip("common_voice", "c-1", speaker_id="shared-1"),
    ]
    with pytest.raises(DisjointnessError, match="shared across sources"):
        check_speaker_disjointness(clips, {"slr54": {"other"}})


def test_disjointness_ignores_null_speakers():
    clips = [_clip("fleurs", f"f-{i}") for i in range(5)]
    assert check_speaker_disjointness(clips, {})["status"] == "pass"


def test_clip_overlap_check_fails_on_shared_recording():
    with pytest.raises(DisjointnessError, match="also in train"):
        check_clip_overlap({"a.wav", "b.wav"}, {"b.wav", "c.wav"}, "fleurs")
    assert check_clip_overlap({"a.wav"}, {"b.wav"}, "fleurs")["status"] == "pass"


# --------------------------------------------------------------------------
# build_manifest end-to-end with fake (offline) fetchers
# --------------------------------------------------------------------------

def test_build_manifest_with_fake_fetchers(monkeypatch, tmp_path):
    def fake_fleurs(session, tmp_dir):
        clips = [make_clip_entry("fleurs", f"fleurs-ne_np-test-{i}",
                                 transcript="परीक्षण",
                                 duration_seconds=3.0, gender="male",
                                 parquet_url="https://example.com/t.parquet",
                                 parquet_row=i,
                                 audio_url="https://example.com/t.parquet",
                                 audio_path=f"row {i}")
                 for i in range(2)]
        return clips, {"status": "ok", "clips": 2}, {}

    def fake_slr54(session, tmp_dir, verify_archives=True):
        clips = [make_clip_entry("slr54", "slr54-aaaa", "परीक्षण",
                                 speaker_id="spk-a",
                                 archive_url="https://example.com/a.zip",
                                 archive_member="asr_nepali/data/aa/aaaa.flac",
                                 audio_url="https://example.com/a.zip",
                                 audio_path="asr_nepali/data/aa/aaaa.flac")]
        return clips, {"status": "ok", "clips": 1}, \
            {"train_speakers": {"spk-z"}, "test_speakers": {"spk-a"}}

    def fake_gated(session, key):
        return [], {"status": "gated", "clips": 0}, {}

    monkeypatch.setitem(build.FETCHERS, "fleurs", fake_fleurs)
    monkeypatch.setitem(build.FETCHERS, "slr54", fake_slr54)
    monkeypatch.setitem(build.FETCHERS, "common_voice", fake_gated)
    monkeypatch.setitem(build.FETCHERS, "indicvoices", fake_gated)
    monkeypatch.setitem(build.FETCHERS, "seamless", fake_gated)

    manifest = build.build_manifest(session=None, tmp_dir=tmp_path, only=None)
    assert manifest["version"] == "v0"
    assert len(manifest["clips"]) == 3
    assert validate_manifest(manifest) == []
    assert manifest["checks"]["speaker_disjointness"]["status"] == "pass"
    assert manifest["realistic_slot"]["status"] == "empty"
    assert manifest["realistic_slot"]["clips"] == 0
    gated = {s["key"]: s["status"] for s in manifest["sources"]}
    assert gated["common_voice"] == "gated"


def test_build_manifest_aborts_when_disjointness_fails(monkeypatch, tmp_path):
    def bad_slr54(session, tmp_dir, verify_archives=True):
        clips = [make_clip_entry("slr54", "slr54-aaaa", "परीक्षण",
                                 speaker_id="spk-a")]
        return clips, {"status": "ok", "clips": 1}, \
            {"train_speakers": {"spk-a"}, "test_speakers": {"spk-a"}}

    monkeypatch.setitem(build.FETCHERS, "slr54", bad_slr54)
    monkeypatch.setitem(build.FETCHERS, "fleurs",
                        lambda s, t: ([], {"status": "ok", "clips": 0}, {}))
    for key in ("common_voice", "indicvoices", "seamless"):
        monkeypatch.setitem(build.FETCHERS, key,
                            lambda s, k: ([], {"status": "gated", "clips": 0}, {}))
    with pytest.raises(DisjointnessError):
        build.build_manifest(session=None, tmp_dir=tmp_path)


# --------------------------------------------------------------------------
# materialise with synthetic local files (no network)
# --------------------------------------------------------------------------

def _make_tiny_fleurs_parquet(path: Path):
    audio_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    table = pa.table({
        "audio": pa.array([
            {"bytes": b"RIFFxxxxWAVEfake-wav-0", "path": "a.wav"},
            {"bytes": b"\x00\x01\x02\x03", "path": "b.wav"},  # raw PCM branch
        ], type=audio_type),
    })
    pq.write_table(table, str(path))


def test_materialise_from_synthetic_files(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _make_tiny_fleurs_parquet(out_dir / "fleurs-ne_np-test.parquet")
    zip_path = out_dir / "asr_nepali_0.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("asr_nepali/data/00/00ee752d8d.flac", b"fake-flac-bytes")
    manifest = _valid_manifest([
        make_clip_entry("fleurs", "fleurs-ne_np-test-aaa", "परीक्षण",
                        duration_seconds=3.0, gender="male",
                        parquet_url="https://example.com/t.parquet",
                        parquet_row=0,
                        audio_url="https://example.com/t.parquet",
                        audio_path="fleurs/fleurs-ne_np-test-aaa.wav"),
        make_clip_entry("fleurs", "fleurs-ne_np-test-bbb", "परीक्षण",
                        duration_seconds=4.0, gender="male",
                        parquet_url="https://example.com/t.parquet",
                        parquet_row=1,
                        audio_url="https://example.com/t.parquet",
                        audio_path="fleurs/fleurs-ne_np-test-bbb.wav"),
        make_clip_entry("slr54", "slr54-00ee752d8d", "परीक्षण",
                        speaker_id="abc12",
                        archive_url="https://example.com/asr_nepali_0.zip",
                        archive_member="asr_nepali/data/00/00ee752d8d.flac",
                        audio_url="https://example.com/asr_nepali_0.zip",
                        audio_path="slr54/slr54-00ee752d8d.flac"),
    ])
    result = materialise(manifest, out_dir)
    assert result["clips_requested"] == 3
    assert result["files_written"] == 3
    wav0 = out_dir / "fleurs" / "fleurs-ne_np-test-aaa.wav"
    assert wav0.read_bytes() == b"RIFFxxxxWAVEfake-wav-0"  # RIFF passthrough
    wav1 = out_dir / "fleurs" / "fleurs-ne_np-test-bbb.wav"
    assert wav1.read_bytes()[:4] == b"RIFF"  # raw PCM wrapped as WAV
    flac = out_dir / "slr54" / "slr54-00ee752d8d.flac"
    assert flac.read_bytes() == b"fake-flac-bytes"


def test_materialise_rejects_unknown_clip_id(tmp_path):
    manifest = _valid_manifest([_clip("fleurs", "fleurs-1")])
    with pytest.raises(build.BuildError, match="not in manifest"):
        materialise(manifest, tmp_path, clip_ids=["nope"])


# --------------------------------------------------------------------------
# JSONL output for the scoring harness (benchmark/evaluate.py)
# --------------------------------------------------------------------------

def test_to_jsonl_row_maps_fleurs_speaker_to_unknown():
    # FLEURS carries no speaker ids: the harness schema requires a
    # non-empty speaker string, so the row uses "unknown" and the real
    # disjointness evidence for FLEURS is the builder's clip-level
    # test/train check, not the harness speaker check.
    row = build.to_jsonl_row(make_clip_entry(
        "fleurs", "fleurs-ne_np-test-aaa", "परीक्षण",
        duration_seconds=3.5, gender="male",
        audio_path="fleurs/fleurs-ne_np-test-aaa.wav"))
    assert row == {
        "clip_id": "fleurs-ne_np-test-aaa",
        "source": "fleurs",
        "speaker": "unknown",
        "transcript": "परीक्षण",
        "audio_path": "fleurs/fleurs-ne_np-test-aaa.wav",
        "gender": "male",
        "region": "unknown",
        "quality": "clean",
        "duration_s": 3.5,
    }


def test_to_jsonl_row_keeps_slr54_speaker_id():
    row = build.to_jsonl_row(make_clip_entry(
        "slr54", "slr54-00ee752d8d", "परीक्षण", speaker_id="abc12",
        audio_path="slr54/slr54-00ee752d8d.flac"))
    assert row["speaker"] == "abc12"
    assert row["gender"] == "unknown"
    assert row["duration_s"] is None


def test_jsonl_output_loads_in_the_scoring_harness(tmp_path):
    from evaluate import load_manifest

    manifest = _valid_manifest([
        make_clip_entry("fleurs", "fleurs-ne_np-test-aaa", "परीक्षण",
                        duration_seconds=3.0, gender="male",
                        audio_path="fleurs/fleurs-ne_np-test-aaa.wav"),
        make_clip_entry("slr54", "slr54-00ee752d8d", "परीक्षण",
                        speaker_id="abc12",
                        audio_path="slr54/slr54-00ee752d8d.flac"),
    ])
    jsonl_path = tmp_path / "v0.jsonl"
    jsonl_path.write_text(build.manifest_to_jsonl(manifest), encoding="utf-8")
    rows = load_manifest(jsonl_path)
    assert len(rows) == 2
    assert {r["clip_id"] for r in rows} == {
        "fleurs-ne_np-test-aaa", "slr54-00ee752d8d"}
    assert {r["source"] for r in rows} == {"fleurs", "slr54"}


def test_jsonl_rows_score_end_to_end_in_the_harness():
    from evaluate import evaluate

    rows = [
        build.to_jsonl_row(make_clip_entry(
            "slr54", "slr54-1", "नेपाल एक सुन्दर देश हो", speaker_id="spk-a",
            audio_path="slr54/slr54-1.flac")),
        build.to_jsonl_row(make_clip_entry(
            "fleurs", "fleurs-ne_np-test-1", "नेपाल एक सुन्दर देश हो",
            gender="male", audio_path="fleurs/fleurs-ne_np-test-1.wav")),
    ]
    result = evaluate(rows, {
        "slr54-1": "नेपाल एक सुन्दर देश हो",
        "fleurs-ne_np-test-1": "नेपाल एक सुन्दर देश छ",
    })
    assert result["n_clips"] == 2
    assert result["overall"]["wer"] == pytest.approx(1 / 10)  # 1 wrong / 10 words
    assert result["slices"]["source"]["slr54"]["wer"] == 0.0
    assert result["slices"]["gender"]["male"]["n"] == 1
