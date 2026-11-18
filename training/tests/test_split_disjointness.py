"""Split-disjointness tests — training vs the v0 holdout (offline).

Worker A's ``test_prepare_data.py`` already covers ``check_train_vs_test``
on hand-made rows. These tests go one step further: they build a small
synthetic TRAIN manifest the way ``prepare_data.py`` does — SLR54 speakers
chosen by the imported holdout rule (``sha256(user_id) % 10 == 0`` is the
holdout, ``benchmark/build.py``) plus FLEURS official-train clips — and
assert zero overlap with (a) a small fixture v0-style manifest and
(b) the REAL ``benchmark/manifests/v0.jsonl`` shipped in the repo
(14,145 rows, read from disk, no network).

The FLEURS case is the subtle one: every FLEURS row carries speaker
``"unknown"`` on both sides, so the speaker check is vacuous there by
design and the clip-level check is what protects it. That is tested
explicitly, including via ``build.check_clip_overlap`` on recording
basenames — the same check ``fetch_fleurs_train`` runs.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build  # noqa: E402
import evaluate  # noqa: E402
from prepare_data import PrepareError, check_train_vs_test  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
V0_MANIFEST = REPO_ROOT / "benchmark" / "manifests" / "v0.jsonl"


def _row(clip_id, source, speaker):
    return {
        "clip_id": clip_id,
        "source": source,
        "speaker": speaker,
        "transcript": "परीक्षण वाक्य",
        "audio_path": f"{source}/{clip_id}.wav",
        "gender": "unknown",
        "region": "unknown",
        "quality": "clean",
        "duration_s": 1.0,
    }


def _speakers_in_bucket(bucket, n, prefix="synth-spk"):
    """Deterministically find n speaker ids in the given holdout bucket."""
    out = []
    i = 0
    while len(out) < n:
        cand = f"{prefix}-{i}"
        if build.speaker_bucket(cand) == bucket:
            out.append(cand)
        i += 1
    return out


def _synthetic_train_rows(n_slr54_speakers=6, n_fleurs=4):
    """Train-side rows, built the way prepare_data.py builds them."""
    rows = []
    for spk in _speakers_in_bucket(1, n_slr54_speakers):
        # Bucket 1 is a train bucket (any bucket != 0 is train).
        assert build.speaker_bucket(spk) != build.SLR54_TEST_BUCKET
        rows.append(_row(f"slr54-{spk}-utt0", "slr54", spk))
    for i in range(n_fleurs):
        rows.append(
            _row(f"fleurs-ne_np-train-1000000000000000{i}", "fleurs", "unknown")
        )
    return rows


def _synthetic_test_rows(n_slr54_speakers=4, n_fleurs=3):
    """v0-style holdout rows: bucket-0 SLR54 speakers + FLEURS test clips."""
    rows = []
    for spk in _speakers_in_bucket(build.SLR54_TEST_BUCKET, n_slr54_speakers):
        assert build.speaker_bucket(spk) == build.SLR54_TEST_BUCKET
        rows.append(_row(f"slr54-{spk}-utt0", "slr54", spk))
    for i in range(n_fleurs):
        rows.append(
            _row(f"fleurs-ne_np-test-1002268537783088316{i}", "fleurs", "unknown")
        )
    return rows


# --- the holdout rule itself ------------------------------------------------

def test_holdout_bucket_is_zero_and_train_is_everything_else():
    assert build.SLR54_TEST_BUCKET == 0
    train_speakers = _speakers_in_bucket(1, 5)
    test_speakers = _speakers_in_bucket(0, 5)
    assert not set(train_speakers) & set(test_speakers)
    assert all(build.speaker_bucket(s) != 0 for s in train_speakers)
    assert all(build.speaker_bucket(s) == 0 for s in test_speakers)


# --- synthetic train vs synthetic v0-style fixture --------------------------

def test_synthetic_train_disjoint_from_fixture_holdout():
    train = _synthetic_train_rows()
    test = _synthetic_test_rows()
    report = check_train_vs_test(train, test)
    assert report["status"] == "pass"
    # The speaker-level check underneath must have seen real speakers.
    assert report["speaker_check"]["disjoint"] is True
    assert report["speaker_check"]["n_overlap"] == 0


def test_synthetic_train_disjoint_from_real_v0_manifest():
    """Strongest offline check: synthetic train rows vs the real v0 file."""
    train = _synthetic_train_rows()
    test = evaluate.load_manifest(V0_MANIFEST)
    assert len(test) == 14145
    report = check_train_vs_test(train, test)
    assert report["status"] == "pass"
    assert report["speaker_check"]["disjoint"] is True


def test_one_holdout_speaker_leaking_into_train_fails():
    train = _synthetic_train_rows()
    test = _synthetic_test_rows()
    leaked = _speakers_in_bucket(build.SLR54_TEST_BUCKET, 1)[0]
    train.append(_row(f"slr54-{leaked}-utt0", "slr54", leaked))
    with pytest.raises(PrepareError, match="SPEAKER OVERLAP"):
        check_train_vs_test(train, test)


# --- FLEURS: speaker "unknown" on both sides, clip level decides ------------

def test_fleurs_unknown_speakers_pass_when_clip_ids_differ():
    train = _synthetic_train_rows(n_slr54_speakers=0)
    test = _synthetic_test_rows(n_slr54_speakers=0)
    assert all(r["speaker"] == "unknown" for r in train + test)
    assert check_train_vs_test(train, test)["status"] == "pass"


def test_fleurs_same_clip_id_on_both_sides_fails():
    train = [_row("fleurs-ne_np-train-10000000000000001", "fleurs", "unknown")]
    # Same clip id, but labelled as a test clip: the clip check must fire
    # even though the speaker check cannot (both "unknown").
    test = [_row("fleurs-ne_np-train-10000000000000001", "fleurs", "unknown")]
    with pytest.raises(PrepareError, match="CLIP OVERLAP"):
        check_train_vs_test(train, test)


def test_fleurs_basename_overlap_fails_via_build_check():
    """Mirrors fetch_fleurs_train: no test recording basename in train."""
    train_paths = {"rec-a.wav", "rec-b.wav"}
    test_paths = {"rec-c.wav", "rec-d.wav"}
    assert build.check_clip_overlap(test_paths, train_paths, "fleurs")["status"] == "pass"
    with pytest.raises(build.DisjointnessError, match="1 test recording"):
        build.check_clip_overlap(
            {"rec-b.wav", "rec-e.wav"}, train_paths, "fleurs train vs test"
        )


def test_fleurs_train_clips_disjoint_from_real_v0_fleurs_clips():
    train = _synthetic_train_rows(n_slr54_speakers=0)
    test = [r for r in evaluate.load_manifest(V0_MANIFEST) if r["source"] == "fleurs"]
    assert len(test) == 726
    assert check_train_vs_test(train, test)["status"] == "pass"
    train_ids = {r["clip_id"] for r in train}
    test_ids = {r["clip_id"] for r in test}
    assert not train_ids & test_ids
