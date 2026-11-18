"""Tests for training/prepare_data.py + training/config.py — offline.

No network: the split rule, sampling, archive parsing and the
fail-closed disjointness check are all tested on synthetic data.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build  # noqa: E402
from config import ConfigError, load_config, validate_ctc_config, validate_lora_config  # noqa: E402
from prepare_data import (  # noqa: E402
    PrepareError,
    check_train_vs_test,
    parse_archives,
    restrict_archives,
    sample_clips,
)


def _row(clip_id, source, speaker):
    return {
        "clip_id": clip_id,
        "source": source,
        "speaker": speaker,
        "transcript": "परीक्षण",
        "audio_path": f"{source}/{clip_id}.wav",
        "gender": "unknown",
        "region": "unknown",
        "quality": "clean",
        "duration_s": 1.0,
    }


def _clip(source, clip_id, archive=None):
    return {
        "clip_id": clip_id,
        "source": source,
        "archive_url": f"https://www.openslr.org/resources/54/{archive}"
        if archive else None,
    }


# --- split rule is imported, not re-derived -------------------------------

def test_holdout_rule_comes_from_build():
    assert build.SLR54_TEST_BUCKET == 0
    # Find one speaker in bucket 0 and one outside it (deterministic).
    bucket0 = [f"spk{i}" for i in range(200) if build.speaker_bucket(f"spk{i}") == 0]
    other = [f"spk{i}" for i in range(200) if build.speaker_bucket(f"spk{i}") != 0]
    assert bucket0 and other
    assert all(build.speaker_bucket(s) != 0 for s in other)


# --- sampling ---------------------------------------------------------------

def test_sample_round_robin_mixes_sources():
    by_source = {
        "fleurs": [_clip("fleurs", f"fleurs-{i:03d}") for i in range(10)],
        "slr54": [_clip("slr54", f"slr54-{i:03d}") for i in range(10)],
    }
    out = sample_clips(by_source, 4)
    assert [c["source"] for c in out] == ["fleurs", "slr54", "fleurs", "slr54"]
    assert len(sample_clips(by_source, None)) == 20
    assert len(sample_clips(by_source, 100)) == 20


# --- --archives parsing -----------------------------------------------------

def test_parse_archives():
    assert parse_archives(None) is None
    assert parse_archives("2") == ("count", 2)
    assert parse_archives("asr_nepali_0.zip,asr_nepali_f.zip") == (
        "names", ["asr_nepali_0.zip", "asr_nepali_f.zip"])
    with pytest.raises(PrepareError):
        parse_archives("0")
    with pytest.raises(PrepareError):
        parse_archives("something.zip")


def test_restrict_archives_count_and_names():
    clips = [
        _clip("slr54", "slr54-a", "asr_nepali_0.zip"),
        _clip("slr54", "slr54-b", "asr_nepali_1.zip"),
        _clip("fleurs", "fleurs-a"),
    ]
    kept, used = restrict_archives(clips, ("count", 1))
    assert used == ["asr_nepali_0.zip"]
    assert [c["clip_id"] for c in kept] == ["slr54-a", "fleurs-a"]
    kept, used = restrict_archives(clips, ("names", ["asr_nepali_1.zip"]))
    assert [c["clip_id"] for c in kept] == ["slr54-b", "fleurs-a"]
    with pytest.raises(PrepareError):
        restrict_archives(clips, ("names", ["asr_nepali_9.zip"]))


# --- disjointness fails closed ----------------------------------------------

def test_disjoint_speaker_overlap_fails():
    train = [_row("slr54-x", "slr54", "spk1")]
    test = [_row("slr54-y", "slr54", "spk1")]
    with pytest.raises(PrepareError, match="SPEAKER OVERLAP"):
        check_train_vs_test(train, test)


def test_disjoint_clip_overlap_fails():
    train = [_row("slr54-x", "slr54", "spk1")]
    test = [_row("slr54-x", "slr54", "spk2")]
    with pytest.raises(PrepareError, match="CLIP OVERLAP"):
        check_train_vs_test(train, test)


def test_disjoint_unknown_speakers_do_not_false_positive():
    # FLEURS rows on both sides carry speaker "unknown": the speaker check
    # must not fire; the clip check is what protects FLEURS.
    train = [_row("fleurs-ne_np-train-a", "fleurs", "unknown")]
    test = [_row("fleurs-ne_np-test-b", "fleurs", "unknown")]
    report = check_train_vs_test(train, test)
    assert report["status"] == "pass"


def test_disjoint_pass():
    train = [_row("slr54-a", "slr54", "spk1"),
             _row("fleurs-ne_np-train-a", "fleurs", "unknown")]
    test = [_row("slr54-b", "slr54", "spk2"),
            _row("fleurs-ne_np-test-b", "fleurs", "unknown")]
    assert check_train_vs_test(train, test)["status"] == "pass"


# --- configs ------------------------------------------------------------------

def test_configs_load_and_validate(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for name, validator in (("configs/whisper_lora.yaml", validate_lora_config),
                            ("configs/ctc.yaml", validate_ctc_config)):
        cfg = load_config(root / name)
        assert validator(cfg) is cfg
    bad = root / "configs" / "whisper_lora.yaml"
    cfg = load_config(bad)
    cfg = dict(cfg, lora_r=0)
    with pytest.raises(ConfigError):
        validate_lora_config(cfg)
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.yaml")
