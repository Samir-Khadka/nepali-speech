"""Tests for the Nepali ASR evaluation harness (benchmark/evaluate.py).

Every expected value below is hand-computed from the (reference, hypothesis)
pair written next to it — no value is copied from the harness itself, so a
bug in the metric math fails the test instead of being baked in.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import (  # noqa: E402
    cer_pair,
    check_speaker_disjoint,
    evaluate,
    load_manifest,
    normalize_text,
    wer_pair,
)


# ---------------------------------------------------------------- WER math
def test_wer_one_substitution():
    # ref 5 words, hyp differs in 1 word -> 1/5 = 0.20
    assert wer_pair("नेपाल एक सुन्दर देश हो", "नेपाल एक सुन्दर देश छ") == pytest.approx(0.20)


def test_wer_one_deletion():
    # ref 4 words, hyp drops "सुन्दर" -> 1/4 = 0.25
    assert wer_pair("नेपाल एक सुन्दर देश", "नेपाल एक देश") == pytest.approx(0.25)


def test_wer_one_insertion():
    # ref 3 words, hyp adds "धेरै" -> 1/3
    assert wer_pair("नेपाल सुन्दर देश", "नेपाल धेरै सुन्दर देश") == pytest.approx(1 / 3)


def test_wer_identical_is_zero():
    assert wer_pair("नेपाल एक सुन्दर देश हो", "नेपाल एक सुन्दर देश हो") == 0.0


def test_wer_empty_hypothesis_is_one():
    # ref 3 words, hyp empty -> 3 deletions / 3 = 1.0
    assert wer_pair("नेपाल एक देश", "") == pytest.approx(1.0)


def test_wer_both_empty_is_zero():
    assert wer_pair("", "") == 0.0


def test_wer_empty_reference_nonempty_hypothesis_is_one():
    # Convention (documented in evaluate.py): all insertions -> 1.0
    assert wer_pair("", "नेपाल") == pytest.approx(1.0)


def test_wer_whitespace_only_counts_as_empty():
    assert wer_pair("   ", "") == 0.0
    assert wer_pair("नेपाल", "   ") == pytest.approx(1.0)


# ---------------------------------------------------------------- CER math
def test_cer_one_deletion():
    # "नेपाल" = न + े + प + ा + ल = 5 code points (NFC);
    # "नेपल" drops the ा -> 1/5 = 0.20
    assert cer_pair("नेपाल", "नेपल") == pytest.approx(0.20)


def test_cer_one_substitution():
    # "काठमाडौं" vs "काठमाडौँ": same length, 1 char differs -> 1/8
    ref = "काठमाडौं"
    hyp = "काठमाडौँ"
    assert len(ref) == len(hyp) == 8
    assert cer_pair(ref, hyp) == pytest.approx(1 / 8)


def test_cer_identical_is_zero():
    assert cer_pair("नेपाल", "नेपाल") == 0.0


def test_cer_empty_cases():
    assert cer_pair("", "") == 0.0
    assert cer_pair("नेपाल", "") == pytest.approx(1.0)
    assert cer_pair("", "नेपाल") == pytest.approx(1.0)


# ------------------------------------------------------- NFC normalisation
def test_nfc_equivalence_devanagari():
    # "ऱ" NFC = U+0931 (DEVANAGARI LETTER RRA); NFD = "र" (U+0930) + nukta
    # (U+093C). Same letter, different byte spelling — must score identically
    # after NFC. (Verified: unicodedata NFC/NFD differ on U+0931.)
    composed = "\u0931"       # ऱ
    decomposed = "र\u093c"     # र + nukta
    assert composed != decomposed  # they really are different byte strings
    assert normalize_text(composed) == normalize_text(decomposed)
    assert wer_pair(composed, decomposed) == 0.0
    assert cer_pair(composed, decomposed) == 0.0


def test_nfc_equivalence_in_sentence():
    composed = "मेरो नाम ऱाम हो"
    decomposed = "मेरो नाम र\u093cाम हो"
    assert wer_pair(composed, decomposed) == 0.0


def test_normalize_collapses_whitespace():
    assert normalize_text("  नेपाल   एक\tदेश\n") == "नेपाल एक देश"


# ------------------------------------------------------------- aggregation
def _manifest(tmp_path, rows):
    path = tmp_path / "manifest.jsonl"
    import json

    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return load_manifest(path)


def test_overall_is_micro_average_not_mean_of_slices():
    # Slice A: 1 clip, ref 10 words, 10 errors -> WER 1.0
    # Slice B: 1 clip, ref 2 words, 0 errors  -> WER 0.0
    # Mean of slice WERs would be 0.50; micro-average is 10/12 = 0.8333.
    rows = [
        {"clip_id": "a1", "source": "fleurs", "speaker": "s1",
         "transcript": "एक दुई तीन चार पाँच छ सात आठ नौ दश", "gender": "female"},
        {"clip_id": "b1", "source": "fleurs", "speaker": "s2",
         "transcript": "नेपाल देश", "gender": "male"},
    ]
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        manifest = _manifest(Path(d), rows)
    hyps = {"a1": "", "b1": "नेपाल देश"}
    result = evaluate(manifest, hyps)
    assert result["overall"]["wer"] == pytest.approx(10 / 12)
    assert result["slices"]["gender"]["female"]["wer"] == pytest.approx(1.0)
    assert result["slices"]["gender"]["male"]["wer"] == pytest.approx(0.0)


def test_empty_slice_does_not_crash_or_pollute():
    rows = [
        {"clip_id": "a1", "source": "fleurs", "speaker": "s1",
         "transcript": "नेपाल एक देश", "gender": "female"},
    ]
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        manifest = _manifest(Path(d), rows)
    # Force-report a gender value with zero clips.
    result = evaluate(manifest, {"a1": "नेपाल एक देश"},
                      forced_slice_values={"gender": ["other"]})
    assert result["slices"]["gender"]["other"] == {"n": 0, "wer": None, "cer": None}
    # Overall is still the real number over the one real clip.
    assert result["overall"]["wer"] == 0.0
    assert result["overall"]["n"] == 1


def test_missing_hypothesis_counts_as_empty_and_is_reported():
    rows = [
        {"clip_id": "a1", "source": "fleurs", "speaker": "s1", "transcript": "नेपाल एक देश"},
        {"clip_id": "a2", "source": "fleurs", "speaker": "s2", "transcript": "काठमाडौं ठूलो शहर"},
    ]
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        manifest = _manifest(Path(d), rows)
    result = evaluate(manifest, {"a1": "नेपाल एक देश"})
    assert result["missing_hypotheses"] == 1
    # a2 scored as empty: 3 deletions / 3 words -> contributes 3/6 overall.
    assert result["overall"]["wer"] == pytest.approx(3 / 6)


def test_slice_dimension_absent_when_field_missing_everywhere():
    rows = [
        {"clip_id": "a1", "source": "fleurs", "speaker": "s1", "transcript": "नेपाल"},
    ]
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        manifest = _manifest(Path(d), rows)
    result = evaluate(manifest, {"a1": "नेपाल"})
    assert "gender" not in result["slices"]  # no gender field anywhere
    assert "source" in result["slices"]      # source is always present


def test_unknown_slice_value_for_missing_field():
    rows = [
        {"clip_id": "a1", "source": "fleurs", "speaker": "s1",
         "transcript": "नेपाल", "gender": "female"},
        {"clip_id": "a2", "source": "fleurs", "speaker": "s2",
         "transcript": "देश"},  # no gender -> "unknown"
    ]
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        manifest = _manifest(Path(d), rows)
    result = evaluate(manifest, {"a1": "नेपाल", "a2": "देश"})
    assert set(result["slices"]["gender"]) == {"female", "unknown"}


# ------------------------------------------------------- manifest validation
def test_load_manifest_rejects_duplicate_clip_id(tmp_path):
    import json

    path = tmp_path / "m.jsonl"
    path.write_text(
        "\n".join(json.dumps({"clip_id": "x", "source": "fleurs", "speaker": "s",
                              "transcript": "नेपाल"}, ensure_ascii=False) for _ in range(2)),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate clip_id"):
        load_manifest(path)


def test_load_manifest_rejects_missing_required_field(tmp_path):
    import json

    path = tmp_path / "m.jsonl"
    path.write_text(json.dumps({"clip_id": "x", "source": "fleurs", "speaker": "s"},
                               ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required field 'transcript'"):
        load_manifest(path)


# ------------------------------------------------------- speaker disjointness
def test_check_speaker_disjoint():
    train = [{"speaker": "s1"}, {"speaker": "s2"}]
    test = [{"speaker": "s3"}, {"speaker": "s4"}]
    out = check_speaker_disjoint(train, test)
    assert out["disjoint"] is True
    assert out["n_overlap"] == 0


def test_check_speaker_disjoint_detects_overlap():
    train = [{"speaker": "s1"}, {"speaker": "s2"}]
    test = [{"speaker": "s2"}, {"speaker": "s3"}]
    out = check_speaker_disjoint(train, test)
    assert out["disjoint"] is False
    assert out["overlap_speakers"] == ["s2"]
