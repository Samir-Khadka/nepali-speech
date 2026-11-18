"""Tests for training/normalize.py — offline, no model, no network.

The key property: the training normaliser IS the benchmark normaliser.
If benchmark/evaluate.py::normalize_text changes, these tests fail here
too, which is the point — training and eval must never drift.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate  # noqa: E402
from normalize import (  # noqa: E402
    NORMALISATION_RULES,
    normalize_text,
    normalize_text_with_options,
)


def test_normalize_is_the_benchmark_function():
    assert normalize_text is evaluate.normalize_text


def test_rules_nfc_strip_collapse():
    # Decomposed nukta + extra whitespace -> NFC, single spaces.
    assert normalize_text("  नमस्ते   संसार  ") == "नमस्ते संसार"
    decomposed = "क" + "\u093c"  # KA + NUKTA (decomposed)
    composed = "\u0915\u093c"  # KA+NUKTA precomposed is NFC form
    assert normalize_text(decomposed) == normalize_text(composed)
    assert normalize_text("a\n\t b") == "a b"
    assert normalize_text("") == ""
    assert normalize_text(None) == ""


def test_numerals_not_converted_by_default():
    # Devanagari digits stay Devanagari; Arabic digits stay Arabic.
    assert normalize_text("सन् १८००") == "सन् १८००"
    assert normalize_text("सन् 1800") == "सन् 1800"
    assert normalize_text_with_options("सन् १८००") == "सन् १८००"


def test_numeral_conversion_is_opt_in_and_flagged():
    assert (
        normalize_text_with_options("सन् १८००", convert_devanagari_numerals=True)
        == "सन् 1800"
    )
    assert (
        normalize_text_with_options("१२३ abc ४५६", convert_devanagari_numerals=True)
        == "123 abc 456"
    )


def test_rules_documented():
    assert "NFC" in NORMALISATION_RULES
    assert "numeral" in NORMALISATION_RULES.lower()
