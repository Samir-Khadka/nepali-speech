"""Shared text normaliser for training and scoring.

The normalisation rules are DEFINED by the benchmark, not by this package:

* NFC normalisation (``unicodedata.normalize("NFC", ...)``)
* strip leading/trailing whitespace
* collapse all internal whitespace runs to a single space

This module REUSES ``benchmark/evaluate.py::normalize_text`` — it imports
that function and re-exports it. The rules are not copied here, so training
text and eval text can never drift apart: there is exactly one code path.

Devanagari numerals — the decision, stated explicitly
----------------------------------------------------
The 2026 benchmark's protocol (Paudel & Sayami) normalises with NFC only
and does NOT convert numerals. Devanagari digits (०-९) and Arabic digits
(0-9) are therefore DIFFERENT tokens, both in training labels and in
scoring. Training must match that, so the default behaviour of this module
is NFC-only — no numeral conversion.

``normalize_text_with_options`` offers an opt-in, flagged
``convert_devanagari_numerals=True`` that maps Devanagari digits to Arabic
digits AFTER the NFC step. It defaults to OFF. It exists only so the
question can be tested as an experiment, and the rule for using it is:

* the training script AND the scoring script must pass the SAME flag, and
* the flag value must be recorded in ``train_config.json`` / the results
  JSON, because a model trained with converted numerals and scored without
  conversion (or vice versa) is silently mis-scored.

The eval side uses the identical function (``score_checkpoint.py`` imports
from here, which imports from ``benchmark/evaluate.py``), so there is one
code path for both sides — no drift.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import load_benchmark_modules  # noqa: E402

_build, _evaluate = load_benchmark_modules()

# The single source of truth. Do not re-implement the rules here.
normalize_text = _evaluate.normalize_text

__all__ = [
    "normalize_text",
    "normalize_text_with_options",
    "DEVANAGARI_TO_ARABIC_DIGITS",
    "NORMALISATION_RULES",
]

NORMALISATION_RULES = (
    "NFC (unicodedata.normalize), strip, collapse internal whitespace "
    "to single spaces. No numeral conversion by default — see module "
    "docstring. Defined by benchmark/evaluate.py::normalize_text; "
    "this module re-exports that function."
)

# Devanagari digits U+0966..U+096F -> ASCII digits. Only used when the
# caller explicitly opts in via normalize_text_with_options(..., True).
DEVANAGARI_TO_ARABIC_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def normalize_text_with_options(
    text: str, *, convert_devanagari_numerals: bool = False
) -> str:
    """Normalise text exactly as the benchmark does, with one opt-in extra.

    With the default (``convert_devanagari_numerals=False``) this is
    byte-for-byte identical to ``benchmark/evaluate.py::normalize_text``:
    NFC + strip + collapse whitespace, no numeral conversion.

    With ``convert_devanagari_numerals=True``, Devanagari digits are mapped
    to Arabic digits after the NFC step. Training and scoring MUST use the
    same value — see the module docstring.
    """
    normalised = normalize_text(text)
    if convert_devanagari_numerals:
        normalised = normalised.translate(DEVANAGARI_TO_ARABIC_DIGITS)
    return normalised
