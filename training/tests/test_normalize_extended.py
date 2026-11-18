"""Extended normaliser tests — identical outputs on a Devanagari corpus.

Worker A's ``test_normalize.py`` already asserts ``normalize_text`` IS the
benchmark function (same object). These tests assert the behavioural side
of the same contract: on a Devanagari corpus that exercises NFC edge
cases — nukta in decomposed vs composed spelling, combining vowel signs,
anusvara/chandrabindu — the training normaliser and
``benchmark/evaluate.py::normalize_text`` return byte-identical output.
If the two ever diverge, training labels and eval scores would silently
disagree, so the corpus is the tripwire.

Also pinned here: the numeral-conversion option defaults OFF everywhere
it exists (the function, and the three script CLIs), per the 2026
benchmark protocol (NFC only, Devanagari and Arabic digits are different
tokens).
"""

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate  # noqa: E402
import train_ctc  # noqa: E402
import train_whisper_lora  # noqa: E402
from normalize import normalize_text, normalize_text_with_options  # noqa: E402

# Devanagari corpus exercising NFC edge cases. Each entry is a pair of
# spellings of the SAME text: decomposed (base + combining mark) vs the
# composed form NFC produces. After normalisation both must be identical
# — and identical between training and benchmark.
CORPUS_PAIRS = [
    # Nukta: KA + NUKTA (decomposed) vs precomposed QA.
    ("क" + "\u093c", "\u0958"),
    # Nukta on KHA: KHA + NUKTA vs precomposed KHA-NUKTA.
    ("ख" + "\u093c", "\u0959"),
    # Nukta on GA: GA + NUKTA vs precomposed GHA-NUKTA form.
    ("ग" + "\u093c", "\u095a"),
    # Vowel sign U + anusvara: कु + ं, decomposed vs composed orderings.
    ("कु" + "\u0902", "कु" + "\u0902"),
    # Chandrabindu vs anusvara are DIFFERENT marks — must stay different.
    ("क" + "\u0901", "क" + "\u0902"),
    # Combining vowel sign I after a consonant cluster.
    ("हिन्दी", "हिन्दी"),
    # Extra whitespace + newline/tab around Devanagari text.
    ("  नमस्ते   संसार\n\t", "नमस्ते संसार"),
    # Mixed Devanagari + Arabic digits + Latin.
    ("सन् १८०० मा ABC 123", "सन् १८०० मा ABC 123"),
]

CORPUS_SINGLES = [
    "नेपाली भाषा",
    "काठमाडौं उपत्यका",
    "मेरो नाम समीर हो",
    "क़िताब",  # precomposed nukta word
    "ऑफिस",  # precomposed vowel + nukta
    "१२३४५६७८९०",  # all Devanagari digits
    "०१२३ abc ४५६",
    "संयुक्त राष्ट्र संघ",
    "हिउँदे बिदा",  # chandrabindu
    "अंङ्ग्रेजी",  # anusvara + nga
]


def test_training_and_benchmark_agree_on_corpus():
    for a, b in CORPUS_PAIRS:
        assert normalize_text(a) == evaluate.normalize_text(a)
        assert normalize_text(b) == evaluate.normalize_text(b)
    for text in CORPUS_SINGLES:
        assert normalize_text(text) == evaluate.normalize_text(text)


def test_decomposed_and_composed_spellings_normalise_identically():
    # The point of NFC: the same spoken word scores the same regardless
    # of which spelling the transcriber produced.
    assert normalize_text("क" + "\u093c") == normalize_text("\u0958")
    assert normalize_text("ख" + "\u093c") == normalize_text("\u0959")
    assert normalize_text("क" + "\u0901") != normalize_text("क" + "\u0902")


def test_numeral_conversion_defaults_off_everywhere():
    # The function itself.
    sig = inspect.signature(normalize_text_with_options)
    assert sig.parameters["convert_devanagari_numerals"].default is False
    # Default call leaves Devanagari digits untouched.
    assert normalize_text_with_options("सन् १८००") == "सन् १८००"
    assert normalize_text("२०८३ साल") == "२०८३ साल"
    # Both train-script CLIs: --convert-devanagari-numerals is store_true,
    # so the default is False unless the flag is passed.
    for module in (train_whisper_lora, train_ctc):
        args = module.parse_args(
            ["--train-manifest", "train.jsonl", "--output-dir", "/tmp/out"]
        )
        assert args.convert_devanagari_numerals is False
    # And the opt-in still works when explicitly requested.
    assert (
        normalize_text_with_options("२०८३ साल", convert_devanagari_numerals=True)
        == "2083 साल"
    )


def test_train_script_default_models_are_the_documented_ones():
    assert train_whisper_lora.DEFAULT_MODEL == "openai/whisper-tiny"
    assert train_whisper_lora.REAL_MODEL == "openai/whisper-large-v3-turbo"
    assert train_ctc.DEFAULT_MODEL == "facebook/wav2vec2-xls-r-300m"
    args = train_whisper_lora.parse_args(
        ["--train-manifest", "train.jsonl", "--output-dir", "/tmp/out"]
    )
    assert args.model == "openai/whisper-tiny"
    args = train_ctc.parse_args(
        ["--train-manifest", "train.jsonl", "--output-dir", "/tmp/out"]
    )
    assert args.model == "facebook/wav2vec2-xls-r-300m"
