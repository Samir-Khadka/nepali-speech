"""Nepali ASR evaluation harness — Phase 3 benchmark infrastructure.

Protocol (Paudel & Sayami 2026, https://p-sumann.github.io/nepali-asr-benchmark/):
  * NFC normalisation of reference and hypothesis text.
  * WER and CER computed with ``jiwer`` on the normalised text.
  * Speaker-disjoint splits (see :func:`check_speaker_disjoint`).
  * Reporting is ALWAYS overall AND per-slice (gender / region / quality
    when the manifest carries those fields) — never a single average.

Reference numbers from that paper (theirs, not ours — do not present as
our results): Whisper-Large-v3-Turbo 14.76% WER in-domain (OpenSLR),
39.56% FLEURS, 48.35% Common Voice; zero-shot Whisper ~94% WER
out-of-domain.

Manifest schema (v0 — defined by Worker 3, 2026-11-18, because Worker 2's
manifests were not present when this harness was written; if Worker 2
later ships a manifest, it must conform to this schema or this file's
``load_manifest`` must be updated and the change noted in
``benchmark/README.md``):

  One JSON object per line (JSONL), UTF-8, with fields:
    clip_id    (str, required)  unique id, e.g. "fleurs-ne-test-000123"
    source     (str, required)  "fleurs" | "common_voice" | "slr54"
                                | "nepali-speech" (Phase 2 collection)
    speaker    (str, required)  speaker id — used for the disjoint check
    transcript (str, required)  reference transcript (Devanagari)
    audio_path (str, optional)  path/URL to the audio (NOT stored in repo)
    gender     (str, optional)  "female" | "male" | "other" | "unknown"
    region     (str, optional)  province / region label, e.g. "Koshi"
    quality    (str, optional)  "clean" | "phone" | "noisy"
    duration_s (float, optional)

  The v0 held-out test set is built from existing public data only
  (FLEURS / Common Voice / SLR54 test splits) because Phase 2 collection
  has produced zero audio so far. The realistic slices (phone-quality,
  noisy, dialect, code-switched Nepali) are defined by the schema but
  cannot be populated until Phase 2 audio exists.

Hypotheses file: JSONL, one object per line: {"clip_id": ..., "hypothesis": ...}
Clips with no hypothesis line are scored as an EMPTY hypothesis (all words
deleted) and counted in ``missing_hypotheses`` — this is reported, never
silent.

Metric conventions (documented, because jiwer itself does not define them):
  * Empty reference AND empty hypothesis  -> WER = CER = 0.0 (perfect).
  * Empty reference, non-empty hypothesis -> WER = CER = 1.0 (all inserted).
  * Overall WER/CER are MICRO-averages: total errors / total reference
    words (or characters) across all clips. Slice WERs are micro-averages
    within the slice. The overall number is computed from the clips, NOT
    as a mean of slice WERs, so a slice with zero clips cannot pollute it.
  * A requested slice value with zero clips is reported with n=0 and
    wer/cer=null (never 0.0, never a crash).

Usage:
  # Score a hypotheses file against a manifest:
  python benchmark/evaluate.py --manifest benchmark/manifests/v0.jsonl \\
      --hypotheses /tmp/run1.jsonl --out /tmp/run1-results.json

  # Transcribe with a Hugging Face / openai-whisper model, then score:
  python benchmark/evaluate.py --manifest benchmark/manifests/v0.jsonl \\
      --transcribe --model tiny --audio-dir /tmp/v0-audio \\
      --out /tmp/tiny-results.json
  (requires benchmark/requirements-eval.txt; CPU-only, no GPU, no account)

  # Verify speaker-disjointness between two manifests:
  python benchmark/evaluate.py --check-splits \\
      --train-manifest data/train.jsonl --test-manifest benchmark/manifests/v0.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import jiwer

__all__ = [
    "normalize_text",
    "wer_pair",
    "cer_pair",
    "load_manifest",
    "load_hypotheses",
    "evaluate",
    "check_speaker_disjoint",
    "summarise",
]

# jiwer transforms matching the 2026 protocol: NFC is applied by us first
# (normalize_text), then jiwer strips / collapses whitespace and tokenises.
_WER_TRANSFORM = jiwer.Compose(
    [
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)
_CER_TRANSFORM = jiwer.Compose(
    [
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfChars(),
    ]
)

# Slice dimensions, in reporting order. A dimension is reported only if at
# least one manifest entry carries the field (or it is explicitly requested
# via --slice-values).
SLICE_FIELDS = ("gender", "region", "quality", "source")


def normalize_text(text: str) -> str:
    """NFC-normalise, strip, collapse internal whitespace to single spaces.

    Devanagari combining marks (e.g. nukta) have composed and decomposed
    spellings; NFC makes them identical so the same spoken word scores the
    same regardless of which spelling the transcriber produced.
    """
    if text is None:
        return ""
    return " ".join(unicodedata.normalize("NFC", str(text)).split())


def _pair_measures(ref: str, hyp: str, transform) -> dict:
    """jiwer per-pair error counts, with the empty-reference convention."""
    ref_n = normalize_text(ref)
    hyp_n = normalize_text(hyp)
    if not ref_n:
        # jiwer divides by the reference length; define the edge case here.
        hyp_len = len(hyp_n.split()) if transform is _WER_TRANSFORM else len(hyp_n)
        return {
            "errors": hyp_len,
            "ref_len": 0,
            "substitutions": 0,
            "deletions": 0,
            "insertions": hyp_len,
            "hits": 0,
        }
    # jiwer 4.x: process_words / process_characters return an output object
    # with .substitutions/.deletions/.insertions/.hits (compute_measures was
    # removed in jiwer 4.0).
    if transform is _WER_TRANSFORM:
        m = jiwer.process_words(
            ref_n, hyp_n, reference_transform=transform, hypothesis_transform=transform
        )
    else:
        m = jiwer.process_characters(
            ref_n, hyp_n, reference_transform=transform, hypothesis_transform=transform
        )
    return {
        "errors": m.substitutions + m.deletions + m.insertions,
        "ref_len": len(ref_n.split()) if transform is _WER_TRANSFORM else len(ref_n),
        "substitutions": m.substitutions,
        "deletions": m.deletions,
        "insertions": m.insertions,
        "hits": m.hits,
    }


def wer_pair(reference: str, hypothesis: str) -> float:
    """Word Error Rate for one (reference, hypothesis) pair, NFC-normalised."""
    m = _pair_measures(reference, hypothesis, _WER_TRANSFORM)
    if m["ref_len"] == 0:
        return 0.0 if m["errors"] == 0 else 1.0
    return m["errors"] / m["ref_len"]


def cer_pair(reference: str, hypothesis: str) -> float:
    """Character Error Rate for one (reference, hypothesis) pair, NFC-normalised.

    Characters are Unicode code points after NFC (so a composed Devanagari
    vowel sign counts as one character, matching jiwer's char tokenisation).
    """
    m = _pair_measures(reference, hypothesis, _CER_TRANSFORM)
    if m["ref_len"] == 0:
        return 0.0 if m["errors"] == 0 else 1.0
    return m["errors"] / m["ref_len"]


def load_manifest(path: str | Path) -> list[dict]:
    """Load a v0 manifest (JSONL). Raises ValueError on schema violations."""
    entries: list[dict] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            for field in ("clip_id", "source", "speaker", "transcript"):
                if field not in row or row[field] in (None, ""):
                    raise ValueError(f"{path}:{lineno}: missing required field {field!r}")
            if row["clip_id"] in seen:
                raise ValueError(f"{path}:{lineno}: duplicate clip_id {row['clip_id']!r}")
            seen.add(row["clip_id"])
            entries.append(row)
    if not entries:
        raise ValueError(f"{path}: manifest is empty")
    return entries


def load_hypotheses(path: str | Path) -> dict[str, str]:
    """Load a hypotheses JSONL into {clip_id: hypothesis}."""
    hyps: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "clip_id" not in row or "hypothesis" not in row:
                raise ValueError(f"{path}:{lineno}: need 'clip_id' and 'hypothesis'")
            hyps[row["clip_id"]] = row["hypothesis"] or ""
    return hyps


def _aggregate(pairs: list[tuple[str, str]]) -> dict:
    """Micro-averaged WER/CER over a list of (reference, hypothesis) pairs."""
    wer_err = wer_ref = cer_err = cer_ref = 0
    subs = dels = ins = 0
    for ref, hyp in pairs:
        wm = _pair_measures(ref, hyp, _WER_TRANSFORM)
        cm = _pair_measures(ref, hyp, _CER_TRANSFORM)
        wer_err += wm["errors"]
        wer_ref += wm["ref_len"]
        cer_err += cm["errors"]
        cer_ref += cm["ref_len"]
        subs += wm["substitutions"]
        dels += wm["deletions"]
        ins += wm["insertions"]
    return {
        "n": len(pairs),
        "wer": (wer_err / wer_ref) if wer_ref else (0.0 if wer_err == 0 else 1.0),
        "cer": (cer_err / cer_ref) if cer_ref else (0.0 if cer_err == 0 else 1.0),
        "wer_errors": wer_err,
        "wer_ref_words": wer_ref,
        "cer_errors": cer_err,
        "cer_ref_chars": cer_ref,
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
    }


def evaluate(
    manifest: list[dict],
    hypotheses: dict[str, str],
    slice_fields: tuple[str, ...] = SLICE_FIELDS,
    forced_slice_values: dict[str, list[str]] | None = None,
) -> dict:
    """Score hypotheses against a manifest.

    Returns a dict with overall micro-averaged metrics, per-slice metrics,
    per-clip scores, and counts of missing/unknown hypotheses. A forced
    slice value with zero clips appears with n=0 and wer/cer=None.
    """
    forced_slice_values = forced_slice_values or {}
    pairs: list[tuple[str, str]] = []
    per_clip: list[dict] = []
    missing = 0
    for entry in manifest:
        hyp = hypotheses.get(entry["clip_id"])
        if hyp is None:
            missing += 1
            hyp = ""
        ref = entry["transcript"]
        pairs.append((ref, hyp))
        per_clip.append(
            {
                "clip_id": entry["clip_id"],
                "speaker": entry["speaker"],
                "wer": wer_pair(ref, hyp),
                "cer": cer_pair(ref, hyp),
            }
        )

    result: dict = {
        "n_clips": len(manifest),
        "n_hypotheses": len(manifest) - missing,
        "missing_hypotheses": missing,
        "overall": _aggregate(pairs),
        "slices": {},
        "per_clip": per_clip,
    }

    for field in slice_fields:
        present = any(field in e and e[field] not in (None, "") for e in manifest)
        forced = forced_slice_values.get(field, [])
        if not present and not forced:
            continue
        buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for entry in manifest:
            value = entry.get(field) or "unknown"
            buckets[str(value)].append(
                (entry["transcript"], hypotheses.get(entry["clip_id"], ""))
            )
        slice_out: dict[str, dict] = {}
        for value in sorted(set(buckets) | set(forced)):
            clip_pairs = buckets.get(value, [])
            if not clip_pairs:
                slice_out[value] = {"n": 0, "wer": None, "cer": None}
            else:
                slice_out[value] = _aggregate(clip_pairs)
        result["slices"][field] = slice_out
    return result


def check_speaker_disjoint(train_manifest: list[dict], test_manifest: list[dict]) -> dict:
    """Verify no speaker appears in both manifests (protocol requirement)."""
    train_speakers = {e["speaker"] for e in train_manifest}
    test_speakers = {e["speaker"] for e in test_manifest}
    overlap = sorted(train_speakers & test_speakers)
    return {
        "n_train_speakers": len(train_speakers),
        "n_test_speakers": len(test_speakers),
        "n_overlap": len(overlap),
        "overlap_speakers": overlap,
        "disjoint": not overlap,
    }


def summarise(result: dict) -> str:
    """Human-readable one-page summary (overall + every slice)."""
    lines = []
    o = result["overall"]
    lines.append(
        f"Clips: {result['n_clips']}  hypotheses: {result['n_hypotheses']}  "
        f"missing: {result['missing_hypotheses']}"
    )
    lines.append(
        f"OVERALL  WER {o['wer'] * 100:6.2f}%  CER {o['cer'] * 100:6.2f}%  "
        f"(errors {o['wer_errors']}/{o['wer_ref_words']} words, "
        f"{o['cer_errors']}/{o['cer_ref_chars']} chars; "
        f"sub {o['substitutions']} del {o['deletions']} ins {o['insertions']})"
    )
    for field, slices in result["slices"].items():
        lines.append(f"-- by {field} --")
        for value, s in slices.items():
            if s["n"] == 0:
                lines.append(f"  {value}: n=0 (no clips)")
            else:
                lines.append(
                    f"  {value}: n={s['n']}  WER {s['wer'] * 100:6.2f}%  "
                    f"CER {s['cer'] * 100:6.2f}%"
                )
    return "\n".join(lines)


def _transcribe_with_whisper(
    manifest: list[dict], model_name: str, audio_dir: str | Path
) -> dict[str, str]:
    """Transcribe manifest clips with openai-whisper (CPU). Lazy import."""
    import whisper  # openai-whisper; see benchmark/requirements-eval.txt

    audio_dir = Path(audio_dir)
    model = whisper.load_model(model_name)
    hyps: dict[str, str] = {}
    for entry in manifest:
        audio_path = entry.get("audio_path")
        if not audio_path:
            continue
        path = Path(audio_path)
        if not path.is_absolute():
            path = audio_dir / path
        if not path.exists():
            print(f"warning: audio missing, skipping {entry['clip_id']}: {path}", file=sys.stderr)
            continue
        text = model.transcribe(str(path), language="ne", fp16=False)["text"]
        hyps[entry["clip_id"]] = text.strip()
    return hyps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="v0 manifest JSONL")
    parser.add_argument("--hypotheses", help="hypotheses JSONL (default: score empty)")
    parser.add_argument("--transcribe", action="store_true",
                        help="transcribe with openai-whisper instead of --hypotheses")
    parser.add_argument("--model", default="tiny",
                        help="whisper model for --transcribe (default: tiny)")
    parser.add_argument("--audio-dir", default=".",
                        help="base dir for relative audio_path values")
    parser.add_argument("--out", help="write full JSON result here (default: stdout only)")
    parser.add_argument("--slice-values", action="append", default=[],
                        help="force-report a slice value, e.g. --slice-values gender=other")
    parser.add_argument("--check-splits", action="store_true",
                        help="with --train-manifest, only check speaker disjointness")
    parser.add_argument("--train-manifest", help="train manifest JSONL for --check-splits")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)

    if args.check_splits:
        if not args.train_manifest:
            parser.error("--check-splits needs --train-manifest")
        check = check_speaker_disjoint(load_manifest(args.train_manifest), manifest)
        print(json.dumps(check, ensure_ascii=False, indent=2))
        if not check["disjoint"]:
            print("FAIL: speakers overlap between train and test splits", file=sys.stderr)
            return 1
        print("OK: train/test splits are speaker-disjoint")
        return 0

    forced: dict[str, list[str]] = defaultdict(list)
    for item in args.slice_values:
        if "=" not in item:
            parser.error(f"--slice-values needs FIELD=VALUE, got {item!r}")
        field, value = item.split("=", 1)
        forced[field].append(value)

    if args.transcribe:
        hypotheses = _transcribe_with_whisper(manifest, args.model, args.audio_dir)
    elif args.hypotheses:
        hypotheses = load_hypotheses(args.hypotheses)
    else:
        hypotheses = {}

    result = evaluate(manifest, hypotheses, forced_slice_values=dict(forced))
    result["manifest"] = str(args.manifest)
    result["model"] = f"whisper-{args.model}" if args.transcribe else None
    print(summarise(result))
    if args.out:
        Path(args.out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"full results written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
