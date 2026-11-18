"""Score a trained checkpoint against the v0 holdout — via the harness.

Given a checkpoint directory (a Whisper LoRA adapter or a CTC model) and
a manifest slice, this script transcribes the clips with the model, writes
a hypotheses JSONL, and scores it by IMPORTING ``benchmark/evaluate.py``
(``evaluate`` + ``load_manifest``) — the same code path as the standalone
harness, so the numbers are harness numbers, never hand-written.

Usage:
    .venv-train/bin/python training/score_checkpoint.py \\
        --checkpoint /tmp/whisper-lora-smoke \\
        --manifest benchmark/manifests/v0.jsonl \\
        --audio-dir /tmp/v0-audio \\
        --out /tmp/whisper-lora-smoke-results.json \\
        [--limit 50] [--model-type whisper] [--base-model openai/whisper-tiny]

``--limit N`` scores only the first N manifest clips (the smoke run).
``--model-type`` is auto-detected when omitted: a checkpoint containing
``adapter_config.json`` is a Whisper LoRA adapter (needs ``--base-model``);
one whose ``config.json`` names a Wav2Vec2 architecture is CTC.

The results JSON contains the EXACT config (model id, checkpoint path,
manifest path + SHA-256, clip count, steps trained and LoRA/CTC settings
read back from the checkpoint's ``train_config.json`` when present) plus
the harness's overall + per-slice WER/CER. No numbers are invented: if
the checkpoint has no ``train_config.json``, ``steps_trained`` is null,
not guessed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import load_benchmark_modules  # noqa: E402
from common import (  # noqa: E402
    apply_peft_whisper_input_ids_fix,
    load_audio_16k,
    load_manifest_rows,
    manifest_sha256,
    require_heavy,
)


def _evaluate_module():
    """Load benchmark/evaluate.py on first use (it needs jiwer)."""
    _build, evaluate = load_benchmark_modules()
    return evaluate


def detect_model_type(checkpoint: Path) -> str | None:
    """Auto-detect 'whisper' (LoRA adapter) or 'ctc' from checkpoint files."""
    if (checkpoint / "adapter_config.json").exists():
        return "whisper"
    config_path = checkpoint / "config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None
        archs = cfg.get("architectures", []) or []
        if any("Wav2Vec2" in a for a in archs):
            return "ctc"
        if any("Whisper" in a for a in archs):
            return "whisper"
    return None


def transcribe_whisper(checkpoint, base_model, rows, audio_dir):
    """Transcribe rows with a Whisper model (+ LoRA adapter if present)."""
    torch = require_heavy("torch")
    transformers = require_heavy("transformers")
    require_heavy("peft")

    from peft import PeftModel

    is_adapter = (checkpoint / "adapter_config.json").exists()
    model_id = base_model or (None if is_adapter else str(checkpoint))
    if model_id is None:
        raise SystemExit(
            "ERROR: checkpoint is a LoRA adapter but no --base-model was given "
            "(the adapter has no base weights of its own)."
        )
    processor = transformers.WhisperProcessor.from_pretrained(
        model_id, language="ne", task="transcribe"
    )
    model = transformers.WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float32
    )
    if is_adapter:
        model = PeftModel.from_pretrained(model, str(checkpoint))
        # Same transformers 5.x + peft 0.21 clash as in training:
        # PeftModelForSeq2SeqLM.forward injects input_ids=None /
        # inputs_embeds=None and Whisper's decoder chokes on them
        # during generate() too — see common.
        model = apply_peft_whisper_input_ids_fix(model)
    model.eval()
    hyps: dict[str, str] = {}
    for row in rows:
        path = resolve_audio(row, audio_dir)
        if path is None or not path.exists():
            print(f"warning: audio missing, skipping {row['clip_id']}",
                  file=sys.stderr)
            continue
        audio, _ = load_audio_16k(path)
        feats = processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features
        with torch.no_grad():
            # transformers 5.x generate() takes inputs as keywords only
            # (and peft's generate() wrapper takes no positional args at
            # all — a positional input_features dies with "takes 1
            # positional argument but 2 were given"). `forced_decoder_ids`
            # is deprecated in 5.x in favour of `language` + `task`,
            # which generate() folds into the generation config.
            ids = model.generate(
                input_features=feats, language="ne", task="transcribe",
                max_new_tokens=256)
        text = processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
        hyps[row["clip_id"]] = text.strip()
    return hyps


def transcribe_ctc(checkpoint, rows, audio_dir):
    """Transcribe rows with a CTC (Wav2Vec2-family) model, greedy decode."""
    torch = require_heavy("torch")
    transformers = require_heavy("transformers")

    processor = transformers.Wav2Vec2Processor.from_pretrained(str(checkpoint))
    model = transformers.Wav2Vec2ForCTC.from_pretrained(
        str(checkpoint), torch_dtype=torch.float32
    )
    model.eval()
    hyps: dict[str, str] = {}
    for row in rows:
        path = resolve_audio(row, audio_dir)
        if path is None or not path.exists():
            print(f"warning: audio missing, skipping {row['clip_id']}",
                  file=sys.stderr)
            continue
        audio, _ = load_audio_16k(path)
        inputs = processor(
            audio, sampling_rate=16000, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(ids)[0]
        hyps[row["clip_id"]] = text.strip()
    return hyps


def resolve_audio(row, audio_dir):
    """Resolve a manifest row's audio_path against --audio-dir (or as-is)."""
    audio_path = row.get("audio_path")
    if not audio_path:
        return None
    path = Path(audio_path)
    if path.is_absolute():
        return path
    if audio_dir:
        return Path(audio_dir) / path
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True,
                   help="checkpoint dir (Whisper adapter or CTC model)")
    p.add_argument("--model-type", choices=["whisper", "ctc"], default=None,
                   help="auto-detected from the checkpoint when omitted")
    p.add_argument("--base-model", default=None,
                   help="base Whisper model for a LoRA adapter checkpoint")
    p.add_argument("--manifest", default="benchmark/manifests/v0.jsonl",
                   help="test manifest JSONL (default: v0 holdout)")
    p.add_argument("--audio-dir", default=None,
                   help="base dir for relative audio_path values")
    p.add_argument("--limit", type=int, default=None,
                   help="score only the first N manifest clips")
    p.add_argument("--max-clips", type=int, default=None,
                   help="alias for --limit")
    p.add_argument("--out", required=True, help="results JSON to write")
    p.add_argument("--hypotheses-out", default=None,
                   help="also write the hypotheses JSONL here")
    p.add_argument("--convert-devanagari-numerals", action="store_true",
                   help="must match the value used in training (default OFF)")
    p.add_argument("--dry-run", action="store_true",
                   help="validate checkpoint + manifest, print the plan, exit 0 "
                        "without loading a model")
    args = p.parse_args(argv)

    limit = args.limit if args.limit is not None else args.max_clips
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"ERROR: checkpoint not found: {checkpoint}", file=sys.stderr)
        return 1
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1
    model_type = args.model_type or detect_model_type(checkpoint)
    if model_type is None:
        print("ERROR: could not auto-detect --model-type from the checkpoint "
              "(no adapter_config.json, no recognisable config.json); pass "
              "--model-type whisper|ctc explicitly.", file=sys.stderr)
        return 1
    try:
        rows = load_manifest_rows(manifest_path, max_clips=limit)
    except ValueError as exc:
        print(f"ERROR: invalid manifest: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if not rows:
        print("ERROR: manifest is empty", file=sys.stderr)
        return 1

    train_config = None
    train_config_path = checkpoint / "train_config.json"
    if train_config_path.exists():
        try:
            train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            print(f"warning: unreadable train_config.json: {exc}", file=sys.stderr)

    if args.dry_run:
        print("DRY RUN — scoring plan (no model loaded, no audio read)")
        print(f"  checkpoint:   {checkpoint} (type: {model_type})")
        print(f"  base model:   {args.base_model or '(from checkpoint)'}")
        print(f"  manifest:     {manifest_path}")
        print(f"    sha256:     {manifest_sha256(manifest_path)}")
        print(f"    clips:      {len(rows)}"
              + (f" (limit {limit})" if limit else ""))
        print(f"  audio dir:    {args.audio_dir or '(manifest paths as-is)'}")
        print(f"  out:          {args.out}")
        print(f"  train_config: {'found' if train_config else 'absent — steps_trained will be null'}")
        print(f"  numerals:     convert={args.convert_devanagari_numerals} (must match training)")
        return 0

    if model_type == "whisper":
        hyps = transcribe_whisper(
            checkpoint, args.base_model, rows, args.audio_dir)
    else:
        hyps = transcribe_ctc(checkpoint, rows, args.audio_dir)
    if args.convert_devanagari_numerals:
        from normalize import normalize_text_with_options  # noqa: E402

        hyps = {k: normalize_text_with_options(v, convert_devanagari_numerals=True)
                for k, v in hyps.items()}

    hypotheses_path = args.hypotheses_out
    if hypotheses_path is None:
        hypotheses_path = str(Path(args.out).with_suffix("")) + ".hypotheses.jsonl"
    Path(hypotheses_path).parent.mkdir(parents=True, exist_ok=True)
    with open(hypotheses_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(
                {"clip_id": row["clip_id"],
                 "hypothesis": hyps.get(row["clip_id"], "")},
                ensure_ascii=False) + "\n")

    evaluate = _evaluate_module()
    result = evaluate.evaluate(rows, hyps)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "training/score_checkpoint.py",
        "config": {
            "model_type": model_type,
            "checkpoint": str(checkpoint),
            "base_model": args.base_model,
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha256(manifest_path),
            "n_clips_scored": len(rows),
            "limit": limit,
            "audio_dir": args.audio_dir,
            "hypotheses_file": hypotheses_path,
            "convert_devanagari_numerals": args.convert_devanagari_numerals,
            "steps_trained": (train_config or {}).get("steps_trained"),
            "train_config": train_config,
        },
        "results": result,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(evaluate.summarise(result))
    print(f"results written to {out_path}")
    print(f"hypotheses written to {hypotheses_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
