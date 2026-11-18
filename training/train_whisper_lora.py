"""Whisper LoRA fine-tune (Nepali) with peft + HF Trainer.

Default model is ``openai/whisper-tiny`` (smoke runs). The real run uses
``openai/whisper-large-v3-turbo`` — the 2026 benchmark's best model
(14.76% WER in-domain); pass it with ``--model``. LoRA settings live in
``training/configs/whisper_lora.yaml`` (r / alpha / dropout / target
modules) and are validated before any model is touched.

Usage:
    # Validate everything without downloading a model:
    .venv-train/bin/python training/train_whisper_lora.py --dry-run \\
        --train-manifest training/train-manifest.jsonl \\
        --output-dir /tmp/whisper-lora-smoke
    # Smoke train (few steps, few clips, CPU):
    .venv-train/bin/python training/train_whisper_lora.py \\
        --train-manifest training/train-manifest.jsonl \\
        --output-dir /tmp/whisper-lora-smoke \\
        --max-steps 10 --max-clips 200 --batch-size 2
    # Real run (GPU runbook — see training/README.md):
    .venv-train/bin/python training/train_whisper_lora.py \\
        --model openai/whisper-large-v3-turbo \\
        --train-manifest training/train-manifest.jsonl \\
        --output-dir /tmp/whisper-lora-v1

Saves into ``--output-dir``: the LoRA adapter (``adapter_model.*`` +
``adapter_config.json``), the processor/tokenizer, and a
``train_config.json`` recording the exact config (model id, manifest path
+ SHA-256, steps, hyperparams, augmentation flags) so any result is
reproducible. The model is NEVER downloaded at import time — heavy
imports happen inside :func:`run_training` only, and a missing
torch/transformers/peft install fails with install directions pointing
at ``training/requirements.txt``.

Augmentation flags (all OFF by default; see training/augment.py):
``--augment-speed 0.9,1.1`` (random pick per clip),
``--augment-noise <noise.wav> --noise-snr-db 10``,
``--augment-phone-codec``. Augmentation quality is UNVALIDATED until
real noisy data exists — see training/README.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import repo_root  # noqa: E402
from common import (  # noqa: E402
    apply_peft_whisper_input_ids_fix,
    load_audio_16k,
    load_manifest_rows,
    manifest_sha256,
    require_heavy,
    warmup_steps_for,
)
from config import ConfigError, load_config, validate_lora_config  # noqa: E402

DEFAULT_MODEL = "openai/whisper-tiny"
REAL_MODEL = "openai/whisper-large-v3-turbo"
DEFAULT_CONFIG = repo_root() / "training" / "configs" / "whisper_lora.yaml"

# Whisper feature extractor pads/truncates to 30 s (3000 mel frames).
MAX_AUDIO_SECONDS = 30.0


def build_train_config(args, cfg, manifest_path, manifest_hash, n_clips) -> dict:
    """Assemble the reproducibility record written as train_config.json."""
    versions = {}
    for pkg in ("torch", "transformers", "peft", "accelerate"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "training/train_whisper_lora.py",
        "model": args.model,
        "config_file": str(args.config),
        "train_manifest": str(manifest_path),
        "train_manifest_sha256": manifest_hash,
        "n_clips_used": n_clips,
        "max_clips_arg": args.max_clips,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr if args.lr is not None else cfg["learning_rate"],
        "gradient_checkpointing": args.gradient_checkpointing,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "lora": {
            "r": cfg["lora_r"],
            "alpha": cfg["lora_alpha"],
            "dropout": cfg["lora_dropout"],
            "target_modules": cfg["target_modules"],
        },
        "training": {
            "num_train_epochs": cfg["num_train_epochs"],
            "warmup_ratio": cfg["warmup_ratio"],
            "logging_steps": cfg["logging_steps"],
            "save_steps": cfg["save_steps"],
        },
        "normalisation": {
            "rule": "NFC + strip + collapse whitespace "
                    "(benchmark/evaluate.py::normalize_text, via training/normalize.py)",
            "convert_devanagari_numerals": args.convert_devanagari_numerals,
        },
        "augmentation": {
            "speed_factors": args.augment_speed,
            "noise_wav": args.augment_noise,
            "noise_snr_db": args.noise_snr_db,
            "phone_codec": args.augment_phone_codec,
            "note": "Augmentation quality is UNVALIDATED until real noisy "
                    "data exists (see training/README.md).",
        },
        "versions": versions,
        "argv": sys.argv,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"HF model id (default {DEFAULT_MODEL}; real run: {REAL_MODEL})")
    p.add_argument("--train-manifest", required=True,
                   help="training manifest JSONL (from training/prepare_data.py)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--config", default=str(DEFAULT_CONFIG),
                   help="LoRA + training config YAML")
    p.add_argument("--max-steps", type=int, default=0,
                   help="max optimiser steps (0 = use num_train_epochs from config)")
    p.add_argument("--max-clips", type=int, default=None,
                   help="use at most N manifest clips")
    p.add_argument("--batch-size", type=int, default=None,
                   help="per-device train batch size (default: from config)")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=None,
                   help="learning rate (default: from config)")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--audio-dir", default=None,
                   help="base dir for relative audio_path values in the manifest")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--convert-devanagari-numerals", action="store_true",
                   help="OPT-IN experiment only (default OFF): map Devanagari "
                        "digits to Arabic digits. Training and scoring MUST "
                        "use the same value; recorded in train_config.json.")
    p.add_argument("--augment-speed", default=None,
                   help="comma-separated speed factors, e.g. 0.9,1.1 (default: off)")
    p.add_argument("--augment-noise", default=None,
                   help="noise WAV to mix in (default: off); see training/augment.py")
    p.add_argument("--noise-snr-db", type=float, default=10.0)
    p.add_argument("--augment-phone-codec", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="load config, validate manifest, print the plan, exit 0 "
                        "without touching a model")
    return p.parse_args(argv)


def print_plan(args, cfg, rows, manifest_hash):
    steps = args.max_steps or f"{cfg['num_train_epochs']} epochs"
    print("DRY RUN — Whisper LoRA training plan (no model downloaded, no training)")
    print(f"  model:            {args.model}")
    print(f"  config:           {args.config}")
    print(f"  manifest:         {args.train_manifest}")
    print(f"    sha256:         {manifest_hash}")
    print(f"    clips in file:  {len(rows)}"
          + (f" (using first {args.max_clips})" if args.max_clips else ""))
    print(f"  output dir:       {args.output_dir}")
    print(f"  steps:            {steps}")
    print(f"  batch size:       {args.batch_size or cfg['per_device_train_batch_size']}"
          f" x grad-accum {args.grad_accum}")
    print(f"  lr:               {args.lr if args.lr is not None else cfg['learning_rate']}")
    print(f"  LoRA:             r={cfg['lora_r']} alpha={cfg['lora_alpha']} "
          f"dropout={cfg['lora_dropout']} targets={cfg['target_modules']}")
    print(f"  grad ckpt:        {args.gradient_checkpointing}  fp16={args.fp16} bf16={args.bf16}")
    print(f"  workers:          {args.num_workers}  seed: {args.seed}")
    print(f"  numerals convert: {args.convert_devanagari_numerals} (default OFF)")
    print(f"  augmentation:     speed={args.augment_speed} noise={args.augment_noise} "
          f"snr={args.noise_snr_db}dB phone={args.augment_phone_codec} (all OFF unless shown)")


def run_training(args, cfg, rows, manifest_hash):
    torch = require_heavy("torch")
    transformers = require_heavy("transformers")
    peft = require_heavy("peft")
    require_heavy("accelerate")

    import numpy as np

    from augment import apply_augmentations
    from normalize import normalize_text_with_options  # noqa: E402

    # --- audio ----------------------------------------------------------------
    noise = None
    if args.augment_noise:
        noise_path = Path(args.augment_noise)
        if not noise_path.exists():
            print(f"ERROR: noise WAV not found: {noise_path}", file=sys.stderr)
            return 1
        noise, _ = load_audio_16k(noise_path)
    speed_factors = None
    if args.augment_speed:
        speed_factors = [float(s) for s in args.augment_speed.split(",")]
        if any(f <= 0 for f in speed_factors):
            print("ERROR: --augment-speed factors must be > 0", file=sys.stderr)
            return 1

    processor = transformers.WhisperProcessor.from_pretrained(
        args.model, language="ne", task="transcribe"
    )
    tokenizer = processor.tokenizer
    rng = np.random.default_rng(args.seed)

    class SpeechDataset(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            row = self.rows[idx]
            audio_path = row.get("audio_path")
            if not audio_path:
                raise ValueError(f"{row['clip_id']}: manifest has no audio_path")
            path = Path(audio_path)
            if not path.is_absolute():
                # audio_path is relative to the dir the audio was
                # materialised into; resolve it against --audio-dir.
                base = Path(args.audio_dir) if args.audio_dir else Path(".")
                path = base / path
            audio, _sr = load_audio_16k(path)
            if len(audio) > int(MAX_AUDIO_SECONDS * 16000):
                audio = audio[: int(MAX_AUDIO_SECONDS * 16000)]
            if speed_factors or noise is not None or args.augment_phone_codec:
                speed = None
                if speed_factors:
                    speed = float(rng.choice(speed_factors))
                audio = apply_augmentations(
                    audio, 16000, speed=speed, noise=noise,
                    snr_db=args.noise_snr_db,
                    phone=args.augment_phone_codec, rng=rng,
                )
            feats = processor.feature_extractor(
                audio, sampling_rate=16000, return_tensors="pt"
            ).input_features[0]
            text = normalize_text_with_options(
                row["transcript"],
                convert_devanagari_numerals=args.convert_devanagari_numerals,
            )
            labels = tokenizer(text).input_ids
            return {"input_features": feats, "labels": labels}

    def collator(batch):
        max_len = max(b["input_features"].shape[1] for b in batch)
        feats = torch.stack([
            torch.nn.functional.pad(
                b["input_features"],
                (0, max_len - b["input_features"].shape[1]))
            for b in batch
        ])
        max_lab = max(len(b["labels"]) for b in batch)
        labels = torch.full((len(batch), max_lab), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            labels[i, : len(b["labels"])] = torch.tensor(b["labels"])
        return {"input_features": feats, "labels": labels}

    model = transformers.WhisperForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float32
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    if args.gradient_checkpointing:
        model.config.use_cache = False
    peft_config = peft.LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=list(cfg["target_modules"]),
        task_type=peft.TaskType.SEQ_2_SEQ_LM,
        bias="none",
    )
    model = peft.get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # transformers 5.x + peft 0.21: peft injects input_ids=None into
    # every forward call and Whisper's decoder chokes on it — see
    # common.apply_peft_whisper_input_ids_fix.
    model = apply_peft_whisper_input_ids_fix(model)

    training_args = transformers.TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=(
            args.batch_size or cfg["per_device_train_batch_size"]
        ),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr if args.lr is not None else cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        # transformers 5.x removed warmup_ratio; convert the configured
        # ratio to steps (see training/common.py::warmup_steps_for).
        warmup_steps=warmup_steps_for(
            cfg.get("warmup_ratio", 0.0),
            len(rows),
            args.batch_size or cfg["per_device_train_batch_size"],
            args.grad_accum,
            cfg["num_train_epochs"],
            args.max_steps,
        ),
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        save_strategy="steps",
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if args.gradient_checkpointing else None
        ),
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=SpeechDataset(rows),
        data_collator=collator,
    )
    t0 = time.time()
    trainer.train()
    print(f"training took {time.time() - t0:.1f}s")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    train_config = build_train_config(
        args, cfg, args.train_manifest, manifest_hash, len(rows)
    )
    train_config["steps_trained"] = trainer.state.global_step
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "train_config.json").write_text(
        json.dumps(train_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"saved adapter + tokenizer + train_config.json to {args.output_dir}")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        cfg = validate_lora_config(load_config(args.config))
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    manifest_path = Path(args.train_manifest)
    if not manifest_path.exists():
        print(f"ERROR: train manifest not found: {manifest_path}", file=sys.stderr)
        return 1
    try:
        rows = load_manifest_rows(manifest_path, max_clips=args.max_clips)
    except ValueError as exc:
        print(f"ERROR: invalid train manifest: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if not rows:
        print("ERROR: train manifest is empty", file=sys.stderr)
        return 1
    manifest_hash = manifest_sha256(manifest_path)
    if args.dry_run:
        print_plan(args, cfg, rows, manifest_hash)
        return 0
    return run_training(args, cfg, rows, manifest_hash)


if __name__ == "__main__":
    raise SystemExit(main())
