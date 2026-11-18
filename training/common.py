"""Shared helpers for the training scripts: manifest handling, audio loading,
and the clean-failure path for missing heavy dependencies.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import load_benchmark_modules  # noqa: E402

__all__ = [
    "manifest_sha256",
    "load_manifest_rows",
    "load_audio_16k",
    "require_heavy",
    "warmup_steps_for",
    "apply_peft_whisper_input_ids_fix",
    "MISSING_HEAVY_MESSAGE",
]

MISSING_HEAVY_MESSAGE = (
    "Missing training dependency: {name}. Training needs the separate "
    "training venv — see training/README.md §0 and install with:\n"
    "  python -m venv .venv-train\n"
    "  .venv-train/bin/pip install --index-url "
    "https://download.pytorch.org/whl/cpu torch\n"
    "  .venv-train/bin/pip install -r training/requirements.txt\n"
    "Do NOT install torch into the repo's main .venv (it must stay lean "
    "for backend/CI)."
)

_EVALUATE = None


def _evaluate_module():
    """Load benchmark/evaluate.py on first use (it needs jiwer).

    Loading it lazily keeps this module importable in a venv without the
    heavy deps, so scripts can fail with the clean message above instead
    of a bare ModuleNotFoundError at import time.
    """
    global _EVALUATE
    if _EVALUATE is None:
        try:
            _build, _EVALUATE = load_benchmark_modules()
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError(
                MISSING_HEAVY_MESSAGE.format(name="jiwer")
            ) from exc
    return _EVALUATE


def manifest_sha256(path: str | Path) -> str:
    """SHA-256 hex digest of a manifest file (reproducibility record)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest_rows(path: str | Path, max_clips: int | None = None) -> list[dict]:
    """Load a training/test manifest JSONL via the benchmark harness loader.

    Uses ``benchmark/evaluate.py::load_manifest`` (same schema validation
    as scoring: required clip_id/source/speaker/transcript, no duplicates),
    so a manifest that scores is a manifest that trains. Raises
    ``ValueError`` (from the harness) on an empty or malformed manifest.
    ``max_clips`` caps the rows returned, in file order.
    """
    rows = _evaluate_module().load_manifest(path)
    if max_clips is not None:
        rows = rows[:max_clips]
    return rows


def load_audio_16k(path: str | Path) -> tuple["object", int]:
    """Load an audio file as a mono float32 numpy array at 16 kHz.

    Returns ``(samples, 16000)``. Multi-channel audio is averaged to mono;
    any sample rate is resampled to 16 kHz (librosa if installed, else the
    numpy fallback in ``training/augment.py``). Requires ``soundfile``.
    """
    try:
        import soundfile as sf
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError(MISSING_HEAVY_MESSAGE.format(name="soundfile")) from exc
    import numpy as np

    from augment import resample_array

    data, sr = sf.read(str(path), always_2d=False)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1).astype(np.float32)
    if sr != 16000:
        data = resample_array(data, sr, 16000)
    return data, 16000


def warmup_steps_for(warmup_ratio, n_rows, per_device_batch_size,
                     grad_accum, num_train_epochs, max_steps):
    """Convert the configured warmup RATIO to warmup STEPS.

    transformers 5.x removed ``TrainingArguments(warmup_ratio=...)``;
    only ``warmup_steps`` survives. The YAML configs still carry
    ``warmup_ratio`` (0.03), so this converts it from the planned total:
    ``max_steps`` wins when positive, else
    ``ceil(n_rows / effective_batch) * num_train_epochs``.
    """
    if max_steps and max_steps > 0:
        total = max_steps
    else:
        effective = per_device_batch_size * grad_accum
        steps_per_epoch = -(-n_rows // effective)
        total = steps_per_epoch * num_train_epochs
    return int(warmup_ratio * total)


def apply_peft_whisper_input_ids_fix(peft_model):
    """Drop peft's spurious ``None`` kwargs before they reach Whisper.

    Compatibility workaround for transformers 5.x + peft 0.21:
    ``PeftModelForSeq2SeqLM.forward`` always passes ``input_ids=None``
    and ``inputs_embeds=None`` explicitly to the base model, and
    transformers 5.x Whisper forwards unknown ``**kwargs`` down to the
    decoder, where they collide with the decoder's own explicit
    ``input_ids=decoder_input_ids`` /
    ``inputs_embeds=decoder_inputs_embeds`` ("got multiple values for
    keyword argument ..."). Whisper takes ``input_features``, never
    ``input_ids``/``inputs_embeds``, so the injected values are always
    ``None`` here; only ``None`` values are dropped, so a genuine
    caller-supplied tensor would still surface as an error instead of
    being silently swallowed. (transformers 4.x did not forward
    ``**kwargs`` to the decoder, which is why this combination worked
    there.)
    """
    hf_model = peft_model.get_base_model()
    orig_forward = hf_model.forward

    def _whisper_forward(*args, **kwargs):
        for _name in ("input_ids", "inputs_embeds"):
            if kwargs.get(_name) is None:
                kwargs.pop(_name, None)
        return orig_forward(*args, **kwargs)

    hf_model.forward = _whisper_forward
    return peft_model


def require_heavy(name: str):
    """Import a heavy training dependency, or fail with install directions.

    ``name`` is one of ``"torch"``, ``"transformers"``, ``"peft"``,
    ``"accelerate"``. Returns the imported module. The message names the
    missing package and points at ``training/requirements.txt`` — a bare
    ``ModuleNotFoundError`` traceback is not an acceptable failure mode.
    """
    try:
        if name == "torch":
            import torch

            return torch
        if name == "transformers":
            import transformers

            return transformers
        if name == "peft":
            import peft

            return peft
        if name == "accelerate":
            import accelerate

            return accelerate
    except ImportError as exc:  # noqa: BLE001
        raise SystemExit(MISSING_HEAVY_MESSAGE.format(name=name)) from exc
    raise ValueError(f"unknown heavy dependency: {name!r}")
