"""Config loading + validation for the training package.

Configs live in ``training/configs/`` as YAML (or JSON). A config is loaded
with :func:`load_config` and validated against a small schema with
:func:`validate_config`; both train scripts call these before touching a
model, so a typo'd config fails fast with a named error instead of halfway
through a GPU run. ``--dry-run`` exercises exactly this path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

class ConfigError(ValueError):
    """A training config is missing, unreadable, or invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON config file into a dict.

    The format is chosen by suffix: ``.yaml`` / ``.yml`` -> YAML,
    ``.json`` -> JSON. Raises :class:`ConfigError` on any problem.
    PyYAML is imported lazily so that importing this module never fails
    on a missing optional dependency — the error names the fix.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    try:
        if path.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:  # noqa: BLE001
                raise ConfigError(
                    "pyyaml is not installed — install the training deps "
                    "into .venv-train (see training/requirements.txt: "
                    ".venv-train/bin/pip install -r training/requirements.txt)"
                ) from exc
            try:
                data = yaml.safe_load(text)
            except yaml.YAMLError as exc:
                raise ConfigError(f"cannot parse config {path}: {exc}") from exc
        elif path.suffix.lower() == ".json":
            import json

            data = json.loads(text)
        else:
            raise ConfigError(
                f"unsupported config suffix {path.suffix!r} "
                "(use .yaml/.yml/.json)"
            )
    except ConfigError:
        raise
    except ValueError as exc:
        raise ConfigError(f"cannot parse config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config {path} must be a mapping, got {type(data)}")
    return data


def validate_config(
    cfg: dict[str, Any], required: dict[str, type], *, name: str = "config"
) -> dict[str, Any]:
    """Check required keys exist and have the expected type.

    ``required`` maps key -> expected type (or a tuple of types). Returns
    the config unchanged so callers can chain ``load_config`` +
    ``validate_config``. Raises :class:`ConfigError` listing every problem
    found, not just the first.
    """
    problems: list[str] = []
    for key, typ in required.items():
        if key not in cfg:
            problems.append(f"missing required key: {key!r}")
        elif not isinstance(cfg[key], typ):
            problems.append(
                f"key {key!r} must be {typ.__name__ if isinstance(typ, type) else typ}, "
                f"got {type(cfg[key]).__name__}"
            )
    if problems:
        raise ConfigError(f"{name} is invalid: " + "; ".join(problems))
    return cfg


# --- Per-script schemas ----------------------------------------------------

LORA_REQUIRED: dict[str, type] = {
    "lora_r": int,
    "lora_alpha": int,
    "lora_dropout": float,
    "target_modules": list,
}

TRAINING_REQUIRED: dict[str, type] = {
    "learning_rate": float,
    "per_device_train_batch_size": int,
    "num_train_epochs": (int, float),
    "warmup_ratio": float,
    "logging_steps": int,
    "save_steps": int,
}


def validate_lora_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate a Whisper LoRA config (``training/configs/whisper_lora.yaml``)."""
    validate_config(cfg, LORA_REQUIRED, name="LoRA config")
    validate_config(cfg, TRAINING_REQUIRED, name="LoRA config")
    problems: list[str] = []
    if cfg["lora_r"] <= 0:
        problems.append("lora_r must be > 0")
    if cfg["lora_alpha"] <= 0:
        problems.append("lora_alpha must be > 0")
    if not 0.0 <= cfg["lora_dropout"] < 1.0:
        problems.append("lora_dropout must be in [0, 1)")
    if not cfg["target_modules"] or not all(
        isinstance(m, str) and m for m in cfg["target_modules"]
    ):
        problems.append("target_modules must be a non-empty list of strings")
    if cfg["learning_rate"] <= 0:
        problems.append("learning_rate must be > 0")
    if cfg["per_device_train_batch_size"] <= 0:
        problems.append("per_device_train_batch_size must be > 0")
    if problems:
        raise ConfigError("LoRA config is invalid: " + "; ".join(problems))
    return cfg


CTC_REQUIRED: dict[str, type] = {
    "learning_rate": float,
    "per_device_train_batch_size": int,
    "num_train_epochs": (int, float),
    "warmup_ratio": float,
    "logging_steps": int,
    "save_steps": int,
    "freeze_feature_extractor": bool,
}


def validate_ctc_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate a CTC config (``training/configs/ctc.yaml``)."""
    validate_config(cfg, CTC_REQUIRED, name="CTC config")
    problems: list[str] = []
    if cfg["learning_rate"] <= 0:
        problems.append("learning_rate must be > 0")
    if cfg["per_device_train_batch_size"] <= 0:
        problems.append("per_device_train_batch_size must be > 0")
    if problems:
        raise ConfigError("CTC config is invalid: " + "; ".join(problems))
    return cfg
