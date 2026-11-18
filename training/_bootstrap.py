"""Shared import bootstrap for the training package.

The training scripts are run either as ``python training/<script>.py`` from
the repo root or as ``python -m training.<script>``. In both cases they need
the benchmark modules (``benchmark/build.py`` and ``benchmark/evaluate.py``)
without the benchmark directory being a package. This helper puts the repo
root and the benchmark directory on ``sys.path`` (the same convention as
``benchmark/tests/conftest.py``) and then imports those modules by name.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BENCHMARK_DIR = _REPO_ROOT / "benchmark"


def ensure_repo_on_path() -> Path:
    """Insert repo root + benchmark dir at the front of ``sys.path``.

    Returns the repo root. Idempotent.
    """
    for entry in (str(_BENCHMARK_DIR), str(_REPO_ROOT)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return _REPO_ROOT


def load_benchmark_modules():
    """Return ``(build, evaluate)`` — the benchmark builder and harness.

    Imports are done here (not at module top) so that a plain
    ``--help`` / ``--dry-run`` on a training script never pays for the
    benchmark's heavy imports, and so the ImportError message can point
    at the real cause.
    """
    ensure_repo_on_path()
    import build  # noqa: E402  (benchmark/build.py)
    import evaluate  # noqa: E402  (benchmark/evaluate.py)

    return build, evaluate


def repo_root() -> Path:
    return _REPO_ROOT
