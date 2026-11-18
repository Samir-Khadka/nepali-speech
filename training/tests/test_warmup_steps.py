"""Regression test for the transformers 5.x warmup_ratio removal.

transformers 5.23.0 dropped ``TrainingArguments(warmup_ratio=...)``; the
training scripts now convert the configured ratio to steps via
``training/common.py::warmup_steps_for``. If a future transformers
release changes the API again, these tests fail loudly instead of
surfacing as a mid-run TypeError after the model has downloaded.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import warmup_steps_for  # noqa: E402


def test_warmup_steps_from_max_steps():
    # 0.03 * 1000 steps = 30.
    assert warmup_steps_for(0.03, 300, 4, 1, 3, 1000) == 30


def test_warmup_steps_from_epochs():
    # No max_steps: 300 rows / batch 4 = 75 steps/epoch * 3 epochs = 225;
    # 0.03 * 225 = 6 (int truncation).
    assert warmup_steps_for(0.03, 300, 4, 1, 3, 0) == 6
    assert warmup_steps_for(0.03, 300, 4, 1, 3, -1) == 6


def test_warmup_steps_grad_accum_and_zero_ratio():
    # Effective batch 4*2=8: 300 rows -> 38 steps/epoch (ceil) * 3 = 114;
    # 0.03 * 114 = 3.
    assert warmup_steps_for(0.03, 300, 4, 2, 3, 0) == 3
    assert warmup_steps_for(0.0, 300, 4, 1, 3, 1000) == 0


def test_training_arguments_accepts_warmup_steps():
    import transformers

    args = transformers.TrainingArguments(
        output_dir="/tmp/warmup-steps-regression",
        warmup_steps=6,
        report_to=[],
    )
    assert args.warmup_steps == 6
