"""Training package init. Heavy deps (torch/transformers/peft) are imported
lazily inside the train/score scripts, never here, so ``--help`` and
``--dry-run`` work in a venv without them.
"""
