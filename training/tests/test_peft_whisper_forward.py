"""Regression test for the peft 0.21 + transformers 5.x Whisper clash.

``PeftModelForSeq2SeqLM.forward`` always passes ``input_ids=None`` and
``inputs_embeds=None`` explicitly to the base model, and transformers
5.x Whisper forwards unknown ``**kwargs`` down to the decoder, where
they collide with the decoder's own explicit arguments::

    TypeError: WhisperDecoder.forward() got multiple values for
    keyword argument 'input_ids'  (then 'inputs_embeds')

``training/common.py::apply_peft_whisper_input_ids_fix`` drops the
spurious ``None`` kwargs on the underlying HF model. These tests pin
that behaviour with a tiny randomly-initialised Whisper (no weights
are downloaded) called exactly the way ``Trainer.compute_loss`` calls
it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import apply_peft_whisper_input_ids_fix  # noqa: E402


def _peft_whisper():
    import peft
    import transformers

    cfg = transformers.WhisperConfig(
        vocab_size=100,
        d_model=64,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=128,
        decoder_ffn_dim=128,
        num_mel_bins=80,
        max_source_positions=1500,
        max_target_positions=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=3,
    )
    model = transformers.WhisperForConditionalGeneration(cfg)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    peft_model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            task_type=peft.TaskType.SEQ_2_SEQ_LM,
            bias="none",
        ),
    )
    return peft_model


def test_peft_whisper_forward_fails_without_fix():
    import torch

    model = _peft_whisper()
    feats = torch.randn(1, 80, 3000)
    labels = torch.tensor([[3, 10, 20, 30, 2]])
    try:
        model(input_features=feats, labels=labels)
    except TypeError as exc:
        assert "input_ids" in str(exc)
    else:
        raise AssertionError(
            "expected the peft/transformers input_ids clash without the fix"
        )


def test_peft_whisper_forward_works_with_fix():
    import torch

    model = apply_peft_whisper_input_ids_fix(_peft_whisper())
    feats = torch.randn(1, 80, 3000)
    labels = torch.tensor([[3, 10, 20, 30, 2]])
    out = model(input_features=feats, labels=labels)
    assert out.loss is not None
    assert torch.isfinite(out.loss).item()


def test_peft_whisper_generate_with_fix():
    # transformers 5.x generate() takes inputs as keywords only, and
    # peft's generate() wrapper takes no positional args at all — the
    # scoring script must call model.generate(input_features=...).
    # `forced_decoder_ids` is deprecated in 5.x in favour of
    # `language` + `task`; the random config has no lang_to_id table,
    # so the test passes `task` only (the real model's language path
    # is covered by the actual scoring run). This pins the keyword
    # form end to end (including the forward fix, which generate()
    # also goes through).
    import torch

    model = apply_peft_whisper_input_ids_fix(_peft_whisper())
    # The random config's generation config has no task_to_id table
    # (real Whisper checkpoints ship one); give it in-vocab ids so
    # the `task` kwarg is accepted. Language is omitted here — the
    # real model's language path is covered by the actual scoring run.
    model.generation_config.task_to_id = {
        "transcribe": 50, "translate": 60}
    feats = torch.randn(1, 80, 3000)
    ids = model.generate(
        input_features=feats,
        task="transcribe",
        max_new_tokens=5,
    )
    assert ids.shape[0] == 1
    assert ids.shape[1] >= 1
