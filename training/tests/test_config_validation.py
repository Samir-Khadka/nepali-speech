"""Config validation tests — both YAMLs load, bad configs fail loudly.

Covers ``training/config.py`` (load + validate) and the train scripts'
CLI-level rejection: a config with a bad LoRA r, a bad dropout, a missing
required key, or a missing ``model`` key must be rejected with a
``ConfigError`` (or a non-zero CLI exit printing ERROR) that names the
problem — never silently, and never halfway through a GPU run. All
offline; no model is loaded.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_ctc  # noqa: E402
import train_whisper_lora  # noqa: E402
from config import (  # noqa: E402
    ConfigError,
    load_config,
    validate_config,
    validate_ctc_config,
    validate_lora_config,
)

ROOT = Path(__file__).resolve().parents[1]
LORA_YAML = ROOT / "configs" / "whisper_lora.yaml"
CTC_YAML = ROOT / "configs" / "ctc.yaml"


# --- shipped configs ---------------------------------------------------------

def test_both_yaml_configs_load_and_validate():
    lora = load_config(LORA_YAML)
    assert validate_lora_config(lora) is lora
    ctc = load_config(CTC_YAML)
    assert validate_ctc_config(ctc) is ctc


def test_shipped_configs_have_all_required_keys():
    lora = load_config(LORA_YAML)
    for key in (
        "lora_r", "lora_alpha", "lora_dropout", "target_modules",
        "learning_rate", "per_device_train_batch_size", "num_train_epochs",
        "warmup_ratio", "logging_steps", "save_steps",
    ):
        assert key in lora, f"whisper_lora.yaml missing {key}"
    assert lora["lora_r"] == 8
    assert lora["target_modules"] == ["q_proj", "v_proj"]
    ctc = load_config(CTC_YAML)
    for key in (
        "learning_rate", "per_device_train_batch_size", "num_train_epochs",
        "warmup_ratio", "logging_steps", "save_steps",
        "freeze_feature_extractor",
    ):
        assert key in ctc, f"ctc.yaml missing {key}"
    assert ctc["freeze_feature_extractor"] is True


# --- invalid LoRA configs ------------------------------------------------------

@pytest.mark.parametrize("bad_r", [0, -1, -8])
def test_bad_lora_r_rejected(bad_r):
    cfg = dict(load_config(LORA_YAML), lora_r=bad_r)
    with pytest.raises(ConfigError, match="lora_r"):
        validate_lora_config(cfg)


@pytest.mark.parametrize("bad_alpha", [0, -16])
def test_bad_lora_alpha_rejected(bad_alpha):
    cfg = dict(load_config(LORA_YAML), lora_alpha=bad_alpha)
    with pytest.raises(ConfigError, match="lora_alpha"):
        validate_lora_config(cfg)


@pytest.mark.parametrize("bad_dropout", [-0.1, 1.0, 1.5])
def test_bad_lora_dropout_rejected(bad_dropout):
    cfg = dict(load_config(LORA_YAML), lora_dropout=bad_dropout)
    with pytest.raises(ConfigError, match="lora_dropout"):
        validate_lora_config(cfg)


@pytest.mark.parametrize("bad_modules", [[], [""], [None], ["q_proj", 3]])
def test_bad_target_modules_rejected(bad_modules):
    cfg = dict(load_config(LORA_YAML), target_modules=bad_modules)
    with pytest.raises(ConfigError, match="target_modules"):
        validate_lora_config(cfg)


@pytest.mark.parametrize("bad_lr", [0, -0.001])
def test_bad_learning_rate_rejected(bad_lr):
    cfg = dict(load_config(LORA_YAML), learning_rate=bad_lr)
    with pytest.raises(ConfigError, match="learning_rate"):
        validate_lora_config(cfg)
    cfg = dict(load_config(CTC_YAML), learning_rate=bad_lr)
    with pytest.raises(ConfigError, match="learning_rate"):
        validate_ctc_config(cfg)


def test_missing_required_key_rejected_and_named():
    cfg = dict(load_config(LORA_YAML))
    del cfg["lora_r"]
    with pytest.raises(ConfigError, match="lora_r"):
        validate_lora_config(cfg)
    cfg = dict(load_config(CTC_YAML))
    del cfg["freeze_feature_extractor"]
    with pytest.raises(ConfigError, match="freeze_feature_extractor"):
        validate_ctc_config(cfg)


def test_wrong_type_rejected():
    cfg = dict(load_config(LORA_YAML), lora_r="8")
    with pytest.raises(ConfigError, match="lora_r"):
        validate_lora_config(cfg)


# --- "missing model": a config that must name a model, but doesn't -----------

def test_missing_model_key_rejected_with_clear_error():
    """The validators reject a config that omits a required key and name it.

    The shipped YAMLs carry hyperparams only; the model id comes from the
    CLI (--model). If a future config is required to name its model, the
    same mechanism must reject its absence — this pins that behaviour.
    """
    cfg = {"learning_rate": 0.0001}
    with pytest.raises(ConfigError, match="missing required key: 'model'"):
        validate_config(cfg, {"model": str, "learning_rate": float},
                        name="training config")
    with pytest.raises(ConfigError, match="missing required key: 'model'"):
        validate_config({}, {"model": str}, name="training config")


# --- load_config failures ------------------------------------------------------

def test_load_config_missing_file_bad_yaml_and_non_mapping(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="cannot parse"):
        load_config(bad)
    lst = tmp_path / "list.yaml"
    lst.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(lst)
    txt = tmp_path / "notes.txt"
    txt.write_text("hello", encoding="utf-8")
    with pytest.raises(ConfigError, match="unsupported config suffix"):
        load_config(txt)


def test_load_config_json_round_trip(tmp_path):
    import json

    cfg = {"model": "openai/whisper-tiny", "learning_rate": 0.0001}
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    assert load_config(path) == cfg


# --- CLI level: bad config exits 1 with ERROR before any model is touched -----

def test_train_scripts_reject_bad_config_at_cli(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        (Path(LORA_YAML).read_text(encoding="utf-8").replace("lora_r: 8", "lora_r: 0")),
        encoding="utf-8",
    )
    rc = train_whisper_lora.main(
        ["--config", str(bad), "--train-manifest", "train.jsonl",
         "--output-dir", str(tmp_path / "out")]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "ERROR" in err
    assert "lora_r" in err
    bad_ctc = tmp_path / "bad_ctc.yaml"
    bad_ctc.write_text(
        (Path(CTC_YAML).read_text(encoding="utf-8")
         .replace("learning_rate: 0.0001", "learning_rate: 0")),
        encoding="utf-8",
    )
    rc = train_ctc.main(
        ["--config", str(bad_ctc), "--train-manifest", "train.jsonl",
         "--output-dir", str(tmp_path / "out2")]
    )
    assert rc == 1
    assert "ERROR" in capsys.readouterr().err
