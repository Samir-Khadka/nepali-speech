"""Tests for training/augment.py — offline, pure numpy, no TTS, no speech.

These run in the repo's main venv (no librosa needed: the numpy fallback
is exercised when librosa is absent, and librosa when present).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from augment import (  # noqa: E402
    add_noise,
    apply_augmentations,
    generate_brown_noise,
    make_synthetic_noise_wav,
    phone_codec,
    speed_perturb,
)

SR = 16000


def _sine(seconds=1.0, freq=440.0):
    t = np.arange(int(seconds * SR), dtype=np.float64) / SR
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_speed_perturb_changes_duration_not_sr():
    y = _sine(1.0)
    faster = speed_perturb(y, SR, 2.0)
    assert len(faster) == pytest.approx(len(y) / 2, rel=0.02)
    slower = speed_perturb(y, SR, 0.5)
    assert len(slower) == pytest.approx(len(y) * 2, rel=0.02)
    assert speed_perturb(y, SR, 1.0).shape == y.shape
    with pytest.raises(ValueError):
        speed_perturb(y, SR, 0.0)


def test_add_noise_hits_target_snr():
    rng = np.random.default_rng(7)
    y = _sine(1.0)
    noise = rng.standard_normal(len(y)).astype(np.float32)
    mixed = add_noise(y, noise, snr_db=10.0, rng=np.random.default_rng(0))
    assert mixed.shape == y.shape
    # Recover the added noise (same seed/crop) and measure the SNR.
    rng2 = np.random.default_rng(0)
    _ = rng2  # add_noise consumed no RNG here (noise already full length)
    residual = mixed - y
    sig_p = float(np.mean(y.astype(np.float64) ** 2))
    noise_p = float(np.mean(residual.astype(np.float64) ** 2))
    snr = 10 * np.log10(sig_p / noise_p)
    assert snr == pytest.approx(10.0, abs=0.5)


def test_add_noise_tiles_short_noise():
    y = _sine(1.0)
    short = np.ones(100, dtype=np.float32)
    out = add_noise(y, short, snr_db=0.0, rng=np.random.default_rng(1))
    assert out.shape == y.shape
    with pytest.raises(ValueError):
        add_noise(y, np.zeros(0, dtype=np.float32), 10.0)


def test_brown_noise_is_synthetic_deterministic_and_labelled(tmp_path):
    a = generate_brown_noise(1.0, sr=SR, seed=3)
    b = generate_brown_noise(1.0, sr=SR, seed=3)
    assert np.array_equal(a, b)
    assert a.dtype == np.float32 and len(a) == SR
    wav = make_synthetic_noise_wav(tmp_path / "noise.wav", duration_s=1.0, seed=3)
    assert wav.exists()
    sidecar = wav.with_suffix(".txt")
    assert sidecar.exists()
    text = sidecar.read_text(encoding="utf-8").lower()
    assert "synthetic" in text and "not speech" in text


def test_phone_codec_bandwidth_proxy():
    y = _sine(1.0)
    out = phone_codec(y, SR)
    assert out.shape == y.shape
    # 8 kHz round-trip removes high frequencies: a 7 kHz tone is gone.
    high = _sine(1.0, freq=7000.0)
    out_high = phone_codec(high, SR)
    assert float(np.mean(out_high**2)) < 0.05 * float(np.mean(high**2))
    with pytest.raises(ValueError):
        phone_codec(y, 22050)


def test_apply_augmentations_default_is_identity():
    y = _sine(0.5)
    assert np.array_equal(apply_augmentations(y, SR), y)
    out = apply_augmentations(y, SR, speed=2.0, phone=True)
    assert len(out) == pytest.approx(len(y) / 2, rel=0.02)
