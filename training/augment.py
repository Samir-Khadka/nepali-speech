"""Cheap, correct audio augmentation — all OFF by default.

Three transforms, each behind its own flag in the train scripts:

* speed perturbation — resample so the clip plays faster/slower at the
  same sample rate (pitch shifts with speed, as with a real faster
  speaker; this is the standard Kaldi-style perturbation).
* additive noise — mix in a short noise sample at a configurable SNR.
  The only noise shipped with this repo is a SYNTHETIC brown-noise WAV
  generated locally by :func:`make_synthetic_noise_wav` (see below); it is
  labelled synthetic and is NOT speech. Real recorded noise must replace
  it before any claim about noisy-speech robustness is made.
* phone-codec simulation — downsample to 8 kHz and back up to 16 kHz,
  which approximates the bandwidth loss of a phone call. It is NOT a real
  codec (no AMR/G.711 encode step); it is a bandwidth proxy.

Everything here is pure functions on numpy arrays (``float32``, mono,
values roughly in [-1, 1]) — no files, no network, no TTS, no synthesised
speech. Unit-testable without torch: resampling uses librosa when it is
installed and a numpy linear-interpolation fallback otherwise.

WARNING (also in training/README.md): augmentation quality is UNVALIDATED
until real noisy / phone-quality Nepali data exists. These transforms make
training harder in a plausible direction; they do not prove the model
handles real phone audio. Do not report augmentation as a robustness
result.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

__all__ = [
    "resample_array",
    "speed_perturb",
    "add_noise",
    "generate_brown_noise",
    "make_synthetic_noise_wav",
    "phone_codec",
    "apply_augmentations",
]

_NOISE_NOTE = (
    "synthetic brown noise for augmentation only — NOT speech, NOT a "
    "recording of any environment"
)


def resample_array(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a mono float array from ``orig_sr`` to ``target_sr``.

    Uses librosa when available (higher quality); otherwise falls back to
    numpy linear interpolation, which is exact enough for the augmentation
    use case and keeps this module testable without librosa.
    """
    y = np.asarray(y, dtype=np.float32)
    if orig_sr == target_sr:
        return y.copy()
    try:
        import librosa

        return np.asarray(
            librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr),
            dtype=np.float32,
        )
    except ImportError:
        n_out = max(1, int(round(len(y) * target_sr / orig_sr)))
        x_old = np.linspace(0.0, 1.0, len(y), dtype=np.float64)
        x_new = np.linspace(0.0, 1.0, n_out, dtype=np.float64)
        return np.interp(x_new, x_old, y).astype(np.float32)


def speed_perturb(y: np.ndarray, sr: int, factor: float) -> np.ndarray:
    """Play the clip ``factor``x faster (``factor > 1``) or slower (< 1).

    Implemented by resampling to ``sr / factor`` and interpreting the
    result at ``sr``: the duration becomes ``len / factor`` and the pitch
    shifts with the speed, matching a speaker talking faster/slower.
    ``factor`` must be > 0; typical values are 0.9 / 1.0 / 1.1.
    """
    if factor <= 0:
        raise ValueError(f"speed factor must be > 0, got {factor}")
    if factor == 1.0:
        return np.asarray(y, dtype=np.float32).copy()
    target_sr = max(1, int(round(sr / factor)))
    return resample_array(np.asarray(y, dtype=np.float32), sr, target_sr)


def add_noise(
    y: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Mix ``noise`` into ``y`` at the given signal-to-noise ratio (dB).

    The noise is scaled so that ``10*log10(P_signal / P_noise) == snr_db``.
    If the noise is shorter than the signal it is tiled; if longer, a
    random contiguous crop is taken (``rng`` seeds that choice; a fixed
    ``rng`` makes the transform reproducible). Returns a float32 array;
    values are NOT hard-clipped — the caller / feature extractor handles
    range.
    """
    y = np.asarray(y, dtype=np.float32)
    noise = np.asarray(noise, dtype=np.float32)
    if len(noise) == 0:
        raise ValueError("noise sample is empty")
    if len(noise) < len(y):
        repeats = int(np.ceil(len(y) / len(noise)))
        noise = np.tile(noise, repeats)
    if len(noise) > len(y):
        rng = rng or np.random.default_rng(0)
        start = int(rng.integers(0, len(noise) - len(y) + 1))
        noise = noise[start : start + len(y)]
    else:
        noise = noise[: len(y)]
    sig_power = float(np.mean(y.astype(np.float64) ** 2)) + 1e-12
    noise_power = float(np.mean(noise.astype(np.float64) ** 2)) + 1e-12
    target_noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    scale = float(np.sqrt(target_noise_power / noise_power))
    return (y + noise * scale).astype(np.float32)


def generate_brown_noise(
    duration_s: float, sr: int = 16000, seed: int = 0
) -> np.ndarray:
    """Generate synthetic brown noise (integrated white noise), normalised.

    Brown noise has a 1/f^2 spectrum — a rough stand-in for low-frequency
    environmental rumble. It is SYNTHETIC (see module docstring): generated
    locally from a seeded RNG, labelled as such wherever it is saved, and
    it contains no speech and no recording of any real environment.
    """
    if duration_s <= 0:
        raise ValueError("duration_s must be > 0")
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(int(round(duration_s * sr))).astype(np.float64)
    brown = np.cumsum(white)
    brown -= brown.mean()
    peak = np.max(np.abs(brown))
    if peak > 0:
        brown /= peak
    return (brown * 0.5).astype(np.float32)


def make_synthetic_noise_wav(
    path: str | Path, duration_s: float = 5.0, sr: int = 16000, seed: int = 0
) -> Path:
    """Write the synthetic brown-noise sample to ``path`` as 16-bit PCM WAV.

    The file is labelled synthetic in its companion ``.txt`` sidecar, which
    is written next to it. Returns the WAV path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    noise = generate_brown_noise(duration_s, sr=sr, seed=seed)
    pcm = np.clip(noise, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(pcm16.tobytes())
    sidecar = path.with_suffix(".txt")
    sidecar.write_text(
        f"Synthetic noise sample for augmentation only.\n"
        f"Generator: training/augment.py::generate_brown_noise "
        f"(seed={seed}, duration_s={duration_s}, sr={sr}).\n"
        f"{_NOISE_NOTE}.\n",
        encoding="utf-8",
    )
    return path


def phone_codec(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Simulate phone bandwidth: downsample to 8 kHz, then back to ``sr``.

    This is a BANDWIDTH proxy only — it drops everything above 4 kHz and
    re-upsamples. It does not run a real codec (AMR/G.711), so it must not
    be described as codec-accurate. ``sr`` must be a multiple of 8000 and
    >= 16000 for the 8 kHz step to be exact; 16000 is the expected input.
    """
    y = np.asarray(y, dtype=np.float32)
    if sr < 8000 or sr % 8000 != 0:
        raise ValueError(f"phone_codec expects sr a multiple of 8000, got {sr}")
    down = resample_array(y, sr, 8000)
    return resample_array(down, 8000, sr)


def apply_augmentations(
    y: np.ndarray,
    sr: int = 16000,
    *,
    speed: float | None = None,
    noise: np.ndarray | None = None,
    snr_db: float = 10.0,
    phone: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply the enabled augmentations in a fixed order: speed, noise, phone.

    Every argument defaults to "off" (``speed=None``, ``noise=None``,
    ``phone=False``), so calling this with no flags is the identity — the
    train scripts rely on that for the default no-augmentation run.
    """
    out = np.asarray(y, dtype=np.float32)
    if speed is not None and speed != 1.0:
        out = speed_perturb(out, sr, speed)
    if noise is not None:
        out = add_noise(out, noise, snr_db, rng=rng)
    if phone:
        out = phone_codec(out, sr)
    return out
