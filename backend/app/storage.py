"""Audio file storage (see CONTRACT.md).

Data root is ``DATA_DIR`` (default ``<project root>/data``). Audio lives at
``data/audio/<recording_id>.<ext>``.
"""

import os
from pathlib import Path

MIME_BY_EXT = {
    "webm": "audio/webm",
    "ogg": "audio/ogg",
    "wav": "audio/wav",
    "mp4": "audio/mp4",
    "mp3": "audio/mpeg",
}

EXT_BY_MIME = {
    "audio/webm": "webm",
    "video/webm": "webm",
    "audio/ogg": "ogg",
    "audio/wav": "wav",
    "audio/mp4": "mp4",
    "audio/mpeg": "mp3",
}

_data_dir: Path | None = None


def default_data_dir() -> Path:
    # storage.py is backend/app/storage.py -> project root is parents[2].
    return Path(__file__).resolve().parents[2] / "data"


def configure(data_dir: str | Path | None) -> None:
    global _data_dir
    _data_dir = Path(data_dir).resolve() if data_dir else None


def get_data_dir() -> Path:
    if _data_dir is not None:
        return _data_dir
    return Path(os.environ.get("DATA_DIR", str(default_data_dir()))).resolve()


def audio_dir() -> Path:
    d = get_data_dir() / "audio"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_audio(data: bytes, recording_id: int, ext: str) -> str:
    """Write audio bytes to ``data/audio/<recording_id>.<ext>``; return the path."""
    path = audio_dir() / f"{recording_id}.{ext}"
    path.write_bytes(data)
    return str(path)


def read_audio(recording_id: int) -> tuple[bytes, str]:
    """Return ``(bytes, mime)`` for a recording id; raises FileNotFoundError."""
    d = get_data_dir() / "audio"
    matches = sorted(d.glob(f"{recording_id}.*")) if d.is_dir() else []
    if not matches:
        raise FileNotFoundError(f"no audio file for recording {recording_id}")
    path = matches[0]
    mime = MIME_BY_EXT.get(path.suffix.lstrip(".").lower(), "application/octet-stream")
    return path.read_bytes(), mime
