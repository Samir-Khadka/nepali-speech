"""Seed prompts from prompts/prompts.json on startup, only when empty."""

import json
from pathlib import Path

from sqlalchemy.orm import Session

from .models import Prompt


def prompts_file() -> Path:
    # seed.py is backend/app/seed.py -> project root is parents[2].
    return Path(__file__).resolve().parents[2] / "prompts" / "prompts.json"


def seed_prompts(db: Session) -> int:
    """Insert prompts from prompts.json if the prompts table is empty.

    Returns the number of rows inserted (0 when already seeded).
    """
    if db.query(Prompt).count() > 0:
        return 0
    path = prompts_file()
    with path.open(encoding="utf-8") as f:
        items = json.load(f)
    for item in items:
        db.add(
            Prompt(
                id=int(item["id"]),
                text=item["text"],
                domain=item["domain"],
                language=item["language"],
                source=item.get("source", "original"),
            )
        )
    db.commit()
    return len(items)
