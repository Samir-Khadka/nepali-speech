"""SQLAlchemy models for nepali-speech Phase 1 (see CONTRACT.md)."""

from sqlalchemy import Boolean, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Prompt(Base):
    __tablename__ = "prompts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    domain: Mapped[str] = mapped_column(String(100), nullable=False)
    language: Mapped[str] = mapped_column(String(10), nullable=False)
    source: Mapped[str] = mapped_column(String(50), nullable=False, default="original")


class Contributor(Base):
    __tablename__ = "contributors"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    consent_version: Mapped[str | None] = mapped_column(String(20), nullable=True)
    consented_at: Mapped[str | None] = mapped_column(String(40), nullable=True)
    province: Mapped[str | None] = mapped_column(String(100), nullable=True)
    age_range: Mapped[str | None] = mapped_column(String(50), nullable=True)
    gender: Mapped[str | None] = mapped_column(String(50), nullable=True)


class Recording(Base):
    __tablename__ = "recordings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prompt_id: Mapped[int] = mapped_column(ForeignKey("prompts.id"), nullable=False)
    contributor_id: Mapped[str] = mapped_column(
        ForeignKey("contributors.id"), nullable=False
    )
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    mime: Mapped[str] = mapped_column(String(50), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="pending_review")
    double_review: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[str | None] = mapped_column(String(40), nullable=True)


class Review(Base):
    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    recording_id: Mapped[int] = mapped_column(
        ForeignKey("recordings.id"), nullable=False
    )
    reviewer_id: Mapped[str] = mapped_column(Text, nullable=False)
    decision: Mapped[str] = mapped_column(String(20), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str | None] = mapped_column(String(40), nullable=True)
