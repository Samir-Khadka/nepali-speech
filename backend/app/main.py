"""nepali-speech Phase 1 API — FastAPI app implementing CONTRACT.md exactly."""

import os
import random
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session, sessionmaker

from . import storage
from .models import Base, Contributor, Prompt, Recording, Review
from .seed import seed_prompts

MAX_UPLOAD_BYTES_DEFAULT = 5 * 1024 * 1024  # 5 MB per CONTRACT.md
DOUBLE_REVIEW_RATE_DEFAULT = 0.2  # ~20% of recordings flagged at creation

ALLOWED_MIME_TYPES = {
    "audio/webm",
    "audio/ogg",
    "audio/mp4",
    "audio/mpeg",
    "audio/wav",
    "video/webm",
}

# Magic-byte prefixes; MP4 has its signature at offset 4 instead.
_MAGIC_PREFIX = {
    "audio/webm": b"\x1a\x45\xdf\xa3",
    "video/webm": b"\x1a\x45\xdf\xa3",
    "audio/ogg": b"OggS",
    "audio/wav": b"RIFF",
}


def _magic_ok(mime: str, data: bytes) -> bool:
    if mime == "audio/mp4":
        return len(data) >= 8 and data[4:8] == b"ftyp"
    if mime == "audio/mpeg":
        # CONTRACT.md lists no magic rule for MP3; accept any bytes.
        return True
    prefix = _MAGIC_PREFIX.get(mime)
    return prefix is not None and data.startswith(prefix)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _admin_emails() -> set[str]:
    raw = os.environ.get("ADMIN_EMAILS", "")
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def _require_admin(x_admin_email: str | None) -> None:
    if not x_admin_email or x_admin_email.strip().lower() not in _admin_emails():
        raise HTTPException(status_code=403, detail="admin access required")


class ConsentIn(BaseModel):
    contributor_id: str
    version: str = "v1"
    province: str | None = None
    age_range: str | None = None
    gender: str | None = None


class ReviewIn(BaseModel):
    reviewer_id: str
    decision: str  # "approve" | "reject" (validated in handler -> 400 on bad value)
    reason: str | None = None


def create_app(
    database_url: str | None = None,
    data_dir: str | Path | None = None,
) -> FastAPI:
    data_dir = Path(data_dir) if data_dir else storage.get_data_dir()
    storage.configure(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)  # sqlite needs the dir to exist

    if database_url is None:
        database_url = os.environ.get(
            "DATABASE_URL", f"sqlite:///{data_dir / 'nepali-speech.db'}"
        )
    max_upload_bytes = int(
        os.environ.get("MAX_UPLOAD_BYTES", str(MAX_UPLOAD_BYTES_DEFAULT))
    )
    double_review_rate = float(
        os.environ.get("DOUBLE_REVIEW_RATE", str(DOUBLE_REVIEW_RATE_DEFAULT))
    )

    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
    engine = create_engine(database_url, connect_args=connect_args)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        Base.metadata.create_all(engine)
        db = session_factory()
        try:
            seed_prompts(db)
        finally:
            db.close()
        yield

    app = FastAPI(title="nepali-speech", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/prompts/next")
    def prompts_next(contributor_id: str, db: Session = Depends(get_db)):
        total = db.query(func.count(Prompt.id)).scalar() or 0
        recorded_ids = {
            r[0]
            for r in db.query(Recording.prompt_id)
            .filter(Recording.contributor_id == contributor_id)
            .all()
        }
        prompt = (
            db.query(Prompt)
            .order_by(Prompt.id)
            .filter(~Prompt.id.in_(recorded_ids) if recorded_ids else True)
            .first()
        )
        if prompt is None:
            return {"prompt_id": None, "done": True, "total": total}
        return {
            "prompt_id": prompt.id,
            "text": prompt.text,
            "domain": prompt.domain,
            "language": prompt.language,
        }

    @app.get("/api/prompts/count")
    def prompts_count(db: Session = Depends(get_db)):
        return {"total": db.query(func.count(Prompt.id)).scalar() or 0}

    @app.post("/api/consent")
    def consent(body: ConsentIn, db: Session = Depends(get_db)):
        contributor = db.get(Contributor, body.contributor_id)
        now = _now_iso()
        if contributor is None:
            contributor = Contributor(
                id=body.contributor_id,
                consent_version=body.version,
                consented_at=now,
                province=body.province,
                age_range=body.age_range,
                gender=body.gender,
            )
            db.add(contributor)
        else:
            contributor.consent_version = body.version
            contributor.consented_at = now
            contributor.province = body.province
            contributor.age_range = body.age_range
            contributor.gender = body.gender
        db.commit()
        return {"ok": True, "contributor_id": body.contributor_id}

    @app.post("/api/recordings", status_code=201)
    async def create_recording(
        prompt_id: int = Form(...),
        contributor_id: str = Form(...),
        audio: UploadFile = File(...),
        db: Session = Depends(get_db),
    ):
        # Validation order per CONTRACT.md:
        # 1. consent -> 403, 2. prompt exists -> 404, 3. size -> 413,
        # 4. content-type -> 400, 5. magic bytes -> 400.
        if db.get(Contributor, contributor_id) is None:
            raise HTTPException(status_code=403, detail="consent required")
        prompt = db.get(Prompt, prompt_id)
        if prompt is None:
            raise HTTPException(status_code=404, detail="prompt not found")
        data = await audio.read()
        if len(data) > max_upload_bytes:
            raise HTTPException(status_code=413, detail="file too large")
        mime = (audio.content_type or "").split(";")[0].strip().lower()
        if mime not in ALLOWED_MIME_TYPES:
            raise HTTPException(status_code=400, detail="unsupported audio type")
        if not _magic_ok(mime, data):
            raise HTTPException(
                status_code=400, detail="audio content does not match its type"
            )
        ext = storage.EXT_BY_MIME[mime]
        recording = Recording(
            prompt_id=prompt_id,
            contributor_id=contributor_id,
            file_path="",
            mime=mime,
            size_bytes=len(data),
            status="pending_review",
            double_review=random.random() < double_review_rate,
            created_at=_now_iso(),
        )
        db.add(recording)
        db.flush()  # assigns recording.id for the filename
        recording.file_path = storage.save_audio(data, recording.id, ext)
        db.commit()
        return JSONResponse(
            status_code=201,
            content={"recording_id": recording.id, "status": "pending_review"},
        )

    @app.get("/api/recordings/mine")
    def my_recordings(contributor_id: str, db: Session = Depends(get_db)):
        recordings = (
            db.query(Recording)
            .filter(Recording.contributor_id == contributor_id)
            .order_by(Recording.id)
            .all()
        )
        prompt_texts = {
            p.id: p.text for p in db.query(Prompt).all()
        }
        counts = {"recorded": 0, "approved": 0, "pending": 0, "rejected": 0}
        items = []
        for r in recordings:
            counts["recorded"] += 1
            if r.status == "approved":
                counts["approved"] += 1
            elif r.status in ("pending_review", "needs_second_review"):
                counts["pending"] += 1
            elif r.status == "rejected":
                counts["rejected"] += 1
            items.append(
                {
                    "recording_id": r.id,
                    "prompt_text": prompt_texts.get(r.prompt_id, ""),
                    "status": r.status,
                }
            )
        return {**counts, "recordings": items}

    @app.get("/api/review/next")
    def review_next(reviewer_id: str, db: Session = Depends(get_db)):
        reviewed_ids = {
            r[0]
            for r in db.query(Review.recording_id)
            .filter(Review.reviewer_id == reviewer_id)
            .all()
        }
        review_counts = {
            row[0]: row[1]
            for row in db.query(Review.recording_id, func.count(Review.id))
            .group_by(Review.recording_id)
            .all()
        }
        candidates = (
            db.query(Recording)
            .filter(Recording.status.in_(["pending_review", "needs_second_review"]))
            .order_by(Recording.id)
            .all()
        )
        candidates = [c for c in candidates if c.id not in reviewed_ids]
        candidates.sort(key=lambda c: (review_counts.get(c.id, 0), c.id))
        if not candidates:
            return {"empty": True, "message": "No recordings waiting for review."}
        rec = candidates[0]
        prompt = db.get(Prompt, rec.prompt_id)
        return {
            "recording_id": rec.id,
            "prompt_text": prompt.text if prompt else "",
            "domain": prompt.domain if prompt else "",
            "language": prompt.language if prompt else "",
            "audio_url": f"/api/review/audio/{rec.id}",
        }

    @app.get("/api/review/audio/{recording_id}")
    def review_audio(recording_id: int, db: Session = Depends(get_db)):
        recording = db.get(Recording, recording_id)
        if recording is None:
            raise HTTPException(status_code=404, detail="recording not found")
        try:
            data, _mime = storage.read_audio(recording_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="audio file not found")
        return Response(content=data, media_type=recording.mime)

    @app.post("/api/review/{recording_id}")
    def review(recording_id: int, body: ReviewIn, db: Session = Depends(get_db)):
        recording = db.get(Recording, recording_id)
        if recording is None:
            raise HTTPException(status_code=404, detail="recording not found")
        if body.decision not in ("approve", "reject"):
            raise HTTPException(status_code=400, detail="decision must be approve or reject")
        if body.decision == "reject" and not (body.reason or "").strip():
            raise HTTPException(status_code=400, detail="reason is required when rejecting")
        existing = (
            db.query(Review)
            .filter(
                Review.recording_id == recording_id,
                Review.reviewer_id == body.reviewer_id,
            )
            .first()
        )
        if existing is not None:
            raise HTTPException(
                status_code=400, detail="reviewer already reviewed this recording"
            )
        if recording.status in ("approved", "rejected"):
            raise HTTPException(status_code=400, detail="recording already decided")
        db.add(
            Review(
                recording_id=recording_id,
                reviewer_id=body.reviewer_id,
                decision=body.decision,
                reason=body.reason,
                created_at=_now_iso(),
            )
        )
        if body.decision == "reject":
            recording.status = "rejected"
        elif recording.double_review and recording.status == "pending_review":
            recording.status = "needs_second_review"
        else:
            recording.status = "approved"
        db.commit()
        return {"ok": True, "status": recording.status}

    @app.get("/api/admin/stats")
    def admin_stats(
        x_admin_email: str | None = Header(default=None),
        db: Session = Depends(get_db),
    ):
        _require_admin(x_admin_email)
        by_status = {s: 0 for s in ("pending_review", "approved", "rejected", "needs_second_review")}
        for status, count in (
            db.query(Recording.status, func.count(Recording.id))
            .group_by(Recording.status)
            .all()
        ):
            by_status[status] = count
        by_domain: dict[str, int] = {}
        for domain, count in (
            db.query(Prompt.domain, func.count(Recording.id))
            .join(Recording, Recording.prompt_id == Prompt.id)
            .group_by(Prompt.domain)
            .all()
        ):
            by_domain[domain] = count
        # Completed double reviews: double_review recordings with >= 2 reviews.
        double_done = (
            db.query(Recording.id)
            .filter(Recording.double_review.is_(True))
            .join(Review, Review.recording_id == Recording.id)
            .group_by(Recording.id)
            .having(func.count(Review.id) >= 2)
            .all()
        )
        double_reviewed = len(double_done)
        agreement_rate = None
        if double_reviewed:
            agree = 0
            for (rec_id,) in double_done:
                decisions = [
                    d[0]
                    for d in db.query(Review.decision)
                    .filter(Review.recording_id == rec_id)
                    .order_by(Review.id)
                    .limit(2)
                    .all()
                ]
                if len(decisions) == 2 and decisions[0] == decisions[1]:
                    agree += 1
            agreement_rate = agree / double_reviewed
        # Coverage: recordings grouped by the contributor's demographics.
        # Blank/missing values fall into an "unspecified" bucket.
        def _coverage(field):
            buckets: dict[str, int] = {}
            for value, count in (
                db.query(field, func.count(Recording.id))
                .join(Contributor, Contributor.id == Recording.contributor_id)
                .group_by(field)
                .all()
            ):
                key = (value or "").strip() or "unspecified"
                buckets[key] = buckets.get(key, 0) + count
            return buckets

        by_province = _coverage(Contributor.province)
        by_gender = _coverage(Contributor.gender)
        by_age_range = _coverage(Contributor.age_range)
        # Hours are an ESTIMATE: ~5 seconds per recording (see hours_basis).
        approved_count = by_status["approved"]
        recorded_count = (db.query(func.count(Recording.id)).scalar() or 0) - by_status[
            "rejected"
        ]
        hours_estimate = {
            "approved_hours": round(approved_count * 5 / 3600, 4),
            "recorded_hours": round(recorded_count * 5 / 3600, 4),
            "hours_basis": "5s average per recording (estimate)",
        }
        return {
            "prompts": db.query(func.count(Prompt.id)).scalar() or 0,
            "contributors": db.query(func.count(Contributor.id)).scalar() or 0,
            "recordings_total": db.query(func.count(Recording.id)).scalar() or 0,
            "by_status": by_status,
            "reviews_total": db.query(func.count(Review.id)).scalar() or 0,
            "double_reviewed": double_reviewed,
            "agreement_rate": agreement_rate,
            "by_domain": by_domain,
            "by_province": by_province,
            "by_gender": by_gender,
            "by_age_range": by_age_range,
            "hours_estimate": hours_estimate,
        }

    return app


app = create_app()
