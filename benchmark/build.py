"""Build the nepali-speech v0 held-out test set — as a MANIFEST, not audio.

Usage:
    python build.py build [--out benchmark/manifests/v0.json]
                          [--tmp-dir /tmp/nepali-speech-benchmark]
                          [--only fleurs,slr54] [--skip-archive-verify]
    python build.py materialise --manifest benchmark/manifests/v0.json
                                --out-dir <dir> [--clip-id ID ... | --all]

Design (see benchmark/manifests/README.md for the full story):
- Sources: FLEURS ne_np official test split, Common Voice ne-NP validated
  split, OpenSLR SLR54 speaker-disjoint holdout. IndicVoices and Seamless
  are catalogued in sources.py but contribute no clips in v0 (gated /
  unverifiable — see sources.py).
- The repo never ships audio. The manifest carries clip ids, source,
  split, speaker id, the official URL of the file containing the audio,
  the clip's location inside that file, transcript, duration when known,
  and gender/region when the source provides them (null otherwise).
- Speaker-disjointness is enforced AND verified by script: no speaker in
  more than one split, and SLR54 test speakers are disjoint from the
  SLR54 training portion (checked against utt_spk_text.tsv). FLEURS
  carries no speaker ids, so its check is clip-level: no test recording
  appears in the train split. A failed check aborts the build with a
  non-zero exit and no manifest is written.
- Gated sources fail gracefully: the source is recorded with its official
  download URL + instructions and zero clips. Nothing is faked.
- Downloads are metadata-only and cheap: HF datasets-server for counts,
  Parquet metadata columns via HTTP Range reads (see http_range.py),
  the 10 MB SLR54 TSV, and ZIP central directories. Full audio is
  downloaded ONLY by `materialise`, into a directory the user chooses —
  never into the repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
import wave
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from http_range import HttpRangeFile  # noqa: E402
from sources import SOURCES, V0_SOURCES  # noqa: E402

VERSION = "v0"
SAMPLE_RATE = 16000  # FLEURS audio is 16 kHz mono
FLEURS_PARQUET_API = (
    "https://huggingface.co/api/datasets/google/fleurs/parquet/ne_np/{split}"
)
FLEURS_SIZE_API = "https://datasets-server.huggingface.co/size"
SLR54_TSV_URL = "https://www.openslr.org/resources/54/utt_spk_text.tsv"
SLR54_RESOURCE_URL = "https://www.openslr.org/54/"
SLR54_ARCHIVE_URL = "https://www.openslr.org/resources/54/asr_nepali_{x}.zip"
SLR54_TEST_BUCKET = 0  # sha256(user_id) % 10 == 0  -> ~10% of speakers
HF_API = "https://huggingface.co/api/datasets/{dataset_id}"
DATASETS_SERVER_SPLITS = "https://datasets-server.huggingface.co/splits"

# Reference sizes of the 2026 benchmark's test sets (Paudel & Sayami),
# counted 2026-11-18 from the released predictions parquet files
# (sumanpaudel1997/nepali-asr-benchmark, whisper_turbo rows). Kept for
# comparison only: v0 defines its own SLR54 holdout because the
# benchmark's exact test speaker list was never published (training code
# not released) and transcript matching is ambiguous (only 2,355/8,894
# references match a unique SLR54 utterance).
BENCHMARK_REFERENCE_SIZES = {"openslr": 8894, "fleurs": 724, "common_voice": 289}


class BuildError(Exception):
    """A source could not be fetched or a check failed."""


class DisjointnessError(BuildError):
    """A speaker-disjointness check failed."""


# --------------------------------------------------------------------------
# Pure helpers (unit-tested, no network)
# --------------------------------------------------------------------------

def nfc(text: str) -> str:
    """NFC-normalise a transcript (the benchmark protocol: NFC + jiwer)."""
    return unicodedata.normalize("NFC", text)


def speaker_bucket(speaker_id: str) -> int:
    """Deterministic 0-9 bucket for a speaker id (SLR54 holdout rule)."""
    digest = hashlib.sha256(speaker_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % 10


def slr54_archive_for(file_id: str) -> tuple[str, str, str]:
    """Map an SLR54 FileID to (archive_name, archive_url, archive_member).

    Verified 2026-11-18 against the archives' central directories and the
    TSV: archive N (hex digit) holds subdirectories (N*16)..(N*16+15);
    member is asr_nepali/data/<first-2-hex>/<file_id>.flac; per-bucket
    TSV counts match the archives' FLAC counts exactly.
    """
    subdir = file_id[:2].lower()
    archive_hex = f"{int(subdir, 16) // 16:x}"
    return (
        f"asr_nepali_{archive_hex}.zip",
        SLR54_ARCHIVE_URL.format(x=archive_hex),
        f"asr_nepali/data/{subdir}/{file_id}.flac",
    )


def make_clip_entry(source_key: str, clip_id: str, transcript: str, **fields) -> dict:
    """Build one manifest clip entry, applying the license gate.

    A source flagged ``redistributable=False`` in sources.py produces
    manifest-only entries: clip id + source + download instructions, and
    NEVER an audio URL/path, even if the caller passed one.
    """
    source = SOURCES[source_key]
    entry = {
        "clip_id": clip_id,
        "source": source_key,
        "source_dataset": source["dataset_id"],
        "split": source["split"],
        "speaker_id": fields.get("speaker_id"),
        "transcript": nfc(transcript),
        "transcript_normalized": nfc(fields.get("transcript_normalized") or transcript),
        "duration_seconds": fields.get("duration_seconds"),
        "gender": fields.get("gender"),
        "region": fields.get("region"),
        # v0 is read speech only, so every clip is "clean"; the realistic
        # portion (phone/noisy) fills the other values when it exists.
        "quality": fields.get("quality", "clean"),
        "license": source["license"],
        "audio_url": fields.get("audio_url"),
        # audio_path is the RELATIVE path of the materialised file, as
        # resolved by evaluate.py --transcribe against --audio-dir and
        # as written by `build.py materialise`:
        #   fleurs/<clip_id>.wav, slr54/<clip_id>.flac
        "audio_path": fields.get("audio_path"),
        "parquet_url": fields.get("parquet_url"),
        "parquet_row": fields.get("parquet_row"),
        "archive_url": fields.get("archive_url"),
        "archive_member": fields.get("archive_member"),
    }
    if not source["redistributable"]:
        entry["audio_url"] = None
        entry["audio_path"] = None
        entry["parquet_url"] = None
        entry["archive_url"] = None
        entry["archive_member"] = None
        entry["access"] = {
            "kind": "manifest-only",
            "reason": f"license {source['license']} does not allow redistribution",
            "download_url": source["download_url"],
        }
    else:
        entry["access"] = {
            "kind": "manifest-only",
            "reason": "audio is never bundled in the repo; fetch it from the official source",
            "download_url": source["download_url"],
        }
    return entry


# --------------------------------------------------------------------------
# Manifest schema validation
# --------------------------------------------------------------------------

CLIP_REQUIRED = {
    "clip_id": str, "source": str, "source_dataset": str, "split": str,
    "transcript": str, "transcript_normalized": str, "license": str,
    "quality": str, "access": dict,
}
CLIP_NULLABLE = {
    "speaker_id": str, "duration_seconds": (int, float),
    "gender": str, "region": str, "audio_url": str, "audio_path": str,
    "parquet_url": str, "parquet_row": int, "archive_url": str,
    "archive_member": str,
}


def validate_manifest(manifest: dict) -> list[str]:
    """Return a list of schema errors (empty = valid)."""
    errors: list[str] = []
    for key in ("version", "generated_at", "sources", "clips", "checks"):
        if key not in manifest:
            errors.append(f"missing top-level key: {key}")
    if errors:
        return errors
    if manifest["version"] != VERSION:
        errors.append(f"version must be {VERSION!r}, got {manifest['version']!r}")
    seen: set[str] = set()
    for i, clip in enumerate(manifest["clips"]):
        where = f"clips[{i}]"
        for field, typ in CLIP_REQUIRED.items():
            if field not in clip:
                errors.append(f"{where}: missing {field}")
            elif not isinstance(clip[field], typ):
                errors.append(f"{where}: {field} must be {typ.__name__}")
        for field, typ in CLIP_NULLABLE.items():
            if field in clip and clip[field] is not None and not isinstance(clip[field], typ):
                errors.append(f"{where}: {field} must be {typ} or null")
        if isinstance(clip.get("clip_id"), str):
            if clip["clip_id"] in seen:
                errors.append(f"{where}: duplicate clip_id {clip['clip_id']!r}")
            seen.add(clip["clip_id"])
        if isinstance(clip.get("transcript"), str) and clip["transcript"] != nfc(clip["transcript"]):
            errors.append(f"{where}: transcript is not NFC-normalised")
        if clip.get("source") not in SOURCES:
            errors.append(f"{where}: unknown source {clip.get('source')!r}")
    return errors


# --------------------------------------------------------------------------
# JSONL output (the scoring harness's schema — benchmark/evaluate.py)
# --------------------------------------------------------------------------

def to_jsonl_row(clip: dict) -> dict:
    """Map one full manifest clip to an evaluate.py manifest row.

    evaluate.py requires clip_id/source/speaker/transcript as non-empty
    strings. FLEURS carries no speaker ids, so its rows use
    speaker="unknown" — the harness's disjoint check is vacuous for
    those rows by design; the real FLEURS check is the clip-level
    test/train non-overlap the builder verifies (see fetch_fleurs).
    Unknown gender/region become "unknown", matching the harness slices.
    """
    return {
        "clip_id": clip["clip_id"],
        "source": clip["source"],
        "speaker": clip.get("speaker_id") or "unknown",
        "transcript": clip["transcript"],
        "audio_path": clip.get("audio_path"),
        "gender": clip.get("gender") or "unknown",
        "region": clip.get("region") or "unknown",
        "quality": clip.get("quality", "clean"),
        "duration_s": clip.get("duration_seconds"),
    }


def manifest_to_jsonl(manifest: dict) -> str:
    """Render the manifest's clips as JSONL in evaluate.py's schema."""
    return "".join(
        json.dumps(to_jsonl_row(c), ensure_ascii=False) + "\n"
        for c in manifest["clips"]
    )


# --------------------------------------------------------------------------
# Speaker-disjointness checks
# --------------------------------------------------------------------------

def check_speaker_disjointness(clips: list[dict],
                               train_speakers_by_source: dict[str, set[str]],
                               ) -> dict:
    """Verify speaker-disjointness; raise DisjointnessError on failure.

    - No (non-null) speaker id appears in more than one split.
    - No (non-null) speaker id is shared across sources.
    - For each source with a known train speaker set, test speakers must
      be disjoint from train speakers (the test set is test-only).
    Sources without speaker ids (FLEURS) are reported, not failed: their
    official fixed splits are used as-is.
    """
    speaker_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    speaker_sources: dict[str, set[str]] = defaultdict(set)
    for clip in clips:
        spk = clip.get("speaker_id")
        if spk is None:
            continue
        speaker_splits[(clip["source"], spk)].add(clip["split"])
        speaker_sources[spk].add(clip["source"])
    multi_split = {k: v for k, v in speaker_splits.items() if len(v) > 1}
    if multi_split:
        raise DisjointnessError(
            f"speakers in more than one split: {sorted(multi_split)[:5]}")
    shared_across_sources = {s: srcs for s, srcs in speaker_sources.items()
                             if len(srcs) > 1}
    if shared_across_sources:
        raise DisjointnessError(
            f"speaker ids shared across sources: {sorted(shared_across_sources)[:5]}")
    result: dict = {
        "status": "pass",
        "speakers_in_more_than_one_split": [],
        "speaker_ids_shared_across_sources": [],
        "per_source": {},
    }
    for source_key, train_speakers in train_speakers_by_source.items():
        test_speakers = {c["speaker_id"] for c in clips
                         if c["source"] == source_key and c.get("speaker_id")}
        overlap = sorted(test_speakers & train_speakers)
        if overlap:
            raise DisjointnessError(
                f"{source_key}: {len(overlap)} test speakers also in train: "
                f"{overlap[:5]}")
        result["per_source"][source_key] = {
            "test_speakers": len(test_speakers),
            "train_speakers": len(train_speakers),
            "overlap": [],
        }
    for source_key in {c["source"] for c in clips}:
        if source_key not in result["per_source"]:
            result["per_source"][source_key] = {
                "note": "source carries no speaker ids; official fixed "
                        "split used as-is, per-clip speaker check not possible",
            }
    return result


def check_clip_overlap(test_paths: set[str], train_paths: set[str],
                       label: str) -> dict:
    """Clip-level non-overlap check (used where speaker ids don't exist)."""
    overlap = sorted(test_paths & train_paths)
    if overlap:
        raise DisjointnessError(
            f"{label}: {len(overlap)} test recordings also in train: {overlap[:5]}")
    return {"status": "pass", "test_clips": len(test_paths),
            "train_clips": len(train_paths), "overlap": []}


# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------

def get_json(session: requests.Session, url: str, **params) -> dict:
    last_exc: Exception | None = None
    for _ in range(3):
        try:
            resp = session.get(url, params=params or None, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:  # noqa: BLE001
            last_exc = exc
    raise BuildError(f"GET {url} failed: {last_exc}")


def download_file(session: requests.Session, url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None
    for _ in range(3):
        try:
            with session.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            return dest
        except requests.RequestException as exc:  # noqa: BLE001
            last_exc = exc
    raise BuildError(f"download {url} failed: {last_exc}")


# --------------------------------------------------------------------------
# Source fetchers (each returns (clips, source_status, extra))
# --------------------------------------------------------------------------

def fetch_fleurs(session: requests.Session, tmp_dir: Path) -> tuple:
    """FLEURS ne_np official test split via Parquet range reads.

    The datasets-server /rows endpoint cannot serve this config
    ("Parquet error: Scan size limit exceeded", 2026-11-18), so the
    metadata columns are read straight from the Parquet file with HTTP
    Range requests — a few MB transferred for a 526 MB file.
    """
    import pyarrow.parquet as pq

    size = get_json(session, FLEURS_SIZE_API,
                    dataset="google/fleurs", config="ne_np", split="test")
    size_block = size.get("size", size)
    splits = {s["split"]: s["num_rows"] for s in size_block["splits"]}
    parquet_urls = session.get(
        FLEURS_PARQUET_API.format(split="test"), timeout=60).json()
    test_url = parquet_urls[0]
    train_urls = session.get(
        FLEURS_PARQUET_API.format(split="train"), timeout=60).json()
    train_url = train_urls[0]

    gender_names = None
    with HttpRangeFile(test_url, session=session) as fh:
        pf = pq.ParquetFile(fh)
        import json as _json
        meta = _json.loads(pf.schema_arrow.metadata[b"huggingface"])
        gender_names = meta["info"]["features"]["gender"]["names"]
        table = pf.read(columns=["id", "path", "num_samples", "transcription",
                                 "raw_transcription", "gender"])
    with HttpRangeFile(train_url, session=session) as fh:
        train_table = pq.ParquetFile(fh).read(columns=["path"])

    clips = []
    test_paths = set()
    for row_idx, row in enumerate(table.to_pylist()):
        basename = Path(row["path"]).name  # unique per recording
        test_paths.add(basename)
        gender_idx = row["gender"]
        gender = gender_names[gender_idx] if gender_idx is not None else None
        clips.append(make_clip_entry(
            "fleurs",
            clip_id=f"fleurs-ne_np-test-{basename[:-4]}",
            transcript=row["transcription"],
            # FLEURS ships `transcription` (normalised) + `raw_transcription`;
            # the manifest stores the NFC form as transcript_normalized.
            transcript_normalized=nfc(row["transcription"]),
            duration_seconds=row["num_samples"] / SAMPLE_RATE,
            gender=gender,
            parquet_url=test_url,
            parquet_row=row_idx,
            audio_url=test_url,
            audio_path=f"fleurs/fleurs-ne_np-test-{basename[:-4]}.wav",
        ))
    train_paths = {Path(p).name for p in train_table.column("path").to_pylist()}
    overlap = check_clip_overlap(test_paths, train_paths, "fleurs test vs train")
    status = {
        "key": "fleurs", "status": "ok", "clips": len(clips),
        "rows_in_official_test_split": splits.get("test"),
        "rows_in_official_train_split": splits.get("train"),
        "hours": round(sum(c["duration_seconds"] for c in clips) / 3600, 3),
        "speakers": None,
        "speaker_note": "FLEURS carries no speaker ids; clip-level "
                        "test/train non-overlap checked instead",
        "clip_overlap_check": overlap,
    }
    return clips, status, {"test_paths": test_paths, "train_paths": train_paths}


def fetch_slr54(session: requests.Session, tmp_dir: Path,
                verify_archives: bool = True) -> tuple:
    """SLR54: TSV download + deterministic 10% speaker holdout."""
    tsv_path = download_file(session, SLR54_TSV_URL, tmp_dir / "utt_spk_text.tsv")
    rows: list[tuple[str, str, str]] = []
    speakers: set[str] = set()
    with open(tsv_path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            file_id, user_id, text = parts[0], parts[1], parts[2]
            rows.append((file_id, user_id, text))
            speakers.add(user_id)
    test_speakers = {s for s in speakers if speaker_bucket(s) == SLR54_TEST_BUCKET}
    train_speakers = speakers - test_speakers
    test_rows = [(f, u, t) for f, u, t in rows if u in test_speakers]

    clips = []
    archives_touched: dict[str, list[str]] = defaultdict(list)
    for file_id, user_id, text in test_rows:
        archive_name, archive_url, member = slr54_archive_for(file_id)
        archives_touched[archive_name].append(file_id)
        clips.append(make_clip_entry(
            "slr54",
            clip_id=f"slr54-{file_id}",
            transcript=text,
            speaker_id=user_id,
            duration_seconds=None,  # TSV carries no durations
            archive_url=archive_url,
            archive_member=member,
            audio_url=archive_url,
            audio_path=f"slr54/slr54-{file_id}.flac",
        ))
    if verify_archives:
        _verify_slr54_archives(session, archives_touched)
    status = {
        "key": "slr54", "status": "ok", "clips": len(clips),
        "utterances_in_corpus": len(rows),
        "speakers_in_corpus": len(speakers),
        "test_speakers": len(test_speakers),
        "train_speakers": len(train_speakers),
        "hours": None,
        "hours_note": "TSV carries no durations; hours unknown without audio",
        "holdout_rule": "sha256(user_id) % 10 == 0 (deterministic, ~10% of speakers)",
        "archives_verified": verify_archives,
    }
    return clips, status, {"train_speakers": train_speakers,
                           "test_speakers": test_speakers}


def _verify_slr54_archives(session: requests.Session,
                           archives_touched: dict[str, list[str]]) -> None:
    """Confirm every test clip's FLAC member exists in its archive.

    Reads only each archive's ZIP central directory via HTTP Range
    requests — no audio is downloaded.
    """
    for archive_name, file_ids in sorted(archives_touched.items()):
        url = SLR54_ARCHIVE_URL.format(x=archive_name.split("_")[-1].split(".")[0])
        members: set[str] | None = None
        last_exc: Exception | None = None
        for _ in range(3):  # transient proxy glitches return bad bytes, not errors
            try:
                with HttpRangeFile(url, session=session) as fh:
                    with zipfile.ZipFile(fh) as zf:
                        members = {Path(n).name[:-5] for n in zf.namelist()
                                   if n.endswith(".flac")}
                break
            except (zipfile.BadZipFile, requests.RequestException) as exc:  # noqa: BLE001
                last_exc = exc
        if members is None:
            raise BuildError(f"{archive_name}: could not list archive: {last_exc}")
        missing = [f for f in file_ids if f not in members]
        if missing:
            raise BuildError(
                f"{archive_name}: {len(missing)} test clips missing from "
                f"archive (e.g. {missing[:3]})")


def _probe_hf_dataset(session: requests.Session, dataset_id: str) -> dict:
    """Minimal HF API probe: gated flag / HTTP status, no auth, no account."""
    try:
        resp = session.get(HF_API.format(dataset_id=dataset_id), timeout=30)
        if resp.status_code == 401:
            return {"reachable": False, "http_status": 401, "gated": True}
        resp.raise_for_status()
        info = resp.json()
        return {"reachable": True, "http_status": 200,
                "gated": info.get("gated", False),
                "description": (info.get("description") or "")[:300]}
    except requests.RequestException as exc:  # noqa: BLE001
        return {"reachable": False, "http_status": None, "gated": None,
                "error": str(exc)}


def fetch_gated_source(session: requests.Session, key: str) -> tuple:
    """A gated/unverifiable source: record status, contribute zero clips."""
    source = SOURCES[key]
    probe: dict = {"key": key, "status": "gated", "clips": 0}
    if key == "common_voice":
        try:
            data = get_json(session, DATASETS_SERVER_SPLITS,
                            dataset="mozilla-foundation/common_voice_17_0")
            probe["datasets_server"] = data
        except BuildError as exc:
            probe["datasets_server_error"] = str(exc)
        probe.update(_probe_hf_dataset(
            session, "mozilla-foundation/common_voice_17_0"))
        probe["status"] = "gated"
        probe["note"] = source["notes"]
    else:
        probe.update(_probe_hf_dataset(session, source["dataset_id"]))
        probe["status"] = "gated" if probe.get("gated") else "unverifiable"
        probe["note"] = source["notes"]
    return [], probe, {}


# --------------------------------------------------------------------------
# Build orchestration
# --------------------------------------------------------------------------

FETCHERS = {
    "fleurs": fetch_fleurs,
    "slr54": fetch_slr54,
    "common_voice": fetch_gated_source,
    "indicvoices": fetch_gated_source,
    "seamless": fetch_gated_source,
}


def build_manifest(session: requests.Session | None = None,
                   tmp_dir: Path | None = None,
                   only: list[str] | None = None,
                   verify_archives: bool = True) -> dict:
    """Assemble the v0 manifest. Raises BuildError/DisjointnessError."""
    session = session or requests.Session()
    tmp_dir = tmp_dir or Path("/tmp/nepali-speech-benchmark")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    wanted = only or V0_SOURCES
    clips: list[dict] = []
    sources_status: list[dict] = []
    train_speakers_by_source: dict[str, set[str]] = {}
    for key in wanted:
        source = SOURCES[key]
        fetcher = FETCHERS[key]
        try:
            if key in ("fleurs", "slr54"):
                got, status, extra = fetcher(session, tmp_dir,
                                             verify_archives=verify_archives) \
                    if key == "slr54" else fetcher(session, tmp_dir)
            else:
                got, status, extra = fetcher(session, key)
        except BuildError as exc:
            status = {"key": key, "status": "failed", "clips": 0,
                      "error": str(exc), "note": source["notes"]}
            got, extra = [], {}
            print(f"[{key}] FAILED: {exc} — recorded, continuing", file=sys.stderr)
        clips.extend(got)
        sources_status.append({
            "key": key, "name": source["name"],
            "dataset_id": source["dataset_id"],
            "license": source["license"],
            "download_url": source["download_url"],
            **{k: v for k, v in status.items() if k != "key"},
        })
        if key == "slr54" and extra.get("train_speakers") is not None:
            train_speakers_by_source["slr54"] = extra["train_speakers"]
        print(f"[{key}] {status.get('status')}: {status.get('clips', 0)} clips")
    disjoint = check_speaker_disjointness(clips, train_speakers_by_source)
    manifest = {
        "version": VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "builder": "benchmark/build.py",
        "protocol": {
            "reference": "Paudel & Sayami (2026), Nepali ASR Benchmark, "
                         "https://p-sumann.github.io/nepali-asr-benchmark/",
            "normalisation": "NFC (unicodedata.normalize); WER with jiwer "
                             "after NFC, as in the 2026 benchmark",
            "speaker_disjointness": "no speaker in more than one split; "
                                    "SLR54 test speakers disjoint from the "
                                    "SLR54 training portion (verified by script)",
            "benchmark_reference_test_sizes": BENCHMARK_REFERENCE_SIZES,
            "benchmark_reference_note": "Test-set sizes of the 2026 "
                "benchmark, counted 2026-11-18 from its released predictions "
                "parquet files. v0 mirrors its protocol (incl. the 80/10/10 "
                "speaker-disjoint SLR54 split) but defines its own SLR54 "
                "holdout, because the benchmark's exact test speaker list "
                "was never published.",
        },
        "sources": sources_status,
        "clips": clips,
        "checks": {"speaker_disjointness": disjoint},
        "realistic_slot": {
            "status": "empty",
            "description": "Phone-quality, noisy, dialect and code-switched "
                           "Nepali — the realistic portion Phase 3 is for.",
            "fills_when": "Phase 2 collection produces approved audio "
                          "(0h collected as of 2026-11-18); the slot is "
                          "then built from this project's own CC0 recordings "
                          "with the same speaker-disjoint rules.",
            "clips": 0,
        },
    }
    errors = validate_manifest(manifest)
    if errors:
        raise BuildError("manifest failed schema validation: " + "; ".join(errors))
    return manifest


# --------------------------------------------------------------------------
# Materialise: fetch the official audio for manifest clips (user's disk)
# --------------------------------------------------------------------------

def _write_wav(path: Path, audio_bytes: bytes) -> None:
    """Write FLEURS embedded audio bytes as a .wav.

    The bytes are a complete WAV file in the released Parquet; if a future
    release embeds raw PCM instead, wrap it as 16 kHz mono s16.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if audio_bytes[:4] == b"RIFF":
        path.write_bytes(audio_bytes)
        return
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(audio_bytes)


def materialise(manifest: dict, out_dir: Path,
                clip_ids: list[str] | None = None) -> dict:
    """Download official audio for the given clips into ``out_dir``.

    FLEURS: downloads the test Parquet once (~526 MB) and extracts the
    requested rows' audio. SLR54: downloads only the archives containing
    requested clips and extracts the FLAC members. Nothing is written
    into the repo; ``out_dir`` is the caller's choice.
    """
    import pyarrow.parquet as pq

    session = requests.Session()
    wanted = set(clip_ids) if clip_ids else None
    clips = [c for c in manifest["clips"]
             if wanted is None or c["clip_id"] in wanted]
    if wanted:
        missing = wanted - {c["clip_id"] for c in clips}
        if missing:
            raise BuildError(f"clip ids not in manifest: {sorted(missing)[:5]}")
    written: list[str] = []
    fleurs = [c for c in clips if c["source"] == "fleurs"]
    if fleurs:
        parquet_path = out_dir / "fleurs-ne_np-test.parquet"
        if not parquet_path.exists():
            print(f"downloading FLEURS test parquet (~526 MB) -> {parquet_path}")
            download_file(session, SOURCES["fleurs"]["download_url"], parquet_path)
        pf = pq.ParquetFile(str(parquet_path))
        rows_by_index = {c["parquet_row"]: c for c in fleurs}
        row_offset = 0
        for batch in pf.iter_batches(batch_size=64, columns=["audio"]):
            for i, row in enumerate(batch.to_pylist()):
                clip = rows_by_index.get(row_offset + i)
                if clip is None:
                    continue
                dest = out_dir / "fleurs" / f"{clip['clip_id']}.wav"
                _write_wav(dest, row["audio"]["bytes"])
                written.append(str(dest))
            row_offset += batch.num_rows
    slr54 = [c for c in clips if c["source"] == "slr54"]
    by_archive: dict[str, list[dict]] = defaultdict(list)
    for clip in slr54:
        by_archive[clip["archive_url"]].append(clip)
    for archive_url, members in by_archive.items():
        zip_path = out_dir / Path(archive_url).name
        if not zip_path.exists():
            print(f"downloading {archive_url} -> {zip_path}")
            download_file(session, archive_url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            for clip in members:
                dest = out_dir / "slr54" / f"{clip['clip_id']}.flac"
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(clip["archive_member"]) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                written.append(str(dest))
    return {"clips_requested": len(clips), "files_written": len(written),
            "files": written}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build / materialise the nepali-speech v0 test-set manifest.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_build = sub.add_parser("build", help="assemble benchmark/manifests/v0.json")
    p_build.add_argument("--out", default="benchmark/manifests/v0.json")
    p_build.add_argument("--tmp-dir", default="/tmp/nepali-speech-benchmark")
    p_build.add_argument("--only", default="",
                         help="comma-separated source keys, e.g. fleurs,slr54")
    p_build.add_argument("--skip-archive-verify", action="store_true",
                         help="skip verifying SLR54 clips against ZIP central directories")
    p_mat = sub.add_parser("materialise", help="download official audio for manifest clips")
    p_mat.add_argument("--manifest", default="benchmark/manifests/v0.json")
    p_mat.add_argument("--out-dir", required=True,
                       help="directory for downloaded audio (NOT the repo)")
    p_mat.add_argument("--clip-id", action="append", default=[],
                       help="materialise only this clip (repeatable); default: all")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            only = [s.strip() for s in args.only.split(",") if s.strip()] or None
            manifest = build_manifest(
                tmp_dir=Path(args.tmp_dir), only=only,
                verify_archives=not args.skip_archive_verify)
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
            jsonl_path = out.with_suffix(".jsonl")
            jsonl_path.write_text(manifest_to_jsonl(manifest), encoding="utf-8")
            total_hours = sum(
                (c["duration_seconds"] or 0) for c in manifest["clips"]) / 3600
            print(f"\nwrote {out} and {jsonl_path}: {len(manifest['clips'])} clips, "
                  f"{total_hours:.2f}h known duration")
            print(f"disjointness: {manifest['checks']['speaker_disjointness']['status']}")
            for src in manifest["sources"]:
                print(f"  {src['key']}: {src['status']} — {src.get('clips', 0)} clips")
            return 0
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        result = materialise(manifest, Path(args.out_dir),
                             clip_ids=args.clip_id or None)
        print(f"materialised {result['files_written']}/{result['clips_requested']} clips "
              f"into {args.out_dir}")
        return 0
    except (BuildError, DisjointnessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
