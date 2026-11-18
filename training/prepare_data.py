"""Build TRAINING manifests (JSONL in the benchmark harness's schema).

Train split = SLR54 NON-holdout speakers + optionally the FLEURS official
train split. NEVER the v0 holdout (SLR54 bucket 0) or FLEURS test.

The split rule is REUSED from ``benchmark/build.py``, not re-derived:

* SLR54 holdout: ``sha256(user_id) % 10 == 0`` — imported as
  ``build.speaker_bucket`` / ``build.SLR54_TEST_BUCKET``. Train speakers
  are every speaker whose bucket is NOT 0.
* SLR54 archive mapping: ``build.slr54_archive_for`` (archive name, URL,
  ZIP member for a FileID).
* FLEURS: the official train split from the same Parquet files
  ``build.fetch_fleurs`` reads; the train/test non-overlap is checked at
  the clip level exactly as the builder does it.

Usage:
    python training/prepare_data.py \\
        --out training/train-manifest.jsonl \\
        --tmp-dir /tmp/nepali-speech-train \\
        [--sources slr54,fleurs] [--max-clips 200] [--archives 2] \\
        [--audio-dir /tmp/nepali-speech-train/audio] \\
        [--test-manifest benchmark/manifests/v0.jsonl] [--dry-run]

``--max-clips N`` samples deterministically (round-robin across sources,
sorted by clip_id within a source, so a smoke run gets a mix, not just
the alphabetically-first source). ``--archives`` limits which SLR54
archives are used — either a count (``--archives 2`` = the first 2
archives, by hex order, among those touched by the sampled clips) or a
comma-separated list of archive names (``--archives
asr_nepali_0.zip,asr_nepali_1.zip``). SLR54 archives are ~1.2 GB each, so
a smoke run must never download all 16: with ``--max-clips`` + ``--audio-dir``
only the archives containing the sampled clips are downloaded, via
``build.materialise`` (the same code the benchmark uses).

Disjointness is ENFORCED, not just checked: after building, the train rows
are checked against the v0 holdout manifest and the script exits non-zero
WITHOUT WRITING ANY MANIFEST if there is ANY speaker overlap (SLR54) or
ANY clip overlap (FLEURS, which carries no speaker ids — its rows use
speaker "unknown", so the speaker check skips "unknown" and the clip-level
check does the real work). Verify afterwards with:
    python benchmark/evaluate.py --check-splits \\
        --train-manifest training/train-manifest.jsonl \\
        --test-manifest benchmark/manifests/v0.jsonl
(note: that CLI check treats "unknown" speakers literally, so a FLEURS
train row and a FLEURS test row both carry speaker "unknown" and the CLI
reports overlap; the clip-level check in this script is the authoritative
one for FLEURS — see README §2).

No audio is ever written into the repo: ``--audio-dir`` must be outside
the repo (checked), and the manifest itself carries only paths/URLs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import ensure_repo_on_path, load_benchmark_modules  # noqa: E402

ensure_repo_on_path()
build, evaluate = load_benchmark_modules()

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_MANIFEST = REPO_ROOT / "benchmark" / "manifests" / "v0.jsonl"
DEFAULT_OUT = REPO_ROOT / "training" / "train-manifest.jsonl"


class PrepareError(Exception):
    """Data prep failed (fetch, validation, or disjointness)."""


# ---------------------------------------------------------------------------
# Source fetchers — train side (reuse build.py's split + archive logic)
# ---------------------------------------------------------------------------

def fetch_slr54_train(session, tmp_dir: Path, verify_archives: bool = True):
    """SLR54 train portion: every utterance whose speaker is NOT bucket 0.

    Reuses ``build.download_file`` (TSV), ``build.speaker_bucket`` /
    ``build.SLR54_TEST_BUCKET`` (the holdout rule), ``build.slr54_archive_for``
    (archive mapping) and ``build.make_clip_entry`` (clip shape). The
    ``split`` field is patched to ``"train"`` — ``make_clip_entry`` stamps
    the source's default split ("test"), which is the v0 holdout's split,
    not this one.
    """
    tsv_path = build.download_file(
        session, build.SLR54_TSV_URL, tmp_dir / "utt_spk_text.tsv"
    )
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
    test_speakers = {
        s for s in speakers if build.speaker_bucket(s) == build.SLR54_TEST_BUCKET
    }
    train_speakers = speakers - test_speakers
    train_rows = [(f, u, t) for f, u, t in rows if u in train_speakers]

    clips: list[dict] = []
    archives_touched: dict[str, list[str]] = defaultdict(list)
    for file_id, user_id, text in train_rows:
        archive_name, archive_url, member = build.slr54_archive_for(file_id)
        archives_touched[archive_name].append(file_id)
        clip = build.make_clip_entry(
            "slr54",
            clip_id=f"slr54-{file_id}",
            transcript=text,
            speaker_id=user_id,
            duration_seconds=None,  # TSV carries no durations
            archive_url=archive_url,
            archive_member=member,
            audio_url=archive_url,
            audio_path=f"slr54/slr54-{file_id}.flac",
        )
        clip["split"] = "train"
        clips.append(clip)
    if verify_archives:
        build._verify_slr54_archives(session, archives_touched)
    status = {
        "key": "slr54",
        "status": "ok",
        "clips": len(clips),
        "utterances_in_corpus": len(rows),
        "speakers_in_corpus": len(speakers),
        "train_speakers": len(train_speakers),
        "test_speakers_excluded": len(test_speakers),
        "hours": None,
        "hours_note": "TSV carries no durations; hours unknown without audio",
        "holdout_rule": "sha256(user_id) % 10 == 0 excluded (benchmark/build.py)",
        "archives_touched": sorted(archives_touched),
        "archives_verified": verify_archives,
    }
    return clips, status, {
        "train_speakers": train_speakers,
        "test_speakers": test_speakers,
    }


def fetch_fleurs_train(session, tmp_dir: Path):
    """FLEURS ne_np OFFICIAL TRAIN split, via Parquet range reads.

    Mirrors ``build.fetch_fleurs`` (same Parquet API, same columns, same
    HttpRangeFile pattern) but for split="train". The clip-level
    train/test non-overlap is checked here exactly as the builder checks
    it for the test split: no test recording basename appears in train.
    """
    import pyarrow.parquet as pq

    size = build.get_json(
        session, build.FLEURS_SIZE_API,
        dataset="google/fleurs", config="ne_np", split="train",
    )
    size_block = size.get("size", size)
    splits = {s["split"]: s["num_rows"] for s in size_block["splits"]}
    parquet_urls = session.get(
        build.FLEURS_PARQUET_API.format(split="train"), timeout=60
    ).json()
    train_url = parquet_urls[0]
    test_urls = session.get(
        build.FLEURS_PARQUET_API.format(split="test"), timeout=60
    ).json()
    test_url = test_urls[0]

    gender_names = None
    with build.HttpRangeFile(train_url, session=session) as fh:
        pf = pq.ParquetFile(fh)
        import json as _json

        meta = _json.loads(pf.schema_arrow.metadata[b"huggingface"])
        gender_names = meta["info"]["features"]["gender"]["names"]
        table = pf.read(columns=["id", "path", "num_samples", "transcription",
                                 "raw_transcription", "gender"])
    with build.HttpRangeFile(test_url, session=session) as fh:
        test_table = pq.ParquetFile(fh).read(columns=["path"])

    clips: list[dict] = []
    train_paths: set[str] = set()
    for row_idx, row in enumerate(table.to_pylist()):
        basename = Path(row["path"]).name
        train_paths.add(basename)
        gender_idx = row["gender"]
        gender = gender_names[gender_idx] if gender_idx is not None else None
        clip = build.make_clip_entry(
            "fleurs",
            clip_id=f"fleurs-ne_np-train-{basename[:-4]}",
            transcript=row["transcription"],
            transcript_normalized=build.nfc(row["transcription"]),
            duration_seconds=row["num_samples"] / build.SAMPLE_RATE,
            gender=gender,
            parquet_url=train_url,
            parquet_row=row_idx,
            audio_url=train_url,
            audio_path=f"fleurs/fleurs-ne_np-train-{basename[:-4]}.wav",
        )
        clip["split"] = "train"
        clips.append(clip)
    test_paths = {Path(p).name for p in test_table.column("path").to_pylist()}
    overlap = build.check_clip_overlap(train_paths, test_paths, "fleurs train vs test")
    status = {
        "key": "fleurs",
        "status": "ok",
        "clips": len(clips),
        "rows_in_official_train_split": splits.get("train"),
        "rows_in_official_test_split": splits.get("test"),
        "hours": round(sum(c["duration_seconds"] for c in clips) / 3600, 3),
        "speakers": None,
        "speaker_note": "FLEURS carries no speaker ids; clip-level "
                        "train/test non-overlap checked instead",
        "clip_overlap_check": overlap,
    }
    return clips, status, {"train_paths": train_paths, "test_paths": test_paths}


# ---------------------------------------------------------------------------
# Sampling / archive limiting
# ---------------------------------------------------------------------------

def parse_archives(value: str | None) -> tuple[str, object] | None:
    """Parse ``--archives`` into ("count", N) or ("names", [names]).

    A bare integer means "the first N archives (hex order) among those
    touched by the sampled clips". Otherwise a comma-separated list of
    archive file names, e.g. ``asr_nepali_0.zip,asr_nepali_1.zip``.
    """
    if value is None:
        return None
    value = value.strip()
    if value.isdigit():
        n = int(value)
        if n <= 0:
            raise PrepareError("--archives count must be > 0")
        return ("count", n)
    names = [s.strip() for s in value.split(",") if s.strip()]
    if not names:
        raise PrepareError("--archives needs names or a count")
    for name in names:
        if not name.startswith("asr_nepali_") or not name.endswith(".zip"):
            raise PrepareError(
                f"--archives: bad archive name {name!r} "
                "(expected e.g. asr_nepali_0.zip)"
            )
    return ("names", names)


def sample_clips(clips_by_source: dict[str, list[dict]], max_clips: int | None):
    """Deterministic sample: round-robin across sources, clip_id order.

    Round-robin (rather than a global sort) keeps a smoke sample mixed
    across sources — a global sort would return FLEURS first for a small
    N because "fleurs-..." < "slr54-...". Within a source, clips are in
    sorted clip_id order, so the sample is reproducible run to run.
    """
    ordered = {
        src: sorted(clips, key=lambda c: c["clip_id"])
        for src, clips in clips_by_source.items()
    }
    if max_clips is None:
        out = [c for src in sorted(ordered) for c in ordered[src]]
        return out
    out: list[dict] = []
    indices = {src: 0 for src in ordered}
    sources = sorted(ordered)
    while len(out) < max_clips:
        progressed = False
        for src in sources:
            if len(out) >= max_clips:
                break
            i = indices[src]
            if i < len(ordered[src]):
                out.append(ordered[src][i])
                indices[src] = i + 1
                progressed = True
        if not progressed:
            break
    return out


def restrict_archives(clips: list[dict], archives):
    """Filter SLR54 clips to the requested archives; return (clips, names)."""
    if archives is None:
        return clips, sorted({Path(c["archive_url"]).name for c in clips
                              if c["source"] == "slr54"})
    kind, value = archives
    slr54 = [c for c in clips if c["source"] == "slr54"]
    touched = sorted({Path(c["archive_url"]).name for c in slr54})
    if kind == "count":
        wanted = touched[: value]
    else:
        wanted = value
        unknown = sorted(set(wanted) - set(touched))
        if unknown:
            raise PrepareError(
                f"--archives: no sampled SLR54 clips in archive(s): {unknown}"
            )
    kept = [c for c in clips
            if c["source"] != "slr54" or Path(c["archive_url"]).name in wanted]
    return kept, wanted


# ---------------------------------------------------------------------------
# Disjointness enforcement (fail closed, before anything is written)
# ---------------------------------------------------------------------------

def check_train_vs_test(train_rows: list[dict], test_rows: list[dict]) -> dict:
    """Verify train rows are disjoint from the v0 holdout. Raises on failure.

    * SLR54 (and any source with real speaker ids): ``evaluate
      .check_speaker_disjoint`` on rows whose speaker is not "unknown".
    * FLEURS (speaker "unknown" on both sides — the speaker check is
      vacuous there by design): clip-id overlap must be empty.
    * Global: no train clip_id may appear in the test manifest at all.
    Returns the check report dict on success.
    """
    train_known = [r for r in train_rows if r.get("speaker") != "unknown"]
    test_known = [r for r in test_rows if r.get("speaker") != "unknown"]
    speaker_check = evaluate.check_speaker_disjoint(train_known, test_known)
    if not speaker_check["disjoint"]:
        raise PrepareError(
            "SPEAKER OVERLAP between train and v0 holdout: "
            f"{speaker_check['overlap_speakers'][:5]} "
            f"({speaker_check['n_overlap']} speakers). Refusing to write a manifest."
        )
    train_ids = {r["clip_id"] for r in train_rows}
    test_ids = {r["clip_id"] for r in test_rows}
    clip_overlap = sorted(train_ids & test_ids)
    if clip_overlap:
        raise PrepareError(
            "CLIP OVERLAP between train and v0 holdout: "
            f"{clip_overlap[:5]} ({len(clip_overlap)} clips). "
            "Refusing to write a manifest."
        )
    fleurs_train = {r["clip_id"] for r in train_rows if r["source"] == "fleurs"}
    fleurs_test = {r["clip_id"] for r in test_rows if r["source"] == "fleurs"}
    fleurs_overlap = sorted(fleurs_train & fleurs_test)
    if fleurs_overlap:
        raise PrepareError(
            "FLEURS clip overlap between train and test: "
            f"{fleurs_overlap[:5]}. Refusing to write a manifest."
        )
    return {
        "status": "pass",
        "speaker_check": speaker_check,
        "clip_overlap": [],
        "fleurs_clip_overlap": [],
        "note": "FLEURS rows carry speaker 'unknown'; FLEURS disjointness "
                "is enforced at the clip level, not the speaker level.",
    }


# ---------------------------------------------------------------------------
# Build orchestration
# ---------------------------------------------------------------------------

def build_train_manifest(
    session=None,
    tmp_dir: Path | None = None,
    sources: list[str] | None = None,
    max_clips: int | None = None,
    archives=None,
    verify_archives: bool = True,
) -> tuple[list[dict], list[dict], list[dict], list[str]]:
    """Fetch train clips (metadata only — no audio) for the given sources.

    Returns ``(full_clips, jsonl_rows, per_source_status, archives_used)``. Audio is NOT
    downloaded here; use :func:`materialise_audio` for that.
    """
    import requests

    session = session or requests.Session()
    tmp_dir = tmp_dir or Path("/tmp/nepali-speech-train")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    wanted = sources or ["slr54", "fleurs"]
    clips_by_source: dict[str, list[dict]] = {}
    statuses: list[dict] = []
    for key in wanted:
        if key == "slr54":
            got, status, _ = fetch_slr54_train(
                session, tmp_dir, verify_archives=verify_archives
            )
        elif key == "fleurs":
            got, status, _ = fetch_fleurs_train(session, tmp_dir)
        else:
            raise PrepareError(f"unknown source {key!r} (expected slr54/fleurs)")
        clips_by_source[key] = got
        statuses.append(status)
        print(f"[{key}] train: {len(got)} clips", file=sys.stderr)
    sampled = sample_clips(clips_by_source, max_clips)
    sampled, archives_used = restrict_archives(sampled, archives)
    rows = [build.to_jsonl_row(c) for c in sampled]
    return sampled, rows, statuses, archives_used


def materialise_audio(full_clips: list[dict], out_dir: Path) -> dict:
    """Download audio for the given clips into ``out_dir`` (not the repo).

    Reuses ``build.materialise`` — only the archives containing the given
    clips are downloaded (SLR54) and only the requested FLEURS rows are
    extracted. ``out_dir`` must be outside the repo.
    """
    out_dir = Path(out_dir).resolve()
    if REPO_ROOT.resolve() in out_dir.parents or out_dir == REPO_ROOT.resolve():
        raise PrepareError(
            f"--audio-dir {out_dir} is inside the repo — audio must go to "
            "a caller-chosen directory outside the repo"
        )
    manifest = {"clips": full_clips}
    clip_ids = [c["clip_id"] for c in full_clips]
    return build.materialise(manifest, out_dir, clip_ids=clip_ids)


def write_outputs(rows, statuses, archives_used, out: Path, disjoint_report: dict,
                  test_manifest_path: Path, max_clips, archives_arg):
    """Write train-manifest.jsonl + train-manifest.summary.json."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    jsonl_text = "".join(
        json.dumps(r, ensure_ascii=False) + "\n" for r in rows
    )
    out.write_text(jsonl_text, encoding="utf-8")
    per_source: dict[str, dict] = {}
    for status in statuses:
        per_source[status["key"]] = {
            "clips_in_manifest": sum(1 for r in rows if r["source"] == status["key"]),
            "clips_available": status.get("clips"),
            "hours_in_manifest": None,
        }
    durations = [r.get("duration_s") for r in rows if r.get("duration_s")]
    hours_known = round(sum(durations) / 3600, 3) if durations else None
    speakers = {r["speaker"] for r in rows if r.get("speaker") != "unknown"}
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "builder": "training/prepare_data.py",
        "train_manifest": str(out),
        "test_manifest_checked_against": str(test_manifest_path),
        "manifest_sha256": hashlib.sha256(jsonl_text.encode("utf-8")).hexdigest(),
        "total_clips": len(rows),
        "hours_known": hours_known,
        "hours_note": (
            "SLR54 clips carry no duration in the TSV; hours are known "
            "only for FLEURS rows. Recompute from audio after materialising."
            if hours_known is not None else
            "No durations known (SLR54-only manifest; TSV carries no durations)."
        ),
        "speakers": len(speakers),
        "speaker_note": "FLEURS rows carry speaker 'unknown' and are excluded "
                        "from the speaker count.",
        "sources": per_source,
        "counts_by_source": dict(Counter(r["source"] for r in rows)),
        "counts_by_quality": dict(Counter(r.get("quality", "clean") for r in rows)),
        "archives_used": archives_used,
        "max_clips": max_clips,
        "archives_arg": archives_arg,
        "holdout_rule": "SLR54 train = sha256(user_id) % 10 != 0 "
                        "(benchmark/build.py::speaker_bucket); FLEURS train = "
                        "official train split only.",
        "disjointness": disjoint_report,
    }
    summary_path = out.with_name(out.stem + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out, summary_path, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a TRAINING manifest (never the v0 holdout).")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="train manifest JSONL to write")
    parser.add_argument("--tmp-dir", default="/tmp/nepali-speech-train")
    parser.add_argument("--sources", default="slr54,fleurs",
                        help="comma-separated: slr54,fleurs")
    parser.add_argument("--test-manifest", default=str(DEFAULT_TEST_MANIFEST),
                        help="v0 holdout manifest JSONL to check against")
    parser.add_argument("--max-clips", type=int, default=None,
                        help="sample at most N clips (deterministic, round-robin)")
    parser.add_argument("--archives", default=None,
                        help="SLR54 archive limit: a count (e.g. 2) or a "
                             "comma-separated list (e.g. asr_nepali_0.zip)")
    parser.add_argument("--audio-dir", default=None,
                        help="also download audio here (outside the repo); "
                             "only archives containing sampled clips are fetched")
    parser.add_argument("--skip-archive-verify", action="store_true",
                        help="skip verifying SLR54 clips against ZIP central directories")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit 0 without network or writes")
    args = parser.parse_args(argv)

    if args.dry_run:
        print("DRY RUN — training manifest plan (nothing fetched, nothing written)")
        print(f"  sources:        {args.sources}")
        print(f"  max-clips:      {args.max_clips}")
        print(f"  archives:       {args.archives}")
        print(f"  out:            {args.out}")
        print(f"  test manifest:  {args.test_manifest}")
        print(f"  tmp dir:        {args.tmp_dir}")
        print(f"  audio dir:      {args.audio_dir or '(none — metadata only)'}")
        print("  split rule:     SLR54 train = sha256(user_id) % 10 != 0 "
              "(benchmark/build.py); FLEURS = official train split only")
        print("  disjoint check: speaker-disjoint (SLR54) + clip-level "
              "(FLEURS) vs v0 holdout; any overlap fails with no manifest written")
        return 0

    try:
        archives = parse_archives(args.archives)
        sources = [s.strip() for s in args.sources.split(",") if s.strip()]
        full_clips, rows, statuses, archives_used = build_train_manifest(
            tmp_dir=Path(args.tmp_dir), sources=sources,
            max_clips=args.max_clips, archives=archives,
            verify_archives=not args.skip_archive_verify,
        )
        if not rows:
            print("ERROR: no train clips built (empty manifest)", file=sys.stderr)
            return 1
        test_rows = evaluate.load_manifest(args.test_manifest)
        disjoint = check_train_vs_test(rows, test_rows)
        print(f"disjointness vs {args.test_manifest}: {disjoint['status']}",
              file=sys.stderr)
        out, summary_path, summary = write_outputs(
            rows, statuses, archives_used, Path(args.out), disjoint,
            Path(args.test_manifest), args.max_clips, args.archives,
        )
        print(f"wrote {out}: {len(rows)} clips, "
              f"{summary['speakers']} speakers, "
              f"hours_known={summary['hours_known']}")
        print(f"wrote {summary_path}")
        if args.audio_dir:
            result = materialise_audio(full_clips, Path(args.audio_dir))
            print(f"materialised {result['files_written']}/"
                  f"{result['clips_requested']} audio files into {args.audio_dir}")
        return 0
    except (PrepareError, build.BuildError, build.DisjointnessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
