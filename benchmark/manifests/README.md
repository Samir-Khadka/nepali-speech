# v0 held-out test-set manifests

**Built 2026-11-18** by `benchmark/build.py` (run it again to rebuild;
the build date and the exact counts it found are recorded inside
`v0.json`). The test set is a **manifest, not audio**: the repo ships
clip IDs, transcripts and download locations — never audio files.

## Files

| File | What it is |
|---|---|
| `v0.json` | Full manifest: protocol, per-source status/counts, all clips with provenance (source, split, speaker, official audio location, transcript, duration/gender where known), the speaker-disjointness check results, and the empty realistic-portion slot. 14,145 clips. |
| `v0.jsonl` | Same clips, one JSON object per line, in the scoring harness's schema (`benchmark/evaluate.py`: `clip_id`, `source`, `speaker`, `transcript`, `audio_path`, `gender`, `region`, `quality`, `duration_s`). This is the file you score against. |
| `v0.example.jsonl` | 2-line format illustration from the harness author — **not** the test set. |

## v0 composition (real numbers from the 2026-11-18 build run)

| Source | Clips | Speakers | Hours | Status |
|---|---|---|---|---|
| FLEURS `ne_np`, official test split | 726 | — (source carries no speaker ids) | 2.28 | ok |
| OpenSLR SLR54, speaker-disjoint holdout | 13,419 | 49 test / 478 train | unknown (TSV has no durations) | ok |
| Mozilla Common Voice ne-NP, validated | 0 | — | — | **gated** (see below) |
| IndicVoices held-out | 0 | — | — | **gated** (see below) |
| Meta Seamless | 0 | — | — | **unverifiable / non-commercial** (see below) |
| **Total** | **14,145** | | **2.28 h known** | |
| Realistic portion (phone/noisy/dialect/code-switched) | 0 | — | — | **empty slot** — Phase 2 has collected 0 h of audio |

Disjointness check (run by the builder, recorded in `v0.json`):
**pass** — no speaker in more than one split; SLR54 test speakers
(49) are disjoint from the SLR54 training portion (478 speakers);
FLEURS test recordings do not overlap FLEURS train (726 test vs 3,332
train recordings, 0 shared). The harness agrees:
`evaluate.py --manifest v0.jsonl --check-splits --train-manifest …`
reports 0 overlapping speakers against an SLR54 train manifest.

Two honest caveats, both recorded in the manifest:

1. **SLR54 has no official test split.** The 2026 benchmark (Paudel &
   Sayami) used an 80/10/10 speaker-disjoint split of SLR54, but its
   exact test speaker list was never published (training code not
   released), and transcript matching cannot recover it (only
   2,355/8,894 of its test references match a unique SLR54 utterance).
   v0 therefore defines its own deterministic holdout under the same
   protocol — speakers with `sha256(user_id) % 10 == 0` — reproducible
   from `utt_spk_text.tsv` alone. Its test size (13,419 clips) differs
   from the benchmark's (8,894); the benchmark's sizes are recorded in
   `v0.json` under `protocol.benchmark_reference_test_sizes` for
   comparison.
2. **FLEURS carries no speaker ids.** Its rows use `speaker: "unknown"`
   in `v0.jsonl` (the harness schema requires a non-empty speaker
   string). Speaker-disjointness for FLEURS rests on the official
   fixed split plus the builder's clip-level test/train non-overlap
   check — not on a speaker check. Also note the real FLEURS `ne_np`
   test split is **726 rows** (datasets-server `/size`, 2026-11-18);
   the 2026 benchmark evaluated 724 of them.

## How to materialise the audio

Nothing here is audio. Fetch it from the official sources with the
builder (downloads go to a directory **you** choose — never the repo):

```bash
# All v0 clips (FLEURS test Parquet ~526 MB once + the SLR54 ZIP
# archives holding the test clips, ~9 GB total if you take everything)
python benchmark/build.py materialise \
    --manifest benchmark/manifests/v0.json \
    --out-dir /path/to/v0-audio

# Just one clip (e.g. to smoke-test the pipeline)
python benchmark/build.py materialise \
    --manifest benchmark/manifests/v0.json \
    --out-dir /tmp/v0-audio \
    --clip-id fleurs-ne_np-test-10022685377830883165
```

Layout under `--out-dir` matches the `audio_path` in `v0.jsonl`, so
the harness resolves it directly:

```bash
python benchmark/evaluate.py --manifest benchmark/manifests/v0.jsonl \
    --transcribe --model tiny --audio-dir /path/to/v0-audio \
    --out /tmp/tiny-results.json
```

Per-source manual download (same files the builder fetches):

- **FLEURS** (`google/fleurs`, CC BY 4.0): the `ne_np` test Parquet —
  https://huggingface.co/datasets/google/fleurs/resolve/main/ne_np/test-00000-of-00001.parquet
  (~526 MB). Each clip is one row (`parquet_row` in `v0.json`);
  `materialise` extracts rows to `fleurs/<clip_id>.wav` (16 kHz mono).
  There is no per-clip audio URL: the datasets-server `/rows`
  endpoint cannot serve this config ("Parquet error: Scan size limit
  exceeded"), so the builder reads the metadata columns straight from
  the Parquet file with HTTP Range requests.
- **SLR54** (CC BY-SA 4.0): 16 ZIP archives from
  https://www.openslr.org/54/ (`asr_nepali_0.zip` … `asr_nepali_f.zip`).
  A clip's archive + member are in `v0.json` (`archive_url`,
  `archive_member`, e.g. `asr_nepali/data/6d/6d49b03fdc.flac`);
  `materialise` extracts them to `slr54/<clip_id>.flac`. The full
  utterance list is
  https://www.openslr.org/resources/54/utt_spk_text.tsv (10 MB).
- **Common Voice ne-NP**: **gated.** Since October 2025 Common Voice
  is distributed only through the Mozilla Data Collective
  (https://commonvoice.mozilla.org/en/datasets — current release
  Scripted Speech 27.0, ne-NP, 40.55 MB); the old Hugging Face
  mirrors were removed and the HF copy is an empty stub. Its terms
  forbid re-hosting, so this source stays manifest-only even once
  reachable. No account was created for this build; fill the slot
  when a maintainer with Data Collective access exports ne-NP.
- **IndicVoices**: gated on Hugging Face (`gated: "auto"`) — needs
  terms acceptance / login. CC BY 4.0 verified. Candidate v0.1
  addition (its held-out set is spontaneous speech); Nepali-subset
  figures must be re-confirmed against the gated repo first.
- **Seamless**: not a v0 source. HF API answers 401, the licence is
  CC BY-NC 4.0 (non-commercial), and SeamlessAlign distributes
  metadata only — there is no Nepali Seamless audio file to fetch.

## How to rebuild

```bash
cd <repo root>
.venv/bin/python benchmark/build.py build
# writes benchmark/manifests/v0.json + v0.jsonl (atomic: a failed
# check writes nothing). Re-run is safe: every step is deterministic
# (same SLR54 holdout, same FLEURS rows) unless the sources change.
```

The builder fetches metadata only (HF datasets-server counts, Parquet
metadata columns via Range reads, the 10 MB SLR54 TSV, ZIP central
directories). It downloads full audio **only** via `materialise`.
It creates no accounts. Tests:
`../.venv/bin/python -m pytest tests/ -q` from `benchmark/`.
