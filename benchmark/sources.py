"""Source catalogue for the nepali-speech v0 held-out test set.

Every fact here was verified against the provider's own pages/APIs on
2026-11-18 (see ``benchmark/manifests/README.md`` for the provenance).
Worker 1 is checking licences in parallel; where this file says a source
is ``gated`` or ``unverifiable`` the builder leaves it out of the v0
manifest clips and records the reason instead of guessing.

Licensing is the hard rule: the repo ships the builder + manifests, never
audio. ``redistributable`` answers one question only — *may the audio be
redistributed under the source licence?* — and the builder's license gate
(``build.make_clip_entry``) turns any ``redistributable=False`` source
into manifest-only entries: clip id + source + download instructions,
never an audio path. Reachability is tracked separately in ``status``:
a source can be redistributable in principle and still unreachable
without an account (Common Voice), in which case it contributes zero
clips. Licence facts here follow the verified table in
``benchmark/README.md`` §7 (checked on the official pages 2026-11-18).
"""

from __future__ import annotations

SOURCES = {
    "fleurs": {
        "key": "fleurs",
        "name": "FLEURS ne_np (official test split)",
        "dataset_id": "google/fleurs",
        "config": "ne_np",
        "split": "test",
        "license": "CC-BY-4.0",
        "license_url": "https://huggingface.co/datasets/google/fleurs",
        "homepage": "https://huggingface.co/datasets/google/fleurs",
        "download_url": (
            "https://huggingface.co/datasets/google/fleurs/resolve/main/"
            "ne_np/test-00000-of-00001.parquet"
        ),
        "redistributable": True,
        "gated": False,
        "notes": (
            "Not gated (HF API gated=false, 2026-11-18). Audio is embedded "
            "in the per-split Parquet file; there is no per-clip audio URL, "
            "so the manifest records the Parquet URL + row index and "
            "`build.py materialise` extracts clips from it."
        ),
    },
    "slr54": {
        "key": "slr54",
        "name": "OpenSLR SLR54 Nepali ASR (speaker-disjoint holdout)",
        "dataset_id": "openslr/slr54",
        "split": "test",
        "license": "CC-BY-SA-4.0",
        "license_url": "https://www.openslr.org/54/",
        "homepage": "https://www.openslr.org/54/",
        "download_url": "https://www.openslr.org/resources/54/utt_spk_text.tsv",
        "redistributable": True,
        "gated": False,
        "notes": (
            "SLR54 ships no official test split: it is a training corpus "
            "(~157k utterances, 527 speakers). The 2026 benchmark "
            "(Paudel & Sayami) used an 80/10/10 speaker-disjoint split of "
            "it; v0 mirrors that protocol with a deterministic 10% speaker "
            "holdout (sha256(user_id) % 10 == 0). Audio is FLAC inside 16 "
            "ZIP archives; the archive + member for a clip are derived from "
            "its FileID (see build.slr54_archive_for) and verified against "
            "the archives' central directories at build time."
        ),
    },
    "common_voice": {
        "key": "common_voice",
        "name": "Mozilla Common Voice ne-NP (validated split)",
        "dataset_id": "mozilla-foundation/common_voice_17_0",
        "split": "validated",
        "license": "CC0-1.0 (audio); Mozilla Data Collective terms forbid re-hosting",
        "license_url": "https://commonvoice.mozilla.org/en/datasets",
        "homepage": "https://commonvoice.mozilla.org/en/datasets",
        "download_url": "https://commonvoice.mozilla.org/en/datasets",
        "redistributable": True,
        "gated": True,
        "notes": (
            "GATED / UNREACHABLE without an account. Current release is "
            "Common Voice Scripted Speech 27.0 (ne-NP, 40.55 MB). Since "
            "October 2025 Common Voice is distributed only through the "
            "Mozilla Data Collective, whose terms forbid re-hosting / "
            "re-sharing, so this source is MANIFEST-ONLY even once "
            "reachable; the old HF mirrors were removed in October 2025 "
            "and the HF copy is an empty stub (datasets-server: "
            "EmptyDatasetError). No clips in v0; fill this slot when a "
            "maintainer with Data Collective access exports ne-NP."
        ),
    },
    "indicvoices": {
        "key": "indicvoices",
        "name": "IndicVoices (held-out 5h)",
        "dataset_id": "ai4bharat/IndicVoices",
        "split": "test",
        "license": "CC-BY-4.0 (verified on the dataset card)",
        "license_url": "https://huggingface.co/datasets/ai4bharat/IndicVoices",
        "homepage": "https://huggingface.co/datasets/ai4bharat/IndicVoices",
        "download_url": "https://huggingface.co/datasets/ai4bharat/IndicVoices",
        "redistributable": True,
        "gated": True,
        "notes": (
            "GATED on Hugging Face (HF API gated='auto', 2026-11-18): "
            "requires accepting terms / a login. No account is created for "
            "this build, so no clips in v0. Candidate v0.1 addition: "
            "its held-out set is spontaneous speech, which would make "
            "v0.1 the first non-read slice. Nepali-subset figures "
            "(100 h train / 496 speakers; 5 h held-out / 114 speakers) "
            "come from a secondary source and must be re-confirmed "
            "against the gated repo before inclusion."
        ),
    },
    "seamless": {
        "key": "seamless",
        "name": "Meta Seamless (~153h Nepali, as-given)",
        "dataset_id": "facebook/seamless",
        "split": "test",
        "license": "CC-BY-NC-4.0 (verified via Meta's SeamlessM4T announcement)",
        "license_url": "https://ai.meta.com/blog/seamless-m4t/",
        "homepage": "https://huggingface.co/datasets/facebook/seamless",
        "download_url": "https://huggingface.co/datasets/facebook/seamless",
        "redistributable": False,
        "gated": True,
        "notes": (
            "UNVERIFIABLE as data and NON-COMMERCIAL by licence: the HF "
            "API answers 401 for this dataset (2026-11-18), and "
            "SeamlessAlign distributes metadata only, no audio — the "
            "audio would have to be reconstructed from Common Crawl. "
            "The '153 h' Nepali figure is as-given, not re-verified. "
            "Fail closed: non-redistributable, left out of v0 entirely."
        ),
    },
}

# Sources whose clips are assembled into the v0 test set, in order.
V0_SOURCES = ["fleurs", "slr54", "common_voice", "indicvoices", "seamless"]
