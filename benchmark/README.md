# Nepali ASR Benchmark — Specification (v0)

> **Status: v0 specification, 2026-11-18.** The benchmark infrastructure
> (this spec + `evaluate.py`) ships now. The v0 held-out test set is built
> **from existing public data only**. Phase 2 collection has produced
> **zero audio so far** (no volunteers yet; live deploy still pending), so
> the realistic portion — phone-quality, noisy, dialect, code-switched
> Nepali — **does not exist yet**. It is defined here as a slot that fills
> in when Phase 2 data lands. Nothing in this document implies the
> benchmark already covers phone / noisy / code-switched speech.

## 1. What this benchmark is

A held-out, test-only evaluation set for Nepali automatic speech
recognition (ASR), with a fixed scoring protocol, so that models trained
on Nepali speech can be compared on the same clips, the same text
normalisation, and the same metric — instead of every paper reporting
its own split and its own number.

It is deliberately modelled on the 2026 benchmark it is meant to match
and extend:

- **Paudel & Sayami (2026)**, TU MSc thesis, "Comparative Analysis of
  Multilingual Pre-trained Models for Nepali ASR" —
  https://p-sumann.github.io/nepali-asr-benchmark/
- Six models fine-tuned on OpenSLR SLR54 under one protocol
  (NFC normalisation + `jiwer`, speaker-disjoint splits, three test
  sets: OpenSLR, FLEURS, Common Voice).
- Their headline numbers (theirs, not ours — see
  [`leaderboard.md`](leaderboard.md)): Whisper-Large-v3-Turbo
  **14.76% WER in-domain** (OpenSLR), **39.56% FLEURS**,
  **48.35% Common Voice**; IndicWav2Vec (94M) ties it in-domain;
  CTC decoding 29× faster; zero-shot Whisper **~94% WER**
  out-of-domain.

Our v0 re-uses their three test sets so new results are directly
comparable to those numbers, and adds the slice reporting and the
realistic-portion slot they did not have.

## 2. Design principles

1. **Test-only, never train.** The benchmark clips are evaluation data.
   A model submission must not have trained on any benchmark clip, any
   benchmark speaker, or any benchmark transcript. Training on the
   benchmark (or on its speakers) disqualifies the submission. This is
   checked, not assumed: see §6.
2. **Speaker-disjoint splits.** No speaker appears in both the training
   data and the test set. The harness verifies this
   (`evaluate.py --check-splits`); a submission whose training speakers
   overlap the test speakers is rejected.
3. **No single-average reporting.** Every result reports WER and CER
   **overall AND sliced** by gender, region/province, and audio quality
   (plus per-source-set). A submission that reports only one overall
   number is incomplete and is not listed.
4. **Fixed protocol.** NFC normalisation + `jiwer`, exactly as in the
   2026 paper (§5). Same normalisation for reference and hypothesis.
5. **Honest scope.** v0 is read speech only (see §3). The realistic
   portion is pending (see §4). Claims must always say which part a
   number comes from.
6. **License-clean by construction.** Each source keeps its own license
   (§7). Sources whose terms forbid re-hosting are manifest-only:
   the benchmark ships file IDs, hashes and download instructions,
   never the audio.

## 3. v0 composition — public-data sources only

v0 is assembled from the test splits of three existing public Nepali
corpora. All three are **read speech** — sentences read aloud in a
quiet setting. That is the whole of v0; it is not representative of
how Nepali is actually spoken on the phone.

| Source (config) | What it is | Size (source) | Role in v0 |
|---|---|---|---|
| OpenSLR SLR54 (`openslr/openslr`, SLR54) | Crowd-sourced read Nepali, 527 speakers, ~157,905 utterances, ~165 h (Kjartansson et al. 2018) | ~165 h | Held-out test split, speaker-disjoint from any SLR54 training split (the 2026 paper used an 80/10/10 train/dev/test partition) |
| FLEURS (`google/fleurs`, `ne_np`) | Read FLoRes-101 sentences; the released `ne_np` test split has 726 rows (the 2026 paper evaluated 724 of them) | ~10 h per language | `ne_np` **test** split (726 sentences) |
| Mozilla Common Voice (`ne`, Scripted Speech 27.0) | Crowdsourced validated read speech, ne-NP | 40.55 MB download (release 27.0); ~1.5–5 h validated in earlier releases | **test** split (validated clips only) |

Notes:

- The v0 manifest itself (`benchmark/manifests/v0.jsonl`) is built by a
  separate step from these sources; the manifest schema is defined in
  `evaluate.py` (`clip_id`, `source`, `speaker`, `transcript`,
  `audio_path`, `gender`, `region`, `quality`, `duration_s`).
  Exact v0 clip counts are **TBD** until that manifest is built —
  no counts are claimed here.
- IndicVoices Nepali (100 h train / 496 speakers + 5 h held-out /
  114 speakers, CC BY 4.0) is a **candidate v0.1 addition**, not a
  v0 component: its license is verified (§7), but the Nepali-subset
  figures come from a secondary source and the HF repo is gated, so
  the subset must be re-confirmed against the repo before it is
  included. Its held-out set is spontaneous speech, which would make
  v0.1 the first non-read slice.
- Meta Seamless / SeamlessAlign is **not** a v0 audio source: it
  distributes metadata only, no audio (§7).

## 4. The pending realistic portion (Phase 2 data slot)

When Phase 2 collection produces audio, a second portion is added
as a separate, clearly labelled slice set — **never mixed into the
v0 numbers**. It is defined now so collection knows what to aim for:

| Slice | Definition | Status |
|---|---|---|
| Phone-quality | Audio recorded on a phone (the project's own collection app), 8 kHz-ish / compressed, as actually spoken | **Pending — 0 clips** |
| Noisy | Background noise present (street, market, household, vehicle) | **Pending — 0 clips** |
| Dialect / regional | Speakers from outside the Kathmandu valley standard — province recorded per clip | **Pending — 0 clips** |
| Code-switched | Nepali mixed with English (and/or Hindi) in the same utterance | **Pending — 0 clips** |
| Spontaneous | Unscripted speech (conversation, not read prompts) | **Pending — 0 clips** |

Rules for the realistic portion:

- It is collected under this project's own consent + CC0 terms
  (see repo `DATASET_LICENSE.md`), so — unlike the public sources —
  its audio **can** be redistributed with the benchmark.
- It is speaker-disjoint from v0 and from any training split, same
  as §2.2.
- It is reported as its own rows in the leaderboard (`leaderboard.md`),
  never folded into the v0 overall number. A model may score well on
  read speech and badly on phone speech; the leaderboard must show
  both.
- Until Phase 2 audio exists, every leaderboard cell for these slices
  reads **TBD / pending** — never a number, never a placeholder
  estimate.

## 5. Metrics

- **Primary: WER** (word error rate), **secondary: CER** (character
  error rate). Both are micro-averages: total errors ÷ total reference
  words (or characters) across all clips in the set/slice — never a
  mean of per-clip rates, never a mean of slice rates.
- **Slices (all reported, none optional):**
  - `source` — OpenSLR / FLEURS / Common Voice (/ realistic slices
    when they exist)
  - `gender` — female / male / other / unknown (where the source
    provides it; FLEURS carries a gender field)
  - `region` — province / region label (where known; mostly
    "unknown" in v0 — reported as such, not hidden)
  - `quality` — clean / phone / noisy (v0 is all `clean`; the other
    values exist only in the pending realistic portion)
- A slice value with zero clips is reported as `n=0, wer/cer=null`
  — never 0.0, never omitted silently.
- Empty-reference edge cases follow the harness conventions in
  `evaluate.py` (empty ref + empty hyp = 0.0; empty ref + non-empty
  hyp = 1.0).
- Latency / speed may be reported alongside (the 2026 paper's
  "CTC 29× faster" observation is a useful secondary signal) but it
  never replaces WER/CER.

## 6. How a model submission is scored

Protocol — identical to Paudel & Sayami (2026):

1. **Text normalisation:** NFC-normalise both reference and
   hypothesis (`unicodedata.normalize("NFC", ...)`), strip, collapse
   whitespace. Devanagari combining marks have composed and
   decomposed spellings; NFC makes the same spoken word score the
   same regardless of spelling.
2. **Scoring:** `jiwer` 4.0 — `process_words` / `process_characters`
   with whitespace stripping and word/character tokenisation (the
   `compute_measures` name in jiwer 3.x was removed in jiwer 4.0; the
   harness pins `jiwer==4.0.0`). WER on words, CER on characters
   (Unicode code points after NFC).
3. **Harness:** `benchmark/evaluate.py` (already in this repo).
   ```bash
   # Score a hypotheses file (JSONL: {"clip_id": ..., "hypothesis": ...})
   python benchmark/evaluate.py --manifest benchmark/manifests/v0.jsonl \
       --hypotheses /tmp/run1.jsonl --out /tmp/run1-results.json

   # Or transcribe directly with openai-whisper (CPU-only):
   python benchmark/evaluate.py --manifest benchmark/manifests/v0.jsonl \
       --transcribe --model tiny --audio-dir /tmp/v0-audio \
       --out /tmp/tiny-results.json
   ```
   Clips with no hypothesis line are scored as an **empty**
   hypothesis (all words deleted) and counted in
   `missing_hypotheses` — reported, never silent.
4. **Speaker-disjoint check** (before scoring, for any submission
   that trained on SLR54 or any overlapping corpus):
   ```bash
   python benchmark/evaluate.py --check-splits \
       --train-manifest data/train.jsonl \
       --test-manifest benchmark/manifests/v0.jsonl
   ```
   Any overlap fails the submission.
5. **Submission must include:** model name + checkpoint link,
   parameters, what it was trained/fine-tuned on (dataset + split),
   the full harness output JSON (overall + every slice), the date,
   and a statement that no benchmark clip/speaker was in training.
   Submissions are listed in [`leaderboard.md`](leaderboard.md)
   only with the full sliced output — no single-average entries.

## 7. License table (verified 2026-11-18)

Every license below was checked on its official page or Hugging Face
dataset card on **2026-11-18** (Asia/Kathmandu). Secondary blogs were
not used as license evidence. Full detail per source follows the table.

| Source | License | Audio redistribution? | Verdict for this benchmark | Verified on |
|---|---|---|---|---|
| FLEURS (`google/fleurs`, `ne_np`) | CC BY 4.0 | Yes, with attribution | **Bundle audio** (with attribution) | https://huggingface.co/datasets/google/fleurs |
| Mozilla Common Voice (`ne`, Scripted Speech 27.0) | CC0 1.0 Universal | CC0 permits it, but Mozilla's distribution terms forbid re-hosting/re-sharing | **Manifest-only** (IDs + hashes + download instructions, no audio) | https://commonvoice.mozilla.org/en/datasets |
| OpenSLR SLR54 | CC BY-SA 4.0 (Attribution-ShareAlike 4.0 International) | Yes, with attribution + share-alike | **Bundle audio** (attribution + CC BY-SA 4.0 on redistributed portions) | https://www.openslr.org/54 |
| IndicVoices (`ai4bharat/IndicVoices`) | CC BY 4.0 | Yes, with attribution | **Bundle audio** (candidate v0.1 addition; Nepali subset to re-confirm) | https://huggingface.co/datasets/ai4bharat/IndicVoices |
| Meta Seamless (SeamlessM4T / SeamlessAlign) | CC BY-NC 4.0 (non-commercial) | No audio distributed (metadata only); NC clause bars commercial use | **Manifest-only** (metadata/alignments only; no audio in v0) | https://ai.meta.com/blog/seamless-m4t/ |

### Per-source verification notes

- **FLEURS — CC BY 4.0, VERIFIED.** The dataset card's front matter
  states `license: cc-by-4.0`, and the card text states "All datasets
  are licensed under the Creative Commons license (CC-BY)". The
  `ne_np` config is listed in the card. Redistribution (including
  commercial) is allowed with attribution. No non-commercial or
  share-alike restriction.
- **Mozilla Common Voice — CC0 1.0, VERIFIED; redistribution
  MANIFEST-ONLY.** The official dataset listing
  (https://commonvoice.mozilla.org/en/datasets) lists
  "Common Voice Scripted Speech 27.0 - Nepali | CC0-1.0 | ne-NP |
  ASR | MP3 | 40.55 MB". The CC0 dedication itself permits any use,
  including redistribution. **However**, since October 2025 Common
  Voice is distributed **exclusively through the Mozilla Data
  Collective**, and the MDC terms forbid re-hosting/re-sharing the
  dataset (official FAQ:
  https://community.mozilladatacollective.com/faq-can-i-get-the-common-voice-or-other-mdc-datasets-from-other-platforms-like-github-or-hugging-face/ —
  "Common Voice datasets are exclusively available through Mozilla
  Data Collective… Our new terms reflect this"). The per-dataset
  Data Consumer License is login-gated and its exact text could not
  be read directly on 2026-11-18; the no-re-hosting wording is
  corroborated by independent reports quoting the MDC dataset page
  ("It is forbidden to re-host or re-share this dataset"). To stay
  inside both the license and the platform terms, this benchmark
  ships a **manifest only** for Common Voice: clip IDs, hashes and
  instructions to download from MDC — never the audio files.
  (Common Voice was also removed from Hugging Face in October 2025;
  do not point users at the old `mozilla-foundation/common_voice`
  mirrors.)
- **OpenSLR SLR54 — CC BY-SA 4.0, VERIFIED.** The official resource
  page (https://www.openslr.org/54) states "License:
  Attribution-ShareAlike 4.0 International" and the dataset is
  "distributed under Creative Commons Attribution-ShareAlike 4.0
  International Public License" (© 2016–2018 Google, Inc.).
  Redistribution is allowed with attribution **and** any
  redistributed or adapted material must carry the same CC BY-SA 4.0
  license. Consequence for this benchmark: SLR54-derived benchmark
  portions are shared under CC BY-SA 4.0, and the benchmark's own
  packaging must preserve the attribution to Kjartansson et al.
  (2018).
- **IndicVoices — CC BY 4.0, VERIFIED.** The official dataset card
  (https://huggingface.co/datasets/ai4bharat/IndicVoices) states
  "This dataset is released under the CC BY 4.0." Access is gated:
  the repo requires accepting its access conditions on Hugging Face
  before files are visible. Redistribution is allowed with
  attribution. Nepali-subset figures (100 h train / 496 speakers;
  5 h held-out / 114 speakers) come from a secondary source, not
  from the card text read on 2026-11-18, and must be re-confirmed
  against the gated repo before the subset is included.
- **Meta Seamless — CC BY-NC 4.0, VERIFIED; MANIFEST-ONLY.**
  Meta's official announcement
  (https://ai.meta.com/blog/seamless-m4t/) states "we're publicly
  releasing SeamlessM4T under CC BY-NC 4.0" and "releasing the
  metadata of SeamlessAlign, the biggest open multimodal translation
  dataset to date, totaling 470,000 hours of mined speech and text
  alignments". Two consequences: (a) the **NC clause forbids
  commercial use** — any benchmark portion built on it is
  non-commercial only; (b) SeamlessAlign releases **metadata only**,
  no audio — the audio must be reconstructed from Common Crawl, so
  there is no Nepali Seamless audio file this benchmark could bundle
  in the first place. The "153 h" Nepali figure quoted in project
  background is **as-given, not re-verified** against a primary
  source on 2026-11-18. Seamless is therefore manifest-only /
  reference-only in this benchmark.

### What the benchmark itself is licensed as

- The scoring code (`evaluate.py`) is under the repo's MIT license.
- The v0 manifest (clip IDs, transcripts, speaker labels) is a
  compilation of the sources above: each source's terms travel with
  its rows, and the strictest applicable term (CC BY-SA 4.0 on the
  SLR54 rows) governs redistribution of the compiled set.
- The future realistic portion (Phase 2 collected audio) is CC0
  under the project's own `DATASET_LICENSE.md` and can be bundled
  freely.
- **No source in this table was left UNVERIFIED** — all five
  licenses were read on their official pages on 2026-11-18. The
  open caveats are: the login-gated Common Voice per-dataset terms
  (exact text not directly read), the IndicVoices Nepali-subset
  figures (secondary source), and the Seamless 153 h figure
  (as-given).

## 8. Limitations (read before citing any number from this benchmark)

- v0 is **read speech only**. It measures how well a model
  transcribes clean, read Devanagari sentences. It says nothing
  about phone calls, noisy rooms, dialects, or code-switching.
- v0 is small in two of its three sets (FLEURS ne_np test = 726
  sentences; Common Voice ne is a ~40 MB release). Small test sets
  mean wide confidence intervals — do not over-claim a 1-point WER
  difference on these sets.
- Region/province slicing is mostly "unknown" in v0: the public
  sources do not record Nepali province. Real regional slicing
  arrives with the Phase 2 portion.
- The 2026 paper's models were fine-tuned on SLR54; any model
  fine-tuned on SLR54 and evaluated on the SLR54 test split is
  "in-domain" by construction, and its number is not comparable to
  a model evaluated on FLEURS/Common Voice without fine-tuning.
  Report the training data with every number.
- No baseline numbers of our own exist yet. The only numbers in
  this repo's benchmark documents are the 2026 paper's, attributed
  to them (see `leaderboard.md`).

## 9. Files

| File | What it is |
|---|---|
| `benchmark/README.md` | This specification |
| `benchmark/evaluate.py` | Scoring harness (NFC + jiwer, sliced reporting, speaker-disjoint check) |
| `benchmark/manifests/v0.jsonl` | v0 held-out manifest (separate build step; schema in `evaluate.py`) |
| `benchmark/manifests/v0.example.jsonl` | 2-line example manifest (format illustration only — not the test set) |
| `benchmark/hf-dataset-card.md` | Draft Hugging Face dataset-card text for v0 (draft only — nothing published) |
| `benchmark/leaderboard.md` | Leaderboard design + reference rows from Paudel & Sayami (2026) |
