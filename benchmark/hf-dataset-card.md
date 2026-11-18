# Draft Hugging Face dataset card — Nepali ASR Benchmark v0

> **DRAFT ONLY — DO NOT PUBLISH.** This is draft dataset-card text for
> the v0 benchmark, prepared 2026-11-18. No Hugging Face account has
> been created and nothing has been uploaded. Publish only after the
> v0 manifest (`benchmark/manifests/v0.jsonl`) is built and its clip
> counts are filled in below (every `TBD` in this draft is a real gap,
> not a placeholder to guess).

---

## Dataset Card for Nepali ASR Benchmark v0

### Dataset Summary

A held-out, test-only benchmark for Nepali automatic speech
recognition (ASR). v0 is built from the test splits of three existing
public Nepali speech corpora — OpenSLR SLR54, FLEURS (`ne_np`), and
Mozilla Common Voice (`ne`) — under a fixed scoring protocol (NFC
normalisation + `jiwer`, speaker-disjoint splits) matching Paudel &
Sayami (2026). It is designed so that models trained on Nepali speech
can be compared on the same clips, the same normalisation, and the
same metric.

**v0 is read speech only.** It does not cover phone-quality, noisy,
dialect, or code-switched Nepali. Those slices are a defined but
empty slot that fills in when the project's own Phase 2 collection
produces audio. Do not use v0 numbers as evidence about real-world
phone-call performance.

### Supported Tasks and Leaderboards

- **Task:** automatic-speech-recognition (`ne`)
- **Metric:** WER (primary), CER (secondary), micro-averaged, reported
  overall AND sliced by source, gender, region/province, and audio
  quality. No single-average reporting.
- **Leaderboard:** see `benchmark/leaderboard.md` in the project
  repository. Reference numbers are from Paudel & Sayami (2026) and
  are labelled as theirs.

### Languages

- Nepali (`ne`, `ne-NP`), Devanagari script.

## Dataset Structure

### Data Instances

One JSON object per line (JSONL manifest), UTF-8, with fields:

| Field | Type | Description |
|---|---|---|
| `clip_id` | string | Unique id, e.g. `fleurs-ne-test-000123` |
| `source` | string | `fleurs` \| `common_voice` \| `slr54` \| `nepali-speech` (Phase 2 collection; empty in v0) |
| `speaker` | string | Speaker id (used for the speaker-disjoint check) |
| `transcript` | string | Reference transcript (Devanagari) |
| `audio_path` | string (optional) | Path/URL to the audio — **not stored in this repo** for manifest-only sources |
| `gender` | string (optional) | `female` \| `male` \| `other` \| `unknown` |
| `region` | string (optional) | Province / region label, e.g. `Koshi` (mostly `unknown` in v0) |
| `quality` | string (optional) | `clean` \| `phone` \| `noisy` (v0 is all `clean`) |
| `duration_s` | float (optional) | Clip duration in seconds |

### Data Fields

As above. Transcripts are the original source transcripts, NFC-normalised
at scoring time (not in the stored file).

### Data Splits

| Split | Source | Clips | Speakers | Hours | Status |
|---|---|---|---|---|---|
| test — OpenSLR | SLR54 held-out test split, speaker-disjoint from training | 13,419 | 49 test / 478 train | unknown (no durations in source) | Manifest built (v0.jsonl) |
| test — FLEURS | `google/fleurs`, `ne_np`, test split (726 sentences) | 726 | — (no speaker ids in source) | 2.28 h | Manifest built (v0.jsonl) |
| test — Common Voice | Common Voice Scripted Speech 27.0, `ne`, validated clips only | TBD | TBD | TBD | Pending manifest build |
| test — realistic (Phase 2) | Project's own collection: phone / noisy / dialect / code-switched | 0 | 0 | 0 h | **Pending — no audio collected yet** |

**There is no training split in this dataset.** It is test-only by
design. Training on any benchmark clip, speaker, or transcript
disqualifies a submission.

## Dataset Creation

### Curation Rationale

Nepali ASR had no single, fixed, public benchmark: the 2026
benchmark (Paudel & Sayami, TU MSc thesis) compared six models on
three test sets under one protocol, but there was no maintained
public test set with slice reporting and no realistic (phone/noisy/
dialect/code-switched) portion. v0 freezes the 2026 protocol into a
reusable test set and defines the realistic portion as a slot for the
project's own collection.

### Source Data

| Source | Size (source) | License | How it enters v0 |
|---|---|---|---|
| OpenSLR SLR54 | ~165 h, 527 speakers, ~157,905 utterances | CC BY-SA 4.0 | Held-out test split; audio bundled with attribution, shared under CC BY-SA 4.0 |
| FLEURS (`google/fleurs`, `ne_np`) | ~10 h per language; test = 726 sentences | CC BY 4.0 | Test split; audio bundled with attribution |
| Mozilla Common Voice (`ne`, Scripted Speech 27.0) | 40.55 MB download (release 27.0) | CC0 1.0 | **Manifest-only**: clip IDs, hashes, download instructions — audio NOT bundled, because Mozilla distributes Common Voice exclusively via the Mozilla Data Collective and its terms forbid re-hosting/re-sharing |
| Meta Seamless / SeamlessAlign | Metadata only, no audio | CC BY-NC 4.0 | **Not included as audio** (reference only) |
| IndicVoices Nepali (candidate v0.1) | 100 h train / 496 speakers + 5 h held-out / 114 speakers (secondary source; to re-confirm) | CC BY 4.0 | Candidate addition; not in v0 |

Licenses verified on the official pages on 2026-11-18; see
`benchmark/README.md` §7 for the full verification table and URLs.

### Annotations

Transcripts are the sources' own human transcripts. No new
annotation was created for v0. No synthetic audio, no TTS, no
machine-generated transcripts are in v0.

### Personal and Sensitive Information

- Speaker IDs in the manifest are the sources' anonymised IDs
  (SLR54 uses anonymised UserIDs; FLEURS/Common Voice carry no
  speaker names). No attempt may be made to identify speakers —
  this is a condition of the Common Voice terms and a rule of this
  benchmark.
- The future Phase 2 portion is collected under explicit consent
  and released CC0 by the project itself (see the project repo's
  `DATASET_LICENSE.md`); no personal data beyond the consented
  recording and its metadata is included.

## Considerations for Using the Data

### Social Impact of Dataset

A fixed Nepali benchmark lowers the barrier to measuring — and
therefore improving — speech recognition for ~32 million Nepali
speakers, most of whom are poorly served by current ASR. Sliced
reporting (gender, region, quality) is intended to make performance
gaps visible rather than hidden behind a single average.

### Discussion of Biases

- v0 is **read speech**: clean recordings of read sentences. It
  over-represents literate, standard-dialect, quiet-environment
  speech and under-represents the way Nepali is actually spoken.
- Gender balance depends on the sources; FLEURS records gender per
  clip, the other sources unevenly. Report the slices; do not
  assume balance.
- Region/province is essentially unrecorded in the public sources,
  so regional bias cannot be measured in v0 at all. That is a
  limitation, not a finding of "no bias".

### Other Known Limitations

- v0 is small in two of its three sets (FLEURS ne_np test = 726
  sentences; Common Voice ne is a ~40 MB release). Small test sets
  mean wide confidence intervals — a 1-point WER difference on
  these sets is not a reliable claim.
- The 2026 paper's models were fine-tuned on SLR54, so SLR54 test
  results are in-domain for them by construction. Always report
  training data alongside any number.
- The realistic portion (phone / noisy / dialect / code-switched)
  is **empty** until Phase 2 collection produces audio. Any claim
  that this benchmark covers those conditions before then is false.

## Additional Information

### Dataset Curators

Nepali speech dataset project — https://github.com/Samir-Khadka/nepali-speech
(Phase 3: Benchmark). No funding, no institutional affiliation.

### Licensing Information

Mixed, per source — see the Source Data table above and
`benchmark/README.md` §7. In short: FLEURS CC BY 4.0; Common Voice
CC0 1.0 (manifest-only due to Mozilla Data Collective terms);
OpenSLR SLR54 CC BY-SA 4.0 (share-alike applies to redistributed
portions); IndicVoices CC BY 4.0; Seamless CC BY-NC 4.0
(manifest/reference only). The scoring code is MIT (repo `LICENSE`).
The future Phase 2 portion is CC0 (repo `DATASET_LICENSE.md`).

### Citation Information

```bibtex
@inproceedings{kjartansson-etal-sltu2018,
  title = {{Crowd-Sourced Speech Corpora for Javanese, Sundanese,
            Sinhala, Nepali, and Bangladeshi Bengali}},
  author = {Oddur Kjartansson and Supheakmungkol Sarin and
            Knot Pipatsrisawat and Martin Jansche and Linne Ha},
  booktitle = {Proc. SLTU},
  year = {2018},
  pages = {52--55},
}
@inproceedings{fleurs2022,
  title = {FLEURS: Few-shot Learning Evaluation of Universal
           Representations of Speech},
  author = {Conneau, Alexis and Ma, Min and Khanuja, Simran and
            Zhang, Yu and Axelrod, Vera and Dalmia, Siddharth and
            Riesa, Jason and Rivera, Clara and Bapna, Ankur},
  journal = {arXiv preprint arXiv:2205.12446},
  year = {2022},
}
@inproceedings{commonvoice,
  author = {Ardila, R. and Branson, M. and Davis, K. and Henretty, M.
            and Kohler, M. and Meyer, J. and Morais, R. and Saunders, L.
            and Tyers, F. M. and Weber, G.},
  title = {Common Voice: A Massively-Multilingual Speech Corpus},
  booktitle = {Proceedings of LREC 2020},
  pages = {4211--4215},
}
@inproceedings{JavedNGJBMSAFPR24,
  author = {Tahir Javed and Janki Nawale and Eldho Ittan George and
            Sakshi Joshi and Kaushal Santosh Bhogale and Deovrat Mehendale
            and Ishvinder Virender Sethi and Aparna Ananthanarayanan and
            Hafsah Faquih and Pratiti Palit and Sneha Ravishankar and
            Saranya Sukumaran and Tripura Panchagnula and Sunjay Murali and
            Kunal Sharad Gandhi and Ambujavalli R and Manickam K. M and
            C. Venkata Vaijayanthi and Krishnan Srinivasa Raghavan Karunganni
            and Pratyush Kumar and Mitesh M. Khapra},
  title = {IndicVoices: Towards building an Inclusive Multilingual Speech
           Dataset for Indian Languages},
  booktitle = {ACL (Findings)},
  pages = {10740--10782},
  year = {2024},
}
```

Benchmark protocol reference (numbers in the leaderboard marked
"Paudel & Sayami 2026" come from here):

- Paudel & Sayami (2026), TU MSc thesis — Nepali ASR benchmark:
  https://p-sumann.github.io/nepali-asr-benchmark/
