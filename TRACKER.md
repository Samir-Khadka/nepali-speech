# nepali-speech — Phase 2 Tracker

Phase 2 is the collection push: recruit speakers across Nepal, record prompts,
review them, and grow the dataset toward a diverse, usable corpus.

## Goal

**50 approved hours** of Nepali speech — roughly **~36,000 recordings** at an
average of ~5 seconds per recording. Hours are always an *estimate* (5 s
average per recording) until real durations are measured.

## Status

Last updated from `GET /api/admin/stats`: **not yet updated — all counts are
zero at the start of Phase 2.**

| Metric | Count |
|---|---|
| Speakers onboarded (contributors) | 0 |
| Recordings submitted | 0 |
| Hours recorded (estimate) | 0 |
| Hours reviewed / approved (estimate) | 0 |
| Recordings approved | 0 |
| Recordings rejected | 0 |
| Recordings pending review | 0 |

## Coverage by province

Target: recordings from **all 7 provinces**.

| Province | Recordings | Approved hours (est.) |
|---|---|---|
| Koshi | 0 | 0 |
| Madhesh | 0 | 0 |
| Bagmati | 0 | 0 |
| Gandaki | 0 | 0 |
| Lumbini | 0 | 0 |
| Karnali | 0 | 0 |
| Sudurpashchim | 0 | 0 |
| Unspecified | 0 | 0 |

## Coverage by gender

Target: a balanced mix of speakers across genders.

| Gender | Recordings |
|---|---|
| (no data yet) | 0 |

## Coverage by age range

Target: speakers across age bands, not only one generation.

| Age range | Recordings |
|---|---|
| (no data yet) | 0 |

## Diversity targets

- **All 7 provinces**: Koshi, Madhesh, Bagmati, Gandaki, Lumbini, Karnali,
  Sudurpashchim — no province left at zero.
- **Gender balance**: aim for a roughly even split; watch the gender table
  after each review batch and recruit where a group is missing.
- **Age range**: cover younger and older speakers, not a single age band.
- **Urban + rural**: recruit beyond the big cities — rural speakers and
  regional accents are part of the goal, not an afterthought.

## How to update

1. After each review batch, open the admin stats page (or call
   `GET /api/admin/stats` with the `X-Admin-Email` admin header).
2. Copy the numbers into the tables above: `contributors`,
   `recordings_total`, `by_status`, `by_province`, `by_gender`,
   `by_age_range`, and `hours_estimate` (approved / recorded hours).
3. Update the "Last updated" line with the date and the batch covered.
4. **Never put personal data in this file** — no names, phone numbers, email
   addresses, or anything that identifies an individual contributor. Only
   aggregate counts go here.
