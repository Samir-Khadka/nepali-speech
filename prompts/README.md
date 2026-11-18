# Prompts

`prompts.json` holds the sentences contributors read aloud when recording.

**All sentences are original, written for this project.** No text has been
copied from any published source — no news sites, books, Wikipedia, lyrics,
or quotes. Every entry has `"source": "original"`.

## Pool size

- Phase 1 (2026-11-18): 200 prompts — farming 40, commands 30, weather 25,
  prices 25, daily-life 25, numbers 20, dates 15, addresses 10,
  code-switched 10.
- Phase 2 (2026-11-18): +900 prompts (IDs 201–1100) — farming +200,
  commands +120, weather +100, prices +100, daily-life +100, numbers +80,
  dates +60, addresses +40, code-switched +100.
- Total: 1,100 prompts — farming 240, commands 150, weather 125, prices 125,
  daily-life 125, numbers 100, dates 75, addresses 50, code-switched 110.
- Phase 2 pool target: ~1,200 sentences for the 50-hour first batch —
  100 prompts still to go.

## Format

Each entry is one object in the JSON array:

```json
{"id": 1, "text": "असारमा धान रोप्ने बेला भइसक्यो।", "domain": "farming", "language": "ne", "source": "original"}
```

- `id` — sequential integer, starting at 1.
- `text` — one short, speakable sentence (5–15 words).
- `domain` — one of: `farming`, `weather`, `prices`, `numbers`, `dates`,
  `addresses`, `commands`, `daily-life`, `code-switched`.
- `language` — `"ne"` for Nepali, `"ne-en"` for Nepali–English code-switched
  sentences (domain `code-switched` only).
- `source` — always `"original"`.

## Adding more prompts

1. Write the sentences yourself, in natural spoken Nepali. Do not copy
   text from anywhere.
2. Keep each sentence to one sentence, 5–15 words, with no personal data,
   no names of real private individuals, and no political or religious
   content.
3. Append the new entries to the end of the array in `prompts.json`.
4. Keep the `id` values sequential — continue from the highest existing id.
5. Set the `domain` and `language` fields correctly (`"ne-en"` only for
   code-switched sentences).
6. Validate the file is still valid JSON before committing, e.g.
   `python3 -m json.tool prompts/prompts.json > /dev/null`.
