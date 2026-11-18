# Prompts

`prompts.json` holds the sentences contributors read aloud when recording.

**All sentences are original, written for this project.** No text has been
copied from any published source — no news sites, books, Wikipedia, lyrics,
or quotes. Every entry has `"source": "original"`.

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
