# ETIS Dataset Schema

Locked schema for `data/processed/dataset.csv`. All data-collection scripts should output rows matching this exactly.

| Column | Type | Description | Example |
|---|---|---|---|
| `id` | string | Unique ID, prefixed by source | `PT-000123`, `SYN-000045` |
| `text` | string | The URL, or the email/message body | `https://inst4pay-secure.com/login` |
| `content_type` | enum | `url`, `text` or `email` | `url` |
| `label` | enum | `phishing` or `legitimate` | `phishing` |
| `language` | enum | `ar`, `en`, or `mixed` | `ar` |
| `source` | string | Where the row came from | `phishtank`, `openphish`, `tranco`, `synthetic_egyptian_brands`, `manual`, `real_collected_messages` |
| `brand_impersonated` | string or empty | Brand being spoofed, if any | `Instapay`, `CIB`, `Vodafone`, (empty) |
| `date_collected` | date (YYYY-MM-DD) | When the row was pulled/created | `2026-07-11` |
| `split` | enum | Filled in later by the stratified splitter | `train`, `val`, `test` |
| `label_detail` | enum | `phishing`, `spam`, `scam`, or `legitimate`. `label` stays binary for compatibility. | `scam` |

## ID prefix convention

- `PT-` PhishTank
- `OP-` OpenPhish
- `TR-` Tranco (legitimate)
- `SYN-` synthetic Arabic brand-impersonation samples
- `MAN-` manually collected/curated samples
- `RM-` real messages collected from contributors (`real_collected_messages`); the id is a hash of the
  normalised text, so it is stable when more files are ingested

Other sources (the published Arabic corpora and the crawled legitimate pages) use their own prefixes;
list them here when this file is next reviewed.

## File locations

- `data/raw/` — untouched pulls from each source (e.g. `data/raw/phishtank_raw.csv`)
- `data/processed/dataset.csv` — the single unified file in this schema, used by everything downstream (rule engine testing, blacklist seeding, model training)
- `data/processed/split_groups.csv` — id, group_id, group_type for every row (host / shared_platform / text_cluster / text_single / synthetic)
- `data/processed/real_messages_meta.csv` — id, category, channel, contributor, brand, label_detail, source_file for the `RM-` rows (no message text)
- `data/raw/real_messages/` — contributor files and `_rejects.csv`. Personal data: **git-ignored, never committed.**

## Notes

- `split` stays empty until the stratified 70/15/15 split step runs — don't hand-assign it earlier.
- One unified file (not separate URL/text files) keeps the model pipelines and Chapter 3.4 statistics simpler.
- Real collected messages are anonymised by the contributor (see `data/raw/COLLECTION_GUIDE.md`) and checked by
  `data/scripts/ingest_real_messages.py`. For `RM-` rows `label` is `legitimate` or `phishing`, and
  `label_detail` keeps `phishing` / `scam` / `spam` / `legitimate`. `brand_impersonated` is only filled for
  non-legitimate rows.
