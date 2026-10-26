"""
clean_dataset.py

Step 1 of the evaluation rebuild (thesis-expansion plan). Run AFTER all the
ingestion scripts and BEFORE the stratified/grouped split.

What it does to data/processed/dataset.csv:

1. Adds a 10th column, `label_detail`, with four values:
       phishing | spam | scam | legitimate
   The existing binary `label` column is left untouched, so every existing
   script (calibration, LSTM preprocessing, baselines) keeps working.
   Mapping (documented in docs/dataset_schema.md):
       github_arabic_text_spam_detection + label=phishing -> spam
           (the source's folder is literally called "spam"; it was mapped to
            "phishing" in merge_arabic_sources.py)
       kaggle_arabic_phishing_fully_dataset + label=phishing -> scam
           (translated 419 / advance-fee-fraud letters, not credential phishing)
       everything else -> same as `label`

2. Removes duplicates ACROSS sources (the old merge only removed them
   within one source):
       - text rows: key = normalised text (NFKC, no diacritics/tatweel,
         alef/yaa unified, punctuation stripped, whitespace collapsed)
       - url rows:  key = canonical URL (scheme-insensitive, lowercase host,
         no www., no default port, no fragment, no trailing slash)
   If the duplicates agree on `label`, one row is kept (highest-priority
   source, see SOURCE_PRIORITY). If they DISAGREE on `label`, every row in
   the group is dropped (the label is unreliable) and listed in
   dedup_conflicts.csv.

3. Never touches synthetic rows' content. They stay identifiable by
   source == "synthetic_egyptian_brands" / SYN- ids. The grouped split
   (step 3) must keep them out of the test set and report them separately.

Safety:
    - first real run copies dataset.csv to dataset.pre_clean.csv (never overwritten)
    - --dry-run prints the report and writes nothing
    - idempotent: running it twice gives the same result

Usage (from the project root):
    python data/scripts/clean_dataset.py --dry-run
    python data/scripts/clean_dataset.py
"""

import argparse
import csv
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # scripts/ -> data/ -> root
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATASET_PATH = PROCESSED_DIR / "dataset.csv"
BACKUP_PATH = PROCESSED_DIR / "dataset.pre_clean.csv"
CONFLICTS_PATH = PROCESSED_DIR / "dedup_conflicts.csv"
REPORT_PATH = PROCESSED_DIR / "clean_report.txt"

DATASET_COLUMNS = [
    "id", "text", "content_type", "label", "language",
    "source", "brand_impersonated", "date_collected", "split",
    "label_detail",
]

GITHUB_SPAM_SOURCE = "github_arabic_text_spam_detection"
KAGGLE_SCAM_SOURCE = "kaggle_arabic_phishing_fully_dataset"
SYNTHETIC_SOURCE = "synthetic_egyptian_brands"
# Real messages collected from contributors (data/scripts/ingest_real_messages.py).
# They arrive with label_detail already set (legitimate / phishing / scam / spam),
# which this script must preserve rather than recompute.
REAL_SOURCE = "real_collected_messages"

# When duplicates agree on label, keep the row from the earliest source in
# this list. Real, peer-reviewed / authentic sources first; translated and
# synthetic last. Unknown sources sort after everything listed here.
SOURCE_PRIORITY = [
    REAL_SOURCE,  # genuinely received messages beat every translated / scraped copy
    "salloum_gaber_bilingual_corpus",
    "kaggle_dialectal_arabic_sms",
    GITHUB_SPAM_SOURCE,
    KAGGLE_SCAM_SOURCE,
    "manual",
    "phishtank",
    "openphish",
    "tranco",
    SYNTHETIC_SOURCE,
]

_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670\u0640]")  # tashkeel + tatweel
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
_DEFAULT_PORT = re.compile(r":(80|443)$")


def normalise_newlines(text: str) -> str:
    """Line breaks inside a message are always stored as \n.

    On Windows, git's autocrlf can check dataset.csv out with \r\n inside quoted
    messages (360 manual rows in the committed data have line breaks). Those \r\n
    reach the models as stray characters, so the stored text would depend on the
    machine. Normalising here makes the pipeline platform-independent.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def assign_label_detail(row: dict) -> str:
    label = row["label"].strip()
    if row["source"] == REAL_SOURCE and row.get("label_detail") in ("phishing", "scam", "spam", "legitimate"):
        return row["label_detail"]
    if label == "phishing" and row["source"] == GITHUB_SPAM_SOURCE:
        return "spam"
    if label == "phishing" and row["source"] == KAGGLE_SCAM_SOURCE:
        return "scam"
    return label


def normalise_text_key(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).lower()
    t = _DIACRITICS.sub("", t)
    t = re.sub(r"[أإآٱ]", "ا", t)
    t = t.replace("ى", "ي")
    t = _NON_WORD.sub(" ", t)
    return _WHITESPACE.sub(" ", t).strip()


def canonical_url_key(url: str) -> str:
    raw = url.strip()
    parsed = urlsplit(raw if "://" in raw else f"http://{raw}")
    host = _DEFAULT_PORT.sub("", (parsed.netloc or "").lower())
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path.rstrip("/")
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{host}{path}{query}"


def dedup_key(row: dict) -> tuple[str, str]:
    if row["content_type"] == "url":
        return ("url", canonical_url_key(row["text"]))
    return ("text", normalise_text_key(row["text"]))


def source_rank(row: dict) -> tuple[int, str]:
    try:
        rank = SOURCE_PRIORITY.index(row["source"])
    except ValueError:
        rank = len(SOURCE_PRIORITY)
    return (rank, row["id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = parser.parse_args()

    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows_in = len(rows)

    for row in rows:
        row["text"] = normalise_newlines(row["text"])
        row["label_detail"] = assign_label_detail(row)

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = dedup_key(row)
        if key[1]:  # empty key (e.g. text that is only punctuation) is never merged
            groups[key].append(row)

    drop_ids: set[str] = set()
    conflict_rows: list[dict] = []
    stats = Counter()
    leaking_groups_before = 0

    for (kind, _), members in groups.items():
        if len(members) < 2:
            continue
        stats[f"{kind}_duplicate_groups"] += 1

        splits = {m["split"] for m in members if m["split"]}
        if len(splits) > 1:
            leaking_groups_before += 1
            stats[f"{kind}_groups_crossing_splits"] += 1

        labels = {m["label"] for m in members}
        if len(labels) > 1:
            stats[f"{kind}_conflicting_label_groups"] += 1
            for m in members:
                drop_ids.add(m["id"])
                conflict_rows.append({
                    "id": m["id"], "source": m["source"], "label": m["label"],
                    "content_type": m["content_type"], "text": m["text"][:200],
                })
            continue

        keeper = min(members, key=source_rank)
        for m in members:
            if m["id"] != keeper["id"]:
                drop_ids.add(m["id"])
                stats[f"{kind}_rows_removed_as_duplicate"] += 1

    kept = [r for r in rows if r["id"] not in drop_ids]

    # ---- report -----------------------------------------------------------
    removed_by_source = Counter(r["source"] for r in rows if r["id"] in drop_ids)
    detail_counts = Counter(r["label_detail"] for r in kept)
    split_counts = Counter(r["split"] or "(empty)" for r in kept)

    lines = [
        "ETIS dataset cleaning report" + ("  [DRY RUN]" if args.dry_run else ""),
        f"Rows in:  {rows_in:,}",
        f"Rows out: {len(kept):,}   (removed {len(drop_ids):,})",
        "",
        "Duplicate handling:",
    ]
    for k in sorted(stats):
        lines.append(f"  {k}: {stats[k]:,}")
    lines += [
        f"  groups that crossed train/val/test BEFORE cleaning: {leaking_groups_before:,}",
        "",
        "Rows removed, by source:",
    ]
    for src, n in removed_by_source.most_common():
        lines.append(f"  {src}: {n:,}")
    lines += ["", "label_detail after cleaning:"]
    for k, n in detail_counts.most_common():
        lines.append(f"  {k}: {n:,}")
    lines += ["", "Existing split column after cleaning (unchanged assignments):"]
    for k, n in split_counts.most_common():
        lines.append(f"  {k}: {n:,}")
    lines += [
        "",
        f"Conflicting-label rows dropped: {len(conflict_rows):,} (see {CONFLICTS_PATH.name})",
        f"Synthetic rows kept: {sum(1 for r in kept if r['source'] == SYNTHETIC_SOURCE):,}",
    ]
    report = "\n".join(lines)
    print(report)

    if args.dry_run:
        return

    if not BACKUP_PATH.exists():
        shutil.copy2(DATASET_PATH, BACKUP_PATH)
        print(f"\nBackup written: {BACKUP_PATH}")

    with open(DATASET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DATASET_COLUMNS)
        writer.writeheader()
        writer.writerows(kept)

    with open(CONFLICTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "source", "label", "content_type", "text"])
        writer.writeheader()
        writer.writerows(conflict_rows)

    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    print(f"\nWrote {DATASET_PATH}, {CONFLICTS_PATH.name}, {REPORT_PATH.name}")


if __name__ == "__main__":
    main()
