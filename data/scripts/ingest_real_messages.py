"""
ingest_real_messages.py

Step B of the demo-readiness plan: bring the real Egyptian messages collected
with data/raw/COLLECTION_GUIDE.md into the unified dataset.

Input   data/raw/real_messages/messages_<initials>.csv   (one file per contributor)
        columns: text, label, category, channel, language, brand,
                 contributor, received_in_real_life
        Saved as CSV UTF-8 (a BOM from Excel is handled).

What it does
  1. Validates every row against the guide (allowed label / category / channel /
     language values, label-vs-category consistency, received_in_real_life = yes).
  2. ANONYMISATION CHECK. Rows containing phone numbers, 13+ digit numbers (cards,
     accounts, IBANs, national IDs) or e-mail addresses are REJECTED (hard rules).
     Long reference numbers, links with long random tokens and language/script
     mismatches are reported as warnings (soft rules, row is kept). Names and
     real balances cannot be detected by a script: the contributor checklist
     in the guide is the only control for those.
  3. Removes duplicates inside the collected set (same normalised text, same
     label -> one row; same text with different labels -> both dropped).
  4. Replaces any earlier real_collected_messages rows in data/processed/dataset.csv
     with the new set (re-runnable; ids are a hash of the text, so they stay
     stable when files are added later), keeping the 10-column schema.
     binary `label`: legitimate stays legitimate; phishing/scam/spam -> phishing.
     `label_detail` keeps phishing / scam / spam / legitimate.
     `split` is left EMPTY: run clean_dataset.py and create_grouped_split.py next.
  5. Writes data/processed/real_messages_meta.csv (id, category, channel,
     contributor, brand, label_detail, source_file; no message text) so Chapter 5
     can report results per category.

Privacy rules for this script's own output
  - Nothing printed or written to a tracked folder contains message text.
    Problems are reported as  file : row number : rule.
  - Rejected rows are written, with text, only to
    data/raw/real_messages/_rejects.csv, which sits in the git-ignored raw folder.

Usage (from the project root):
    python data/scripts/ingest_real_messages.py --dry-run     # check only, write nothing
    python data/scripts/ingest_real_messages.py               # refuses if any row is rejected
    python data/scripts/ingest_real_messages.py --drop-flagged   # skip rejected rows, ingest the rest
"""

import argparse
import csv
import hashlib
import re
import shutil
import sys
from collections import Counter
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # scripts/ -> data/ -> root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from clean_dataset import DATASET_COLUMNS, REAL_SOURCE, normalise_newlines, normalise_text_key  # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "real_messages"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATASET_PATH = PROCESSED_DIR / "dataset.csv"
BACKUP_PATH = PROCESSED_DIR / "dataset.pre_ingest.csv"
META_PATH = PROCESSED_DIR / "real_messages_meta.csv"
REPORT_PATH = PROCESSED_DIR / "ingest_report.txt"
REJECTS_PATH = RAW_DIR / "_rejects.csv"  # raw folder is git-ignored; contains text

REQUIRED_COLUMNS = ["text", "label", "category", "channel", "language",
                    "brand", "contributor", "received_in_real_life"]
LABELS = {"legitimate", "phishing", "scam", "spam"}
LEGITIMATE_CATEGORIES = {
    "bank_notification", "otp", "wallet_payment", "telecom", "delivery",
    "government_service", "personal_work", "anti_fraud_warning",
}
PHISHING_CATEGORY = "phishing_received"
CHANNELS = {"sms", "whatsapp", "email"}
LANGUAGES = {"ar", "en", "mixed"}

MIN_TEXT_LENGTH = 5

# The two example rows shipped inside collection_template.csv. A contributor who copies the
# template and forgets to delete them would otherwise inject two invented messages.
TEMPLATE_EXAMPLE_TEXTS = (
    'بنك مصر: تم خصم مبلغ 320.00 جنيه من حسابكم المنتهي بـ XXXX. للاستفسار اتصل بالخط الساخن.',
    'كود التحقق الخاص بك هو 482915. لا تشاركه مع أحد.',
)
TEMPLATE_EXAMPLE_KEYS = {normalise_text_key(t) for t in TEMPLATE_EXAMPLE_TEXTS}
LONG_TEXT_WARNING = 4000
MAX_ROWS_LISTED = 40  # per rule, in the console report

# ---------------------------------------------------------------------------
# Anonymisation check
# ---------------------------------------------------------------------------

_ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹"
_ASCII_DIGITS = "01234567890123456789"
_DIGIT_TABLE = str.maketrans(_ARABIC_INDIC, _ASCII_DIGITS)

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_SEPARATOR_BETWEEN_DIGITS = re.compile(r"(?<=\d)[ \-.](?=\d)")
_LONG_NUMBER = re.compile(r"(?<!\d)\d{13,}(?!\d)")
_EG_MOBILE = re.compile(r"(?<!\d)(?:(?:\+|00)20|0)1[0125]\d{8}(?!\d)")
_PHONE_LIKE = re.compile(r"(?<!\d)0\d{9,10}(?!\d)")
_INTERNATIONAL = re.compile(r"\+\d{8,15}(?!\d)")
_REFERENCE_NUMBER = re.compile(r"(?<!\d)\d{9,12}(?!\d)")
_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_URL_TOKEN = re.compile(r"[A-Za-z0-9_-]{16,}")
_PLACEHOLDERS = re.compile(r"\[[A-Z]+\]|X{3,}", re.IGNORECASE)
_ARABIC_LETTER = re.compile(r"[\u0600-\u06FF]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")


def hard_anonymisation_hits(text: str) -> list[str]:
    """Rules that must reject the row (personal identifiers)."""
    t = text.translate(_DIGIT_TABLE)
    collapsed = _SEPARATOR_BETWEEN_DIGITS.sub("", t)
    hits = []
    if _EMAIL.search(t):
        hits.append("email_address")
    if _LONG_NUMBER.search(collapsed):
        hits.append("card_account_or_id_number")
    if _EG_MOBILE.search(collapsed) or _PHONE_LIKE.search(collapsed) or _INTERNATIONAL.search(collapsed):
        hits.append("phone_number")
    return hits


def soft_warnings(text: str, language: str) -> list[str]:
    """Rules that are reported for a human to look at; the row is kept."""
    t = text.translate(_DIGIT_TABLE)
    collapsed = _SEPARATOR_BETWEEN_DIGITS.sub("", t)
    warnings = []
    if _REFERENCE_NUMBER.search(collapsed) and not _EG_MOBILE.search(collapsed) \
            and not _PHONE_LIKE.search(collapsed):
        warnings.append("long_reference_number")
    for url in _URL.findall(t):
        after_host = url.split("://", 1)[1].partition("/")[2]
        if _URL_TOKEN.search(after_host):
            warnings.append("url_with_long_token")
            break

    stripped = _PLACEHOLDERS.sub(" ", _URL.sub(" ", text))
    has_arabic = bool(_ARABIC_LETTER.search(stripped))
    has_latin = bool(_LATIN_LETTER.search(stripped))
    if language == "ar" and not has_arabic:
        warnings.append("language_ar_but_no_arabic_letters")  # Franco-Arabic is legitimate: check by eye
    elif language == "en" and has_arabic:
        warnings.append("language_en_but_has_arabic_letters")
    elif language == "mixed" and not (has_arabic and has_latin):
        warnings.append("language_mixed_but_single_script")
    if len(text) > LONG_TEXT_WARNING:
        warnings.append("very_long_text")
    return warnings


# ---------------------------------------------------------------------------
# Reading and validating
# ---------------------------------------------------------------------------

def read_contributor_file(path: Path) -> tuple[list[dict], list[str]]:
    """Returns (rows with a __row number, problems that stop the whole file)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = [(name or "").strip().lower() for name in (reader.fieldnames or [])]
        missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
        if missing:
            return [], [f"missing column(s): {', '.join(missing)}"]
        reader.fieldnames = fieldnames
        rows = []
        for number, row in enumerate(reader, start=2):  # row 1 is the header
            clean = {k: (v or "").strip() for k, v in row.items() if k}
            clean["text"] = normalise_newlines(clean.get("text", ""))  # Excel/Windows may use \r\n inside a cell
            clean["__row"] = number
            rows.append(clean)
    return rows, []


def validate_row(row: dict) -> list[str]:
    """Structural problems -> the row is rejected."""
    problems = []
    text = row["text"]
    label = row["label"].lower()
    category = row["category"].lower()
    if len(text) < MIN_TEXT_LENGTH:
        problems.append("text_empty_or_too_short")
    if normalise_text_key(text) in TEMPLATE_EXAMPLE_KEYS:
        problems.append("template_example_row")
    if label not in LABELS:
        problems.append("bad_label")
    if row["channel"].lower() not in CHANNELS:
        problems.append("bad_channel")
    if row["language"].lower() not in LANGUAGES:
        problems.append("bad_language")
    if row["received_in_real_life"].lower() != "yes":
        problems.append("not_received_in_real_life")
    if not row["contributor"]:
        problems.append("missing_contributor")
    if category not in LEGITIMATE_CATEGORIES and category != PHISHING_CATEGORY:
        problems.append("bad_category")
    elif label in LABELS:
        if category == PHISHING_CATEGORY and label == "legitimate":
            problems.append("label_legitimate_but_category_phishing_received")
        if category in LEGITIMATE_CATEGORIES and label != "legitimate":
            problems.append("label_not_legitimate_but_category_is_legitimate_type")
    return problems


def message_id(text: str) -> str:
    return "RM-" + hashlib.sha1(normalise_text_key(text).encode("utf-8")).hexdigest()[:10].upper()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="check and report only, write nothing")
    parser.add_argument("--drop-flagged", action="store_true",
                        help="exclude rejected rows and ingest the rest (default: refuse if any row is rejected)")
    parser.add_argument("--date", default=date.today().isoformat(), help="date_collected value (YYYY-MM-DD)")
    parser.add_argument("--input-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    files = sorted(p for p in args.input_dir.glob("messages_*.csv"))
    if not files:
        sys.exit(f"No messages_*.csv files found in {args.input_dir}")

    accepted: list[dict] = []
    rejected: list[dict] = []  # with text, for the raw-folder rejects file
    reject_rules: Counter = Counter()
    warning_rules: Counter = Counter()
    listing: dict[str, list[str]] = {}
    file_problems: list[str] = []
    rows_read = 0

    def note(rule: str, where: str) -> None:
        listing.setdefault(rule, []).append(where)

    for path in files:
        rows, problems = read_contributor_file(path)
        if problems:
            file_problems += [f"{path.name}: {p}" for p in problems]
            continue
        for row in rows:
            rows_read += 1
            where = f"{path.name}:{row['__row']}"
            rules = validate_row(row)
            rules += hard_anonymisation_hits(row["text"]) if row["text"] else []
            if rules:
                for rule in rules:
                    reject_rules[rule] += 1
                    note(rule, where)
                rejected.append({**row, "rejected_for": ";".join(rules), "source_file": path.name})
                continue
            for rule in soft_warnings(row["text"], row["language"].lower()):
                warning_rules[rule] += 1
                note("WARN " + rule, where)
            row["source_file"] = path.name
            accepted.append(row)

    # ---- duplicates inside the collected set ---------------------------------
    by_key: dict[str, list[dict]] = {}
    for row in accepted:
        by_key.setdefault(normalise_text_key(row["text"]) or row["text"], []).append(row)
    unique: list[dict] = []
    duplicates_removed = 0
    label_conflicts = 0
    for members in by_key.values():
        binary = {("legitimate" if m["label"].lower() == "legitimate" else "phishing") for m in members}
        if len(binary) > 1:
            label_conflicts += 1
            duplicates_removed += len(members)
            for m in members:
                note("label_conflict_between_duplicates", f"{m['source_file']}:{m['__row']}")
            continue
        unique.append(members[0])
        duplicates_removed += len(members) - 1

    # ---- build dataset rows + meta ----------------------------------------------
    dataset_rows, meta_rows = [], []
    for row in sorted(unique, key=lambda r: message_id(r["text"])):
        label = row["label"].lower()
        category = row["category"].lower()
        channel = row["channel"].lower()
        legit = label == "legitimate"
        dataset_rows.append({
            "id": message_id(row["text"]),
            "text": row["text"],
            "content_type": "email" if channel == "email" else "text",
            "label": "legitimate" if legit else "phishing",
            "language": row["language"].lower(),
            "source": REAL_SOURCE,
            "brand_impersonated": "" if legit else row["brand"],
            "date_collected": args.date,
            "split": "",
            "label_detail": label,
        })
        meta_rows.append({
            "id": message_id(row["text"]), "category": category, "channel": channel,
            "contributor": row["contributor"], "brand": row["brand"],
            "label_detail": label, "source_file": row["source_file"],
        })

    # ---- report (no message text anywhere) ----------------------------------------
    detail = Counter(r["label_detail"] for r in dataset_rows)
    lines = ["ETIS real-message ingest report" + ("  [DRY RUN]" if args.dry_run else ""),
             f"Files read: {len(files)}   rows read: {rows_read:,}",
             f"Rejected (hard rules / structure): {len(rejected):,}",
             f"Duplicates removed inside the collected set: {duplicates_removed:,}"
             f"   (label-conflict groups dropped: {label_conflicts})",
             f"Rows to ingest: {len(dataset_rows):,}", ""]
    if file_problems:
        lines += ["FILES SKIPPED:"] + [f"  {p}" for p in file_problems] + [""]
    lines.append("By label_detail: " + ", ".join(f"{k}={v}" for k, v in detail.most_common()))
    lines.append("By category x label (legitimate / non-legitimate):")
    cat_counts: Counter = Counter()
    for m in meta_rows:
        cat_counts[(m["category"], "legitimate" if m["label_detail"] == "legitimate" else "phishing-like")] += 1
    for (cat, kind), n in sorted(cat_counts.items()):
        lines.append(f"  {cat:<20}{kind:<15}{n:>5}")
    lines.append("By channel: " + ", ".join(f"{k}={v}" for k, v in Counter(m["channel"] for m in meta_rows).most_common()))
    lines.append("By language: " + ", ".join(f"{k}={v}" for k, v in Counter(r["language"] for r in dataset_rows).most_common()))
    lines.append("By contributor: " + ", ".join(f"{k}={v}" for k, v in Counter(m["contributor"] for m in meta_rows).most_common()))
    if reject_rules:
        lines += ["", "Reject rules: " + ", ".join(f"{k}={v}" for k, v in reject_rules.most_common())]
    if warning_rules:
        lines += ["Warnings (rows kept, please look): " + ", ".join(f"{k}={v}" for k, v in warning_rules.most_common())]
    lines += ["", "Where (file:row) -- message text is never printed:"]
    for rule in sorted(listing):
        shown = listing[rule][:MAX_ROWS_LISTED]
        extra = len(listing[rule]) - len(shown)
        lines.append(f"  {rule}: " + ", ".join(shown) + (f"  (+{extra} more)" if extra else ""))
    report = "\n".join(lines)
    print(report)

    if args.dry_run:
        return
    if (rejected or file_problems) and not args.drop_flagged:
        sys.exit("\nRefusing to write: fix the rows/files above, or re-run with --drop-flagged "
                 "to ingest everything that passed. Nothing was changed.")

    # ---- write -----------------------------------------------------------------------
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        existing = [r for r in csv.DictReader(f) if r["source"] != REAL_SOURCE]
    if not BACKUP_PATH.exists():
        shutil.copy2(DATASET_PATH, BACKUP_PATH)
        print(f"\nBackup written: {BACKUP_PATH}")
    with open(DATASET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DATASET_COLUMNS)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerows(dataset_rows)
    with open(META_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "category", "channel", "contributor",
                                               "brand", "label_detail", "source_file"])
        writer.writeheader()
        writer.writerows(meta_rows)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    if rejected:
        with open(REJECTS_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["source_file", "__row", "rejected_for"] + REQUIRED_COLUMNS,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rejected)
        print(f"Rejected rows (with text) saved to the git-ignored {REJECTS_PATH}")

    print(f"\nWrote {len(dataset_rows):,} real rows into {DATASET_PATH.name} "
          f"(replacing any earlier real rows), plus {META_PATH.name} and {REPORT_PATH.name}.")
    print("Next: python data/scripts/clean_dataset.py  then  python data/scripts/create_grouped_split.py")


if __name__ == "__main__":
    main()
