"""
merge_arabic_sources.py

Merges the 4 vetted real Arabic sources into data/processed/dataset.csv:
  1. Salloum/Gaber et al. bilingual academic corpus (phishing + legit, 300 rows)
  2. Kaggle "Fully Dataset" Arabic phishing/legit emails (1,250 rows, translated 419-style)
  3. Dialectal Arabic phishing/safe SMS dataset (500 rows, deduplicated -> ~333 unique)
  4. GitHub Arabic-Text-Spam-Detection raw messages (78 rows, Saudi dialect)

All rows get the MAN- prefix (manually collected/curated, not an automated feed).
See docs/dataset_schema.md for column definitions and docs/arabic_sources_citations.md
for the citations each of these needs in Chapter 3.3 / References.

Usage:
    python merge_arabic_sources.py
"""

import csv
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # scripts/ -> data/ -> root
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "arabic_sources"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATASET_PATH = PROCESSED_DIR / "dataset.csv"

DATASET_COLUMNS = [
    "id", "text", "content_type", "label", "language",
    "source", "brand_impersonated", "date_collected", "split",
]

TODAY = date.today().isoformat()


def load_salloum_gaber_corpus() -> list[dict]:
    """Source 1: bilingual academic corpus -- use the Arabic Text column only."""
    folder = RAW_DIR / "salloum_gaber_corpus"
    rows = []

    for filename, label in [("phishing-bilingual.csv", "phishing"), ("legit-bilingual.csv", "legitimate")]:
        path = folder / filename
        with open(path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for entry in reader:
                text = entry.get("Arabic Text", "").strip()
                if text:
                    rows.append({
                        "text": text,
                        "content_type": "text",
                        "label": label,
                        "language": "ar",
                        "source": "salloum_gaber_bilingual_corpus",
                    })
    return rows


def load_kaggle_fully_dataset() -> list[dict]:
    """Source 2: 1,250 rows, label derived from 'Phish'/'Legal' prefix in the ID column."""
    path = RAW_DIR / "kaggle_fully_dataset" / "Arabic Phishing and Legitimate emails - Fully Dataset.xlsx"
    df = pd.read_excel(path)
    df["derived_label"] = df["نوع الإيميل"].str.extract(r"^(Phish|Legal)")

    rows = []
    for _, entry in df.iterrows():
        text = str(entry["محتوى الإيميل"]).strip()
        derived = entry["derived_label"]
        if not text or pd.isna(derived):
            continue
        label = "phishing" if derived == "Phish" else "legitimate"
        rows.append({
            "text": text,
            "content_type": "text",
            "label": label,
            "language": "ar",
            "source": "kaggle_arabic_phishing_fully_dataset",
        })
    return rows


def load_kaggle_dialectal() -> list[dict]:
    """Source 3: 500 rows, deduplicated by exact (text, label) match."""
    path = RAW_DIR / "kaggle_dialectal" / "dialectal_arabic_phishing_dataset.csv"
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["text", "label"])

    rows = []
    for _, entry in df.iterrows():
        label = "phishing" if entry["label"].strip().lower() == "phishing" else "legitimate"
        rows.append({
            "text": str(entry["text"]).strip(),
            "content_type": "text",
            "label": label,
            "language": "mixed",  # dialectal Arabic frequently code-switches with English
            "source": "kaggle_dialectal_arabic_sms",
        })
    return rows


def load_github_arabic_spam() -> list[dict]:
    """Source 4: one message per .txt file, split across spam/ and 'non spam/' folders."""
    folder = RAW_DIR / "github_arabic_spam"
    rows = []

    for subfolder, label in [("spam", "phishing"), ("non spam", "legitimate")]:
        for txt_file in (folder / subfolder).glob("*.txt"):
            try:
                text = txt_file.read_text(encoding="utf-8").strip()
            except UnicodeDecodeError:
                # A handful of files in this dataset are legacy Windows-1256 Arabic encoding
                text = txt_file.read_text(encoding="windows-1256").strip()
            if text:
                rows.append({
                    "text": text,
                    "content_type": "text",
                    "label": label,
                    "language": "ar",
                    "source": "github_arabic_text_spam_detection",
                })
    return rows


def main() -> None:
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_rows = list(reader)

    existing_man_count = sum(1 for row in existing_rows if row["id"].startswith("MAN-"))
    next_id_num = existing_man_count + 1

    loaders = [
        ("Salloum/Gaber bilingual corpus", load_salloum_gaber_corpus),
        ("Kaggle Fully Dataset", load_kaggle_fully_dataset),
        ("Kaggle dialectal Arabic SMS", load_kaggle_dialectal),
        ("GitHub Arabic-Text-Spam-Detection", load_github_arabic_spam),
    ]

    all_new_rows = []
    for name, loader in loaders:
        source_rows = loader()
        print(f"{name}: {len(source_rows)} rows")
        all_new_rows.extend(source_rows)

    final_rows = []
    for i, row in enumerate(all_new_rows, start=next_id_num):
        final_rows.append({
            "id": f"MAN-{i:06d}",
            "text": row["text"],
            "content_type": row["content_type"],
            "label": row["label"],
            "language": row["language"],
            "source": row["source"],
            "brand_impersonated": "",
            "date_collected": TODAY,
            "split": "",
        })

    all_rows = existing_rows + final_rows
    with open(DATASET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DATASET_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    print()
    print(f"Total new Arabic rows added: {len(final_rows)}")
    print(f"dataset.csv now has {len(all_rows):,} total rows")


if __name__ == "__main__":
    main()
