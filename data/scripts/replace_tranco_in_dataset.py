"""
replace_tranco_in_dataset.py

Removes the old, unfiltered TR- rows from data/processed/dataset.csv and
replaces them with fresh rows built from the reviewed data/raw/tranco_final_approved.csv
(see review_tranco.py). Run this once, after review_tranco.py has produced
tranco_final_approved.csv.

Usage:
    python replace_tranco_in_dataset.py
"""

import csv
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # scripts/ -> data/ -> root
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATASET_PATH = PROCESSED_DIR / "dataset.csv"
APPROVED_PATH = RAW_DIR / "tranco_final_approved.csv"

DATASET_COLUMNS = [
    "id", "text", "content_type", "label", "language",
    "source", "brand_impersonated", "date_collected", "split",
]


def main() -> None:
    # 1. Load the existing dataset and drop old TR- rows
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_rows = [row for row in reader if not row["id"].startswith("TR-")]

    removed_count = 0
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        removed_count = sum(1 for row in reader if row["id"].startswith("TR-"))

    print(f"Removed {removed_count:,} old TR- rows from dataset.csv")

    # 2. Load the approved, filtered Tranco domains
    with open(APPROVED_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        approved_domains = list(reader)

    # 3. Build fresh TR- rows
    new_rows = []
    for i, entry in enumerate(approved_domains, start=1):
        new_rows.append({
            "id": f"TR-{i:06d}",
            "text": f"https://{entry['domain']}",
            "content_type": "url",
            "label": "legitimate",
            "language": "en",
            "source": "tranco",
            "brand_impersonated": "",
            "date_collected": date.today().isoformat(),
            "split": "",
        })

    print(f"Adding {len(new_rows):,} reviewed TR- rows")

    # 4. Write everything back
    all_rows = existing_rows + new_rows
    with open(DATASET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DATASET_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"dataset.csv now has {len(all_rows):,} total rows")


if __name__ == "__main__":
    main()
