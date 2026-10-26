"""
create_stratified_split.py

Fills in the 'split' column of data/processed/dataset.csv with train/val/test
labels, using a 70/15/15 stratified split. Stratification is done on
(label, language) pairs, so the phishing/legitimate ratio AND the
ar/en/mixed language ratio are both preserved across train, val, and test --
not just the overall label balance.

Uses a fixed random seed (42) for reproducibility, matching standard practice
and worth stating as such in Chapter 3.4 (Dataset Specifications).

Usage:
    python create_stratified_split.py
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # scripts/ -> data/ -> root
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATASET_PATH = PROCESSED_DIR / "dataset.csv"

DATASET_COLUMNS = [
    "id", "text", "content_type", "label", "language",
    "source", "brand_impersonated", "date_collected", "split",
]

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO is whatever remains (0.15)


def assign_splits_for_group(group_rows: list[dict], rng: random.Random) -> None:
    """Shuffle a group in place and assign split labels directly onto the row dicts."""
    shuffled = group_rows[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = round(n * TRAIN_RATIO)
    val_end = train_end + round(n * VAL_RATIO)

    for row in shuffled[:train_end]:
        row["split"] = "train"
    for row in shuffled[train_end:val_end]:
        row["split"] = "val"
    for row in shuffled[val_end:]:
        row["split"] = "test"


def main() -> None:
    rng = random.Random(RANDOM_SEED)

    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    groups = defaultdict(list)
    for row in rows:
        key = (row["label"], row["language"])
        groups[key].append(row)

    small_group_warnings = []
    for key, group_rows in groups.items():
        if len(group_rows) < 20:
            small_group_warnings.append((key, len(group_rows)))
        assign_splits_for_group(group_rows, rng)

    with open(DATASET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DATASET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    split_counts = defaultdict(int)
    for row in rows:
        split_counts[row["split"]] += 1

    print(f"Stratified {len(rows):,} rows across {len(groups)} (label, language) groups.")
    print()
    print("Overall split sizes:")
    for split_name in ["train", "val", "test"]:
        count = split_counts[split_name]
        pct = 100 * count / len(rows)
        print(f"  {split_name}: {count:,} ({pct:.1f}%)")

    if small_group_warnings:
        print()
        print("Note: these (label, language) groups had fewer than 20 rows --")
        print("split ratios may be uneven for these due to rounding on small numbers:")
        for (label, language), count in small_group_warnings:
            print(f"  ({label}, {language}): {count} rows")


if __name__ == "__main__":
    main()
