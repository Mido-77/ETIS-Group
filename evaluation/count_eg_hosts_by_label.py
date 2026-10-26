"""
count_eg_hosts_by_label.py

READ-ONLY diagnostic. Counts how many URL rows in data/processed/dataset.csv
have a host ending in ".eg", by label (phishing / legitimate) and split.

Why: the false-positive measurement on real Egyptian sites found ~0% on .eg
domains, while the held-out controls found 7/36 real Egyptian .com sites flagged
by the URL model. If .eg hosts are almost exclusively LEGITIMATE in the training
data, the URL model could have learned ".eg means legitimate" (a TLD shortcut),
which would (a) make the .eg false-positive rate look better than it is and
(b) under-detect real phishing hosted on .eg domains. This script only looks at
the DATA; it does not change anything and does not test the model.

Usage (project root):
    python evaluation/count_eg_hosts_by_label.py
"""

import csv
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"


def host_of(text: str) -> str:
    t = text.strip().replace("\\", "/")
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", t):
        t = "http://" + t
    try:
        host = urlsplit(t).hostname or ""
    except ValueError:
        return ""
    host = host.lower()
    return (host[4:] if host.startswith("www.") else host).rstrip(".")


def main() -> None:
    if not DATASET_PATH.exists():
        sys.exit(f"{DATASET_PATH} not found.")
    rows_by_cell: Counter = Counter()
    domains_by_label: dict[str, set] = {"phishing": set(), "legitimate": set()}
    totals: Counter = Counter()
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["content_type"] != "url":
                continue
            label, split = row["label"], row["split"]
            totals[label] += 1
            host = host_of(row["text"])
            if host.endswith(".eg"):
                rows_by_cell[(label, split)] += 1
                if label in domains_by_label:
                    domains_by_label[label].add(host)

    print("URL rows whose host ends in .eg, by label and split (rows):")
    print(f"  {'label':<12}{'train':>8}{'val':>8}{'test':>8}{'total':>8}   {'of all URL rows with this label':>34}   distinct hosts")
    for label in ("phishing", "legitimate"):
        counts = [rows_by_cell[(label, s)] for s in ("train", "val", "test")]
        total = sum(counts)
        share = 100 * total / totals[label] if totals[label] else 0
        print(f"  {label:<12}{counts[0]:>8}{counts[1]:>8}{counts[2]:>8}{total:>8}   {total:>10,} / {totals[label]:<10,} = {share:5.2f}%   {len(domains_by_label[label])}")
    other = {k: v for k, v in totals.items() if k not in ("phishing", "legitimate")}
    if other:
        print(f"  (other labels not shown: {other})")
    print("\nRead: if phishing .eg rows are ~0 while legitimate .eg rows number in the thousands, a '.eg means legitimate'")
    print("shortcut is possible in the URL model; testing the MODEL for it is a separate step.")


if __name__ == "__main__":
    main()
