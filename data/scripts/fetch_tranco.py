"""
fetch_tranco.py

Pulls the top 20,000 domains from the Tranco list (a research-grade,
manipulation-resistant ranking of popular websites), saves the raw list,
and appends clean "legitimate" rows to the unified processed dataset in
the ETIS schema (see docs/dataset_schema.md).

Tranco's permanent download URL always points to the current daily list --
no API key or registration required.

Usage:
    python fetch_tranco.py
"""

import csv
import io
import zipfile
from datetime import date
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # scripts/ -> data/ -> root
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATASET_PATH = PROCESSED_DIR / "dataset.csv"

TRANCO_ZIP_URL = "https://tranco-list.eu/top-1m.csv.zip"
DOMAIN_LIMIT = 20_000

DATASET_COLUMNS = [
    "id",
    "text",
    "content_type",
    "label",
    "language",
    "source",
    "brand_impersonated",
    "date_collected",
    "split",
]


class FeedFetchError(Exception):
    """Raised when the Tranco list can't be downloaded or parsed."""


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

def fetch_tranco_rows(limit: int = DOMAIN_LIMIT) -> list[dict]:
    """Download the Tranco list, save the raw top-N slice, and convert to schema rows."""
    response = requests.get(TRANCO_ZIP_URL, timeout=60)
    if response.status_code != 200:
        raise FeedFetchError(
            f"Tranco list request failed with status {response.status_code}"
        )

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        # The zip contains a single CSV named top-1m.csv with columns: rank,domain
        csv_name = archive.namelist()[0]
        with archive.open(csv_name) as f:
            reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
            all_domains = [row[1].strip() for row in reader if len(row) >= 2]

    top_domains = all_domains[:limit]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / "tranco_raw.csv"
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "domain"])
        for i, domain in enumerate(top_domains, start=1):
            writer.writerow([i, domain])

    rows = []
    for i, domain in enumerate(top_domains, start=1):
        rows.append({
            "id": f"TR-{i:06d}",
            "text": f"https://{domain}",
            "content_type": "url",
            "label": "legitimate",
            "language": "en",  # Tranco domains skew global/English; not language-tagged
            "source": "tranco",
            "brand_impersonated": "",
            "date_collected": date.today().isoformat(),
            "split": "",
        })
    return rows


# ---------------------------------------------------------------------------
# Dataset writer
# ---------------------------------------------------------------------------

def append_to_dataset(rows: list[dict]) -> None:
    """Append rows to data/processed/dataset.csv, creating it if needed."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = DATASET_PATH.exists()

    with open(DATASET_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DATASET_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Fetching Tranco top {DOMAIN_LIMIT:,} domains...")
    rows = fetch_tranco_rows()
    print(f"  -> {len(rows)} domains saved to data/raw/tranco_raw.csv")

    append_to_dataset(rows)
    print(f"Appended {len(rows)} legitimate rows to data/processed/dataset.csv")


if __name__ == "__main__":
    main()
