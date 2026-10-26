"""
fetch_threat_feeds.py

Pulls confirmed phishing URLs from PhishTank and OpenPhish's free community
feeds, saves the untouched raw responses, and appends clean rows to the
unified processed dataset in the ETIS schema (see docs/dataset_schema.md).

Usage:
    python fetch_threat_feeds.py

Run from anywhere -- paths are resolved relative to the project root
regardless of your current working directory.
"""

import csv
import io
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

PHISHTANK_URL = "https://data.phishtank.com/data/online-valid.csv"
OPENPHISH_URL = "https://openphish.com/feed.txt"

# PhishTank asks that requests identify the application, per their API docs
REQUEST_HEADERS = {
    "User-Agent": "ETIS-GraduationProject/1.0 (Egyptian phishing detection thesis)"
}

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
    """Raised when a threat feed can't be downloaded or parsed."""


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def fetch_phishtank_rows() -> list[dict]:
    """Download the PhishTank CSV feed and convert rows to the ETIS schema."""
    response = requests.get(PHISHTANK_URL, headers=REQUEST_HEADERS, timeout=30)
    if response.status_code != 200:
        raise FeedFetchError(
            f"PhishTank feed request failed with status {response.status_code}"
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / "phishtank_raw.csv"
    raw_path.write_text(response.text, encoding="utf-8")

    reader = csv.DictReader(io.StringIO(response.text))
    rows = []
    for i, entry in enumerate(reader, start=1):
        rows.append({
            "id": f"PT-{i:06d}",
            "text": entry.get("url", "").strip(),
            "content_type": "url",
            "label": "phishing",
            "language": "en",  # PhishTank URLs are not language-tagged; default en
            "source": "phishtank",
            "brand_impersonated": (entry.get("target") or "").strip(),
            "date_collected": date.today().isoformat(),
            "split": "",
        })
    return rows


def fetch_openphish_rows() -> list[dict]:
    """Download the OpenPhish text feed and convert rows to the ETIS schema."""
    response = requests.get(OPENPHISH_URL, headers=REQUEST_HEADERS, timeout=30)
    if response.status_code != 200:
        raise FeedFetchError(
            f"OpenPhish feed request failed with status {response.status_code}"
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / "openphish_raw.txt"
    raw_path.write_text(response.text, encoding="utf-8")

    urls = [line.strip() for line in response.text.splitlines() if line.strip()]
    rows = []
    for i, url in enumerate(urls, start=1):
        rows.append({
            "id": f"OP-{i:06d}",
            "text": url,
            "content_type": "url",
            "label": "phishing",
            "language": "en",
            "source": "openphish",
            "brand_impersonated": "",  # OpenPhish's free feed has no brand field
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
    print("Fetching PhishTank feed...")
    phishtank_rows = fetch_phishtank_rows()
    print(f"  -> {len(phishtank_rows)} PhishTank URLs saved to data/raw/phishtank_raw.csv")

    print("Fetching OpenPhish feed...")
    openphish_rows = fetch_openphish_rows()
    print(f"  -> {len(openphish_rows)} OpenPhish URLs saved to data/raw/openphish_raw.txt")

    all_rows = phishtank_rows + openphish_rows
    append_to_dataset(all_rows)
    print(f"Appended {len(all_rows)} total rows to data/processed/dataset.csv")


if __name__ == "__main__":
    main()
