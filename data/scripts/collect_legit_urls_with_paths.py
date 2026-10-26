"""
collect_legit_urls_with_paths.py

Fixes a measured flaw in the URL dataset: every legitimate URL is a bare
Tranco homepage ("https://example.com", 0% carry a path) while 66% of phishing
URLs carry one. A model can reach ~0.999 AUC by learning "has a path =
phishing" -- which would flag every ordinary page on a real site. See
ml/results/url_model_comparison.txt (full-URL vs host-only models).

This script builds REAL legitimate URLs with paths and from less-famous hosts:

  1. Picks domains from the Tranco top-1M list:
       - LONGTAIL_COUNT random domains ranked 20,001-200,000 (not famous)
       - up to EGYPT_COUNT random .eg domains from anywhere in the list
         (keeps the thesis focus on Egyptian sites)
  2. Fetches each domain's homepage ONCE (identifying User-Agent, short
     timeout, no logins, no crawling beyond the homepage) and reads its
     <a href> links.
  3. Keeps up to MAX_PER_DOMAIN internal links that have a real path, skipping
     assets (images, scripts, archives, PDFs) and adult-content domains/URLs.
  4. Writes data/raw/legit_urls_with_paths.csv (rank, domain, url).
  5. With --append, also adds the rows to data/processed/dataset.csv with ids
     LP-000001.., source "crawl_legit_paths", label legitimate, empty split.

After --append, re-run in this order (all idempotent):
     python data/scripts/clean_dataset.py
     python data/scripts/create_grouped_split.py
     (then re-seed the evaluation blacklist: seed_database.py --only-split train)

Cite in Chapter 3.3: public homepages fetched once each on <date>, links read
from the HTML, no authentication, no personal data, one request per site.

Usage:
    python data/scripts/collect_legit_urls_with_paths.py
    python data/scripts/collect_legit_urls_with_paths.py --append
    python data/scripts/collect_legit_urls_with_paths.py --domains-csv my_domains.csv   # rank,domain (for testing)
"""

import argparse
import csv
import io
import random
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "legit_urls_with_paths.csv"
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"

TRANCO_ZIP_URL = "https://tranco-list.eu/top-1m.csv.zip"
HEADERS = {"User-Agent": "ETIS-GraduationProject/1.0 (Egyptian phishing detection thesis; contact: student project)"}

SEED = 42
LONGTAIL_RANK_RANGE = (20_001, 200_000)
LONGTAIL_COUNT = 1_800
EGYPT_COUNT = 700
MAX_PER_DOMAIN = 8
MAX_URL_LENGTH = 200
TIMEOUT_SECONDS = 6
WORKERS = 20

ASSET_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico", ".css", ".js", ".json", ".xml",
    ".pdf", ".zip", ".rar", ".exe", ".apk", ".mp3", ".mp4", ".avi", ".doc", ".docx", ".xls", ".xlsx",
)
ADULT_SUBSTRINGS = [  # same intent as review_tranco.py
    "porn", "xxx", "xnxx", "xvideos", "sex", "adult", "gambl", "casino", "escort", "chaturbate",
    "onlyfans", "hentai", "redtube", "youporn", "brazzers", "bdsm",
]
ADULT_SAFE_WORDS = ["essex", "sussex", "middlesex", "wessex", "unisex", "intersex"]

DATASET_COLUMNS = [
    "id", "text", "content_type", "label", "language", "source",
    "brand_impersonated", "date_collected", "split", "label_detail",
]


class LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value:
                    self.hrefs.append(value.strip())


def looks_adult(text: str) -> bool:
    lowered = text.lower()
    if any(safe in lowered for safe in ADULT_SAFE_WORDS):
        return False
    return any(sub in lowered for sub in ADULT_SUBSTRINGS)


def same_site(host: str, domain: str) -> bool:
    host, domain = host.lower().removeprefix("www."), domain.lower().removeprefix("www.")
    return host == domain or host.endswith("." + domain)


def candidate_links(base_url: str, domain: str, html: str) -> list[str]:
    parser = LinkExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    seen, links = set(), []
    for href in parser.hrefs:
        if href.startswith(("javascript:", "mailto:", "tel:", "#", "data:")):
            continue
        full = urljoin(base_url, href).split("#")[0]
        parts = urlsplit(full)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            continue
        if not same_site(parts.hostname, domain):
            continue
        if len(parts.path) <= 1:  # homepage itself, or query-only
            continue
        if parts.path.lower().endswith(ASSET_EXTENSIONS):
            continue
        if len(full) > MAX_URL_LENGTH or looks_adult(full):
            continue
        if full not in seen:
            seen.add(full)
            links.append(full)
    return links


def fetch_links(rank: int, domain: str) -> list[tuple[int, str, str]]:
    if looks_adult(domain):
        return []
    for scheme in ("https", "http"):
        try:
            response = requests.get(f"{scheme}://{domain}", headers=HEADERS, timeout=TIMEOUT_SECONDS)
        except requests.RequestException:
            continue
        if response.status_code != 200 or "html" not in response.headers.get("Content-Type", "").lower():
            continue
        links = candidate_links(response.url, domain, response.text)
        random.Random(f"{SEED}-{domain}").shuffle(links)
        return [(rank, domain, u) for u in links[:MAX_PER_DOMAIN]]
    return []


def load_tranco_domains() -> list[tuple[int, str]]:
    print("Downloading Tranco top-1M list...")
    response = requests.get(TRANCO_ZIP_URL, timeout=120)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        with archive.open(archive.namelist()[0]) as f:
            reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
            return [(int(row[0]), row[1].strip()) for row in reader if len(row) >= 2]


def pick_domains(all_domains: list[tuple[int, str]]) -> list[tuple[int, str]]:
    rng = random.Random(SEED)
    lo, hi = LONGTAIL_RANK_RANGE
    longtail = [(r, d) for r, d in all_domains if lo <= r <= hi and not d.endswith(".eg")]
    egypt = [(r, d) for r, d in all_domains if d.endswith(".eg")]
    chosen = rng.sample(longtail, min(LONGTAIL_COUNT, len(longtail))) + rng.sample(egypt, min(EGYPT_COUNT, len(egypt)))
    return sorted(chosen)


def append_to_dataset(found: list[tuple[int, str, str]]) -> None:
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        existing_rows = list(csv.DictReader(f))
    existing_texts = {r["text"] for r in existing_rows}
    next_num = 1 + sum(1 for r in existing_rows if r["id"].startswith("LP-"))

    new_rows = []
    for _, domain, url in found:
        if url in existing_texts:
            continue
        existing_texts.add(url)
        new_rows.append({
            "id": f"LP-{next_num:06d}", "text": url, "content_type": "url", "label": "legitimate",
            "language": "mixed" if domain.endswith(".eg") else "en", "source": "crawl_legit_paths",
            "brand_impersonated": "", "date_collected": date.today().isoformat(), "split": "",
            "label_detail": "legitimate",
        })
        next_num += 1

    with open(DATASET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DATASET_COLUMNS)
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerows(new_rows)
    print(f"Appended {len(new_rows):,} rows to dataset.csv (now {len(existing_rows) + len(new_rows):,}).")
    print("Next: python data/scripts/clean_dataset.py ; python data/scripts/create_grouped_split.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--append", action="store_true", help="append the collected URLs to dataset.csv")
    parser.add_argument("--domains-csv", help="CSV with columns rank,domain instead of downloading Tranco")
    args = parser.parse_args()

    if args.domains_csv:
        with open(args.domains_csv, newline="", encoding="utf-8") as f:
            domains = [(int(r["rank"]), r["domain"]) for r in csv.DictReader(f)]
    else:
        domains = pick_domains(load_tranco_domains())
    print(f"Fetching {len(domains):,} homepages ({WORKERS} workers)...")

    found: list[tuple[int, str, str]] = []
    ok_domains = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(fetch_links, r, d) for r, d in domains]
        for i, fut in enumerate(as_completed(futures), start=1):
            result = fut.result()
            if result:
                ok_domains += 1
                found.extend(result)
            if i % 200 == 0:
                print(f"  ...{i}/{len(domains)} domains, {len(found):,} URLs so far")

    found.sort()
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RAW_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "domain", "url"])
        writer.writerows(found)

    egypt = sum(1 for _, d, _ in found if d.endswith(".eg"))
    print(f"\n{len(found):,} URLs from {ok_domains:,} domains ({egypt:,} URLs from .eg domains)")
    print(f"Saved to {RAW_PATH}")

    if args.append:
        append_to_dataset(found)
    else:
        print("Re-run with --append to add them to dataset.csv.")


if __name__ == "__main__":
    sys.exit(main())
