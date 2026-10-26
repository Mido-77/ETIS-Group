"""
build_tranco_eg_sample.py   (false-positive measurement on real Egyptian sites, step 1 of 2)

Builds an EXTERNAL sample of real, popular-to-obscure Egyptian domains from a
Tranco list the user downloaded (tranco-list.eu, top 1M, CSV "rank,domain"),
for measuring the false-positive rate of the final system on sites the model
has never seen.

What it enforces
  * only domains ending in ".eg";
  * a domain is EXCLUDED if its registrable domain occurs anywhere in
    data/processed/dataset.csv (any split, URL rows): the URL model or the
    blacklist may have seen it;
  * brand-registry domains (app/core/egyptian_brands.py) are excluded: the
    allowlist makes them LOW and they would say nothing about the models;
  * sampled per popularity band (Tranco rank): A 1-100,000, B 100,001-500,000,
    C 500,001-1,000,000; up to 100 domains per band, seed 20261024. Bands exist
    because the unmeasured question is "what about obscure sites";
  * two URL forms per domain: apex "https://domain" (PRIMARY: the form the
    dataset's own Tranco rows use) and "https://www.domain" (secondary: the
    URL model reads "www." as slightly safer).

Honest limits (state them in the thesis)
  * A Tranco-listed domain is NOT verified legitimate: flagged URLs must be
    checked by hand (look the URL up on virustotal.com in the browser, do not
    visit suspicious sites).
  * Tranco lists registrable domains only: this sample measures HOMEPAGE-style
    URLs. URLs with paths are measured from the dataset's own .eg test rows.
  * Do not tune thresholds or rules on this sample; if a mitigation follows,
    measure it on a fresh sample (new seed / newer Tranco list).

Output: data/benchmarks/eg_legit_external.csv and eg_legit_external_manifest.json

Usage (project root, venv active):
    python data/scripts/build_tranco_eg_sample.py --tranco data/external/tranco_top1m.csv --list-id <ID shown on tranco-list.eu>
"""

import argparse
import csv
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
from app.core.egyptian_brands import EGYPTIAN_BRAND_DOMAINS  # noqa: E402

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"
OUT_PATH = PROJECT_ROOT / "data" / "benchmarks" / "eg_legit_external.csv"
MANIFEST_PATH = PROJECT_ROOT / "data" / "benchmarks" / "eg_legit_external_manifest.json"
SEED = 20261024
PER_BAND = 100
BANDS = [("A_top_100k", 1, 100_000), ("B_100k_500k", 100_001, 500_000), ("C_500k_1M", 500_001, 1_000_000)]
EG_SECOND_LEVEL = {"com", "gov", "edu", "org", "net", "sci", "eun", "mil", "name", "info", "biz"}


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


def registrable(host: str) -> str:
    labels = host.strip(".").split(".")
    if len(labels) >= 3 and labels[-1] == "eg" and labels[-2] in EG_SECOND_LEVEL:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:]) if len(labels) >= 2 else host


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tranco", required=True, help="path to the downloaded Tranco CSV (rank,domain)")
    parser.add_argument("--list-id", required=True, help="the Tranco list ID/date shown on tranco-list.eu (provenance)")
    args = parser.parse_args()

    tranco_path = Path(args.tranco)
    if not tranco_path.exists():
        sys.exit(f"{tranco_path} not found.")
    if not DATASET_PATH.exists():
        sys.exit(f"{DATASET_PATH} not found.")

    dataset_registrables: set[str] = set()
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["content_type"] == "url":
                host = host_of(row["text"])
                if host.endswith(".eg"):
                    dataset_registrables.add(registrable(host))
    brand_registrables = {registrable(d) for _, d in EGYPTIAN_BRAND_DOMAINS}

    eg_domains: list[tuple[int, str]] = []
    total_rows = 0
    with open(tranco_path, newline="", encoding="utf-8-sig") as f:
        for cells in csv.reader(f):
            if len(cells) < 2 or not cells[0].strip().isdigit():
                continue
            total_rows += 1
            domain = cells[1].strip().lower().rstrip(".")
            if domain.endswith(".eg"):
                eg_domains.append((int(cells[0]), domain))

    excluded = Counter()
    available: dict[str, list[tuple[int, str]]] = {name: [] for name, _, _ in BANDS}
    for rank, domain in eg_domains:
        reg = registrable(domain)
        if reg in dataset_registrables:
            excluded["in_dataset"] += 1
            continue
        if reg in brand_registrables:
            excluded["brand_registry"] += 1
            continue
        for name, low, high in BANDS:
            if low <= rank <= high:
                available[name].append((rank, domain))

    rows = []
    for name, _, _ in BANDS:
        pool = sorted(available[name])
        random.Random(f"{SEED}|{name}").shuffle(pool)
        for rank, domain in pool[:PER_BAND]:
            rows.append({"domain": domain, "rank": rank, "band": name})
    rows.sort(key=lambda r: r["rank"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "url", "domain", "rank", "band", "form"])
        writer.writeheader()
        for n, r in enumerate(rows, start=1):
            writer.writerow({"id": f"EGX-{n:04d}a", "url": f"https://{r['domain']}", "domain": r["domain"], "rank": r["rank"], "band": r["band"], "form": "apex"})
            writer.writerow({"id": f"EGX-{n:04d}w", "url": f"https://www.{r['domain']}", "domain": r["domain"], "rank": r["rank"], "band": r["band"], "form": "www"})

    manifest = {
        "tranco_list_id": args.list_id, "tranco_file": tranco_path.name,
        "tranco_sha256": hashlib.sha256(tranco_path.read_bytes()).hexdigest(),
        "tranco_rows": total_rows, "eg_domains_in_list": len(eg_domains), "excluded": dict(excluded),
        "available_per_band": {k: len(v) for k, v in available.items()},
        "sampled_per_band": dict(Counter(r["band"] for r in rows)), "seed": SEED, "git_head": git_head(),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Tranco rows read: {total_rows:,}; .eg domains in the list: {len(eg_domains):,}")
    print(f"excluded: {dict(excluded)}")
    print(f"available per band: {manifest['available_per_band']}")
    print(f"sampled per band:   {manifest['sampled_per_band']}  (x2 URL forms each)")
    print(f"Wrote {len(rows) * 2} URLs to {OUT_PATH}\nManifest: {MANIFEST_PATH}")
    if any(len(available[n]) < PER_BAND for n, _, _ in BANDS):
        print("NOTE: a band has fewer than 100 available domains; its interval will be wider.")


if __name__ == "__main__":
    main()
