"""
build_vt_honest_sample.py   (Task B, step 1 of 3)

Draws a FRESH stratified sample (100 phishing + 100 legitimate URLs, seed 42)
from the CURRENT test split for the ETIS-vs-VirusTotal comparison.

Why a fresh sample: the original 200-URL sample was drawn from the OLD
12,702-row test split. Most of those URLs now sit in the new TRAIN split, so
the URL model has seen them and the etis_eval blacklist contains them;
scoring them again would reproduce a leaky "ETIS 100%".

What this script enforces
  * split == "test" in data/processed/dataset.csv, content_type == "url",
    label in {phishing, legitimate};
  * rows tagged as shared-platform in data/processed/split_groups.csv are
    EXCLUDED (their split is assigned row-wise, so host-unseenness is not
    guaranteed; report shared platforms separately, never here);
  * any test URL whose host also occurs in a TRAIN row is dropped (belt and
    braces; the grouped split should already guarantee this);
  * at most ONE URL per host, so a handful of hosts cannot dominate a class;
  * deterministic (seed 42, ids sorted before shuffling).

Output: evaluation/vt_honest_sample.csv  (id, url, label, host)

It also prints what differs between the classes (share with a path, .eg share,
source mix), because a comparison on these URLs inherits the dataset's
construction: legitimate URLs are Tranco top-20k homepages plus crawled pages.

Usage (project root, venv active):
    python evaluation/build_vt_honest_sample.py
"""

import csv
import random
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"
GROUPS_PATH = PROJECT_ROOT / "data" / "processed" / "split_groups.csv"
OUT_PATH = PROJECT_ROOT / "evaluation" / "vt_honest_sample.csv"
SEED = 42
N_PER_CLASS = 100


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


def has_path(url: str) -> bool:
    try:
        parts = urlsplit(url if "://" in url else "http://" + url)
    except ValueError:
        return False
    return len(parts.path.strip("/")) > 0 or bool(parts.query)


def main() -> None:
    for path in (DATASET_PATH, GROUPS_PATH):
        if not path.exists():
            sys.exit(f"{path} not found.")

    group_type_by_id: dict[str, str] = {}
    with open(GROUPS_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            group_type_by_id[row["id"]] = row["group_type"]
    type_counts = Counter(group_type_by_id.values())
    print("split_groups.csv group_type values:", dict(type_counts))
    shared_types = {t for t in type_counts if "shared" in t.lower()}
    if not shared_types:
        sys.exit("No group_type containing 'shared' found; I cannot exclude shared-platform rows. "
                 "Tell me the real value from the line above and I will adjust the script.")
    print("treating as shared-platform:", shared_types)

    train_hosts: set[str] = set()
    candidates: dict[str, list[dict]] = {"phishing": [], "legitimate": []}
    dropped = Counter()
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["content_type"] != "url":
                continue
            host = host_of(row["text"])
            if row["split"] == "train":
                train_hosts.add(host)
            elif row["split"] == "test" and row["label"] in candidates:
                if group_type_by_id.get(row["id"], "") in shared_types:
                    dropped["shared_platform"] += 1
                    continue
                candidates[row["label"]].append({"id": row["id"], "url": row["text"], "label": row["label"],
                                                 "host": host, "source": row.get("source", "")})

    sample: list[dict] = []
    for label, rows in candidates.items():
        kept = []
        for r in rows:
            if r["host"] in train_hosts:
                dropped["host_also_in_train"] += 1
            else:
                kept.append(r)
        kept.sort(key=lambda r: r["id"])
        random.Random(f"{SEED}|{label}").shuffle(kept)
        chosen, seen_hosts = [], set()
        for r in kept:
            if r["host"] and r["host"] not in seen_hosts:
                seen_hosts.add(r["host"])
                chosen.append(r)
            if len(chosen) == N_PER_CLASS:
                break
        if len(chosen) < N_PER_CLASS:
            sys.exit(f"Only {len(chosen)} distinct-host {label} URLs available (needed {N_PER_CLASS}).")
        print(f"{label}: {len(rows):,} test candidates -> {len(kept):,} unseen-host -> sampled {len(chosen)}")
        sample += chosen

    print("dropped:", dict(dropped))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "url", "label", "host"])
        writer.writeheader()
        writer.writerows({k: r[k] for k in ("id", "url", "label", "host")} for r in sample)

    print(f"\nWrote {len(sample)} rows to {OUT_PATH}")
    print("Class differences that any comparison on this sample inherits:")
    for label in ("phishing", "legitimate"):
        part = [r for r in sample if r["label"] == label]
        with_path = sum(has_path(r["url"]) for r in part)
        eg = sum(r["host"].endswith(".eg") for r in part)
        print(f"  {label:<11} with a path/query: {with_path}/{len(part)}   .eg hosts: {eg}/{len(part)}   "
              f"sources: {dict(Counter(r['source'] for r in part))}")


if __name__ == "__main__":
    main()
