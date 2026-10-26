"""
create_grouped_split.py

Step 3a of the evaluation rebuild. Replaces create_stratified_split.py.

The old split assigned rows to train/val/test at random, so the same host
(URLs) and near-identical messages (texts) landed on both sides of the split.
That made test results optimistic. This script splits by GROUP instead:

  URL rows   group = the real host (normalize_domain). Every URL on one host
             goes to the same split, so test URLs come from hosts the model
             and blacklist never saw. Hosts with more than MAX_HOST_GROUP
             rows are shared platforms (docs.google.com, netlify.app, ...):
             one split for all of them would unbalance everything, and "unseen
             host" is meaningless for a platform, so those rows are assigned
             one by one and tagged group_type = shared_platform.
  Text rows  group = a cluster of near-duplicate messages (character 3-5-gram
             TF-IDF cosine >= TEXT_SIM_THRESHOLD, linked transitively), so
             template-style messages never straddle train/test. A cluster
             larger than MAX_TEXT_GROUP is a whole template FAMILY (e.g. the
             translated 419 letters): keeping it in one split would starve
             val/test of that class, so its rows are assigned one by one and
             tagged group_type = text_template_family -- report them as a
             separate group, like shared platforms.
  Synthetic  the 54 SYN- rows (source synthetic_egyptian_brands) are only ever
             put in train (70%) or val (30%) -- NEVER test -- so they can be
             reported separately and cannot flatter the test numbers.

Stratification: groups are assigned to the split with the largest remaining
deficit, separately for each stratum -- URLs by label; texts by (label,
language) -- targeting 70/15/15 within every stratum.

URL, text and synthetic assignment each use their own seeded random
generator, so adding more text rows later (e.g. the real collected messages)
does NOT change any URL's split. That matters because the training-only
blacklist is built from the URL train split.

Run AFTER clean_dataset.py. Re-runnable: same data + same seed = same split.

Outputs:
    data/processed/dataset.csv          split column rewritten
    data/processed/split_groups.csv     id, group_id, group_type (sidecar)
    data/processed/split_report.txt     balance + leakage before/after
    data/processed/dataset.pre_split.csv  backup of the first run's input

Usage (from the project root):
    python data/scripts/create_grouped_split.py --dry-run
    python data/scripts/create_grouped_split.py
"""

import argparse
import csv
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # scripts/ -> data/ -> root
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from clean_dataset import DATASET_COLUMNS, SYNTHETIC_SOURCE, normalise_text_key  # noqa: E402
from app.engines.rule_based.domain_typosquatting_detector import normalize_domain  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATASET_PATH = PROCESSED_DIR / "dataset.csv"
BACKUP_PATH = PROCESSED_DIR / "dataset.pre_split.csv"
GROUPS_PATH = PROCESSED_DIR / "split_groups.csv"
REPORT_PATH = PROCESSED_DIR / "split_report.txt"

SEED = 42
SPLITS = ("train", "val", "test")
RATIOS = (0.70, 0.15, 0.15)
SYNTHETIC_RATIOS = (0.70, 0.30, 0.0)  # never test

MAX_HOST_GROUP = 50          # hosts with more rows than this are treated as shared platforms
TEXT_SIM_THRESHOLD = 0.90    # char n-gram TF-IDF cosine for "near-duplicate"
MAX_TEXT_GROUP = 25          # larger clusters are template families (see build_text_groups)


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def build_url_groups(url_rows: list[dict]) -> tuple[dict[str, list[dict]], dict[str, str]]:
    """Returns (group_id -> rows, group_id -> group_type)."""
    by_host: dict[str, list[dict]] = defaultdict(list)
    for row in url_rows:
        host = normalize_domain(row["text"]) or f"__nohost__{row['id']}"
        by_host[host].append(row)

    groups: dict[str, list[dict]] = {}
    types: dict[str, str] = {}
    for host, rows in by_host.items():
        if len(rows) > MAX_HOST_GROUP:
            for row in rows:  # shared platform: one row per group
                gid = f"plat:{row['id']}"
                groups[gid] = [row]
                types[gid] = "shared_platform"
        else:
            gid = f"host:{host}"
            groups[gid] = rows
            types[gid] = "host"
    return groups, types


def build_text_groups(text_rows: list[dict]) -> tuple[dict[str, list[dict]], dict[str, str]]:
    n = len(text_rows)
    uf = UnionFind(n)
    if n > 1:
        keys = [normalise_text_key(r["text"]) or r["id"] for r in text_rows]
        vectoriser = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        matrix = vectoriser.fit_transform(keys)
        similarity = (matrix @ matrix.T).toarray()
        rows_idx, cols_idx = np.where(np.triu(similarity >= TEXT_SIM_THRESHOLD, k=1))
        for a, b in zip(rows_idx.tolist(), cols_idx.tolist()):
            uf.union(a, b)

    clusters: dict[int, list[dict]] = defaultdict(list)
    for i, row in enumerate(text_rows):
        clusters[uf.find(i)].append(row)

    groups: dict[str, list[dict]] = {}
    types: dict[str, str] = {}
    for members in clusters.values():
        if len(members) > MAX_TEXT_GROUP:
            for m in members:
                gid = f"fam:{m['id']}"
                groups[gid] = [m]
                types[gid] = "text_template_family"
            continue
        gid = "txt:" + min(m["id"] for m in members)
        groups[gid] = members
        types[gid] = "text_cluster" if len(members) > 1 else "text_single"
    return groups, types


def majority(values: list[str]) -> str:
    counts = Counter(values)
    top = max(counts.values())
    return sorted(v for v, c in counts.items() if c == top)[0]


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------

def assign_groups(groups: dict[str, list[dict]], stratum_of: dict[str, tuple],
                  ratios: tuple, rng: random.Random) -> dict[str, str]:
    """Greedy: within each stratum, biggest groups first, each to the split
    with the largest RELATIVE remaining deficit, (target - filled) / target.

    Relative, not absolute: with absolute deficits every big group goes to
    train until train is more than half full, so val/test would contain only
    one-row groups and look nothing like train (found 2026-10-13: all 8,000
    crawled legitimate URLs with paths landed in train)."""
    by_stratum: dict[tuple, list[str]] = defaultdict(list)
    for gid in groups:
        by_stratum[stratum_of[gid]].append(gid)

    assignment: dict[str, str] = {}
    active = [(s, r) for s, r in zip(SPLITS, ratios) if r > 0]
    for stratum in sorted(by_stratum):
        gids = sorted(by_stratum[stratum])
        rng.shuffle(gids)
        gids.sort(key=lambda g: len(groups[g]), reverse=True)  # stable: ties keep shuffled order
        total = sum(len(groups[g]) for g in gids)
        target = {s: total * r for s, r in active}
        filled = {s: 0 for s, _ in active}
        ratio_of = dict(active)
        for gid in gids:
            best = max((s for s, _ in active),
                       key=lambda s: ((target[s] - filled[s]) / target[s], ratio_of[s]))
            assignment[gid] = best
            filled[best] += len(groups[gid])
    return assignment


# ---------------------------------------------------------------------------
# Leakage metrics
# ---------------------------------------------------------------------------

def host_leakage(url_rows: list[dict], split_of: dict[str, str], host_of: dict[str, str]) -> tuple[int, int]:
    """(test phishing URL rows whose host also occurs in train, total test phishing URL rows)."""
    train_hosts = {host_of[r["id"]] for r in url_rows if split_of[r["id"]] == "train"}
    test_phish = [r for r in url_rows if split_of[r["id"]] == "test" and r["label"] == "phishing"]
    shared = sum(1 for r in test_phish if host_of[r["id"]] in train_hosts)
    return shared, len(test_phish)


def text_cluster_leaks(groups: dict[str, list[dict]], split_of: dict[str, str]) -> int:
    return sum(1 for rows in groups.values()
               if len(rows) > 1 and len({split_of[r["id"]] for r in rows}) > 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = parser.parse_args()

    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "label_detail" not in (reader.fieldnames or []):
            sys.exit("dataset.csv has no label_detail column -- run clean_dataset.py first.")
        rows = list(reader)

    old_split = {r["id"]: r["split"] for r in rows}

    synthetic = [r for r in rows if r["source"] == SYNTHETIC_SOURCE]
    url_rows = [r for r in rows if r["source"] != SYNTHETIC_SOURCE and r["content_type"] == "url"]
    text_rows = [r for r in rows if r["source"] != SYNTHETIC_SOURCE and r["content_type"] != "url"]

    host_of = {r["id"]: normalize_domain(r["text"]) for r in url_rows}

    # --- URLs --------------------------------------------------------------
    url_groups, url_types = build_url_groups(url_rows)
    url_strata = {gid: ("url", majority([r["label"] for r in rs])) for gid, rs in url_groups.items()}
    url_assign = assign_groups(url_groups, url_strata, RATIOS, random.Random(f"{SEED}-url"))

    # --- Texts -------------------------------------------------------------
    text_groups, text_types = build_text_groups(text_rows)
    text_strata = {
        gid: ("text", majority([r["label"] for r in rs]), majority([r["language"] for r in rs]))
        for gid, rs in text_groups.items()
    }
    text_assign = assign_groups(text_groups, text_strata, RATIOS, random.Random(f"{SEED}-text"))

    # --- Synthetic: train/val only ----------------------------------------
    syn_groups = {f"syn:{r['id']}": [r] for r in synthetic}
    syn_types = {gid: "synthetic" for gid in syn_groups}
    syn_strata = {gid: ("syn", rs[0]["label"]) for gid, rs in syn_groups.items()}
    syn_assign = assign_groups(syn_groups, syn_strata, SYNTHETIC_RATIOS, random.Random(f"{SEED}-syn"))

    # --- Apply -------------------------------------------------------------
    new_split: dict[str, str] = {}
    sidecar: list[dict] = []
    for groups, assign, types in [
        (url_groups, url_assign, url_types),
        (text_groups, text_assign, text_types),
        (syn_groups, syn_assign, syn_types),
    ]:
        for gid, members in groups.items():
            for r in members:
                new_split[r["id"]] = assign[gid]
                sidecar.append({"id": r["id"], "group_id": gid, "group_type": types[gid]})

    for r in rows:
        r["split"] = new_split[r["id"]]

    # --- Report ------------------------------------------------------------
    total = len(rows)
    lines = ["ETIS grouped split report" + ("  [DRY RUN]" if args.dry_run else ""),
             f"seed={SEED}  ratios={RATIOS}  MAX_HOST_GROUP={MAX_HOST_GROUP}  "
             f"TEXT_SIM_THRESHOLD={TEXT_SIM_THRESHOLD}", ""]

    lines.append("Overall:")
    for s in SPLITS:
        n = sum(1 for r in rows if r["split"] == s)
        lines.append(f"  {s:<5} {n:>7,}  ({100 * n / total:.1f}%)")

    lines += ["", "By content kind x label_detail (train / val / test):"]
    table: dict[tuple, Counter] = defaultdict(Counter)
    for r in rows:
        kind = "synthetic" if r["source"] == SYNTHETIC_SOURCE else ("url" if r["content_type"] == "url" else "text")
        table[(kind, r["label_detail"])][r["split"]] += 1
    for (kind, detail), c in sorted(table.items()):
        lines.append(f"  {kind:<10}{detail:<12}{c['train']:>8,}{c['val']:>8,}{c['test']:>8,}")

    lines += ["", "URL group types (rows):"]
    for t, n in Counter(url_types[g] for g, rs in url_groups.items() for _ in rs).most_common():
        lines.append(f"  {t}: {n:,}")
    host_groups = [rs for g, rs in url_groups.items() if url_types[g] == "host"]
    lines.append(f"  distinct host groups: {len(host_groups):,}")

    old_shared, old_total = host_leakage(url_rows, old_split, host_of)
    new_shared, new_total = host_leakage(url_rows, new_split, host_of)
    plat_ids = {r["id"] for g, rs in url_groups.items() if url_types[g] == "shared_platform" for r in rs}
    new_shared_nonplat = sum(
        1 for r in url_rows
        if new_split[r["id"]] == "test" and r["label"] == "phishing" and r["id"] not in plat_ids
        and host_of[r["id"]] in {host_of[x["id"]] for x in url_rows if new_split[x["id"]] == "train"}
    )
    lines += [
        "",
        "Leakage -- test phishing URLs whose host also appears in train:",
        f"  BEFORE (old split): {old_shared:,} of {old_total:,}"
        + (f"  ({100 * old_shared / old_total:.1f}%)" if old_total else ""),
        f"  AFTER  (new split): {new_shared:,} of {new_total:,}"
        + (f"  ({100 * new_shared / new_total:.1f}%)" if new_total else ""),
        f"    of which on non-platform hosts (must be 0): {new_shared_nonplat:,}",
        "    (the remainder are shared-platform rows, which share a host by construction;",
        "     report them as a separate group in Chapter 5)",
    ]

    old_text_leaks = text_cluster_leaks(text_groups, old_split)
    new_text_leaks = text_cluster_leaks(text_groups, new_split)
    type_rows = Counter(text_types[g] for g, rs in text_groups.items() for _ in rs)
    sizes = Counter(len(rs) for rs in text_groups.values() if len(rs) > 1)
    lines += [
        "",
        "Text group types (rows): "
        + ", ".join(f"{t}={n:,}" for t, n in type_rows.most_common()),
        f"  near-duplicate clusters kept together (2..{MAX_TEXT_GROUP} rows): "
        f"{sum(sizes.values()):,}",
        f"  clusters spanning >1 split BEFORE: {old_text_leaks:,}   AFTER (must be 0): {new_text_leaks:,}",
        "  cluster size distribution (size: count): "
        + (", ".join(f"{k}:{v}" for k, v in sorted(sizes.items())) or "none"),
        "",
        f"Synthetic rows in test (must be 0): {sum(1 for r in synthetic if new_split[r['id']] == 'test')}",
    ]

    report = "\n".join(lines)
    print(report)

    if args.dry_run:
        return

    if not BACKUP_PATH.exists():
        shutil.copy2(DATASET_PATH, BACKUP_PATH)
        print(f"\nBackup written: {BACKUP_PATH}")

    with open(DATASET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DATASET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    with open(GROUPS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "group_id", "group_type"])
        writer.writeheader()
        writer.writerows(sorted(sidecar, key=lambda d: d["id"]))

    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    print(f"\nWrote {DATASET_PATH.name}, {GROUPS_PATH.name}, {REPORT_PATH.name}")


if __name__ == "__main__":
    main()
