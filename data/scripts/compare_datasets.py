"""
compare_datasets.py

Answers one question: when data/processed/dataset.csv shows up as "92,811
lines changed" in git after a deterministic re-run of clean_dataset.py and
create_grouped_split.py, did the DATA change, or only how the file is written?

It compares two versions at three levels:
  1. bytes      size, BOM, CRLF vs bare-LF line endings
  2. structure  header, row count, id order, ids only in one version
  3. content    per-column differences for the ids both versions share, with
                text differences classed as "whitespace/line-ending only" or
                "real"; and a split-change table (kind x old split -> new split)

Each argument is either a file path or a git ref. A ref is read with
`git show <ref>:data/processed/dataset.csv` as raw bytes (no shell
re-encoding, which would corrupt Arabic text in PowerShell).

Message text is NEVER printed: only ids, counts and lengths.

Usage (from the project root):
    python data/scripts/compare_datasets.py HEAD "stash@{0}"
    python data/scripts/compare_datasets.py old.csv data/processed/dataset.csv
"""

import csv
import io
import subprocess
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REL_PATH = "data/processed/dataset.csv"
MAX_IDS_SHOWN = 10


def load_bytes(arg: str) -> bytes:
    path = Path(arg)
    if path.exists():
        return path.read_bytes()
    result = subprocess.run(["git", "show", f"{arg}:{REL_PATH}"], capture_output=True, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        sys.exit(f"'{arg}' is neither a file nor a git ref containing {REL_PATH}:\n"
                 f"{result.stderr.decode('utf-8', 'replace').strip()}")
    return result.stdout


def byte_facts(raw: bytes) -> dict:
    crlf = raw.count(b"\r\n")
    return {"bytes": len(raw), "bom": raw.startswith(b"\xef\xbb\xbf"),
            "crlf": crlf, "bare_lf": raw.count(b"\n") - crlf, "bare_cr": raw.count(b"\r") - crlf}


def parse(raw: bytes) -> tuple[list[str], list[dict]]:
    text = raw.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    return list(reader.fieldnames or []), list(reader)


def squash(value: str) -> str:
    return " ".join(value.replace("\r", " ").replace("\n", " ").split())


def kind_of(row: dict) -> str:
    if row.get("source") == "synthetic_egyptian_brands":
        return "synthetic"
    return "url" if row.get("content_type") == "url" else "text"


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    name_a, name_b = sys.argv[1], sys.argv[2]
    raw_a, raw_b = load_bytes(name_a), load_bytes(name_b)

    print(f"A = {name_a}\nB = {name_b}\n\n== 1. bytes ==")
    for label, raw in (("A", raw_a), ("B", raw_b)):
        f = byte_facts(raw)
        print(f"  {label}: {f['bytes']:,} bytes, BOM={f['bom']}, CRLF lines={f['crlf']:,}, "
              f"bare-LF={f['bare_lf']:,}, bare-CR={f['bare_cr']:,}")
    print(f"  byte-identical: {raw_a == raw_b}")

    cols_a, rows_a = parse(raw_a)
    cols_b, rows_b = parse(raw_b)
    print("\n== 2. structure ==")
    print(f"  header identical: {cols_a == cols_b}" + ("" if cols_a == cols_b else f"\n    A: {cols_a}\n    B: {cols_b}"))
    print(f"  rows: A={len(rows_a):,}  B={len(rows_b):,}")
    ids_a, ids_b = [r["id"] for r in rows_a], [r["id"] for r in rows_b]
    dup_a, dup_b = len(ids_a) - len(set(ids_a)), len(ids_b) - len(set(ids_b))
    print(f"  duplicate ids: A={dup_a}  B={dup_b}")
    only_a, only_b = sorted(set(ids_a) - set(ids_b)), sorted(set(ids_b) - set(ids_a))
    print(f"  ids only in A: {len(only_a):,} {only_a[:MAX_IDS_SHOWN]}")
    print(f"  ids only in B: {len(only_b):,} {only_b[:MAX_IDS_SHOWN]}")
    set_a, set_b = set(ids_a), set(ids_b)  # built once: building them inside the loop is O(n^2)
    common_order_a = [i for i in ids_a if i in set_b]
    common_order_b = [i for i in ids_b if i in set_a]
    print(f"  same row order for shared ids: {common_order_a == common_order_b}")

    print("\n== 3. content (ids present in both) ==")
    by_id_a = {r["id"]: r for r in rows_a}
    by_id_b = {r["id"]: r for r in rows_b}
    shared = [i for i in ids_a if i in by_id_b]
    differing = Counter()
    text_whitespace_only, text_real, example_ids = 0, 0, {}
    split_changes: Counter = Counter()
    for i in shared:
        a, b = by_id_a[i], by_id_b[i]
        for column in cols_a:
            if column not in b or a[column] == b[column]:
                continue
            differing[column] += 1
            example_ids.setdefault(column, []).append(i)
            if column == "text":
                if squash(a[column]) == squash(b[column]):
                    text_whitespace_only += 1
                else:
                    text_real += 1
            if column == "split":
                split_changes[(kind_of(a), a["split"] or "(empty)", b["split"] or "(empty)")] += 1
    if not differing:
        print("  every shared row is IDENTICAL in every column: only the file's bytes differ.")
    for column, n in differing.most_common():
        print(f"  column '{column}' differs in {n:,} rows, e.g. {example_ids[column][:MAX_IDS_SHOWN]}")
    if "text" in differing:
        print(f"    text: {text_whitespace_only:,} differ in whitespace/line endings only, {text_real:,} differ for real")
    if split_changes:
        print("  split changes (kind, old -> new): count")
        for (kind, old, new), n in sorted(split_changes.items()):
            print(f"    {kind:<9} {old:>7} -> {new:<7} {n:,}")


if __name__ == "__main__":
    main()
