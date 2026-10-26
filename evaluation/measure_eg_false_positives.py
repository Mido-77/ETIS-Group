"""
measure_eg_false_positives.py   (false-positive measurement on real Egyptian sites, step 2 of 2)

Measures how often the final system flags REAL Egyptian sites, replacing the
small hand-written control list in the held-out typosquat benchmark (7/36,
effective n = 18 domains, list written from memory).

PRE-DECLARED BEFORE ANY RESULT EXISTS (commit this file and the sample first):
  * "flagged" = verdict MEDIUM or HIGH (primary); HIGH-only reported second.
  * Rows the Egyptian-brand allowlist overrides are EXCLUDED from the rates
    (the allowlist cancels URL signals, so they say nothing about the models)
    and counted separately.
  * Rates are given per URL type / popularity band, never pooled across them.
  * The unit of independence is the DOMAIN, not the URL: each section also
    reports the share of DOMAINS with at least one flagged URL.
  * Nothing here is used to tune thresholds or rules. If a mitigation follows,
    measure it on a fresh sample (new seed or a newer Tranco list).

Sections
  1. DATASET: legitimate URLs with host ending ".eg" in the TEST split of
     data/processed/dataset.csv (hosts unseen by the models by construction;
     shared-platform rows excluded). Covers bare hosts AND URLs with paths
     (Tranco rows vs crawled pages).
  2. EXTERNAL: data/benchmarks/eg_legit_external.csv (build_tranco_eg_sample.py):
     Tranco-listed .eg domains absent from the dataset, by popularity band,
     apex form primary and www form secondary. Homepage-style URLs only.

Assumption and limits (state in the thesis)
  * Tranco-listed or dataset-labelled "legitimate" is an ASSUMPTION. Every
    flagged URL printed below must be checked by hand (look it up on
    virustotal.com in the browser; do not visit suspicious sites). Report the
    rate both as measured and, if some flagged URLs turn out to be malicious,
    after removing them, saying so.
  * The legitimate rows of the dataset are Tranco top-20k + crawled pages,
    so they lean towards well-known sites.

Usage (project root, venv active, DATABASE_URL -> etis_eval in THIS window):
    python evaluation/measure_eg_false_positives.py
"""

import csv
import math
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"
GROUPS_PATH = PROJECT_ROOT / "data" / "processed" / "split_groups.csv"
EXTERNAL_PATH = PROJECT_ROOT / "data" / "benchmarks" / "eg_legit_external.csv"
SCORES_PATH = PROJECT_ROOT / "evaluation" / "eg_false_positive_scores.csv"
REPORT_PATH = PROJECT_ROOT / "evaluation" / "eg_false_positive_report.txt"
EG_SECOND_LEVEL = {"com", "gov", "edu", "org", "net", "sci", "eun", "mil", "name", "info", "biz"}
THIN_N = 20


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


def has_path(url: str) -> bool:
    try:
        parts = urlsplit(url if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url) else "http://" + url)
    except ValueError:
        return False
    return len(parts.path.strip("/")) > 0 or bool(parts.query)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(k: int, n: int) -> str:
    if n == 0:
        return "n/a (n=0)"
    lo, hi = wilson(k, n)
    return f"{100 * k / n:5.1f}% [{100 * lo:4.1f}, {100 * hi:5.1f}]  ({k}/{n})" + ("  [thin]" if n < THIN_N else "")


def load_dataset_rows() -> list[dict]:
    for path in (DATASET_PATH, GROUPS_PATH):
        if not path.exists():
            sys.exit(f"{path} not found.")
    with open(GROUPS_PATH, newline="", encoding="utf-8") as f:
        group_type = {r["id"]: r["group_type"] for r in csv.DictReader(f)}
    rows = []
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["content_type"] != "url" or r["label"] != "legitimate" or r["split"] != "test":
                continue
            host = host_of(r["text"])
            if not host.endswith(".eg") or "shared" in group_type.get(r["id"], "").lower():
                continue
            rows.append({"section": "dataset", "id": r["id"], "url": r["text"], "domain": registrable(host),
                         "group": "with_path" if has_path(r["text"]) else "bare_host", "subgroup": r.get("source", "")})
    return rows


def load_external_rows() -> list[dict]:
    if not EXTERNAL_PATH.exists():
        print(f"NOTE: {EXTERNAL_PATH} not found; section 2 skipped.")
        return []
    with open(EXTERNAL_PATH, newline="", encoding="utf-8") as f:
        return [{"section": "external", "id": r["id"], "url": r["url"], "domain": r["domain"],
                 "group": r["band"], "subgroup": r["form"]} for r in csv.DictReader(f)]


def score(rows: list[dict]) -> list[dict]:
    from app.core.database import SessionLocal, engine
    from app.engines.risk_scoring import calculate_final_risk_score

    database = engine.url.database or ""
    print(f"Database: {engine.url.render_as_string(hide_password=True)}")
    if not database.endswith("eval"):
        sys.exit("Refusing to run: DATABASE_URL does not point at the evaluation database (etis_eval). "
                 "Set it in THIS PowerShell window with the real password (no angle brackets).")
    db, consecutive, started = SessionLocal(), 0, time.time()
    for n, row in enumerate(rows, start=1):
        try:
            result = calculate_final_risk_score(url=row["url"], db=db)
            row.update({"verdict": result["verdict"], "final_score": result["final_score"],
                        "decided_by": result["decided_by"] or "", "allowlisted": int(result["allowlisted"]), "error": ""})
            consecutive = 0
        except Exception as exc:
            db.rollback()
            row.update({"verdict": "", "final_score": "", "decided_by": "", "allowlisted": 0, "error": f"{type(exc).__name__}: {exc}"[:150]})
            consecutive += 1
            if consecutive >= 25:
                sys.exit("25 consecutive scoring failures; check DATABASE_URL and the venv.")
        if n % 200 == 0:
            print(f"  ...{n}/{len(rows)} ({time.time() - started:.0f}s)")
    db.close()
    return rows


def flagged(row: dict, level: str = "MEDIUM+") -> bool:
    return row["verdict"] in ("MEDIUM", "HIGH") if level == "MEDIUM+" else row["verdict"] == "HIGH"


def block(rows: list[dict], title: str, group_label: str) -> list[str]:
    out = [f"{title}"]
    by_group = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    out.append(f"  {group_label:<22}{'URLs >= MEDIUM  [95% CI]':<46}{'URLs = HIGH':<30}{'DOMAINS with >= 1 flagged URL'}")
    for g in sorted(by_group):
        part = by_group[g]
        domains = defaultdict(list)
        for r in part:
            domains[r["domain"]].append(flagged(r))
        flagged_domains = sum(any(v) for v in domains.values())
        out.append(f"  {g:<22}{rate(sum(flagged(r) for r in part), len(part)):<46}"
                   f"{rate(sum(flagged(r, 'HIGH') for r in part), len(part)):<30}{rate(flagged_domains, len(domains))}")
    return out


def main() -> None:
    rows = score(load_dataset_rows() + load_external_rows())
    errors = [r for r in rows if r["error"]]
    rows = [r for r in rows if not r["error"]]
    allowlisted = [r for r in rows if r["allowlisted"]]
    rows = [r for r in rows if not r["allowlisted"]]

    fields = ["section", "id", "url", "domain", "group", "subgroup", "verdict", "final_score", "decided_by", "allowlisted"]
    with open(SCORES_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows + allowlisted)

    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        head = "unknown"
    out = ["ETIS false-positive measurement on real Egyptian sites (final system, etis_eval)",
           f"git HEAD: {head}; scoring errors excluded: {len(errors)}; allowlist-overridden rows excluded from rates: {len(allowlisted)}",
           "'flagged' = MEDIUM or HIGH. Wilson 95% intervals. URLs of one domain are not independent: read the DOMAIN column first.",
           "ASSUMPTION: these URLs are legitimate. Check every flagged URL by hand before quoting a rate (see the end)."]

    dataset_rows = [r for r in rows if r["section"] == "dataset"]
    external_rows = [r for r in rows if r["section"] == "external"]

    out += ["", "== 1. DATASET: legitimate .eg URLs in the TEST split (hosts unseen by the models; shared platforms excluded) =="]
    out += block(dataset_rows, "by URL type:", "URL type")
    by_source = defaultdict(list)
    for r in dataset_rows:
        by_source[r["subgroup"]].append(r)
    out.append("by dataset source (descriptive):")
    for src in sorted(by_source):
        out.append(f"  {src:<22}{rate(sum(flagged(r) for r in by_source[src]), len(by_source[src]))}")

    if external_rows:
        for form, label in (("apex", "PRIMARY: apex form (https://domain)"), ("www", "SECONDARY: www form (https://www.domain)")):
            part = [r for r in external_rows if r["subgroup"] == form]
            out += ["", f"== 2. EXTERNAL Tranco .eg domains absent from the dataset, by popularity band; {label} =="]
            out += block(part, "by Tranco rank band:", "band")

    flagged_rows = [r for r in rows if flagged(r)]
    out += ["", f"== Flagged URLs: CHECK EACH BY HAND ({len(flagged_rows)}); decided_by counts: {dict(Counter(r['decided_by'] for r in flagged_rows))} =="]
    for r in flagged_rows[:80]:
        out.append(f"  {r['verdict']:<6} {r['decided_by']:<10} {r['section']:<9}{r['group']:<14}{r['url'][:100]}")
    if len(flagged_rows) > 80:
        out.append(f"  ... {len(flagged_rows) - 80} more in {SCORES_PATH.name}")

    out += ["", "How to use this",
            " - Quote rates per URL type and per band with their intervals and the domain-level column, never one pooled number.",
            " - The dataset section is biased to well-known sites (Tranco top-20k + crawled pages); the external section covers obscure",
            "   domains but only homepage-style URLs. Paths on obscure sites are not measured.",
            " - Before quoting: inspect each flagged URL. Any that prove malicious are not false positives; say how many and why.",
            " - Do not tune thresholds/rules on this report; measure any mitigation on a fresh sample."]

    report = "\n".join(out)
    print("\n" + report)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved {REPORT_PATH} and {SCORES_PATH}")


if __name__ == "__main__":
    main()
