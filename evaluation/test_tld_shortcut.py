"""
test_tld_shortcut.py

DIAGNOSTIC (not tuning): does the URL model rely on a ".eg means legitimate"
shortcut? In the dataset every training URL whose host ends in .eg is
legitimate (807 legitimate, 0 phishing), so such a shortcut would be a perfect
rule on the training data.

Design
  * Sample (seed 20261025) 300 phishing + 300 legitimate URLs from the TEST
    split: unseen hosts, one URL per host, shared-platform rows excluded, hosts
    not already ending in .eg, no IP hosts, no ports/userinfo.
  * Score each URL in three forms with the real scorer against etis_eval:
        original                  as in the dataset
        swap_eg                   last host label replaced by "eg"
        swap_net                  last host label replaced by "net" (CONTROL for
                                  "any TLD change perturbs the model")
    Only the last label changes; path, query and subdomains stay.
  * Main quantity: the URL MODEL tier alone (the rule engine and the allowlist
    react to the host in their own ways), plus the final verdict for context.

PRE-DECLARED (commit this file before running):
  * Evidence for a TLD shortcut = the url_model MEDIUM+ recall on PHISHING drops
    clearly from original to swap_eg (paired exact McNemar p < 0.01 and a drop of
    more than 10 points) WITHOUT a comparable drop for swap_net, and/or the
    url_model false-positive rate on LEGITIMATE falls for swap_eg but not swap_net.
  * Anything smaller is "no evidence at this sample size", NOT "no shortcut".
  * The swapped URLs are synthetic. A result describes the model's sensitivity,
    not how real .eg phishing would score.
  * No rule, threshold or model change follows from this script. A mitigation
    (e.g. removing TLD information or adding .eg phishing examples) would have to be
    measured on a fresh sample.

Usage (project root, venv active, DATABASE_URL -> etis_eval in THIS window):
    python evaluation/test_tld_shortcut.py
"""

import csv
import math
import random
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"
GROUPS_PATH = PROJECT_ROOT / "data" / "processed" / "split_groups.csv"
SCORES_PATH = PROJECT_ROOT / "evaluation" / "tld_shortcut_scores.csv"
REPORT_PATH = PROJECT_ROOT / "evaluation" / "tld_shortcut_report.txt"
SEED = 20261025
N_PER_CLASS = 300
VARIANTS = ["original", "swap_eg", "swap_net"]


def split_url(url: str):
    t = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", t):
        return None
    try:
        parts = urlsplit(t)
        host, port = parts.hostname, parts.port
    except ValueError:
        return None
    if not host or "@" in parts.netloc or port is not None:
        return None
    return parts, host.lower()


def swap_tld(url: str, new_label: str) -> str | None:
    parsed = split_url(url)
    if parsed is None:
        return None
    parts, host = parsed
    labels = host.split(".")
    if len(labels) < 2 or labels[-1].isdigit() or labels[-1] in ("eg", new_label):
        return None
    labels[-1] = new_label
    return urlunsplit((parts.scheme, ".".join(labels), parts.path, parts.query, parts.fragment))


def host_key(url: str) -> str:
    parsed = split_url(url)
    if parsed is None:
        return ""
    host = parsed[1]
    return host[4:] if host.startswith("www.") else host


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{100 * k / n:5.1f}% [{100 * lo:4.1f}, {100 * hi:5.1f}]  ({k}/{n})"


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n)


def build_sample() -> list[dict]:
    from app.core.egyptian_brands import EGYPTIAN_BRAND_DOMAINS

    for path in (DATASET_PATH, GROUPS_PATH):
        if not path.exists():
            sys.exit(f"{path} not found.")
    with open(GROUPS_PATH, newline="", encoding="utf-8") as f:
        shared = {r["id"] for r in csv.DictReader(f) if "shared" in r["group_type"].lower()}
    brand_labels = {d.split(".")[0] for _, d in EGYPTIAN_BRAND_DOMAINS}

    train_hosts: set[str] = set()
    pool: dict[str, list[dict]] = {"phishing": [], "legitimate": []}
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["content_type"] != "url":
                continue
            key = host_key(r["text"])
            if r["split"] == "train":
                train_hosts.add(key)
            elif r["split"] == "test" and r["label"] in pool and r["id"] not in shared:
                pool[r["label"]].append({"id": r["id"], "url": r["text"], "label": r["label"], "host": key})

    sample = []
    for label, rows in pool.items():
        eligible = []
        for r in sorted(rows, key=lambda x: x["id"]):
            if not r["host"] or r["host"] in train_hosts or r["host"].endswith(".eg"):
                continue
            if any(swap_tld(r["url"], t) is None for t in ("eg", "net")):
                continue
            if host_key(swap_tld(r["url"], "eg")).split(".")[0] in brand_labels:
                continue          # a swap that lands on a brand-looking host would be confounded by the allowlist/rules
            eligible.append(r)
        random.Random(f"{SEED}|{label}").shuffle(eligible)
        chosen, seen = [], set()
        for r in eligible:
            if r["host"] not in seen:
                seen.add(r["host"])
                chosen.append(r)
            if len(chosen) == N_PER_CLASS:
                break
        if len(chosen) < N_PER_CLASS:
            sys.exit(f"Only {len(chosen)} eligible {label} URLs (needed {N_PER_CLASS}).")
        print(f"{label}: {len(rows):,} test rows -> {len(eligible):,} eligible -> sampled {len(chosen)}")
        sample += chosen
    return sample


def score(sample: list[dict]) -> list[dict]:
    from app.core.database import SessionLocal, engine
    from app.engines.risk_scoring import calculate_final_risk_score

    database = engine.url.database or ""
    print(f"Database: {engine.url.render_as_string(hide_password=True)}")
    if not database.endswith("eval"):
        sys.exit("Refusing to run: DATABASE_URL does not point at the evaluation database (etis_eval).")
    db, consecutive, started = SessionLocal(), 0, time.time()
    records = []
    for n, row in enumerate(sample, start=1):
        for variant in VARIANTS:
            url = row["url"] if variant == "original" else swap_tld(row["url"], variant.split("_")[1])
            rec = {"id": row["id"], "label": row["label"], "variant": variant, "url": url}
            try:
                result = calculate_final_risk_score(url=url, db=db)
                rec.update({"verdict": result["verdict"], "decided_by": result["decided_by"] or "",
                            "allowlisted": int(result["allowlisted"]),
                            "url_model_tier": result["engine_tiers"].get("url_model", ""),
                            "url_model_probability": (result["engine_details"].get("url_model") or {}).get("probability", ""),
                            "error": ""})
                consecutive = 0
            except Exception as exc:
                db.rollback()
                rec.update({"verdict": "", "decided_by": "", "allowlisted": 0, "url_model_tier": "",
                            "url_model_probability": "", "error": f"{type(exc).__name__}: {exc}"[:150]})
                consecutive += 1
                if consecutive >= 25:
                    sys.exit("25 consecutive scoring failures; check DATABASE_URL and the venv.")
            records.append(rec)
        if n % 100 == 0:
            print(f"  ...{n}/{len(sample)} URLs ({time.time() - started:.0f}s)")
    db.close()
    return records


def median(values: list[float]) -> float:
    values = sorted(values)
    return float("nan") if not values else values[len(values) // 2]


def main() -> None:
    records = score(build_sample())
    errors = [r for r in records if r["error"]]
    ok_ids = {r["id"] for r in records} - {r["id"] for r in errors}
    records = [r for r in records if r["id"] in ok_ids]      # keep only URLs scored in all three forms

    with open(SCORES_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "variant", "url", "verdict", "decided_by", "allowlisted",
                                                "url_model_tier", "url_model_probability"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    by = {(r["label"], r["variant"], r["id"]): r for r in records}
    ids = {lab: sorted({r["id"] for r in records if r["label"] == lab}) for lab in ("phishing", "legitimate")}

    def url_flag(label, variant, i):
        return by[(label, variant, i)]["url_model_tier"] in ("MEDIUM", "HIGH")

    def verdict_flag(label, variant, i):
        return by[(label, variant, i)]["verdict"] in ("MEDIUM", "HIGH")

    out = ["TLD-shortcut diagnostic: re-scoring unseen-host test URLs with the last host label swapped",
           f"sample: {len(ids['phishing'])} phishing + {len(ids['legitimate'])} legitimate URLs scored in all three forms; "
           f"URLs dropped for scoring errors: {len({r['id'] for r in errors})}",
           "swap_eg = last label -> 'eg'; swap_net = last label -> 'net' (control). Wilson 95% intervals.",
           "The swapped URLs are SYNTHETIC; this measures the model's sensitivity, not real .eg phishing.",
           "Hosts already ending in .eg or .net are not sampled (they cannot take both swaps); only the LAST host label changes (x.co.uk -> x.co.eg).", ""]

    out.append("== URL model tier alone: share flagged (MEDIUM or HIGH) ==")
    out.append(f"{'class':<12}{'variant':<10}{'URL model MEDIUM+':<38}{'final verdict MEDIUM+':<38}{'median model probability'}")
    for lab, what in (("phishing", "recall"), ("legitimate", "false-positive rate")):
        for variant in VARIANTS:
            n = len(ids[lab])
            probs = [float(by[(lab, variant, i)]["url_model_probability"]) for i in ids[lab]
                     if by[(lab, variant, i)]["url_model_probability"] != ""]
            out.append(f"{lab:<12}{variant:<10}{rate(sum(url_flag(lab, variant, i) for i in ids[lab]), n):<38}"
                       f"{rate(sum(verdict_flag(lab, variant, i) for i in ids[lab]), n):<38}{median(probs):.3f}   ({what})")

    out += ["", "== Paired outcomes, URL model MEDIUM+ (same URLs; exact McNemar on the discordant pairs) =="]
    for lab in ("phishing", "legitimate"):
        for variant in ("swap_eg", "swap_net"):
            a = sum(url_flag(lab, "original", i) and url_flag(lab, variant, i) for i in ids[lab])
            lost = sum(url_flag(lab, "original", i) and not url_flag(lab, variant, i) for i in ids[lab])
            gained = sum(not url_flag(lab, "original", i) and url_flag(lab, variant, i) for i in ids[lab])
            neither = len(ids[lab]) - a - lost - gained
            out.append(f"  {lab:<11} original -> {variant:<9} flagged both {a:>3}   flag LOST {lost:>3}   flag GAINED {gained:>3}   neither {neither:>3}"
                       f"   exact p = {exact_mcnemar_p(lost, gained):.4f}")

    out += ["", "How to read it (pre-declared)",
            " - Evidence for a TLD shortcut: phishing url_model recall drops by more than 10 points original -> swap_eg with p < 0.01",
            "   and NOT comparably for swap_net, and/or legitimate false positives vanish for swap_eg but not swap_net.",
            " - Anything smaller is 'no evidence at this sample size', not 'no shortcut'. n = 300 per class.",
            " - No rule/threshold/model change follows from this; a mitigation must be measured on a fresh sample."]

    report = "\n".join(out)
    print("\n" + report)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved {REPORT_PATH} and {SCORES_PATH}")


if __name__ == "__main__":
    main()
