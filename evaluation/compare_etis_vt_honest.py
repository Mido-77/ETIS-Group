"""
compare_etis_vt_honest.py   (Task B, step 3 of 3)

Scores ETIS on the SAME URLs that were looked up on VirusTotal
(evaluation/vt_honest_sample.csv + evaluation/vt_honest_results.csv) and
writes evaluation/vt_honest_comparison_report.txt.

PRE-DECLARED BEFORE ANY RESULT EXISTS (this file is committed before the
VirusTotal lookup is run):
  * ETIS operating point: verdict MEDIUM or HIGH = flagged (the same point used
    in the final-system evaluation); HIGH-only is reported as a second row.
    ETIS thresholds were chosen on the validation split, never on this sample.
  * VirusTotal primary operating point: at least ONE engine reports "malicious"
    (malicious >= 1). Secondary, also declared in advance: malicious >= 3.
    A sweep over 1..10 is printed for information only: choosing the best VT
    threshold afterwards would be optimistic and must not be quoted.
  * A URL VirusTotal has never seen (not_found) counts as NOT flagged.
  * Report recall (phishing flagged) and false-positive rate (legitimate flagged)
    with Wilson 95% intervals. No pooled accuracy and no ROC-AUC: the sample is a
    balanced 100/100 draw, so accuracy/precision depend on the chosen prevalence.

What this comparison can and cannot say
  * Both tools are scored on URLs from historical PhishTank/OpenPhish feeds;
    VirusTotal aggregates many of the same feeds, so its recall is an upper
    bound for novel threats. It is a lookup, not a fresh scan.
  * ETIS is scored against etis_eval (blacklist seeded from the TRAIN split
    only) and the sample contains only unseen hosts, but the sample is small:
    n = 100 per class. The legitimate class is mostly popular homepages
    (Tranco) plus crawled pages, which VirusTotal finds easy; the false-positive
    rate on obscure legitimate sites is NOT measured here.
  * Google Safe Browsing is NOT part of this script. Its earlier result (recall
    7.7% on 12,702 historical URLs) measures live-threat coverage against a
    historical feed and must never be presented as "ETIS beats Google".

Usage (project root, venv active, DATABASE_URL -> etis_eval in THIS window):
    python evaluation/compare_etis_vt_honest.py
"""

import csv
import math
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

SAMPLE_PATH = PROJECT_ROOT / "evaluation" / "vt_honest_sample.csv"
VT_PATH = PROJECT_ROOT / "evaluation" / "vt_honest_results.csv"
SCORES_PATH = PROJECT_ROOT / "evaluation" / "vt_honest_etis_scores.csv"
REPORT_PATH = PROJECT_ROOT / "evaluation" / "vt_honest_comparison_report.txt"
VT_PRIMARY, VT_SECONDARY = 1, 3


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
    """Two-sided exact binomial test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_inputs() -> tuple[list[dict], dict[str, dict]]:
    for path in (SAMPLE_PATH, VT_PATH):
        if not path.exists():
            sys.exit(f"{path} not found.")
    with open(SAMPLE_PATH, newline="", encoding="utf-8") as f:
        sample = list(csv.DictReader(f))
    with open(VT_PATH, newline="", encoding="utf-8") as f:
        vt = {r["id"]: r for r in csv.DictReader(f)}
    problems = [r["id"] for r in sample if r["id"] not in vt or vt[r["id"]]["status"] not in ("found", "not_found")]
    if problems:
        sys.exit(f"{len(problems)} sample URLs have no completed VirusTotal result (e.g. {problems[:5]}). "
                 "Run run_vt_lookup_honest.py again to finish/retry them.")
    return sample, vt


def score_etis(sample: list[dict]) -> dict[str, dict]:
    from app.core.database import SessionLocal, engine
    from app.engines.risk_scoring import calculate_final_risk_score

    database = engine.url.database or ""
    print(f"Database: {engine.url.render_as_string(hide_password=True)}")
    if not database.endswith("eval"):
        sys.exit("Refusing to run: DATABASE_URL does not point at the evaluation database (etis_eval). "
                 "Set it in THIS PowerShell window with the real password (no angle brackets).")
    db = SessionLocal()
    scores, consecutive_errors, started = {}, 0, time.time()
    for n, row in enumerate(sample, start=1):
        try:
            result = calculate_final_risk_score(url=row["url"], db=db)
            scores[row["id"]] = {"verdict": result["verdict"], "final_score": result["final_score"],
                                 "decided_by": result["decided_by"] or "", "error": ""}
            consecutive_errors = 0
        except Exception as exc:
            db.rollback()
            scores[row["id"]] = {"verdict": "", "final_score": "", "decided_by": "", "error": f"{type(exc).__name__}: {exc}"[:150]}
            consecutive_errors += 1
            if consecutive_errors >= 25:
                sys.exit("25 consecutive scoring failures; check DATABASE_URL and the venv.")
        if n % 50 == 0:
            print(f"  ...{n}/{len(sample)} ({time.time() - started:.0f}s)")
    db.close()
    return scores


def main() -> None:
    sample, vt = load_inputs()
    scores = score_etis(sample)

    with open(SCORES_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "url", "label", "etis_verdict", "etis_score", "decided_by", "error",
                         "vt_status", "vt_malicious", "vt_suspicious"])
        for r in sample:
            s, v = scores[r["id"]], vt[r["id"]]
            writer.writerow([r["id"], r["url"], r["label"], s["verdict"], s["final_score"], s["decided_by"], s["error"],
                             v["status"], v["malicious"], v["suspicious"]])

    errored = [r for r in sample if scores[r["id"]]["error"]]
    rows = [r for r in sample if not scores[r["id"]]["error"]]
    by_label = {lab: [r for r in rows if r["label"] == lab] for lab in ("phishing", "legitimate")}

    def vt_malicious(r) -> int:
        return int(vt[r["id"]]["malicious"] or 0) if vt[r["id"]]["status"] == "found" else 0

    tools = {
        "ETIS  MEDIUM or HIGH   (pre-declared primary)": lambda r: scores[r["id"]]["verdict"] in ("MEDIUM", "HIGH"),
        "ETIS  HIGH only": lambda r: scores[r["id"]]["verdict"] == "HIGH",
        f"VT    malicious >= {VT_PRIMARY}      (pre-declared primary)": lambda r: vt_malicious(r) >= VT_PRIMARY,
        f"VT    malicious >= {VT_SECONDARY}      (pre-declared secondary)": lambda r: vt_malicious(r) >= VT_SECONDARY,
    }

    out = ["ETIS vs VirusTotal on a FRESH unseen-host sample (current test split)",
           f"sample: {len(by_label['phishing'])} phishing + {len(by_label['legitimate'])} legitimate URLs "
           f"(one URL per host; shared-platform rows excluded); ETIS scoring errors excluded: {len(errored)}",
           "ETIS scored against etis_eval (blacklist seeded from the TRAIN split only). Intervals: Wilson 95%.",
           "", "== Operating points (pre-declared) ==",
           f"{'tool / threshold':<52}{'recall (phishing flagged)':<38}{'false-positive rate (legit flagged)'}"]
    for name, flag in tools.items():
        k_p, k_l = sum(flag(r) for r in by_label["phishing"]), sum(flag(r) for r in by_label["legitimate"])
        out.append(f"{name:<52}{rate(k_p, len(by_label['phishing'])):<38}{rate(k_l, len(by_label['legitimate']))}")

    out += ["", "== VirusTotal lookup status =="]
    for lab in ("phishing", "legitimate"):
        counts = Counter(vt[r["id"]]["status"] for r in by_label[lab])
        out.append(f"  {lab:<11} {dict(counts)}   (not_found counts as NOT flagged)")

    out += ["", "== Paired outcomes, ETIS (MEDIUM+) vs VT (malicious >= 1), same URLs =="]
    etis_flag, vt_flag = tools[list(tools)[0]], tools[list(tools)[2]]
    for lab in ("phishing", "legitimate"):
        part = by_label[lab]
        both = sum(etis_flag(r) and vt_flag(r) for r in part)
        only_e = sum(etis_flag(r) and not vt_flag(r) for r in part)
        only_v = sum(vt_flag(r) and not etis_flag(r) for r in part)
        neither = len(part) - both - only_e - only_v
        out.append(f"  {lab:<11} both flag {both}   ETIS only {only_e}   VT only {only_v}   neither {neither}"
                   f"   exact McNemar p = {exact_mcnemar_p(only_e, only_v):.3f}")
    out.append("  (the test uses only the discordant pairs; with n = 100 per class, small differences are not significant)")

    out += ["", "== FOR INFORMATION ONLY: VirusTotal threshold sweep (post hoc; do NOT quote a 'best' threshold) =="]
    for t in range(1, 11):
        out.append(f"  malicious >= {t:<2} recall {rate(sum(vt_malicious(r) >= t for r in by_label['phishing']), len(by_label['phishing']))}"
                   f"   FPR {rate(sum(vt_malicious(r) >= t for r in by_label['legitimate']), len(by_label['legitimate']))}")

    out += ["", "== ETIS misses and false positives (URLs only) =="]
    for r in by_label["phishing"]:
        if not etis_flag(r):
            out.append(f"  ETIS MISSED phishing   {r['url'][:110]}   (VT malicious: {vt_malicious(r)})")
    for r in by_label["legitimate"]:
        if etis_flag(r):
            out.append(f"  ETIS FLAGGED legitimate {r['url'][:110]}   (decided by {scores[r['id']]['decided_by']}; VT malicious: {vt_malicious(r)})")
    for r in by_label["legitimate"]:
        if vt_flag(r):
            out.append(f"  VT   FLAGGED legitimate {r['url'][:110]}   (VT malicious: {vt_malicious(r)}; ETIS {scores[r['id']]['verdict']})")

    out += ["", "Methodology notes (carry into Chapter 5.3 / 6.3; do not omit)",
            " - Fresh sample from the CURRENT test split, unseen hosts only; the earlier 200-URL sample came from the old split and",
            "   overlaps the new training split, so its 'ETIS 100%' is not reportable and is not used here.",
            " - VirusTotal is a lookup of existing community/engine reports (v3 GET /urls/{id}), not a rescan; unseen URLs count as",
            "   not flagged. VirusTotal aggregates feeds that overlap this dataset's phishing source, so its recall is an upper bound.",
            " - The legitimate class is mostly popular Tranco homepages plus crawled pages; false-positive rates on obscure",
            "   legitimate sites are not measured here (see the controls in the held-out typosquat report).",
            " - n = 100 per class: intervals are wide; do not claim one tool is better unless the intervals and the paired test agree.",
            " - Google Safe Browsing is reported separately and never as a head-to-head with ETIS (live threats vs historical feed)."]

    report = "\n".join(out)
    print("\n" + report)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved {REPORT_PATH} and {SCORES_PATH}")


if __name__ == "__main__":
    main()
