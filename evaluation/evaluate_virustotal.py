"""
evaluate_virustotal.py

Master Plan Section 5 "worth adding" item: runs a STRATIFIED SAMPLE (not
the full test set -- see rate-limit note below) of test-split URLs
through VirusTotal's v3 API and reports precision/recall/accuracy/F1/
ROC-AUC for comparison against ETIS's own engines and against the
Safe Browsing comparison (evaluate_safe_browsing.py).

SAMPLE SIZE, NOT FULL TEST SET (methodology note -- carry into Chapter
5.3/6.3): VirusTotal's free tier is rate-limited to 4 requests/minute.
Querying the full ~12,700-row test-split URL subset would take roughly
53 hours, which is not a reasonable one-sitting evaluation run. Instead
this script draws a STRATIFIED random sample (SAMPLE_SIZE_PER_CLASS per
label, seed=42 for reproducibility, matching the project's existing split
seed convention), balanced 50/50 phishing/legitimate so precision/recall
are not distorted by the dataset's natural ~76/24 class imbalance. State
the sample size and the free-tier constraint explicitly in the thesis --
do not present this as full-test-set coverage.

LOOKUP METHOD: uses the v3 "get URL analysis" endpoint (GET /urls/{id},
where {id} is the URL-safe-base64-encoded URL with padding stripped).
This retrieves EXISTING community scan reports rather than submitting a
fresh scan-and-wait -- much faster, but means URLs VirusTotal has never
seen before return 404 (treated here as "not flagged," i.e. a negative
prediction -- the same real-world behavior a user of the actual VT
website would see: "no report found" reads as "nothing detected," not
as "definitely malicious"). This is disclosed here so the comparison
isn't overstated as an apples-to-apples live-scan comparison.

Run from the project root:
    python evaluation/evaluate_virustotal.py

Requires VIRUSTOTAL_API_KEY in .env.

Outputs (written to evaluation/):
    virustotal_comparison_report.txt
"""

import base64
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

API_KEY = os.getenv("VIRUSTOTAL_API_KEY")
if not API_KEY:
    raise SystemExit(
        "VIRUSTOTAL_API_KEY not found in .env. Add it first -- see setup steps."
    )

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "evaluation"
LABEL_MAP = {"legitimate": 0, "phishing": 1}

# Free tier: 4 requests/minute. 100 per class = 200 total = ~50 minutes
# at one request per 15s. Adjust down if you want a faster, smaller run.
SAMPLE_SIZE_PER_CLASS = 100
SEED = 42  # matches the project's existing stratified-split seed convention
SECONDS_BETWEEN_REQUESTS = 16  # slightly over the 15s minimum for safety margin


def url_to_vt_id(url: str) -> str:
    return base64.urlsafe_b64encode(url.encode()).decode().strip("=")


def query_url(url: str) -> dict | None:
    """
    Returns the VT analysis stats dict (malicious/suspicious/harmless/
    undetected counts) if VT has a report for this URL, else None (VT has
    never seen it -- treated as "not flagged," see module docstring).
    """
    url_id = url_to_vt_id(url)
    response = requests.get(
        f"https://www.virustotal.com/api/v3/urls/{url_id}",
        headers={"x-apikey": API_KEY},
        timeout=30,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()["data"]["attributes"]["last_analysis_stats"]


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    test_url_df = df[(df["split"] == "test") & (df["content_type"] == "url")].copy()

    phishing_pool = test_url_df[test_url_df["label"] == "phishing"]
    legit_pool = test_url_df[test_url_df["label"] == "legitimate"]

    n_phishing = min(SAMPLE_SIZE_PER_CLASS, len(phishing_pool))
    n_legit = min(SAMPLE_SIZE_PER_CLASS, len(legit_pool))

    sample = pd.concat([
        phishing_pool.sample(n=n_phishing, random_state=SEED),
        legit_pool.sample(n=n_legit, random_state=SEED),
    ]).sample(frac=1, random_state=SEED)  # shuffle

    urls = sample["text"].tolist()
    labels = sample["label"].map(LABEL_MAP).tolist()
    total = len(urls)
    est_minutes = total * SECONDS_BETWEEN_REQUESTS / 60

    print(f"Stratified sample: {n_phishing} phishing + {n_legit} legitimate = {total} URLs")
    print(f"Estimated runtime at {SECONDS_BETWEEN_REQUESTS}s/request: ~{est_minutes:.0f} minutes\n")

    y_true = []
    y_pred = []
    y_score = []  # continuous score for ROC-AUC, where available
    n_unseen = 0
    n_errors = 0

    for i, (url, label) in enumerate(zip(urls, labels)):
        try:
            stats = query_url(url)
        except requests.exceptions.RequestException as exc:
            print(f"  [{i+1}/{total}] ERROR on {url[:50]!r}: {exc} -- skipping")
            n_errors += 1
            time.sleep(SECONDS_BETWEEN_REQUESTS)
            continue

        y_true.append(label)
        if stats is None:
            y_pred.append(0)
            y_score.append(0.0)
            n_unseen += 1
        else:
            total_engines = sum(stats.values())
            flagged = stats.get("malicious", 0) + stats.get("suspicious", 0)
            y_pred.append(1 if flagged > 0 else 0)
            y_score.append(flagged / total_engines if total_engines > 0 else 0.0)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{total}] done ({n_unseen} unseen so far, {n_errors} errors)")

        time.sleep(SECONDS_BETWEEN_REQUESTS)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_score)
        auc_str = f"{auc:.4f}"
    except ValueError:
        auc_str = "n/a (single class in sample)"

    print(f"\n=== VirusTotal vs. ETIS (stratified sample, N={len(y_true)}) ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {auc_str}")
    print(f"Confusion matrix [[TN,FP],[FN,TP]]:\n{cm}")
    print(f"URLs with no VT report (treated as not-flagged): {n_unseen}/{len(y_true)}")

    report_lines = [
        "ETIS -- VirusTotal Commercial-Tool Comparison",
        f"Stratified sample: {n_phishing} phishing + {n_legit} legitimate = {len(y_true)} URLs",
        f"(drawn from {len(test_url_df)}-row test-split URL subset, seed={SEED})",
        "",
        f"Accuracy:  {acc:.4f}",
        f"Precision: {prec:.4f}",
        f"Recall:    {rec:.4f}",
        f"F1:        {f1:.4f}",
        f"ROC-AUC:   {auc_str}",
        "",
        f"Confusion matrix [[TN,FP],[FN,TP]]:\n{cm}",
        "",
        f"URLs with no VT report (treated as not-flagged): {n_unseen}/{len(y_true)}",
        f"Request errors (excluded from sample): {n_errors}",
        "",
        "=" * 70,
        "METHODOLOGY NOTES -- carry into Chapter 5.3/6.3, do not omit:",
        f"- SAMPLE, not full test set: VirusTotal's free tier (4 req/min)",
        f"  makes a full {len(test_url_df)}-row run impractical (~53 hours).",
        f"  This is a {len(y_true)}-URL stratified sample, seed={SEED}.",
        "- Lookup, not fresh scan: queries existing community reports via",
        "  the v3 URL-report endpoint. URLs VT has never seen return no",
        "  report and are treated as 'not flagged' here -- matching what a",
        "  real user checking that URL on virustotal.com would see, but",
        "  meaning this is not a live-rescan comparison.",
        "- Same PhishTank/OpenPhish source-overlap caveat as the Safe",
        "  Browsing comparison applies: VirusTotal aggregates many of the",
        "  same public feeds ETIS's blacklist is seeded from, so recall",
        "  here should be read as an upper-bound estimate, not a clean",
        "  measure of generalization to genuinely novel threats.",
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "virustotal_comparison_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\nSaved report to {OUTPUT_DIR / 'virustotal_comparison_report.txt'}")


if __name__ == "__main__":
    main()
