"""
evaluate_safe_browsing.py

Master Plan Section 5 "worth adding" item: runs the FULL test-split URL
subset through Google Safe Browsing's Lookup API v4 and reports precision/
recall/accuracy/F1 for direct comparison against ETIS's own engines
(see evaluation/test_configuration_report.txt for those numbers).

Uses the batched threatMatches:find endpoint (up to 500 URLs per request),
NOT per-URL scanning -- this is why the full ~12,700-row URL subset is
feasible here in a handful of requests, unlike VirusTotal's free-tier
per-URL rate limit (see evaluate_virustotal.py for that one).

IMPORTANT METHODOLOGY NOTE: Safe Browsing only returns a binary match/
no-match per URL (present in its known-bad-URL database, or not) -- it
does NOT return a continuous risk score. This means:
  - Precision/recall/accuracy/F1 ARE meaningful and directly comparable.
  - ROC-AUC is NOT computable (no probability/score to threshold across) --
    do not report it or attempt to fabricate one.
  - A "no match" result means "not currently in Google's known-bad
    database," not "verified safe" -- the same absence-is-not-evidence-
    of-safety principle already noted for ETIS's own blacklist engine
    (see blacklist_lookup.py's docstring) applies here too, arguably more
    so since Safe Browsing's matching criteria are not disclosed.

Run from the project root:
    python evaluation/evaluate_safe_browsing.py

Requires SAFE_BROWSING_API_KEY in .env (see CLAUDE.md / setup notes).

Outputs (written to evaluation/):
    safe_browsing_comparison_report.txt
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

API_KEY = os.getenv("SAFE_BROWSING_API_KEY")
if not API_KEY:
    raise SystemExit(
        "SAFE_BROWSING_API_KEY not found in .env. Add it first -- see setup steps."
    )

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "evaluation"
API_URL = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={API_KEY}"
BATCH_SIZE = 500  # Safe Browsing's per-request limit
LABEL_MAP = {"legitimate": 0, "phishing": 1}


def query_batch(urls: list[str]) -> set[str]:
    """
    Returns the SUBSET of `urls` that Safe Browsing flagged as a known
    threat. Everything not in the returned set is "no match" (unknown to
    Safe Browsing, not necessarily safe -- see module docstring).
    """
    body = {
        "client": {"clientId": "etis-thesis-evaluation", "clientVersion": "1.0.0"},
        "threatInfo": {
            "threatTypes": [
                "MALWARE",
                "SOCIAL_ENGINEERING",
                "UNWANTED_SOFTWARE",
                "POTENTIALLY_HARMFUL_APPLICATION",
            ],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": u} for u in urls],
        },
    }
    response = requests.post(API_URL, json=body, timeout=30)
    response.raise_for_status()
    matches = response.json().get("matches", [])
    return {m["threat"]["url"] for m in matches}


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    test_url_df = df[(df["split"] == "test") & (df["content_type"] == "url")].copy()
    urls = test_url_df["text"].tolist()
    labels = test_url_df["label"].map(LABEL_MAP).tolist()

    print(f"Querying Safe Browsing for {len(urls)} test-split URL rows "
          f"in batches of {BATCH_SIZE}...")

    flagged_urls: set[str] = set()
    for i in range(0, len(urls), BATCH_SIZE):
        batch = urls[i : i + BATCH_SIZE]
        try:
            flagged_urls |= query_batch(batch)
        except requests.exceptions.RequestException as exc:
            print(f"  Batch {i}-{i+len(batch)} failed: {exc} -- skipping this batch.")
            continue
        print(f"  ...batch {i + len(batch)}/{len(urls)} done "
              f"({len(flagged_urls)} flagged so far)")
        time.sleep(1)  # polite pacing, well under any reasonable quota

    y_true = labels
    y_pred = [1 if u in flagged_urls else 0 for u in urls]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== Google Safe Browsing vs. ETIS test-split URL subset ===")
    print(f"N: {len(urls)}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print("ROC-AUC:   not computable (Safe Browsing returns binary match only)")
    print(f"Confusion matrix [[TN,FP],[FN,TP]]:\n{cm}")
    print(f"\nTotal flagged by Safe Browsing: {len(flagged_urls)} / {len(urls)}")

    report_lines = [
        "ETIS -- Google Safe Browsing Commercial-Tool Comparison",
        f"Test-split URL rows evaluated: {len(urls)}",
        "",
        f"Accuracy:  {acc:.4f}",
        f"Precision: {prec:.4f}",
        f"Recall:    {rec:.4f}",
        f"F1:        {f1:.4f}",
        "ROC-AUC:   not computable (binary match only, no continuous score)",
        "",
        f"Confusion matrix [[TN,FP],[FN,TP]]:\n{cm}",
        "",
        f"Total flagged by Safe Browsing: {len(flagged_urls)} / {len(urls)}",
        "",
        "=" * 70,
        "METHODOLOGY NOTES -- carry into Chapter 5.3/6.3, do not omit:",
        "- Safe Browsing returns a binary known-bad match, not a probability;",
        "  ROC-AUC is not applicable here, unlike ETIS's own 0-100 scores.",
        "- A 'no match' result means the URL is not in Google's disclosed",
        "  known-threat database at query time, not that it is verified safe.",
        "  This is the same absence-is-not-evidence-of-safety caveat that",
        "  applies to ETIS's own blacklist engine.",
        "- Safe Browsing's matching criteria/database coverage are not",
        "  publicly disclosed, so this is a black-box comparison -- reported",
        "  as an external reference point, not a methodology audit.",
        "- Recall here reflects Safe Browsing's coverage of THIS SPECIFIC",
        "  test set (largely PhishTank/OpenPhish-sourced), which Safe",
        "  Browsing itself may partially ingest as a data source -- treat",
        "  as an upper-bound estimate of real-world novel-threat recall,",
        "  similar in spirit to ETIS's own blacklist-leakage caveat.",
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "safe_browsing_comparison_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\nSaved report to {OUTPUT_DIR / 'safe_browsing_comparison_report.txt'}")


if __name__ == "__main__":
    main()
