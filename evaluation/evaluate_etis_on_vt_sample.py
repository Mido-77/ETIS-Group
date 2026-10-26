"""
evaluate_etis_on_vt_sample.py

Companion to evaluate_virustotal.py: scores ETIS's own hybrid system on
the EXACT SAME 200-URL stratified sample (same seed=42, same sampling
logic) used for the VirusTotal comparison, so that row is a true
apples-to-apples comparison rather than comparing VirusTotal-on-a-200-
sample against ETIS-on-the-full-13,003-row-test-set.

The full-test-set ETIS numbers (99.91% accuracy etc., from
evaluate_test_configurations.py) remain the correct, primary numbers for
Table 5.1 -- this script exists ONLY to produce a fair same-N comparison
row for the VirusTotal table in Chapter 5.3/6.3, not to replace the main
evaluation.

Run from the project root:
    python evaluation/evaluate_etis_on_vt_sample.py

Outputs (written to evaluation/):
    etis_on_vt_sample_report.txt
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.database import SessionLocal  # noqa: E402
from app.engines.risk_scoring import calculate_final_risk_score  # noqa: E402

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "evaluation"
LABEL_MAP = {"legitimate": 0, "phishing": 1}
BINARY_THRESHOLD = 50.0

# MUST match evaluate_virustotal.py exactly, or this won't be the same sample.
SAMPLE_SIZE_PER_CLASS = 100
SEED = 42


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    test_url_df = df[(df["split"] == "test") & (df["content_type"] == "url")].copy()

    phishing_pool = test_url_df[test_url_df["label"] == "phishing"]
    legit_pool = test_url_df[test_url_df["label"] == "legitimate"]

    n_phishing = min(SAMPLE_SIZE_PER_CLASS, len(phishing_pool))
    n_legit = min(SAMPLE_SIZE_PER_CLASS, len(legit_pool))

    # Identical sampling + shuffle sequence to evaluate_virustotal.py, so
    # this reproduces the SAME 200 rows, in the same order.
    sample = pd.concat([
        phishing_pool.sample(n=n_phishing, random_state=SEED),
        legit_pool.sample(n=n_legit, random_state=SEED),
    ]).sample(frac=1, random_state=SEED)

    urls = sample["text"].tolist()
    labels = sample["label"].map(LABEL_MAP).tolist()

    print(f"Scoring ETIS on the same {len(urls)}-URL sample used for VirusTotal "
          f"({n_phishing} phishing + {n_legit} legitimate, seed={SEED})...")

    db = SessionLocal()
    y_true = []
    y_pred = []
    y_score = []

    for i, (url, label) in enumerate(zip(urls, labels)):
        try:
            result = calculate_final_risk_score(url=url, db=db)
        except Exception as exc:
            print(f"  [{i+1}/{len(urls)}] ERROR on {url[:50]!r}: {exc} -- skipping")
            continue
        y_true.append(label)
        y_score.append(result["final_score"] / 100.0)
        y_pred.append(1 if result["final_score"] >= BINARY_THRESHOLD else 0)

    db.close()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_score)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n=== ETIS on the same {len(y_true)}-URL VirusTotal sample ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Confusion matrix [[TN,FP],[FN,TP]]:\n{cm}")

    report_lines = [
        "ETIS -- scored on the SAME 200-URL sample used for the VirusTotal comparison",
        f"({n_phishing} phishing + {n_legit} legitimate, seed={SEED})",
        "",
        f"Accuracy:  {acc:.4f}",
        f"Precision: {prec:.4f}",
        f"Recall:    {rec:.4f}",
        f"F1:        {f1:.4f}",
        f"ROC-AUC:   {auc:.4f}",
        "",
        f"Confusion matrix [[TN,FP],[FN,TP]]:\n{cm}",
        "",
        "This is a same-N companion to the full-test-set ETIS numbers in",
        "evaluation/test_configuration_report.txt (N=13,003) -- use THIS",
        "row specifically when comparing against VirusTotal's 200-URL",
        "sample, and the full-test-set numbers everywhere else (Table 5.1).",
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "etis_on_vt_sample_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\nSaved report to {OUTPUT_DIR / 'etis_on_vt_sample_report.txt'}")


if __name__ == "__main__":
    main()
