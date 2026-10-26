"""
diagnose_allowlist_recall_impact.py

One-off diagnostic (not a permanent evaluation script): checks whether the
Egyptian brand allowlist override in risk_scoring.py is responsible for the
Full Hybrid recall dip seen in evaluate_test_configurations.py's 4-engine
re-run (99.95% -> 99.57%) versus the old 3-engine Table 5.1 numbers.

Only scores URL rows (allowlist only applies when a URL is provided) on
the TEST split, and prints every row where the allowlist fired AND the
row is labeled phishing -- these are exactly the rows that would have
been correctly flagged before the allowlist existed, and are now forced
to LOW/0.0 regardless of what the engines found.

Run from the project root:
    python evaluation/diagnose_allowlist_recall_impact.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.database import SessionLocal  # noqa: E402
from app.engines.risk_scoring import calculate_final_risk_score  # noqa: E402

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    test_df = df[(df["split"] == "test") & (df["content_type"] == "url")]
    print(f"Checking {len(test_df)} URL rows in the test split for allowlist overrides...")

    db = SessionLocal()
    allowlisted_rows = []

    for _, row in test_df.iterrows():
        try:
            result = calculate_final_risk_score(url=row["text"], db=db)
        except Exception:
            continue
        if result["allowlisted"]:
            allowlisted_rows.append((row["text"], row["label"], row["source"]))

    db.close()

    print(f"\nTotal allowlisted rows (any label): {len(allowlisted_rows)}")
    phishing_allowlisted = [r for r in allowlisted_rows if r[1] == "phishing"]
    print(f"Of those, labeled PHISHING (the recall-impact candidates): {len(phishing_allowlisted)}")

    if phishing_allowlisted:
        print("\n=== Phishing-labeled rows forced to LOW by the allowlist ===")
        for url, label, source in phishing_allowlisted:
            print(f"  {url}  (source={source})")
    else:
        print("\nNone. The allowlist is not the cause of the recall dip -- look elsewhere.")

    legit_allowlisted = [r for r in allowlisted_rows if r[1] == "legitimate"]
    print(f"\n(For reference: {len(legit_allowlisted)} legitimate rows were also allowlisted -- "
          "these are the expected, correct cases the feature exists for.)")


if __name__ == "__main__":
    main()
