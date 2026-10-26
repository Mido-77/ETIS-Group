"""
calibrate_thresholds.py

Runs the actual risk_scoring aggregator (app/engines/risk_scoring.py)
against every VALIDATION-split row (never test -- picking thresholds using
test data would let them overfit to numbers meant to be reported unbiased
later) and uses the real score distributions to pick LOW/MEDIUM/HIGH
cutoffs, replacing the provisional 39/69 round numbers.

Methodology: set the HIGH cutoff at the score below which 95% of TRUE
LEGITIMATE validation examples fall (so only the riskiest-looking 5% of
legitimate traffic would be HIGH-flagged), and the LOW cutoff at the score
above which 95% of TRUE PHISHING validation examples fall (so only the
least-obvious 5% of real phishing could slip into LOW). Both cutoffs are
then rounded in the conservative direction -- LOW rounded down, HIGH
rounded up -- shrinking the outer bands and growing MEDIUM, consistent
with this project's stated preference (Chapter 5.2/5.4) for over-flagging
over under-flagging in a security tool.

KNOWN CAVEAT (see conversation record / Chapter 7): blacklist_entries was
seeded from the full PhishTank/OpenPhish feeds, not just the training
split, so some validation-split phishing URLs are themselves already in
the blacklist. This inflates their scores via trivial exact-match rather
than genuine generalization. The resulting thresholds reflect how the
actual deployed system scores these inputs (still meaningful), but are
NOT a clean measure of generalization to unseen phishing -- worth stating
as a limitation.

Run from the project root:
    python evaluation/calibrate_thresholds.py
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))  # so `app.*` imports resolve

from app.core.database import SessionLocal  # noqa: E402
from app.engines.risk_scoring import calculate_final_risk_score  # noqa: E402

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "evaluation"
TEXT_CONTENT_TYPES = ("text", "email")

PURITY_TARGET = 0.95  # 95th percentile in each direction


def score_row(row, db):
    """Run one dataset row through the aggregator. Returns final_score, or
    None if the row errors out (logged, not fatal -- a handful of bad rows
    shouldn't kill a 13,000-row run)."""
    try:
        if row["content_type"] in TEXT_CONTENT_TYPES:
            result = calculate_final_risk_score(text=row["text"])
        else:
            result = calculate_final_risk_score(url=row["text"], db=db)
        return result["final_score"]
    except Exception as e:
        return None, str(e)


def main():
    df = pd.read_csv(DATASET_PATH)
    val_df = df[df["split"] == "val"].copy()
    print(f"Scoring {len(val_df)} validation-split rows through the full aggregator...")
    print("(this includes real blacklist DB lookups + LSTM inference on text rows -- expect a few minutes)\n")

    db = SessionLocal()
    scores = []
    labels = []
    errors = 0
    start = time.time()

    for i, (_, row) in enumerate(val_df.iterrows()):
        result = score_row(row, db)
        if isinstance(result, tuple):  # error case
            errors += 1
            continue
        scores.append(result)
        labels.append(row["label"])

        if (i + 1) % 2000 == 0:
            elapsed = time.time() - start
            print(f"  ...{i + 1}/{len(val_df)} rows ({elapsed:.0f}s elapsed)")

    db.close()

    elapsed_total = time.time() - start
    print(f"\nDone in {elapsed_total:.0f}s. {errors} rows errored and were skipped.\n")

    results_df = pd.DataFrame({"score": scores, "label": labels})
    legit_scores = results_df[results_df["label"] == "legitimate"]["score"]
    phishing_scores = results_df[results_df["label"] == "phishing"]["score"]

    print("=== Score distribution by true label ===")
    print("Legitimate:")
    print(legit_scores.describe())
    print("\nPhishing:")
    print(phishing_scores.describe())

    # LOW's upper edge comes from the LEGITIMATE distribution (95th percentile
    # of legitimate scores). HIGH's lower edge comes from the PHISHING
    # distribution (5th percentile of phishing scores).
    low_raw = np.percentile(legit_scores, PURITY_TARGET * 100)
    high_raw = np.percentile(phishing_scores, (1 - PURITY_TARGET) * 100)

    low_threshold = int(np.floor(low_raw))    # round DOWN -- more conservative, stricter LOW

    # HIGH threshold: floor rather than ceil, by deliberate product decision --
    # a bare blacklist exact-match at the default reputation score (no other
    # corroborating signal) produces a score of ~51.43 for URL-only
    # submissions. Ceiling this to 52 would exclude that entire common case
    # from HIGH, pushing the majority of real phishing into MEDIUM instead.
    # Decision: a bare blacklist match alone should be enough to reach HIGH.
    high_threshold = int(np.floor(high_raw))

    print(f"\n=== Calibration result ===")
    print(f"Raw 95th-percentile-of-legitimate (LOW boundary):  {low_raw:.2f} -> rounded down to {low_threshold}")
    print(f"Raw 5th-percentile-of-phishing (HIGH boundary):     {high_raw:.2f} -> rounded down to {high_threshold}")

    if low_threshold >= high_threshold:
        print("\n*** WARNING: low_threshold >= high_threshold. Score distributions overlap")
        print("*** too much at these percentiles for clean separation. Do not use these")
        print("*** thresholds as-is -- inspect the histogram and consider a less strict")
        print("*** purity target (e.g. 90th percentile instead of 95th).")
    else:
        print(f"\nRecommended: LOW = 0-{low_threshold}, MEDIUM = {low_threshold + 1}-{high_threshold - 1}, HIGH = {high_threshold}-100")

    # Bucket purity check: with the NEW thresholds, how many of each true
    # label land in each bucket? This is the real-world validation of the
    # choice above, not just the percentile math in isolation.
    def bucket(score):
        if score <= low_threshold:
            return "LOW"
        if score < high_threshold:
            return "MEDIUM"
        return "HIGH"

    results_df["bucket"] = results_df["score"].apply(bucket)
    crosstab = pd.crosstab(results_df["bucket"], results_df["label"])
    print("\n=== Bucket composition under recommended thresholds ===")
    print(crosstab)

    # Outlier rows worth manual inspection: legitimate examples that scored
    # highest (candidate false positives), and phishing examples that scored
    # lowest (candidate false negatives / hardest cases). Useful for
    # thesis Chapter 6 error analysis and Section 5.8 worked examples.
    results_with_input = results_df.copy()
    results_with_input["text"] = val_df["text"].values
    results_with_input["content_type"] = val_df["content_type"].values

    top_legit_outliers = results_with_input[results_with_input["label"] == "legitimate"].nlargest(5, "score")
    bottom_phishing_outliers = results_with_input[results_with_input["label"] == "phishing"].nsmallest(5, "score")

    print("\n=== Top 5 highest-scoring LEGITIMATE examples (candidate false positives) ===")
    print(top_legit_outliers[["text", "content_type", "score"]].to_string(index=False))

    print("\n=== Bottom 5 lowest-scoring PHISHING examples (candidate false negatives) ===")
    print(bottom_phishing_outliers[["text", "content_type", "score"]].to_string(index=False))

    # Diagnostic re-run: for the false-positive candidates specifically, pull
    # the full engine breakdown (not just the final score) so we can see
    # WHICH engine drove the high score -- e.g. confirming whether these are
    # hitting the blacklist's domain-level match on platforms not covered by
    # SHARED_HOSTING_DOMAINS (see blacklist_lookup.py), the same false-positive
    # mechanism that list was originally built to prevent for docs.google.com etc.
    print("\n=== Diagnostic: engine breakdown for false-positive candidates ===")
    db_diag = SessionLocal()
    for _, outlier_row in top_legit_outliers.iterrows():
        detail = calculate_final_risk_score(
            url=outlier_row["text"] if outlier_row["content_type"] == "url" else None,
            text=outlier_row["text"] if outlier_row["content_type"] != "url" else None,
            db=db_diag,
        )
        bl = detail["engine_details"]["blacklist"]
        print(f"\n{outlier_row['text'][:60]!r} (score {outlier_row['score']})")
        if bl:
            print(f"  blacklist match_type={bl['match_type']}, score={bl['score']}, matched_entry={bl['matched_entry']}")
        else:
            print("  blacklist: not applicable")
    db_diag.close()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "threshold_calibration_report.txt", "w", encoding="utf-8") as f:
        f.write("ETIS Risk Scoring -- Threshold Calibration Report\n")
        f.write(f"Validation rows scored: {len(results_df)} ({errors} errored/skipped)\n\n")
        f.write("Legitimate score distribution:\n")
        f.write(str(legit_scores.describe()) + "\n\n")
        f.write("Phishing score distribution:\n")
        f.write(str(phishing_scores.describe()) + "\n\n")
        f.write(f"Recommended LOW threshold:  <= {low_threshold}\n")
        f.write(f"Recommended HIGH threshold: >= {high_threshold}\n\n")
        f.write("Bucket composition:\n")
        f.write(str(crosstab) + "\n\n")
        f.write("CAVEAT: blacklist_entries was seeded from the full PhishTank/OpenPhish\n")
        f.write("feeds, not just the training split, so some validation-split phishing\n")
        f.write("URLs are themselves already in the blacklist. This inflates their scores\n")
        f.write("via exact-match rather than genuine generalization -- see module docstring.\n\n")
        f.write("Top 5 highest-scoring LEGITIMATE examples (candidate false positives):\n")
        f.write(top_legit_outliers[["text", "content_type", "score"]].to_string(index=False) + "\n\n")
        f.write("Bottom 5 lowest-scoring PHISHING examples (candidate false negatives):\n")
        f.write(bottom_phishing_outliers[["text", "content_type", "score"]].to_string(index=False) + "\n")

    # Histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(legit_scores, bins=30, alpha=0.6, label="legitimate", color="#4f5fe0")
    ax.hist(phishing_scores, bins=30, alpha=0.6, label="phishing", color="#d6453c")
    ax.axvline(low_threshold, color="green", linestyle="--", label=f"LOW cutoff ({low_threshold})")
    ax.axvline(high_threshold, color="red", linestyle="--", label=f"HIGH cutoff ({high_threshold})")
    ax.set_xlabel("Final risk score")
    ax.set_ylabel("Count")
    ax.set_title("Validation score distribution by true label")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "threshold_calibration_histogram.png", dpi=150)

    print(f"\nSaved report and histogram to {OUTPUT_DIR}")
    print(f"\nNext step: update PROVISIONAL_LOW_MAX and PROVISIONAL_MEDIUM_MAX")
    print(f"in app/engines/risk_scoring.py to {low_threshold} and {high_threshold - 1}.")


if __name__ == "__main__":
    main()
