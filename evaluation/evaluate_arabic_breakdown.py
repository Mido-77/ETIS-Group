"""
evaluate_arabic_breakdown.py

Chapter 5.6 (Arabic Phishing Detection Results) needs LSTM/hybrid
performance broken out by language, not just the aggregate text/email
subset numbers already in Table 5.2. This script re-scores the TEST
split's text/email rows (same 301-row subset as Table 5.1/5.2), grouped
by the dataset's `language` column (ar / en / mixed), for two engines:

    - LSTM only    -- the engine actually built to read Arabic phishing
                      language (Section 3.8's rationale)
    - Full hybrid  -- included so 5.6 can also state whether Arabic-
                      language submissions get materially different
                      overall risk scores, not just LSTM sub-scores

SCOPE CAVEAT (carry into 5.6 prose, don't omit): per Table 3.2, the
text/email subset is already overwhelmingly Arabic (1,660 of 1,996 text
rows) with very few English text rows (2 of 1,996) -- so the English
bucket in this breakdown will have a tiny N, possibly too small for a
stable accuracy/F1/ROC-AUC reading, and definitely too small to support
any claim of an English-vs-Arabic performance gap. Report the English
numbers if they exist, but do not draw conclusions from them -- say so
explicitly rather than silently omitting them.

Run from the project root:
    python evaluation/evaluate_arabic_breakdown.py

Outputs (written to evaluation/):
    arabic_breakdown_report.txt      -- full metrics + caveats, text
    arabic_breakdown_comparison.png  -- grouped bar chart by language
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))  # so `app.*` imports resolve

from app.core.database import SessionLocal  # noqa: E402
from app.core.exceptions import InvalidScanInputError  # noqa: E402
from app.engines.deep_learning.lstm_inference import calculate_lstm_phishing_score  # noqa: E402
from app.engines.risk_scoring import calculate_final_risk_score  # noqa: E402

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "evaluation"
TEXT_CONTENT_TYPES = ("text", "email")
BINARY_THRESHOLD = 50.0  # same convention as evaluate_test_configurations.py

LABEL_MAP = {"legitimate": 0, "phishing": 1}
LANGUAGES = ["ar", "mixed", "en"]
LANGUAGE_DISPLAY = {"ar": "Arabic", "mixed": "Mixed", "en": "English"}
MIN_N_FOR_ROC_AUC = 10  # below this, ROC-AUC on a single class or tiny N is not meaningful


def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    y_pred = (scores >= BINARY_THRESHOLD).astype(int)
    probs = scores / 100.0
    metrics = {
        "n": len(y_true),
        "n_phishing": int(y_true.sum()),
        "n_legitimate": int(len(y_true) - y_true.sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if len(y_true) >= MIN_N_FOR_ROC_AUC and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, probs)
    else:
        metrics["roc_auc"] = None
    return metrics


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_PATH)
    test_text_df = df[(df["split"] == "test") & (df["content_type"].isin(TEXT_CONTENT_TYPES))].copy()
    print(f"Test-split text/email rows: {len(test_text_df)}")
    print(test_text_df["language"].value_counts())
    print()

    db = SessionLocal()
    results = {lang: {"lstm": {"scores": [], "labels": []}, "hybrid": {"scores": [], "labels": []}} for lang in LANGUAGES}

    for _, row in test_text_df.iterrows():
        lang = row["language"]
        if lang not in LANGUAGES:
            continue
        label = LABEL_MAP[row["label"]]
        text = row["text"]

        try:
            lstm_result = calculate_lstm_phishing_score(text)
            results[lang]["lstm"]["scores"].append(lstm_result["score"])
            results[lang]["lstm"]["labels"].append(label)
        except InvalidScanInputError:
            pass

        try:
            hybrid_result = calculate_final_risk_score(text=text, db=db)
            results[lang]["hybrid"]["scores"].append(hybrid_result["final_score"])
            results[lang]["hybrid"]["labels"].append(label)
        except InvalidScanInputError:
            pass

    db.close()

    all_metrics = {}
    for lang in LANGUAGES:
        all_metrics[lang] = {}
        for engine in ("lstm", "hybrid"):
            labels = np.array(results[lang][engine]["labels"])
            scores = np.array(results[lang][engine]["scores"])
            if len(labels) == 0:
                all_metrics[lang][engine] = None
                continue
            all_metrics[lang][engine] = compute_metrics(labels, scores)

    # --- Print + report ---
    lines = ["ETIS -- Arabic/Mixed/English Breakdown Report (Test Split, Text/Email Subset)\n"]
    for lang in LANGUAGES:
        header = f"=== {LANGUAGE_DISPLAY[lang]} ({lang}) ==="
        print(header)
        lines.append(header)
        for engine in ("lstm", "hybrid"):
            m = all_metrics[lang][engine]
            if m is None:
                line = f"  {engine}: no rows"
                print(line)
                lines.append(line)
                continue
            auc_str = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "n/a (N too small or single-class)"
            line = (
                f"  {engine:<7} N={m['n']:<4} (phishing={m['n_phishing']}, legitimate={m['n_legitimate']})  "
                f"acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  rec={m['recall']:.4f}  "
                f"f1={m['f1']:.4f}  roc_auc={auc_str}"
            )
            print(line)
            lines.append(line)
        print()
        lines.append("")

    lines.append("=" * 70)
    lines.append("CAVEAT: the English bucket is expected to have a very small N (per Table")
    lines.append("3.2, only 2 of 1,996 text rows are English) -- treat any English-bucket")
    lines.append("numbers as illustrative only, not as evidence of an English/Arabic gap.")
    lines.append("Do not report English-bucket ROC-AUC if it printed as n/a above.")

    with open(OUTPUT_DIR / "arabic_breakdown_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved report to {OUTPUT_DIR / 'arabic_breakdown_report.txt'}")

    # --- Bar chart: accuracy/precision/recall/F1 grouped by language, LSTM only ---
    # (hybrid included in the text report for completeness, but the chart focuses
    # on LSTM since that's the engine Section 3.8 specifically built for this)
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
    langs_present = [l for l in LANGUAGES if all_metrics[l]["lstm"] is not None]

    x = np.arange(len(metric_keys))
    width = 0.8 / max(len(langs_present), 1)
    colors = {"ar": "#4f5fe0", "mixed": "#c4881a", "en": "#2e7d4f"}

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, lang in enumerate(langs_present):
        m = all_metrics[lang]["lstm"]
        values = [m[k] for k in metric_keys]
        offset = (i - (len(langs_present) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=f"{LANGUAGE_DISPLAY[lang]} (N={m['n']})", color=colors[lang])
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("LSTM Performance by Language (Test Split)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "arabic_breakdown_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved chart to {OUTPUT_DIR / 'arabic_breakdown_comparison.png'}")


if __name__ == "__main__":
    main()
