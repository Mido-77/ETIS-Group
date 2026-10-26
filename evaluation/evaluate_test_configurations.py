"""
evaluate_test_configurations.py

Scores the TEST split (never touched during threshold calibration, which
used validation only -- see evaluation/calibrate_thresholds.py) through
four configurations:

    1. rule-based only   -- calculate_rule_based_score(), all rows
                             (every row has either text or a URL, so this
                             engine is always applicable)
    2. blacklist only     -- calculate_blacklist_reputation_score(), URL
                             rows only (blacklist has no text-input path)
    3. LSTM only          -- calculate_lstm_phishing_score(), text/email
                             rows only (same subset as Table 5.1/5.2 in
                             FYP_REPORT.docx -- this script reproduces
                             those numbers on the SAME split so they can
                             sit in the same comparison table/chart
                             honestly, not just be pasted in from a
                             different run)
    4. full hybrid        -- calculate_final_risk_score(), all rows
                             (rule-based + blacklist + LSTM, weighted
                             and redistributed per risk_scoring.py)

Decision on evaluation methodology (see conversation record): threshold-
DEPENDENT metrics (accuracy/precision/recall/F1) use a plain binary split
at score >= 50 -> predicted phishing, for ALL FOUR configs uniformly.
This is deliberate, not an oversight -- the hybrid's calibrated
LOW/MEDIUM/HIGH bands (0 / 50) were derived from the HYBRID's own combined
score distribution (see threshold_calibration_report.txt) and do not
carry equivalent meaning for a lone engine's raw 0-100 score, which was
never separately calibrated. Borrowing those bands for e.g. rule-based-only
would make its accuracy/F1 numbers an artifact of a threshold built for a
different distribution, not a fair single-engine baseline. A plain
50-point midpoint split is threshold-neutral and mirrors what
ml/baseline_sanity_check.py already used for the TF-IDF+LogReg baseline
(0.5 probability cutoff) -- so this table's methodology stays consistent
with Table 5.1's.

ROC-AUC is reported as the primary, threshold-independent comparison
metric across all four configs for exactly this reason.

IMPORTANT SCOPE CAVEATS (state these in Chapter 5.3/5.8, don't omit):
  - Blacklist-only and full-hybrid's blacklist component share the SAME
    known caveat as calibration: blacklist_entries was seeded from full
    PhishTank/OpenPhish feeds, not train-split-only, so some test-split
    phishing URLs may already sit in the blacklist. This inflates
    blacklist-only (and partially hybrid) recall via trivial exact-match
    rather than pure generalization. Rule-based-only and LSTM-only do NOT
    have this issue (both learned/derived purely from train-split data).
  - Blacklist-only metrics are computed ONLY over the URL-row subset of
    test (blacklist has no text-input path) -- not the full test set.
    Same restriction applies to LSTM-only (text/email-row subset only).
  - Rule-based-only and full-hybrid ARE computed over the full test set,
    since both apply to every row (rule-based always runs; hybrid
    redistributes weights automatically per-row applicability).
  - Because the four configs are evaluated over different row subsets,
    do NOT present this as one clean apples-to-apples table without that
    caveat attached -- footnote it, same spirit as the LSTM subset note
    already in FYP_REPORT.docx Section 3.4/5.1.

Run from the project root (after calibrate_thresholds.py has already run
once, so ETIS_API_KEY / DB / models are all confirmed working):

    python evaluation/evaluate_test_configurations.py

Outputs (written to evaluation/):
    test_configuration_report.txt        -- full metrics + caveats, text
    test_confusion_matrices.png          -- 2x2 grid, one CM per config
    test_roc_curves.png                  -- all 4 ROC curves overlaid
    test_configuration_comparison.png    -- Figure 5.5 bar chart
                                             (accuracy/precision/recall/F1
                                             grouped by config)
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))  # so `app.*` imports resolve

from app.core.database import SessionLocal  # noqa: E402
from app.core.exceptions import InvalidScanInputError  # noqa: E402
from app.engines.deep_learning.lstm_inference import calculate_lstm_phishing_score  # noqa: E402
from app.engines.rule_based.rule_based_engine import calculate_rule_based_score  # noqa: E402
from app.engines.risk_scoring import calculate_final_risk_score  # noqa: E402
from app.engines.threat_intel.blacklist_lookup import calculate_blacklist_reputation_score  # noqa: E402

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "evaluation"
TEXT_CONTENT_TYPES = ("text", "email")
BINARY_THRESHOLD = 50.0  # score >= 50 -> predicted phishing, for ALL configs (see module docstring)

LABEL_MAP = {"legitimate": 0, "phishing": 1}

CONFIG_NAMES = ["rule_based_only", "blacklist_only", "lstm_only", "full_hybrid"]
CONFIG_DISPLAY_NAMES = {
    "rule_based_only": "Rule-Based Only",
    "blacklist_only": "Blacklist Only",
    "lstm_only": "LSTM Only",
    "full_hybrid": "Full Hybrid",
}
CONFIG_COLORS = {
    "rule_based_only": "#4f5fe0",
    "blacklist_only": "#c4881a",
    "lstm_only": "#2e7d4f",
    "full_hybrid": "#d6453c",
}


def score_row(row, db):
    """
    Returns a dict of per-config scores (0-100 or None if not applicable)
    for one dataset row. None entries are excluded from that config's
    metrics later, not treated as 0 -- a config that doesn't apply to a
    row is missing data, not a confident "not phishing" prediction.
    """
    is_url_row = row["content_type"] not in TEXT_CONTENT_TYPES
    text = row["text"] if not is_url_row else None
    url = row["text"] if is_url_row else None

    scores = {"rule_based_only": None, "blacklist_only": None, "lstm_only": None, "full_hybrid": None}

    try:
        rb_result = calculate_rule_based_score(text=text, url=url)
        scores["rule_based_only"] = rb_result["score"]
    except InvalidScanInputError:
        pass

    if is_url_row:
        try:
            bl_result = calculate_blacklist_reputation_score(url, db)
            scores["blacklist_only"] = bl_result["score"]
        except InvalidScanInputError:
            pass
    else:
        try:
            lstm_result = calculate_lstm_phishing_score(text)
            scores["lstm_only"] = lstm_result["score"]
        except InvalidScanInputError:
            pass

    try:
        hybrid_result = calculate_final_risk_score(text=text, url=url, db=db)
        scores["full_hybrid"] = hybrid_result["final_score"]
    except (InvalidScanInputError, ValueError):
        pass

    return scores


def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """scores are 0-100 continuous. Binary prediction at BINARY_THRESHOLD."""
    y_pred = (scores >= BINARY_THRESHOLD).astype(int)
    probs = scores / 100.0

    metrics = {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    # ROC-AUC needs both classes present in y_true to be defined.
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, probs)
    else:
        metrics["roc_auc"] = None

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    metrics["y_true"] = y_true
    metrics["probs"] = probs
    return metrics


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_PATH)
    test_df = df[df["split"] == "test"].copy()
    print(f"Scoring {len(test_df)} TEST-split rows through all four configurations...")
    print("(never used during threshold calibration -- this is the unbiased held-out set)\n")

    db = SessionLocal()
    per_config_scores: dict[str, list] = {name: [] for name in CONFIG_NAMES}
    per_config_labels: dict[str, list] = {name: [] for name in CONFIG_NAMES}
    errors = 0
    start = time.time()

    for i, (_, row) in enumerate(test_df.iterrows()):
        row_scores = score_row(row, db)
        label = LABEL_MAP[row["label"]]

        for config_name in CONFIG_NAMES:
            score = row_scores[config_name]
            if score is not None:
                per_config_scores[config_name].append(score)
                per_config_labels[config_name].append(label)
            elif config_name == "full_hybrid":
                errors += 1  # full_hybrid should apply to every row; a None here is a real error

        if (i + 1) % 2000 == 0:
            elapsed = time.time() - start
            print(f"  ...{i + 1}/{len(test_df)} rows ({elapsed:.0f}s elapsed)")

    db.close()
    elapsed_total = time.time() - start
    print(f"\nDone in {elapsed_total:.0f}s. {errors} full_hybrid rows errored and were skipped.\n")

    # --- Compute metrics per config ---
    all_metrics: dict[str, dict] = {}
    for config_name in CONFIG_NAMES:
        y_true = np.array(per_config_labels[config_name])
        scores = np.array(per_config_scores[config_name])
        if len(y_true) == 0:
            print(f"WARNING: no applicable rows for {config_name}, skipping.")
            continue
        all_metrics[config_name] = compute_metrics(y_true, scores)

    # --- Print summary table ---
    print("=== Test-split results by configuration ===")
    header = f"{'Config':<18}{'N':>8}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}{'ROC-AUC':>9}"
    print(header)
    print("-" * len(header))
    for config_name in CONFIG_NAMES:
        if config_name not in all_metrics:
            continue
        m = all_metrics[config_name]
        auc_str = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "n/a"
        print(
            f"{CONFIG_DISPLAY_NAMES[config_name]:<18}{m['n']:>8}{m['accuracy']:>10.4f}"
            f"{m['precision']:>11.4f}{m['recall']:>9.4f}{m['f1']:>8.4f}{auc_str:>9}"
        )

    # --- Text report ---
    report_lines = []
    report_lines.append("ETIS -- Test-Split Configuration Comparison Report")
    report_lines.append(f"Total test-split rows: {len(test_df)}\n")
    report_lines.append(
        "Binary decision threshold for accuracy/precision/recall/F1: score >= "
        f"{BINARY_THRESHOLD:.0f} -> predicted phishing, applied UNIFORMLY across all "
        "four configs (see script docstring for why this differs from the hybrid's "
        "calibrated LOW/MEDIUM/HIGH bands).\n"
    )
    report_lines.append(header)
    report_lines.append("-" * len(header))
    for config_name in CONFIG_NAMES:
        if config_name not in all_metrics:
            continue
        m = all_metrics[config_name]
        auc_str = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "n/a"
        report_lines.append(
            f"{CONFIG_DISPLAY_NAMES[config_name]:<18}{m['n']:>8}{m['accuracy']:>10.4f}"
            f"{m['precision']:>11.4f}{m['recall']:>9.4f}{m['f1']:>8.4f}{auc_str:>9}"
        )
    report_lines.append("")
    report_lines.append("Confusion matrices (rows=true, cols=predicted, [[TN,FP],[FN,TP]]):")
    for config_name in CONFIG_NAMES:
        if config_name not in all_metrics:
            continue
        report_lines.append(f"\n{CONFIG_DISPLAY_NAMES[config_name]}:")
        report_lines.append(str(all_metrics[config_name]["confusion_matrix"]))
    report_lines.append("\n" + "=" * 70)
    report_lines.append("SCOPE CAVEATS -- include in Chapter 5.3/5.8, do not omit:")
    report_lines.append(
        "- Blacklist-only and LSTM-only are evaluated on DIFFERENT row subsets "
        "(URL rows only / text-email rows only respectively), not the full test "
        "set. Rule-based-only and full-hybrid ARE evaluated on the full test set. "
        "This table is not a clean apples-to-apples comparison without that caveat."
    )
    report_lines.append(
        "- Blacklist-only (and the blacklist component of full-hybrid) shares the "
        "same known limitation as threshold calibration: blacklist_entries was "
        "seeded from full PhishTank/OpenPhish feeds, not train-split-only, so some "
        "test-split phishing URLs may already sit in the blacklist. This inflates "
        "blacklist-driven recall via trivial exact-match rather than pure "
        "generalization to unseen phishing. Rule-based-only and LSTM-only do not "
        "have this issue."
    )
    report_lines.append(
        f"- Binary threshold (score >= {BINARY_THRESHOLD:.0f}) is a plain midpoint "
        "split chosen for threshold-neutral comparability across configs, not a "
        "calibrated decision boundary for any individual engine -- ROC-AUC is the "
        "primary metric that should carry weight in the writeup, not the F1/accuracy "
        "numbers in isolation."
    )

    with open(OUTPUT_DIR / "test_configuration_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\nSaved text report to {OUTPUT_DIR / 'test_configuration_report.txt'}")

    # --- Confusion matrices, 2x2 grid ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, config_name in zip(axes.flat, CONFIG_NAMES):
        if config_name not in all_metrics:
            ax.axis("off")
            continue
        cm = all_metrics[config_name]["confusion_matrix"]
        ConfusionMatrixDisplay(cm, display_labels=["legitimate", "phishing"]).plot(
            ax=ax, cmap="Blues", colorbar=False
        )
        ax.set_title(CONFIG_DISPLAY_NAMES[config_name])
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "test_confusion_matrices.png", dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrices to {OUTPUT_DIR / 'test_confusion_matrices.png'}")

    # --- ROC curves overlay ---
    fig, ax = plt.subplots(figsize=(6, 6))
    for config_name in CONFIG_NAMES:
        if config_name not in all_metrics or all_metrics[config_name]["roc_auc"] is None:
            continue
        m = all_metrics[config_name]
        # NOTE: `color=` is intentionally NOT passed to from_predictions() --
        # some scikit-learn versions don't forward it through to the
        # underlying matplotlib call and raise TypeError. Setting it on the
        # returned display's line_ afterwards works across versions.
        roc_display = RocCurveDisplay.from_predictions(
            m["y_true"],
            m["probs"],
            name=f"{CONFIG_DISPLAY_NAMES[config_name]} (AUC={m['roc_auc']:.3f})",
            ax=ax,
        )
        roc_display.line_.set_color(CONFIG_COLORS[config_name])
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title("ROC Curves by Configuration (Test Split)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "test_roc_curves.png", dpi=150)
    plt.close(fig)
    print(f"Saved ROC curve overlay to {OUTPUT_DIR / 'test_roc_curves.png'}")

    # --- Figure 5.5: grouped bar chart ---
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
    configs_present = [c for c in CONFIG_NAMES if c in all_metrics]

    x = np.arange(len(metric_keys))
    width = 0.8 / len(configs_present)

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, config_name in enumerate(configs_present):
        m = all_metrics[config_name]
        values = [m[k] for k in metric_keys]
        offset = (i - (len(configs_present) - 1) / 2) * width
        bars = ax.bar(
            x + offset, values, width,
            label=CONFIG_DISPLAY_NAMES[config_name],
            color=CONFIG_COLORS[config_name],
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Detection Method Comparison -- Test Split (Figure 5.5)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "test_configuration_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved Figure 5.5 comparison chart to {OUTPUT_DIR / 'test_configuration_comparison.png'}")

    print("\nDone. Remember to carry the SCOPE CAVEATS section of the text report")
    print("into Chapter 5.3 (Baseline Comparison) / 5.8 (Discussion) verbatim, not")
    print("just the numbers -- the subset-size and blacklist-leakage caveats change")
    print("how the numbers should be interpreted, especially blacklist-only's recall.")


if __name__ == "__main__":
    main()
