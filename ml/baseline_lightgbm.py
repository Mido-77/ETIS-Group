"""
baseline_lightgbm.py

Second ML baseline (Master Plan Section 5 "worth adding" item): LightGBM
on the SAME TF-IDF features already used for the TF-IDF + Logistic
Regression baseline in baseline_sanity_check.py. This gives Chapter 5.3
a stronger "why LSTM over classical ML" comparison than the Logistic
Regression baseline alone -- LightGBM is a considerably stronger
classical learner, so if the LSTM still wins (or ties), that's a more
convincing result than beating plain LogReg.

Deliberately reuses preprocess_lstm_data's load_subset()/normalize_arabic_text()
and the same train/val/test split (seed 42, stratified) as every other
evaluation script in this project, so results are directly comparable
across baseline_sanity_check.py, train_lstm.py, and this script.

Run from the project root (after preprocess_lstm_data.py has been run,
same precondition as baseline_sanity_check.py):
    python ml/baseline_lightgbm.py

Requires: pip install lightgbm --break-system-packages (if not already installed)
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from lightgbm import LGBMClassifier
except ImportError as exc:
    raise SystemExit(
        "LightGBM is not installed. Run:\n"
        "    pip install lightgbm --break-system-packages\n"
        "then re-run this script."
    ) from exc

from preprocess_lstm_data import load_subset, normalize_arabic_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Same vocabulary size as the TF-IDF + Logistic Regression baseline, so the
# two classical baselines are compared on identical input features -- only
# the classifier differs.
TFIDF_MAX_FEATURES = 5000


def main() -> None:
    subset = load_subset()
    subset["clean_text"] = subset["text"].apply(normalize_arabic_text)

    train_df = subset[subset["split"] == "train"]
    val_df = subset[subset["split"] == "val"]
    test_df = subset[subset["split"] == "test"]

    label_map = {"legitimate": 0, "phishing": 1}
    y_train = train_df["label"].map(label_map)
    y_val = val_df["label"].map(label_map)
    y_test = test_df["label"].map(label_map)

    # --- Same TF-IDF vectorizer setup as baseline_sanity_check.py ---
    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
    X_train = vectorizer.fit_transform(train_df["clean_text"])
    X_val = vectorizer.transform(val_df["clean_text"])
    X_test = vectorizer.transform(test_df["clean_text"])

    # class_weight="balanced" mirrors the same imbalance-handling choice
    # used for both the Logistic Regression baseline and the LSTM's
    # compute_class_weight("balanced", ...) call, so all three models are
    # compared under the same imbalance-correction policy, not just the
    # same features.
    clf = LGBMClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    )
    clf.fit(X_train, y_train)

    val_pred = clf.predict(X_val)
    val_proba = clf.predict_proba(X_val)[:, 1]
    test_pred = clf.predict(X_test)
    test_proba = clf.predict_proba(X_test)[:, 1]

    print("=== Baseline: TF-IDF + LightGBM ===")
    print("-- Validation --")
    print(f"  Accuracy:  {accuracy_score(y_val, val_pred):.4f}")
    print(f"  Precision: {precision_score(y_val, val_pred):.4f}")
    print(f"  Recall:    {recall_score(y_val, val_pred):.4f}")
    print(f"  F1:        {f1_score(y_val, val_pred):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_val, val_proba):.4f}")
    print()
    print("-- Test --")
    print(f"  Accuracy:  {accuracy_score(y_test, test_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, test_pred):.4f}")
    print(f"  Recall:    {recall_score(y_test, test_pred):.4f}")
    print(f"  F1:        {f1_score(y_test, test_pred):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, test_proba):.4f}")
    print()
    print(classification_report(y_test, test_pred, target_names=["legitimate", "phishing"]))

    print("Reference points (same split, from other evaluation scripts):")
    print("  Majority-class:            69.44% test accuracy")
    print("  TF-IDF + Logistic Reg.:    94.68% test accuracy, 98.47% ROC-AUC")
    print("  BiLSTM (this project):     96.01% test accuracy, 96.39% ROC-AUC")

    # Persist to results/, matching train_lstm.py's convention, so this
    # slots straight into Chapter 5.3's Table 5.1 alongside the other rows.
    results_dir = PROJECT_ROOT / "ml" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "lightgbm_baseline_metrics.txt", "w") as f:
        f.write("TF-IDF + LightGBM baseline\n\n")
        f.write("-- Validation --\n")
        f.write(f"Accuracy:  {accuracy_score(y_val, val_pred):.4f}\n")
        f.write(f"Precision: {precision_score(y_val, val_pred):.4f}\n")
        f.write(f"Recall:    {recall_score(y_val, val_pred):.4f}\n")
        f.write(f"F1:        {f1_score(y_val, val_pred):.4f}\n")
        f.write(f"ROC-AUC:   {roc_auc_score(y_val, val_proba):.4f}\n\n")
        f.write("-- Test --\n")
        f.write(f"Accuracy:  {accuracy_score(y_test, test_pred):.4f}\n")
        f.write(f"Precision: {precision_score(y_test, test_pred):.4f}\n")
        f.write(f"Recall:    {recall_score(y_test, test_pred):.4f}\n")
        f.write(f"F1:        {f1_score(y_test, test_pred):.4f}\n")
        f.write(f"ROC-AUC:   {roc_auc_score(y_test, test_proba):.4f}\n\n")
        f.write(classification_report(y_test, test_pred, target_names=["legitimate", "phishing"]))
    print(f"\nSaved metrics to {results_dir / 'lightgbm_baseline_metrics.txt'}")


if __name__ == "__main__":
    main()
