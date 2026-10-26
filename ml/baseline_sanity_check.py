"""
baseline_sanity_check.py

Two cheap reference points to run BEFORE spending time tuning the LSTM:

1. Majority-class baseline -- what accuracy do you get by predicting the
   most common label every time, no model at all? Any real model needs to
   clearly beat this or it isn't learning anything useful.

2. TF-IDF + Logistic Regression baseline -- a classical, fast-to-train
   model on the same train/val/test split. If the LSTM can't outperform
   this, something is likely wrong in the pipeline (leakage, a bug, or
   the dataset genuinely being too small for a deep model to help), not
   just "needs more tuning."

This also directly produces numbers for thesis Chapter 5.3 (Baseline
Comparison), so it isn't throwaway work -- it belongs in the final report
either way.

Run from the project root:
    python ml/baseline_sanity_check.py
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from preprocess_lstm_data import CONTENT_TYPES, load_subset, normalize_arabic_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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

    print("=== Split label balance ===")
    for name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        print(f"  {name}: {dict(y.value_counts())}")
    print()

    # --- Baseline 1: majority class ---
    majority_label = y_train.mode()[0]
    majority_pred_val = [majority_label] * len(y_val)
    majority_pred_test = [majority_label] * len(y_test)

    print("=== Baseline 1: Majority-class prediction ===")
    print(f"  Always predicting: {'phishing' if majority_label == 1 else 'legitimate'}")
    print(f"  Val accuracy:  {accuracy_score(y_val, majority_pred_val):.4f}")
    print(f"  Test accuracy: {accuracy_score(y_test, majority_pred_test):.4f}")
    print("  (Any real model MUST clearly beat this, or it has learned nothing.)")
    print()

    # --- Baseline 2: TF-IDF + Logistic Regression ---
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df["clean_text"])
    X_val = vectorizer.transform(val_df["clean_text"])
    X_test = vectorizer.transform(test_df["clean_text"])

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)

    val_pred = clf.predict(X_val)
    val_proba = clf.predict_proba(X_val)[:, 1]
    test_pred = clf.predict(X_test)
    test_proba = clf.predict_proba(X_test)[:, 1]

    print("=== Baseline 2: TF-IDF + Logistic Regression ===")
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


if __name__ == "__main__":
    main()
