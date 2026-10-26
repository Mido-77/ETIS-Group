"""
compare_text_models.py

Step 4 of the evaluation rebuild: an honest model comparison for the
message-text classifier, on the new group-aware split.

Candidates (all trained on the TRAIN split only):
    word_tfidf_logreg   word TF-IDF + logistic regression (the existing baseline)
    char_tfidf_logreg   character n-gram TF-IDF + logistic regression
                        (suits Arabic dialect spelling variation better than words)
    word_tfidf_lightgbm word TF-IDF + LightGBM
    bilstm              the saved BiLSTM (ml/models/lstm_phishing_classifier.keras)
    ensemble_char_bilstm  mean of the char model and the BiLSTM scores (no fitting)

Selection rule (decided here, BEFORE looking at test): the candidate with the
highest VALIDATION ROC-AUC. The regularisation strength C of the two logistic
regressions is also tuned on validation only. Test metrics are computed for
every candidate for reporting, but nothing is chosen with them.

Reporting on the test split (N small, so every metric carries a 95% bootstrap
confidence interval -- 2,000 resamples, seed 42):
    - accuracy, precision, recall, F1, ROC-AUC with CIs
    - paired bootstrap of the DIFFERENCE vs the BiLSTM (does the interval
      contain 0? if so the models are not distinguishable at this test size)
    - recall by label_detail (phishing / scam / spam) and by source, so a
      weak corner is visible instead of averaged away

Positive class = every label_detail except legitimate (the binary `label`
column, as before). Synthetic rows can never be in test (create_grouped_split).

Run from the project root, after preprocess_lstm_data.py and train_lstm.py:
    python ml/compare_text_models.py

Outputs:
    ml/results/model_comparison.txt
    ml/results/model_comparison_test_predictions.csv  (id + each model's score)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess_lstm_data import load_subset, normalize_arabic_text  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "ml" / "data"
MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "lstm_phishing_classifier.keras"
RESULTS_DIR = PROJECT_ROOT / "ml" / "results"

SEED = 42
N_BOOTSTRAP = 2000
C_GRID = [0.3, 1.0, 3.0, 10.0, 30.0]


# ---------------------------------------------------------------------------
# Candidates -- each returns (validation_scores, test_scores) in [0, 1]
# ---------------------------------------------------------------------------

def _logreg_candidate(vectoriser_factory, train_text, y_train, val_text, y_val, test_text):
    vec = vectoriser_factory()
    X_train = vec.fit_transform(train_text)
    X_val, X_test = vec.transform(val_text), vec.transform(test_text)

    best_auc, best_c = -1.0, None
    for c in C_GRID:  # tune C on validation only
        clf = LogisticRegression(C=c, max_iter=2000, class_weight="balanced")
        clf.fit(X_train, y_train)
        auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
        if auc > best_auc:
            best_auc, best_c = auc, c
    clf = LogisticRegression(C=best_c, max_iter=2000, class_weight="balanced").fit(X_train, y_train)
    return clf.predict_proba(X_val)[:, 1], clf.predict_proba(X_test)[:, 1], f"C={best_c}"


def word_tfidf_logreg(tr, y_tr, va, y_va, te):
    return _logreg_candidate(
        lambda: TfidfVectorizer(max_features=5000), tr, y_tr, va, y_va, te)


def char_tfidf_logreg(tr, y_tr, va, y_va, te):
    return _logreg_candidate(
        lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                sublinear_tf=True, max_features=60000),
        tr, y_tr, va, y_va, te)


def word_tfidf_lightgbm(tr, y_tr, va, y_va, te):
    import lightgbm as lgb  # optional dependency
    vec = TfidfVectorizer(max_features=5000)
    X_train = vec.fit_transform(tr)
    clf = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10,
        subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
        random_state=SEED, verbose=-1,
    )
    clf.fit(X_train, y_tr)
    return clf.predict_proba(vec.transform(va))[:, 1], clf.predict_proba(vec.transform(te))[:, 1], "300 trees"


def bilstm_saved(*_unused):
    from tensorflow.keras.models import load_model  # imported lazily
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found -- run ml/train_lstm.py first")
    model = load_model(MODEL_PATH)
    X_val, X_test = np.load(DATA_DIR / "X_val.npy"), np.load(DATA_DIR / "X_test.npy")
    return (model.predict(X_val, verbose=0).ravel(),
            model.predict(X_test, verbose=0).ravel(), "saved model")


# ---------------------------------------------------------------------------
# Metrics + bootstrap
# ---------------------------------------------------------------------------

def metric_row(y, score, threshold=0.5) -> dict:
    pred = (score >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, score),
    }


def bootstrap_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=(N_BOOTSTRAP, n))


def bootstrap_ci(y, score, idx) -> dict:
    samples = {k: [] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}
    for row in idx:
        yb, sb = y[row], score[row]
        if yb.min() == yb.max():  # resample with one class only: skip (AUC undefined)
            continue
        for k, v in metric_row(yb, sb).items():
            samples[k].append(v)
    return {k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in samples.items()}


def paired_difference_ci(y, score_a, score_b, idx, key) -> tuple[float, float, float]:
    """CI of metric(a) - metric(b) on the SAME resampled rows."""
    diffs = []
    for row in idx:
        yb = y[row]
        if yb.min() == yb.max():
            continue
        diffs.append(metric_row(yb, score_a[row])[key] - metric_row(yb, score_b[row])[key])
    point = metric_row(y, score_a)[key] - metric_row(y, score_b)[key]
    return point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main() -> None:
    subset = load_subset()
    subset["clean_text"] = subset["text"].apply(normalize_arabic_text)
    label_map = {"legitimate": 0, "phishing": 1}

    train_df = subset[subset["split"] == "train"]
    val_df = subset[subset["split"] == "val"].reset_index(drop=True)
    test_df = subset[subset["split"] == "test"].reset_index(drop=True)
    y_train = train_df["label"].map(label_map).to_numpy()
    y_val = val_df["label"].map(label_map).to_numpy()
    y_test = test_df["label"].map(label_map).to_numpy()

    candidates = [
        ("word_tfidf_logreg", word_tfidf_logreg),
        ("char_tfidf_logreg", char_tfidf_logreg),
        ("word_tfidf_lightgbm", word_tfidf_lightgbm),
        ("bilstm", bilstm_saved),
    ]

    val_scores, test_scores, notes = {}, {}, {}
    lines = ["ETIS text-model comparison (group-aware split)",
             f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}  "
             f"(test positives={int(y_test.sum())})", ""]

    for name, fn in candidates:
        try:
            v, t, note = fn(train_df["clean_text"], y_train, val_df["clean_text"], y_val,
                            test_df["clean_text"]) if name != "bilstm" else fn()
        except ImportError as e:
            lines.append(f"[skipped] {name}: {e}")
            continue
        except FileNotFoundError as e:
            lines.append(f"[skipped] {name}: {e}")
            continue
        val_scores[name], test_scores[name], notes[name] = np.asarray(v), np.asarray(t), note

    # ---- ensemble candidate: plain mean of the char model and the BiLSTM ----
    # (no fitting involved, so nothing is tuned on test; it competes under
    # the same validation rule as every other candidate)
    if "char_tfidf_logreg" in val_scores and "bilstm" in val_scores:
        name = "ensemble_char_bilstm"
        val_scores[name] = (val_scores["char_tfidf_logreg"] + val_scores["bilstm"]) / 2
        test_scores[name] = (test_scores["char_tfidf_logreg"] + test_scores["bilstm"]) / 2
        notes[name] = "mean of char LogReg + BiLSTM"

    # ---- selection on VALIDATION only ----
    lines += ["", "== Validation (used for selection and for tuning C) =="]
    lines.append(f"{'model':<22}{'val ROC-AUC':>12}{'val F1':>9}{'val recall':>12}  note")
    val_auc = {}
    for name in val_scores:
        m = metric_row(y_val, val_scores[name])
        val_auc[name] = m["roc_auc"]
        lines.append(f"{name:<22}{m['roc_auc']:>12.4f}{m['f1']:>9.4f}{m['recall']:>12.4f}  {notes[name]}")
    selected = max(val_auc, key=val_auc.get)
    lines += ["", f"SELECTED on validation ROC-AUC: {selected}", ""]

    # ---- test reporting with CIs ----
    rng = np.random.default_rng(SEED)
    idx = bootstrap_indices(len(y_test), rng)
    lines.append("== Test (95% bootstrap CI, 2,000 resamples) ==")
    for name in test_scores:
        m = metric_row(y_test, test_scores[name])
        ci = bootstrap_ci(y_test, test_scores[name], idx)
        tag = "  <- selected" if name == selected else ""
        lines.append(f"{name}{tag}")
        for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            lines.append(f"    {k:<10}{m[k]:.4f}   [{ci[k][0]:.4f}, {ci[k][1]:.4f}]")

    if "bilstm" in test_scores:
        lines += ["", "== Paired difference vs BiLSTM (positive = model beats BiLSTM) =="]
        for name in test_scores:
            if name == "bilstm":
                continue
            for key in ("f1", "roc_auc"):
                d, lo, hi = paired_difference_ci(y_test, test_scores[name], test_scores["bilstm"], idx, key)
                verdict = "distinguishable" if (lo > 0 or hi < 0) else "NOT distinguishable (CI contains 0)"
                lines.append(f"  {name:<22}{key:<8}{d:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {verdict}")

    # ---- where does each model miss? ----
    lines += ["", "== Test recall by label_detail (positives only) =="]
    pos = test_df["label"] == "phishing"
    for detail in sorted(test_df.loc[pos, "label_detail"].unique()):
        rows = (test_df["label_detail"] == detail).to_numpy()
        cells = ", ".join(
            f"{n}={((test_scores[n][rows] >= 0.5).mean()):.2f}" for n in test_scores)
        lines.append(f"  {detail:<10} n={int(rows.sum()):<4} {cells}")

    lines += ["", "== Test accuracy by source =="]
    for source in sorted(test_df["source"].unique()):
        rows = (test_df["source"] == source).to_numpy()
        accs = ", ".join(
            f"{n}={(((test_scores[n][rows] >= 0.5).astype(int)) == y_test[rows]).mean():.2f}"
            for n in test_scores)
        lines.append(f"  {source:<40} n={int(rows.sum()):<4} {accs}")

    lines += ["", "== Test accuracy by language =="]
    for language in sorted(test_df["language"].unique()):
        rows = (test_df["language"] == language).to_numpy()
        accs = ", ".join(
            f"{n}={(((test_scores[n][rows] >= 0.5).astype(int)) == y_test[rows]).mean():.2f}"
            for n in test_scores)
        lines.append(f"  {language:<8} n={int(rows.sum()):<4} {accs}")

    report = "\n".join(lines)
    print(report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "model_comparison.txt").write_text(report + "\n", encoding="utf-8")
    out = pd.DataFrame({"id": test_df["id"], "label": y_test, **test_scores})
    out.to_csv(RESULTS_DIR / "model_comparison_test_predictions.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
