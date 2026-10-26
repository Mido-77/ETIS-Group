"""
export_models.py

Trains the two models selected in Chapter 5 and saves them as deployable
artifacts for the backend:

    ml/models/text_char_logreg.joblib   character TF-IDF + logistic regression
                                        (message text; replaces the BiLSTM as the live text model)
    ml/models/url_ensemble.joblib       char n-gram LogReg + LightGBM lexical model,
                                        mean of the two, with MEDIUM/HIGH tier thresholds

Both are fitted on the TRAIN split only, with hyper-parameters (C) and the URL
tier thresholds chosen on the VALIDATION split -- exactly the protocol of
ml/compare_text_models.py and ml/train_url_model.py, so the numbers reported
in the thesis describe these files. (A final model refitted on train+val
would be slightly stronger but would no longer match the reported
evaluation; that is future work.)

Self-check (on by default): the backend inference modules are called on test
rows and their scores are compared with the scores saved by the evaluation
scripts (ml/results/*_test_predictions.csv). This proves the deployed code
path reproduces the evaluated models. It fails loudly if feature extraction
or preprocessing has drifted.

URL model variants (--variant):
    v1  (default for now)  TLD visible: the model deployed until the TLD-shortcut finding
                           (evaluation/test_tld_shortcut.py). Reproducible exactly: the
                           ablation's baseline check reproduced the deployed artifact
                           with a maximum score difference of 0.0.
    v2                     TLD masked: the character model reads the last host label as
                           "tld" and the LightGBM TLD columns are dropped
                           (ml/ablation_url_tld_mask.py; criteria C1-C4 met). The artifact
                           carries "tld_masked": True and the serving code applies the same
                           masking. The self-check for v2 compares the served scores with
                           the ablation's scoring code applied to the NEW artifact, and the
                           new artifact with the ablation's masked artifact (ml/ablation/),
                           i.e. it proves the export reproduces the model that was evaluated.

Run from the project root, after the comparison scripts:
    python ml/export_models.py
    python ml/export_models.py --variant v2 --only url      (rebuild ONLY the URL model; the text artifact is not touched)
    python ml/export_models.py --skip-selfcheck
"""

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from preprocess_lstm_data import load_subset, normalize_arabic_text  # noqa: E402
from app.engines.ml.url_features import (  # noqa: E402
    HOST_FEATURES, PATH_FEATURES, TOP_TLDS, build_matrix, char_model_text, feature_frame, mask_tld_in_text,
)

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
RESULTS_DIR = PROJECT_ROOT / "ml" / "results"
TEXT_ARTIFACT = MODELS_DIR / "text_char_logreg.joblib"
URL_ARTIFACT = MODELS_DIR / "url_ensemble.joblib"

SEED = 42
SYNTHETIC_SOURCE = "synthetic_egyptian_brands"
TEXT_C_GRID = [0.3, 1.0, 3.0, 10.0, 30.0]
URL_C_GRID = [1.0, 10.0]
URL_MEDIUM_FPR = 0.01   # MEDIUM tier: top 1% of legitimate validation URLs
URL_HIGH_FPR = 0.001    # HIGH tier: top 0.1%
TEXT_MEDIUM_FPR = 0.05  # text MEDIUM tier: top 5% of legitimate validation messages
TEXT_HIGH_FPR = 0.01    # text HIGH tier: top 1%
SELFCHECK_URL_SAMPLE = 2000


# ---------------------------------------------------------------------------
# Text model
# ---------------------------------------------------------------------------

def train_text_model() -> dict:
    subset = load_subset()
    subset["clean_text"] = subset["text"].apply(normalize_arabic_text)
    label_map = {"legitimate": 0, "phishing": 1}
    tr, va = subset[subset["split"] == "train"], subset[subset["split"] == "val"]
    y_tr, y_va = tr["label"].map(label_map).to_numpy(), va["label"].map(label_map).to_numpy()

    vectoriser = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                 sublinear_tf=True, max_features=60000)
    X_tr = vectoriser.fit_transform(tr["clean_text"])
    X_va = vectoriser.transform(va["clean_text"])

    best_auc, best_c = -1.0, None
    for c in TEXT_C_GRID:
        clf = LogisticRegression(C=c, max_iter=2000, class_weight="balanced").fit(X_tr, y_tr)
        auc = roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1])
        if auc > best_auc:
            best_auc, best_c = auc, c
    classifier = LogisticRegression(C=best_c, max_iter=2000, class_weight="balanced").fit(X_tr, y_tr)
    p_val = classifier.predict_proba(X_va)[:, 1]
    # Thresholds are estimated on REAL validation messages only: the synthetic
    # rows are excluded from every reported evaluation (evaluate_fusion.py, the
    # test split), so including their hard legitimate examples here would make
    # the deployed tiers differ from the ones that were evaluated. Re-estimate
    # on the collected real messages when they arrive.
    is_real = (va["source"] != SYNTHETIC_SOURCE).to_numpy()
    legit_val = p_val[(y_va == 0) & is_real]
    thresholds = {
        "medium": float(np.quantile(legit_val, 1 - TEXT_MEDIUM_FPR)),
        "high": float(np.quantile(legit_val, 1 - TEXT_HIGH_FPR)),
    }
    print(f"Text model: C={best_c}, validation ROC-AUC={best_auc:.4f}, train rows={len(tr)}, "
          f"thresholds medium={thresholds['medium']:.3f} high={thresholds['high']:.3f}")
    return {
        "vectorizer": vectoriser, "classifier": classifier, "c": best_c,
        "thresholds": thresholds,
        "validation_roc_auc": best_auc, "trained_on": "train split only",
        "created": date.today().isoformat(),
    }


# ---------------------------------------------------------------------------
# URL ensemble
# ---------------------------------------------------------------------------

def train_url_model(masked: bool = False) -> dict:
    import lightgbm as lgb

    df = pd.read_csv(DATASET_PATH)
    urls = df[df["content_type"] == "url"].reset_index(drop=True)
    y = (urls["label"] == "phishing").astype(int).to_numpy()
    tr = (urls["split"] == "train").to_numpy()
    va = (urls["split"] == "val").to_numpy()

    feats = feature_frame(urls["text"].tolist())
    char_text = [char_model_text(u) for u in urls["text"]]
    if masked:
        char_text = [mask_tld_in_text(t) for t in char_text]
    tld_categories = [] if masked else feats.loc[tr, "tld"].value_counts().head(TOP_TLDS).index.tolist()

    vectoriser = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=3,
                                 sublinear_tf=True, max_features=200_000)
    X_tr = vectoriser.fit_transform([t for t, m in zip(char_text, tr) if m])
    X_va = vectoriser.transform([t for t, m in zip(char_text, va) if m])
    y_tr, y_va = y[tr], y[va]

    best_c = max(URL_C_GRID, key=lambda c: roc_auc_score(
        y_va, LogisticRegression(C=c, max_iter=300, class_weight="balanced").fit(X_tr, y_tr).predict_proba(X_va)[:, 1]))
    char_logreg = LogisticRegression(C=best_c, max_iter=300, class_weight="balanced").fit(X_tr, y_tr)

    columns = HOST_FEATURES + PATH_FEATURES
    X_all = build_matrix(feats, columns, tld_categories)
    if masked:
        X_all = X_all[columns]          # drops the constant tld_other column
    lightgbm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                  subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbose=-1)
    lightgbm.fit(X_all[tr], y_tr)

    p_val = (char_logreg.predict_proba(X_va)[:, 1] + lightgbm.predict_proba(X_all[va])[:, 1]) / 2
    legit_val = p_val[y_va == 0]
    thresholds = {
        "medium": float(np.quantile(legit_val, 1 - URL_MEDIUM_FPR)),
        "high": float(np.quantile(legit_val, 1 - URL_HIGH_FPR)),
    }
    print(f"URL ensemble ({'v2, TLD masked' if masked else 'v1, TLD visible'}): C={best_c}, validation ROC-AUC={roc_auc_score(y_va, p_val):.4f}, "
          f"train rows={int(tr.sum())}, thresholds medium={thresholds['medium']:.3f} high={thresholds['high']:.3f}")
    return {
        "vectorizer": vectoriser, "char_logreg": char_logreg, "lightgbm": lightgbm,
        "lgbm_columns": columns, "tld_categories": tld_categories, "thresholds": thresholds,
        "c": best_c, "tld_masked": masked, "trained_on": "train split only", "created": date.today().isoformat(),
    }


# ---------------------------------------------------------------------------
# Self-check: deployed code path vs saved evaluation scores
# ---------------------------------------------------------------------------

def selfcheck_text() -> bool:
    from app.engines.ml.text_char_inference import calculate_text_char_phishing_score

    reference_path = RESULTS_DIR / "model_comparison_test_predictions.csv"
    if not reference_path.exists():
        print("  [skip] text: ml/results/model_comparison_test_predictions.csv not found")
        return True
    reference = pd.read_csv(reference_path)
    if "char_tfidf_logreg" not in reference.columns:
        print("  [skip] text: reference file has no char_tfidf_logreg column")
        return True
    subset = load_subset().set_index("id")
    served = np.array([calculate_text_char_phishing_score(subset.loc[i, "text"])["probability"]
                       for i in reference["id"]])
    diff = np.abs(served - reference["char_tfidf_logreg"].to_numpy())
    ok = bool(diff.max() < 1e-4)
    print(f"  text:  {len(diff)} test rows, max |served - evaluated| = {diff.max():.2e}  -> {'PASS' if ok else 'FAIL'}")
    return ok


def selfcheck_url() -> bool:
    from app.engines.ml.url_model_inference import calculate_url_model_score

    reference_path = RESULTS_DIR / "url_model_test_predictions.csv"
    if not reference_path.exists():
        print("  [skip] url: ml/results/url_model_test_predictions.csv not found")
        return True
    reference = pd.read_csv(reference_path)
    if "ensemble_full" not in reference.columns:
        print("  [skip] url: reference file has no ensemble_full column")
        return True
    sample = reference.sample(n=min(SELFCHECK_URL_SAMPLE, len(reference)), random_state=SEED)
    texts = pd.read_csv(DATASET_PATH, usecols=["id", "text"]).set_index("id")
    start = time.time()
    served = np.array([calculate_url_model_score(texts.loc[i, "text"])["probability"] for i in sample["id"]])
    per_call_ms = 1000 * (time.time() - start) / len(sample)
    diff = np.abs(served - sample["ensemble_full"].to_numpy())
    ok = bool(diff.max() < 0.02)
    print(f"  url:   {len(diff)} test rows, max |served - evaluated| = {diff.max():.2e}, "
          f"mean = {diff.mean():.2e}, {per_call_ms:.1f} ms/URL  -> {'PASS' if ok else 'FAIL'}")
    return ok


def selfcheck_url_masked() -> bool:
    """v2 self-check. There is no saved evaluation file for the masked model, so the reference is the
    evaluation code itself: ml/ablation_url_tld_mask.py::score_urls (which produced the reported numbers)."""
    import joblib as _joblib
    import ablation_url_tld_mask as ablation
    from app.engines.ml.url_model_inference import calculate_url_model_score

    new_artifact = _joblib.load(URL_ARTIFACT)
    if not new_artifact.get("tld_masked"):
        print("  url:   FAIL: the exported artifact is not flagged tld_masked")
        return False
    texts = pd.read_csv(DATASET_PATH, usecols=["id", "text", "content_type", "split"])
    sample = texts[(texts["content_type"] == "url") & (texts["split"] == "test")]
    sample = sample.sample(n=min(SELFCHECK_URL_SAMPLE, len(sample)), random_state=SEED)["text"].tolist()

    start = time.time()
    served = np.array([calculate_url_model_score(u)["probability"] for u in sample])
    per_call_ms = 1000 * (time.time() - start) / len(sample)
    evaluated = ablation.score_urls(new_artifact, sample)
    diff_a = float(np.abs(served - evaluated).max())
    ok_a = diff_a < 1e-6
    print(f"  url:   serving code vs evaluation scoring code on {len(sample)} test URLs: max |difference| = {diff_a:.2e}, "
          f"{per_call_ms:.1f} ms/URL  -> {'PASS' if ok_a else 'FAIL'}")

    if not ablation.MASKED_ARTIFACT.exists():
        print("  [skip] url: ml/ablation/url_ensemble_tldmasked.joblib not found; cannot compare with the evaluated model")
        return ok_a
    reference = _joblib.load(ablation.MASKED_ARTIFACT)
    diff_b = float(np.abs(evaluated - ablation.score_urls(reference, sample)).max())
    thr_diff = max(abs(new_artifact["thresholds"][k] - reference["thresholds"][k]) for k in ("medium", "high"))
    ok_b = diff_b < 0.02 and thr_diff < 1e-3
    print(f"  url:   new artifact vs the evaluated ablation artifact: max score difference = {diff_b:.2e}, "
          f"max threshold difference = {thr_diff:.2e}  -> {'PASS' if ok_b else 'FAIL'}")
    return ok_a and ok_b


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-selfcheck", action="store_true")
    parser.add_argument("--variant", choices=["v1", "v2"], default="v1",
                        help="URL model: v1 = TLD visible (deployed so far), v2 = TLD masked")
    parser.add_argument("--only", choices=["url", "text"], default=None,
                        help="rebuild only one artifact (default: both); the other file is not touched")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    do_text, do_url = args.only in (None, "text"), args.only in (None, "url")

    if do_text:
        joblib.dump(train_text_model(), TEXT_ARTIFACT, compress=3)
    if do_url:
        joblib.dump(train_url_model(masked=(args.variant == "v2")), URL_ARTIFACT, compress=3)
    for path, built in ((TEXT_ARTIFACT, do_text), (URL_ARTIFACT, do_url)):
        if built:
            print(f"Saved {path.relative_to(PROJECT_ROOT)}  ({path.stat().st_size / 1024:.0f} KB)")

    if args.skip_selfcheck:
        return
    print("\nSelf-check (deployed inference code vs saved evaluation scores):")
    url_ok = (selfcheck_url() if args.variant == "v1" else selfcheck_url_masked()) if do_url else True
    text_ok = selfcheck_text() if do_text else True
    if not (text_ok and url_ok):
        sys.exit("SELF-CHECK FAILED -- do not commit these artifacts; the deployed path does not reproduce the evaluation.")
    print("Self-check passed.")


if __name__ == "__main__":
    main()
