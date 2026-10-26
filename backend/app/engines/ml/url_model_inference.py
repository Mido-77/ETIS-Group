"""
url_model_inference.py

Serves the learned URL detector: the mean of a character n-gram TF-IDF
logistic regression and a LightGBM lexical-feature model (ensemble_full in
ml/train_url_model.py). It is the only part of ETIS that can flag a phishing
URL on a host it has never seen.

Output matches the other engines' dict shape:
    {
        "score": float,        # 0-100 (probability * 100)
        "probability": float,  # 0-1, mean of the two models
        "tier": str,           # "HIGH" | "MEDIUM" | "LOW"
        "components": {"char_logreg": float, "lightgbm": float},
        "thresholds": {"medium": float, "high": float},
    }

Tier thresholds were chosen on the VALIDATION split at fixed false-positive
rates of 1% (MEDIUM) and 0.1% (HIGH) of legitimate URLs, and stored inside
the artifact (ml/models/url_ensemble.joblib, built by ml/export_models.py).
Measured on unseen test hosts for the deployed v2 (TLD-masked) artifact:
MEDIUM-or-higher flags ~93% of phishing at ~0.8% false positives; HIGH flags
~86% at ~0.0% (ml/results/url_tld_ablation_report.txt and
ml/results/url_high_tier_check_report.txt; see Chapter 5).

TLD masking (v2): an artifact trained with tld_masked=True (ml/export_models.py
--variant v2) reads the host WITHOUT its last label: the character model sees
"tld" in its place and the LightGBM TLD columns are dropped. The reason is the
finding in evaluation/test_tld_shortcut.py: the v1 model treated ".eg" as a
sign of legitimacy (phishing recall 93% -> 25% when only the last label was
swapped to .eg); the masked model loses 1 point. Artifacts without the
"tld_masked" key (v1) are scored exactly as before. The v1 model reached ~66%
at HIGH on the same URLs (same MEDIUM figures).

KNOWN LIMITS: the legitimate training URLs are Tranco top-20k homepages plus
crawled internal pages, so precision on obscure legitimate sites is not
measured, and URLs with a path have a higher false-positive rate (~2.8% at
the MEDIUM threshold) than bare hosts (~0%).
"""

from pathlib import Path

import joblib

from app.core.exceptions import InvalidScanInputError
from app.engines.ml.url_features import build_matrix, char_model_text, feature_frame, mask_tld_in_text

PROJECT_ROOT = Path(__file__).resolve().parents[4]  # ml -> engines -> app -> backend -> root
ARTIFACT_PATH = PROJECT_ROOT / "ml" / "models" / "url_ensemble.joblib"

_artifact = None


def ensure_scheme(url: str) -> str:
    """Give a scheme-less URL the scheme the model was trained on.

    Every URL in the training data carries http:// or https:// (99.1% of the
    legitimate and 94.3% of the phishing rows are https). A URL typed or pasted
    as "wikipedia.org" is therefore outside the training format: without a
    scheme the is_https feature is 0, which the model reads as the (more
    phishing-like) http case. Normalising here is serving-time only; training
    and the saved evaluation scores are untouched.
    """
    raw = url.strip()
    return raw if "://" in raw else "https://" + raw


def _load_artifact() -> dict:
    global _artifact
    if _artifact is None:
        if not ARTIFACT_PATH.exists():
            raise FileNotFoundError(
                f"URL model not found at {ARTIFACT_PATH}. Run ml/export_models.py first."
            )
        _artifact = joblib.load(ARTIFACT_PATH)
    return _artifact


def calculate_url_model_score(url: str) -> dict:
    if not isinstance(url, str) or not url.strip():
        raise InvalidScanInputError("Input must be a non-empty string.")

    artifact = _load_artifact()
    raw = ensure_scheme(url)

    tld_masked = bool(artifact.get("tld_masked", False))
    char_text = char_model_text(raw)
    if tld_masked:
        char_text = mask_tld_in_text(char_text)
    p_char = float(artifact["char_logreg"].predict_proba(
        artifact["vectorizer"].transform([char_text]))[0, 1])
    features = build_matrix(feature_frame([raw]), artifact["lgbm_columns"], artifact["tld_categories"])
    if tld_masked:
        features = features[artifact["lgbm_columns"]]   # drops the constant tld_other column
    p_lgbm = float(artifact["lightgbm"].predict_proba(features)[0, 1])

    probability = (p_char + p_lgbm) / 2
    thresholds = artifact["thresholds"]
    if probability >= thresholds["high"]:
        tier = "HIGH"
    elif probability >= thresholds["medium"]:
        tier = "MEDIUM"
    else:
        tier = "LOW"

    return {
        "score": round(probability * 100, 2),
        "probability": probability,
        "tier": tier,
        "components": {"char_logreg": p_char, "lightgbm": p_lgbm},
        "thresholds": dict(thresholds),
    }
