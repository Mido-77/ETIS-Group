"""
text_char_inference.py

Serves the selected message-text classifier: character n-gram TF-IDF
(char_wb, 2-5) + logistic regression, chosen on validation ROC-AUC in
ml/compare_text_models.py over word TF-IDF, LightGBM and the BiLSTM.
Measured on the held-out test split: F1 95.1% [91.5, 98.2], ROC-AUC 0.996;
paired F1 vs the BiLSTM +0.036 (CI excludes 0).

Return shape (a superset of lstm_inference.calculate_lstm_phishing_score):
    {
        "score": float,        # 0-100 (probability * 100)
        "probability": float,  # 0-1
        "tier": str,           # "HIGH" | "MEDIUM" | "LOW"
        "thresholds": {"medium": float, "high": float},
    }

Tier thresholds were chosen on the VALIDATION split at fixed false-positive
rates on legitimate messages -- 5% for MEDIUM, 1% for HIGH -- and are stored
inside the artifact by ml/export_models.py. Measured on the test split
(193 legitimate / 82 phishing, so 1% FPR is about two messages): HIGH flags
~84% of phishing at ~0% false positives; MEDIUM-or-higher ~96% at ~6%.

Unlike lstm_inference.py this module does NOT import TensorFlow: inference
takes about a millisecond instead of ~175 ms, and there is no Keras model
file to ship.

normalize_arabic_text is the same function as in ml/preprocess_lstm_data.py
and lstm_inference.py (third copy -- consolidate into one shared module when
the BiLSTM engine is retired).
"""

from pathlib import Path

import joblib
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import (
    normalize_alef_ar,
    normalize_alef_maksura_ar,
    normalize_teh_marbuta_ar,
)

from app.core.exceptions import InvalidScanInputError

PROJECT_ROOT = Path(__file__).resolve().parents[4]  # ml -> engines -> app -> backend -> root
ARTIFACT_PATH = PROJECT_ROOT / "ml" / "models" / "text_char_logreg.joblib"

_artifact = None


def _load_artifact() -> dict:
    global _artifact
    if _artifact is None:
        if not ARTIFACT_PATH.exists():
            raise FileNotFoundError(
                f"Text model not found at {ARTIFACT_PATH}. Run ml/export_models.py first."
            )
        _artifact = joblib.load(ARTIFACT_PATH)
    return _artifact


def normalize_arabic_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = dediac_ar(text)
    text = normalize_alef_ar(text)
    text = normalize_alef_maksura_ar(text)
    text = normalize_teh_marbuta_ar(text)
    return text


def calculate_text_char_phishing_score(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        raise InvalidScanInputError("Input text must be a non-empty string.")

    artifact = _load_artifact()
    thresholds = artifact.get("thresholds")
    if thresholds is None:
        raise RuntimeError(
            "Text model artifact has no tier thresholds -- re-run python ml/export_models.py."
        )
    features = artifact["vectorizer"].transform([normalize_arabic_text(text)])
    probability = float(artifact["classifier"].predict_proba(features)[0, 1])
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
        "thresholds": dict(thresholds),
    }
