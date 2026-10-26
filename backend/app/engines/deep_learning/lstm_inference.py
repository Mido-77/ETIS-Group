"""
lstm_inference.py

Wraps the trained BiLSTM (ml/train_lstm.py) for single-submission inference,
matching the other engines' dict-return shape so risk_scoring.py can treat
all three engines uniformly.

Model and tokenizer are loaded once at module import time, not per-call --
loading a Keras model on every request would be needlessly slow.

KNOWN GAP (flagged 2026-07-28, unresolved): the model file
(ml/models/lstm_phishing_classifier.keras) and tokenizer
(ml/data/tokenizer.pkl) both currently live inside .gitignore'd
directories. That's fine for local development, where these files already
exist on disk, but it means a fresh clone or a hosted deployment (Railway/
Render) will NOT have these files unless a deliberate decision is made to
commit them as tracked artifacts (they are not bit-identical-reproducible
via retraining, unlike ml/data/*.npy, since weight initialization is
random). This needs to be resolved before Phase 8 (deployment) -- see
CLAUDE.md.

Preprocessing note: normalize_arabic_text is duplicated here from
ml/preprocess_lstm_data.py rather than imported, since ml/ and backend/app/
are separate top-level packages without a shared import path today. This
is a conscious, small duplication (6 lines), not an oversight -- worth
consolidating into a shared module if the two ever drift out of sync.
"""

import pickle
from pathlib import Path

from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import (
    normalize_alef_ar,
    normalize_alef_maksura_ar,
    normalize_teh_marbuta_ar,
)
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from app.core.exceptions import InvalidScanInputError

PROJECT_ROOT = Path(__file__).resolve().parents[4]  # deep_learning -> engines -> app -> backend -> root
MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "lstm_phishing_classifier.keras"
TOKENIZER_PATH = PROJECT_ROOT / "ml" / "data" / "tokenizer.pkl"
MAX_LEN = 200

_model = None
_tokenizer = None


def _load_artifacts():
    global _model, _tokenizer
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"LSTM model not found at {MODEL_PATH}. Run ml/train_lstm.py first, "
                "or see the artifact-storage gap noted in this module's docstring."
            )
        _model = load_model(MODEL_PATH)
    if _tokenizer is None:
        if not TOKENIZER_PATH.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {TOKENIZER_PATH}. Run ml/preprocess_lstm_data.py first."
            )
        with open(TOKENIZER_PATH, "rb") as f:
            _tokenizer = pickle.load(f)


def normalize_arabic_text(text: str) -> str:
    """Duplicated from ml/preprocess_lstm_data.py -- see module docstring."""
    if not isinstance(text, str):
        return ""
    text = dediac_ar(text)
    text = normalize_alef_ar(text)
    text = normalize_alef_maksura_ar(text)
    text = normalize_teh_marbuta_ar(text)
    return text


def calculate_lstm_phishing_score(text: str) -> dict:
    """
    Run the trained BiLSTM on a single piece of message text.

    Returns a dict:
        {
            "score": float,        # 0-100 (probability * 100, matching other engines)
            "probability": float,  # raw sigmoid output, 0.0-1.0
        }

    Raises InvalidScanInputError if text is not a non-empty string.
    """
    if not isinstance(text, str) or not text.strip():
        raise InvalidScanInputError("Input text must be a non-empty string.")

    _load_artifacts()

    cleaned = normalize_arabic_text(text)
    sequence = _tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post", truncating="post")

    probability = float(_model.predict(padded, verbose=0)[0][0])
    score = round(probability * 100, 2)

    return {"score": score, "probability": probability}
