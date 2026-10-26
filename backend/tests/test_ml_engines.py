"""
test_ml_engines.py

Light checks for the two learned engines (text_char_inference,
url_model_inference). Skipped automatically if the artifacts have not been
built yet (python ml/export_models.py). The deeper correctness check -- that
the served scores equal the evaluated scores -- is export_models.py's
self-check; these tests only guard the interface and basic sanity.
"""

import pytest

from app.core.exceptions import InvalidScanInputError
from app.engines.ml import text_char_inference, url_model_inference


def _require(path):
    if not path.exists():
        pytest.skip(f"{path.name} not built -- run python ml/export_models.py")


# ---- URL engine -------------------------------------------------------------

def test_url_result_shape_and_ranges():
    _require(url_model_inference.ARTIFACT_PATH)
    result = url_model_inference.calculate_url_model_score("https://example.com/login")
    assert set(result) == {"score", "probability", "tier", "components", "thresholds"}
    assert 0.0 <= result["score"] <= 100.0
    assert 0.0 <= result["probability"] <= 1.0
    assert result["tier"] in ("LOW", "MEDIUM", "HIGH")
    assert result["thresholds"]["medium"] < result["thresholds"]["high"]


def test_url_obviously_random_phishing_style_scores_above_famous_site():
    _require(url_model_inference.ARTIFACT_PATH)
    famous = url_model_inference.calculate_url_model_score("https://wikipedia.org")
    suspicious = url_model_inference.calculate_url_model_score(
        "http://xk3j9a-secure-login-verify.top/account/update?id=83hd7"
    )
    assert suspicious["probability"] > famous["probability"]


@pytest.mark.parametrize("bad", ["", "   ", None, 123])
def test_url_invalid_input_raises(bad):
    with pytest.raises(InvalidScanInputError):
        url_model_inference.calculate_url_model_score(bad)


def test_url_malformed_input_does_not_crash():
    _require(url_model_inference.ARTIFACT_PATH)
    for url in ["javascript:alert(1)", "http://[::1", "http://", "https://cibeg.com:443@evil.example/login"]:
        assert 0.0 <= url_model_inference.calculate_url_model_score(url)["score"] <= 100.0


# ---- Text engine ------------------------------------------------------------

def test_text_result_shape_and_ranges():
    _require(text_char_inference.ARTIFACT_PATH)
    result = text_char_inference.calculate_text_char_phishing_score("بطاقتك محظورة يرجى ادخال كود التحقق فورا")
    assert set(result) == {"score", "probability", "tier", "thresholds"}
    assert 0.0 <= result["probability"] <= 1.0
    assert result["score"] == round(result["probability"] * 100, 2)
    assert result["tier"] in ("LOW", "MEDIUM", "HIGH")
    assert result["thresholds"]["medium"] < result["thresholds"]["high"]


def test_text_phishing_message_scores_above_ordinary_message():
    _require(text_char_inference.ARTIFACT_PATH)
    phishing = text_char_inference.calculate_text_char_phishing_score("بطاقتك محظورة يرجى ادخال كود التحقق فورا")
    ordinary = text_char_inference.calculate_text_char_phishing_score("اجتماع الفريق غدا الساعة العاشرة صباحا")
    assert phishing["probability"] > ordinary["probability"]


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_text_invalid_input_raises(bad):
    with pytest.raises(InvalidScanInputError):
        text_char_inference.calculate_text_char_phishing_score(bad)
