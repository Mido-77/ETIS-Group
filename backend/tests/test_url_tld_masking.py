"""
test_url_tld_masking.py

The v2 URL model reads the host without its last label (see
ml/ablation_url_tld_mask.py and evaluation/test_tld_shortcut.py: the v1 model
treated ".eg" as a sign of legitimacy). These tests cover the masking function
and the serving path with a small artifact trained inside the test, so they do
not depend on ml/models/*.joblib.
"""

import pytest

from app.engines.ml import url_model_inference
from app.engines.ml.url_features import (
    HOST_FEATURES, PATH_FEATURES, build_matrix, char_model_text, feature_frame, mask_tld_in_text,
)


@pytest.mark.parametrize("given, expected", [
    ("cairo123.eg/p?x=1", "cairo123.tld/p?x=1"),
    ("shop.example.com.eg/a", "shop.example.com.tld/a"),
    ("user@host.com:8080/path", "user@host.tld:8080/path"),
    ("www.example.co.uk/", "www.example.co.tld/"),
    ("example.com./a", "example.tld/a"),
    ("1.2.3.4/x", "1.2.3.4/x"),            # IP addresses are left alone
    ("localhost/x", "localhost/x"),        # no dot: nothing to mask
    ("example.com", "example.tld"),
    ("example.com?a=b.c", "example.tld?a=b.c"),
])
def test_mask_tld_in_text(given, expected):
    assert mask_tld_in_text(given) == expected


def test_swapping_the_tld_does_not_change_the_masked_text():
    for a, b in [("x.example.top/a?b=1", "x.example.eg/a?b=1"), ("x.example.net/", "x.example.xyz/")]:
        assert mask_tld_in_text(a) == mask_tld_in_text(b)


def test_masking_works_on_char_model_text_output():
    assert mask_tld_in_text(char_model_text("HTTPS://Login.Example.EG/Verify")) == "login.example.tld/verify"


def _tiny_artifact(masked: bool) -> dict:
    lightgbm = pytest.importorskip("lightgbm")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    phishing = [f"https://x{i}-secure-login.example{i}.{tld}/verify/{i}.php" for i in range(40) for tld in [("top", "xyz", "com")[i % 3]]]
    legit = [f"https://www.news{i}.{tld}/" for i in range(40) for tld in [("org", "com", "net")[i % 3]]]
    urls, y = phishing + legit, [1] * len(phishing) + [0] * len(legit)
    texts = [char_model_text(u) for u in urls]
    if masked:
        texts = [mask_tld_in_text(t) for t in texts]
    vectoriser = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    char_logreg = LogisticRegression(C=10.0, max_iter=300).fit(vectoriser.fit_transform(texts), y)
    columns = HOST_FEATURES + PATH_FEATURES
    X = build_matrix(feature_frame(urls), columns, [])
    if masked:
        X = X[columns]
    model = lightgbm.LGBMClassifier(n_estimators=20, num_leaves=4, min_child_samples=2, random_state=0, verbose=-1).fit(X, y)
    artifact = {"vectorizer": vectoriser, "char_logreg": char_logreg, "lightgbm": model, "lgbm_columns": columns,
                "tld_categories": [], "thresholds": {"medium": 0.5, "high": 0.9}}
    if masked:
        artifact["tld_masked"] = True
    return artifact


def test_masked_artifact_character_model_is_blind_to_the_last_label(monkeypatch):
    monkeypatch.setattr(url_model_inference, "_artifact", _tiny_artifact(masked=True))
    top = url_model_inference.calculate_url_model_score("https://login-secure.example.top/a/b")
    eg = url_model_inference.calculate_url_model_score("https://login-secure.example.eg/a/b")
    assert top["components"]["char_logreg"] == eg["components"]["char_logreg"]


def test_masked_artifact_keeps_the_output_shape(monkeypatch):
    monkeypatch.setattr(url_model_inference, "_artifact", _tiny_artifact(masked=True))
    result = url_model_inference.calculate_url_model_score("https://example.com/login")
    assert set(result) == {"score", "probability", "tier", "components", "thresholds"}
    assert set(result["components"]) == {"char_logreg", "lightgbm"}
    assert 0.0 <= result["probability"] <= 1.0
    assert result["tier"] in {"LOW", "MEDIUM", "HIGH"}


def test_artifact_without_the_flag_is_scored_as_v1(monkeypatch):
    """v1 artifacts have no 'tld_masked' key (or False): the unmasked path must be used."""
    artifact = _tiny_artifact(masked=False)
    assert "tld_masked" not in artifact
    monkeypatch.setattr(url_model_inference, "_artifact", artifact)
    result = url_model_inference.calculate_url_model_score("https://example.com/login")
    assert set(result) == {"score", "probability", "tier", "components", "thresholds"}
    # the v1 character model may read the TLD: identical scores for different TLDs are NOT required
    expected = artifact["char_logreg"].predict_proba(
        artifact["vectorizer"].transform([char_model_text("https://example.com/login")]))[0, 1]
    assert result["components"]["char_logreg"] == pytest.approx(float(expected))
