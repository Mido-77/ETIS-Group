"""
test_risk_scoring.py

Tests for the decision-tier scorer (risk_scoring.py): every applicable
engine gives its own tier and the verdict is the highest one.

Pure unit tests cover the tier/band helpers. Integration tests call the real
engines and use the same `db` fixture as test_blacklist_lookup.py (a real
session against the seeded dev database), consistent with the project's
integration-style testing. The learned models must exist
(python ml/export_models.py).

What these tests deliberately pin (each guards a past or likely regression):
  - the verdict is the highest engine tier and agrees with final_score's band
  - adding an input never lowers a strong signal (review item: a clean URL
    used to dilute a 95% phishing text)
  - the allowlist cancels URL-derived evidence only, never message text
  - a domain-embedding typosquat is HIGH on its own, without blacklist
    corroboration (it used to land in MEDIUM)
"""

import pytest

from app.core.exceptions import InvalidScanInputError
from app.engines.risk_scoring import (
    _band_score,
    _noisy_or,
    _tier_from_score,
    calculate_final_risk_score,
)

PHISHING_TEXT = "بطاقتك محظورة يرجى ادخال كود التحقق فورا"
BENIGN_TEXT = "مرحبا بكم في اجتماع الفريق"
TIER_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
ALL_ENGINES = ["rule_based", "blacklist", "url_model", "text_model"]


# ---- pure helpers -------------------------------------------------------------

def test_tier_from_score_boundaries():
    assert _tier_from_score(0.0) == "LOW"
    assert _tier_from_score(33.99) == "LOW"
    assert _tier_from_score(34.0) == "MEDIUM"
    assert _tier_from_score(66.99) == "MEDIUM"
    assert _tier_from_score(67.0) == "HIGH"
    assert _tier_from_score(100.0) == "HIGH"


def test_band_score_maps_each_tier_into_its_band():
    medium, high = 0.666, 0.983
    assert _band_score(0.0, medium, high, 1.0) == 0.0
    assert _band_score(0.665, medium, high, 1.0) <= 33.0
    assert _band_score(medium, medium, high, 1.0) == 34.0
    assert 34.0 < _band_score(0.9, medium, high, 1.0) < 67.0
    assert _band_score(high, medium, high, 1.0) == 67.0
    assert _band_score(1.0, medium, high, 1.0) == 100.0


def test_band_score_is_monotonic_and_agrees_with_tier_thresholds():
    medium, high = 30.0, 60.0
    previous = -1.0
    for value in range(0, 101):
        score = _band_score(float(value), medium, high, 100.0)
        assert score >= previous
        previous = score
        expected = "HIGH" if value >= high else "MEDIUM" if value >= medium else "LOW"
        assert _tier_from_score(score) == expected


def test_noisy_or():
    assert _noisy_or([]) == 0.0
    assert _noisy_or([50.0]) == pytest.approx(50.0)
    assert _noisy_or([50.0, 50.0]) == pytest.approx(75.0)
    assert _noisy_or([100.0, 0.0]) == pytest.approx(100.0)


# ---- applicability ------------------------------------------------------------

def test_text_only_runs_rule_based_and_text_model():
    result = calculate_final_risk_score(text=PHISHING_TEXT)

    assert result["engines_applied"] == ["rule_based", "text_model"]
    assert result["engine_details"]["blacklist"] is None
    assert result["engine_details"]["url_model"] is None
    assert result["engine_details"]["text_model"] is not None
    assert result["skip_reasons"] == {
        "blacklist": "no_url_provided", "url_model": "no_url_provided", "text_model": None,
    }
    assert set(result["engine_tiers"]) == {"rule_based", "text_model"}
    assert result["allowlisted"] is False

    # Deterministic: two HIGH-weight Arabic keyword phrases give a rule score of 60,
    # which is exactly the text-part HIGH threshold.
    assert result["engine_details"]["rule_based"]["score"] == 60.0
    assert result["engine_tiers"]["rule_based"] == "HIGH"
    assert result["verdict"] == "HIGH"
    assert result["decided_by"] in ("rule_based", "text_model")


def test_url_only_runs_rule_based_blacklist_and_url_model(db):
    result = calculate_final_risk_score(url="http://192.168.1.50/login", db=db)

    assert result["engines_applied"] == ["rule_based", "blacklist", "url_model"]
    assert result["engine_details"]["text_model"] is None
    assert result["engine_details"]["blacklist"] is not None
    assert result["engine_details"]["url_model"] is not None
    assert result["skip_reasons"]["text_model"] == "no_text_provided"

    url_pattern_result = result["engine_details"]["rule_based"]["engine_results"]["url_pattern"]
    assert any(flag_name == "ip_address_host" for flag_name, _, _ in url_pattern_result["flags"])
    # IP host (25) + "login" path keyword (8) + no https (8) = 41 >= the 40 MEDIUM threshold.
    assert result["verdict"] in ("MEDIUM", "HIGH")


def test_both_inputs_run_all_four_engines(db):
    result = calculate_final_risk_score(text="بطاقتك محظورة", url="http://192.168.1.50/login", db=db)
    assert result["engines_applied"] == ALL_ENGINES
    assert all(result["engine_details"][engine] is not None for engine in ALL_ENGINES)
    assert set(result["engine_tiers"]) == set(ALL_ENGINES)
    assert set(result["stage_timings"]) == {
        "rule_based_ms", "blacklist_ms", "url_model_ms", "text_model_ms", "total_ms",
    }


def test_edge_case_neither_text_nor_url_raises():
    with pytest.raises(InvalidScanInputError):
        calculate_final_risk_score()


def test_edge_case_url_without_db_raises_value_error():
    """A URL was provided but no database session -- a caller/integration
    error, not a user-input error, so ValueError rather than InvalidScanInputError."""
    with pytest.raises(ValueError):
        calculate_final_risk_score(url="http://example.com")


# ---- verdict consistency --------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        {"text": PHISHING_TEXT},
        {"text": BENIGN_TEXT},
        {"url": "http://192.168.1.50/login"},
        {"url": "https://www.wikipedia.org"},
        {"url": "cibeg.com.verify-account-now.net"},
        {"text": PHISHING_TEXT, "url": "https://www.cibeg.com/login"},
    ],
)
def test_verdict_is_highest_engine_tier_and_matches_score_band(db, kwargs):
    result = calculate_final_risk_score(db=db, **kwargs)
    highest = max(result["engine_tiers"].values(), key=lambda tier: TIER_RANK[tier])
    assert result["verdict"] == highest
    assert _tier_from_score(result["final_score"]) == result["verdict"]
    assert 0.0 <= result["final_score"] <= 100.0
    if result["verdict"] == "LOW":
        assert result["decided_by"] is None
    else:
        assert result["engine_tiers"][result["decided_by"]] == result["verdict"]


def test_adding_a_benign_text_never_lowers_a_strong_url_signal(db):
    url = "cibeg.com.verify-account-now.net"
    url_only = calculate_final_risk_score(url=url, db=db)
    both = calculate_final_risk_score(url=url, text=BENIGN_TEXT, db=db)
    assert TIER_RANK[both["verdict"]] >= TIER_RANK[url_only["verdict"]]
    assert both["verdict"] == "HIGH"


def test_adding_a_benign_url_never_lowers_a_strong_text_signal(db):
    text_only = calculate_final_risk_score(text=PHISHING_TEXT)
    both = calculate_final_risk_score(text=PHISHING_TEXT, url="https://www.wikipedia.org", db=db)
    assert TIER_RANK[both["verdict"]] >= TIER_RANK[text_only["verdict"]]
    assert both["verdict"] == "HIGH"


# ---- Egyptian brand allowlist ---------------------------------------------------

def test_verified_egyptian_domain_is_allowlisted_to_low(db):
    """The brand's own domain cancels the URL-derived signals (including the
    learned URL model, which has a higher false-positive rate on URLs that
    carry a path). engine_details still shows every engine's real output."""
    result = calculate_final_risk_score(url="https://www.cibeg.com/login", db=db)

    assert result["allowlisted"] is True
    assert result["final_score"] == 0.0
    assert result["verdict"] == "LOW"
    assert set(result["engine_tiers"].values()) == {"LOW"}
    for engine in ("rule_based", "blacklist", "url_model"):
        assert result["engine_details"][engine] is not None


def test_allowlisted_url_does_not_cancel_phishing_text(db):
    """A phishing message that links to a real bank URL is still phishing."""
    result = calculate_final_risk_score(text=PHISHING_TEXT, url="https://www.cibeg.com/login", db=db)
    assert result["allowlisted"] is True
    assert result["verdict"] == "HIGH"
    assert result["decided_by"] in ("rule_based", "text_model")
    assert result["engine_tiers"]["blacklist"] == "LOW"
    assert result["engine_tiers"]["url_model"] == "LOW"


def test_real_brand_subdomain_is_allowlisted_low(db):
    result = calculate_final_risk_score(url="https://online.cibeg.com/login", db=db)
    assert result["allowlisted"] is True
    assert result["verdict"] == "LOW"
    assert result["final_score"] == 0.0


def test_typosquat_domain_is_not_allowlisted(db):
    """A Levenshtein-close typosquat is not an exact match, so it is not allowlisted;
    the rule engine's typosquat score puts it at MEDIUM or higher."""
    result = calculate_final_risk_score(url="http://cib3g.com/secure", db=db)
    assert result["allowlisted"] is False
    assert result["verdict"] in ("MEDIUM", "HIGH")


def test_domain_embedding_is_high_without_blacklist_corroboration(db):
    """The real brand domain embedded verbatim in a longer, different domain.
    Under the old additive scorer this landed in MEDIUM (40.89) because the
    blacklist contributed 0; as a decision tier, a typosquat score >= 90 decides HIGH."""
    result = calculate_final_risk_score(url="cibeg.com.verify-account-now.net", db=db)

    assert result["allowlisted"] is False
    assert result["verdict"] == "HIGH"
    typosquat = result["engine_details"]["rule_based"]["engine_results"]["domain_typosquatting"]
    assert typosquat["technique"] == "domain_embedding"
    assert typosquat["score"] >= 90


def test_unrelated_legitimate_domain_is_not_allowlisted(db):
    """A clean, unrelated domain is scored normally, not shortcut."""
    result = calculate_final_risk_score(url="https://www.wikipedia.org", db=db)
    assert result["allowlisted"] is False
    assert result["verdict"] == "LOW"


def test_lookalike_and_userinfo_tricks_are_not_allowlisted(db):
    for url in [
        "https://cibeg.com.verify-account-now.net/login",
        "https://notcibeg.com/login",
        "https://evil.net\\.cibeg.com/login",
        "https://cibeg.com@evil.net/login",
        "https://cibeg.com:443@evil.example/login",
    ]:
        assert calculate_final_risk_score(url=url, db=db)["allowlisted"] is False
