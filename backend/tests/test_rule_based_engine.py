import pytest

from app.engines.rule_based.rule_based_engine import calculate_rule_based_score
from app.core.exceptions import InvalidScanInputError


def test_strong_single_engine_signal_is_not_diluted():
    # A domain-embedding typosquat (score ~95) with NO text provided should
    # stay high, not get averaged down by absent engines.
    result = calculate_rule_based_score(url="cibeg.com.verify-account-now.net")
    assert result["score"] >= 90
    assert result["engine_results"]["arabic_keyword"] is None  # no text -> not applicable
    assert result["engine_results"]["domain_typosquatting"] is not None
    assert result["engine_results"]["url_pattern"] is not None


def test_text_only_input_runs_only_keyword_engine():
    text = "مبروك لقد فزت بجائزة من بنك مصر! أكد بياناتك عبر الرابط."
    result = calculate_rule_based_score(text=text)
    assert result["engine_results"]["arabic_keyword"] is not None
    assert result["engine_results"]["domain_typosquatting"] is None
    assert result["engine_results"]["url_pattern"] is None
    assert result["score"] > 0


def test_combined_text_and_url_runs_all_three_engines():
    result = calculate_rule_based_score(
        text="حسابك محظور، تحديث بياناتك البنكية فورا",
        url="http://192.168.1.1/verify",
    )
    assert all(v is not None for v in result["engine_results"].values())


def test_clean_input_scores_near_zero():
    result = calculate_rule_based_score(
        text="تم إيداع راتبكم الشهري بنجاح في حسابكم.",
        url="https://www.cibeg.com/personal-banking",
    )
    assert result["score"] < 15


def test_no_input_raises():
    with pytest.raises(InvalidScanInputError):
        calculate_rule_based_score()


def test_empty_strings_raise():
    with pytest.raises(InvalidScanInputError):
        calculate_rule_based_score(text="", url="")


def test_noisy_or_math_stays_low_for_clean_legitimate_url():
    # exact_match on typosquat scores 0. url_pattern gives a small LOW-weight
    # signal (8) because the path "/login" literally contains the word
    # "login" -- a real, expected weak signal, not a false alarm (see
    # url_pattern_analyzer's own validation: 0 false positives above 25).
    result = calculate_rule_based_score(url="https://www.cibeg.com/login")
    assert result["score"] <= 10
