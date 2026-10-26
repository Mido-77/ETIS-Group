import pytest

from app.engines.rule_based.english_keyword_detector import (
    calculate_english_keyword_phishing_score,
    normalize_english_text,
    InvalidScanInputError,
)


def test_lottery_scam_scores_high():
    text = (
        "Congratulations! You have been selected as the lucky winner of "
        "the International Lottery. Please provide your bank account "
        "number for wire transfer. Reply immediately, this is confidential."
    )
    result = calculate_english_keyword_phishing_score(text)
    assert result["score"] >= 60
    assert len(result["matched_high"]) >= 1


def test_clear_legitimate_example_scores_low():
    text = "Your monthly statement is now available in your online banking portal."
    result = calculate_english_keyword_phishing_score(text)
    assert result["score"] < 20


def test_edge_case_empty_string_raises():
    with pytest.raises(InvalidScanInputError):
        calculate_english_keyword_phishing_score("")


def test_edge_case_non_string_raises():
    with pytest.raises(InvalidScanInputError):
        calculate_english_keyword_phishing_score(None)


def test_case_insensitivity():
    text = "WIRE TRANSFER required immediately to CLAIM YOUR PRIZE"
    normalized = normalize_english_text(text)
    assert "wire transfer" in normalized
    result = calculate_english_keyword_phishing_score(text)
    assert result["score"] > 0


def test_low_weight_terms_alone_do_not_inflate_score():
    # Deliberately avoids any HIGH/MEDIUM phrase substrings (e.g. "verify
    # your account", "click the link") -- only bare LOW-tier words
    # ("account", "reply", "address") should fire here. An earlier version
    # of this test accidentally embedded the HIGH-weight phrase "verify
    # your account" in its sample text, which correctly scored 90 -- that
    # was a test-authoring bug, not a detector bug.
    text = "Please check your account and reply with your address."
    result = calculate_english_keyword_phishing_score(text)
    assert result["low_weight_counted"] is True  # multiple low terms present
    # but score should still be modest since no high/medium signal exists
    assert result["score"] < 40
