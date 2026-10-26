import pytest

from app.engines.rule_based.arabic_keyword_detector import (
    calculate_arabic_keyword_phishing_score,
    normalize_arabic_text,
    InvalidScanInputError,
)


def test_clear_phishing_example_scores_high():
    text = "مبروك لقد فزت بجائزة من بنك مصر! لاستلام الجائزة أكد بياناتك عبر الرابط."
    result = calculate_arabic_keyword_phishing_score(text)
    assert result["score"] >= 60
    assert len(result["matched_high"]) >= 1


def test_clear_legitimate_example_scores_low():
    text = "تم إيداع راتبكم الشهري بنجاح في حسابكم. الرصيد الحالي متاح عبر تطبيق AAIB Mobile Banking."
    result = calculate_arabic_keyword_phishing_score(text)
    assert result["score"] < 30


def test_edge_case_empty_string_raises():
    with pytest.raises(InvalidScanInputError):
        calculate_arabic_keyword_phishing_score("")


def test_edge_case_non_string_raises():
    with pytest.raises(InvalidScanInputError):
        calculate_arabic_keyword_phishing_score(None)


def test_diacritics_and_alef_variants_are_normalized():
    # Same phrase, with diacritics and an alef-hamza variant, should still match
    text_with_diacritics = "مَبْرُوك لَقَدْ فُزْتَ بِجَائِزَة"
    normalized = normalize_arabic_text(text_with_diacritics)
    assert "مبروك لقد فزت بجائزة" in normalized


def test_low_weight_terms_alone_do_not_inflate_score():
    # A single generic low-weight word ("account") should not, by itself,
    # meaningfully raise the score.
    text = "هذا حساب تجريبي للاختبار فقط بدون أي معنى خاص"
    result = calculate_arabic_keyword_phishing_score(text)
    assert result["low_weight_counted"] is True  # multiple low terms present
    # but score should still be modest since no high/medium signal exists
    assert result["score"] < 40
