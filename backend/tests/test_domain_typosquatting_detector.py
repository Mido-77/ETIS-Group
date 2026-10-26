import pytest

from app.engines.rule_based.domain_typosquatting_detector import (
    calculate_domain_typosquatting_score,
    calculate_levenshtein_distance,
    normalize_domain,
    InvalidScanInputError,
)


def test_exact_match_is_not_flagged():
    result = calculate_domain_typosquatting_score("https://www.cibeg.com/login")
    assert result["score"] == 0.0
    assert result["technique"] == "exact_match"
    assert result["matched_brand"] == "Commercial International Bank (CIB)"


def test_domain_embedding_scores_very_high():
    result = calculate_domain_typosquatting_score("cibeg.com.verify-account-now.net")
    assert result["score"] >= 90
    assert result["technique"] == "domain_embedding"


def test_homoglyph_cyrillic_letter_substitution_detected():
    # Cyrillic "е" (U+0435) substituted for Latin "e" in cibeg.com --
    # renders visually identical in a browser, but is a different string.
    # Checked before Levenshtein (see technique ordering note in
    # domain_typosquatting_detector.py's module docstring) -- a single
    # confusable-character substitution is also "1 edit" in plain
    # Levenshtein terms, so homoglyph must be checked first or it never
    # gets a chance to fire for that brand.
    homoglyph_domain = "cib\u0435g.com"  # cib + Cyrillic е + g.com
    result = calculate_domain_typosquatting_score(homoglyph_domain)
    assert result["technique"] == "homoglyph_confusable"
    assert result["matched_brand"] == "Commercial International Bank (CIB)"
    assert result["score"] >= 90


def test_homoglyph_greek_letter_substitution_detected():
    # Greek "α" (U+03B1) substituted for Latin "a" in banquemisr.com.
    homoglyph_domain = "b\u03b1nquemisr.com"
    result = calculate_domain_typosquatting_score(homoglyph_domain)
    assert result["technique"] == "homoglyph_confusable"
    assert result["matched_brand"] == "Banque Misr"


def test_homoglyph_punycode_encoded_domain_detected():
    # Same Cyrillic substitution as above, but encoded as punycode (the
    # form it would actually appear in as a registered domain / in a raw
    # URL) rather than raw Unicode -- computed dynamically via Python's
    # built-in idna codec rather than a hardcoded punycode string, so this
    # test doesn't depend on hand-transcribing punycode correctly.
    homoglyph_label = "cib\u0435g"
    punycode_label = homoglyph_label.encode("idna").decode("ascii")
    candidate = f"{punycode_label}.com"
    result = calculate_domain_typosquatting_score(candidate)
    assert result["technique"] == "homoglyph_confusable"
    assert result["matched_brand"] == "Commercial International Bank (CIB)"


def test_homoglyph_does_not_fire_on_plain_ascii_unrelated_domain():
    # Regression guard: a domain with zero confusable/punycode characters
    # should never trigger the homoglyph technique, even if it happens to
    # score via a different technique or not at all.
    result = calculate_domain_typosquatting_score("https://www.wikipedia.org")
    assert result["technique"] != "homoglyph_confusable"


def test_malformed_punycode_does_not_crash():
    # A broken/invalid punycode label should degrade gracefully (fall back
    # to treating it as an ordinary, non-decoded label) rather than raising
    # -- a malformed label is exactly the kind of thing a hostile input
    # might contain, deliberately or not.
    result = calculate_domain_typosquatting_score("xn--!!invalid.com")
    assert isinstance(result["score"], float)


def test_levenshtein_typosquat_detected():
    # cib3g.com vs cibeg.com -- one character substitution
    result = calculate_domain_typosquatting_score("http://cib3g.com/secure")
    assert result["score"] > 0
    assert result["matched_brand"] == "Commercial International Bank (CIB)"


def test_brand_label_reuse_detected():
    result = calculate_domain_typosquatting_score("secure-vodafone-com-eg.net")
    assert result["score"] > 0


def test_unrelated_domain_scores_zero():
    result = calculate_domain_typosquatting_score("https://www.wikipedia.org")
    assert result["score"] == 0.0
    assert result["matched_brand"] is None


def test_short_brand_domain_does_not_cause_false_positive():
    # "te.eg" (Telecom Egypt) is very short -- an unrelated short domain
    # should NOT trigger a Levenshtein match against it.
    result = calculate_domain_typosquatting_score("https://tv.eg")
    assert result["technique"] != "levenshtein" or result["matched_brand"] != "Telecom Egypt (WE)"


def test_normalize_domain_strips_scheme_www_path_port():
    assert normalize_domain("https://www.cibeg.com:443/login?x=1") == "cibeg.com"
    assert normalize_domain("HTTP://CIBEG.COM/") == "cibeg.com"


def test_levenshtein_distance_basic():
    assert calculate_levenshtein_distance("cibeg.com", "cibeg.com") == 0
    assert calculate_levenshtein_distance("cibeg.com", "cib3g.com") == 1
    assert calculate_levenshtein_distance("", "abc") == 3


def test_edge_case_empty_string_raises():
    with pytest.raises(InvalidScanInputError):
        calculate_domain_typosquatting_score("")


def test_edge_case_non_string_raises():
    with pytest.raises(InvalidScanInputError):
        calculate_domain_typosquatting_score(None)


def test_unrelated_domain_containing_brand_substring_is_not_flagged():
    for url in ["https://afternoon.com", "https://website.eg", "https://afternoon-login.com"]:
        assert calculate_domain_typosquatting_score(url)["score"] == 0.0


def test_real_brand_subdomain_is_not_flagged():
    for url in ["https://online.cibeg.com", "https://mobile.nbe.com.eg"]:
        result = calculate_domain_typosquatting_score(url)
        assert result["score"] == 0.0
        assert result["technique"] == "brand_subdomain"


def test_embedding_attacks_still_flagged_after_boundary_fix():
    for url in ["cibeg.com.verify-account-now.net", "noon.com.secure-login.net", "te.eg.verify-now.net"]:
        result = calculate_domain_typosquatting_score(url)
        assert result["score"] >= 90
        assert result["technique"] == "domain_embedding"


def test_backslash_host_trick_is_not_treated_as_brand_subdomain():
    result = calculate_domain_typosquatting_score("https://evil.net\\.cibeg.com/login")
    assert result["technique"] != "brand_subdomain"
    assert result["score"] >= 90


