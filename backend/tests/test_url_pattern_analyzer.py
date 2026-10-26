import pytest

from app.engines.rule_based.url_pattern_analyzer import (
    calculate_url_pattern_score,
    InvalidScanInputError,
)


def test_ip_address_host_flagged():
    result = calculate_url_pattern_score("http://192.168.1.1/login")
    flag_names = [f[0] for f in result["flags"]]
    assert "ip_address_host" in flag_names
    assert result["score"] > 0


def test_at_symbol_redirect_trick_flagged():
    result = calculate_url_pattern_score("http://cibeg.com@malicious-site.net/login")
    flag_names = [f[0] for f in result["flags"]]
    assert "at_symbol_redirect_trick" in flag_names


def test_known_shortener_flagged():
    result = calculate_url_pattern_score("https://bit.ly/2Tjfhgl")
    flag_names = [f[0] for f in result["flags"]]
    assert "known_url_shortener" in flag_names


def test_clean_legitimate_url_scores_zero():
    result = calculate_url_pattern_score("https://www.cibeg.com/personal-banking")
    assert result["score"] == 0.0


def test_excessive_subdomains_flagged():
    result = calculate_url_pattern_score("https://secure.login.verify.account.cibeg-fake.net/")
    flag_names = [f[0] for f in result["flags"]]
    assert "excessive_subdomains" in flag_names


def test_suspicious_path_keyword_flagged():
    result = calculate_url_pattern_score("https://example.com/verify-account/update")
    flag_names = [f[0] for f in result["flags"]]
    assert "suspicious_path_keyword" in flag_names


def test_multiple_flags_combine_score():
    # IP host (25) + suspicious path keyword (8) + no https (8) should stack to >= 33
    result = calculate_url_pattern_score("http://10.0.0.5/secure/login/verify")
    assert result["score"] >= 33


def test_edge_case_empty_string_raises():
    with pytest.raises(InvalidScanInputError):
        calculate_url_pattern_score("")


def test_edge_case_non_string_raises():
    with pytest.raises(InvalidScanInputError):
        calculate_url_pattern_score(None)


def test_edge_case_url_without_scheme_still_parses():
    result = calculate_url_pattern_score("bit.ly/2Tjfhgl")
    flag_names = [f[0] for f in result["flags"]]
    assert "known_url_shortener" in flag_names


# ---- 2026-10: scheme-less input, malformed input, '@' placement -------------

def _names(url):
    return [f[0] for f in calculate_url_pattern_score(url)["flags"]]


def test_scheme_less_url_is_not_flagged_as_no_https():
    assert "no_https" not in _names("wikipedia.org")
    assert "no_https" not in _names("example.com/some/page")


def test_explicit_http_is_still_flagged_as_no_https():
    assert "no_https" in _names("http://example.com/")
    assert "no_https" in _names("HTTP://EXAMPLE.COM/")


@pytest.mark.parametrize("url", [
    "javascript:alert(1)", "http://example.com:99999/x", "http://example.com:abc/",
    "http://[::1", "https://example.com:-1/",
])
def test_unparseable_urls_do_not_crash_and_are_flagged(url):
    result = calculate_url_pattern_score(url)
    assert [f[0] for f in result["flags"]] == ["malformed_url"]
    assert 0 < result["score"] < 50


def test_at_sign_in_query_string_is_not_the_redirect_trick():
    assert "at_symbol_redirect_trick" not in _names("https://example.com/?email=a@b.com")


def test_at_sign_before_host_is_still_flagged():
    assert "at_symbol_redirect_trick" in _names("https://cibeg.com:443@evil.example/login")


def test_non_standard_port_still_flagged():
    assert "non_standard_port" in _names("https://example.com:8443/")
