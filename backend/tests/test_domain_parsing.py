"""
test_domain_parsing.py

Adoption step 1: host parsing. normalize_domain() must return the host a
browser would actually connect to, and the Egyptian-brand allowlist must
never trust a URL whose real host is something else.

No database needed -- these call the pure functions directly.
"""

import pytest

from app.engines.rule_based.domain_typosquatting_detector import (
    calculate_domain_typosquatting_score,
    extract_userinfo,
    normalize_domain,
)
from app.engines.risk_scoring import _is_verified_egyptian_domain


# ---- normalize_domain: real host ------------------------------------------

@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://www.cibeg.com:443/login?x=1", "cibeg.com"),
        ("HTTP://CIBEG.COM/", "cibeg.com"),
        ("cibeg.com.verify-account-now.net", "cibeg.com.verify-account-now.net"),
        ("cibeg.com:443/login", "cibeg.com"),
        ("example.com?x=1", "example.com"),
        ("https://example.com.", "example.com"),
        ("https://cibeg.com@evil.net/login", "evil.net"),
        ("https://cibeg.com:443@evil.example/login", "evil.example"),
        ("https://user:pw@www.Evil.Example:8443/a#frag", "evil.example"),
        ("https://evil.net\\.cibeg.com/login", "evil.net"),
        ("https://exa\tmple.com/\n", "example.com"),
    ],
)
def test_normalize_domain_returns_real_host(url, expected):
    assert normalize_domain(url) == expected


@pytest.mark.parametrize(
    "url", ["javascript:alert(1)", "http://[::1", "http://", "http://[bad/path", "   "]
)
def test_normalize_domain_never_raises_on_malformed_input(url):
    assert isinstance(normalize_domain(url), str)


def test_extract_userinfo():
    assert extract_userinfo("http://cibeg.com@evil.net/x") == "cibeg.com"
    assert extract_userinfo("https://cibeg.com:443@evil.example/login") == "cibeg.com:443"
    assert extract_userinfo("https://cibeg.com/login") == ""


# ---- allowlist: forged hosts must not be trusted ---------------------------

@pytest.mark.parametrize(
    "url",
    [
        "https://cibeg.com:443@evil.example/login",
        "https://cibeg.com:pw@evil.net/login",
        "https://cibeg.com@evil.net/login",
        "https://evil.net\\.cibeg.com/login",
        "https://cibeg.com.verify-account-now.net/login",
        "https://notcibeg.com/login",
        "javascript:alert(1)",
    ],
)
def test_forged_hosts_are_not_allowlisted(url):
    assert _is_verified_egyptian_domain(url) is False


@pytest.mark.parametrize(
    "url",
    [
        "https://cibeg.com/login",
        "https://CIBEG.COM:443/x",
        "https://www.cibeg.com/personal-banking",
        "https://online.cibeg.com/login",
    ],
)
def test_real_brand_hosts_are_allowlisted(url):
    assert _is_verified_egyptian_domain(url) is True


# ---- typosquatting engine: brand in userinfo is still caught ---------------

@pytest.mark.parametrize(
    "url",
    [
        "http://cibeg.com@malicious-site.net/login",
        "https://cibeg.com:443@evil.example/login",
    ],
)
def test_brand_in_userinfo_scores_high(url):
    result = calculate_domain_typosquatting_score(url)
    assert result["score"] >= 90
    assert result["technique"] == "userinfo_embedding"
    assert result["matched_brand"] == "Commercial International Bank (CIB)"


def test_real_brand_host_with_harmless_userinfo_stays_clean():
    result = calculate_domain_typosquatting_score("https://someone@cibeg.com/login")
    assert result["score"] == 0.0
    assert result["technique"] == "exact_match"


def test_existing_behaviour_unchanged():
    assert calculate_domain_typosquatting_score("cibeg.com.verify-account-now.net")["score"] >= 90
    assert calculate_domain_typosquatting_score("https://www.wikipedia.org")["score"] == 0.0


def test_backslash_trick_still_scores_high_with_real_host_evil_net():
    result = calculate_domain_typosquatting_score("https://evil.net\\.cibeg.com/login")
    assert normalize_domain("https://evil.net\\.cibeg.com/login") == "evil.net"
    assert result["score"] >= 90
    assert result["technique"] == "backslash_embedding"
    assert result["matched_brand"] == "Commercial International Bank (CIB)"


def test_backslash_inside_path_is_not_flagged():
    result = calculate_domain_typosquatting_score("https://example.com/a\\cibeg.com")
    # Not the deceptive-host trick (95): the real host is example.com. Since the
    # 2026-10 brand-in-path technique it is a MEDIUM-tier path mention (60),
    # never HIGH.
    assert result["technique"] != "backslash_embedding"
    assert result["score"] < 90.0
