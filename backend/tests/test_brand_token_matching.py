"""
Tests for the brand-token, hyphen-split, TLD-swap and brand-in-path
techniques of domain_typosquatting_detector.py (added 2026-10 after the
Egyptian-brand typosquat benchmark), plus false-positive guards for the
multinational-label fix. Pure string logic: no database, no models.
"""

import pytest

from app.engines.rule_based.domain_typosquatting_detector import (
    calculate_domain_typosquatting_score as score_url,
)


@pytest.mark.parametrize("url", [
    "https://nbe-egypt.com",
    "https://support-nbe.org",
    "http://nbe-eg.com",
    "https://nbe.verify-egypt.xyz",
    "https://cib-egypt.com",
])
def test_brand_token_next_to_context_word_is_flagged(url):
    result = score_url(url)
    assert result["technique"] == "brand_token_combo"
    assert 60.0 <= result["score"] < 90.0  # MEDIUM-tier evidence, never HIGH on its own
    assert result["matched_brand"] is not None


@pytest.mark.parametrize("url,brand_fragment", [
    ("https://banque-misr.top", "Banque Misr"),
    ("https://qnb-alahli.top", "QNB"),
    ("https://insta-pay.net", "Instapay"),
    ("https://egypt-air.top", "EgyptAir"),
])
def test_hyphen_split_brand_label_is_flagged(url, brand_fragment):
    result = score_url(url)
    assert result["technique"] == "brand_hyphen_split"
    assert brand_fragment in result["matched_brand"]


@pytest.mark.parametrize("url", ["https://cibeg.info", "https://cibeg.org/login", "http://fawry.net"])
def test_distinctive_label_with_other_suffix_is_flagged(url):
    result = score_url(url)
    assert result["technique"] == "brand_tld_swap"
    assert result["score"] == 60.0


def test_brand_domain_in_path_of_other_host_is_flagged_but_capped():
    result = score_url("https://evil-host.top/cibeg.com/login")
    assert result["technique"] == "brand_in_path"
    assert result["score"] == 60.0
    assert result["matched_brand"] == "Commercial International Bank (CIB)"


def test_very_short_brand_domain_in_path_is_not_checked():
    assert score_url("https://evil-host.top/te.eg/login")["score"] == 0.0


def test_archive_host_mentioning_brand_in_path_is_not_flagged():
    assert score_url("https://web.archive.org/web/2024/https://cibeg.com/")["score"] == 0.0


@pytest.mark.parametrize("url", [
    "https://www.vodafone.com", "https://www.vodafone.co.uk", "https://www.vodafone.de",
    "https://www.orange.fr", "https://www.amazon.de/dp/123", "https://www.etisalat.ae",
    "https://www.jumia.com.ng", "https://qnb.com",
])
def test_multinational_parent_sites_are_not_flagged(url):
    # Before the 2026-10 fix, Levenshtein scored several of these at 60-80.
    assert score_url(url)["score"] == 0.0


@pytest.mark.parametrize("url", [
    "https://nbe.com", "https://aib.com", "https://noon-day.com", "https://afternoon-login.com",
    "https://www.travelegypt.com", "https://digital-egypt.example.org", "https://eg.linkedin.com",
])
def test_ambiguous_or_unrelated_domains_are_not_flagged(url):
    assert score_url(url)["score"] == 0.0


def test_multinational_label_on_abuse_prone_suffix_is_still_flagged():
    assert score_url("https://vodafone.top")["score"] > 0


def test_real_brand_hosts_unchanged_by_new_techniques():
    assert score_url("https://online.cibeg.com/login")["technique"] == "brand_subdomain"
    assert score_url("https://www.nbe.com.eg/")["score"] == 0.0


def test_existing_stronger_techniques_still_win():
    assert score_url("cibeg.com.verify-account-now.net")["technique"] == "domain_embedding"
    assert score_url("https://cibeg.com@evil.example/login")["technique"] == "userinfo_embedding"
