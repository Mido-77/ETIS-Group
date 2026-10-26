"""Scheme-less URLs must be scored as https:// URLs (see ensure_scheme)."""

import pytest

from app.engines.ml.url_model_inference import calculate_url_model_score, ensure_scheme


@pytest.mark.parametrize("given, expected", [
    ("wikipedia.org", "https://wikipedia.org"),
    ("  wikipedia.org/wiki/Egypt  ", "https://wikipedia.org/wiki/Egypt"),
    ("http://example.com", "http://example.com"),
    ("https://example.com", "https://example.com"),
    ("HTTPS://EXAMPLE.COM", "HTTPS://EXAMPLE.COM"),
    ("ftp://example.com/file", "ftp://example.com/file"),
])
def test_ensure_scheme(given, expected):
    assert ensure_scheme(given) == expected


def test_scheme_less_and_https_inputs_score_identically():
    bare = calculate_url_model_score("wikipedia.org/wiki/Egypt")
    explicit = calculate_url_model_score("https://wikipedia.org/wiki/Egypt")
    assert bare["probability"] == explicit["probability"]
    assert bare["tier"] == explicit["tier"]
