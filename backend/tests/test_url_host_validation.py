"""
test_url_host_validation.py

A URL with no host ("https://") used to be scored LOW; it is now rejected as
invalid input. URLs that have a host but are unparseable (javascript:, bad
port) must keep going through the existing malformed_url handling.
"""

import pytest

from app.core.exceptions import InvalidScanInputError
from app.core.url_validation import ensure_url_has_host
from app.engines.risk_scoring import calculate_final_risk_score


@pytest.mark.parametrize("url", [
    "https://", "http://", "https:///login", "http://:80/", "https://@", "//", "   https://   ",
])
def test_url_without_host_is_rejected(url):
    with pytest.raises(InvalidScanInputError):
        ensure_url_has_host(url)


@pytest.mark.parametrize("url", [
    "https://wikipedia.org",
    "wikipedia.org",
    "http://192.168.1.50:8080/secure/login",
    "//example.com/x",
    "https://user@host.example/",
    "javascript:alert(1)",      # has a "host" as far as parsing goes; malformed_url handles it
    "http://[::1",              # broken IPv6 literal: unparseable, not hostless
    "http://host:99999/",       # out-of-range port
])
def test_url_with_a_host_or_unparseable_is_not_rejected_here(url):
    ensure_url_has_host(url)  # must not raise


def test_scorer_rejects_hostless_url(db):
    with pytest.raises(InvalidScanInputError):
        calculate_final_risk_score(url="https://", db=db)


def test_scorer_rejects_hostless_url_even_when_text_is_valid(db):
    with pytest.raises(InvalidScanInputError):
        calculate_final_risk_score(text="تم إيداع راتبكم الشهري بنجاح", url="https://", db=db)


def test_analyze_endpoint_returns_400_for_hostless_url(monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    # Make the test independent of any ETIS_API_KEY set in the developer's shell.
    monkeypatch.delenv("ETIS_API_KEY", raising=False)
    monkeypatch.setattr("app.core.security.EXPECTED_API_KEY", None, raising=False)

    response = TestClient(app).post("/analyze", json={"url": "https://"})
    assert response.status_code == 400
    assert response.json()["error"] == "InvalidScanInputError"
