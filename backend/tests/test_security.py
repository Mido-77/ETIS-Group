import pytest

from app.core import security
from app.core.exceptions import AuthenticationError


def test_no_key_configured_allows_any_request(monkeypatch):
    monkeypatch.setattr(security, "EXPECTED_API_KEY", None)
    security.verify_api_key(None)
    security.verify_api_key("anything")


def test_correct_key_is_accepted(monkeypatch):
    monkeypatch.setattr(security, "EXPECTED_API_KEY", "s3cret")
    security.verify_api_key("s3cret")


def test_wrong_key_raises_authentication_error(monkeypatch):
    monkeypatch.setattr(security, "EXPECTED_API_KEY", "s3cret")
    with pytest.raises(AuthenticationError):
        security.verify_api_key("nope")


def test_missing_key_raises_authentication_error(monkeypatch):
    monkeypatch.setattr(security, "EXPECTED_API_KEY", "s3cret")
    with pytest.raises(AuthenticationError):
        security.verify_api_key(None)


def test_non_ascii_key_is_rejected_not_crashed(monkeypatch):
    monkeypatch.setattr(security, "EXPECTED_API_KEY", "s3cret")
    with pytest.raises(AuthenticationError):
        security.verify_api_key("مفتاح")


def test_authentication_error_maps_to_401():
    from app.core.exceptions import AuthenticationError
    from app.main import _STATUS_BY_EXCEPTION

    assert _STATUS_BY_EXCEPTION[AuthenticationError] == 401