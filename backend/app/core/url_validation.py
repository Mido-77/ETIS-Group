"""
url_validation.py

Input check shared by the scoring layer: a submitted URL must have a host.

Before this check, a bare "https://" (or "http:///login", "https://@") reached
the engines and came back LOW (4.53 in a browser scan on 2026-10-19), because
the URL model scored an empty string. There is nothing to analyse in such an
input, and a green "no indicators" result is the wrong answer to it, so it is
rejected with a 400 like any other unusable input.

Deliberately NOT rejected here: URLs that have a host but cannot be fully
parsed (javascript:alert(1), a non-numeric or out-of-range port, a broken IPv6
literal). The rule engine already reports those with a malformed_url flag, and
changing that would change documented, tested behaviour.
"""

from urllib.parse import urlsplit

from app.core.exceptions import InvalidScanInputError


def ensure_url_has_host(url: str) -> None:
    candidate = url.strip()
    if candidate.startswith("//"):  # scheme-relative, e.g. //example.com/path
        with_scheme = "http:" + candidate
    elif "://" in candidate:
        with_scheme = candidate
    else:  # scheme-less, e.g. wikipedia.org (scored as https:// by the URL model)
        with_scheme = "http://" + candidate
    try:
        host = urlsplit(with_scheme).hostname
    except ValueError:
        return  # unparseable but not hostless: left to the existing malformed_url handling
    if not host:
        raise InvalidScanInputError("The URL has no host name (for example 'https://' on its own).")
