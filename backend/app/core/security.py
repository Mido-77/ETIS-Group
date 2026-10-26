"""
security.py

Minimal API-key gate for publicly hosted ETIS endpoints. Deliberately not
JWT/RBAC -- that scope was cut at the group-proposal stage (see
Group_Project_Proposal, section 3.3) and a single-operator manual-scan demo
doesn't need session-based auth. This exists purely so a hosted /analyze
endpoint isn't wide open.

A missing or wrong key raises AuthenticationError, which the global handler
in main.py maps to HTTP 401 (not 400 -- the request is well-formed, it just
isn't authenticated).
"""

import os
import secrets

from fastapi import Header

from app.core.exceptions import AuthenticationError

EXPECTED_API_KEY = os.getenv("ETIS_API_KEY")


def verify_api_key(x_api_key: str = Header(default=None)) -> None:
    """FastAPI dependency: raises AuthenticationError if the request's
    X-API-Key header doesn't match ETIS_API_KEY from the environment. If
    ETIS_API_KEY isn't set at all (local dev), the check is skipped entirely
    -- so local development never needs a key, only a real deployment with
    the env var set.

    The comparison is constant-time (secrets.compare_digest on bytes, which
    also avoids a TypeError on non-ASCII header values) so response timing
    can't leak how much of a guessed key was correct."""
    if EXPECTED_API_KEY is None:
        return
    if x_api_key is None or not secrets.compare_digest(
        x_api_key.encode("utf-8"), EXPECTED_API_KEY.encode("utf-8")
    ):
        raise AuthenticationError("Missing or invalid API key.")
