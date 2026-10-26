"""
exceptions.py

Shared ETIS exception hierarchy, per CLAUDE.md's error-handling convention:
custom exception classes + a global FastAPI exception handler (added in
Phase 6), rather than bare try/except with HTTPException scattered through
routes.
"""


class ETISException(Exception):
    """Base exception for all ETIS-specific errors."""


class InvalidScanInputError(ETISException):
    """Raised when input passed to a detection engine is not usable
    (empty, wrong type, or otherwise malformed)."""


class EngineFailureError(ETISException):
    """Raised when a detection engine fails to complete scoring for a
    reason other than invalid input (e.g. an unexpected internal error)."""


class BlacklistLookupError(ETISException):
    """Raised when a threat-intelligence blacklist lookup fails
    (used starting in Phase 3)."""


class AuthenticationError(ETISException):
    """Raised when a request's API key is missing or invalid
    (mapped to HTTP 401 in main.py)."""