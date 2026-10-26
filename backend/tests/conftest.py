"""
conftest.py

Shared pytest fixtures for backend/tests/. pytest auto-discovers this file
and makes its fixtures available to every test module in this directory
without an explicit import.

The `db` fixture was originally defined locally inside
test_blacklist_lookup.py; centralizing it here so test_risk_scoring.py
(and any future test file, including Phase 6 API endpoint tests) can use
the same session fixture without duplicating it.

ETIS_SKIP_WARMUP is set before anything imports the app, so a test that
starts the FastAPI lifespan (e.g. TestClient used as a context manager)
does not spend seconds pre-loading models the test may not need.
"""

import os

os.environ.setdefault("ETIS_SKIP_WARMUP", "1")

import pytest  # noqa: E402

from app.core.database import SessionLocal  # noqa: E402


@pytest.fixture
def db():
    session = SessionLocal()
    yield session
    session.close()
