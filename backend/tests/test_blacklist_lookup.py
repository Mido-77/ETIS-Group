import pytest

from app.engines.threat_intel.blacklist_lookup import (
    calculate_blacklist_reputation_score,
    SHARED_HOSTING_DOMAINS,
)
from app.core.exceptions import InvalidScanInputError
from app.models import BlacklistEntry



def test_exact_match_returns_stored_reputation_score(db):
    entry = db.query(BlacklistEntry).first()
    if entry is None:
        pytest.skip("Database not seeded -- run data/scripts/seed_database.py first.")
    result = calculate_blacklist_reputation_score(entry.url, db)
    assert result["match_type"] == "exact"
    assert result["score"] == entry.reputation_score


def test_unrelated_clean_url_scores_zero(db):
    result = calculate_blacklist_reputation_score("https://www.wikipedia.org/wiki/Egypt", db)
    assert result["score"] == 0.0
    assert result["match_type"] is None


def test_shared_hosting_domain_suppresses_domain_level_match(db):
    result = calculate_blacklist_reputation_score(
        "https://docs.google.com/document/d/some-unrelated-legit-doc", db
    )
    assert result["score"] == 0.0


def test_shared_hosting_domain_exact_match_still_works(db):
    entry = db.query(BlacklistEntry).filter(
        BlacklistEntry.domain.in_(SHARED_HOSTING_DOMAINS)
    ).first()
    if entry is None:
        pytest.skip("No shared-hosting-domain entries in seeded data.")
    result = calculate_blacklist_reputation_score(entry.url, db)
    assert result["match_type"] == "exact"
    assert result["score"] > 0


def test_edge_case_empty_string_raises(db):
    with pytest.raises(InvalidScanInputError):
        calculate_blacklist_reputation_score("", db)


def test_edge_case_non_string_raises(db):
    with pytest.raises(InvalidScanInputError):
        calculate_blacklist_reputation_score(None, db)
