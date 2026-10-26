"""
blacklist_lookup.py

The threat-intelligence engine (max 40 pts in the overall risk score, per
CLAUDE.md's architecture) -- queries blacklist_entries for a match against
a submitted URL, at two levels of confidence:

1. Exact match (via url_hash) -- this specific URL is a known phishing URL.
   Highest confidence: returns the stored reputation_score directly.
2. Domain match -- the URL's domain has appeared in the blacklist before,
   even if this exact path hasn't. Real phishing infrastructure commonly
   reuses one malicious domain across many different paths/campaigns.
   Score scales with how many times that domain has been seen.
3. No match -- score 0. (Absence from a blacklist is not evidence of
   safety; this is exactly why the rule-based and LSTM engines exist
   alongside this one.)
"""

import hashlib

from sqlalchemy.orm import Session

from app.core.exceptions import InvalidScanInputError
from app.engines.rule_based.domain_typosquatting_detector import normalize_domain
from app.models import BlacklistEntry

DOMAIN_MATCH_BASE_SCORE = 60.0
DOMAIN_MATCH_SCORE_PER_EXTRA_HIT = 5.0
DOMAIN_MATCH_SCORE_CAP = 85.0  # stays below a typical exact-match score

# Shared hosting / URL-shortening / form-builder platforms discovered by
# analyzing the seeded blacklist itself: these dominate the "most repeated
# domain" list (e.g. docs.google.com: 5,539 entries) not because the domain
# is malicious, but because attackers abuse the platform's free public-share
# feature. Domain-level scoring is meaningless here -- virtually any path on
# docs.google.com could be legitimate or malicious, unrelated to prior
# entries. These get exact-match-only treatment; domain-level matches are
# suppressed (return 0) to avoid flagging the platform's legitimate traffic.
SHARED_HOSTING_DOMAINS = {
    "docs.google.com", "sites.google.com", "drive.google.com", "forms.gle",
    "qrco.de", "q-r.to", "l.ead.me", "l.wl.co", "did.li", "qr-codes.io",
    "bit.ly", "tinyurl.com", "t.co",
    "new.express.adobe.com", "dropbox.com",
    "eu.jotform.com", "app.jotform.com",
    "share-eu1.hsforms.com", "share.hsforms.com",
    "ipfs.io", "flowcode.com",
    "rebrand.ly", "t.me", "gravatar.com", "genspark.ai",  # found via Phase 5 calibration
    "beacons.ai", "taplink.cc",  # found via 2026-08 recalibration (link-in-bio platforms)
}


def _hash_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def calculate_blacklist_reputation_score(url: str, db: Session) -> dict:
    """
    Look up a URL against the threat-intelligence blacklist.

    Returns a dict:
        {
            "score": float,           # 0-100
            "match_type": str | None, # "exact" | "domain" | None
            "matched_entry": {...} | None,   # details of the matching row(s)
        }

    Raises InvalidScanInputError if url is not a non-empty string.
    """
    if not isinstance(url, str) or not url.strip():
        raise InvalidScanInputError("Input must be a non-empty string.")

    url_hash = _hash_url(url)

    exact_match = db.query(BlacklistEntry).filter_by(url_hash=url_hash, is_active=True).first()
    if exact_match is not None:
        return {
            "score": exact_match.reputation_score,
            "match_type": "exact",
            "matched_entry": {
                "url": exact_match.url,
                "source": exact_match.source,
                "brand_target": exact_match.brand_target,
                "first_seen": str(exact_match.first_seen),
                "last_seen": str(exact_match.last_seen),
            },
        }

    domain = normalize_domain(url)

    if domain in SHARED_HOSTING_DOMAINS:
        # Exact-match-only for these platforms -- domain-level "seen before"
        # is not a meaningful signal here (see SHARED_HOSTING_DOMAINS above).
        return {"score": 0.0, "match_type": None, "matched_entry": None}

    domain_matches = (
        db.query(BlacklistEntry).filter_by(domain=domain, is_active=True).all()
    )
    if domain_matches:
        hit_count = len(domain_matches)
        score = min(
            DOMAIN_MATCH_SCORE_CAP,
            DOMAIN_MATCH_BASE_SCORE + (hit_count - 1) * DOMAIN_MATCH_SCORE_PER_EXTRA_HIT,
        )
        most_recent = max(domain_matches, key=lambda e: e.last_seen)
        return {
            "score": score,
            "match_type": "domain",
            "matched_entry": {
                "domain": domain,
                "hit_count": hit_count,
                "most_recent_source": most_recent.source,
                "most_recent_last_seen": str(most_recent.last_seen),
            },
        }

    return {"score": 0.0, "match_type": None, "matched_entry": None}
