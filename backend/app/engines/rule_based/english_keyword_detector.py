"""
english_keyword_detector.py

Rule-based sub-engine: scores English-language text for generic
scam/phishing signals (lottery/prize scams, advance-fee/inheritance
scams, generic account-phishing pressure language).

Scope note (deliberately "light" coverage, not a full English NLP
engine): ETIS's core scope is Arabic-language, Egypt-specific phishing
(see CLAUDE.md). This detector was added to close a real gap found
during testing: a purely English-language scam text previously scored
near-zero, because arabic_keyword_detector.py only matches Arabic
terms, and the LSTM's training data is ~99.9% Arabic/mixed for text
rows (see CLAUDE.md "LSTM scope"), so it has essentially never seen
English phishing examples. This module is a narrow, generic-scam-
pattern keyword list -- not a general English NLP classifier -- added
specifically to give English-only submissions a non-zero rule-based
signal. Same weighted-tier design as arabic_keyword_detector.py, so it
combines cleanly via noisy-OR in rule_based_engine.py.
"""

import re
from app.core.exceptions import InvalidScanInputError

# ---------------------------------------------------------------------------
# Weighted keyword tiers
# ---------------------------------------------------------------------------

# HIGH: specific, low-ambiguity scam phrases.
HIGH_WEIGHT_TERMS = {
    "wire transfer": "requests a wire transfer",
    "wire the funds": "requests a wire transfer (alt. phrasing)",
    "claim your prize": "prize-claim scam phrasing",
    "verify your account": "credential-harvesting phrasing",
    "confirm your identity": "identity-confirmation phrasing",
    "bank account number": "requests bank account number",
    "social security number": "requests SSN",
    "you have been selected": "lottery/prize selection scam opener",
    "processing fee": "advance-fee scam signal",
    "next of kin": "advance-fee/inheritance scam phrasing",
}

# MEDIUM: suspicious but more ambiguous on their own.
MEDIUM_WEIGHT_TERMS = {
    "lottery": "lottery reference",
    "prize": "prize reference",
    "winner": "winner reference",
    "inheritance": "inheritance reference",
    "beneficiary": "beneficiary reference",
    "sweepstakes": "sweepstakes reference",
    "tax-free": "tax-free reference (common lottery-scam phrasing)",
    "act now": "urgency pressure",
    "urgent": "urgency pressure",
    "immediately": "urgency pressure",
    "click the link": "click-through pressure",
    "do not share this information": "isolation/secrecy pressure",
    "confidential": "secrecy pressure",
}

# LOW: generic words, common in ordinary English text. Only scored if 2+
# distinct low-weight terms appear (clustering signal), same rule as the
# Arabic detector.
LOW_WEIGHT_TERMS = {
    "account": "generic account reference",
    "verify": "generic verification reference",
    "link": "generic link reference",
    "identity": "generic identity reference",
    "phone number": "generic phone number reference",
    "address": "generic address reference",
    "reply": "generic reply reference",
    "confirm": "generic confirmation reference",
}

HIGH_WEIGHT = 3
MEDIUM_WEIGHT = 2
LOW_WEIGHT = 1
LOW_WEIGHT_MIN_CLUSTER = 2

# Same normalization divisor as arabic_keyword_detector.py, so the two
# engines' raw scores stay on a comparable scale for noisy-OR combination.
SCORE_NORMALIZATION_DIVISOR = 10


def normalize_english_text(text: str) -> str:
    """Lowercase and collapse whitespace. No stemming/lemmatization --
    matches the "light coverage" scope: a generic-scam-phrase keyword
    list, not a general English NLP pipeline."""
    normalized = text.lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def calculate_english_keyword_phishing_score(text: str) -> dict:
    """
    Score English text for generic scam/phishing keyword signals.

    Returns a dict:
        {
            "score": float,               # 0-100
            "matched_high": [(term, meaning), ...],
            "matched_medium": [(term, meaning), ...],
            "matched_low": [(term, meaning), ...],
            "low_weight_counted": bool,
        }

    Raises InvalidScanInputError if text is not a non-empty string.
    """
    if not isinstance(text, str) or not text.strip():
        raise InvalidScanInputError("Input text must be a non-empty string.")

    normalized = normalize_english_text(text)

    matched_high = [(term, meaning) for term, meaning in HIGH_WEIGHT_TERMS.items() if term in normalized]
    matched_medium = [(term, meaning) for term, meaning in MEDIUM_WEIGHT_TERMS.items() if term in normalized]
    matched_low = [(term, meaning) for term, meaning in LOW_WEIGHT_TERMS.items() if term in normalized]

    raw_score = len(matched_high) * HIGH_WEIGHT + len(matched_medium) * MEDIUM_WEIGHT

    low_weight_counted = len(matched_low) >= LOW_WEIGHT_MIN_CLUSTER
    if low_weight_counted:
        raw_score += len(matched_low) * LOW_WEIGHT

    score = min(100.0, round(100 * raw_score / SCORE_NORMALIZATION_DIVISOR, 2))

    return {
        "score": score,
        "matched_high": matched_high,
        "matched_medium": matched_medium,
        "matched_low": matched_low,
        "low_weight_counted": low_weight_counted,
    }
