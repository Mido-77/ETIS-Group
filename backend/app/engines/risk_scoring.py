"""
risk_scoring.py

The unified risk-scoring layer. Combines the detection engines into one
LOW / MEDIUM / HIGH verdict (plus a 0-100 display score) with a per-engine
explainability breakdown.

DESIGN (decision tiers, adopted 2026-10 after evaluation/evaluate_fusion.py)
---------------------------------------------------------------------------
Earlier versions ADDED engine points (30/40/30, later 45/15 for text). On an
honest evaluation (blacklist knowing only training URLs) that design left
67.8% of unseen phishing URLs at LOW and put 99% of legitimate messages at
MEDIUM (LOW was unreachable for text). Averaging unlike evidence also let a
clean second input dilute a strong signal. The replacement is simpler:

    every applicable engine gives its OWN tier (LOW / MEDIUM / HIGH);
    the verdict is the HIGHEST tier any engine reaches.

Engines and their tiers
    rule_based   URL part  (typosquat + URL pattern, noisy-OR): >= 40 MEDIUM, >= 90 HIGH
                 text part (Arabic + English keywords, noisy-OR): >= 30 MEDIUM, >= 60 HIGH
    blacklist    exact URL match -> HIGH; domain-level match -> MEDIUM; none -> LOW
    url_model    learned URL ensemble (ml/models/url_ensemble.joblib): MEDIUM / HIGH at the
                 validation thresholds for 1% / 0.1% false positives on legitimate URLs
    text_model   learned character TF-IDF + logistic regression
                 (ml/models/text_char_logreg.joblib): MEDIUM / HIGH at the validation
                 thresholds for 5% / 1% false positives on legitimate messages

Applicability: URL only -> rule_based + blacklist + url_model; text only ->
rule_based + text_model; both -> all four. Adding an input can only add
evidence: the verdict of a URL+text submission is never lower than that of
the URL alone or the text alone.

Display score: each engine's value is mapped piecewise-linearly into the
band of its tier (LOW 0-33, MEDIUM 34-66, HIGH 67-100) and final_score is the
maximum, so the 0-100 number and the verdict can never disagree
(final_score < 34 -> LOW, < 67 -> MEDIUM, else HIGH).

Egyptian brand allowlist: a URL whose host is a verified Egyptian brand
domain (or a genuine subdomain, e.g. online.cibeg.com) is the brand's own
site, so the URL-derived signals (URL part of rule_based, blacklist,
url_model) are set to LOW. The allowlist does NOT cancel evidence from the
message text: a phishing text that links to a real bank URL still gets its
text tier. engine_details always shows each engine's real, un-overridden
output; only engine_tiers / final_score / verdict reflect the override.

What the evaluation does and does not support (see Chapter 5)
    - Unseen-host URLs: the URL model reaches ROC-AUC 0.995; MEDIUM-or-higher
      flags ~93% of phishing at ~0.8% false positives, HIGH ~86% at ~0.0%
      (TLD-masked v2 model; the v1 model reached ~66% at HIGH and lost ~68
      points of recall when a phishing URL's TLD was swapped to .eg).
    - In the evaluation database the blacklist cannot fire on unseen URLs, so
      its value for KNOWN threats is not measured there; "exact blacklist hit
      -> HIGH" is a design rule, not a measured effect.
    - The rule engine's Egyptian-brand typosquat tier (>= 90 -> HIGH) is evaluated
      on constructed benchmarks only (data/benchmarks/); the dataset contains no
      real Egyptian-brand look-alike URLs.
    - URLs with a path have a higher false-positive rate (~2.8% at MEDIUM)
      than bare hosts (~0%).

History (kept because the thesis discusses it): the additive design had a
CALIBRATED_LOW_MAX/CALIBRATED_MEDIUM_MAX percentile calibration and a 45/15
text-only reweighting (found to be silently lost once from this file). Both
are gone; tier thresholds come from fixed false-positive rates chosen on the
validation split. evaluation/calibrate_thresholds.py is obsolete.

Other response fields
    skip_reasons    machine-readable reason each engine did not run
    stage_timings   per-engine and total milliseconds (explainability only)
    confidence      heuristic coverage/agreement of the raw engine scores; kept
                    for API compatibility but no longer shown in the dashboard
                    (with one deciding engine, low "agreement" with engines
                    that have nothing to say reads as doubt, which it is not)
    engine_tiers    tier of each applicable engine after the allowlist override
    decided_by      engine that produced the verdict (None when LOW)
"""

import time

from sqlalchemy.orm import Session

from app.core.egyptian_brands import EGYPTIAN_BRAND_DOMAIN_SET
from app.core.exceptions import InvalidScanInputError
from app.core.url_validation import ensure_url_has_host
from app.engines.ml.text_char_inference import calculate_text_char_phishing_score
from app.engines.ml.url_model_inference import calculate_url_model_score
from app.engines.rule_based.domain_typosquatting_detector import (
    is_plain_hostname,
    normalize_domain,
)
from app.engines.rule_based.rule_based_engine import calculate_rule_based_score
from app.engines.threat_intel.blacklist_lookup import calculate_blacklist_reputation_score

# --- verdict bands on the 0-100 display score -------------------------------
VERDICT_MEDIUM_MIN = 34.0
VERDICT_HIGH_MIN = 67.0

# --- rule-based tier thresholds on its 0-100 score (design-time, from the
# fusion evaluation: URL part 40/90, text part 30/60) --------------------------
RULE_TIER_THRESHOLDS = {
    "url": (40.0, 90.0),   # (MEDIUM min, HIGH min)
    "text": (30.0, 60.0),
}

ENGINE_ORDER = ["rule_based", "blacklist", "url_model", "text_model"]
TOTAL_ENGINE_COUNT = len(ENGINE_ORDER)  # used by the heuristic confidence "coverage"

SKIP_REASON_NO_URL_PROVIDED = "no_url_provided"
SKIP_REASON_NO_TEXT_PROVIDED = "no_text_provided"

CONFIDENCE_COVERAGE_WEIGHT = 0.5
CONFIDENCE_AGREEMENT_WEIGHT = 0.5


# ---------------------------------------------------------------------------
# Tier / band helpers
# ---------------------------------------------------------------------------

def _tier_from_score(score: float) -> str:
    if score >= VERDICT_HIGH_MIN:
        return "HIGH"
    if score >= VERDICT_MEDIUM_MIN:
        return "MEDIUM"
    return "LOW"


def _band_score(value: float, medium_min: float, high_min: float, upper: float) -> float:
    """Map an engine value onto the 0-100 display bands by tier:
    below medium_min -> 0..33, medium_min..high_min -> 34..66, >= high_min -> 67..100."""
    value = max(0.0, min(value, upper))
    if value >= high_min:
        span = max(upper - high_min, 1e-9)
        return round(67.0 + (value - high_min) / span * 33.0, 2)
    if value >= medium_min:
        span = max(high_min - medium_min, 1e-9)
        return round(min(66.0, 34.0 + (value - medium_min) / span * 32.0), 2)
    if medium_min <= 0:
        return 0.0
    return round(min(33.0, value / medium_min * 33.0), 2)


def _noisy_or(scores_0_100: list[float]) -> float:
    remaining = 1.0
    for score in scores_0_100:
        remaining *= 1.0 - max(0.0, min(score, 100.0)) / 100.0
    return 100.0 * (1.0 - remaining)


def _model_decision_score(result: dict) -> float:
    thresholds = result["thresholds"]
    return _band_score(result["probability"], thresholds["medium"], thresholds["high"], 1.0)


def _blacklist_decision_score(result: dict) -> float:
    match_type = result.get("match_type")
    score = float(result.get("score") or 0.0)
    if match_type == "exact":
        return round(67.0 + min(100.0, score) / 100.0 * 33.0, 2)
    if match_type == "domain":
        return round(min(66.0, 34.0 + max(0.0, score - 60.0) / 25.0 * 32.0), 2)
    return 0.0


def _rule_based_decision_score(rule_result: dict, url_provided: bool, text_provided: bool,
                               ignore_url_part: bool) -> float:
    """Tier-band score of the rule engine, judged per input kind: the URL part
    (typosquat + URL pattern) against 40/90, the text part (Arabic + English
    keywords) against 30/60. ignore_url_part drops the URL part (allowlisted
    brand domain)."""
    results = rule_result.get("engine_results") or {}

    def sub_score(name: str):
        entry = results.get(name)
        return None if not entry else entry["score"]

    candidates: list[float] = []
    if url_provided and not ignore_url_part:
        parts = [s for s in (sub_score("domain_typosquatting"), sub_score("url_pattern")) if s is not None]
        if parts:
            medium, high = RULE_TIER_THRESHOLDS["url"]
            candidates.append(_band_score(_noisy_or(parts), medium, high, 100.0))
    if text_provided:
        parts = [s for s in (sub_score("arabic_keyword"), sub_score("english_keyword")) if s is not None]
        if parts:
            medium, high = RULE_TIER_THRESHOLDS["text"]
            candidates.append(_band_score(_noisy_or(parts), medium, high, 100.0))
    return max(candidates) if candidates else 0.0


# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------

def _is_verified_egyptian_domain(url: str) -> bool:
    """
    True when the submitted URL's host IS a verified Egyptian brand domain
    (app/core/egyptian_brands.py), or a genuine subdomain of one -- i.e. the
    real bank/telecom/service's own infrastructure, not something that
    merely resembles it. Typosquats, brand-label reuse and domain embedding
    ("cibeg.com.verify-now.net") all fail this check by design.

    Subdomains are trusted on the assumption that a brand controls
    everything under its own registered domain (a dangling-DNS subdomain
    takeover would defeat that, which is a stated limitation). The subdomain
    rule only applies to plain ASCII hostnames (is_plain_hostname): URLs
    containing a backslash, '@', '%' or non-ASCII characters are never
    treated as subdomains, because browsers read those differently from a
    naive string comparison (e.g. 'evil.net\\\\.cibeg.com' is a request to
    evil.net).
    """
    host = normalize_domain(url)
    if host in EGYPTIAN_BRAND_DOMAIN_SET:
        return True
    if not is_plain_hostname(host):
        return False
    return any(host.endswith("." + brand_domain) for brand_domain in EGYPTIAN_BRAND_DOMAIN_SET)


# ---------------------------------------------------------------------------
# Heuristic confidence (kept for API compatibility; not shown in the UI)
# ---------------------------------------------------------------------------

def _calculate_confidence(engine_scores: dict[str, float]) -> dict:
    """coverage = engines applied / 4; agreement = 100 - (max - min) of the raw
    0-100 engine scores; score = 0.5 * coverage + 0.5 * agreement. A stated
    design choice, not a calibrated probability."""
    applicable_count = len(engine_scores)
    coverage = round((applicable_count / TOTAL_ENGINE_COUNT) * 100, 2)
    scores = list(engine_scores.values())
    agreement = round(100.0 - (max(scores) - min(scores)), 2)
    confidence_score = round(
        CONFIDENCE_COVERAGE_WEIGHT * coverage + CONFIDENCE_AGREEMENT_WEIGHT * agreement, 2
    )
    return {
        "score": confidence_score,
        "coverage": coverage,
        "agreement": agreement,
        "engines_considered": applicable_count,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def calculate_final_risk_score(
    text: str | None = None,
    url: str | None = None,
    db: Session | None = None,
) -> dict:
    """
    Run all applicable engines and combine them by decision tiers.

    At least one of `text` or `url` must be provided. `db` is required
    whenever `url` is provided (needed for the blacklist lookup) -- this is
    a caller/integration error, not a user-input error.

    Returns a dict:
        {
            "final_score": float,          # 0-100 display score (max over engines' band scores)
            "verdict": str,                # "LOW" | "MEDIUM" | "HIGH" (highest engine tier)
            "engines_applied": [str, ...], # subset of rule_based/blacklist/url_model/text_model
            "engine_tiers": {engine: "LOW"|"MEDIUM"|"HIGH"},   # after the allowlist override
            "decided_by": str | None,      # engine that produced the verdict; None when LOW
            "engine_details": {
                "rule_based": {...},           # always present
                "blacklist": {...} | None,     # None if no url
                "url_model": {...} | None,     # None if no url
                "text_model": {...} | None,    # None if no text
            },
            "skip_reasons": {"blacklist": str|None, "url_model": str|None, "text_model": str|None},
            "stage_timings": {"rule_based_ms", "blacklist_ms", "url_model_ms",
                              "text_model_ms", "total_ms"},   # engines that did not run are None
            "confidence": {...},           # heuristic, see _calculate_confidence
            "allowlisted": bool,           # URL is a verified Egyptian brand domain
        }

    Raises InvalidScanInputError if neither text nor url is provided, or if the url has no host.
    Raises ValueError if url is provided without a db session.
    """
    text_provided = isinstance(text, str) and text.strip() != ""
    url_provided = isinstance(url, str) and url.strip() != ""

    if not text_provided and not url_provided:
        raise InvalidScanInputError("At least one of 'text' or 'url' must be provided.")

    if url_provided:
        ensure_url_has_host(url)  # raises InvalidScanInputError for "https://" and similar

    if url_provided and db is None:
        raise ValueError(
            "A database session is required when 'url' is provided "
            "(needed for the blacklist lookup)."
        )

    total_start = time.perf_counter()

    raw_scores: dict[str, float] = {}  # each engine's own 0-100 score, for the heuristic confidence
    engine_details: dict[str, dict | None] = {
        "rule_based": None, "blacklist": None, "url_model": None, "text_model": None,
    }
    stage_timings: dict[str, float | None] = {
        "rule_based_ms": None, "blacklist_ms": None, "url_model_ms": None, "text_model_ms": None,
    }

    def timed(name: str, call):
        start = time.perf_counter()
        result = call()
        stage_timings[f"{name}_ms"] = round((time.perf_counter() - start) * 1000, 2)
        return result

    # Rule-based always runs: it handles text-only / url-only / both itself.
    rule_result = timed("rule_based", lambda: calculate_rule_based_score(text=text, url=url))
    engine_details["rule_based"] = rule_result
    raw_scores["rule_based"] = rule_result["score"]

    if url_provided:
        blacklist_result = timed("blacklist", lambda: calculate_blacklist_reputation_score(url, db))
        engine_details["blacklist"] = blacklist_result
        raw_scores["blacklist"] = blacklist_result["score"]

        url_model_result = timed("url_model", lambda: calculate_url_model_score(url))
        engine_details["url_model"] = url_model_result
        raw_scores["url_model"] = url_model_result["score"]

    if text_provided:
        text_model_result = timed("text_model", lambda: calculate_text_char_phishing_score(text))
        engine_details["text_model"] = text_model_result
        raw_scores["text_model"] = text_model_result["score"]

    skip_reasons: dict[str, str | None] = {
        "blacklist": None if url_provided else SKIP_REASON_NO_URL_PROVIDED,
        "url_model": None if url_provided else SKIP_REASON_NO_URL_PROVIDED,
        "text_model": None if text_provided else SKIP_REASON_NO_TEXT_PROVIDED,
    }

    applicable_engines = [e for e in ENGINE_ORDER if engine_details[e] is not None]
    confidence = _calculate_confidence(raw_scores)

    # Allowlist: the brand's own domain cancels the URL-derived signals only.
    allowlisted = url_provided and _is_verified_egyptian_domain(url)

    decision_scores: dict[str, float] = {
        "rule_based": _rule_based_decision_score(
            rule_result, url_provided, text_provided, ignore_url_part=allowlisted
        ),
    }
    if url_provided:
        decision_scores["blacklist"] = 0.0 if allowlisted else _blacklist_decision_score(engine_details["blacklist"])
        decision_scores["url_model"] = 0.0 if allowlisted else _model_decision_score(engine_details["url_model"])
    if text_provided:
        decision_scores["text_model"] = _model_decision_score(engine_details["text_model"])

    engine_tiers = {engine: _tier_from_score(decision_scores[engine]) for engine in applicable_engines}
    final_score = round(max(decision_scores.values()), 2)
    verdict = _tier_from_score(final_score)
    decided_by = None
    if verdict != "LOW":
        decided_by = max(applicable_engines, key=lambda engine: decision_scores[engine])

    stage_timings["total_ms"] = round((time.perf_counter() - total_start) * 1000, 2)

    return {
        "final_score": final_score,
        "verdict": verdict,
        "engines_applied": applicable_engines,
        "engine_tiers": engine_tiers,
        "decided_by": decided_by,
        "engine_details": engine_details,
        "skip_reasons": skip_reasons,
        "stage_timings": stage_timings,
        "confidence": confidence,
        "allowlisted": allowlisted,
    }
