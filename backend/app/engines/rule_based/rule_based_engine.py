"""
rule_based_engine.py

Combines the rule-based sub-engines into a single explainable
rule-based sub-score (0-100):

  1. Arabic keyword detector    (arabic_keyword_detector.py)
  2. English keyword detector   (english_keyword_detector.py) -- added
     to close a gap where English-only text got near-zero coverage
     from both the Arabic-only keyword list and the Arabic-trained
     LSTM (see english_keyword_detector.py's module docstring)
  3. Domain typosquatting check (domain_typosquatting_detector.py)
  4. URL pattern analyzer       (url_pattern_analyzer.py)

This sub-score is what feeds into the unified risk-scoring layer (Phase 5),
where it will be weighted against the threat-intel blacklist and LSTM
sub-scores per CLAUDE.md's 30/40/30 point architecture.

Combination method: noisy-OR, not a plain average.

Why: not every scan input has both text and a URL (a bare URL scan can't
run the keyword detectors; plain text with no link can't run the URL-based
checks). A plain average would badly understate a strong single-engine
signal -- e.g. a domain-embedding typosquat scoring 95 would get diluted
if averaged against engines that simply had nothing to analyze and
defaulted to 0. Noisy-OR treats each engine's score as an independent
probability estimate and combines only the engines that actually ran:

    combined = 1 - PRODUCT(1 - score_i / 100) over applicable engines

This keeps one confident detection confident, regardless of how many other
engines had no input to work with.
"""

from app.core.exceptions import InvalidScanInputError
from app.engines.rule_based.arabic_keyword_detector import calculate_arabic_keyword_phishing_score
from app.engines.rule_based.english_keyword_detector import calculate_english_keyword_phishing_score
from app.engines.rule_based.domain_typosquatting_detector import calculate_domain_typosquatting_score
from app.engines.rule_based.url_pattern_analyzer import calculate_url_pattern_score


def calculate_rule_based_score(text: str | None = None, url: str | None = None) -> dict:
    """
    Run all applicable rule-based sub-engines and combine their scores.

    At least one of `text` or `url` must be provided (non-empty).

    Returns a dict:
        {
            "score": float,   # 0-100, combined via noisy-OR
            "engine_results": {
                "arabic_keyword":       {...} | None,  # None if not applicable (no text)
                "english_keyword":      {...} | None,  # None if not applicable (no text)
                "domain_typosquatting": {...} | None,  # None if not applicable (no url)
                "url_pattern":          {...} | None,  # None if not applicable (no url)
            },
        }

    Raises InvalidScanInputError if neither text nor url is provided.
    """
    text_provided = isinstance(text, str) and text.strip() != ""
    url_provided = isinstance(url, str) and url.strip() != ""

    if not text_provided and not url_provided:
        raise InvalidScanInputError("At least one of 'text' or 'url' must be provided.")

    engine_results = {
        "arabic_keyword": None,
        "english_keyword": None,
        "domain_typosquatting": None,
        "url_pattern": None,
    }
    applicable_probabilities = []

    if text_provided:
        keyword_result = calculate_arabic_keyword_phishing_score(text)
        engine_results["arabic_keyword"] = keyword_result
        applicable_probabilities.append(keyword_result["score"] / 100)

        english_keyword_result = calculate_english_keyword_phishing_score(text)
        engine_results["english_keyword"] = english_keyword_result
        applicable_probabilities.append(english_keyword_result["score"] / 100)

    if url_provided:
        typosquat_result = calculate_domain_typosquatting_score(url)
        engine_results["domain_typosquatting"] = typosquat_result
        applicable_probabilities.append(typosquat_result["score"] / 100)

        url_pattern_result = calculate_url_pattern_score(url)
        engine_results["url_pattern"] = url_pattern_result
        applicable_probabilities.append(url_pattern_result["score"] / 100)

    combined_probability = 1.0
    for p in applicable_probabilities:
        combined_probability *= (1 - p)
    combined_probability = 1 - combined_probability

    combined_score = round(100 * combined_probability, 2)

    return {
        "score": combined_score,
        "engine_results": engine_results,
    }
