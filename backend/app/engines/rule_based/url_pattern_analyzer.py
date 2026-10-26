"""
url_pattern_analyzer.py

Rule-based sub-engine: scores a URL for structural phishing red flags,
independent of any specific brand (that's the typosquatting detector's job).
Flag design informed by the PhiUSIIL Phishing URL Dataset's published
feature set (URLSimilarityIndex, HasObfuscation, NoOfSubDomain, etc. --
see docs/rule_based_engine_design.md) and the URL-shortener patterns
observed in the GitHub Arabic-Text-Spam-Detection blacklist (source #11,
Phase 1) -- all of which were bit.ly links.
"""

import re
from urllib.parse import urlparse
from app.core.exceptions import InvalidScanInputError


# ---------------------------------------------------------------------------
# Flag weights, tiered by how strong a standalone signal each pattern is.
# ---------------------------------------------------------------------------

HIGH_WEIGHT = 25    # strong, low-ambiguity indicators
MEDIUM_WEIGHT = 15  # suspicious but more context-dependent
LOW_WEIGHT = 8      # weak signals, only meaningful in combination

KNOWN_URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "shorte.st", "cutt.ly", "rb.gy", "shorturl.at", "tiny.cc",
}

SUSPICIOUS_PATH_KEYWORDS = re.compile(
    r"secure|verify|login|update|confirm|account|wallet|unlock|alert|"
    r"suspend|billing|refund|activate",
    re.IGNORECASE,
)

MAX_REASONABLE_URL_LENGTH = 75
MAX_REASONABLE_HYPHENS_IN_DOMAIN = 2
MAX_REASONABLE_SUBDOMAIN_LEVELS = 3  # e.g. a.b.c.example.com = 3 levels before "example.com"

_IPV4_PATTERN = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")


def _is_ip_address_host(hostname: str) -> bool:
    if not hostname:
        return False
    if _IPV4_PATTERN.match(hostname):
        return True
    if ":" in hostname:  # crude IPv6 check -- hostnames don't normally contain colons
        return True
    return False


def calculate_url_pattern_score(url: str) -> dict:
    """
    Score a URL for structural phishing red flags.

    Returns a dict:
        {
            "score": float,     # 0-100
            "flags": [(flag_name, weight, detail), ...],
        }

    Raises InvalidScanInputError if url is not a non-empty string.
    """
    if not isinstance(url, str) or not url.strip():
        raise InvalidScanInputError("Input must be a non-empty string.")

    raw = url.strip()
    # urlparse needs a scheme to correctly split netloc from path; add one if
    # missing. That added scheme is a PARSING aid only: it must never be reported
    # as the user's scheme (see no_https below).
    parsed_input = raw if "://" in raw else f"http://{raw}"
    try:
        parsed = urlparse(parsed_input)
        hostname = parsed.hostname or ""
        port = parsed.port  # raises ValueError for non-numeric / out-of-range ports
    except ValueError:
        # "javascript:alert(1)", "http://[::1", "http://host:99999/" ... A URL no
        # browser can open is itself a (weak) red flag, and must not crash the API.
        return {"score": float(MEDIUM_WEIGHT),
                "flags": [("malformed_url", MEDIUM_WEIGHT, "URL could not be parsed")]}
    flags = []

    # --- HIGH weight ---
    if _is_ip_address_host(hostname):
        flags.append(("ip_address_host", HIGH_WEIGHT, hostname))

    if "@" in parsed.netloc:  # userinfo before the host; an '@' in a query string is harmless
        flags.append(("at_symbol_redirect_trick", HIGH_WEIGHT, "'@' present before path, browsers ignore everything before it"))

    if hostname.startswith("xn--") or ".xn--" in hostname:
        flags.append(("punycode_domain", HIGH_WEIGHT, hostname))

    # --- MEDIUM weight ---
    if hostname in KNOWN_URL_SHORTENERS:
        flags.append(("known_url_shortener", MEDIUM_WEIGHT, hostname))

    subdomain_levels = max(0, hostname.count(".") - 1)  # e.g. "a.b.example.com" -> 2
    if subdomain_levels > MAX_REASONABLE_SUBDOMAIN_LEVELS:
        flags.append(("excessive_subdomains", MEDIUM_WEIGHT, f"{subdomain_levels} levels"))

    if port is not None and port not in (80, 443):
        flags.append(("non_standard_port", MEDIUM_WEIGHT, str(port)))

    # --- LOW weight ---
    if SUSPICIOUS_PATH_KEYWORDS.search(parsed.path) or SUSPICIOUS_PATH_KEYWORDS.search(parsed.query):
        flags.append(("suspicious_path_keyword", LOW_WEIGHT, parsed.path))

    hyphen_count = hostname.count("-")
    if hyphen_count > MAX_REASONABLE_HYPHENS_IN_DOMAIN:
        flags.append(("excessive_hyphens", LOW_WEIGHT, f"{hyphen_count} hyphens"))

    if len(raw) > MAX_REASONABLE_URL_LENGTH:
        flags.append(("excessive_url_length", LOW_WEIGHT, f"{len(raw)} chars"))

    # Only when the user's own text says http://. A URL typed without any scheme
    # ("wikipedia.org") says nothing about https, and flagging it was a false
    # positive on every scheme-less input (found in the 2026-10 browser smoke test).
    if raw.lower().startswith("http://"):
        flags.append(("no_https", LOW_WEIGHT, None))

    score = min(100.0, float(sum(weight for _, weight, _ in flags)))

    return {"score": score, "flags": flags}
