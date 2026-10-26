"""
domain_typosquatting_detector.py

Rule-based sub-engine: checks a URL/domain against the Egyptian brand
registry (app/core/egyptian_brands.py) for typosquatting using the
techniques below, checked in order of confidence:

1. Exact match -> not typosquatting, it IS the real brand domain.
1b. Real brand subdomain (added 2026-10) -> the host is a subdomain of a
   real brand domain (e.g. "online.cibeg.com" under "cibeg.com"). The
   brand owns everything under its own domain, so this is not
   impersonation: score 0, technique "brand_subdomain". matched_brand is
   deliberately left None (unlike exact_match) so the dashboard doesn't
   print "Impersonates: <brand>" next to a genuine brand host.
2. Full-domain embedding -> the brand's real domain string appears verbatim
   inside a longer, different domain (e.g. "cibeg.com.verify-account.net").
   This is a very common, highly deceptive real phishing technique and gets
   the highest confidence score. The brand string must sit on a domain
   boundary -- not directly preceded or followed by a letter or digit --
   (fixed 2026-10): without that, "afternoon.com" matched "noon.com" and
   "website.eg" matched "te.eg" at 95 points.
3. Homoglyph/confusable-character detection (added 2026-08) -> the domain
   uses visually-confusable Unicode characters (Cyrillic, Greek lookalikes,
   or punycode-encoded IDN labels that decode to such characters) instead
   of the real ASCII letters, so it RENDERS identically to a real brand
   domain in a browser address bar while being a completely different
   string to a computer. Checked BEFORE Levenshtein deliberately -- a
   single confusable-character substitution is also just "1 edit" in plain
   Levenshtein distance terms, so checking Levenshtein first would match
   and short-circuit before this more specific technique ever ran (found
   and fixed during testing, see test_homoglyph_punycode_encoded_domain_detected
   and its siblings in backend/tests/test_domain_typosquatting_detector.py).
   This is distinct from technique 4 (Levenshtein) and technique 5 (brand-
   label reuse) -- neither catches this on its own terms, since the
   confusable characters aren't "close" in an edit-distance sense, they
   just visually resemble the real letter. See CONFUSABLE_CHAR_MAP below
   for the curated (not exhaustive) set of characters covered -- this is a
   deliberately scoped subset of the most common phishing homoglyphs, not
   the full ~6,000-entry Unicode Consortium confusables table, consistent
   with this project's "don't over-invest" testing/scope philosophy
   (CLAUDE.md). Complements url_pattern_analyzer.py's existing punycode_domain
   flag, which detects that a domain IS punycode-encoded but doesn't decode
   it or identify which brand (if any) it's impersonating -- this technique
   does both.
4. Levenshtein edit-distance closeness -> classic character-level typosquats
   (e.g. "cib3g.com", "cibeg-eg.com"). Only applied to brand domains at least
   MIN_DOMAIN_LENGTH_FOR_LEVENSHTEIN characters long, since edit-distance
   comparisons against very short domains (e.g. "te.eg") produce unreliable,
   false-positive-prone results -- lots of unrelated short domains sit within
   1-2 edits of a 5-character string by chance.
1c. Brand in the userinfo part (added 2026-10) -> "http://cibeg.com@evil.net/":
   the real host is evil.net; the brand sits before the '@'. Score 95,
   technique "userinfo_embedding". normalize_domain() now returns the real
   host (evil.net), so this check reads the userinfo separately.
1d. Brand before a backslash in the authority (added 2026-10) ->
   "https://evil.net\\.cibeg.com/login": browsers read the backslash as the
   start of the path (real host evil.net) while a human reads cibeg.com.
   Score 95, technique "backslash_embedding".
5. Brand-label reuse -> the brand's core label (domain minus TLD) appears as
   a substring alongside a suspicious modifier, but the full domain isn't
   close enough to be caught by rule 4 (e.g. "secure-cibeg-support.net").
   Short labels (SHORT_LABEL_MAX_LENGTH_REQUIRING_BOUNDARY characters or
   fewer, e.g. "noon", "aaib") must also sit on a boundary, otherwise they
   match inside ordinary words ("afternoon-login.com").
6. Brand tokens (added 2026-10, after evaluation/evaluate_typosquat_benchmark.py
   showed technique 5 missed most hyphenated look-alikes): the host's labels
   are split on hyphens and compared TOKEN BY TOKEN against each brand's
   label and official aliases (EGYPTIAN_BRAND_ALIAS_LABELS). Whole-token
   comparison is what keeps "afternoon-login.com" at 0 while catching
   "nbe-egypt.com", "support-nbe.org" and "nbe.verify-egypt.xyz" (technique 5
   skips labels shorter than 4 characters, so it could never see "nbe").
   - brand_token_combo   a brand token next to a context word (verify, login,
     egypt, online, app, ...) -> 70. A distinctive brand token (5+ letters,
     not a multinational name) next to ANY other token -> 60.
   - brand_hyphen_split  adjacent tokens that join into a brand label
     ("banque-misr.com", "qnb-alahli.com") -> 70.
   Multinational labels (vodafone, orange, amazon, ...) are not flagged on a
   plain country-code domain ("vodafone-support.de" is the real company).
7. brand_tld_swap (added 2026-10): the registered label equals a DISTINCTIVE
   brand label under a different suffix ("cibeg.info") -> 60. Short or
   multinational labels are only flagged under abuse-prone TLDs, because
   "nbe.com" / "orange.fr" / "aib.com"-style domains belong to other,
   legitimate organisations and cannot be told apart by string alone.
8. brand_in_path (added 2026-10): a verified brand DOMAIN string appears in
   the path or query of a URL on some other host
   ("evil.top/cibeg.com/login") -> 60. Capped at MEDIUM on purpose: honest
   pages (news, archives, link lists) also mention brand domains in paths.
   Only brand domains of 7+ characters are checked ("te.eg" is too short).
Scores of 60-70 land in the rule engine's MEDIUM tier (>= 40) and never reach
HIGH (>= 90) on their own: string-only evidence is not decisive evidence.
"""

import re
from urllib.parse import unquote, urlsplit

from app.core.egyptian_brands import EGYPTIAN_BRAND_ALIAS_LABELS, EGYPTIAN_BRAND_DOMAINS
from app.core.exceptions import InvalidScanInputError


MIN_DOMAIN_LENGTH_FOR_LEVENSHTEIN = 9
MIN_LABEL_LENGTH_FOR_REUSE_CHECK = 4
SHORT_LABEL_MAX_LENGTH_REQUIRING_BOUNDARY = 4
LEVENSHTEIN_SIMILARITY_THRESHOLD = 0.75
HOMOGLYPH_SCORE = 92.0  # just below domain_embedding's 95 -- both are highly
                         # deceptive, deliberate impersonation techniques,
                         # but domain embedding additionally still contains
                         # the brand's real string verbatim, which is a
                         # marginally stronger signal than a visual-only match

# Brand-label reuse (technique 5) requires one of these nearby, not just bare
# substring containment -- otherwise legitimate infrastructure domains that
# happen to contain a brand's label (e.g. "amazonaws.com", "nooncdn.com") or
# domains containing an accidental common-word match (e.g. "digital" from
# "digital.gov.eg") get flagged incorrectly.
SUSPICIOUS_MODIFIER_PATTERN = re.compile(
    r"secure|verify|login|update|confirm|account|support|service|"
    r"activate|alert|billing|payment|wallet|refund|unlock"
)

# Curated set of visually-confusable Unicode characters mapped to the Latin
# letter they're commonly substituted for in real phishing/homoglyph
# attacks. Covers the Cyrillic and Greek lookalikes most frequently seen in
# the wild -- not exhaustive by design (see module docstring).
# Written as \u escapes so the file's behaviour can't depend on an editor's
# or console's text encoding.
CONFUSABLE_CHAR_MAP = {
    # Cyrillic -> Latin
    "\u0430": "a",  # а
    "\u0435": "e",  # е
    "\u043e": "o",  # о
    "\u0440": "p",  # р
    "\u0441": "c",  # с
    "\u0445": "x",  # х
    "\u0443": "y",  # у
    "\u0456": "i",  # і
    "\u0455": "s",  # ѕ
    "\u04bb": "h",  # һ
    "\u0458": "j",  # ј
    "\u0501": "d",  # ԁ
    "\u0461": "w",  # ѡ
    "\u0475": "v",  # ѵ
    # Greek -> Latin
    "\u03b1": "a",  # α
    "\u03bf": "o",  # ο
    "\u03c1": "p",  # ρ
    "\u03c5": "u",  # υ
    "\u03bd": "v",  # ν
    "\u03ba": "k",  # κ
    "\u03b9": "i",  # ι
}
CONFUSABLE_CHARS = set(CONFUSABLE_CHAR_MAP.keys())


_REMOVED_URL_CHARS = re.compile(r"[\t\r\n]")


def _split_url(url_or_domain: str):
    """Parse like a browser would, as far as urllib allows: tabs/newlines are
    dropped, backslashes count as slashes, and a missing scheme is tolerated.
    Returns a SplitResult, or None if the string cannot be parsed at all
    (e.g. an unterminated IPv6 literal)."""
    text = _REMOVED_URL_CHARS.sub("", url_or_domain.strip()).replace("\\", "/")
    if "://" not in text:
        text = "http://" + text
    try:
        return urlsplit(text)
    except ValueError:
        return None


def normalize_domain(url_or_domain: str) -> str:
    """The REAL host a browser would connect to, lowercased, without 'www.',
    port, userinfo, path, query or fragment.

    Uses urllib's URL parser instead of string splitting (fixed 2026-10):
    the old split-on-'/' then split-on-':' logic turned
    'https://cibeg.com:443@evil.example/login' into 'cibeg.com' (the real
    host is evil.example). Anything before the last '@' is userinfo, not the
    host. Never raises on malformed input: unparseable strings fall back to
    a crude best-effort host so bulk jobs (seeding, evaluation) don't crash;
    rejecting bad input is the API validation layer's job.
    """
    parsed = _split_url(url_or_domain)
    if parsed is None:
        text = url_or_domain.strip().lower()
        text = re.sub(r"^[a-z]+://", "", text)
        host = re.split(r"[/?#\\]", text)[0].rpartition("@")[2].split(":")[0]
    else:
        host = (parsed.hostname or "")
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    return host.rstrip(".")


def extract_userinfo(url_or_domain: str) -> str:
    """The 'user:password' part before the last '@' in the authority, or ''.
    Phishing URLs put a brand name here (http://cibeg.com@evil.net/) because
    browsers ignore it for routing but users read it as the destination."""
    parsed = _split_url(url_or_domain)
    if parsed is None:
        return ""
    userinfo, at, _ = parsed.netloc.rpartition("@")
    return userinfo.lower() if at else ""


def extract_backslash_authority(url_or_domain: str) -> str:
    """The text a naive reader (or old-style string split) would take as the
    host: everything after the scheme up to the first '/', '?' or '#'.
    Returned only when it contains a backslash, which browsers read as the
    START of the path -- so 'https://evil.net\\.cibeg.com/login' goes to
    evil.net while a human reads 'cibeg.com'. Otherwise returns ''."""
    text = _REMOVED_URL_CHARS.sub("", url_or_domain.strip().lower())
    text = re.sub(r"^[a-z][a-z0-9+.-]*://", "", text)
    text = re.split(r"[/?#]", text)[0]
    return text if "\\" in text else ""


_PLAIN_HOSTNAME_PATTERN = re.compile(r"[a-z0-9.-]+")


def is_plain_hostname(host: str) -> bool:
    """True only for a plain ASCII hostname (letters, digits, dots, hyphens).
    Anything with a backslash, '@', '%', spaces or non-ASCII characters is NOT
    plain. Used to gate the "genuine subdomain of a real brand" conclusions
    (here and in risk_scoring's allowlist): browsers treat a backslash like a
    slash and '@' as a userinfo separator, so a string such as
    'evil.net\\.cibeg.com' ends with '.cibeg.com' but is really a request to
    evil.net. Such inputs must never be trusted as brand subdomains."""
    return bool(_PLAIN_HOSTNAME_PATTERN.fullmatch(host))


def _contains_on_domain_boundary(candidate: str, fragment: str) -> bool:
    """True if `fragment` appears in `candidate` NOT directly preceded or
    followed by a letter or digit -- i.e. it starts/ends at the edge of the
    string or next to a dot or hyphen. 'cibeg.com' counts inside
    'cibeg.com.verify-now.net' and 'secure-cibeg.com.evil.net', but 'noon.com'
    does not count inside 'afternoon.com'."""
    pattern = r"(?<![a-z0-9])" + re.escape(fragment) + r"(?![a-z0-9])"
    return re.search(pattern, candidate) is not None


def _decode_punycode_labels(domain: str) -> str:
    """
    Decode any 'xn--' punycode-encoded labels in a domain back to their
    real Unicode form, leaving non-punycode labels unchanged. Falls back
    to the original domain unchanged if a label fails to decode (malformed
    punycode should not crash the engine -- it just won't be checked for
    homoglyphs; url_pattern_analyzer's punycode_domain flag still catches
    it as a raw structural signal either way).
    """
    labels = domain.split(".")
    decoded_labels = []
    for label in labels:
        if label.startswith("xn--"):
            try:
                decoded_labels.append(label.encode("ascii").decode("idna"))
            except (UnicodeError, UnicodeDecodeError):
                decoded_labels.append(label)
        else:
            decoded_labels.append(label)
    return ".".join(decoded_labels)


def _skeletonize(domain: str) -> str:
    """Map each confusable Unicode character to its Latin lookalike."""
    return "".join(CONFUSABLE_CHAR_MAP.get(char, char) for char in domain)


def _contains_confusable_chars(domain: str) -> bool:
    return any(char in CONFUSABLE_CHARS for char in domain)


def calculate_levenshtein_distance(a: str, b: str) -> int:
    """Standard dynamic-programming edit distance. No external dependency."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    previous_row = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current_row = [i]
        for j, char_b in enumerate(b, start=1):
            insertion = current_row[j - 1] + 1
            deletion = previous_row[j] + 1
            substitution = previous_row[j - 1] + (0 if char_a == char_b else 1)
            current_row.append(min(insertion, deletion, substitution))
        previous_row = current_row

    return previous_row[-1]


def _brand_label(domain: str) -> str:
    """The core label of a domain, e.g. 'cibeg' from 'cibeg.com'."""
    return domain.split(".")[0]


# ---------------------------------------------------------------------------
# Techniques 6-8: brand tokens, TLD swap, brand in path (added 2026-10)
# ---------------------------------------------------------------------------

MIN_BRAND_TOKEN_LENGTH = 3
DISTINCTIVE_MIN_LABEL_LENGTH = 5
MIN_BRAND_DOMAIN_LENGTH_FOR_PATH_CHECK = 7

BRAND_TOKEN_CONTEXT_SCORE = 70.0
BRAND_TOKEN_ANY_OTHER_TOKEN_SCORE = 60.0
BRAND_HYPHEN_SPLIT_SCORE = 70.0
BRAND_TLD_SWAP_SCORE = 60.0
BRAND_IN_PATH_SCORE = 60.0

# Words that, next to a brand name inside a hostname, turn "mentions a brand"
# into "pretends to be the brand".
CONTEXT_TOKENS = frozenset({
    "secure", "security", "safe", "verify", "login", "signin", "update",
    "confirm", "account", "support", "service", "services", "activate",
    "alert", "billing", "payment", "pay", "wallet", "refund", "unlock",
    "egypt", "eg", "masr", "online", "mobile", "app", "bank", "banking",
    "ebanking", "cash", "portal", "official", "care", "help", "card",
    "cards", "customer", "customers", "offer", "offers", "bonus", "gift",
    "prize", "claim", "web", "my",
})

# Brand names that are also real organisations elsewhere (parents, sister
# companies in other countries). A look-alike under an ordinary country-code
# domain is more likely the real company than an attacker.
MULTINATIONAL_LABELS = frozenset({
    "vodafone", "orange", "amazon", "etisalat", "eand", "jumia", "talabat",
    "noon", "qnb",
})

# Labels that are ordinary words; never distinctive.
GENERIC_BRAND_LABELS = frozenset({"digital"})

# Suffixes (last label) that are heavily abused and cheap; a multinational
# or short brand label under one of these IS suspicious.
ABUSE_PRONE_TLDS = frozenset({
    "top", "xyz", "click", "site", "online", "icu", "vip", "live", "shop",
    "buzz", "cfd", "sbs", "info", "work", "support", "tk", "ml", "ga", "cf",
    "gq", "cc", "ws", "pw",
})

# Hosts whose job is to show other sites' URLs inside their own paths.
PATH_CHECK_EXEMPT_HOSTS = frozenset({
    "archive.org", "web.archive.org", "archive.ph", "archive.is",
})

_SECOND_LEVEL_LABELS = frozenset({"com", "co", "gov", "org", "net", "edu", "ac", "sch", "mil"})


def _build_brand_profiles() -> list[dict]:
    brand_name_by_domain: dict[str, str] = {}
    for brand_name, brand_domain in EGYPTIAN_BRAND_DOMAINS:
        brand_name_by_domain.setdefault(brand_domain, brand_name)
    profiles = []
    for brand_domain, brand_name in brand_name_by_domain.items():
        label = _brand_label(brand_domain)
        tokens = {label, *EGYPTIAN_BRAND_ALIAS_LABELS.get(brand_domain, ())}
        tokens = {t for t in tokens
                  if len(t) >= MIN_BRAND_TOKEN_LENGTH and t not in GENERIC_BRAND_LABELS}
        profiles.append({
            "brand": brand_name,
            "domain": brand_domain,
            "label": label,
            "tokens": tokens,
            "multinational": label in MULTINATIONAL_LABELS
                             or any(t in MULTINATIONAL_LABELS for t in tokens),
            "distinctive": len(label) >= DISTINCTIVE_MIN_LABEL_LENGTH
                           and label not in GENERIC_BRAND_LABELS
                           and label not in MULTINATIONAL_LABELS,
        })
    return profiles


_BRAND_PROFILES = _build_brand_profiles()


def _split_registrable(host: str):
    """(registrable label, subdomain labels, suffix) for a plain hostname, or
    None. Handles the common two-part suffixes (com.eg, co.uk ...) without a
    full public-suffix list -- a deliberate simplification."""
    labels = host.split(".")
    if len(labels) < 2 or not all(labels):
        return None
    if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in _SECOND_LEVEL_LABELS:
        return labels[-3], labels[:-3], ".".join(labels[-2:])
    return labels[-2], labels[:-2], labels[-1]


def _tld_is_ordinary_country_code(suffix: str) -> bool:
    last = suffix.rsplit(".", 1)[-1]
    return len(last) == 2 and last not in ABUSE_PRONE_TLDS


def _is_multinational_label_on_ordinary_suffix(candidate: str, brand_domain: str) -> bool:
    label = _brand_label(brand_domain)
    if label not in MULTINATIONAL_LABELS or not is_plain_hostname(candidate):
        return False
    split = _split_registrable(candidate)
    if split is None:
        return False
    registrable, subdomains, suffix = split
    last = suffix.rsplit(".", 1)[-1]
    return registrable == label and last not in ABUSE_PRONE_TLDS


def _adjacent_joins(tokens: list[str]) -> set[str]:
    joins = set()
    for size in (2, 3):
        for i in range(len(tokens) - size + 1):
            joins.add("".join(tokens[i:i + size]))
    return joins


def _check_brand_tokens_tld_and_path(url_or_domain: str, candidate: str, best_result: dict) -> dict:
    """Techniques 6-8. Only ever raises best_result's score, never lowers it."""

    def consider(score, profile, technique):
        nonlocal best_result
        if score > best_result["score"]:
            best_result = {
                "score": score,
                "matched_brand": profile["brand"],
                "matched_brand_domain": profile["domain"],
                "technique": technique,
                "normalized_domain": candidate,
            }

    if not is_plain_hostname(candidate):
        return best_result
    split = _split_registrable(candidate)
    if split is None:
        return best_result
    registrable, subdomains, suffix = split

    registrable_tokens = [t for t in registrable.split("-") if t]
    subdomain_tokens = [t for label in subdomains for t in label.split("-") if t]
    all_tokens = registrable_tokens + subdomain_tokens

    for profile in _BRAND_PROFILES:
        country_code_only = _tld_is_ordinary_country_code(suffix)
        skip_multinational = profile["multinational"] and country_code_only

        # --- 6. brand tokens ------------------------------------------------
        if not skip_multinational and len(all_tokens) >= 2:
            for token in profile["tokens"]:
                if token not in all_tokens:
                    continue
                others = list(all_tokens)
                others.remove(token)
                has_context = any(other in CONTEXT_TOKENS for other in others)
                if has_context:
                    consider(BRAND_TOKEN_CONTEXT_SCORE, profile, "brand_token_combo")
                elif profile["distinctive"] and token == profile["label"]:
                    consider(BRAND_TOKEN_ANY_OTHER_TOKEN_SCORE, profile, "brand_token_combo")

        # --- 6b. hyphen-split brand label ("banque-misr.com") ----------------
        if not skip_multinational and len(registrable_tokens) >= 2:
            joins = _adjacent_joins(registrable_tokens)
            for token in profile["tokens"]:
                if len(token) >= DISTINCTIVE_MIN_LABEL_LENGTH and token in joins:
                    consider(BRAND_HYPHEN_SPLIT_SCORE, profile, "brand_hyphen_split")

        # --- 7. same label, different suffix ----------------------------------
        if registrable == profile["label"] and not subdomains and candidate != profile["domain"]:
            if profile["distinctive"]:
                consider(BRAND_TLD_SWAP_SCORE, profile, "brand_tld_swap")
            elif len(profile["label"]) >= MIN_BRAND_TOKEN_LENGTH and not country_code_only \
                    and suffix.rsplit(".", 1)[-1] in ABUSE_PRONE_TLDS \
                    and profile["label"] not in GENERIC_BRAND_LABELS:
                consider(BRAND_TLD_SWAP_SCORE, profile, "brand_tld_swap")

    # --- 8. brand domain named in the path / query of another host -------------
    if candidate not in PATH_CHECK_EXEMPT_HOSTS:
        parsed = _split_url(url_or_domain)
        if parsed is not None:
            path_text = unquote(f"{parsed.path}?{parsed.query}#{parsed.fragment}").lower()
            for profile in _BRAND_PROFILES:
                if len(profile["domain"]) >= MIN_BRAND_DOMAIN_LENGTH_FOR_PATH_CHECK \
                        and _contains_on_domain_boundary(path_text, profile["domain"]):
                    consider(BRAND_IN_PATH_SCORE, profile, "brand_in_path")

    return best_result


def calculate_domain_typosquatting_score(url_or_domain: str) -> dict:
    """
    Check a URL or bare domain against the Egyptian brand registry.

    Returns a dict:
        {
            "score": float,              # 0-100
            "matched_brand": str | None,
            "matched_brand_domain": str | None,
            "technique": str | None,      # "exact_match" | "brand_subdomain" |
                                           # "domain_embedding" | "levenshtein" |
                                           # "brand_label_reuse" |
                                           # "homoglyph_confusable" |
                                           # "userinfo_embedding" |
                                           # "backslash_embedding" |
                                           # "brand_token_combo" |
                                           # "brand_hyphen_split" |
                                           # "brand_tld_swap" |
                                           # "brand_in_path" | None
            "normalized_domain": str,
        }

    Raises InvalidScanInputError if input is not a non-empty string.
    """
    if not isinstance(url_or_domain, str) or not url_or_domain.strip():
        raise InvalidScanInputError("Input must be a non-empty string.")

    candidate = normalize_domain(url_or_domain)

    best_result = {
        "score": 0.0,
        "matched_brand": None,
        "matched_brand_domain": None,
        "technique": None,
        "normalized_domain": candidate,
    }

    # --- Technique 3 setup: decode punycode, then skeletonize -----------
    # Done once, outside the per-brand loop, since it doesn't depend on
    # which brand we're comparing against.
    decoded_candidate = _decode_punycode_labels(candidate)
    candidate_skeleton = _skeletonize(decoded_candidate)
    # Only a genuine homoglyph signal if something actually changed --
    # i.e. the domain really did contain confusable/punycode-decoded
    # characters, not just a coincidental plain-ASCII match (which would
    # already be caught by exact_match above, at score 0).
    is_potential_homoglyph = (
        decoded_candidate != candidate or _contains_confusable_chars(decoded_candidate)
    )

    # --- Brand named in the userinfo part (http://cibeg.com@evil.net/) ----
    # The host is evil.net, but the brand string is what the user reads.
    # Scored like domain embedding. A genuine exact brand host below still
    # returns 0 immediately, so this cannot flag the brand's own site.
    # Same idea for the backslash trick: the brand sits where a human reads
    # the host, but the browser treats the backslash as the start of the path.
    deceptive_zones = [
        (extract_userinfo(url_or_domain), "userinfo_embedding"),
        (extract_backslash_authority(url_or_domain), "backslash_embedding"),
    ]
    for zone_text, technique_name in deceptive_zones:
        if not zone_text or best_result["score"] > 0:
            continue
        for brand_name, brand_domain in EGYPTIAN_BRAND_DOMAINS:
            if _contains_on_domain_boundary(zone_text, brand_domain):
                best_result = {
                    "score": 95.0,
                    "matched_brand": brand_name,
                    "matched_brand_domain": brand_domain,
                    "technique": technique_name,
                    "normalized_domain": candidate,
                }
                break

    for brand_name, brand_domain in EGYPTIAN_BRAND_DOMAINS:
        # 1. Exact match -- this IS the real domain, not a typosquat.
        if candidate == brand_domain:
            return {
                "score": 0.0,
                "matched_brand": brand_name,
                "matched_brand_domain": brand_domain,
                "technique": "exact_match",
                "normalized_domain": candidate,
            }

        # 1b. Real brand subdomain -- e.g. "online.cibeg.com". Owned by the
        # brand, not an impersonation. matched_brand stays None on purpose
        # (see module docstring).
        if candidate.endswith("." + brand_domain) and is_plain_hostname(candidate):
            return {
                "score": 0.0,
                "matched_brand": None,
                "matched_brand_domain": brand_domain,
                "technique": "brand_subdomain",
                "normalized_domain": candidate,
            }

        # 2. Full-domain embedding, on a domain boundary.
        if _contains_on_domain_boundary(candidate, brand_domain):
            score = 95.0
            if score > best_result["score"]:
                best_result = {
                    "score": score,
                    "matched_brand": brand_name,
                    "matched_brand_domain": brand_domain,
                    "technique": "domain_embedding",
                    "normalized_domain": candidate,
                }
            continue  # a stronger technique already matched this brand; skip weaker checks for it

        # 3. Homoglyph/confusable-character detection. Checked BEFORE
        # Levenshtein (deliberately, not accidentally) -- a single
        # confusable-character substitution (e.g. Cyrillic small ie,
        # U+0435, for Latin "e") is also just "1 edit" as far as plain
        # Levenshtein distance is concerned, so if Levenshtein were
        # checked first it would match and `continue` before this more
        # specific, higher-confidence technique ever got a chance to run
        # for the same brand. The skeletonized (confusable-normalized)
        # domain matches a real brand domain exactly, but the raw
        # candidate did NOT (we'd have returned at technique 1 otherwise)
        # -- meaning it renders identically but is a different string.
        # Only fires when the domain genuinely contained punycode or
        # confusable characters (is_potential_homoglyph), so a domain that
        # happens to already equal its own skeleton (pure ASCII, no
        # substitution) can't spuriously trigger this.
        if is_potential_homoglyph and candidate_skeleton == brand_domain:
            if HOMOGLYPH_SCORE > best_result["score"]:
                best_result = {
                    "score": HOMOGLYPH_SCORE,
                    "matched_brand": brand_name,
                    "matched_brand_domain": brand_domain,
                    "technique": "homoglyph_confusable",
                    "normalized_domain": candidate,
                }
            continue  # HOMOGLYPH_SCORE (92) already beats anything
                      # Levenshtein or brand-label-reuse could produce
                      # for this same brand

        # 3b. The same multinational brand label on an ordinary country/.com
        # suffix ("vodafone.com", "orange.fr", "amazon.de", "jumia.com.ng") is
        # the parent or a sister company, not a look-alike. Before this check
        # (found 2026-10 while testing technique 6-8 on legitimate domains)
        # Levenshtein scored those at 60-80 (MEDIUM). Abuse-prone suffixes
        # ("vodafone.top") are still checked.
        if _is_multinational_label_on_ordinary_suffix(candidate, brand_domain):
            continue

        # 4. Levenshtein closeness (only for long-enough brand domains).
        if len(brand_domain) >= MIN_DOMAIN_LENGTH_FOR_LEVENSHTEIN:
            distance = calculate_levenshtein_distance(candidate, brand_domain)
            max_len = max(len(candidate), len(brand_domain))
            similarity = 1 - (distance / max_len) if max_len else 0
            if similarity >= LEVENSHTEIN_SIMILARITY_THRESHOLD:
                score = round(60 + (similarity - LEVENSHTEIN_SIMILARITY_THRESHOLD) / (1 - LEVENSHTEIN_SIMILARITY_THRESHOLD) * 30, 2)
                if score > best_result["score"]:
                    best_result = {
                        "score": score,
                        "matched_brand": brand_name,
                        "matched_brand_domain": brand_domain,
                        "technique": "levenshtein",
                        "normalized_domain": candidate,
                    }
                continue

        # 5. Brand-label reuse with a modifier (hyphen, digits, extra words).
        # Bare substring containment isn't enough -- a suspicious keyword
        # must also be present, or legitimate infrastructure/common-word
        # domains get falsely flagged (see docs/rule_based_engine_design.md).
        # Short labels must additionally sit on a domain boundary, or they
        # match inside ordinary words (e.g. "noon" in "afternoon-login.com").
        label = _brand_label(brand_domain)
        if len(label) >= MIN_LABEL_LENGTH_FOR_REUSE_CHECK:
            if len(label) <= SHORT_LABEL_MAX_LENGTH_REQUIRING_BOUNDARY:
                label_present = _contains_on_domain_boundary(candidate, label)
            else:
                label_present = label in candidate
            if label_present and SUSPICIOUS_MODIFIER_PATTERN.search(candidate):
                score = 70.0
                if score > best_result["score"]:
                    best_result = {
                        "score": score,
                        "matched_brand": brand_name,
                        "matched_brand_domain": brand_domain,
                        "technique": "brand_label_reuse",
                        "normalized_domain": candidate,
                    }

    return _check_brand_tokens_tld_and_path(url_or_domain, candidate, best_result)
