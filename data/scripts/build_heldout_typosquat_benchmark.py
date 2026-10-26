"""
build_heldout_typosquat_benchmark.py

Builds the HELD-OUT Egyptian-brand typosquat benchmark. The first benchmark
(data/benchmarks/egypt_brand_typosquats.csv) became a DEVELOPMENT set when the
rule-engine fixes were designed from its miss list. This set exists so the
claim behind RQ2/RQ3 can be measured on URLs the rules were NOT designed from.

DEFENSIVE USE ONLY. Output is a CSV of URL STRINGS for testing detectors.
Nothing is registered, resolved, fetched or published; no organisation is
contacted. Never use it for training, and never tune rules on its results:
if a fix follows a look at these results, this set becomes a development set
and the thesis must say so.

What is different from the development set (by design)
  * seed 20261021 (development: 42), chosen before any result was seen;
  * keyword vocabulary: none of the 12 development keywords. Two strata,
    reported separately: words that happen to be in the detector's own
    context-word list ("in_detector_vocab") and words that are not ("novel",
    incl. Franco-Arabic: khidma, hesab, tahweel, tafeel);
  * hand-written aliases and full-name spellings that are not in the
    development ALIASES nor in EGYPTIAN_BRAND_ALIAS_LABELS;
  * TLD pools: ordinary / abuse-prone-and-in-the-detector's-list / abuse-prone-
    but-NOT-in-the-detector's-list, reported as tld_group;
  * new techniques (bitsquat, random substitution/insertion, vowel swap,
    ASCII multi-character look-alikes, plural suffix, dot-to-hyphen, glued www,
    glued combos, typo+combo, full-name, query/fragment embedding, IDN
    homoglyphs in punycode form, hyphen/chain embedding, userinfo credentials);
  * random attacker hosts are half word-based, so the URL model cannot win on
    "random-looking string" alone;
  * technique renamed brand_subdomain -> brand_as_subdomain (the detector's
    genuine-subdomain technique already uses the old name);
  * every row whose host also appears in the development benchmark or in
    data/processed/dataset.csv is DROPPED (typosquat rows only).

Row kinds: typosquat | genuine (constructed, on the brands' own domains) |
control (real other organisations, see below).
  genuine  14 per brand domain: paths, genuine subdomains, URL quirks
           (upper case, :443, trailing dot, http). CONSTRUCTED, not crawled.
           They mostly test the allowlist, not the models.
  control  control_namesake_org: real foreign sister companies / namesakes
           (vodafone.de, orange.fr, cibc.com ...); control_egypt_site: real
           popular Egyptian sites. Written from memory, NOT verified live:
           eyeball the list and delete any line you doubt. They measure false
           positives on real sites and are reported separately, never pooled.

Real-company handling: no DNS/WHOIS lookup is done (that would contact the
domains). Rows that may well be a real, unrelated organisation get an
`ambiguity` value and are EXCLUDED from the headline (kept in the file,
reported separately): real_org_curated, namesake_label_ordinary_tld,
short_label_collision (registrable label of 4 or fewer characters on an
ordinary TLD, e.g. aib.com).

Freeze: heldout_manifest.json records sha256 (line-ending-normalised) of the
detector, registry and scoring files plus git HEAD at generation time.
evaluate_heldout_typosquat_benchmark.py refuses to run if any of them changed.

Usage (project root, venv active):
    python data/scripts/build_heldout_typosquat_benchmark.py
    python data/scripts/build_heldout_typosquat_benchmark.py --allow-missing-dev   (testing only)
"""

import argparse
import csv
import hashlib
import json
import random
import re
import string
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
from app.core.egyptian_brands import EGYPTIAN_BRAND_ALIAS_LABELS, EGYPTIAN_BRAND_DOMAINS  # noqa: E402

OUT_PATH = PROJECT_ROOT / "data" / "benchmarks" / "egypt_brand_typosquats_heldout.csv"
MANIFEST_PATH = PROJECT_ROOT / "data" / "benchmarks" / "heldout_manifest.json"
DEV_PATH = PROJECT_ROOT / "data" / "benchmarks" / "egypt_brand_typosquats.csv"
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"

SEED = 20261021
PER_TECHNIQUE = 4          # kept per (brand domain, technique[, vocab stratum])
MIN_LABEL_FOR_EDITS = 3    # te (2 letters) gets no character edits
SHORT_BRAND_MAX = 4        # brand label length counted as a "short brand"

# Files whose content must not change between generating and scoring.
FREEZE_FILES = [
    "backend/app/engines/rule_based/domain_typosquatting_detector.py",
    "backend/app/engines/rule_based/url_pattern_analyzer.py",
    "backend/app/engines/rule_based/arabic_keyword_detector.py",
    "backend/app/engines/rule_based/rule_based_engine.py",
    "backend/app/engines/threat_intel/blacklist_lookup.py",
    "backend/app/engines/risk_scoring.py",
    "backend/app/engines/ml/url_features.py",
    "backend/app/engines/ml/url_model_inference.py",
    "backend/app/engines/ml/text_char_inference.py",
    "backend/app/core/egyptian_brands.py",
    "backend/app/core/url_validation.py",
]
FREEZE_GLOBS = ["ml/models/*.joblib"]

# ---------------------------------------------------------------------------
# Vocabularies (all different from the development benchmark)
# ---------------------------------------------------------------------------

DEV_KEYWORDS = {"secure", "login", "verify", "online", "support", "account",
                "egypt", "eg", "app", "wallet", "service", "update"}

# Words that ARE in the detector's CONTEXT_TOKENS (as of the frozen version).
KW_IN_DETECTOR_VOCAB = [
    "security", "safe", "confirm", "activate", "alert", "billing", "payment", "refund", "unlock",
    "masr", "mobile", "bank", "cash", "portal", "official", "care", "help", "card", "customer",
    "offer", "bonus", "gift", "prize", "claim", "my", "signin",
]
# Words that are NOT in CONTEXT_TOKENS nor (as substrings) in the brand-label
# reuse modifier pattern.
KW_NOVEL = [
    "reset", "recovery", "restore", "validate", "authenticate", "suspended", "limit", "invoice",
    "receipt", "tracking", "delivery", "shipment", "loan", "transfer", "statement", "otp", "kyc",
    "identity", "compliance", "notice", "case", "resolution", "member", "direct", "center", "hub",
    "plus", "khidma", "hesab", "tahweel", "tafeel",
]
assert not ((set(KW_IN_DETECTOR_VOCAB) | set(KW_NOVEL)) & DEV_KEYWORDS), "keyword vocab overlaps development set"
VOCABS = [("in_detector_vocab", KW_IN_DETECTOR_VOCAB), ("novel", KW_NOVEL)]

ORD_TLDS = [".com", ".net", ".org", ".biz", ".co", ".io", ".me"]
ABUSE_IN_LIST = [".shop", ".live", ".icu", ".vip", ".site", ".click", ".cc"]
ABUSE_NOT_IN_LIST = [".store", ".fun", ".space", ".website", ".bond", ".lol", ".link", ".today", ".world", ".digital"]
ALL_TLDS = ORD_TLDS + ABUSE_IN_LIST + ABUSE_NOT_IN_LIST
TLD_GROUP = {
    **{t.lstrip("."): "ordinary" for t in ORD_TLDS},
    **{t.lstrip("."): "abuse_in_detector_list" for t in ABUSE_IN_LIST},
    **{t.lstrip("."): "abuse_not_in_detector_list" for t in ABUSE_NOT_IN_LIST},
}
ABUSE_ALL = {t.lstrip(".") for t in ABUSE_IN_LIST + ABUSE_NOT_IN_LIST} | {
    "top", "xyz", "info", "online", "work", "support", "tk", "ml", "ga", "cf", "gq", "ws", "pw",
    "buzz", "cfd", "sbs"}

PATHS_HO = ["/ar/signin", "/auth/reset", "/cards/unblock", "/kyc/update", "/app/download",
            "/claim/reward", "/portal/session", "/otp/confirm"]
QUERY_PARAMS = ["url", "next", "redirect", "continue", "return_to", "goto"]
WORDS = ["bluebird", "sunrise", "cedar", "harbor", "pixel", "maple", "orbit", "delta", "lotus",
         "nile", "falcon", "amber"]

QWERTY = {
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh", "u": "yij", "i": "uok", "o": "ipl", "p": "ol",
    "a": "qsz", "s": "awdx", "d": "serfc", "f": "drtgv", "g": "ftyhb", "h": "gyujn", "j": "huikm", "k": "jiol", "l": "kop",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk",
}
LEET = {"o": "0", "l": "1", "i": "1", "e": "3", "a": "4", "s": "5", "b": "8"}
ASCII_MULTI = {"m": "rn", "w": "vv", "d": "cl", "i": "l", "l": "i", "g": "q", "u": "v", "n": "m"}
# Homoglyphs that ARE in the detector's curated confusable map (Greek / Cyrillic) ...
IDN_MAPPED = {"o": "\u03bf", "a": "\u03b1", "i": "\u03b9", "v": "\u03bd", "p": "\u03c1",
              "s": "\u0455", "j": "\u0458", "h": "\u04bb"}
# ... and ones that are NOT in it (Armenian, Latin-extended). Some may be refused
# by Python's IDNA-2003 codec and are then simply skipped.
IDN_UNMAPPED = {"o": "\u0585", "g": "\u0261", "i": "\u0131", "a": "\u0251", "n": "\u0578", "u": "\u057d", "l": "\u04cf"}

# Hand-written, Egypt-specific spellings. NOT in the development ALIASES.
ALIASES_HO = {
    "cibeg.com": ["cibbank", "cib-misr", "commercialintlbank"],
    "nbe.com.eg": ["nbebank", "nbe-masr", "nationalbankegypt"],
    "banquemisr.com": ["banquemasr", "bank-masr", "bm-online"],
    "qnbalahli.com": ["qnbalahly", "qnb-ahli", "qnbbank"],
    "qnb.com.eg": ["qnbmasr", "qnb-misr", "qnbalahlieg"],
    "aaib.com": ["aaibbank", "arabafricanbank", "aaib-masr"],
    "vodafone.com.eg": ["vodacash", "vodafone-masr", "vfcash", "vodaeg"],
    "orange.eg": ["orangemoney", "orange-masr", "orangeegy"],
    "etisalat.eg": ["etisalatmisr", "etisalat-eg", "eandmisr"],
    "eand.com.eg": ["e-and-masr", "eandcash", "and-egypt"],
    "te.eg": ["telecomegypt", "te-data", "we-egypt"],
    "instapay.eg": ["instapayeg", "instapayapp", "instapay-masr"],
    "fawry.com": ["fawrypayments", "fawry-masr", "fawrycash"],
    "jumia.com.eg": ["jumiamasr", "jumia-misr", "jumiashop"],
    "noon.com": ["noonmasr", "noon-misr", "noonshop"],
    "amazon.eg": ["amazonmasr", "amazon-misr", "amazonshop-eg"],
    "talabat.com": ["talabatmasr", "talabat-misr", "talabatfood"],
    "breadfast.com": ["breadfastmasr", "breadfastapp", "bread-fast-masr"],
    "egyptair.com": ["egyptairlines", "misrairline", "masrair"],
    "digital.gov.eg": ["digitalmasr", "egyptdigital", "digital-masr-gov"],
}
FULLNAMES_HO = {
    "cibeg.com": ["commercial-international-bank", "commercialinternationalbank"],
    "nbe.com.eg": ["national-bank-of-egypt", "nationalbankofegypt"],
    "banquemisr.com": ["banquemisregypt"],
    "qnbalahli.com": ["qnb-alahli-bank", "qatar-national-bank-alahli"],
    "aaib.com": ["arab-african-international-bank", "arabafricaninternationalbank"],
    "te.eg": ["telecomegyptwe", "telecom-egypt-we"],
    "egyptair.com": ["egyptair-airlines"],
    "digital.gov.eg": ["digital-egypt-gov", "egypt-digital-portal"],
}
_all_alias_labels = {a for aliases in EGYPTIAN_BRAND_ALIAS_LABELS.values() for a in aliases}
assert not (_all_alias_labels & {a for v in ALIASES_HO.values() for a in v}), "alias overlaps registry aliases"

# Real other organisations (written from memory; verify before trusting).
CONTROL_NAMESAKE = [
    "vodafone.com", "vodafone.de", "vodafone.co.uk", "vodafone.in", "vodafone.it", "vodafone.es",
    "orange.fr", "orange.com", "orange.pl", "orange.es", "orange.ro",
    "amazon.com", "amazon.de", "amazon.co.uk", "amazon.in", "amazon.fr", "amazon.es",
    "etisalat.ae", "eand.com", "jumia.com.ng", "jumia.co.ke", "jumia.ma", "cibc.com", "qnb.com",
]
CONTROL_EGYPT_SITES = [
    "ahram.org.eg", "youm7.com", "masrawy.com", "shorouknews.com", "almasryalyoum.com",
    "elwatannews.com", "olx.com.eg", "elmenus.com", "vezeeta.com", "swvl.com", "hsbc.com.eg",
    "alexbank.com", "bdc.com.eg", "adib.eg", "cbe.org.eg", "mcit.gov.eg", "sis.gov.eg", "tax.gov.eg",
]
CONTROL_PATHS = ["/news", "/ar/home", "/about", "/contact"]
REAL_ORG_REGISTRABLES = set(CONTROL_NAMESAKE) | {"qnb.com", "aib.com", "orange.com", "vodafone.com"}
NAMESAKE_LABELS = {"vodafone", "orange", "amazon", "etisalat", "eand", "jumia", "talabat", "noon", "qnb"}
SECOND_LEVEL = {"com", "co", "gov", "org", "net", "edu", "ac"}

DEV_TECHNIQUE_NAMES = {
    "omission", "repetition", "transposition", "keyboard_adjacent", "digit_lookalike", "hyphen_insertion",
    "tld_swap", "combo_suffix", "combo_prefix", "alias_variant", "embedding", "brand_as_subdomain",
    "homoglyph_unicode", "path_embedding", "userinfo_trick",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def host_of(text: str) -> str:
    t = text.strip().replace("\\", "/")
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", t):   # scheme only counts at the START
        t = "http://" + t
    try:
        host = urlsplit(t).hostname or ""
    except ValueError:
        return ""
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    return host.rstrip(".")


def split_domain(domain: str) -> tuple[str, str]:
    label = domain.split(".")[0]
    return label, domain[len(label):]


def registrable_parts(host: str) -> tuple[str, str, str]:
    """(registrable label, registrable domain, last label) -- simplified PSL."""
    labels = host.split(".")
    if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in SECOND_LEVEL:
        return labels[-3], ".".join(labels[-3:]), labels[-1]
    if len(labels) >= 2:
        return labels[-2], ".".join(labels[-2:]), labels[-1]
    return host, host, host


def tld_group_of(host: str) -> str:
    return TLD_GROUP.get(host.rsplit(".", 1)[-1], "brand_suffix_or_other")


# Techniques where the brand sits INSIDE a host/URL the attacker controls: the registrable
# domain is a wrapper, not the look-alike, so "might be a real company" does not apply.
NO_AMBIGUITY_TECHNIQUES = {
    "embedding_hyphen_suffix", "embedding_subdomain_chain", "brand_as_subdomain", "userinfo_credentials",
    "query_embedding", "fragment_embedding", "path_embedding",
}


def ambiguity_of(host: str, technique: str) -> str:
    if technique in NO_AMBIGUITY_TECHNIQUES:
        return ""
    reg_label, reg_domain, last = registrable_parts(host)
    if reg_domain in REAL_ORG_REGISTRABLES:
        return "real_org_curated"
    if reg_label in NAMESAKE_LABELS and last not in ABUSE_ALL:
        return "namesake_label_ordinary_tld"
    if len(reg_label) <= 4 and last in {"com", "net", "org", "biz", "co", "io", "me", "edu"}:
        return "short_label_collision"
    return ""


def rand_host(rng: random.Random) -> str:
    if rng.random() < 0.5:
        body = "".join(rng.choices(string.ascii_lowercase + string.digits, k=rng.randint(7, 12)))
    else:
        body = rng.choice(WORDS) + rng.choice(["", "-"]) + rng.choice(WORDS) + rng.choice(["", "", "2", "24", "77"])
    return body + rng.choice(ALL_TLDS)


def normalised_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def fingerprint() -> dict:
    files = {}
    for rel in FREEZE_FILES:
        p = PROJECT_ROOT / rel
        files[rel] = normalised_sha256(p) if p.exists() else None
    for pattern in FREEZE_GLOBS:
        for p in sorted(PROJECT_ROOT.glob(pattern)):
            files[str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")] = normalised_sha256(p)
    return files


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return None
    return {"head": run("rev-parse", "HEAD"), "dirty_backend_or_ml": run("status", "--porcelain", "--", "backend", "ml")}


# ---------------------------------------------------------------------------
# Candidate generators: (L = label, S = suffix, D = full domain) -> [(remainder, vocab_tag)]
# "remainder" is everything after "scheme://".
# ---------------------------------------------------------------------------

NA = "n/a"


def g_omission(L, S, D, rng):
    return [(L[:i] + L[i + 1:] + S, NA) for i in range(len(L))]


def g_repetition(L, S, D, rng):
    return [(L[:i + 1] + L[i] + L[i + 1:] + S, NA) for i in range(len(L))]


def g_transposition(L, S, D, rng):
    return [(L[:i] + L[i + 1] + L[i] + L[i + 2:] + S, NA) for i in range(len(L) - 1) if L[i] != L[i + 1]]


def g_keyboard(L, S, D, rng):
    return [(L[:i] + n + L[i + 1:] + S, NA) for i, ch in enumerate(L) for n in QWERTY.get(ch, "")]


def g_digit(L, S, D, rng):
    return [(L[:i] + LEET[ch] + L[i + 1:] + S, NA) for i, ch in enumerate(L) if ch in LEET]


def g_hyphen_insertion(L, S, D, rng):
    return [(L[:i] + "-" + L[i:] + S, NA) for i in range(2, len(L) - 1)]


def g_substitution(L, S, D, rng):
    return [(L[:i] + c + L[i + 1:] + S, NA) for i in range(len(L)) for c in string.ascii_lowercase if c != L[i]]


def g_insertion(L, S, D, rng):
    return [(L[:i] + c + L[i:] + S, NA) for i in range(len(L) + 1) for c in string.ascii_lowercase]


def g_vowel_swap(L, S, D, rng):
    return [(L[:i] + v + L[i + 1:] + S, NA) for i, ch in enumerate(L) if ch in "aeiou" for v in "aeiou" if v != ch]


def g_bitsquat(L, S, D, rng):
    ok = set(string.ascii_lowercase + string.digits)
    out = []
    for i, ch in enumerate(L):
        for bit in range(8):
            flipped = chr(ord(ch) ^ (1 << bit))
            if flipped in ok and flipped != ch:
                out.append((L[:i] + flipped + L[i + 1:] + S, NA))
    return out


def g_ascii_multichar(L, S, D, rng):
    out = []
    for i, ch in enumerate(L):
        if ch in ASCII_MULTI:
            out.append((L[:i] + ASCII_MULTI[ch] + L[i + 1:] + S, NA))
    return out


def g_plural(L, S, D, rng):
    return [(L + "s" + S, NA), (L + "es" + S, NA), (L + "x" + S, NA)]


def _idn(mapping):
    def gen(L, S, D, rng):
        out = []
        for i, ch in enumerate(L):
            if ch in mapping:
                try:
                    out.append(((L[:i] + mapping[ch] + L[i + 1:] + S).encode("idna").decode("ascii"), NA))
                except UnicodeError:
                    continue
        return out
    return gen


def g_tld_swap(L, S, D, rng):
    return [(L + t, NA) for t in ALL_TLDS if L + t != D]


def g_dot_to_hyphen(L, S, D, rng):
    return [(D.replace(".", "-") + t, NA) for t in ALL_TLDS]


def g_glued_www(L, S, D, rng):
    return [(p + L + t, NA) for p in ("www", "www-", "www1-") for t in [S] + ALL_TLDS]


def _kw_samples(rng, n):
    """Yield (vocab tag, word, tld) triples, n per word-vocab, deterministic."""
    out = []
    for tag, words in VOCABS:
        for k in words:
            for t in rng.sample(ALL_TLDS, n):
                out.append((tag, k, t))
    return out


def g_combo_suffix(L, S, D, rng):
    return [(f"{L}-{k}{t}", tag) for tag, k, t in _kw_samples(rng, 3)]


def g_combo_prefix(L, S, D, rng):
    return [(f"{k}-{L}{t}", tag) for tag, k, t in _kw_samples(rng, 3)]


def g_combo_concat(L, S, D, rng):
    out = []
    for tag, k, t in _kw_samples(rng, 2):
        out.append((f"{L}{k}{t}", tag))
        out.append((f"{k}{L}{t}", tag))
    return out


def g_combo_two_words(L, S, D, rng):
    out = []
    for tag, words in VOCABS:
        for _ in range(50):
            k1, k2 = rng.sample(words, 2)
            out.append((f"{L}-{k1}-{k2}{rng.choice(ALL_TLDS)}", tag))
    return out


def g_brand_as_subdomain(L, S, D, rng):
    out = []
    for tag, words in VOCABS:
        for _ in range(50):
            k1, k2 = rng.sample(words, 2)
            sub = f"{k1}-{k2}" if rng.random() < 0.5 else k1
            out.append((f"{L}.{sub}{rng.choice(ALL_TLDS)}", tag))
    return out


def g_typo_plus_combo(L, S, D, rng):
    out = []
    for tag, words in VOCABS:
        for _ in range(50):
            kind, i = rng.choice(["omit", "dup", "swap"]), rng.randrange(len(L))
            if kind == "omit":
                typo = L[:i] + L[i + 1:]
            elif kind == "dup":
                typo = L[:i + 1] + L[i] + L[i + 1:]
            else:
                typo = L[:i] + L[i + 1] + L[i] + L[i + 2:] if i < len(L) - 1 else L
            if typo != L:
                out.append((f"{typo}-{rng.choice(words)}{rng.choice(ALL_TLDS)}", tag))
    return out


def g_alias(L, S, D, rng):
    return [(a + t, NA) for a in ALIASES_HO.get(D, []) for t in ALL_TLDS]


def g_fullname(L, S, D, rng):
    return [(a + t, NA) for a in FULLNAMES_HO.get(D, []) for t in ALL_TLDS]


def g_embedding_hyphen_suffix(L, S, D, rng):
    return [(f"{D}-{k}{rng.choice(ALL_TLDS)}", tag) for tag, words in VOCABS for k in words]


def g_embedding_chain(L, S, D, rng):
    out = []
    for tag, words in VOCABS:
        for _ in range(40):
            k1, k2 = rng.sample(words, 2)
            out.append((f"{k1}.{D}.{k2}{rng.choice(ALL_TLDS)}", tag))
    return out


def g_userinfo_credentials(L, S, D, rng):
    out = []
    for _ in range(40):
        kw = rng.choice(KW_IN_DETECTOR_VOCAB + KW_NOVEL)
        cred = rng.choice([f"www.{D}", f"{D}:443", f"secure.{D}", f"{D}:{kw}"])
        out.append((f"{cred}@{rand_host(rng)}{rng.choice(PATHS_HO + ['/'])}", NA))
    return out


def g_query_embedding(L, S, D, rng):
    vals = [f"https://{D}/", D, quote(f"https://{D}/ar", safe=""), f"//{D}"]
    return [(f"{rand_host(rng)}{rng.choice(PATHS_HO)}?{rng.choice(QUERY_PARAMS)}={rng.choice(vals)}", NA) for _ in range(40)]


def g_fragment_embedding(L, S, D, rng):
    return [(f"{rand_host(rng)}{rng.choice(PATHS_HO)}#{rng.choice([D, 'https://' + D + '/'])}", NA) for _ in range(40)]


def g_path_embedding(L, S, D, rng):
    return [(f"{rand_host(rng)}/{rng.choice(['cdn', 'files', 'u', 'go', 'sites'])}/{D}{rng.choice(['', '/ar/signin', '/auth/reset'])}", NA)
            for _ in range(40)]


# (technique name, generator, minimum brand-label length, host-only => random path may be appended)
TECHNIQUES = [
    ("omission", g_omission, MIN_LABEL_FOR_EDITS, True),
    ("repetition", g_repetition, MIN_LABEL_FOR_EDITS, True),
    ("transposition", g_transposition, MIN_LABEL_FOR_EDITS, True),
    ("keyboard_adjacent", g_keyboard, MIN_LABEL_FOR_EDITS, True),
    ("digit_lookalike", g_digit, MIN_LABEL_FOR_EDITS, True),
    ("hyphen_insertion", g_hyphen_insertion, 4, True),
    ("substitution_random", g_substitution, MIN_LABEL_FOR_EDITS, True),
    ("insertion_random", g_insertion, MIN_LABEL_FOR_EDITS, True),
    ("vowel_swap", g_vowel_swap, MIN_LABEL_FOR_EDITS, True),
    ("bitsquat", g_bitsquat, MIN_LABEL_FOR_EDITS, True),
    ("ascii_multichar", g_ascii_multichar, MIN_LABEL_FOR_EDITS, True),
    ("plural_suffix", g_plural, 2, True),
    ("idn_homoglyph_mapped", _idn(IDN_MAPPED), MIN_LABEL_FOR_EDITS, True),
    ("idn_homoglyph_unmapped", _idn(IDN_UNMAPPED), MIN_LABEL_FOR_EDITS, True),
    ("tld_swap", g_tld_swap, 2, True),
    ("dot_to_hyphen", g_dot_to_hyphen, 2, True),
    ("glued_www", g_glued_www, 2, True),
    ("combo_suffix", g_combo_suffix, 2, True),
    ("combo_prefix", g_combo_prefix, 2, True),
    ("combo_concat", g_combo_concat, 2, True),
    ("combo_two_words", g_combo_two_words, 2, True),
    ("brand_as_subdomain", g_brand_as_subdomain, 2, True),
    ("typo_plus_combo", g_typo_plus_combo, 4, True),
    ("alias_variant", g_alias, 2, True),
    ("fullname_variant", g_fullname, 2, True),
    ("embedding_hyphen_suffix", g_embedding_hyphen_suffix, 2, True),
    ("embedding_subdomain_chain", g_embedding_chain, 2, True),
    ("userinfo_credentials", g_userinfo_credentials, 2, False),
    ("query_embedding", g_query_embedding, 2, False),
    ("fragment_embedding", g_fragment_embedding, 2, False),
    ("path_embedding", g_path_embedding, 2, False),
]
# brand domains shorter than 7 characters ("te.eg") are not meaningful for the three
# path-like techniques (the rule and any human would not read them as a domain):
SKIP_SHORT_DOMAIN = {"query_embedding", "fragment_embedding", "path_embedding"}
MIN_DOMAIN_FOR_PATHLIKE = 7


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def load_hosts_from_csv(path: Path, content_filter=None) -> set[str]:
    hosts: set[str] = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if content_filter and not content_filter(row):
                continue
            text = row.get("url") or row.get("text") or ""
            h = host_of(text)
            if h:
                hosts.add(h)
    return hosts


def build(dev_hosts: set[str], dataset_hosts: set[str] | None) -> tuple[list[dict], Counter]:
    real_domains = {d for _, d in EGYPTIAN_BRAND_DOMAINS}
    rows: list[dict] = []
    seen_urls: set[str] = set()
    stats: Counter = Counter()
    n_ts = 0

    for brand, domain in EGYPTIAN_BRAND_DOMAINS:
        L, S = split_domain(domain)
        brand_size = "short" if len(L) <= SHORT_BRAND_MAX else "normal"
        for technique, gen, min_label, host_only in TECHNIQUES:
            if len(L) < min_label:
                continue
            if technique in SKIP_SHORT_DOMAIN and len(domain) < MIN_DOMAIN_FOR_PATHLIKE:
                continue
            rng = random.Random(f"{SEED}|{domain}|{technique}")
            candidates = list(dict.fromkeys(gen(L, S, domain, rng)))
            by_tag: dict[str, list[str]] = defaultdict(list)
            for remainder, tag in candidates:
                host = host_of(remainder)
                if not host or host in real_domains:
                    stats["dropped_is_real_brand_domain"] += 1
                    continue
                if host in dev_hosts:
                    stats["dropped_host_in_dev_benchmark"] += 1
                    continue
                if dataset_hosts is not None and host in dataset_hosts:
                    stats["dropped_host_in_dataset"] += 1
                    continue
                by_tag[tag].append(remainder)
            per = max(1, PER_TECHNIQUE // max(1, len(by_tag)))
            for tag in sorted(by_tag):
                pool = by_tag[tag]
                rng.shuffle(pool)
                for remainder in pool[:per]:
                    scheme = "https" if rng.random() < 0.6 else "http"
                    path = rng.choice(PATHS_HO) if (host_only and rng.random() < 0.5) else ""
                    url = f"{scheme}://{remainder}{path}"
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)
                    host = host_of(remainder)
                    n_ts += 1
                    rows.append({
                        "id": f"HT-{n_ts:06d}", "url": url, "brand": brand, "brand_domain": domain,
                        "technique": technique, "kind": "typosquat",
                        "family": "repeat" if technique in DEV_TECHNIQUE_NAMES else "new",
                        "keyword_vocab": tag, "tld_group": tld_group_of(host), "brand_size": brand_size,
                        "ambiguity": ambiguity_of(host, technique),
                    })

    # genuine own-domain controls (constructed): 14 per brand domain
    n_gn = 0
    for brand, domain in EGYPTIAN_BRAND_DOMAINS:
        L, _ = split_domain(domain)
        brand_size = "short" if len(L) <= SHORT_BRAND_MAX else "normal"
        urls = [("genuine_path", f"https://{domain}{p}") for p in
                ["/ar/home", "/en/home", "/products/cards?lang=ar", "/branches?city=cairo", "/fraud-awareness#tips"]]
        urls += [("genuine_subdomain", f"https://{s}.{domain}/{'login' if s in ('secure', 'my') else ''}")
                 for s in ["secure", "my", "portal", "careers", "help"]]
        urls += [("genuine_quirk", u) for u in
                 [f"https://WWW.{domain.upper()}/", f"https://{domain}:443/ar/home", f"https://{domain}./", f"http://{domain}/ar/home"]]
        for technique, url in urls:
            n_gn += 1
            rows.append({"id": f"HG-{n_gn:06d}", "url": url, "brand": brand, "brand_domain": domain,
                         "technique": technique, "kind": "genuine", "family": "genuine", "keyword_vocab": NA,
                         "tld_group": NA, "brand_size": brand_size, "ambiguity": ""})

    # real other organisations (false-positive controls, not part of any headline)
    n_c = 0
    crng = random.Random(f"{SEED}|controls")
    for group, domains in (("control_namesake_org", CONTROL_NAMESAKE), ("control_egypt_site", CONTROL_EGYPT_SITES)):
        for d in domains:
            for url in (f"https://www.{d}/", f"https://{d}{crng.choice(CONTROL_PATHS)}"):
                n_c += 1
                rows.append({"id": f"HC-{n_c:06d}", "url": url, "brand": "", "brand_domain": "", "technique": group,
                             "kind": "control", "family": "control", "keyword_vocab": NA, "tld_group": NA,
                             "brand_size": NA, "ambiguity": ""})
    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing-dev", action="store_true", help="testing only: proceed without the development CSV")
    args = parser.parse_args()

    if DEV_PATH.exists():
        dev_hosts = load_hosts_from_csv(DEV_PATH)
    elif args.allow_missing_dev:
        print("WARNING: development benchmark missing; overlap with it NOT excluded (testing only).")
        dev_hosts = set()
    else:
        sys.exit(f"Development benchmark not found at {DEV_PATH}. Without it overlap cannot be excluded.")

    dataset_hosts = None
    if DATASET_PATH.exists():
        dataset_hosts = load_hosts_from_csv(DATASET_PATH, lambda r: r.get("content_type") == "url")
    else:
        print(f"WARNING: {DATASET_PATH} not found; overlap with the dataset NOT excluded.")

    rows, stats = build(dev_hosts, dataset_hosts)

    # tag which genuine/control hosts also occur in the dataset (information only)
    for r in rows:
        r["host_in_dataset"] = ("unknown" if dataset_hosts is None else str(int(host_of(r["url"]) in dataset_hosts)))

    fields = ["id", "url", "brand", "brand_domain", "technique", "kind", "family", "keyword_vocab",
              "tld_group", "brand_size", "ambiguity", "host_in_dataset"]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    typo = [r for r in rows if r["kind"] == "typosquat"]
    kinds = Counter(r["kind"] for r in rows)
    manifest = {
        "seed": SEED, "per_technique": PER_TECHNIQUE,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "csv_sha256": normalised_sha256(OUT_PATH),
        "rows": dict(kinds), "dropped": dict(stats),
        "ambiguity_counts": dict(Counter(r["ambiguity"] or "none" for r in typo)),
        "git": git_state(), "freeze_files": fingerprint(),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {len(rows):,} rows to {OUT_PATH}")
    print(f"  typosquat={kinds['typosquat']:,}  genuine={kinds['genuine']:,}  control={kinds['control']:,}")
    print(f"  typosquats marked ambiguous (excluded from headline): {sum(1 for r in typo if r['ambiguity']):,}")
    print(f"  dropped candidates: {dict(stats)}")
    print("typosquats per technique (family):")
    per_tech = Counter(r["technique"] for r in typo)
    for t in sorted(per_tech):
        fam = "repeat" if t in DEV_TECHNIQUE_NAMES else "new"
        print(f"  {t:<28}{per_tech[t]:>5}  ({fam})")
    print("\nExamples:")
    for t in sorted(per_tech):
        print(f"  {t:<28}{next(r['url'] for r in typo if r['technique'] == t)}")
    print(f"\nManifest: {MANIFEST_PATH}\nFreeze fingerprint recorded for {sum(v is not None for v in manifest['freeze_files'].values())} files.")
    print("Do not change the detector, registry or scoring files before running the evaluation.")


if __name__ == "__main__":
    main()
