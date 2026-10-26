"""
build_typosquat_benchmark.py

Builds a DEFENSIVE test set for the Egyptian-brand impersonation claim of the
thesis (RQ2/RQ3), which the main dataset cannot evidence: it contains no
Egyptian-brand typosquat URLs at all.

The output is a CSV of URL STRINGS used only to test detectors. Nothing here
is registered, resolved, fetched or published; no real organisation is
contacted. (Generating look-alike candidates is standard defensive practice,
cf. tools such as dnstwist.) The set must never be used for training --
it exists to measure the finished system.

Two kinds of rows:
  typosquat  look-alikes of the 20 verified brand domains in
             app/core/egyptian_brands.py, by technique:
       omission, repetition, transposition, keyboard_adjacent, digit_lookalike,
       hyphen_insertion, tld_swap, combo_suffix, combo_prefix, alias_variant,
       embedding, brand_as_subdomain, homoglyph_unicode, path_embedding, userinfo_trick
  genuine    CONSTRUCTED URLs on the brands' own domains (homepage, www, common
             paths, genuine subdomains such as online.<brand>). They are
             plausible, not crawled: they test that the allowlist and the models
             do not flag a brand's own site. Report them as "constructed".

Honesty note for the thesis: some techniques (embedding, homoglyph_unicode,
userinfo_trick, and the Levenshtein-like edits) are exactly what the rule
engine was built to detect, so detecting them is partly circular. The other
techniques (tld_swap, combo_prefix/suffix, alias_variant, path_embedding,
brand_as_subdomain, keyboard_adjacent) are NOT implemented rules and show where
the system really stands. Always report per technique, never one pooled number.

Output: data/benchmarks/egypt_brand_typosquats.csv
Columns: id, url, brand, brand_domain, technique, kind
Deterministic (seed 42). Usage (project root):
    python data/scripts/build_typosquat_benchmark.py
"""

import csv
import random
import string
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
from app.core.egyptian_brands import EGYPTIAN_BRAND_DOMAINS  # noqa: E402

OUT_PATH = PROJECT_ROOT / "data" / "benchmarks" / "egypt_brand_typosquats.csv"
SEED = 42
PER_TECHNIQUE = 4          # candidates kept per (brand domain, technique)
MIN_LABEL_FOR_EDITS = 4    # skip character edits on very short labels (te, nbe)

TLDS = [".com", ".net", ".org", ".co", ".info", ".xyz", ".top", ".online"]
KEYWORDS = ["secure", "login", "verify", "online", "support", "account", "egypt", "eg", "app", "wallet", "service", "update"]
PATHS = ["/login", "/verify", "/account/update", "/secure/signin", "/support/refund", "/wallet/activate"]
GENUINE_PATHS = ["/login", "/en/personal", "/ar/support", "/contact-us", "/careers", "/offers/summer-2026", "/help/faq"]
GENUINE_SUBDOMAINS = ["online", "mobile", "ebanking", "m"]

QWERTY = {
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh", "u": "yij", "i": "uok", "o": "ipl", "p": "ol",
    "a": "qsz", "s": "awdx", "d": "serfc", "f": "drtgv", "g": "ftyhb", "h": "gyujn", "j": "huikm", "k": "jiol", "l": "kop",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk",
}
LEET = {"o": "0", "l": "1", "i": "1", "e": "3", "a": "4", "s": "5", "b": "8"}
CYRILLIC = {"a": "\u0430", "e": "\u0435", "o": "\u043e", "c": "\u0441", "p": "\u0440", "i": "\u0456", "x": "\u0445", "y": "\u0443"}

# Hand-written, Egypt-specific spellings people actually use for these brands.
ALIASES = {
    "cibeg.com": ["cib-egypt", "cibegypt", "cib-eg", "cibonline"],
    "nbe.com.eg": ["nbe-egypt", "nbeonline", "nationalbank-egypt"],
    "banquemisr.com": ["bankmisr", "banque-misr", "misrbank", "banquemisr-online"],
    "qnbalahli.com": ["qnb-alahli", "qnbalahli-online"],
    "qnb.com.eg": ["qnb-egypt", "qnbeg"],
    "aaib.com": ["aaib-egypt", "aaibonline"],
    "vodafone.com.eg": ["vodafone-cash", "vodafonecash", "vodafone-egypt", "vodafoneeg"],
    "orange.eg": ["orange-egypt", "orangecash", "orange-eg"],
    "etisalat.eg": ["etisalat-misr", "etisalatcash", "eand-egypt"],
    "eand.com.eg": ["eand-egypt", "e-and-egypt"],
    "te.eg": ["telecom-egypt", "tegypt", "we-telecom-egypt"],
    "instapay.eg": ["insta-pay", "instapay-eg", "instapayegypt"],
    "fawry.com": ["fawrypay", "fawry-pay", "fawry-egypt"],
    "jumia.com.eg": ["jumia-egypt", "jumiaeg", "jumia-eg"],
    "noon.com": ["noon-egypt", "nooneg", "noon-eg"],
    "amazon.eg": ["amazon-egypt", "amazoneg"],
    "talabat.com": ["talabat-egypt", "talabateg"],
    "breadfast.com": ["bread-fast", "breadfast-eg"],
    "egyptair.com": ["egypt-air", "egyptair-eg", "egyptair-online"],
    "digital.gov.eg": ["digital-egypt", "egypt-digital-gov", "gov-eg-digital"],
}


def random_host(rng: random.Random) -> str:
    body = "".join(rng.choices(string.ascii_lowercase + string.digits, k=rng.randint(6, 10)))
    return body + rng.choice([".xyz", ".top", ".online", ".site", ".click", ".net"])


def split_domain(domain: str) -> tuple[str, str]:
    label = domain.split(".")[0]
    return label, domain[len(label):]  # e.g. ("cibeg", ".com"), ("nbe", ".com.eg")


def candidates(domain: str, rng: random.Random) -> dict[str, list[str]]:
    """technique -> list of candidate HOSTS (no scheme/path)."""
    label, suffix = split_domain(domain)
    out: dict[str, list[str]] = {}

    if len(label) >= MIN_LABEL_FOR_EDITS:
        out["omission"] = [label[:i] + label[i + 1:] + suffix for i in range(len(label))]
        out["repetition"] = [label[:i + 1] + label[i] + label[i + 1:] + suffix for i in range(len(label))]
        out["transposition"] = [label[:i] + label[i + 1] + label[i] + label[i + 2:] + suffix
                                for i in range(len(label) - 1) if label[i] != label[i + 1]]
        out["keyboard_adjacent"] = [label[:i] + n + label[i + 1:] + suffix
                                    for i, ch in enumerate(label) for n in QWERTY.get(ch, "")]
        out["digit_lookalike"] = [label[:i] + LEET[ch] + label[i + 1:] + suffix
                                  for i, ch in enumerate(label) if ch in LEET]
        out["hyphen_insertion"] = [label[:i] + "-" + label[i:] + suffix for i in range(2, len(label) - 1)]
        out["homoglyph_unicode"] = [label[:i] + CYRILLIC[ch] + label[i + 1:] + suffix
                                    for i, ch in enumerate(label) if ch in CYRILLIC]

    out["tld_swap"] = [label + t for t in TLDS if label + t != domain] + ([label + ".com"] if suffix != ".com" else [])
    out["combo_suffix"] = [f"{label}-{k}{t}" for k in KEYWORDS for t in (".com", ".net", ".org", ".top")]
    out["combo_prefix"] = [f"{k}-{label}{t}" for k in KEYWORDS for t in (".com", ".net", ".org", ".top")]
    out["alias_variant"] = [f"{a}{t}" for a in ALIASES.get(domain, []) for t in (".com", ".net", ".org", ".co")]
    out["embedding"] = [f"{domain}.{k1}-{k2}{t}" for k1 in KEYWORDS[:6] for k2 in KEYWORDS[6:9] for t in (".net", ".top", ".com")]
    out["brand_as_subdomain"] = [f"{label}.{k1}-{k2}{t}" for k1 in KEYWORDS[:6] for k2 in KEYWORDS[6:9] for t in (".net", ".top", ".xyz")]
    out["path_embedding"] = [random_host(rng) for _ in range(PER_TECHNIQUE * 2)]   # host unrelated; brand goes in the path
    out["userinfo_trick"] = [random_host(rng) for _ in range(PER_TECHNIQUE * 2)]   # brand goes before the '@'
    return out


def build() -> list[dict]:
    rng = random.Random(SEED)
    real_domains = {d for _, d in EGYPTIAN_BRAND_DOMAINS}
    rows: list[dict] = []
    seen: set[str] = set()
    n_ts = n_gn = 0

    for brand, domain in EGYPTIAN_BRAND_DOMAINS:
        per_domain = random.Random(f"{SEED}-{domain}")
        for technique, hosts in candidates(domain, per_domain).items():
            hosts = [h for h in dict.fromkeys(hosts) if h not in real_domains]
            per_domain.shuffle(hosts)
            for host in hosts[:PER_TECHNIQUE]:
                scheme = "https" if rng.random() < 0.6 else "http"
                path = "" if rng.random() < 0.5 else rng.choice(PATHS)
                if technique == "path_embedding":
                    url = f"{scheme}://{host}/{domain}{rng.choice(PATHS)}"
                elif technique == "userinfo_trick":
                    url = f"{scheme}://{domain}@{host}{path or '/login'}"
                else:
                    url = f"{scheme}://{host}{path}"
                if url in seen:
                    continue
                seen.add(url)
                n_ts += 1
                rows.append({"id": f"TS-{n_ts:06d}", "url": url, "brand": brand, "brand_domain": domain,
                             "technique": technique, "kind": "typosquat"})

        genuine = [f"https://{domain}/", *(f"https://{domain}{p}" for p in GENUINE_PATHS),
                   *(f"https://{s}.{domain}/login" for s in GENUINE_SUBDOMAINS)]
        if not domain.startswith("www."):
            genuine.append(f"https://www.{domain}/")
        for url in genuine:
            n_gn += 1
            rows.append({"id": f"GN-{n_gn:06d}", "url": url, "brand": brand, "brand_domain": domain,
                         "technique": "genuine_own_domain", "kind": "genuine"})
    return rows


def main() -> None:
    rows = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "url", "brand", "brand_domain", "technique", "kind"])
        writer.writeheader()
        writer.writerows(rows)

    by_kind = Counter(r["kind"] for r in rows)
    print(f"Wrote {len(rows):,} rows to {OUT_PATH}  (typosquat={by_kind['typosquat']:,}, genuine={by_kind['genuine']:,})")
    print("typosquats per technique:")
    for technique, n in sorted(Counter(r["technique"] for r in rows if r["kind"] == "typosquat").items()):
        print(f"  {technique:<20}{n:>5}")
    print("\nExamples (typosquat):")
    for technique in sorted({r["technique"] for r in rows if r["kind"] == "typosquat"}):
        example = next(r for r in rows if r["technique"] == technique)
        print(f"  {technique:<20}{example['url']}")


if __name__ == "__main__":
    main()
