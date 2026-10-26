"""
review_tranco.py

Reviews data/raw/tranco_raw.csv and splits it into:
  - tranco_clean.csv            : domains that pass all checks, ready to use as-is
  - tranco_flagged_for_review.csv : domains flagged for a closer look (infra / adult / non_standard)

Also writes tranco_final_approved.csv -- the recommended final list after
automatically applying the agreed rules:
  - infra        -> keep (legitimate CDN/ad-tech/analytics infrastructure)
  - non_standard -> keep, EXCEPT *.in-addr.arpa entries (reverse-DNS zones, not real websites)
  - adult        -> exclude (kept out of the legitimate-brand dataset for defense-safety reasons)

Usage:
    python review_tranco.py
"""

import csv
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # scripts/ -> data/ -> root
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INPUT_PATH = RAW_DIR / "tranco_raw.csv"
CLEAN_PATH = RAW_DIR / "tranco_clean.csv"
FLAGGED_PATH = RAW_DIR / "tranco_flagged_for_review.csv"
APPROVED_PATH = RAW_DIR / "tranco_final_approved.csv"

# ---------------------------------------------------------------------------
# Flagging rules
# ---------------------------------------------------------------------------

INFRA_SUBSTRINGS = [
    "cdn", "cloudflare", "akamai", "fastly", "amazonaws", "googletagmanager",
    "doubleclick", "googlesyndication", "google-analytics", "googleusercontent",
    "gstatic", "edgekey", "edgesuite", "azureedge", "googleadservices",
    "googleapis", "sentry", "gravatar", "bugsnag", "unpkg", "jsdelivr",
]

ADULT_SUBSTRINGS = [
    "porn", "xxx", "xnxx", "xvideos", "sex", "adult", "gambl", "casino", "bet",
    "escort", "chaturbate", "onlyfans", "cam4", "hentai", "redtube", "youporn",
    "javhd", "javporn", "brazzers", "bdsm",
]

# Domains where an adult substring match is a false positive (real word/company
# that happens to contain one of the substrings above, e.g. "sex" inside "Essex").
ADULT_FALSE_POSITIVE_WHITELIST = [
    "essex", "sussex", "middlesex", "wessex", "unisex", "intersex",
    "saasexch", "dnsexit", "exchange",
]

NON_STANDARD_PATTERN = re.compile(r"^in-addr\.arpa$|\.in-addr\.arpa$|\.ip6\.arpa$")


def is_reverse_dns_zone(domain: str) -> bool:
    """True for reverse-DNS infrastructure entries like '23.in-addr.arpa' -- not real websites."""
    return bool(NON_STANDARD_PATTERN.search(domain))


def classify(domain: str) -> str | None:
    """Return a flag reason ('infra', 'adult', 'non_standard') or None if the domain looks fine."""
    lowered = domain.lower()

    if any(sub in lowered for sub in INFRA_SUBSTRINGS):
        return "infra"

    for sub in ADULT_SUBSTRINGS:
        if sub in lowered:
            if any(safe_word in lowered for safe_word in ADULT_FALSE_POSITIVE_WHITELIST):
                break  # false positive -- don't flag as adult, fall through to other checks
            return "adult"

    # Numeric-looking or otherwise junk-pattern domains (e.g. "927589452.space"),
    # plus reverse-DNS zones -- genuinely unusual, not just "short."
    label = lowered.split(".")[0]
    if label.isdigit() or is_reverse_dns_zone(lowered):
        return "non_standard"

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    with open(INPUT_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    clean_rows = []
    flagged_rows = []
    approved_rows = []
    flag_counts = {"infra": 0, "adult": 0, "non_standard": 0}

    for row in rows:
        domain = row["domain"]
        reason = classify(domain)

        if reason is None:
            clean_rows.append(row)
            approved_rows.append(row)
        else:
            flagged_rows.append({**row, "flag_reason": reason})
            flag_counts[reason] += 1

            if reason == "infra":
                approved_rows.append(row)
            elif reason == "non_standard" and not is_reverse_dns_zone(domain.lower()):
                approved_rows.append(row)
            # adult, and non_standard reverse-DNS zones, are excluded from approved_rows

    with open(CLEAN_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "domain"])
        writer.writeheader()
        writer.writerows(clean_rows)

    with open(FLAGGED_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "domain", "flag_reason"])
        writer.writeheader()
        writer.writerows(flagged_rows)

    with open(APPROVED_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "domain"])
        writer.writeheader()
        writer.writerows(approved_rows)

    excluded_count = len(rows) - len(approved_rows)

    print(f"Checked {len(rows):,} domains.")
    print(f"  -> {len(clean_rows):,} clean, saved to tranco_clean.csv")
    print(f"  -> {len(flagged_rows):,} flagged for review, saved to tranco_flagged_for_review.csv")
    print(f"Flag breakdown:")
    for reason, count in flag_counts.items():
        print(f"  {reason}: {count}")
    print()
    print(f"Final approved list: {len(approved_rows):,} domains saved to tranco_final_approved.csv")
    print(f"  ({excluded_count:,} excluded: adult content + reverse-DNS zones)")


if __name__ == "__main__":
    main()
