"""
add_egyptian_domains.py

Appends the curated list of real Egyptian brand domains (banks, telecom,
payments, e-commerce, delivery, government) to data/processed/dataset.csv
as legitimate rows. This list was manually researched and verified rather
than scraped -- see docs/dataset_schema.md for the MAN- prefix convention.

This list doubles as the seed for the Egyptian brand registry needed later
for the threat-intel database (Phase 3, CLAUDE.md architecture step 2).

Usage:
    python add_egyptian_domains.py
"""

import csv
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # scripts/ -> data/ -> root
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATASET_PATH = PROCESSED_DIR / "dataset.csv"

DATASET_COLUMNS = [
    "id", "text", "content_type", "label", "language",
    "source", "brand_impersonated", "date_collected", "split",
]

# (brand name, domain) -- domain only, https:// added when building rows.
# Verified via web search on 2026-07-15 unless noted otherwise.
EGYPTIAN_BRAND_DOMAINS = [
    ("Commercial International Bank (CIB)", "cibeg.com"),
    ("National Bank of Egypt (NBE)", "nbe.com.eg"),
    ("Banque Misr", "banquemisr.com"),
    ("QNB Al Ahli", "qnbalahli.com"),
    ("QNB Al Ahli", "qnb.com.eg"),
    ("Arab African International Bank", "aaib.com"),
    ("Vodafone Egypt", "vodafone.com.eg"),
    ("Orange Egypt", "orange.eg"),
    ("Etisalat Misr (e&)", "etisalat.eg"),
    ("Etisalat Misr (e&)", "eand.com.eg"),
    ("Telecom Egypt (WE)", "te.eg"),
    ("Instapay", "instapay.eg"),
    ("Fawry", "fawry.com"),
    ("Jumia Egypt", "jumia.com.eg"),
    ("Noon Egypt", "noon.com"),
    ("Amazon Egypt", "amazon.eg"),
    ("Talabat Egypt", "talabat.com"),
    ("Breadfast", "breadfast.com"),
    ("EgyptAir", "egyptair.com"),
    ("Egypt e-Government portal", "digital.gov.eg"),
]


def main() -> None:
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_rows = list(reader)

    existing_man_ids = [row["id"] for row in existing_rows if row["id"].startswith("MAN-")]
    next_id_num = len(existing_man_ids) + 1

    new_rows = []
    for i, (brand, domain) in enumerate(EGYPTIAN_BRAND_DOMAINS, start=next_id_num):
        new_rows.append({
            "id": f"MAN-{i:06d}",
            "text": f"https://{domain}",
            "content_type": "url",
            "label": "legitimate",
            "language": "mixed",  # Egyptian corporate sites are typically bilingual AR/EN
            "source": "manual",
            "brand_impersonated": "",
            "date_collected": date.today().isoformat(),
            "split": "",
        })

    all_rows = existing_rows + new_rows
    with open(DATASET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DATASET_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Added {len(new_rows)} Egyptian brand domain rows to dataset.csv")
    print(f"dataset.csv now has {len(all_rows):,} total rows")


if __name__ == "__main__":
    main()
