"""
seed_database.py

Seeds the ETIS database from Phase 1 data:
  1. blacklist_entries  <- data/processed/dataset.csv (PhishTank + OpenPhish rows)
  2. egyptian_brands + brand_domains  <- app/core/egyptian_brands.py (canonical registry)

Also logs a FeedUpdate row summarizing the run.

Re-running is safe -- existing URLs are skipped (not duplicated), and
brands/domains are only inserted if not already present. Tables are created
if they don't exist yet (needed for a fresh evaluation database).

NEW (evaluation rebuild): --only-split train seeds ONLY the training split's
phishing URLs. Use it with a SEPARATE evaluation database so the blacklist
never contains validation/test answers:

    # 1. create the evaluation database once
    psql -U postgres -c "CREATE DATABASE etis_eval;"
    # 2. point this terminal at it (copy DATABASE_URL from .env, change the db name)
    $env:DATABASE_URL = "postgresql://postgres:PASSWORD@localhost:5432/etis_eval"
    # 3. seed it from the training split only
    python data/scripts/seed_database.py --only-split train

The demo/production database ("etis") keeps the full feed (no flag). An
environment variable DATABASE_URL overrides the value in .env, so the
evaluation scripts (calibrate_thresholds.py, evaluate_*.py) use whichever
database the current terminal points at.

Usage:
    python data/scripts/seed_database.py                      # full feed (demo DB)
    python data/scripts/seed_database.py --only-split train   # evaluation DB
"""

import argparse
import csv
import hashlib
import sys
from datetime import datetime
from pathlib import Path

# Make the backend/ "app" package importable when running this script directly.
BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.core.database import Base, SessionLocal, engine
from app.core.egyptian_brands import EGYPTIAN_BRAND_DOMAINS
from app.engines.rule_based.domain_typosquatting_detector import normalize_domain
from app.models import BlacklistEntry, BrandDomain, EgyptianBrand, FeedUpdate  # registers all tables

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"

BATCH_SIZE = 2000


def _hash_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def seed_blacklist_entries(db, only_split: str | None = None) -> tuple[int, int]:
    """Load PhishTank/OpenPhish rows from dataset.csv into blacklist_entries.
    If only_split is given (e.g. "train"), rows from other splits are ignored.
    Returns (rows_added, rows_skipped_as_duplicate)."""

    existing_hashes = {row[0] for row in db.query(BlacklistEntry.url_hash).all()}
    seen_in_this_run = set()

    rows_added = 0
    rows_skipped = 0
    batch = []

    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["source"] not in ("phishtank", "openphish"):
                continue
            if only_split is not None and row["split"] != only_split:
                continue

            url = row["text"]
            url_hash = _hash_url(url)
            if url_hash in existing_hashes or url_hash in seen_in_this_run:
                rows_skipped += 1
                continue
            seen_in_this_run.add(url_hash)

            collected_date = datetime.strptime(row["date_collected"], "%Y-%m-%d").date()
            brand_target = row["brand_impersonated"].strip() or None
            # PhishTank uses "Other" for unidentified brands -- not a real brand name.
            if brand_target == "Other":
                brand_target = None

            batch.append(
                BlacklistEntry(
                    url=url,
                    url_hash=url_hash,
                    domain=normalize_domain(url),
                    source=row["source"],
                    brand_target=brand_target,
                    first_seen=collected_date,
                    last_seen=collected_date,
                )
            )

            if len(batch) >= BATCH_SIZE:
                db.add_all(batch)
                db.commit()
                rows_added += len(batch)
                print(f"  ...{rows_added:,} rows inserted so far")
                batch = []

    if batch:
        db.add_all(batch)
        db.commit()
        rows_added += len(batch)

    return rows_added, rows_skipped


def seed_brand_registry(db) -> tuple[int, int]:
    """Load the canonical Egyptian brand registry. Returns (brands_added, domains_added)."""
    existing_brand_names = {row[0] for row in db.query(EgyptianBrand.brand_name).all()}
    existing_domains = {row[0] for row in db.query(BrandDomain.domain).all()}

    brands_by_name: dict[str, EgyptianBrand] = {}
    brands_added = 0
    domains_added = 0
    today = datetime.now().date()

    for brand_name, domain in EGYPTIAN_BRAND_DOMAINS:
        if domain in existing_domains:
            continue

        if brand_name not in brands_by_name:
            if brand_name in existing_brand_names:
                brand = db.query(EgyptianBrand).filter_by(brand_name=brand_name).one()
            else:
                # Category left generic here -- categories were tracked informally
                # in Phase 1; refine per-brand if needed later.
                brand = EgyptianBrand(brand_name=brand_name, category="Uncategorized")
                db.add(brand)
                db.flush()  # get brand.id without a full commit
                brands_added += 1
            brands_by_name[brand_name] = brand

        db.add(BrandDomain(brand_id=brands_by_name[brand_name].id, domain=domain, verified_date=today))
        domains_added += 1

    db.commit()
    return brands_added, domains_added


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only-split",
        choices=["train", "val", "test"],
        default=None,
        help="seed only this split's URLs (use 'train' for the evaluation database)",
    )
    args = parser.parse_args()

    # Show exactly which database is being written to -- guards against
    # accidentally seeding the wrong one.
    target = engine.url.render_as_string(hide_password=True)
    print(f"Target database: {target}")
    if args.only_split is None:
        print("Mode: FULL feed (all splits)")
    else:
        print(f"Mode: {args.only_split.upper()} split only")

    Base.metadata.create_all(bind=engine)  # idempotent; needed for a fresh database

    db = SessionLocal()

    print("Seeding blacklist_entries from PhishTank + OpenPhish...")
    blacklist_added, blacklist_skipped = seed_blacklist_entries(db, args.only_split)
    print(f"  -> {blacklist_added:,} added, {blacklist_skipped:,} already present (skipped)")

    print("Seeding egyptian_brands + brand_domains...")
    brands_added, domains_added = seed_brand_registry(db)
    print(f"  -> {brands_added} brands added, {domains_added} domains added")

    feed_source = "initial_seed" if args.only_split is None else f"eval_seed_{args.only_split}"
    db.add(FeedUpdate(source=feed_source, rows_added=blacklist_added, rows_updated=0))
    db.commit()

    db.close()
    print()
    print("Seeding complete.")


if __name__ == "__main__":
    main()
