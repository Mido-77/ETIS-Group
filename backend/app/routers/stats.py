"""
routers/stats.py

Phase 6: GET /stats -- summary numbers for the dashboard, pulled from the
blacklist (seeded threat intel) and scan history (live usage, Phase 6+).

Deliberately returns a plain dict rather than a rigid Pydantic response
model with a fixed set of fields -- stats requirements tend to grow as the
dashboard evolves (e.g. time-windowed counts, most-targeted brand), and a
flexible dict means adding a new stat later is a one-line addition to
_build_stats(), not a schema migration. Each section below is a natural
place to extend.
"""

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.database import get_db_session
from app.core.security import verify_api_key
from app.models.blacklist import BlacklistEntry
from app.models.scan import ScanResult

router = APIRouter(prefix="/stats", tags=["stats"])


def _blacklist_stats(db: Session) -> dict:
    total = db.query(func.count(BlacklistEntry.id)).scalar()
    active = db.query(func.count(BlacklistEntry.id)).filter(BlacklistEntry.is_active.is_(True)).scalar()

    by_source_rows = (
        db.query(BlacklistEntry.source, func.count(BlacklistEntry.id))
        .group_by(BlacklistEntry.source)
        .all()
    )

    return {
        "total_entries": total,
        "active_entries": active,
        "by_source": {source: count for source, count in by_source_rows},
        # Future extension point: most-targeted brand, entries added in
        # last 7 days, etc.
    }


def _scan_stats(db: Session) -> dict:
    total_scans = db.query(func.count(ScanResult.id)).scalar()

    by_verdict_rows = (
        db.query(ScanResult.verdict, func.count(ScanResult.id))
        .group_by(ScanResult.verdict)
        .all()
    )
    by_verdict = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    by_verdict.update({verdict: count for verdict, count in by_verdict_rows})

    avg_score = db.query(func.avg(ScanResult.final_score)).scalar()

    return {
        "total_scans": total_scans,
        "by_verdict": by_verdict,
        "average_final_score": round(avg_score, 2) if avg_score is not None else None,
        # Future extension point: scans in last 24h, average per-engine
        # contribution, most common brand_target among HIGH verdicts.
    }


@router.get("", dependencies=[Depends(verify_api_key)])
async def get_stats(db: Session = Depends(get_db_session)) -> dict:
    """Summary numbers for the dashboard: blacklist size and scan history."""
    return {
        "blacklist": _blacklist_stats(db),
        "scans": _scan_stats(db),
    }