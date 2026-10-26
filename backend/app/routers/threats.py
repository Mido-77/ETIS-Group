"""
routers/threats.py

Phase 6: GET /threats -- paginated, filterable read access to the seeded
threat-intelligence blacklist (BlacklistEntry). Domain + metadata only,
deliberately not the raw url/url_hash -- exposing 65K+ live phishing URLs
in bulk through a public API is unnecessary surface area for a defensive
tool to hand out, even though the underlying data (PhishTank/OpenPhish) is
itself public. If a future use case genuinely needs the raw URL (e.g. an
authenticated analyst view), that's a deliberate addition, not a default.
"""

from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db_session
from app.core.security import verify_api_key
from app.models.blacklist import BlacklistEntry

router = APIRouter(prefix="/threats", tags=["threats"])


class ThreatEntry(BaseModel):
    id: int
    domain: str
    source: str
    brand_target: Optional[str]
    reputation_score: float
    is_active: bool
    first_seen: str
    last_seen: str

    class Config:
        from_attributes = True


class ThreatsResponse(BaseModel):
    total_matching: int
    limit: int
    offset: int
    results: list[ThreatEntry]


@router.get("", response_model=ThreatsResponse, dependencies=[Depends(verify_api_key)])
async def list_threats(
    source: Optional[str] = Query(default=None, description="Filter by source: phishtank, openphish, manual"),
    is_active: Optional[bool] = Query(default=None, description="Filter by active status"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db_session),
) -> dict:
    """
    Browse the seeded threat-intelligence blacklist. Domain-level info
    only -- see module docstring for why the raw URL is omitted.
    """
    query = db.query(BlacklistEntry)

    if source is not None:
        query = query.filter(BlacklistEntry.source == source)
    if is_active is not None:
        query = query.filter(BlacklistEntry.is_active == is_active)

    total_matching = query.count()
    entries = (
        query.order_by(BlacklistEntry.last_seen.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return {
        "total_matching": total_matching,
        "limit": limit,
        "offset": offset,
        "results": [
            ThreatEntry(
                id=e.id,
                domain=e.domain,
                source=e.source,
                brand_target=e.brand_target,
                reputation_score=e.reputation_score,
                is_active=e.is_active,
                first_seen=e.first_seen.isoformat(),
                last_seen=e.last_seen.isoformat(),
            )
            for e in entries
        ],
    }