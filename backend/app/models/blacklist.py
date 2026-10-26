"""
blacklist.py

BlacklistEntry: the core threat-intelligence table, seeded from PhishTank
and OpenPhish (Phase 1 data) and refreshed periodically (see FeedUpdate).

Uniqueness is enforced on `url_hash` (a SHA-256 hash of the URL), not on
`url` directly -- PostgreSQL's B-tree index has an 8191-byte row limit, and
real phishing URLs are sometimes deliberately obfuscated to be extremely
long, which exceeds that limit if the raw URL itself is indexed.

`domain` is stored separately (extracted at seed time) so the lookup engine
can efficiently match on domain alone, not just exact URL -- real phishing
often reuses one malicious domain across many different paths.
"""

from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, Float, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base

BLACKLIST_SOURCES = ("phishtank", "openphish", "manual")


class BlacklistEntry(Base):
    __tablename__ = "blacklist_entries"

    id: Mapped[int] = mapped_column(primary_key=True)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    url_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    domain: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    brand_target: Mapped[str | None] = mapped_column(String(255), nullable=True)
    reputation_score: Mapped[float] = mapped_column(Float, default=90.0, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    first_seen: Mapped[date] = mapped_column(Date, nullable=False)
    last_seen: Mapped[date] = mapped_column(Date, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def __repr__(self) -> str:
        return f"<BlacklistEntry id={self.id} source={self.source} url={self.url[:50]!r}>"
