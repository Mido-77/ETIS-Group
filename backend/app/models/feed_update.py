"""
feed_update.py

FeedUpdate: logs each run of the blacklist feed refresh (PhishTank,
OpenPhish, etc.) -- the "update mechanisms" requirement from plan.docx
section 3.7.
"""

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base

FEED_UPDATE_STATUSES = ("success", "failed")


class FeedUpdate(Base):
    __tablename__ = "feed_updates"

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    rows_added: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    rows_updated: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="success", nullable=False)
    run_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def __repr__(self) -> str:
        return f"<FeedUpdate id={self.id} source={self.source} status={self.status}>"
