"""
community_report.py

CommunityReport: user-submitted "this looks suspicious" reports, per the
community reporting feature mentioned in FYP_REPORT.docx's abstract.
Pending review by an analyst before being promoted into blacklist_entries.
"""

from datetime import datetime

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base

REPORT_STATUSES = ("pending", "confirmed", "rejected")


class CommunityReport(Base):
    __tablename__ = "community_reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    reported_value: Mapped[str] = mapped_column(Text, nullable=False)
    reporter_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)
    submitted_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"<CommunityReport id={self.id} status={self.status}>"
