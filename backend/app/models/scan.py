"""
scan.py

ScanRequest + ScanResult: records every scan submitted through the
dashboard (Phase 6+) and the resulting per-engine score breakdown.
One-to-one: each ScanRequest has exactly one ScanResult once scoring
completes.

url_value / text_value are both nullable, independently, rather than a
single input_value + input_type tag -- a scan can legitimately include a
URL, text, or both at once (see risk_scoring.py's three-case engine
applicability), and forcing that into one field + a type label would
either lose the second value or require an arbitrary priority choice
between them. At least one of the two is always non-null, enforced at the
application layer (same InvalidScanInputError precondition already used
by calculate_final_risk_score()).
"""

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

SCAN_VERDICTS = ("LOW", "MEDIUM", "HIGH")


class ScanRequest(Base):
    __tablename__ = "scan_requests"

    id: Mapped[int] = mapped_column(primary_key=True)
    url_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    text_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), index=True
    )

    result: Mapped["ScanResult | None"] = relationship(
        back_populates="scan_request", uselist=False, cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        kind = "url+text" if self.url_value and self.text_value else (
            "url" if self.url_value else "text"
        )
        return f"<ScanRequest id={self.id} kind={kind}>"


class ScanResult(Base):
    __tablename__ = "scan_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    scan_request_id: Mapped[int] = mapped_column(
        ForeignKey("scan_requests.id"), unique=True, nullable=False
    )
    rule_based_score: Mapped[float] = mapped_column(Float, nullable=False)
    blacklist_score: Mapped[float | None] = mapped_column(Float, nullable=True)  # null when no URL submitted
    lstm_score: Mapped[float | None] = mapped_column(Float, nullable=True)  # null when no text submitted
    final_score: Mapped[float] = mapped_column(Float, nullable=False)
    verdict: Mapped[str] = mapped_column(String(10), nullable=False)
    engine_breakdown: Mapped[dict] = mapped_column(JSON, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    scan_request: Mapped["ScanRequest"] = relationship(back_populates="result")

    def __repr__(self) -> str:
        return f"<ScanResult id={self.id} verdict={self.verdict} final_score={self.final_score}>"