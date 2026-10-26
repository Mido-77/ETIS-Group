"""
brand.py

EgyptianBrand + BrandDomain: the verified Egyptian brand registry, seeded
from data/scripts/add_egyptian_domains.py (Phase 1) via
data/scripts/seed_database.py (Phase 3). One brand can have multiple real
domains (e.g. QNB Al Ahli has both qnbalahli.com and qnb.com.eg).
"""

from datetime import date, datetime

from sqlalchemy import Date, DateTime, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class EgyptianBrand(Base):
    __tablename__ = "egyptian_brands"

    id: Mapped[int] = mapped_column(primary_key=True)
    brand_name: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    domains: Mapped[list["BrandDomain"]] = relationship(
        back_populates="brand", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<EgyptianBrand id={self.id} brand_name={self.brand_name!r}>"


class BrandDomain(Base):
    __tablename__ = "brand_domains"

    id: Mapped[int] = mapped_column(primary_key=True)
    brand_id: Mapped[int] = mapped_column(ForeignKey("egyptian_brands.id"), nullable=False)
    domain: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    verified_date: Mapped[date] = mapped_column(Date, nullable=False)

    brand: Mapped["EgyptianBrand"] = relationship(back_populates="domains")

    def __repr__(self) -> str:
        return f"<BrandDomain id={self.id} domain={self.domain!r}>"
