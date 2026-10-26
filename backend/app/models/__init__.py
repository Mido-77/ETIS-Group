"""
Import all models here so Base.metadata.create_all() (or Alembic) can
discover every table by importing this package once.
"""

from app.models.blacklist import BlacklistEntry
from app.models.brand import BrandDomain, EgyptianBrand
from app.models.community_report import CommunityReport
from app.models.feed_update import FeedUpdate
from app.models.scan import ScanRequest, ScanResult

__all__ = [
    "BlacklistEntry",
    "EgyptianBrand",
    "BrandDomain",
    "ScanRequest",
    "ScanResult",
    "CommunityReport",
    "FeedUpdate",
]
