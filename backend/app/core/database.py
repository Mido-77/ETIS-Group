"""
database.py

SQLAlchemy engine, session factory, and declarative base for ETIS.
Reads DATABASE_URL from the .env file (see SETUP.md step 4).
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# Explicitly locate .env at the project root, regardless of which directory
# this script is run from (load_dotenv() with no args only checks the
# current working directory, which caused connection failures when running
# from inside backend/ instead of the project root).
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # core -> app -> backend -> root
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/etis"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db_session() -> Session:
    """FastAPI dependency (used starting Phase 6): yields a DB session and
    always closes it, even if the request raises."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
