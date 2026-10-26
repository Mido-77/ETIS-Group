"""
main.py

FastAPI application entrypoint (Phase 6). Schema creation runs
automatically on startup via the lifespan handler rather than the manual
Base.metadata.create_all() one-liner used in Phases 0-5 — safe because
create_all() is idempotent, and this removes a manual step that would
otherwise need to be repeated on every fresh clone or hosted deployment
(Railway/Render).

The lifespan handler also warms up the learned models so the first scan is
not slow (see app/core/warmup.py; skip with ETIS_SKIP_WARMUP=1).
"""
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from app.core.rate_limit import limiter

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from fastapi.middleware.cors import CORSMiddleware

from app.core.database import engine, Base
import app.models  # noqa: F401 — registers all models on Base.metadata
from app.core.exceptions import (
    ETISException,
    InvalidScanInputError,
    EngineFailureError,
    BlacklistLookupError,
    AuthenticationError,
)
from app.core.warmup import warm_up_engines


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    warm_up_engines()
    yield


app = FastAPI(
    title="ETIS — Egypt-Specific Threat Intelligence System",
    description="Hybrid phishing detection API for Egyptian SMEs.",
    version="0.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.routers import analyze, threats, stats

app.include_router(analyze.router)
app.include_router(threats.router)
app.include_router(stats.router)

# --- Global exception handling ------------------------------------------
# Maps the ETISException hierarchy to HTTP responses in one place, per
# CLAUDE.md's error-handling convention. Routes and engine code raise the
# custom exceptions directly; they never construct HTTPException inline.

_STATUS_BY_EXCEPTION = {
    InvalidScanInputError: 400,
    BlacklistLookupError: 502,
    EngineFailureError: 500,
    AuthenticationError: 401,
}


@app.exception_handler(ETISException)
async def etis_exception_handler(request: Request, exc: ETISException) -> JSONResponse:
    status_code = _STATUS_BY_EXCEPTION.get(type(exc), 500)
    return JSONResponse(
        status_code=status_code,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


@app.get("/")
async def root() -> dict:
    return {"status": "ok", "service": "ETIS"}
