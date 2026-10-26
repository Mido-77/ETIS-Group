"""
routers/analyze.py

Phase 6: the POST /analyze endpoint. Thin wiring layer over
calculate_final_risk_score() -- validation and scoring logic live in the
engines/scoring layer, not here. (Updated 2026-10 for the decision-tier
scorer: engine_tiers/decided_by replace weighted_contributions; engines are
rule_based, blacklist, url_model, text_model.)

Each successful scan is persisted as a ScanRequest + ScanResult pair for
history/stats purposes. Persistence is treated as non-fatal: the computed
score is always returned to the caller, even if saving to history fails.
For a security tool, losing the result because of an unrelated storage
problem is a worse failure mode than losing history -- the response's
`saved` / `save_error` fields tell the caller whether the scan was
recorded, so the dashboard can offer a retry without discarding the score
the user already has in front of them.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db_session
from app.core.security import verify_api_key
from app.engines.risk_scoring import calculate_final_risk_score
from app.models.scan import ScanRequest, ScanResult

from fastapi import Request
from app.core.rate_limit import limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyze", tags=["analyze"])


class AnalyzeRequest(BaseModel):
    text: Optional[str] = Field(
        default=None, description="Email or message body to analyze."
    )
    url: Optional[str] = Field(
        default=None, description="URL to analyze."
    )


class SkipReasons(BaseModel):
    blacklist: Optional[str] = Field(
        default=None,
        description=(
            "Why the blacklist engine did not run, e.g. 'no_url_provided'. "
            "Null when the engine ran."
        ),
    )
    url_model: Optional[str] = Field(
        default=None,
        description=(
            "Why the learned URL model did not run, e.g. 'no_url_provided'. "
            "Null when the engine ran."
        ),
    )
    text_model: Optional[str] = Field(
        default=None,
        description=(
            "Why the learned text model did not run, e.g. 'no_text_provided'. "
            "Null when the engine ran."
        ),
    )


class StageTimings(BaseModel):
    rule_based_ms: float = Field(
        description="Wall-clock time spent in the rule-based engine, in milliseconds. Always present."
    )
    blacklist_ms: Optional[float] = Field(
        default=None,
        description="Wall-clock time spent in the blacklist engine, in milliseconds. Null when it didn't run.",
    )
    url_model_ms: Optional[float] = Field(
        default=None,
        description="Wall-clock time spent in the learned URL model, in milliseconds. Null when it didn't run.",
    )
    text_model_ms: Optional[float] = Field(
        default=None,
        description="Wall-clock time spent in the learned text model, in milliseconds. Null when it didn't run.",
    )
    total_ms: float = Field(
        description=(
            "Total wall-clock time for the full aggregation -- every "
            "applicable engine call plus the tier/fusion step. "
            "Not the same as the HTTP request's total latency (excludes "
            "persistence and response serialization)."
        )
    )


class Confidence(BaseModel):
    """
    A heuristic "how much independent evidence backed this result" signal --
    NOT a calibrated statistical confidence. Computed from the applicable
    engines' raw 0-100 scores, before the Egyptian brand allowlist override.
    Kept for API compatibility; the dashboard no longer displays it (see
    app/engines/risk_scoring.py).
    """

    score: float = Field(
        description=(
            "0-100 headline confidence: round(0.5 * coverage + 0.5 * "
            "agreement, 2). A design-choice weighting, not a calibrated one."
        )
    )
    coverage: float = Field(
        description=(
            "0-100. How many of the four engines applied to this "
            "submission (engines_considered / 4 * 100): 75 (url-only), "
            "50 (text-only) or 100 (both provided)."
        )
    )
    agreement: float = Field(
        description=(
            "0-100. How closely the applicable engines' raw scores agreed: "
            "100 - (max(scores) - min(scores)). Lower when one engine "
            "flags strongly and another finds nothing."
        )
    )
    engines_considered: int = Field(
        description="How many engines' scores fed this calculation. Between 2 and 4."
    )


class AnalyzeResponse(BaseModel):
    final_score: float
    verdict: str
    engines_applied: list[str]
    engine_tiers: dict[str, str] = Field(
        description=(
            "Tier (LOW/MEDIUM/HIGH) of each applicable engine, after the "
            "Egyptian brand allowlist override. The verdict is the highest tier."
        ),
    )
    decided_by: Optional[str] = Field(
        default=None,
        description="The engine that produced the verdict. Null when the verdict is LOW.",
    )
    engine_details: dict
    skip_reasons: SkipReasons = Field(
        description=(
            "Machine-readable reason for each engine that did not apply "
            "to this submission (everything except rule_based, which always "
            "runs). Null per-field when that engine ran."
        ),
    )
    stage_timings: StageTimings = Field(
        description=(
            "Per-engine and total processing time in milliseconds, for "
            "explainability/presentation purposes (e.g. showing where "
            "latency comes from). Engines that did not run have null timings."
        ),
    )
    confidence: Confidence = Field(
        description=(
            "Heuristic explainability signal -- how much independent "
            "evidence backed this result. Purely additive: never derived "
            "from, and never affects, final_score/verdict/engine_details."
        ),
    )
    allowlisted: bool = Field(
        default=False,
        description=(
            "True when the submitted URL is a verified Egyptian brand domain "
            "(or a genuine subdomain) -- the URL-derived engines' tiers were "
            "set to LOW. Evidence from the message text is NOT cancelled. "
            "engine_details still shows each engine's real, un-overridden output."
        ),
    )
    saved: bool = Field(
        description="Whether this scan was successfully saved to history."
    )
    save_error: Optional[str] = Field(
        default=None,
        description="If saving failed, a brief explanation. Null if saved=true.",
    )


@router.post("", response_model=AnalyzeResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("20/minute")
async def analyze(
    request: Request,
    payload: AnalyzeRequest,
    db: Session = Depends(get_db_session),
) -> dict:
    """
    Submit a URL and/or text for phishing risk analysis.

    The score is always returned, even if saving to history fails. If
    saving fails, `saved` is false and `save_error` explains why; the
    caller can resubmit the same request to attempt saving again.
    """
    result = calculate_final_risk_score(text=payload.text, url=payload.url, db=db)

    try:
        scan_request = ScanRequest(url_value=payload.url, text_value=payload.text)
        db.add(scan_request)
        db.flush()  # populate scan_request.id before creating the linked ScanResult

        scan_result = ScanResult(
            scan_request_id=scan_request.id,
            rule_based_score=result["engine_details"]["rule_based"]["score"],
            blacklist_score=(
                result["engine_details"]["blacklist"]["score"]
                if result["engine_details"]["blacklist"] is not None
                else None
            ),
            # The BiLSTM is no longer part of the live scorer; the learned
            # text/URL model outputs are stored in engine_breakdown (JSON).
            # The scan_results table has no columns for them, and adding
            # columns would need a migration for existing databases.
            lstm_score=None,
            final_score=result["final_score"],
            verdict=result["verdict"],
            engine_breakdown=result["engine_details"],
        )
        db.add(scan_result)
        db.commit()
        result["saved"] = True
        result["save_error"] = None
    except Exception as exc:
        db.rollback()
        logger.error("Failed to save scan to history: %s", exc, exc_info=True)
        result["saved"] = False
        result["save_error"] = (
            "Your score is accurate, but this scan could not be saved to "
            "history. You can copy your result now, or resubmit to try "
            "saving again."
        )

    return result
