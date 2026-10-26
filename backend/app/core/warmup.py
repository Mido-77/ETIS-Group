"""
warmup.py

Pre-loads the learned models (and touches the database once) at server
startup, so the first real /analyze request does not pay the one-off cost of
joblib-loading the artifacts and the first predict calls. Measured before
this change: first scan ~10.3 s, later scans ~38 ms (local dev server).

Skippable with ETIS_SKIP_WARMUP=1 (set automatically in tests by
backend/tests/conftest.py, and handy for `uvicorn --reload` development).

A warm-up failure is logged but never stops the server from starting: a
missing artifact will still surface as a clear error on the first real
scan, exactly as before.
"""

import logging
import os
import time

from app.core.database import SessionLocal
from app.engines.ml.text_char_inference import calculate_text_char_phishing_score
from app.engines.ml.url_model_inference import calculate_url_model_score
from app.engines.threat_intel.blacklist_lookup import calculate_blacklist_reputation_score

# uvicorn's logger, so the messages appear in the default uvicorn console output.
logger = logging.getLogger("uvicorn.error")

WARMUP_URL = "https://warmup.invalid/"
WARMUP_TEXT = "تم إيداع راتبكم الشهري بنجاح"


def warmup_is_disabled() -> bool:
    return os.getenv("ETIS_SKIP_WARMUP", "").strip().lower() in ("1", "true", "yes")


def warm_up_engines() -> None:
    if warmup_is_disabled():
        logger.info("ETIS warm-up skipped (ETIS_SKIP_WARMUP is set).")
        return

    start = time.perf_counter()

    steps = [
        ("URL model", lambda: calculate_url_model_score(WARMUP_URL)),
        ("text model", lambda: calculate_text_char_phishing_score(WARMUP_TEXT)),
    ]
    for name, call in steps:
        step_start = time.perf_counter()
        try:
            call()
            logger.info("ETIS warm-up: %s ready (%.0f ms).", name, (time.perf_counter() - step_start) * 1000)
        except Exception:
            logger.warning("ETIS warm-up: %s failed; the first real scan will load it instead.", name, exc_info=True)

    step_start = time.perf_counter()
    db = SessionLocal()
    try:
        calculate_blacklist_reputation_score(WARMUP_URL, db)
        logger.info("ETIS warm-up: database/blacklist ready (%.0f ms).", (time.perf_counter() - step_start) * 1000)
    except Exception:
        logger.warning("ETIS warm-up: database/blacklist check failed.", exc_info=True)
    finally:
        db.close()

    logger.info("ETIS warm-up finished in %.0f ms.", (time.perf_counter() - start) * 1000)
