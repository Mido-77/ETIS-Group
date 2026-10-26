"""
rate_limit.py

Minimal rate limiting for publicly hosted ETIS endpoints, via slowapi
(built on the `limits` package). Deliberately simple -- keyed by client
IP address only, no per-user/per-key tiers, matching the same
scope-cutting rationale already used for verify_api_key in security.py
(a single-operator demo doesn't need a tiered rate-limiting policy).

Applied selectively, not globally: /analyze is the expensive endpoint
(real LSTM inference + a DB write per call, ~175ms per Table 5.4's own
benchmark) and is the one worth protecting. /threats and /stats are
cheap reads and are left unlimited.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)