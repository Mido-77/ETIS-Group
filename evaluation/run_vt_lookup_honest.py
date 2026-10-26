"""
run_vt_lookup_honest.py   (Task B, step 2 of 3)

Looks up each URL of evaluation/vt_honest_sample.csv on VirusTotal (API v3,
GET /urls/{id}) and stores the per-URL engine counts in
evaluation/vt_honest_results.csv.

  * LOOKUP only: it reads the existing report for a URL. It never submits a
    URL for scanning, so it neither contacts the (possibly live) phishing
    site nor adds anything to VirusTotal.
  * A URL VirusTotal has never seen returns 404 -> status "not_found". The
    comparison script counts that as "not flagged", i.e. what a user checking
    the URL on virustotal.com would see. It is NOT a fresh scan.
  * Free tier: 4 requests/minute -> one request every ~15.5 s (200 URLs ~ 52 min).
    Leave the window alone. The script is RESUMABLE: re-run it and it skips
    rows already stored as found/not_found and retries errors.
  * API key: read from the environment variable VT_API_KEY, in the same
    PowerShell window. It is never printed, logged or written to a file.

Usage (project root):
    $env:VT_API_KEY = "<your key>"      # in this window only; do not paste it in chat
    python evaluation/run_vt_lookup_honest.py
"""

import base64
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = PROJECT_ROOT / "evaluation" / "vt_honest_sample.csv"
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "vt_honest_results.csv"
API_BASE = "https://www.virustotal.com/api/v3/urls/"
SECONDS_BETWEEN_REQUESTS = 15.5
RATE_LIMIT_BACKOFF_SECONDS = 65
MAX_RETRIES_ON_429 = 3
FIELDS = ["id", "url", "label", "status", "malicious", "suspicious", "harmless", "undetected", "timeout",
          "last_analysis_date"]


def url_id(url: str) -> str:
    """VirusTotal URL identifier: URL-safe base64 of the URL, without padding."""
    return base64.urlsafe_b64encode(url.encode("utf-8")).decode("ascii").rstrip("=")


def lookup(url: str, api_key: str) -> dict:
    request = urllib.request.Request(API_BASE + url_id(url), headers={"x-apikey": api_key, "accept": "application/json"})
    for attempt in range(MAX_RETRIES_ON_429 + 1):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.load(response)
            attrs = data["data"]["attributes"]
            stats = attrs.get("last_analysis_stats", {}) or {}
            return {"status": "found", "malicious": stats.get("malicious", 0), "suspicious": stats.get("suspicious", 0),
                    "harmless": stats.get("harmless", 0), "undetected": stats.get("undetected", 0),
                    "timeout": stats.get("timeout", 0), "last_analysis_date": attrs.get("last_analysis_date", "")}
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return {"status": "not_found"}
            if exc.code == 429 and attempt < MAX_RETRIES_ON_429:
                print(f"    rate limited (429); waiting {RATE_LIMIT_BACKOFF_SECONDS}s then retrying")
                time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
                continue
            if exc.code in (401, 403):
                sys.exit(f"VirusTotal refused the key (HTTP {exc.code}). Check VT_API_KEY in this window.")
            return {"status": f"error_http_{exc.code}"}
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            return {"status": f"error_{type(exc).__name__}"}
    return {"status": "error_rate_limited"}


def main() -> None:
    api_key = os.environ.get("VT_API_KEY", "").strip()
    if not api_key:
        sys.exit('VT_API_KEY is not set in this window. Run:  $env:VT_API_KEY = "<your key>"   (do not paste the key in chat)')
    if not SAMPLE_PATH.exists():
        sys.exit(f"{SAMPLE_PATH} not found: run build_vt_honest_sample.py first.")

    with open(SAMPLE_PATH, newline="", encoding="utf-8") as f:
        sample = list(csv.DictReader(f))

    done: dict[str, dict] = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["status"] in ("found", "not_found"):
                    done[row["id"]] = row
    todo = [r for r in sample if r["id"] not in done]
    print(f"{len(sample)} URLs in sample, {len(done)} already stored, {len(todo)} to look up "
          f"(~{len(todo) * SECONDS_BETWEEN_REQUESTS / 60:.0f} min).")

    # rewrite the results file keeping only completed rows (errors are retried)
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in done.values():
            writer.writerow({k: row.get(k, "") for k in FIELDS})

    started = time.time()
    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        for n, row in enumerate(todo, start=1):
            result = lookup(row["url"], api_key)
            writer.writerow({"id": row["id"], "url": row["url"], "label": row["label"],
                             **{k: result.get(k, "") for k in FIELDS[3:]}})
            f.flush()
            if n % 10 == 0 or n == len(todo):
                print(f"  ...{n}/{len(todo)} looked up ({(time.time() - started) / 60:.1f} min elapsed); last status: {result['status']}")
            if n < len(todo):
                time.sleep(SECONDS_BETWEEN_REQUESTS)

    with open(RESULTS_PATH, newline="", encoding="utf-8") as f:
        statuses = [r["status"] for r in csv.DictReader(f)]
    from collections import Counter
    print("\nStatus counts:", dict(Counter(statuses)))
    if any(s.startswith("error") for s in statuses):
        print("Some rows are errors: simply run the script again to retry them.")


if __name__ == "__main__":
    main()
