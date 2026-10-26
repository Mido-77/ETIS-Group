"""
benchmark_api_performance.py

Chapter 5.7 (System Performance) needs response time, throughput, and
resource utilization numbers -- none of which can come from scoring the
dataset directly, unlike every other evaluation script so far. This one
hits your ACTUALLY RUNNING FastAPI server over real HTTP, the same way a
real user's browser would, and times it.

REQUIRES: uvicorn running (uvicorn app.main:app --reload, from backend/)
in a SEPARATE terminal before running this script. Unlike
calibrate_thresholds.py / evaluate_test_configurations.py /
evaluate_arabic_breakdown.py, which call engine functions directly and
therefore only need Postgres running, this script needs the full HTTP
stack up -- it is testing the stack, not just the engines.

Only stdlib is used (urllib, concurrent.futures, statistics) -- no new
pip install needed. Resource-utilization sampling via psutil is
attempted but optional -- if psutil isn't installed, that section is
skipped with a note rather than failing the whole run (consistent with
CLAUDE.md's "don't over-invest" testing philosophy: response time and
throughput are the numbers Chapter 5.7 actually needs; resource sampling
is a bonus if easy, not a blocker if not).

What gets measured:
    1. Sequential latency, broken out by input type (url-only,
       text-only, both) -- because these hit different engine
       combinations (blacklist DB lookup vs. LSTM inference vs. both),
       and Chapter 5.7 should report them separately, not pooled, since
       pooling would hide that LSTM inference is the slow path.
    2. Sequential throughput (requests/sec, one request at a time).
    3. Concurrent throughput (requests/sec, CONCURRENCY requests in
       flight at once) -- more representative of real multi-user SME
       usage than pure sequential timing.
    4. GET /threats and GET /stats latency (lighter read paths, useful
       contrast against /analyze's heavier compute path).
    5. Server process CPU/memory during the concurrent burst, IF psutil
       is available AND you tell the script the uvicorn process's PID.

CAVEAT for Chapter 5.7 prose: these numbers reflect local development
hardware (see Chapter 5.1's hardware spec) over localhost -- no network
latency, no hosted-deployment contention. State this explicitly; do not
present these as representative of the Railway/Render hosted demo's
performance without a separate benchmark against that deployment.

Run from anywhere, with uvicorn already running in another terminal:
    python evaluation/benchmark_api_performance.py

Outputs (written to evaluation/):
    api_benchmark_report.txt
    api_benchmark_latency.png
    api_benchmark_throughput.png
"""

import concurrent.futures
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "evaluation"

BASE_URL = os.getenv("ETIS_BENCHMARK_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("ETIS_API_KEY")  # matches backend's verify_api_key no-op when unset

# IMPORTANT (Windows-specific): urllib's default opener checks system proxy
# settings on every request, including a WPAD (Web Proxy Auto-Discovery) DNS
# lookup even when no proxy is actually configured. On Windows this commonly
# adds a fixed ~1-2 SECOND delay to EVERY request regardless of what the
# server does -- easy to mistake for real server latency (it isn't). Using
# "127.0.0.1" instead of "localhost" avoids one source of this; explicitly
# building an opener with an empty ProxyHandler avoids the other. Together
# these remove the fixed per-request tax so the numbers reported actually
# reflect ETIS's own processing time, not Windows network-stack overhead.
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
urllib.request.install_opener(_opener)

N_PER_CATEGORY = 40       # sequential latency sample size per input-type category
CONCURRENCY = 10          # simultaneous in-flight requests for the concurrent throughput test
CONCURRENT_TOTAL = 100    # total requests fired during the concurrent burst

# Optional: set this to the uvicorn process's PID (shown in the terminal
# running uvicorn, or via Task Manager) to sample server-side CPU/memory
# during the concurrent burst. Leave as None to skip resource sampling.
UVICORN_PID = os.getenv("ETIS_UVICORN_PID")


def _post_analyze(payload: dict) -> tuple[float, int]:
    """Returns (latency_seconds, http_status). Raises on connection failure."""
    url = f"{BASE_URL}/analyze"
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
            status = resp.status
    except urllib.error.HTTPError as e:
        e.read()
        status = e.code
    elapsed = time.perf_counter() - start
    return elapsed, status


def _get(path: str) -> tuple[float, int]:
    url = f"{BASE_URL}{path}"
    headers = {}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    req = urllib.request.Request(url, headers=headers, method="GET")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
            status = resp.status
    except urllib.error.HTTPError as e:
        e.read()
        status = e.code
    elapsed = time.perf_counter() - start
    return elapsed, status


def check_server_up() -> None:
    try:
        elapsed, status = _get("/")
        print(f"Server reachable at {BASE_URL} (status {status}, {elapsed*1000:.0f}ms)\n")
    except Exception as e:
        print(f"ERROR: cannot reach {BASE_URL} -- is uvicorn running?")
        print(f"  ({e})")
        print("  Start it first: uvicorn app.main:app --reload  (from backend/)")
        sys.exit(1)


def build_sample_inputs():
    """Pull real rows from the test split to use as realistic benchmark
    inputs -- not synthetic Lorem-ipsum text, actual data the system was
    evaluated on elsewhere in Chapter 5."""
    df = pd.read_csv(DATASET_PATH)
    test_df = df[df["split"] == "test"]

    url_rows = test_df[test_df["content_type"] == "url"].sample(n=N_PER_CATEGORY, random_state=42)
    text_rows = test_df[test_df["content_type"].isin(["text", "email"])].sample(
        n=min(N_PER_CATEGORY, len(test_df[test_df["content_type"].isin(["text", "email"])])),
        random_state=42,
    )

    url_only = [{"url": row["text"]} for _, row in url_rows.iterrows()]
    text_only = [{"text": row["text"]} for _, row in text_rows.iterrows()]
    # "both" case: pair each sampled URL with a sampled text row (synthetic
    # combination -- realistic for exercising the code path, not meant to
    # represent a naturally-occurring dataset row).
    both = [
        {"url": url_rows.iloc[i]["text"], "text": text_rows.iloc[i % len(text_rows)]["text"]}
        for i in range(min(N_PER_CATEGORY, len(url_rows)))
    ]

    return {"url_only": url_only, "text_only": text_only, "both": both}


def summarize(latencies: list) -> dict:
    if not latencies:
        return {}
    sorted_l = sorted(latencies)
    return {
        "n": len(sorted_l),
        "mean_ms": statistics.mean(sorted_l) * 1000,
        "median_ms": statistics.median(sorted_l) * 1000,
        "p95_ms": sorted_l[int(len(sorted_l) * 0.95) - 1] * 1000,
        "p99_ms": sorted_l[min(int(len(sorted_l) * 0.99), len(sorted_l) - 1)] * 1000,
        "min_ms": min(sorted_l) * 1000,
        "max_ms": max(sorted_l) * 1000,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    check_server_up()

    print(f"Sampling {N_PER_CATEGORY} rows per category from the test split...")
    samples = build_sample_inputs()

    # --- 1. Sequential latency by category ---
    latencies_by_category = {}
    errors_by_category = {}
    for category, payloads in samples.items():
        print(f"\nSequential latency test: {category} (N={len(payloads)})")
        lat = []
        errs = 0
        for i, payload in enumerate(payloads):
            try:
                elapsed, status = _post_analyze(payload)
                if status == 200:
                    lat.append(elapsed)
                else:
                    errs += 1
            except Exception as e:
                errs += 1
            if (i + 1) % 10 == 0:
                print(f"  ...{i+1}/{len(payloads)}")
        latencies_by_category[category] = lat
        errors_by_category[category] = errs

    # --- 2. Sequential throughput (reuse url_only requests, timed as a batch) ---
    print(f"\nSequential throughput test ({len(samples['url_only'])} requests, one at a time)...")
    start = time.perf_counter()
    for payload in samples["url_only"]:
        try:
            _post_analyze(payload)
        except Exception:
            pass
    seq_elapsed = time.perf_counter() - start
    seq_throughput = len(samples["url_only"]) / seq_elapsed if seq_elapsed > 0 else 0

    # --- 3. Concurrent throughput ---
    print(f"\nConcurrent throughput test ({CONCURRENT_TOTAL} requests, {CONCURRENCY} concurrent)...")
    concurrent_payloads = (samples["url_only"] * ((CONCURRENT_TOTAL // len(samples["url_only"])) + 1))[:CONCURRENT_TOTAL]

    resource_samples = []
    stop_sampling = False

    def sample_resources():
        try:
            import psutil
        except ImportError:
            return
        if not UVICORN_PID:
            return
        try:
            proc = psutil.Process(int(UVICORN_PID))
        except Exception:
            print(f"  Could not attach psutil to PID {UVICORN_PID} -- skipping resource sampling.")
            return
        while not stop_sampling:
            try:
                resource_samples.append({
                    "cpu_percent": proc.cpu_percent(interval=0.2),
                    "memory_mb": proc.memory_info().rss / (1024 * 1024),
                })
            except Exception:
                break

    resource_thread = None
    if UVICORN_PID:
        import threading
        resource_thread = threading.Thread(target=sample_resources, daemon=True)
        resource_thread.start()

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = [executor.submit(_post_analyze, p) for p in concurrent_payloads]
        concurrent_statuses = [f.result()[1] for f in concurrent.futures.as_completed(futures)]
    conc_elapsed = time.perf_counter() - start
    stop_sampling = True
    if resource_thread:
        resource_thread.join(timeout=1)

    conc_throughput = len(concurrent_payloads) / conc_elapsed if conc_elapsed > 0 else 0
    conc_errors = sum(1 for s in concurrent_statuses if s != 200)

    # --- 4. Lighter endpoints ---
    print("\nTiming GET /threats and GET /stats...")
    threats_latencies = []
    stats_latencies = []
    for _ in range(20):
        try:
            e, s = _get("/threats?limit=20")
            if s == 200:
                threats_latencies.append(e)
        except Exception:
            pass
        try:
            e, s = _get("/stats")
            if s == 200:
                stats_latencies.append(e)
        except Exception:
            pass

    # --- Report ---
    lines = ["ETIS -- API Performance Benchmark Report", f"Base URL: {BASE_URL}\n"]
    lines.append("=== /analyze sequential latency, by input type ===")
    for category in ("url_only", "text_only", "both"):
        stats = summarize(latencies_by_category[category])
        errs = errors_by_category[category]
        if not stats:
            lines.append(f"{category}: no successful requests")
            continue
        line = (
            f"{category:<10} N={stats['n']:<4} errors={errs:<3} "
            f"mean={stats['mean_ms']:.1f}ms  median={stats['median_ms']:.1f}ms  "
            f"p95={stats['p95_ms']:.1f}ms  p99={stats['p99_ms']:.1f}ms  "
            f"min={stats['min_ms']:.1f}ms  max={stats['max_ms']:.1f}ms"
        )
        print(line)
        lines.append(line)

    lines.append("\n=== Throughput ===")
    seq_line = f"Sequential: {seq_throughput:.2f} req/s ({len(samples['url_only'])} requests in {seq_elapsed:.1f}s)"
    conc_line = (
        f"Concurrent ({CONCURRENCY} workers): {conc_throughput:.2f} req/s "
        f"({len(concurrent_payloads)} requests in {conc_elapsed:.1f}s, {conc_errors} non-200 responses)"
    )
    print(seq_line)
    print(conc_line)
    lines.append(seq_line)
    lines.append(conc_line)

    lines.append("\n=== Lighter endpoints ===")
    for name, lat in [("GET /threats", threats_latencies), ("GET /stats", stats_latencies)]:
        stats = summarize(lat)
        if stats:
            line = f"{name}: mean={stats['mean_ms']:.1f}ms  median={stats['median_ms']:.1f}ms  p95={stats['p95_ms']:.1f}ms"
        else:
            line = f"{name}: no successful requests"
        print(line)
        lines.append(line)

    lines.append("\n=== Resource utilization (server process) ===")
    if resource_samples:
        cpu_vals = [s["cpu_percent"] for s in resource_samples]
        mem_vals = [s["memory_mb"] for s in resource_samples]
        line = (
            f"During concurrent burst: CPU mean={statistics.mean(cpu_vals):.1f}%  "
            f"max={max(cpu_vals):.1f}%  |  Memory mean={statistics.mean(mem_vals):.1f}MB  "
            f"max={max(mem_vals):.1f}MB  ({len(resource_samples)} samples)"
        )
        print(line)
        lines.append(line)
    else:
        note = (
            "Not measured -- set ETIS_UVICORN_PID env var to the uvicorn process's "
            "PID before running, and ensure psutil is installed (pip install psutil), "
            "to enable this. Not a blocker for Chapter 5.7 -- response time and "
            "throughput are the primary numbers needed there."
        )
        print(note)
        lines.append(note)

    lines.append("\n" + "=" * 70)
    lines.append("CAVEAT: measured against localhost on local development hardware")
    lines.append("(see Chapter 5.1 hardware spec) -- no network latency, no hosted-")
    lines.append("deployment contention. Do not present as representative of the")
    lines.append("Railway/Render hosted demo without a separate benchmark against it.")

    with open(OUTPUT_DIR / "api_benchmark_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved report to {OUTPUT_DIR / 'api_benchmark_report.txt'}")

    # --- Latency chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = [c for c in ("url_only", "text_only", "both") if latencies_by_category.get(c)]
    box_data = [[v * 1000 for v in latencies_by_category[c]] for c in categories]
    ax.boxplot(box_data, tick_labels=[c.replace("_", " ") for c in categories])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("POST /analyze Latency by Input Type")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "api_benchmark_latency.png", dpi=150)
    plt.close(fig)
    print(f"Saved latency chart to {OUTPUT_DIR / 'api_benchmark_latency.png'}")

    # --- Throughput chart ---
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["Sequential", f"Concurrent ({CONCURRENCY} workers)"], [seq_throughput, conc_throughput], color=["#4f5fe0", "#d6453c"])
    ax.bar_label(bars, fmt="%.2f req/s")
    ax.set_ylabel("Requests per second")
    ax.set_title("Throughput: Sequential vs. Concurrent")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "api_benchmark_throughput.png", dpi=150)
    plt.close(fig)
    print(f"Saved throughput chart to {OUTPUT_DIR / 'api_benchmark_throughput.png'}")


if __name__ == "__main__":
    main()
