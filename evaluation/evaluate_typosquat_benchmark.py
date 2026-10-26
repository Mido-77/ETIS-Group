"""
evaluate_typosquat_benchmark.py

Runs the benchmark built by data/scripts/build_typosquat_benchmark.py through
the real scorer (calculate_final_risk_score) and reports how well ETIS
detects look-alikes of the 20 verified Egyptian brand domains -- the claim
behind RQ2/RQ3 that the main dataset cannot evidence.

How to read it
  - Report PER TECHNIQUE, never one pooled number. Some techniques (embedding,
    homoglyph_unicode, userinfo_trick, small character edits) are what the
    rule engine was designed for, so detecting them is partly circular. The
    others (tld_swap, combo_prefix/suffix, alias_variant, path_embedding,
    brand_as_subdomain, keyboard_adjacent) are not implemented rules: they show
    the real standing of the system, including the learned URL model, which
    never saw an Egyptian-brand look-alike in training.
  - 'rule' and 'url_model' columns show which engine alone would have flagged
    a row (tier >= MEDIUM), i.e. how much each contributes and whether they
    overlap.
  - 'genuine' rows are CONSTRUCTED plausible URLs on the brands' own domains
    (not crawled). They check that nothing flags a brand's own site.
  - The benchmark is for evaluation only. Do NOT add it to training data
    without building a separate, disjoint test set first.

Usage (project root, venv active):
    python evaluation/evaluate_typosquat_benchmark.py
"""

import math
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmarks" / "egypt_brand_typosquats.csv"
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "typosquat_benchmark_results.csv"
REPORT_PATH = PROJECT_ROOT / "evaluation" / "typosquat_benchmark_report.txt"
ENGINES = ["rule_based", "blacklist", "url_model"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def pct(k: int, n: int) -> str:
    if n == 0:
        return "   n/a"
    return f"{100 * k / n:5.1f}%"


def pct_ci(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]  ({k}/{n})"


def score_all() -> pd.DataFrame:
    from app.core.database import SessionLocal, engine
    from app.engines.risk_scoring import calculate_final_risk_score

    print(f"Database: {engine.url.render_as_string(hide_password=True)} (only the blacklist uses it; benchmark URLs are not in it)")
    bench = pd.read_csv(BENCHMARK_PATH)
    db = SessionLocal()
    records, start = [], time.time()
    for n, row in enumerate(bench.itertuples(index=False), start=1):
        record = row._asdict()
        try:
            result = calculate_final_risk_score(url=row.url, db=db)
            typosquat = (result["engine_details"]["rule_based"]["engine_results"].get("domain_typosquatting") or {})
            record.update({
                "verdict": result["verdict"], "final_score": result["final_score"],
                "decided_by": result["decided_by"] or "", "allowlisted": int(result["allowlisted"]),
                "rule_typosquat_technique": typosquat.get("technique") or "",
                "url_model_probability": (result["engine_details"].get("url_model") or {}).get("probability"),
                **{f"tier_{e}": result["engine_tiers"].get(e, "") for e in ENGINES},
            })
        except Exception as exc:
            db.rollback()
            record["error"] = f"{type(exc).__name__}: {exc}"[:200]
        records.append(record)
        if n % 300 == 0:
            print(f"  ...{n}/{len(bench)} ({time.time() - start:.0f}s)")
    db.close()
    df = pd.DataFrame(records)
    df.to_csv(RESULTS_PATH, index=False)
    return df


def flagged(df: pd.DataFrame, engine: str | None = None, level: str = "MEDIUM+") -> pd.Series:
    column = "verdict" if engine is None else f"tier_{engine}"
    values = df[column].fillna("")
    return values.isin(["MEDIUM", "HIGH"]) if level == "MEDIUM+" else values == "HIGH"


def main() -> None:
    df = score_all()
    errors = int(df["error"].notna().sum()) if "error" in df.columns else 0
    if errors:
        df = df[df["error"].isna()]
    typo, genuine = df[df["kind"] == "typosquat"], df[df["kind"] == "genuine"]

    out = ["ETIS Egyptian-brand typosquat benchmark (final system)",
           f"typosquat URLs: {len(typo):,}   genuine own-domain URLs (constructed): {len(genuine):,}   errors excluded: {errors}"]

    out += ["", "== Genuine controls: must NOT be flagged ==",
            f"flagged >= MEDIUM: {pct_ci(int(flagged(genuine).sum()), len(genuine))}",
            f"flagged  = HIGH  : {pct_ci(int(flagged(genuine, level='HIGH').sum()), len(genuine))}",
            f"allowlisted      : {pct_ci(int(genuine['allowlisted'].sum()), len(genuine))}"]
    for r in genuine[flagged(genuine)].head(10).itertuples():
        out.append(f"  FLAGGED {r.verdict:<6} by {r.decided_by:<10} {r.url}")

    out += ["", "== Typosquats: overall ==",
            f"flagged >= MEDIUM: {pct_ci(int(flagged(typo).sum()), len(typo))}",
            f"flagged  = HIGH  : {pct_ci(int(flagged(typo, level='HIGH').sum()), len(typo))}"]

    out += ["", "== Typosquats by technique  (share flagged >= MEDIUM; 'rule' / 'url_model' = that engine alone) ==",
            f"{'technique':<20}{'n':>5}{'system':>9}{'HIGH':>9}{'rule':>9}{'url_model':>11}{'either':>9}  note"]
    designed_for = {"embedding", "homoglyph_unicode", "userinfo_trick", "omission", "repetition", "transposition", "digit_lookalike"}
    for technique in sorted(typo["technique"].unique()):
        part = typo[typo["technique"] == technique]
        n = len(part)
        rule, model = flagged(part, "rule_based"), flagged(part, "url_model")
        out.append(f"{technique:<20}{n:>5}{pct(int(flagged(part).sum()), n):>9}{pct(int(flagged(part, level='HIGH').sum()), n):>9}"
                   f"{pct(int(rule.sum()), n):>9}{pct(int(model.sum()), n):>11}{pct(int((rule | model).sum()), n):>9}"
                   f"  {'designed-for (partly circular)' if technique in designed_for else 'NOT a built-in rule'}")

    out += ["", "== Typosquats by brand (share flagged >= MEDIUM) =="]
    for brand in sorted(typo["brand"].unique()):
        part = typo[typo["brand"] == brand]
        out.append(f"  {brand:<40}{pct(int(flagged(part).sum()), len(part)):>8}  (n={len(part)})")

    out += ["", "== Misses (verdict LOW): up to 3 per technique =="]
    for technique in sorted(typo["technique"].unique()):
        misses = typo[(typo["technique"] == technique) & (typo["verdict"] == "LOW")]
        if len(misses):
            out.append(f"  {technique} ({len(misses)} missed)")
            for url in misses["url"].head(3):
                out.append(f"      {url}")

    out += ["", "Reading guide: report the per-technique table. 'NOT a built-in rule' rows are the real",
            "measure; 'designed-for' rows partly test whether the rule engine does what it was built to do.",
            "The genuine rows are constructed URLs, not crawled."]

    report = "\n".join(out)
    print("\n" + report)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved {REPORT_PATH} and {RESULTS_PATH}")


if __name__ == "__main__":
    main()
