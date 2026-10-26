"""
evaluate_heldout_typosquat_benchmark.py   (v2: relabelled technique classes + --from-results)

Scores the HELD-OUT Egyptian-brand typosquat benchmark
(data/benchmarks/egypt_brand_typosquats_heldout.csv, built by
data/scripts/build_heldout_typosquat_benchmark.py) through the real scorer.

Two modes
  default         score every URL (needs DATABASE_URL -> etis_eval), write
                  evaluation/heldout_typosquat_results.csv and
                  evaluation/heldout_typosquat_report.txt
  --from-results  NO scoring: re-read the saved results CSV and write a
                  re-labelled report to evaluation/heldout_typosquat_report_relabelled.txt.
                  The scores are exactly those of the committed run; only the
                  technique-class labels and groupings differ (see CHANGE LOG).

CHANGE LOG vs the first version (made AFTER seeing the first report; labels only)
  The first report labelled substitution_random, vowel_swap, bitsquat,
  ascii_multichar, insertion_random, keyboard_adjacent and plural_suffix
  "no dedicated rule" and quoted 75.6% for them as a generalisation measure.
  That was wrong: the Levenshtein rule, which predates the development set,
  catches them (for brand domains of 9+ characters). They are now their own
  class. The "no dedicated rule" class is now only: alias_variant (these
  aliases are NOT in the registry), combo_concat, typo_plus_combo, glued_www,
  fullname_variant, idn_homoglyph_unmapped. No scores, thresholds or rules changed.

Rules this script enforces or reports
  * FREEZE (scoring mode). data/benchmarks/heldout_manifest.json holds sha256
    fingerprints of the detector, registry, scoring files and ML artifacts
    taken when the set was generated. If any differ now, or the CSV was edited,
    the script refuses to run (--allow-changed-files = DEVELOPMENT run, report
    stamped "NOT HELD-OUT").
  * Scoring mode runs against the evaluation database (etis_eval; blacklist
    seeded from the TRAIN split only), refused otherwise unless --allow-any-db.
  * Per technique / per brand / per stratum, never one pooled headline. Pooled
    figures depend on how many easy techniques the benchmark happens to contain.
    Wilson 95% intervals assume independent rows; rows of one technique share 20
    brands, so the intervals are OPTIMISTIC (treat differences < ~5 points as noise).
  * Rows marked `ambiguous` (may be a real, unrelated organisation) are
    excluded from the headline and reported separately.
  * Do NOT tune rules on this report. If a fix follows, say in the thesis that
    this set became a development set.

Usage (project root, venv active):
    python evaluation/evaluate_heldout_typosquat_benchmark.py                 (DATABASE_URL -> etis_eval, same window)
    python evaluation/evaluate_heldout_typosquat_benchmark.py --from-results
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmarks" / "egypt_brand_typosquats_heldout.csv"
MANIFEST_PATH = PROJECT_ROOT / "data" / "benchmarks" / "heldout_manifest.json"
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "heldout_typosquat_results.csv"
REPORT_PATH = PROJECT_ROOT / "evaluation" / "heldout_typosquat_report.txt"
RELABELLED_REPORT_PATH = PROJECT_ROOT / "evaluation" / "heldout_typosquat_report_relabelled.txt"
ENGINES = ["rule_based", "blacklist", "url_model"]
THIN_N = 20

FREEZE_GLOBS = ["ml/models/*.joblib"]

# ---- technique classes, as of the frozen detector --------------------------------------------
CLASS_BEFORE_DEV = "A. rule built BEFORE the dev set (partly circular)"
CLASS_EDIT_DISTANCE = "B. pre-existing edit-distance rule (brand domains >= 9 chars only)"
CLASS_AFTER_DEV = "C. rule built FROM dev misses (new vocabulary here)"
CLASS_NONE = "D. no dedicated rule (generalisation)"

TECHNIQUE_CLASS = {
    # A: homoglyph map, userinfo, domain embedding, and the first Levenshtein-style edits
    "embedding_hyphen_suffix": CLASS_BEFORE_DEV, "embedding_subdomain_chain": CLASS_BEFORE_DEV,
    "userinfo_credentials": CLASS_BEFORE_DEV, "idn_homoglyph_mapped": CLASS_BEFORE_DEV,
    "omission": CLASS_BEFORE_DEV, "repetition": CLASS_BEFORE_DEV, "transposition": CLASS_BEFORE_DEV,
    "digit_lookalike": CLASS_BEFORE_DEV,
    # B: caught by the Levenshtein rule (similarity >= 0.75 against brand domains of 9+ characters)
    "keyboard_adjacent": CLASS_EDIT_DISTANCE, "substitution_random": CLASS_EDIT_DISTANCE,
    "vowel_swap": CLASS_EDIT_DISTANCE, "bitsquat": CLASS_EDIT_DISTANCE,
    "ascii_multichar": CLASS_EDIT_DISTANCE, "insertion_random": CLASS_EDIT_DISTANCE,
    "plural_suffix": CLASS_EDIT_DISTANCE,
    # C: brand-token / hyphen-split / TLD-swap / brand-in-path rules, designed from the dev miss list
    "tld_swap": CLASS_AFTER_DEV, "combo_prefix": CLASS_AFTER_DEV, "combo_suffix": CLASS_AFTER_DEV,
    "combo_two_words": CLASS_AFTER_DEV, "brand_as_subdomain": CLASS_AFTER_DEV, "path_embedding": CLASS_AFTER_DEV,
    "query_embedding": CLASS_AFTER_DEV, "fragment_embedding": CLASS_AFTER_DEV, "dot_to_hyphen": CLASS_AFTER_DEV,
    "hyphen_insertion": CLASS_AFTER_DEV,
    # D: nothing in the detector targets these (alias strings here are NOT in the registry)
    "alias_variant": CLASS_NONE, "combo_concat": CLASS_NONE, "typo_plus_combo": CLASS_NONE,
    "glued_www": CLASS_NONE, "fullname_variant": CLASS_NONE, "idn_homoglyph_unmapped": CLASS_NONE,
}
CLASS_ORDER = [CLASS_BEFORE_DEV, CLASS_EDIT_DISTANCE, CLASS_AFTER_DEV, CLASS_NONE]


def technique_class(technique: str) -> str:
    return TECHNIQUE_CLASS.get(technique, "? unclassified (add to TECHNIQUE_CLASS)")


def normalised_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def current_fingerprint(recorded: dict) -> dict:
    files = {}
    for rel in recorded:
        p = PROJECT_ROOT / rel
        files[rel] = normalised_sha256(p) if p.exists() else None
    for pattern in FREEZE_GLOBS:
        for p in sorted(PROJECT_ROOT.glob(pattern)):
            files[str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")] = normalised_sha256(p)
    return files


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        sys.exit(f"{MANIFEST_PATH} not found: run build_heldout_typosquat_benchmark.py first.")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def check_freeze(manifest: dict, allow_changed: bool) -> list[str]:
    problems = []
    if normalised_sha256(BENCHMARK_PATH) != manifest["csv_sha256"]:
        problems.append("benchmark CSV differs from the one the manifest was written for")
    now, then = current_fingerprint(manifest["freeze_files"]), manifest["freeze_files"]
    for rel in sorted(set(now) | set(then)):
        if now.get(rel) != then.get(rel):
            problems.append(f"changed since generation: {rel}")
    if problems and not allow_changed:
        print("FREEZE CHECK FAILED -- refusing to run:")
        for p in problems:
            print("  -", p)
        sys.exit("Restore the files (or regenerate the benchmark BEFORE looking at any result). "
                 "--allow-changed-files makes this a development run.")
    return problems


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def pct(k: int, n: int) -> str:
    return "  n/a" if n == 0 else f"{100 * k / n:5.1f}%"


def pct_ci(k: int, n: int) -> str:
    if n == 0:
        return "n/a (n=0)"
    lo, hi = wilson(k, n)
    thin = "  [thin]" if n < THIN_N else ""
    return f"{100 * k / n:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]  ({k}/{n}){thin}"


def score_all(args) -> pd.DataFrame:
    from app.core.database import SessionLocal, engine
    from app.engines.risk_scoring import calculate_final_risk_score

    database = engine.url.database or ""
    print(f"Database: {engine.url.render_as_string(hide_password=True)}")
    if not database.endswith("eval") and not args.allow_any_db:
        sys.exit("Refusing to run: DATABASE_URL does not point at the evaluation database (etis_eval). "
                 "Set it in THIS PowerShell window, or pass --allow-any-db.")

    bench = pd.read_csv(BENCHMARK_PATH, dtype=str, keep_default_na=False)
    db = SessionLocal()
    records, start, consecutive_errors = [], time.time(), 0
    for n, row in enumerate(bench.to_dict("records"), start=1):
        record = dict(row)
        try:
            result = calculate_final_risk_score(url=row["url"], db=db)
            typosquat = (result["engine_details"]["rule_based"]["engine_results"].get("domain_typosquatting") or {})
            record.update({
                "verdict": result["verdict"], "final_score": result["final_score"],
                "decided_by": result["decided_by"] or "", "allowlisted": int(result["allowlisted"]),
                "rule_typosquat_technique": typosquat.get("technique") or "",
                "rule_typosquat_brand_domain": typosquat.get("matched_brand_domain") or "",
                "url_model_probability": (result["engine_details"].get("url_model") or {}).get("probability"),
                **{f"tier_{e}": result["engine_tiers"].get(e, "") for e in ENGINES},
            })
            consecutive_errors = 0
        except Exception as exc:
            db.rollback()
            record["error"] = f"{type(exc).__name__}: {exc}"[:200]
            consecutive_errors += 1
            if consecutive_errors >= 25:
                sys.exit("25 consecutive scoring failures -- check DATABASE_URL (real password, no angle brackets) "
                         f"and that the backend venv is active. Last error: {record['error']}")
        records.append(record)
        if n % 300 == 0:
            print(f"  ...{n}/{len(bench)} ({time.time() - start:.0f}s)")
    db.close()
    df = pd.DataFrame(records)
    df.to_csv(RESULTS_PATH, index=False)
    return df


def normalise_results(df: pd.DataFrame) -> pd.DataFrame:
    """Same dtypes whether the frame came from scoring or from the saved CSV."""
    df = df.copy()
    if "error" not in df.columns:
        df["error"] = ""
    df["error"] = df["error"].fillna("").astype(str)
    for column in ["verdict", "decided_by", "rule_typosquat_brand_domain", "rule_typosquat_technique",
                   *[f"tier_{e}" for e in ENGINES]]:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].fillna("").astype(str)
    df["allowlisted"] = pd.to_numeric(df.get("allowlisted", 0), errors="coerce").fillna(0).astype(int)
    return df


def flagged(df: pd.DataFrame, engine: str | None = None, level: str = "MEDIUM+") -> pd.Series:
    column = "verdict" if engine is None else f"tier_{engine}"
    values = df[column].fillna("")
    return values.isin(["MEDIUM", "HIGH"]) if level == "MEDIUM+" else values == "HIGH"


def strata_table(part: pd.DataFrame, column: str, title: str) -> list[str]:
    out = ["", f"-- {title} --",
           f"{column:<30}{'n':>5}  {'system >= MEDIUM  [95% CI]':<40}{'HIGH':>8}{'rule':>8}{'url_model':>11}"]
    for value in sorted(part[column].unique()):
        sub = part[part[column] == value]
        k = int(flagged(sub).sum())
        lo, hi = wilson(k, len(sub))
        out.append(f"{value:<30}{len(sub):>5}  {100 * k / len(sub):5.1f}% [{100 * lo:5.1f}, {100 * hi:5.1f}]".ljust(77)
                   + f"{pct(int(flagged(sub, level='HIGH').sum()), len(sub)):>8}"
                   + f"{pct(int(flagged(sub, 'rule_based').sum()), len(sub)):>8}"
                   + f"{pct(int(flagged(sub, 'url_model').sum()), len(sub)):>11}"
                   + ("  [thin]" if len(sub) < THIN_N else ""))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-results", action="store_true", help="no scoring: recompute the report from the saved results CSV")
    parser.add_argument("--allow-changed-files", action="store_true", help="DEVELOPMENT run: skip the freeze refusal")
    parser.add_argument("--allow-any-db", action="store_true", help="testing only")
    args = parser.parse_args()

    manifest = load_manifest()
    problems: list[str] = []
    if args.from_results:
        if not RESULTS_PATH.exists():
            sys.exit(f"{RESULTS_PATH} not found.")
        df = pd.read_csv(RESULTS_PATH, dtype=str, keep_default_na=False)
        report_path = RELABELLED_REPORT_PATH
    else:
        problems = check_freeze(manifest, args.allow_changed_files)
        df = score_all(args)
        report_path = REPORT_PATH

    df = normalise_results(df)
    errors = int((df["error"] != "").sum())
    df = df[df["error"] == ""]

    typo_all = df[df["kind"] == "typosquat"]
    ambiguous = typo_all[typo_all["ambiguity"] != ""]
    typo = typo_all[typo_all["ambiguity"] == ""].copy()
    typo["technique_class"] = typo["technique"].map(technique_class)
    genuine = df[df["kind"] == "genuine"]
    controls = df[df["kind"] == "control"]

    out = ["ETIS held-out Egyptian-brand typosquat benchmark (final system)"]
    if args.from_results:
        out += ["*** RELABELLED REPORT: recomputed from the saved results CSV, NO rescoring. Scores are identical to the",
                "    committed raw run; only technique-class labels/groupings were corrected after the first report was read",
                "    (see CHANGE LOG in the script header). ***"]
    if problems:
        out += ["*** NOT HELD-OUT: files changed since generation (development run) ***", *[f"    {p}" for p in problems]]
    elif not args.from_results:
        out.append("Freeze check: PASSED (detector, registry, scoring files and ML artifacts unchanged since generation)")
    git = manifest.get("git", {})
    out += [f"Generated {manifest.get('generated_utc')} with seed {manifest.get('seed')}; git HEAD at generation: {git.get('head')}",
            f"rows scored: typosquat {len(typo_all):,} (headline {len(typo):,}, ambiguous {len(ambiguous):,}), "
            f"genuine {len(genuine):,}, controls {len(controls):,}; errors excluded: {errors}",
            f"candidates dropped at generation: {manifest.get('dropped')}",
            "CIs assume independent rows; rows share brands, so they are optimistic (treat < ~5 points as noise)."]

    # ---- genuine own-domain controls -------------------------------------------------
    out += ["", "== Genuine own-domain URLs (constructed; mostly test the allowlist) ==",
            f"flagged >= MEDIUM: {pct_ci(int(flagged(genuine).sum()), len(genuine))}",
            f"flagged  = HIGH  : {pct_ci(int(flagged(genuine, level='HIGH').sum()), len(genuine))}",
            f"allowlisted      : {pct_ci(int(genuine['allowlisted'].sum()), len(genuine))}"]
    for technique in sorted(genuine["technique"].unique()):
        sub = genuine[genuine["technique"] == technique]
        out.append(f"  {technique:<20} flagged {pct_ci(int(flagged(sub).sum()), len(sub))}")
    for r in genuine[flagged(genuine)].head(15).itertuples():
        out.append(f"  FLAGGED {r.verdict:<6} by {r.decided_by:<10} {r.url}")

    # ---- real other organisations ------------------------------------------------------
    out += ["", "== Controls: real OTHER organisations (written from memory, not verified; separate, never pooled) =="]
    for group in sorted(controls["technique"].unique()):
        sub = controls[controls["technique"] == group]
        out.append(f"{group}: flagged >= MEDIUM {pct_ci(int(flagged(sub).sum()), len(sub))};  HIGH {pct_ci(int(flagged(sub, level='HIGH').sum()), len(sub))}")
        for r in sub[flagged(sub)].itertuples():
            out.append(f"    FLAGGED {r.verdict:<6} by {r.decided_by:<10} {r.url}   (host in dataset: {r.host_in_dataset})")
    out.append("  Note: 18-24 domains x 2 URLs each, so the effective n is the number of DOMAINS, not URLs.")
    out.append("  Note: host_in_dataset=1 only says the host occurs in the dataset (label not checked); popular hosts may be in Tranco training rows.")

    # ---- typosquats -------------------------------------------------------------------
    out += ["", "== Typosquats (ambiguous rows excluded) ==",
            "A pooled number depends on how many easy techniques this benchmark contains; quote the class and technique lines instead.",
            f"pooled >= MEDIUM: {pct_ci(int(flagged(typo).sum()), len(typo))}   (do NOT quote as 'the' result)",
            f"pooled  = HIGH  : {pct_ci(int(flagged(typo, level='HIGH').sum()), len(typo))}"]

    out += ["", "== By technique class (pooled rows; 'macro' = unweighted mean of the per-technique rates, techniques with n >= 20) ==",
            f"{'class':<72}{'techs':>6}{'n':>6}  {'pooled >= MEDIUM [95% CI]':<28}{'macro':>7}"]
    for cls in CLASS_ORDER:
        sub = typo[typo["technique_class"] == cls]
        if sub.empty:
            continue
        per_tech = [float(flagged(g).mean()) for _, g in sub.groupby("technique") if len(g) >= THIN_N]
        macro = f"{100 * sum(per_tech) / len(per_tech):5.1f}%" if per_tech else "  n/a"
        k = int(flagged(sub).sum())
        lo, hi = wilson(k, len(sub))
        out.append(f"{cls:<72}{sub['technique'].nunique():>6}{len(sub):>6}  "
                   f"{100 * k / len(sub):5.1f}% [{100 * lo:5.1f}, {100 * hi:5.1f}]".ljust(112) + f"{macro:>7}")
    unclassified = typo[typo["technique_class"].str.startswith("?")]
    if len(unclassified):
        out.append(f"  WARNING: unclassified techniques: {sorted(unclassified['technique'].unique())}")
    out.append("  Class D is the closest thing to a generalisation measure; class A is partly circular; class B only works for brand")
    out.append("  domains of 9+ characters (hence the short-brand gap); class C was designed from the dev misses (here with new vocabulary).")

    out += strata_table(typo, "brand_size", "short brands (label <= 4: aaib, noon, nbe, qnb, te) vs normal")
    out += strata_table(typo, "family", "technique name also in the development set (repeat) vs new")

    out += ["", "== By technique (share flagged >= MEDIUM; 'rule' / 'url_model' = that engine alone) ==",
            f"{'technique':<26}{'n':>4}  {'system [95% CI]':<22}{'HIGH':>7}{'rule':>7}{'model':>7}{'either':>7}  class"]
    for cls in CLASS_ORDER:
        for technique in sorted(typo[typo["technique_class"] == cls]["technique"].unique()):
            sub = typo[typo["technique"] == technique]
            n = len(sub)
            k = int(flagged(sub).sum())
            lo, hi = wilson(k, n)
            rule, model = flagged(sub, "rule_based"), flagged(sub, "url_model")
            out.append(f"{technique:<26}{n:>4}  {100 * k / n:5.1f}% [{100 * lo:4.0f},{100 * hi:4.0f}]".ljust(54)
                       + f"{pct(int(flagged(sub, level='HIGH').sum()), n):>7}{pct(int(rule.sum()), n):>7}"
                       + f"{pct(int(model.sum()), n):>7}{pct(int((rule | model).sum()), n):>7}  {cls[:1]}"
                       + ("  [thin]" if n < THIN_N else ""))

    out += ["", "== By brand (share flagged >= MEDIUM; per brand domain; short brands are a known ceiling) =="]
    for domain in sorted(typo["brand_domain"].unique()):
        sub = typo[typo["brand_domain"] == domain]
        k = int(flagged(sub).sum())
        lo, hi = wilson(k, len(sub))
        out.append(f"  {domain:<18}{sub['brand_size'].iloc[0]:<8}{100 * k / len(sub):5.1f}% [{100 * lo:5.1f}, {100 * hi:5.1f}]  (n={len(sub)})")

    out += strata_table(typo[typo["tld_group"] != "n/a"], "tld_group", "by TLD of the look-alike host")
    kw = typo[typo["keyword_vocab"].isin(["in_detector_vocab", "novel"])]
    out += strata_table(kw, "keyword_vocab", "keyword vocabulary (combo/embedding/typo+combo techniques only)")

    flagged_by_rule = typo[flagged(typo, "rule_based")]
    if len(flagged_by_rule):
        right = int((flagged_by_rule["rule_typosquat_brand_domain"] == flagged_by_rule["brand_domain"]).sum())
        out += ["", "== Attribution: when the rule engine flags a row, does it name the intended brand? ==",
                f"  {pct_ci(right, len(flagged_by_rule))}   (rows with no rule typosquat match, e.g. URL-pattern flags only, count as wrong)"]

    out += ["", "== Ambiguous rows (may be a real, unrelated organisation; excluded above) =="]
    for reason in sorted(ambiguous["ambiguity"].unique()):
        sub = ambiguous[ambiguous["ambiguity"] == reason]
        out.append(f"  {reason:<30} flagged >= MEDIUM {pct_ci(int(flagged(sub).sum()), len(sub))}")

    out += ["", "== Misses (verdict LOW): up to 5 per technique =="]
    for technique in sorted(typo["technique"].unique()):
        misses = typo[(typo["technique"] == technique) & (typo["verdict"] == "LOW")]
        if len(misses):
            out.append(f"  {technique} ({len(misses)} missed of {int((typo['technique'] == technique).sum())})")
            out += [f"      {u}" for u in misses["url"].head(5)]

    out += ["", "Reading guide",
            " - Report the class table and the per-technique table; never quote one pooled number as 'the' result.",
            " - 'repeat' rows reuse a technique name from the development set but with new keywords, aliases and TLDs;",
            "   'new' rows are techniques the development set never contained.",
            " - Genuine rows are constructed and mostly exercise the allowlist; the controls are the false-positive check",
            "   on real sites, and are small and written from memory.",
            " - Do NOT tune on this report. If the rules change after reading it, this set becomes a development set."]

    report = "\n".join(out)
    print("\n" + report)
    report_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved {report_path}" + ("" if args.from_results else f" and {RESULTS_PATH}"))


if __name__ == "__main__":
    main()
