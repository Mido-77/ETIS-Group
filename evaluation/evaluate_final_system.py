"""
evaluate_final_system.py

End-to-end evaluation of the CURRENT ETIS scorer (app/engines/risk_scoring.py,
decision tiers) on the TEST split. This is the headline-numbers script for
Chapter 5: every row is submitted the way a user would submit it (URL rows as
a URL only, text/email rows as text only) through calculate_final_risk_score.

Honesty rules baked in
  * Test split only. Nothing here is used to choose a threshold: every tier
    threshold was fixed on VALIDATION data before this script exists.
  * Synthetic rows (source synthetic_egyptian_brands) are excluded; they are
    reported separately, never in the test headline.
  * Run against the evaluation database (training-split-only blacklist),
    never the demo database. The script refuses otherwise.
  * Every rate carries a Wilson 95% confidence interval.
  * Results are broken down by host group, path / no path, source, language
    and by which engine decided the verdict. Report those, not one pooled
    number. Shared-platform URLs are listed as their own group.
  * What it cannot measure: the blacklist's value on already-known threats
    (the eval database only knows training URLs by design).

Phase 1  scoring (slow, resumable)      -> evaluation/final_system_scores[_TAG].csv
Phase 2  analysis (instant, from CSV)   -> evaluation/final_system_report[_TAG].txt
                                           evaluation/final_system_errors[_TAG].csv

Usage (project root, venv active):
    $env:DATABASE_URL = "postgresql://postgres:PASSWORD@localhost:5432/etis_eval"
    python evaluation/evaluate_final_system.py                    # score + analyse
    python evaluation/evaluate_final_system.py --analyze-only
    python evaluation/evaluate_final_system.py --rescore
    python evaluation/evaluate_final_system.py --only url --tag after
    python evaluation/evaluate_final_system.py --compare before after

Before/after workflow for a code change (e.g. the rule-engine commit):
    1. run with --only url --tag after        (current code)
    2. check out the old versions of the changed files, run --only url --tag before,
       restore the files
    3. python evaluation/evaluate_final_system.py --compare before after
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"
GROUPS_PATH = PROJECT_ROOT / "data" / "processed" / "split_groups.csv"
META_PATH = PROJECT_ROOT / "data" / "processed" / "real_messages_meta.csv"  # written by ingest_real_messages.py
OUT_DIR = PROJECT_ROOT / "evaluation"

SYNTHETIC_SOURCE = "synthetic_egyptian_brands"
REAL_SOURCE = "real_collected_messages"
MIN_QUOTABLE_ROWS = 20  # below this a per-category rate is shown but flagged as too thin to quote
ENGINES = ("rule_based", "blacklist", "url_model", "text_model")
FIELDS = ["id", "kind", "content_type", "label", "language", "group_type", "source",
          "has_path", "is_eg", "verdict", "final_score", "decided_by",
          "tier_rule_based", "tier_blacklist", "tier_url_model", "tier_text_model",
          "allowlisted", "error"]
TIER_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def _paths(tag: str) -> tuple[Path, Path, Path]:
    suffix = f"_{tag}" if tag else ""
    return (OUT_DIR / f"final_system_scores{suffix}.csv",
            OUT_DIR / f"final_system_report{suffix}.txt",
            OUT_DIR / f"final_system_errors{suffix}.csv")


# ---------------------------------------------------------------------------
# Phase 1: scoring
# ---------------------------------------------------------------------------

def _url_features(url: str) -> tuple[int, int]:
    try:
        parts = urlsplit(url if "://" in url else "http://" + url)
        return int(len(parts.path) > 1), int((parts.hostname or "").endswith(".eg"))
    except ValueError:
        return 0, 0


def score_rows(scores_path: Path, rescore: bool, only: str | None) -> None:
    from app.core.database import SessionLocal, engine
    database_name = engine.url.database or ""
    if "eval" not in database_name:
        sys.exit(f"Refusing to run against database '{database_name}': it is not an evaluation database. "
                 "Set DATABASE_URL to etis_eval (training-split-only blacklist).")
    print(f"Target database: {engine.url.render_as_string(hide_password=True)}")
    if engine.url.password in ("YOUR_PASSWORD", "PASSWORD", "password_here"):
        sys.exit("DATABASE_URL still contains a placeholder password. Put your real PostgreSQL password in it "
                 "(URL rows need the database for the blacklist lookup; text rows do not, which is why a "
                 "placeholder makes every URL row fail while text rows succeed).")

    from app.engines.risk_scoring import calculate_final_risk_score

    df = pd.read_csv(DATASET_PATH)
    df = df[(df["split"] == "test") & (df["source"] != SYNTHETIC_SOURCE)]
    if only == "url":
        df = df[df["content_type"] == "url"]
    elif only == "text":
        df = df[df["content_type"] != "url"]
    if GROUPS_PATH.exists():
        df = df.merge(pd.read_csv(GROUPS_PATH)[["id", "group_type"]], on="id", how="left")
    else:
        df["group_type"] = ""

    if scores_path.exists() and rescore:
        scores_path.unlink()
    done: set = set()
    if scores_path.exists():
        previous = pd.read_csv(scores_path)
        good = previous[previous["error"].isna()]
        if len(good) < len(previous):
            print(f"{len(previous) - len(good):,} previously ERRORED rows will be retried (not counted as done).")
            good.reindex(columns=FIELDS).to_csv(scores_path, index=False, encoding="utf-8")
        done = set(good["id"])
    todo = df[~df["id"].isin(done)]
    print(f"{len(df):,} test rows selected; {len(done):,} already scored; scoring {len(todo):,} now.")

    new_file = not scores_path.exists()
    start = time.time()
    error_messages: list[str] = []
    db = SessionLocal()
    with open(scores_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for n, row in enumerate(todo.itertuples(index=False), start=1):
            is_url = row.content_type == "url"
            record = {"id": row.id, "kind": "url" if is_url else "text", "content_type": row.content_type,
                      "label": row.label, "language": row.language, "group_type": row.group_type,
                      "source": row.source}
            if is_url:
                record["has_path"], record["is_eg"] = _url_features(row.text)
            try:
                result = (calculate_final_risk_score(url=row.text, db=db) if is_url
                          else calculate_final_risk_score(text=row.text, db=db))
                tiers = result.get("engine_tiers", {})
                record.update({
                    "verdict": result["verdict"], "final_score": result["final_score"],
                    "decided_by": result.get("decided_by") or "",
                    "allowlisted": int(bool(result.get("allowlisted"))),
                    **{f"tier_{name}": tiers.get(name, "") for name in ENGINES},
                })
            except Exception as exc:  # one bad row must not kill a long run
                db.rollback()
                record["error"] = f"{type(exc).__name__}: {exc}"[:200]
                error_messages.append(record["error"])
            writer.writerow(record)
            if n == 25 and len(error_messages) == 25:
                handle.flush()
                sys.exit(f"The first 25 rows ALL failed, so the run was stopped. First error:\n  {error_messages[0]}\n"
                         "Fix that (database password? models missing?) and re-run; errored rows are retried.")
            if n % 1000 == 0:
                handle.flush()
                print(f"  ...{n:,}/{len(todo):,} rows ({time.time() - start:.0f}s)")
    db.close()
    print(f"Scoring done in {time.time() - start:.0f}s -> {scores_path}")


# ---------------------------------------------------------------------------
# Phase 2: analysis
# ---------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def rate(k: int, n: int) -> str:
    if n == 0:
        return "   n/a"
    lo, hi = wilson(k, n)
    return f"{100 * k / n:5.1f}% [{100 * lo:4.1f}, {100 * hi:5.1f}] ({k:,}/{n:,})"


def crosstab(out: list, title: str, d: pd.DataFrame) -> None:
    out += ["", f"-- {title} --", f"{'':<12}{'LOW':>8}{'MEDIUM':>8}{'HIGH':>8}   (row %)"]
    for label in ("phishing", "legitimate"):
        part = d[d["label"] == label]
        counts = [int((part["verdict"] == v).sum()) for v in ("LOW", "MEDIUM", "HIGH")]
        total = max(1, sum(counts))
        out.append(f"{label:<12}" + "".join(f"{100 * c / total:>7.1f}%" for c in counts) + f"   (n={sum(counts):,})")


def operating_points(out: list, title: str, d: pd.DataFrame) -> None:
    phishing, legit = d[d["label"] == "phishing"], d[d["label"] == "legitimate"]
    out += ["", f"-- {title}  (phishing n={len(phishing):,}, legitimate n={len(legit):,}) --"]
    for name, mask in (("flagged MEDIUM or HIGH", lambda x: x["verdict"] != "LOW"),
                       ("flagged HIGH only", lambda x: x["verdict"] == "HIGH")):
        tp, fp = int(mask(phishing).sum()), int(mask(legit).sum())
        precision = f"{100 * tp / (tp + fp):5.1f}%" if tp + fp else "  n/a"
        out.append(f"  {name:<24} recall {rate(tp, len(phishing))}")
        out.append(f"  {'':<24} FPR    {rate(fp, len(legit))}   precision {precision}")


def per_engine(out: list, d: pd.DataFrame) -> None:
    out += ["", "-- Each engine ALONE (tier >= MEDIUM among rows where the engine applied) --"]
    for engine in ENGINES:
        column = f"tier_{engine}"
        applied = d[d[column].fillna("") != ""]
        if applied.empty:
            continue
        phishing, legit = applied[applied["label"] == "phishing"], applied[applied["label"] == "legitimate"]
        tp = int((phishing[column] != "LOW").sum())
        fp = int((legit[column] != "LOW").sum())
        out.append(f"  {engine:<11} recall {rate(tp, len(phishing))}")
        out.append(f"  {'':<11} FPR    {rate(fp, len(legit))}")
    out += ["", "-- Which engine DECIDED the verdict (flagged rows only) --"]
    flagged = d[d["verdict"] != "LOW"]
    for label in ("phishing", "legitimate"):
        counts = flagged[flagged["label"] == label]["decided_by"].value_counts()
        out.append(f"  {label:<11} " + (", ".join(f"{k}={v:,}" for k, v in counts.items()) or "(none flagged)"))


def real_message_breakdown(out: list, texts: pd.DataFrame) -> None:
    """Real collected messages by category and channel (needs real_messages_meta.csv).

    Legitimate categories report the false-positive rate (flagged >= MEDIUM, and HIGH
    alone); phishing_received reports recall. Cells under MIN_QUOTABLE_ROWS are marked.
    """
    if not META_PATH.exists():
        return
    real = texts[texts["source"] == REAL_SOURCE]
    if real.empty:
        return
    meta = pd.read_csv(META_PATH)[["id", "category", "channel"]]
    real = real.merge(meta, on="id", how="left")
    out += ["", "=" * 70, "REAL COLLECTED MESSAGES (test split), by category and channel", "=" * 70,
            f"legitimate categories: false-positive rate; phishing_received: recall. "
            f"Rows marked [thin] have n < {MIN_QUOTABLE_ROWS}: look at them, do not quote them."]

    def line(name: str, part: pd.DataFrame) -> str:
        thin = "  [thin]" if len(part) < MIN_QUOTABLE_ROWS else ""
        flagged, high = int((part["verdict"] != "LOW").sum()), int((part["verdict"] == "HIGH").sum())
        if (part["label"] == "phishing").all():
            return f"  {name:<22} recall {rate(flagged, len(part))}   HIGH only {rate(high, len(part))}{thin}"
        return f"  {name:<22} FPR    {rate(flagged, len(part))}   HIGH only {rate(high, len(part))}{thin}"

    for category, part in real.groupby("category", dropna=False):
        out.append(line(str(category), part))
    out.append("  -- by channel, legitimate messages only --")
    for channel, part in real[real["label"] == "legitimate"].groupby("channel", dropna=False):
        out.append(line(str(channel), part))


def error_rows(d: pd.DataFrame, out_path: Path) -> None:
    text = pd.read_csv(DATASET_PATH, usecols=["id", "text"])
    wrong = d[((d["label"] == "legitimate") & (d["verdict"] != "LOW"))
              | ((d["label"] == "phishing") & (d["verdict"] == "LOW"))].merge(text, on="id", how="left")
    wrong["error_type"] = wrong["label"].map({"legitimate": "false_positive", "phishing": "false_negative"})
    wrong = wrong.sort_values(["error_type", "final_score"], ascending=[True, False])
    wrong[["id", "error_type", "kind", "label", "verdict", "final_score", "decided_by",
           "group_type", "source", "language", "text"]].to_csv(out_path, index=False, encoding="utf-8-sig")


def analyze(scores_path: Path, report_path: Path, errors_path: Path) -> None:
    df = pd.read_csv(scores_path)
    n_errors = int(df["error"].notna().sum())
    df = df[df["error"].isna()].copy()
    out = ["ETIS final-system evaluation: current decision-tier scorer, TEST split",
           "(training-split-only blacklist; synthetic rows excluded; Wilson 95% CIs in brackets)",
           f"rows scored: {len(df):,}   rows that errored (excluded): {n_errors:,}"]
    if n_errors:
        out.append("  >>> errored rows are NOT counted anywhere below; investigate before quoting numbers.")
        top = pd.read_csv(scores_path)["error"].dropna().value_counts().head(3)
        out += [f"  most common errors: {count:,} x {message}" for message, count in top.items()]

    # ---------------- URLs ----------------
    urls = df[df["kind"] == "url"].copy()
    if len(urls):
        out += ["", "=" * 70, "URL SUBMISSIONS", "=" * 70,
                f"group_type counts (blank = no group): {urls['group_type'].fillna('').value_counts().to_dict()}"]
        crosstab(out, "URLs, TEST, all", urls)
        operating_points(out, "URLs, TEST, all", urls)
        operating_points(out, "URLs, TEST, UNSEEN-HOST (group_type == 'host')", urls[urls["group_type"] == "host"])
        unseen = urls[urls["group_type"] == "host"]
        operating_points(out, "URLs, UNSEEN-HOST, with a path", unseen[unseen["has_path"] == 1])
        operating_points(out, "URLs, UNSEEN-HOST, no path (bare host)", unseen[unseen["has_path"] == 0])
        for group in sorted(set(urls["group_type"].dropna()) - {"host"}):
            operating_points(out, f"URLs, group_type == '{group}'  (report separately)", urls[urls["group_type"] == group])
        out += ["", "-- Legitimate URLs only: false-positive rate by source, and .eg domains --"]
        legit = urls[urls["label"] == "legitimate"]
        for source, part in legit.groupby("source"):
            out.append(f"  source={source:<22} FPR {rate(int((part['verdict'] != 'LOW').sum()), len(part))}"
                       f"   HIGH {rate(int((part['verdict'] == 'HIGH').sum()), len(part))}")
        eg = legit[legit["is_eg"] == 1]
        out.append(f"  .eg hosts                    FPR {rate(int((eg['verdict'] != 'LOW').sum()), len(eg))}")
        out.append(f"  allowlisted (verified brand) {rate(int((legit['allowlisted'] == 1).sum()), len(legit))} of legitimate rows")
        per_engine(out, urls)

    # ---------------- Text ----------------
    texts = df[df["kind"] == "text"].copy()
    if len(texts):
        out += ["", "=" * 70, "TEXT / EMAIL SUBMISSIONS", "=" * 70,
                f"small sample: {len(texts):,} rows; interpret intervals, not point estimates"]
        crosstab(out, "TEXT, TEST, all", texts)
        operating_points(out, "TEXT, TEST, all", texts)
        for language, part in texts.groupby("language"):
            operating_points(out, f"TEXT, language == {language}", part)
        for source, part in texts.groupby("source"):
            operating_points(out, f"TEXT, source == {source}", part)
        per_engine(out, texts)
        real_message_breakdown(out, texts)

    out += ["", "=" * 70, "LIMITS TO STATE WITH THESE NUMBERS", "=" * 70,
            "- Blacklist: the evaluation database knows only training-split URLs, so its value on",
            "  already-known threats is NOT measured here.",
            "- Legitimate URLs are Tranco top-20k plus crawled pages: precision on obscure legitimate",
            "  sites is unmeasured. Shared-platform URLs are a separate group above.",
            "- The text test set is small and mostly translated/short messages; see the intervals.",
            "- No threshold was tuned on this split; the tier thresholds came from validation data."]

    report = "\n".join(out)
    print(report)
    report_path.write_text(report + "\n", encoding="utf-8")
    error_rows(df, errors_path)
    print(f"\nSaved {report_path}\nSaved {errors_path}  (false positives / false negatives with their text)")


# ---------------------------------------------------------------------------
# Compare two runs (same rows, different code)
# ---------------------------------------------------------------------------

def compare(tag_a: str, tag_b: str) -> None:
    path_a, path_b = _paths(tag_a)[0], _paths(tag_b)[0]
    a, b = pd.read_csv(path_a), pd.read_csv(path_b)
    a, b = a[a["error"].isna()], b[b["error"].isna()]
    merged = a[["id", "label", "kind", "group_type", "source", "verdict", "final_score"]].merge(
        b[["id", "verdict", "final_score", "decided_by"]], on="id", suffixes=(f"_{tag_a}", f"_{tag_b}"))
    va, vb = f"verdict_{tag_a}", f"verdict_{tag_b}"
    out = [f"ETIS paired comparison: '{tag_a}' -> '{tag_b}'  ({len(merged):,} rows scored in both runs)"]
    for label in ("legitimate", "phishing"):
        part = merged[merged["label"] == label]
        flagged_a, flagged_b = (part[va] != "LOW"), (part[vb] != "LOW")
        newly, dropped = int((~flagged_a & flagged_b).sum()), int((flagged_a & ~flagged_b).sum())
        out += ["", f"{label}: n={len(part):,}",
                f"  flagged (>= MEDIUM): {tag_a} {rate(int(flagged_a.sum()), len(part))}",
                f"                       {tag_b} {rate(int(flagged_b.sum()), len(part))}",
                f"  newly flagged: {newly:,}   no longer flagged: {dropped:,}"
                f"   (discordant pairs; a sign test on these two numbers tells you if the change is real)"]
        out.append("  verdict changes: " + (", ".join(
            f"{x}->{y}={n:,}" for (x, y), n in part.groupby([va, vb]).size().items() if x != y) or "none"))
    changed = merged[(merged[va] != merged[vb])]
    text = pd.read_csv(DATASET_PATH, usecols=["id", "text"])
    changed = changed.merge(text, on="id", how="left").sort_values(["label", "final_score_" + tag_b], ascending=[True, False])
    path = OUT_DIR / f"final_system_changes_{tag_a}_to_{tag_b}.csv"
    changed.to_csv(path, index=False, encoding="utf-8-sig")
    out += ["", f"Saved every changed row (with its text) to {path}",
            "READ the legitimate rows that became flagged: each one is a false positive the change introduced."]
    report = "\n".join(out)
    print(report)
    (OUT_DIR / f"final_system_compare_{tag_a}_to_{tag_b}.txt").write_text(report + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--rescore", action="store_true", help="discard cached scores and rescore everything")
    parser.add_argument("--only", choices=["url", "text"], help="score only URL rows or only text rows")
    parser.add_argument("--tag", default="", help="suffix for output files (e.g. before / after)")
    parser.add_argument("--compare", nargs=2, metavar=("TAG_A", "TAG_B"), help="paired comparison of two tagged runs")
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    scores_path, report_path, errors_path = _paths(args.tag)
    if not args.analyze_only:
        score_rows(scores_path, args.rescore, args.only)
    if not scores_path.exists():
        sys.exit(f"{scores_path} not found -- run without --analyze-only first.")
    analyze(scores_path, report_path, errors_path)


if __name__ == "__main__":
    main()
