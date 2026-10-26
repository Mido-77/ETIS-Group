"""
evaluate_fusion.py

Offline comparison of HOW to combine ETIS's engines, before changing the live
scorer. Run it against the evaluation database (training-split-only
blacklist), never the demo database.

Phase 1 -- scoring (slow, resumable, ~20-30 min once)
    Every VALIDATION and TEST row (synthetic rows excluded) is submitted the
    way a user would: URL rows as a URL only, text/email rows as text only.
    For each row we record the CURRENT system's outputs (final score, verdict,
    rule-based / blacklist / BiLSTM scores, via calculate_final_risk_score)
    plus the two NEW engines (URL ensemble, character text model).
    Results are appended to evaluation/fusion_engine_scores.csv; if the run is
    interrupted, re-running continues where it stopped.

Phase 2 -- analysis (instant, from the CSV)
    URL submissions, candidates:
        legacy_current        the system as it is today
        url_model_alone       the learned URL ensemble only
        additive (+URL X)     rule 30 + blacklist 40 + url model X points, rescaled
                              (Option 1); X chosen on VALIDATION
        decision_tiers        exact blacklist hit / typosquat >= 90 / URL tier HIGH
                              decide outright; URL tier MEDIUM, domain match or
                              rule >= 40 -> MEDIUM; allowlisted domains -> LOW (Option 2)
    Text submissions, candidates:
        legacy_current        rule 15 + BiLSTM 45, rescaled
        text_char_alone       the new character model only
        additive (rule R)     R rule points + 45 text-model points; R chosen on VALIDATION
        decision_tiers        text model above its 1%-FPR validation threshold or
                              rule >= 60 -> HIGH; above its 5%-FPR threshold or
                              rule >= 30 -> MEDIUM
    All thresholds / weights are chosen on validation only; test is reported.

IMPORTANT HONESTY NOTE: in the evaluation database the blacklist knows only
training-split URLs, so on validation/test it almost never fires -- by design
(it is the honest "unseen URL" setting). A blacklist's real job is known
threats, which this snapshot cannot represent. Therefore the additive
candidate keeps the blacklist at its fixed 40 points instead of tuning it,
and the tiers keep "exact blacklist hit -> HIGH" as a design rule. Do not
conclude from this evaluation that the blacklist is useless.

Usage (project root, venv active):
    $env:DATABASE_URL = ...etis_eval...      # see seed_database.py
    python evaluation/evaluate_fusion.py
    python evaluation/evaluate_fusion.py --analyze-only
    python evaluation/evaluate_fusion.py --rescore        # discard cached scores
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"
GROUPS_PATH = PROJECT_ROOT / "data" / "processed" / "split_groups.csv"
SCORES_PATH = PROJECT_ROOT / "evaluation" / "fusion_engine_scores.csv"
REPORT_PATH = PROJECT_ROOT / "evaluation" / "fusion_report.txt"

SYNTHETIC_SOURCE = "synthetic_egyptian_brands"
FIELDS = ["id", "split", "kind", "label", "group_type", "source", "has_path", "allowlisted",
          "legacy_score", "legacy_verdict", "rule_score", "bl_score", "bl_match",
          "lstm_score", "new_prob", "new_tier", "error"]

URL_TARGETS = (0.01, 0.001)
TEXT_TARGETS = (0.05, 0.01)
URL_X_GRID = [10, 20, 30, 40, 60, 80]
TEXT_RULE_GRID = [0, 5, 15, 30, 45]
N_BOOTSTRAP = 1000
SEED = 42


# ---------------------------------------------------------------------------
# Phase 1: scoring
# ---------------------------------------------------------------------------

def _has_path(url: str) -> int:
    try:
        return int(len(urlsplit(url if "://" in url else "http://" + url).path) > 1)
    except ValueError:
        return 0


def score_rows(rescore: bool) -> None:
    from app.core.database import SessionLocal, engine
    database_name = engine.url.database or ""
    if "eval" not in database_name:
        sys.exit(f"Refusing to run against database '{database_name}': it is not an evaluation database. "
                 "Set DATABASE_URL to etis_eval (training-split-only blacklist).")
    print(f"Target database: {engine.url.render_as_string(hide_password=True)}")

    from app.engines.ml.text_char_inference import calculate_text_char_phishing_score
    from app.engines.ml.url_model_inference import calculate_url_model_score
    from app.engines.risk_scoring import calculate_final_risk_score

    df = pd.read_csv(DATASET_PATH)
    df = df[df["split"].isin(["val", "test"]) & (df["source"] != SYNTHETIC_SOURCE)]
    if GROUPS_PATH.exists():
        df = df.merge(pd.read_csv(GROUPS_PATH)[["id", "group_type"]], on="id", how="left")
    else:
        df["group_type"] = ""

    if SCORES_PATH.exists() and rescore:
        SCORES_PATH.unlink()
    done = set(pd.read_csv(SCORES_PATH, usecols=["id"])["id"]) if SCORES_PATH.exists() else set()
    todo = df[~df["id"].isin(done)]
    print(f"{len(df):,} rows in val+test; {len(done):,} already scored; scoring {len(todo):,} now.")

    new_file = not SCORES_PATH.exists()
    start = time.time()
    db = SessionLocal()
    with open(SCORES_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for n, row in enumerate(todo.itertuples(index=False), start=1):
            is_url = row.content_type == "url"
            record = {"id": row.id, "split": row.split, "kind": "url" if is_url else "text",
                      "label": row.label, "group_type": row.group_type, "source": row.source,
                      "has_path": _has_path(row.text) if is_url else ""}
            try:
                if is_url:
                    legacy = calculate_final_risk_score(url=row.text, db=db)
                    new = calculate_url_model_score(row.text)
                    record["new_prob"], record["new_tier"] = new["probability"], new["tier"]
                else:
                    legacy = calculate_final_risk_score(text=row.text, db=db)
                    record["new_prob"] = calculate_text_char_phishing_score(row.text)["probability"]
                details = legacy["engine_details"]
                blacklist, lstm = details.get("blacklist"), details.get("lstm")
                record.update({
                    "allowlisted": int(bool(legacy.get("allowlisted") or details.get("allowlisted"))),
                    "legacy_score": legacy["final_score"], "legacy_verdict": legacy["verdict"],
                    "rule_score": details["rule_based"]["score"],
                    "bl_score": blacklist["score"] if blacklist else "",
                    "bl_match": (blacklist.get("match_type") or "") if blacklist else "",
                    "lstm_score": lstm["score"] if lstm else "",
                })
            except Exception as exc:  # one bad row must not kill a 25-minute run
                db.rollback()
                record["error"] = f"{type(exc).__name__}: {exc}"[:200]
            writer.writerow(record)
            if n % 1000 == 0:
                f.flush()
                print(f"  ...{n:,}/{len(todo):,} rows ({time.time() - start:.0f}s)")
    db.close()
    print(f"Scoring done in {time.time() - start:.0f}s -> {SCORES_PATH}")


# ---------------------------------------------------------------------------
# Phase 2: analysis helpers
# ---------------------------------------------------------------------------

def quantile_or_inf(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if len(values) else float("inf")


def recall_fpr(scores: np.ndarray, y: np.ndarray, threshold: float) -> tuple[float, float]:
    pred = scores >= threshold
    pos, neg = y == 1, y == 0
    recall = (pred & pos).sum() / pos.sum() if pos.sum() else float("nan")
    fpr = (pred & neg).sum() / neg.sum() if neg.sum() else float("nan")
    return float(recall), float(fpr)


def auc_or_nan(y, s) -> float:
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def auc_ci(y, s, rng) -> tuple[float, float]:
    vals = []
    for _ in range(N_BOOTSTRAP):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            vals.append(roc_auc_score(y[i], s[i]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"),) * 2


def f(v: float) -> str:
    return "  n/a " if v != v else f"{v:.3f}"


def table(out: list, title: str, candidates: dict, val_y, test_y, mask, targets) -> None:
    """candidates: name -> (val_scores, test_scores). Thresholds come from VALIDATION legit rows."""
    out += ["", f"-- {title}  (n={int(mask.sum()):,}, phishing={int(test_y[mask].sum()):,}) --"]
    header = f"{'candidate':<28}{'AUC':>7}"
    for t in targets:
        header += f"{f'recall@{t:.1%}':>14}{'(FPR)':>8}"
    out.append(header)
    for name, (val_s, test_s) in candidates.items():
        line = f"{name:<28}{f(auc_or_nan(test_y[mask], test_s[mask])):>7}"
        for t in targets:
            thr = quantile_or_inf(val_s[val_y == 0], 1 - t)
            r, fp = recall_fpr(test_s[mask], test_y[mask], thr)
            line += f"{f(r):>14}{f(fp):>8}"
        out.append(line)


def natural_points(out: list, title: str, points: dict, test_y, mask) -> None:
    out += ["", f"-- {title} (fixed, design-time operating points) --",
            f"{'operating point':<44}{'recall':>9}{'FPR':>9}"]
    for name, flagged in points.items():
        pred = flagged[mask]
        y = test_y[mask]
        r = (pred & (y == 1)).sum() / max(1, (y == 1).sum())
        fp = (pred & (y == 0)).sum() / max(1, (y == 0).sum())
        out.append(f"{name:<44}{f(float(r)):>9}{f(float(fp)):>9}")


def verdict_crosstab(out: list, title: str, sub: pd.DataFrame) -> None:
    out += ["", f"-- {title} --", f"{'':<14}{'LOW':>8}{'MEDIUM':>8}{'HIGH':>8}   (row %)"]
    for label in ("phishing", "legitimate"):
        part = sub[sub["label"] == label]
        counts = [int((part["legacy_verdict"] == v).sum()) for v in ("LOW", "MEDIUM", "HIGH")]
        total = max(1, sum(counts))
        out.append(f"{label:<14}" + "".join(f"{100 * c / total:>7.1f}%" for c in counts) + f"   (n={sum(counts):,})")


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

def additive_url(d: pd.DataFrame, x: float) -> np.ndarray:
    rule, bl, p = d["rule_score"].fillna(0).to_numpy(), d["bl_score"].fillna(0).to_numpy(), d["new_prob"].to_numpy()
    return (30 * rule / 100 + 40 * bl / 100 + x * p) / (70 + x) * 100


def tiers_url(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    rule, bl_match = d["rule_score"].fillna(0).to_numpy(), d["bl_match"].fillna("").to_numpy()
    tier_name, allow = d["new_tier"].fillna("LOW").to_numpy(), d["allowlisted"].fillna(0).to_numpy() == 1
    high = (bl_match == "exact") | (rule >= 90) | (tier_name == "HIGH")
    medium = (tier_name == "MEDIUM") | (bl_match == "domain") | (rule >= 40)
    tier = np.where(high, 2, np.where(medium, 1, 0))
    tier = np.where(allow, 0, tier)
    return tier, tier + 0.99 * d["new_prob"].to_numpy()


def additive_text(d: pd.DataFrame, rule_points: float) -> np.ndarray:
    rule, p = d["rule_score"].fillna(0).to_numpy(), d["new_prob"].to_numpy()
    return (45 * p + rule_points * rule / 100) / (45 + rule_points) * 100


def best_by_validation(grid, make, val, val_y, target) -> float:
    def key(x):
        s = make(val, x)
        thr = quantile_or_inf(s[val_y == 0], 1 - target)
        return (recall_fpr(s, val_y, thr)[0], auc_or_nan(val_y, s))
    return max(grid, key=key)


def analyze(out: list) -> None:
    df = pd.read_csv(SCORES_PATH)
    errors = int(df["error"].notna().sum())
    df = df[df["error"].isna()].copy()
    df["y"] = (df["label"] == "phishing").astype(int)
    out += ["ETIS fusion evaluation (honest, training-split-only blacklist)",
            f"rows scored: {len(df):,}  (errors excluded: {errors})",
            f"  URL rows: val={int(((df.kind == 'url') & (df.split == 'val')).sum()):,} "
            f"test={int(((df.kind == 'url') & (df.split == 'test')).sum()):,}   "
            f"text rows: val={int(((df.kind == 'text') & (df.split == 'val')).sum()):,} "
            f"test={int(((df.kind == 'text') & (df.split == 'test')).sum()):,}"]
    out += ["Reading the tables: 'recall@1.0% (FPR)' = detection rate when the threshold is set so that",
            "1% of LEGITIMATE VALIDATION rows are flagged; the (FPR) next to it is what actually happened on",
            "TEST. If that FPR is far above the target, the candidate cannot reach that false-positive rate",
            "(many tied scores, e.g. the current system gives most legitimate rows exactly 0)."]
    rng = np.random.default_rng(SEED)

    # ================= URL submissions =================
    val, test = (df[(df.kind == "url") & (df.split == s)].reset_index(drop=True) for s in ("val", "test"))
    vy, ty = val["y"].to_numpy(), test["y"].to_numpy()

    verdict_crosstab(out, "URLs, TEST: verdicts of the CURRENT system (deployed thresholds, honest blacklist)", test)

    best_x = best_by_validation(URL_X_GRID, additive_url, val, vy, URL_TARGETS[0])
    out += ["", f"URL additive: url-model points X chosen on validation (recall at 1% FPR) = {best_x}"]

    v_tier, v_tier_score = tiers_url(val)
    t_tier, t_tier_score = tiers_url(test)
    candidates = {
        "legacy_current": (val["legacy_score"].to_numpy(), test["legacy_score"].to_numpy()),
        "url_model_alone": (val["new_prob"].to_numpy(), test["new_prob"].to_numpy()),
        f"additive (+URL {best_x})": (additive_url(val, best_x), additive_url(test, best_x)),
        "decision_tiers": (v_tier_score, t_tier_score),
    }
    group, has_path = test["group_type"].to_numpy(), test["has_path"].to_numpy() == 1
    views = [("URLs, TEST, all", np.ones(len(test), bool)),
             ("URLs, TEST, UNSEEN-HOST", group == "host"),
             ("URLs, TEST, UNSEEN-HOST that have a path", (group == "host") & has_path)]
    out += ["", "================ URL submissions: thresholds chosen on validation ================"]
    for title, mask in views:
        table(out, title, candidates, vy, ty, mask, URL_TARGETS)

    unseen = group == "host"
    natural_points(out, "URLs, TEST, UNSEEN-HOST", {
        "legacy_current: MEDIUM or HIGH (score > 0)": (test["legacy_score"] > 0).to_numpy(),
        "legacy_current: HIGH (score > 50)": (test["legacy_score"] > 50).to_numpy(),
        "decision_tiers: MEDIUM or HIGH": t_tier >= 1,
        "decision_tiers: HIGH": t_tier >= 2,
    }, ty, unseen)
    out += ["", "Bootstrap 95% CI of ROC-AUC, UNSEEN-HOST URLs:"]
    for name, (_, ts) in candidates.items():
        lo, hi = auc_ci(ty[unseen], ts[unseen], rng)
        out.append(f"  {name:<28}[{lo:.4f}, {hi:.4f}]")

    # ================= Text submissions =================
    val, test = (df[(df.kind == "text") & (df.split == s)].reset_index(drop=True) for s in ("val", "test"))
    vy, ty = val["y"].to_numpy(), test["y"].to_numpy()
    verdict_crosstab(out, "TEXT, TEST: verdicts of the CURRENT system", test)

    best_r = best_by_validation(TEXT_RULE_GRID, additive_text, val, vy, TEXT_TARGETS[0])
    out += ["", f"Text additive: rule points R (text model fixed at 45) chosen on validation = {best_r}"]

    thr_high = quantile_or_inf(val.loc[vy == 0, "new_prob"].to_numpy(), 1 - 0.01)
    thr_med = quantile_or_inf(val.loc[vy == 0, "new_prob"].to_numpy(), 1 - 0.05)

    def tiers_text(d):
        rule, p = d["rule_score"].fillna(0).to_numpy(), d["new_prob"].to_numpy()
        high, med = (p >= thr_high) | (rule >= 60), (p >= thr_med) | (rule >= 30)
        tier = np.where(high, 2, np.where(med, 1, 0))
        return tier, tier + 0.99 * p

    v_tier, v_tier_score = tiers_text(val)
    t_tier, t_tier_score = tiers_text(test)
    candidates = {
        "legacy_current": (val["legacy_score"].to_numpy(), test["legacy_score"].to_numpy()),
        "text_char_alone": (val["new_prob"].to_numpy(), test["new_prob"].to_numpy()),
        f"additive (rule {best_r})": (additive_text(val, best_r), additive_text(test, best_r)),
        "decision_tiers": (v_tier_score, t_tier_score),
    }
    out += ["", "================ Text submissions: thresholds chosen on validation ================"]
    table(out, "TEXT, TEST, all", candidates, vy, ty, np.ones(len(test), bool), TEXT_TARGETS)
    natural_points(out, "TEXT, TEST", {
        "legacy_current: MEDIUM or HIGH (score > 0)": (test["legacy_score"] > 0).to_numpy(),
        "legacy_current: HIGH (score > 50)": (test["legacy_score"] > 50).to_numpy(),
        "decision_tiers: MEDIUM or HIGH": t_tier >= 1,
        "decision_tiers: HIGH": t_tier >= 2,
    }, ty, np.ones(len(test), bool))
    out += ["", f"(text tier thresholds from validation: HIGH p >= {thr_high:.3f} (1% FPR), MEDIUM p >= {thr_med:.3f} (5% FPR))",
            f"(text test set is small: {int((ty == 0).sum())} legitimate rows, so 1% FPR means about {max(1, round(0.01 * (ty == 0).sum()))} message)"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--rescore", action="store_true", help="discard cached scores and rescore everything")
    args = parser.parse_args()

    if not args.analyze_only:
        score_rows(args.rescore)
    if not SCORES_PATH.exists():
        sys.exit(f"{SCORES_PATH} not found -- run without --analyze-only first.")

    out: list[str] = []
    analyze(out)
    report = "\n".join(out)
    print("\n" + report)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved {REPORT_PATH}")


if __name__ == "__main__":
    main()
