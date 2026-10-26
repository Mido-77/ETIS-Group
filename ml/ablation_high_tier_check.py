"""
ablation_high_tier_check.py

STAGE 1 before any promotion of the TLD-masked URL model (see
ml/ablation_url_tld_mask.py). The ablation showed the masked model's HIGH
recall on unseen-host test URLs rising from 66.2% to 85.8%, but it did not
report the false-positive rate at the HIGH tier. This script does, for both
models, from the same two artifacts the ablation used:
  ml/models/url_ensemble.joblib            deployed (never modified)
  ml/ablation/url_ensemble_tldmasked.joblib   masked, written by the ablation

HIGH was defined for both models as the top 0.1% of legitimate VALIDATION URLs
(deployed cutoff 0.983, masked cutoff 0.872). A rise in recall is only a gain
if the false-positive rate at that tier stays near its nominal 0.1%.

PRE-DECLARED (fixed before the result):
  C4  masked model, unseen-host test legitimate URLs: false-positive rate at
      HIGH <= 0.20% (at most 8 of 4,147 URLs), i.e. no more than twice the
      nominal 0.1%.
If C4 fails, the masked model's HIGH tier is not equivalent to the deployed
one and the promotion plan needs revisiting (e.g. a stricter HIGH cutoff
chosen on VALIDATION only), not a quiet change of the criterion here.

Also reported (descriptive): HIGH recall and FPR split by bare host / with a
path; the validation tail counts behind both cutoffs; a tier cross-table
(deployed rows x masked columns) for phishing and for legitimate URLs; and the
legitimate URLs each model puts in HIGH.

No model is changed, nothing is retrained, the test split is only scored.

Usage (project root, venv active; ~1 minute, no database):
    python ml/ablation_high_tier_check.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation_url_tld_mask import (  # noqa: E402
    DATASET_PATH, GROUPS_PATH, LIVE_ARTIFACT, MASKED_ARTIFACT, PROJECT_ROOT, feature_frame, rate, score_urls,
)

REPORT_PATH = PROJECT_ROOT / "ml" / "results" / "url_high_tier_check_report.txt"
C4_MAX_HIGH_FPR = 0.002
LIST_LIMIT = 40


def tiers_of(probabilities: np.ndarray, thresholds: dict) -> np.ndarray:
    return np.where(probabilities >= thresholds["high"], "HIGH", np.where(probabilities >= thresholds["medium"], "MEDIUM", "LOW"))


def main() -> None:
    for path in (DATASET_PATH, GROUPS_PATH, LIVE_ARTIFACT, MASKED_ARTIFACT):
        if not path.exists():
            sys.exit(f"{path} not found. Run ml/ablation_url_tld_mask.py first (it writes the masked artifact).")
    models = {"deployed": joblib.load(LIVE_ARTIFACT), "TLD-masked": joblib.load(MASKED_ARTIFACT)}
    names = list(models)

    df = pd.read_csv(DATASET_PATH)
    urls = df[df["content_type"] == "url"].reset_index(drop=True).merge(pd.read_csv(GROUPS_PATH)[["id", "group_type"]], on="id", how="left")
    te = urls[(urls["split"] == "test") & (urls["group_type"] == "host")].reset_index(drop=True)
    va = urls[urls["split"] == "val"].reset_index(drop=True)
    y_te, y_va = (te["label"] == "phishing").to_numpy(), (va["label"] == "phishing").to_numpy()
    has_path = feature_frame(te["text"].tolist())["has_path"].to_numpy() == 1

    s_te = {n: score_urls(models[n], te["text"].tolist()) for n in names}
    s_va = {n: score_urls(models[n], va["text"].tolist()) for n in names}
    thr = {n: models[n]["thresholds"] for n in names}
    tier_te = {n: tiers_of(s_te[n], thr[n]) for n in names}

    out = ["HIGH-tier check, deployed vs TLD-masked URL model (stage 1 before any promotion; nothing is changed)",
           f"unseen-host test URLs: {len(te):,} (phishing {int(y_te.sum()):,}, legitimate {int((~y_te).sum()):,}); Wilson 95% intervals",
           f"HIGH cutoff: deployed {thr['deployed']['high']:.3f}, masked {thr['TLD-masked']['high']:.3f}   "
           f"MEDIUM cutoff: deployed {thr['deployed']['medium']:.3f}, masked {thr['TLD-masked']['medium']:.3f}", ""]

    out += ["== HIGH tier on the test split ==", f"{'model':<14}{'HIGH recall':<36}{'HIGH false-positive rate'}"]
    for n in names:
        high = tier_te[n] == "HIGH"
        out.append(f"{n:<14}{rate(int(high[y_te].sum()), int(y_te.sum())):<36}{rate(int(high[~y_te].sum()), int((~y_te).sum()))}")
    out.append("  by URL type:")
    for label, sel in (("bare host", ~has_path), ("with a path", has_path)):
        for n in names:
            high = tier_te[n] == "HIGH"
            out.append(f"    {label:<12}{n:<12}recall {rate(int(high[y_te & sel].sum()), int((y_te & sel).sum())):<36}"
                       f"FPR {rate(int(high[~y_te & sel].sum()), int((~y_te & sel).sum()))}")

    out += ["", "== Validation tails behind the cutoffs (legitimate validation URLs at or above each cutoff) =="]
    for n in names:
        legit = s_va[n][~y_va]
        out.append(f"  {n:<12}n legit = {len(legit):,}; >= MEDIUM cutoff: {int((legit >= thr[n]['medium']).sum())}   >= HIGH cutoff: {int((legit >= thr[n]['high']).sum())}"
                   f"   (nominal: {0.01 * len(legit):.0f} and {0.001 * len(legit):.0f})")

    for label, sel in (("phishing", y_te), ("legitimate", ~y_te)):
        out += ["", f"== Tier cross-table on {label} test URLs (rows: deployed, columns: masked) =="]
        table = pd.crosstab(pd.Series(tier_te["deployed"][sel], name="deployed"), pd.Series(tier_te["TLD-masked"][sel], name="masked"))
        table = table.reindex(index=["LOW", "MEDIUM", "HIGH"], columns=["LOW", "MEDIUM", "HIGH"], fill_value=0)
        out += ["  " + line for line in table.to_string().splitlines()]

    out += ["", f"== Legitimate URLs placed in HIGH (first {LIST_LIMIT} per model) =="]
    for n in names:
        idx = np.where((tier_te[n] == "HIGH") & ~y_te)[0]
        out.append(f"  {n}: {len(idx)}")
        out += [f"      {te['text'][i][:110]}   (p = {s_te[n][i]:.3f})" for i in idx[:LIST_LIMIT]]

    fpr_masked = float(((tier_te["TLD-masked"] == "HIGH") & ~y_te).sum() / (~y_te).sum())
    c4 = fpr_masked <= C4_MAX_HIGH_FPR
    out += ["", "== Pre-declared criterion ==",
            f"  C4 masked HIGH false-positive rate = {100 * fpr_masked:.3f}% (allowed <= {100 * C4_MAX_HIGH_FPR:.2f}%)  -> {'PASS' if c4 else 'FAIL'}",
            "  This script promotes nothing and changes nothing."]

    report = "\n".join(out)
    print(report)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved {REPORT_PATH}")


if __name__ == "__main__":
    main()
