"""
ablation_url_tld_mask.py

OFFLINE ABLATION. Does hiding the TLD fix the ".eg means legitimate" shortcut
found by evaluation/test_tld_shortcut.py (URL-model phishing recall 93.0% ->
19.7% when only the last host label is swapped to .eg), and what does it cost?

NOTHING LIVE CHANGES. The baseline is the DEPLOYED artifact
(ml/models/url_ensemble.joblib), loaded and scored, never modified. The masked
model is trained here and saved under ml/ablation/ (NOT ml/models/: the
held-out benchmark's freeze manifest fingerprints ml/models/*.joblib and a new
file there would make its evaluator refuse to run). No backend file is touched.

The masked model = the deployed training protocol (ml/export_models.py:
train split only, C and tier thresholds chosen on the VALIDATION split,
MEDIUM = 1% and HIGH = 0.1% false-positive rate of legitimate validation URLs),
with exactly two differences:
  * character model: the last host label is replaced by the token "tld" in the
    text it reads ("x.com.eg/p" -> "x.com.tld/p");
  * LightGBM: the TLD one-hot columns are dropped.
It does NOT remove everything correlated with the TLD (host length, a "com"
second-level label, ...); it removes the TLD identity. State that.

PRE-DECLARED PROMOTION CRITERIA (fixed before any result; all three must hold
before replacing the live model is even discussed; nothing is promoted by this
script):
  C1 cost on ordinary phishing, UNSEEN-HOST test URLs, MEDIUM tier:
       masked recall >= baseline recall - 3.0 points   AND
       masked false-positive rate <= baseline + 0.5 points.
  C2 benefit: for the masked model, recall on phishing URLs whose last label is
       swapped to .eg is at most 25 points below its recall on the originals
       (the live model loses about 73 points).
  C3 side effect on REAL Egyptian sites (URL model alone, MEDIUM tier): masked
       false-positive rate <= 5% on BOTH the Tranco .eg sample (apex form) and
       the dataset's .eg test URLs (brand-registry domains excluded).
Test-split numbers are looked at once, for reporting; the model is never tuned
on them. If you change the masking or the criteria after reading this report,
say so: this script then has become a development tool and a fresh sample is
needed to evaluate the result.

Usage (project root, venv active; takes a few minutes, no database needed):
    python ml/ablation_url_tld_mask.py
    python ml/ablation_url_tld_mask.py --skip-baseline-check     (skips retraining the unmasked variant)
"""

import argparse
import csv
import math
import re
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.egyptian_brands import EGYPTIAN_BRAND_DOMAINS  # noqa: E402
from app.engines.ml.url_features import (  # noqa: E402
    HOST_FEATURES, PATH_FEATURES, TOP_TLDS, build_matrix, char_model_text, feature_frame,
)
from app.engines.ml.url_model_inference import ensure_scheme  # noqa: E402

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"
GROUPS_PATH = PROJECT_ROOT / "data" / "processed" / "split_groups.csv"
EXTERNAL_PATH = PROJECT_ROOT / "data" / "benchmarks" / "eg_legit_external.csv"
HELDOUT_PATH = PROJECT_ROOT / "data" / "benchmarks" / "egypt_brand_typosquats_heldout.csv"
LIVE_ARTIFACT = PROJECT_ROOT / "ml" / "models" / "url_ensemble.joblib"
ABLATION_DIR = PROJECT_ROOT / "ml" / "ablation"
MASKED_ARTIFACT = ABLATION_DIR / "url_ensemble_tldmasked.joblib"
REPORT_PATH = PROJECT_ROOT / "ml" / "results" / "url_tld_ablation_report.txt"

SEED = 42
URL_C_GRID = [1.0, 10.0]
URL_MEDIUM_FPR, URL_HIGH_FPR = 0.01, 0.001
THIN_N = 20
EG_SECOND_LEVEL = {"com", "gov", "edu", "org", "net", "sci", "eun", "mil", "name", "info", "biz"}

# Pre-declared criteria
C1_MAX_RECALL_LOSS, C1_MAX_FPR_GAIN = 0.03, 0.005
C2_MAX_SWAP_DROP = 0.25
C3_MAX_EG_FPR = 0.05


# ---------------------------------------------------------------------------
# Masking and swapping
# ---------------------------------------------------------------------------

def mask_tld(text: str) -> str:
    """text = char_model_text output (lowercase, scheme removed). Replace the last host label by 'tld'."""
    match = re.match(r"^([^/?#]*)(.*)$", text, flags=re.S)
    authority, rest = match.group(1), match.group(2)
    userinfo, sep, hostport = authority.rpartition("@")
    port_match = re.match(r"^(.*?)(:\d+)?$", hostport)
    host, port = port_match.group(1).rstrip("."), port_match.group(2) or ""
    if "." in host and not re.fullmatch(r"[\d.]+", host):      # leave IP addresses alone
        host = host.rsplit(".", 1)[0] + ".tld"
    return userinfo + sep + host + port + rest


def swap_tld(url: str, new_label: str) -> str | None:
    """Same rule as evaluation/test_tld_shortcut.py: only the LAST host label changes."""
    t = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", t):
        return None
    try:
        parts = urlsplit(t)
        host, port = parts.hostname, parts.port
    except ValueError:
        return None
    if not host or "@" in parts.netloc or port is not None:
        return None
    labels = host.lower().split(".")
    if len(labels) < 2 or labels[-1].isdigit() or labels[-1] in ("eg", new_label):
        return None
    labels[-1] = new_label
    return urlunsplit((parts.scheme, ".".join(labels), parts.path, parts.query, parts.fragment))


def host_of(text: str) -> str:
    t = text.strip().replace("\\", "/")
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", t):
        t = "http://" + t
    try:
        host = urlsplit(t).hostname or ""
    except ValueError:
        return ""
    host = host.lower()
    return (host[4:] if host.startswith("www.") else host).rstrip(".")


def registrable(host: str) -> str:
    labels = host.strip(".").split(".")
    if len(labels) >= 3 and labels[-1] == "eg" and labels[-2] in EG_SECOND_LEVEL:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:]) if len(labels) >= 2 else host


# ---------------------------------------------------------------------------
# Training (identical protocol to ml/export_models.py train_url_model, plus the mask)
# ---------------------------------------------------------------------------

def train_variant(urls: pd.DataFrame, feats: pd.DataFrame, masked: bool) -> dict:
    import lightgbm as lgb

    y = (urls["label"] == "phishing").astype(int).to_numpy()
    tr, va = (urls["split"] == "train").to_numpy(), (urls["split"] == "val").to_numpy()
    texts = [char_model_text(u) for u in urls["text"]]
    if masked:
        texts = [mask_tld(t) for t in texts]
    tld_categories = [] if masked else feats.loc[tr, "tld"].value_counts().head(TOP_TLDS).index.tolist()

    vectoriser = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=200_000)
    X_tr = vectoriser.fit_transform([t for t, m in zip(texts, tr) if m])
    X_va = vectoriser.transform([t for t, m in zip(texts, va) if m])
    y_tr, y_va = y[tr], y[va]
    best_c = max(URL_C_GRID, key=lambda c: roc_auc_score(
        y_va, LogisticRegression(C=c, max_iter=300, class_weight="balanced").fit(X_tr, y_tr).predict_proba(X_va)[:, 1]))
    char_logreg = LogisticRegression(C=best_c, max_iter=300, class_weight="balanced").fit(X_tr, y_tr)

    columns = HOST_FEATURES + PATH_FEATURES
    X_all = build_matrix(feats, columns, tld_categories)
    if masked:
        X_all = X_all[columns]                       # drops the constant tld_other column
    lightgbm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8,
                                  colsample_bytree=0.8, random_state=SEED, verbose=-1)
    lightgbm.fit(X_all[tr], y_tr)

    p_val = (char_logreg.predict_proba(X_va)[:, 1] + lightgbm.predict_proba(X_all[va])[:, 1]) / 2
    legit_val = p_val[y_va == 0]
    thresholds = {"medium": float(np.quantile(legit_val, 1 - URL_MEDIUM_FPR)),
                  "high": float(np.quantile(legit_val, 1 - URL_HIGH_FPR))}
    print(f"  {'masked' if masked else 'unmasked'}: C={best_c}, validation ROC-AUC={roc_auc_score(y_va, p_val):.4f}, "
          f"train rows={int(tr.sum())}, thresholds medium={thresholds['medium']:.3f} high={thresholds['high']:.3f}")
    return {"vectorizer": vectoriser, "char_logreg": char_logreg, "lightgbm": lightgbm, "lgbm_columns": columns,
            "tld_categories": tld_categories, "thresholds": thresholds, "c": best_c, "tld_masked": masked,
            "trained_on": "train split only", "created": date.today().isoformat()}


def score_urls(model: dict, urls: list[str]) -> np.ndarray:
    """Ensemble probability, same code path as serving (plus the mask for the masked model)."""
    raws = [ensure_scheme(u) for u in urls]
    texts = [char_model_text(r) for r in raws]
    masked = bool(model.get("tld_masked"))
    if masked:
        texts = [mask_tld(t) for t in texts]
    p_char = model["char_logreg"].predict_proba(model["vectorizer"].transform(texts))[:, 1]
    X = build_matrix(feature_frame(raws), model["lgbm_columns"], model["tld_categories"])
    if masked:
        X = X[model["lgbm_columns"]]
    p_lgbm = model["lightgbm"].predict_proba(X)[:, 1]
    return (p_char + p_lgbm) / 2


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(k: int, n: int) -> str:
    if n == 0:
        return "n/a (n=0)"
    lo, hi = wilson(k, n)
    return f"{100 * k / n:5.1f}% [{100 * lo:4.1f}, {100 * hi:5.1f}] ({k}/{n})" + (" [thin]" if n < THIN_N else "")


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n)


def pair(a: np.ndarray, b: np.ndarray) -> tuple[int, int, int, int]:
    return (int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum()))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-baseline-check", action="store_true")
    args = parser.parse_args()

    for path in (DATASET_PATH, GROUPS_PATH, LIVE_ARTIFACT):
        if not path.exists():
            sys.exit(f"{path} not found.")

    df = pd.read_csv(DATASET_PATH)
    urls = df[df["content_type"] == "url"].reset_index(drop=True)
    groups = pd.read_csv(GROUPS_PATH)[["id", "group_type"]]
    urls = urls.merge(groups, on="id", how="left")
    print(f"URL rows: {len(urls):,} (train {int((urls['split'] == 'train').sum()):,}, "
          f"val {int((urls['split'] == 'val').sum()):,}, test {int((urls['split'] == 'test').sum()):,})")

    started = time.time()
    feats = feature_frame(urls["text"].tolist())
    print(f"features extracted in {time.time() - started:.0f}s; training...")
    masked_model = train_variant(urls, feats, masked=True)
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(masked_model, MASKED_ARTIFACT, compress=3)
    live_model = joblib.load(LIVE_ARTIFACT)

    baseline_check = "skipped (--skip-baseline-check)"
    test_mask_all = (urls["split"] == "test").to_numpy()
    if not args.skip_baseline_check:
        print("baseline check: retraining the UNMASKED variant with this script's protocol...")
        retrained = train_variant(urls, feats, masked=False)
        sample = urls[test_mask_all].sample(n=min(2000, int(test_mask_all.sum())), random_state=SEED)["text"].tolist()
        diff = np.abs(score_urls(retrained, sample) - score_urls(live_model, sample))
        baseline_check = (f"retrained unmasked vs deployed artifact on {len(sample)} test URLs: max |difference| = {diff.max():.2e}, "
                          f"mean = {diff.mean():.2e}  -> {'matches' if diff.max() < 0.02 else 'DOES NOT MATCH: investigate before trusting the comparison'}")
        print("  " + baseline_check)

    models = {"deployed (baseline)": live_model, "TLD-masked": masked_model}
    names = list(models)
    thr = {n: models[n]["thresholds"] for n in names}

    # ---- 1. unseen-host test URLs ----
    te = urls[(urls["split"] == "test") & (urls["group_type"] == "host")].reset_index(drop=True)
    y_te = (te["label"] == "phishing").to_numpy()
    scores_te = {n: score_urls(models[n], te["text"].tolist()) for n in names}
    has_path = feature_frame(te["text"].tolist())["has_path"].to_numpy() == 1

    # ---- 2. TLD swap (all eligible unseen-host test URLs) ----
    eligible = [i for i, u in enumerate(te["text"]) if swap_tld(u, "eg") and swap_tld(u, "net") and not host_of(u).endswith(".eg")]
    sw_urls = {"original": [te["text"][i] for i in eligible],
               "swap_eg": [swap_tld(te["text"][i], "eg") for i in eligible],
               "swap_net": [swap_tld(te["text"][i], "net") for i in eligible]}
    sw_y = y_te[eligible]
    sw_scores = {(n, v): score_urls(models[n], sw_urls[v]) for n in names for v in sw_urls}

    # ---- 3. real Egyptian sites ----
    brand_regs = {registrable(d) for _, d in EGYPTIAN_BRAND_DOMAINS}
    eg_rows = te[(te["label"] == "legitimate") & te["text"].map(lambda u: host_of(u).endswith(".eg"))]
    eg_rows = eg_rows[~eg_rows["text"].map(lambda u: registrable(host_of(u)) in brand_regs)]
    eg_scores = {n: score_urls(models[n], eg_rows["text"].tolist()) for n in names}
    eg_domains = eg_rows["text"].map(lambda u: registrable(host_of(u))).to_numpy()

    external = pd.read_csv(EXTERNAL_PATH) if EXTERNAL_PATH.exists() else None
    ext_scores = {n: score_urls(models[n], external["url"].tolist()) for n in names} if external is not None else {}
    controls = None
    if HELDOUT_PATH.exists():
        heldout = pd.read_csv(HELDOUT_PATH)
        controls = heldout[(heldout["kind"] == "control") & (heldout["technique"] == "control_egypt_site")]
        ctl_scores = {n: score_urls(models[n], controls["url"].tolist()) for n in names}

    # ---------------------------------------------------------------- report
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        head = "unknown"
    out = ["URL model TLD-masking ablation (OFFLINE; the live model and backend are untouched)",
           f"git HEAD: {head}; masked artifact: {MASKED_ARTIFACT.relative_to(PROJECT_ROOT)} (not used by the backend)",
           f"baseline = the DEPLOYED artifact {LIVE_ARTIFACT.relative_to(PROJECT_ROOT)}",
           f"baseline check: {baseline_check}",
           "Tiers: MEDIUM = 1% and HIGH = 0.1% false-positive rate of legitimate VALIDATION URLs, per model (thresholds in the table below).",
           "Rows of one host are not independent and CIs are Wilson 95%: treat small differences as noise.", "",
           f"thresholds   deployed: medium {thr[names[0]]['medium']:.3f}, high {thr[names[0]]['high']:.3f}    "
           f"masked: medium {thr[names[1]]['medium']:.3f}, high {thr[names[1]]['high']:.3f}", ""]

    out += [f"== 1. UNSEEN-HOST test URLs (group_type = host; n = {len(te):,}, phishing {int(y_te.sum()):,}) ==",
            f"{'model':<22}{'ROC-AUC':>8}  {'MEDIUM+ recall':<34}{'MEDIUM+ false-positive rate':<34}{'HIGH recall':<34}"]
    flags_med = {}
    for n in names:
        s = scores_te[n]
        med, high = s >= thr[n]["medium"], s >= thr[n]["high"]
        flags_med[n] = med
        out.append(f"{n:<22}{roc_auc_score(y_te, s):>8.4f}  {rate(int(med[y_te].sum()), int(y_te.sum())):<34}"
                   f"{rate(int(med[~y_te].sum()), int((~y_te).sum())):<34}{rate(int(high[y_te].sum()), int(y_te.sum())):<34}")
    out.append("  by URL type (MEDIUM+):")
    for label, sel in (("bare host", ~has_path), ("with a path", has_path)):
        for n in names:
            med = flags_med[n]
            out.append(f"    {label:<12}{n:<22}recall {rate(int(med[y_te & sel].sum()), int((y_te & sel).sum())):<34}"
                       f"FPR {rate(int(med[~y_te & sel].sum()), int((~y_te & sel).sum()))}")
    a, b = flags_med[names[0]], flags_med[names[1]]
    both, lost, gained, _ = pair(a[y_te], b[y_te])
    out.append(f"  paired, phishing: both flag {both}, only deployed {lost}, only masked {gained}, exact McNemar p = {exact_mcnemar_p(lost, gained):.4f}")
    both, lost, gained, _ = pair(a[~y_te], b[~y_te])
    out.append(f"  paired, legitimate: both flag {both}, only deployed {lost}, only masked {gained}, exact McNemar p = {exact_mcnemar_p(lost, gained):.4f}")

    out += ["", f"== 2. TLD SWAP, URL model alone, MEDIUM+ (all {len(eligible):,} eligible unseen-host test URLs; swapped URLs are SYNTHETIC) ==",
            f"{'model':<22}{'variant':<10}{'phishing recall':<34}{'legitimate false-positive rate':<34}"]
    swap_recall = {}
    for n in names:
        for v in ("original", "swap_eg", "swap_net"):
            med = sw_scores[(n, v)] >= thr[n]["medium"]
            swap_recall[(n, v)] = med[sw_y].mean() if sw_y.any() else float("nan")
            out.append(f"{n:<22}{v:<10}{rate(int(med[sw_y].sum()), int(sw_y.sum())):<34}{rate(int(med[~sw_y].sum()), int((~sw_y).sum()))}")
    for n in names:
        drop_eg = swap_recall[(n, "original")] - swap_recall[(n, "swap_eg")]
        drop_net = swap_recall[(n, "original")] - swap_recall[(n, "swap_net")]
        o = sw_scores[(n, "original")] >= thr[n]["medium"]
        e = sw_scores[(n, "swap_eg")] >= thr[n]["medium"]
        _, lost, gained, _ = pair(o[sw_y], e[sw_y])
        out.append(f"  {n}: recall drop original -> swap_eg = {100 * drop_eg:.1f} points (lost {lost}, gained {gained}, p = {exact_mcnemar_p(lost, gained):.4f}); "
                   f"original -> swap_net = {100 * drop_net:.1f} points")

    out += ["", "== 3. REAL EGYPTIAN SITES, URL model alone, MEDIUM+ (assumed legitimate; the allowlist is NOT applied here) =="]
    out.append(f"  (a) dataset .eg test URLs, unseen hosts, brand-registry domains excluded (n = {len(eg_rows)} URLs, {len(set(eg_domains))} domains)")
    for n in names:
        med = eg_scores[n] >= thr[n]["medium"]
        flagged_domains = len(set(eg_domains[med]))
        out.append(f"      {n:<22}URLs {rate(int(med.sum()), len(med)):<34}domains {rate(flagged_domains, len(set(eg_domains)))}")
    if external is not None:
        for form in ("apex", "www"):
            sel = (external["form"] == form).to_numpy()
            out.append(f"  (b) Tranco .eg sample absent from the dataset, {form} form ({int(sel.sum())} domains)")
            for n in names:
                med = ext_scores[n] >= thr[n]["medium"]
                bands = "; ".join(f"{band} {int(med[sel & (external['band'] == band).to_numpy()].sum())}/{int((sel & (external['band'] == band).to_numpy()).sum())}"
                                  for band in sorted(external["band"].unique()))
                out.append(f"      {n:<22}{rate(int(med[sel].sum()), int(sel.sum())):<34}by band: {bands}")
    else:
        out.append("  (b) Tranco sample not found; skipped")
    if controls is not None and len(controls):
        out.append(f"  (c) held-out controls, real Egyptian sites (list written from memory; informational; {len(controls)} URLs)")
        for n in names:
            med = ctl_scores[n] >= thr[n]["medium"]
            flagged_hosts = sorted({host_of(u) for u, m in zip(controls["url"], med) if m})
            out.append(f"      {n:<22}{rate(int(med.sum()), len(med)):<34}flagged hosts: {', '.join(flagged_hosts) or 'none'}")

    # ---- pre-declared criteria
    base, mask = names
    rec = {n: flags_med[n][y_te].mean() for n in names}
    fpr = {n: flags_med[n][~y_te].mean() for n in names}
    c1 = (rec[mask] >= rec[base] - C1_MAX_RECALL_LOSS) and (fpr[mask] <= fpr[base] + C1_MAX_FPR_GAIN)
    drop_masked = swap_recall[(mask, "original")] - swap_recall[(mask, "swap_eg")]
    c2 = drop_masked <= C2_MAX_SWAP_DROP
    eg_fpr_dataset = float((eg_scores[mask] >= thr[mask]["medium"]).mean()) if len(eg_rows) else float("nan")
    if external is not None:
        sel = (external["form"] == "apex").to_numpy()
        eg_fpr_tranco = float((ext_scores[mask][sel] >= thr[mask]["medium"]).mean())
    else:
        eg_fpr_tranco = float("nan")
    c3 = (eg_fpr_dataset <= C3_MAX_EG_FPR) and (eg_fpr_tranco <= C3_MAX_EG_FPR)
    out += ["", "== Pre-declared promotion criteria (all three must hold before replacing the live model is even discussed) ==",
            f"  C1 cost: recall {100 * rec[mask]:.1f}% vs {100 * rec[base]:.1f}% (allowed loss {100 * C1_MAX_RECALL_LOSS:.1f} pts), "
            f"FPR {100 * fpr[mask]:.2f}% vs {100 * fpr[base]:.2f}% (allowed gain {100 * C1_MAX_FPR_GAIN:.1f} pts)  -> {'PASS' if c1 else 'FAIL'}",
            f"  C2 benefit: masked recall drop on swap_eg = {100 * drop_masked:.1f} points (allowed <= {100 * C2_MAX_SWAP_DROP:.0f}; deployed model: "
            f"{100 * (swap_recall[(base, 'original')] - swap_recall[(base, 'swap_eg')]):.1f})  -> {'PASS' if c2 else 'FAIL'}",
            f"  C3 side effect: masked FPR on real .eg sites: dataset {100 * eg_fpr_dataset:.1f}%, Tranco apex {100 * eg_fpr_tranco:.1f}% "
            f"(allowed <= {100 * C3_MAX_EG_FPR:.0f}% on both)  -> {'PASS' if c3 else 'FAIL'}",
            f"  OVERALL: {'all three hold' if (c1 and c2 and c3) else 'NOT all three hold'}. This script promotes nothing.",
            "", "Caveats",
            " - The swap test is synthetic and only the LAST host label changes; real .eg phishing may differ (public feeds contain almost none).",
            " - Masking removes the TLD identity, not everything correlated with it (host length, a 'com' second-level label, ...).",
            " - The Egyptian-site sets are small and assumed legitimate; Egyptian .com sites are covered only by the memory-written controls (c).",
            " - Promoting the masked model would invalidate the frozen held-out results (the URL model is part of the system): re-scoring",
            "   every evaluation with ONE consistent model would be required first."]

    report = "\n".join(out)
    print("\n" + report)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved {REPORT_PATH}")


if __name__ == "__main__":
    main()
