"""
train_url_model.py

Step 5 (item 2 of the expansion plan): the first detector in ETIS that can
flag a phishing URL it has never seen. Today URLs are judged by (a) a
blacklist lookup and (b) a few structural rules; on unseen hosts the whole
system scores ~0 (see evaluation with the training-only blacklist).

Four candidates, all trained on the URL TRAIN split only:

    char_lr_full        character 3-5-gram TF-IDF of the whole URL + logistic regression
    char_lr_hostonly    the same on the HOST string only
    lgbm_lex_full       LightGBM on lexical features (lengths, digits, hyphens,
                        entropy, subdomain depth, TLD, path/query features...)
    lgbm_lex_hostonly   LightGBM on the host-level lexical features only
    ensemble_full       mean of char_lr_full and lgbm_lex_full (no fitting)

Why the host-only variants exist: your legitimate URLs are bare Tranco
homepages ("https://example.com") while phishing URLs usually carry a path.
A model could score 99% just by learning "has a path = phishing". The
host-only models cannot see the path, so if they perform similarly the
result is not a path artefact. The report also prints how often each class
has a path, so the shortcut is visible.

Selection: highest VALIDATION ROC-AUC. The decision threshold for each model
is the one that maximises F1 on VALIDATION; it is then applied unchanged to
test. Test results are reported for three views, because they mean different
things:

    all test URLs
    unseen-host      URLs on hosts that never occur in train (group_type = host)
    shared-platform  URLs on big shared hosts (docs.google.com ...), where the
                     host says nothing -- reported separately, mostly one class

Run from the project root:
    python ml/train_url_model.py

Outputs:
    ml/results/url_model_comparison.txt
    ml/results/url_model_test_predictions.csv   (id + each model's score)
"""

import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
from app.engines.rule_based.domain_typosquatting_detector import _split_url, normalize_domain  # noqa: E402

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"
GROUPS_PATH = PROJECT_ROOT / "data" / "processed" / "split_groups.csv"
RESULTS_DIR = PROJECT_ROOT / "ml" / "results"

SEED = 42
N_BOOTSTRAP = 1000
TOP_TLDS = 30
LR_C_GRID = [1.0, 10.0]


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    return -sum(c / len(s) * math.log2(c / len(s)) for c in counts.values())


def parse_url(url: str) -> dict:
    parsed = _split_url(url)
    host = normalize_domain(url)
    path = parsed.path if parsed else ""
    query = parsed.query if parsed else ""
    scheme = parsed.scheme if parsed else ""
    return {"host": host, "path": path, "query": query, "scheme": scheme}


def lexical_features(parts: dict, raw: str) -> dict:
    host, path, query = parts["host"], parts["path"], parts["query"]
    labels = [l for l in host.split(".") if l]
    tld = labels[-1] if labels else ""
    return {
        # --- host-level (visible without any path) ---
        "host_len": len(host),
        "host_dots": host.count("."),
        "subdomain_depth": max(0, len(labels) - 2),
        "host_hyphens": host.count("-"),
        "host_digits": sum(ch.isdigit() for ch in host),
        "host_digit_ratio": sum(ch.isdigit() for ch in host) / max(1, len(host)),
        "host_entropy": _entropy(host),
        "longest_label": max((len(l) for l in labels), default=0),
        "host_is_ip": int(bool(host) and all(p.isdigit() for p in host.split(".")) and len(host.split(".")) == 4),
        "host_has_punycode": int("xn--" in host),
        "has_at": int("@" in raw),
        "tld": tld,
        # --- path/query-level (only in the *_full models) ---
        "url_len": len(raw),
        "has_path": int(len(path) > 1),
        "path_len": len(path),
        "path_depth": path.count("/"),
        "query_len": len(query),
        "num_special": sum(not ch.isalnum() for ch in raw),
        "is_https": int(parts["scheme"] == "https"),
    }


HOST_FEATURES = [
    "host_len", "host_dots", "subdomain_depth", "host_hyphens", "host_digits",
    "host_digit_ratio", "host_entropy", "longest_label", "host_is_ip",
    "host_has_punycode", "has_at",
]
PATH_FEATURES = ["url_len", "has_path", "path_len", "path_depth", "query_len", "num_special", "is_https"]


def build_matrix(feature_df: pd.DataFrame, columns: list[str], tld_categories: list[str]) -> pd.DataFrame:
    X = feature_df[columns].copy()
    tld = feature_df["tld"].where(feature_df["tld"].isin(tld_categories), "other")
    for t in tld_categories + ["other"]:
        X[f"tld_{t}"] = (tld == t).astype(int)
    return X


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(np.quantile(score, np.linspace(0.01, 0.99, 197))):
        pred = score >= t
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def describe(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    pred = score >= threshold
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    out = {
        "n": len(y), "pos": int(y.sum()),
        "recall": tp / max(1, tp + fn) if y.sum() else float("nan"),
        "fpr": fp / max(1, fp + tn) if (y == 0).sum() else float("nan"),
        "precision": tp / max(1, tp + fp) if (tp + fp) else float("nan"),
    }
    if 0 < y.sum() < len(y):
        out["roc_auc"] = roc_auc_score(y, score)
        out["pr_auc"] = average_precision_score(y, score)
    return out


def bootstrap_auc_recall(y, score, threshold, rng) -> dict:
    aucs, recalls = [], []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        yb, sb = y[idx], score[idx]
        if 0 < yb.sum() < len(yb):
            aucs.append(roc_auc_score(yb, sb))
        if yb.sum():
            recalls.append(((sb >= threshold) & (yb == 1)).sum() / yb.sum())
    f = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (float("nan"),) * 2
    return {"roc_auc": f(aucs), "recall": f(recalls)}


def fmt(v: float) -> str:
    return "  n/a " if v != v else f"{v:.4f}"


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    urls = df[df["content_type"] == "url"].copy()
    if GROUPS_PATH.exists():
        groups = pd.read_csv(GROUPS_PATH)[["id", "group_type"]]
        urls = urls.merge(groups, on="id", how="left")
    else:
        urls["group_type"] = "host"
    urls["y"] = (urls["label"] == "phishing").astype(int)

    parts = [parse_url(u) for u in urls["text"]]
    urls["host"] = [p["host"] for p in parts]
    urls["path_part"] = [p["path"] for p in parts]
    feats = pd.DataFrame([lexical_features(p, u) for p, u in zip(parts, urls["text"])], index=urls.index)
    urls["full_text"] = [u.lower().split("://", 1)[-1] for u in urls["text"]]

    tr, va, te = (urls["split"] == "train"), (urls["split"] == "val"), (urls["split"] == "test")
    y_tr, y_va, y_te = urls.loc[tr, "y"].to_numpy(), urls.loc[va, "y"].to_numpy(), urls.loc[te, "y"].to_numpy()

    lines = ["ETIS URL model comparison (host-grouped split)",
             f"train={int(tr.sum()):,}  val={int(va.sum()):,}  test={int(te.sum()):,}  "
             f"(test phishing={int(y_te.sum()):,})", ""]

    # ---- the shortcut check ----
    lines.append("== Dataset shortcut check: share of URLs that carry a path ==")
    for split_name, mask in [("train", tr), ("val", va), ("test", te)]:
        for label in (0, 1):
            m = mask & (urls["y"] == label)
            lines.append(f"  {split_name:<5} {'phishing' if label else 'legitimate':<11}"
                         f"{feats.loc[m, 'has_path'].mean():>8.1%}   (n={int(m.sum()):,})")
    lines.append("  -> if legitimate is ~0% and phishing is high, *_full models can cheat on the path;")
    lines.append("     compare them with the *_hostonly models.")
    lines.append("")

    tld_categories = feats.loc[tr, "tld"].value_counts().head(TOP_TLDS).index.tolist()

    def lr_model(text_col: str):
        vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=200_000)
        Xtr = vec.fit_transform(urls.loc[tr, text_col])
        Xva, Xte = vec.transform(urls.loc[va, text_col]), vec.transform(urls.loc[te, text_col])
        best = max(LR_C_GRID, key=lambda c: roc_auc_score(
            y_va, LogisticRegression(C=c, max_iter=300, class_weight="balanced").fit(Xtr, y_tr).predict_proba(Xva)[:, 1]))
        clf = LogisticRegression(C=best, max_iter=300, class_weight="balanced").fit(Xtr, y_tr)
        return clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1], f"C={best}"

    def lgbm_model(columns: list[str]):
        import lightgbm as lgb
        X = build_matrix(feats, columns, tld_categories)
        clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                 subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbose=-1)
        clf.fit(X[tr.to_numpy()], y_tr)
        return clf.predict_proba(X[va.to_numpy()])[:, 1], clf.predict_proba(X[te.to_numpy()])[:, 1], "400 trees"

    candidates = {
        "char_lr_full": lambda: lr_model("full_text"),
        "char_lr_hostonly": lambda: lr_model("host"),
        "lgbm_lex_full": lambda: lgbm_model(HOST_FEATURES + PATH_FEATURES),
        "lgbm_lex_hostonly": lambda: lgbm_model(HOST_FEATURES),
    }

    val_s, test_s, notes, thresholds = {}, {}, {}, {}
    for name, fn in candidates.items():
        try:
            v, t, note = fn()
        except ImportError as e:
            lines.append(f"[skipped] {name}: {e}")
            continue
        val_s[name], test_s[name], notes[name] = np.asarray(v), np.asarray(t), note
        thresholds[name] = best_f1_threshold(y_va, val_s[name])

    # ---- ensemble candidate: mean of the two full-URL models (no fitting) ----
    # char n-grams read the path better, LightGBM reads host structure better
    if "char_lr_full" in val_s and "lgbm_lex_full" in val_s:
        name = "ensemble_full"
        val_s[name] = (val_s["char_lr_full"] + val_s["lgbm_lex_full"]) / 2
        test_s[name] = (test_s["char_lr_full"] + test_s["lgbm_lex_full"]) / 2
        notes[name] = "mean of char_lr_full + lgbm_lex_full"
        thresholds[name] = best_f1_threshold(y_va, val_s[name])

    lines += ["== Validation (selection + threshold tuning) ==",
              f"{'model':<20}{'val ROC-AUC':>12}{'threshold':>11}  note"]
    for name in val_s:
        lines.append(f"{name:<20}{roc_auc_score(y_va, val_s[name]):>12.4f}{thresholds[name]:>11.3f}  {notes[name]}")
    selected = max(val_s, key=lambda n: roc_auc_score(y_va, val_s[n]))
    lines += ["", f"SELECTED on validation ROC-AUC: {selected}", ""]

    group_te = urls.loc[te, "group_type"].to_numpy()
    has_path_te = feats.loc[te, "has_path"].to_numpy() == 1
    views = [("all test URLs", np.ones(len(y_te), bool)),
             ("UNSEEN-HOST URLs", group_te == "host"),
             ("shared-platform URLs", group_te == "shared_platform"),
             ("UNSEEN-HOST URLs that HAVE a path (path shortcut cannot decide these)",
              (group_te == "host") & has_path_te),
             ("UNSEEN-HOST URLs with NO path (bare host)", (group_te == "host") & ~has_path_te)]
    rng = np.random.default_rng(SEED)
    lines.append("== Test (thresholds from validation) ==")
    for view_name, mask in views:
        lines += ["", f"-- {view_name}  (n={int(mask.sum()):,}, phishing={int(y_te[mask].sum()):,}) --",
                  f"{'model':<20}{'ROC-AUC':>9}{'PR-AUC':>9}{'recall':>9}{'FPR':>9}{'precision':>11}"]
        for name in test_s:
            d = describe(y_te[mask], test_s[name][mask], thresholds[name])
            tag = "  <- selected" if name == selected else ""
            lines.append(f"{name:<20}{fmt(d.get('roc_auc', float('nan'))):>9}{fmt(d.get('pr_auc', float('nan'))):>9}"
                         f"{fmt(d['recall']):>9}{fmt(d['fpr']):>9}{fmt(d['precision']):>11}{tag}")

    mask = views[1][1]
    lines += ["", f"== 95% bootstrap CI, UNSEEN-HOST view ({N_BOOTSTRAP} resamples) =="]
    for name in test_s:
        ci = bootstrap_auc_recall(y_te[mask], test_s[name][mask], thresholds[name], rng)
        lines.append(f"{name:<20}ROC-AUC [{ci['roc_auc'][0]:.4f}, {ci['roc_auc'][1]:.4f}]   "
                     f"recall [{ci['recall'][0]:.4f}, {ci['recall'][1]:.4f}]")

    # ---- operating points at a fixed false-positive rate (chosen on VALIDATION) ----
    # F1-optimal thresholds ignore that real traffic is overwhelmingly legitimate
    # (this dataset is ~70% phishing). For a warning tool the false-positive
    # rate is what users feel, so report recall at fixed low FPR.
    lines += ["", "== Recall at a fixed FPR (threshold from validation legitimate URLs), test =="]
    for view_name, view_mask in (views[1], views[3], views[4]):
        lines += ["", f"-- {view_name} --",
                  f"{'model':<20}{'target FPR':>11}{'threshold':>11}{'test recall':>13}{'test FPR':>10}"]
        for name in test_s:
            for target in (0.01, 0.001):
                thr = float(np.quantile(val_s[name][y_va == 0], 1 - target))
                d = describe(y_te[view_mask], test_s[name][view_mask], thr)
                lines.append(f"{name:<20}{target:>11.3%}{thr:>11.3f}{fmt(d['recall']):>13}{fmt(d['fpr']):>10}")

    lines += ["", "Caveat: the legitimate URLs are Tranco top-20k homepages (popular, short, no path).",
              "A model may partly learn 'is this a famous domain'. Real-world precision on unpopular",
              "legitimate sites is NOT measured here; add long-tail legitimate URLs with paths before",
              "claiming it."]

    report = "\n".join(lines)
    print(report)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "url_model_comparison.txt").write_text(report + "\n", encoding="utf-8")
    pd.DataFrame({"id": urls.loc[te, "id"].to_numpy(), "label": y_te, "group_type": group_te, **test_s}).to_csv(
        RESULTS_DIR / "url_model_test_predictions.csv", index=False)
    print(f"\nSaved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
