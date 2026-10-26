"""
diagnose_url_input_variants.py

How much does the URL model's score depend on HOW a URL is written, not on
which site it is? Scores the same well-known legitimate sites in four writings:

    bare          wikipedia.org
    https         https://wikipedia.org
    https_www     https://www.wikipedia.org
    http_www      http://www.wikipedia.org

and prints the raw model probability of each, WITHOUT the serving-time scheme
normalisation, so it measures the model itself. Large gaps between 'https' and
'https_www' mean the model learned "www." as a legitimacy signal (13.7% of
legitimate training URLs contain it, against 4.1% of phishing URLs) -- a
shortcut, not evidence about the site.

Run from the project root (venv active):
    python ml/diagnose_url_input_variants.py
Output: ml/results/url_input_variants.txt
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.engines.ml.url_features import build_matrix, char_model_text, feature_frame  # noqa: E402
from app.engines.ml.url_model_inference import _load_artifact  # noqa: E402

SITES = [
    "wikipedia.org", "github.com", "bbc.com", "mozilla.org", "python.org",
    "stackoverflow.com", "reuters.com", "nytimes.com", "ahram.org.eg",
    "youm7.com", "sis.gov.eg", "cbe.org.eg", "egx.com.eg", "alexu.edu.eg",
    "microsoft.com", "apple.com", "linkedin.com", "who.int",
]
WRITINGS = {
    "bare": "{d}",
    "https": "https://{d}",
    "https_www": "https://www.{d}",
    "http_www": "http://www.{d}",
}


def raw_probability(url: str) -> float:
    artifact = _load_artifact()
    p_char = float(artifact["char_logreg"].predict_proba(
        artifact["vectorizer"].transform([char_model_text(url)]))[0, 1])
    features = build_matrix(feature_frame([url]), artifact["lgbm_columns"], artifact["tld_categories"])
    p_lgbm = float(artifact["lightgbm"].predict_proba(features)[0, 1])
    return (p_char + p_lgbm) / 2


def main() -> None:
    thresholds = _load_artifact()["thresholds"]
    lines = [f"URL model probability by writing style (MEDIUM threshold {thresholds['medium']:.3f}, "
             f"HIGH {thresholds['high']:.3f}). Legitimate sites only; lower is better.",
             f"{'site':<22}" + "".join(f"{name:>11}" for name in WRITINGS)]
    sums = {name: 0.0 for name in WRITINGS}
    over = {name: 0 for name in WRITINGS}
    for site in SITES:
        probabilities = {name: raw_probability(template.format(d=site)) for name, template in WRITINGS.items()}
        for name, p in probabilities.items():
            sums[name] += p
            over[name] += p >= thresholds["medium"]
        lines.append(f"{site:<22}" + "".join(f"{p:>11.3f}" for p in probabilities.values()))
    lines.append(f"{'MEAN':<22}" + "".join(f"{sums[n] / len(SITES):>11.3f}" for n in WRITINGS))
    lines.append(f"{'flagged >= MEDIUM':<22}" + "".join(f"{over[n]:>9d}/{len(SITES)}" for n in WRITINGS))
    report = "\n".join(lines)
    print(report)
    out = PROJECT_ROOT / "ml" / "results" / "url_input_variants.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
