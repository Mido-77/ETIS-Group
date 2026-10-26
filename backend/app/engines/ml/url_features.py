"""
url_features.py

Feature extraction for the learned URL model. Shared by ml/export_models.py
(training the deployed artifact) and url_model_inference.py (serving), so a
URL is turned into features by exactly one piece of code.

The logic is identical to the evaluated version in ml/train_url_model.py
(which produced the numbers reported in Chapter 5). Do not change it without
re-running that evaluation: export_models.py --selfcheck compares the served
scores with the saved evaluation scores and fails if they drift.

mask_tld_in_text() was added for the v2 (TLD-masked) URL model, which reads the
host WITHOUT its last label. char_model_text() itself is unchanged, so v1
artifacts score exactly as before. See ml/ablation_url_tld_mask.py for why.
"""

import math
import re
from collections import Counter

import pandas as pd

from app.engines.rule_based.domain_typosquatting_detector import _split_url, normalize_domain

TOP_TLDS = 30

HOST_FEATURES = [
    "host_len", "host_dots", "subdomain_depth", "host_hyphens", "host_digits",
    "host_digit_ratio", "host_entropy", "longest_label", "host_is_ip",
    "host_has_punycode", "has_at",
]
PATH_FEATURES = ["url_len", "has_path", "path_len", "path_depth", "query_len", "num_special", "is_https"]


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    return -sum(c / len(s) * math.log2(c / len(s)) for c in counts.values())


def parse_url(url: str) -> dict:
    parsed = _split_url(url)
    return {
        "host": normalize_domain(url),
        "path": parsed.path if parsed else "",
        "query": parsed.query if parsed else "",
        "scheme": parsed.scheme if parsed else "",
    }


def lexical_features(parts: dict, raw: str) -> dict:
    host, path, query = parts["host"], parts["path"], parts["query"]
    labels = [l for l in host.split(".") if l]
    tld = labels[-1] if labels else ""
    return {
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
        "url_len": len(raw),
        "has_path": int(len(path) > 1),
        "path_len": len(path),
        "path_depth": path.count("/"),
        "query_len": len(query),
        "num_special": sum(not ch.isalnum() for ch in raw),
        "is_https": int(parts["scheme"] == "https"),
    }


def feature_frame(urls: list[str]) -> pd.DataFrame:
    return pd.DataFrame([lexical_features(parse_url(u), u) for u in urls])


def char_model_text(url: str) -> str:
    """Text fed to the character n-gram model: lowercase, scheme removed."""
    return url.lower().split("://", 1)[-1]


def mask_tld_in_text(text: str) -> str:
    """Replace the last host label by 'tld' in char_model_text() output.

    text is lowercase with the scheme removed, e.g. "user@host.com.eg:8080/p?q".
    Userinfo, port, path, query and IP addresses are left alone, so
    "x.example.top/a" and "x.example.eg/a" become identical. Used only by
    artifacts trained with tld_masked=True.
    """
    match = re.match(r"^([^/?#]*)(.*)$", text, flags=re.S)
    authority, rest = match.group(1), match.group(2)
    userinfo, sep, hostport = authority.rpartition("@")
    port_match = re.match(r"^(.*?)(:\d+)?$", hostport)
    host, port = port_match.group(1).rstrip("."), port_match.group(2) or ""
    if "." in host and not re.fullmatch(r"[\d.]+", host):
        host = host.rsplit(".", 1)[0] + ".tld"
    return userinfo + sep + host + port + rest


def build_matrix(feature_df: pd.DataFrame, columns: list[str], tld_categories: list[str]) -> pd.DataFrame:
    X = feature_df[columns].copy()
    tld = feature_df["tld"].where(feature_df["tld"].isin(tld_categories), "other")
    for t in tld_categories + ["other"]:
        X[f"tld_{t}"] = (tld == t).astype(int)
    return X
