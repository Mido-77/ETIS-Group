"""
arabic_keyword_detector.py

Rule-based sub-engine: scores Arabic text for phishing-keyword signals using a
tiered, weighted keyword list rather than flat equal-weight matching.

Design rationale (see docs/rule_based_engine_design.md for full writeup):
- Raw source keyword lists (e.g. the GitHub Arabic-Text-Spam-Detection blacklist)
  contain many generic single words ("account", "number", "without") that are
  extremely common in ordinary Arabic text. Treating these as equal-weight
  phishing signals would cause a high false-positive rate.
- HIGH-weight terms are specific, low-ambiguity phishing phrases (e.g. "your
  card is blocked", "congratulations you won a prize") grounded either in the
  GitHub blacklist's original compound phrases or in real, currently-documented
  Egyptian fraud patterns (CBE, Vodafone Egypt, Instapay, Bank Misr).
- LOW-weight terms are generic words that are only counted toward the score
  if multiple distinct low-weight terms appear together (a clustering signal),
  since any single one of them appearing alone is not meaningful.
"""

import re
from app.core.exceptions import InvalidScanInputError

# ---------------------------------------------------------------------------
# Weighted keyword tiers
# ---------------------------------------------------------------------------

# HIGH: specific, low-ambiguity phishing phrases. Worth the most per match.
HIGH_WEIGHT_TERMS = {
    "بطاقتك محظورة": "card blocked (direct)",
    "حظر بطاقة": "card blocked",
    "الصراف الالي": "ATM reference (combined with block/update context)",
    "قسيمة شرائية مجانا": "free purchase voucher",
    "دخل اضافي بدون ترك عملك": "extra income without leaving your job",
    "ضاعف ثروتك": "double your wealth",
    "ضاعف ثروته": "double your wealth (alt. form)",
    "مبروك لقد فزت": "congratulations you won",
    "فزت بجائزة": "won a prize",
    "تحديث بياناتك البنكية": "update your banking data",
    "كود التحقق": "verification code (OTP) request",  # grounded: CBE/Vodafone/Instapay warnings
    "الرمز السري": "secret PIN request",  # grounded: CBE/Vodafone/Instapay warnings
    "تفعيل حسابك": "activate your account",  # grounded: Instapay documented pattern
    "تفعيل الخدمة": "activate the service",  # grounded: Instapay documented pattern
    "استرجاع المبلغ": "refund/return the amount",  # grounded: Vodafone Cash documented pattern
    "تم تحويل مبلغ": "amount was transferred (mistaken-transfer scam opener)",
}

# MEDIUM: suspicious but more ambiguous on their own.
MEDIUM_WEIGHT_TERMS = {
    "تحديث": "update",
    "الاستثمار": "investment",
    "استثمار": "investment",
    "اتصل بالرقم": "call the number",
    "احصل": "get/obtain",
    "حصلت": "obtained (past tense)",
    "سارع": "hurry",
    "اعلان التوظيف": "job posting",
    "رابط": "link",
}

# LOW: generic words, common in ordinary Arabic text. Only scored if 2+
# distinct low-weight terms appear in the same text (clustering signal).
LOW_WEIGHT_TERMS = {
    "العميل": "the customer",
    "حساب": "account",  # covers "حسابك"/"حسابكم" etc. as substrings -- don't also list those
    "الرقم": "the number",
    "الارقام": "the numbers",
    "الرجاء": "please",
    "الاتصال": "the call/contact",
    "بدون": "without",
    "ترك": "leaving",
    "عملك": "your job",
    "دخل": "income",
    "النت": "the internet",
    "خاص": "private",
    "مجانا": "free",
    "لقد": "(auxiliary word)",
    "فزت": "won",
    "جائزة": "prize",
    "التواصل": "contact",
    "السحب": "withdrawal",
}

HIGH_WEIGHT = 3
MEDIUM_WEIGHT = 2
LOW_WEIGHT = 1
LOW_WEIGHT_MIN_CLUSTER = 2  # low-weight terms only count if this many distinct ones appear

# Raw weighted sum needed to reach a 100 score. Tuned so ~3 high-weight hits
# (a strong, unambiguous phishing message) lands in the 90-100 range.
SCORE_NORMALIZATION_DIVISOR = 10


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

_TASHKEEL_PATTERN = re.compile(r"[\u064B-\u0652\u0670\u0640]")  # diacritics + tatweel


def normalize_arabic_text(text: str) -> str:
    """
    Normalize Arabic text for reliable keyword matching:
    - strip diacritics (tashkeel) and tatweel elongation characters
    - unify alef variants (أ, إ, آ -> ا)
    - unify taa marbuta / haa (ة -> ه) is intentionally NOT done, to avoid
      merging distinct words -- only alef and yaa variants are unified
    - unify alef maqsura / yaa (ى -> ي)
    - collapse repeated whitespace
    """
    normalized = _TASHKEEL_PATTERN.sub("", text)
    normalized = re.sub(r"[أإآ]", "ا", normalized)
    normalized = re.sub(r"ى", "ي", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def calculate_arabic_keyword_phishing_score(text: str) -> dict:
    """
    Score Arabic text for phishing-keyword signals.

    Returns a dict:
        {
            "score": float,               # 0-100
            "matched_high": [(term, meaning), ...],
            "matched_medium": [(term, meaning), ...],
            "matched_low": [(term, meaning), ...],   # only non-empty list; may
                                                       # not have contributed to
                                                       # score if below cluster minimum
            "low_weight_counted": bool,   # whether low-weight terms met the cluster minimum
        }

    Raises InvalidScanInputError if text is not a non-empty string.
    """
    if not isinstance(text, str) or not text.strip():
        raise InvalidScanInputError("Input text must be a non-empty string.")

    normalized = normalize_arabic_text(text)

    matched_high = [(term, meaning) for term, meaning in HIGH_WEIGHT_TERMS.items() if term in normalized]
    matched_medium = [(term, meaning) for term, meaning in MEDIUM_WEIGHT_TERMS.items() if term in normalized]
    matched_low = [(term, meaning) for term, meaning in LOW_WEIGHT_TERMS.items() if term in normalized]

    raw_score = len(matched_high) * HIGH_WEIGHT + len(matched_medium) * MEDIUM_WEIGHT

    low_weight_counted = len(matched_low) >= LOW_WEIGHT_MIN_CLUSTER
    if low_weight_counted:
        raw_score += len(matched_low) * LOW_WEIGHT

    score = min(100.0, round(100 * raw_score / SCORE_NORMALIZATION_DIVISOR, 2))

    return {
        "score": score,
        "matched_high": matched_high,
        "matched_medium": matched_medium,
        "matched_low": matched_low,
        "low_weight_counted": low_weight_counted,
    }
