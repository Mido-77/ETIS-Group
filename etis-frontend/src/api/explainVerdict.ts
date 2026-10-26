import type { TFunction } from 'i18next';

import type { AnalyzeResponse } from './types';

/**
 * Turns the raw engine breakdown into a short, human-readable reason,
 * so a verdict badge never appears alone. This directly addresses the
 * calibration finding that a low-looking score (e.g. 3.43, 37.71) can
 * legitimately carry a MEDIUM/HIGH verdict -- without a reason attached,
 * that reads as a bug to anyone unfamiliar with the calibration story.
 *
 * Takes `t` explicitly rather than calling useTranslation() internally --
 * this is a plain function (usable outside component render, easier to
 * reason about/test), and it keeps this file consistent with the rest of
 * api/ being framework-agnostic. Every string here goes through i18n so
 * the explanation actually respects the language toggle, not just the
 * static page labels around it.
 *
 * Checks BOTH arabic_keyword and english_keyword matches (added
 * alongside english_keyword_detector.py).
 *
 * Decision-tier scorer (2026-10): the learned URL and text models report
 * their own tier; a model is only cited as a reason when its tier is
 * MEDIUM or HIGH, i.e. when it actually contributed to the verdict.
 *
 * Allowlist override handled FIRST and returned early (added alongside
 * risk_scoring.py's Egyptian-brand allowlist override): when
 * result.allowlisted is true, engine_details still contain the engines'
 * real, un-overridden output -- including domain_typosquatting, whose
 * "exact_match" technique still populates matched_brand even at score
 * 0.0 (it IS the real brand domain, not merely resembling it). Falling
 * through to the normal reason-building logic would misleadingly say
 * "domain closely resembles the brand X" and list URL pattern flags
 * right next to a "verified" badge. An allowlisted result gets its own
 * dedicated, honest explanation instead.
 */
export function explainVerdict(t: TFunction, result: AnalyzeResponse): string {
  if (result.allowlisted) {
    return t('scan.reasonAllowlisted');
  }

  const { rule_based, blacklist, url_model, text_model } = result.engine_details;
  const reasons: string[] = [];

  if (blacklist?.match_type === 'exact') {
    reasons.push(t('scan.reasonBlacklistExact'));
  } else if (blacklist?.match_type === 'domain' && blacklist.matched_entry && 'hit_count' in blacklist.matched_entry) {
    const entry = blacklist.matched_entry as { hit_count: number };
    reasons.push(t('scan.reasonBlacklistDomain', { count: entry.hit_count }));
  }

  if (url_model && url_model.tier !== 'LOW') {
    reasons.push(
      t(url_model.tier === 'HIGH' ? 'scan.reasonUrlModelHigh' : 'scan.reasonUrlModelMedium', {
        percent: Math.round(url_model.probability * 100),
      }),
    );
  }

  if (text_model && text_model.tier !== 'LOW') {
    reasons.push(t('scan.reasonTextModel', { percent: Math.round(text_model.probability * 100) }));
  }

  const arabicKeywordMatch = rule_based.engine_results.arabic_keyword;
  const englishKeywordMatch = rule_based.engine_results.english_keyword;

  const arabicTerms =
    arabicKeywordMatch && (arabicKeywordMatch.matched_high.length > 0 || arabicKeywordMatch.matched_medium.length > 0)
      ? [...arabicKeywordMatch.matched_high, ...arabicKeywordMatch.matched_medium]
      : [];
  const englishTerms =
    englishKeywordMatch && (englishKeywordMatch.matched_high.length > 0 || englishKeywordMatch.matched_medium.length > 0)
      ? [...englishKeywordMatch.matched_high, ...englishKeywordMatch.matched_medium]
      : [];

  if (arabicTerms.length > 0 || englishTerms.length > 0) {
    const terms = [...arabicTerms, ...englishTerms].map(([term]) => term).join('، ');
    reasons.push(t('scan.reasonKeywordMatch', { terms }));
  }

  // Only report typosquatting when the technique actually indicates a
  // typosquat, not an exact match on the real domain -- an "exact_match"
  // result (score 0.0) means this IS the verified brand's own domain,
  // and would only reach here for a URL NOT covered by the allowlist
  // (e.g. matched via the rule-based engine's brand registry but not yet
  // in EGYPTIAN_BRAND_DOMAIN_SET), so it's still worth guarding against
  // a misleading "resembles the brand" message in that edge case too.
  const typosquat = rule_based.engine_results.domain_typosquatting;
  if (typosquat?.matched_brand && typosquat.technique !== 'exact_match') {
    reasons.push(t('scan.reasonTyposquat', { brand: typosquat.matched_brand }));
  }

  const urlPattern = rule_based.engine_results.url_pattern;
  if (urlPattern && urlPattern.flags.length > 0) {
    reasons.push(t('scan.reasonUrlFlags', { count: urlPattern.flags.length }));
  }

  if (reasons.length === 0) {
    reasons.push(result.verdict === 'LOW' ? t('scan.reasonNoIndicators') : t('scan.reasonLowConfidence'));
  }

  return reasons.join(' ');
}
