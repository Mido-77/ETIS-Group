/**
 * Mirrors app/routers/analyze.py, threats.py, stats.py and the
 * ScanResult/BlacklistEntry models, corrected against real responses
 * from scripts/smoke-test-api.mjs (see comments below for what's
 * confirmed vs. still unverified).
 */

// ---- Error responses ----
// Confirmed from main.py's global exception handler: any ETISException
// subclass (InvalidScanInputError, BlacklistLookupError, EngineFailureError,
// or an unmapped one falling back to 500) returns this shape. A raw,
// non-ETISException error (e.g. a Pydantic validation 422) would NOT match
// this — FastAPI's default handler uses {"detail": ...} with no "error" key.

export interface EtisErrorResponse {
  error: string; // exception class name, e.g. "InvalidScanInputError"
  detail: string;
}

// ---- POST /analyze ----

export interface AnalyzeRequest {
  text?: string;
  url?: string;
}

// --- rule_based engine ---

export interface ArabicKeywordMatch {
  score: number;
  // Each match is a [arabic_term, english_gloss] pair, e.g. ["تحديث", "update"]
  matched_high: [string, string][];
  matched_medium: [string, string][];
  matched_low: [string, string][];
  low_weight_counted: boolean;
}

// Added alongside backend/app/engines/rule_based/english_keyword_detector.py
// (generic English scam-keyword coverage). Same shape as ArabicKeywordMatch,
// but matched pairs are [english_term, meaning] instead of [arabic, gloss].
export interface EnglishKeywordMatch {
  score: number;
  matched_high: [string, string][];
  matched_medium: [string, string][];
  matched_low: [string, string][];
  low_weight_counted: boolean;
}

export interface DomainTyposquattingResult {
  score: number;
  matched_brand: string | null;
  matched_brand_domain: string | null;
  technique: string | null;
  normalized_domain: string;
}

// A flag is [flag_name, weight, detail]. Confirmed both forms from real
// data: detail is a string for path/keyword-based flags (e.g.
// ["suspicious_path_keyword", 8, "/login"]), but null for flags with no
// specific matched substring (e.g. ["no_https", 8, null]).
export type UrlPatternFlag = [string, number, string | null];

export interface UrlPatternResult {
  score: number;
  flags: UrlPatternFlag[];
}

export interface RuleBasedEngineResults {
  // Each sub-check is null when it doesn't apply to the given input
  // (e.g. arabic_keyword/english_keyword are null for a url-only
  // submission; domain_typosquatting/url_pattern are null for a
  // text-only submission).
  arabic_keyword: ArabicKeywordMatch | null;
  english_keyword: EnglishKeywordMatch | null;
  domain_typosquatting: DomainTyposquattingResult | null;
  url_pattern: UrlPatternResult | null;
}

export interface RuleBasedDetail {
  score: number;
  engine_results: RuleBasedEngineResults;
}

// --- blacklist engine ---

// Confirmed from blacklist_lookup.py source (not just observed data).

export interface BlacklistExactMatch {
  url: string;
  source: string;
  brand_target: string | null;
  first_seen: string;
  last_seen: string;
}

export interface BlacklistDomainMatch {
  domain: string;
  hit_count: number;
  most_recent_source: string;
  most_recent_last_seen: string;
}

export type BlacklistMatchType = 'exact' | 'domain' | null;

export interface BlacklistDetail {
  score: number;
  match_type: BlacklistMatchType;
  // Shape depends on match_type: BlacklistExactMatch when 'exact',
  // BlacklistDomainMatch when 'domain', null when match_type is null.
  matched_entry: BlacklistExactMatch | BlacklistDomainMatch | null;
}

// --- learned engines (decision-tier scorer, 2026-10) ---

export type Verdict = 'LOW' | 'MEDIUM' | 'HIGH';

// Probability thresholds at which the engine's own tier changes, chosen on
// the validation split at fixed false-positive rates (see risk_scoring.py).
export interface ModelThresholds {
  medium: number;
  high: number;
}

// Learned URL ensemble (character n-gram LogReg + LightGBM, mean of the two).
export interface UrlModelDetail {
  score: number; // probability * 100
  probability: number; // 0-1
  tier: Verdict;
  components: { char_logreg: number; lightgbm: number };
  thresholds: ModelThresholds;
}

// Learned message-text classifier (character TF-IDF + logistic regression).
export interface TextModelDetail {
  score: number; // probability * 100
  probability: number; // 0-1
  tier: Verdict;
  thresholds: ModelThresholds;
}

export interface EngineDetails {
  // Always present — analyze.py accesses this without a null check.
  rule_based: RuleBasedDetail;
  // Null when no URL was submitted (blacklist engine doesn't apply).
  blacklist: BlacklistDetail | null;
  // Null when no URL was submitted (the URL model needs a URL).
  url_model: UrlModelDetail | null;
  // Null when no text was submitted (the text model needs text).
  text_model: TextModelDetail | null;
}

// Confirmed from risk_scoring.py: a sibling to engine_details giving a
// stable, machine-readable reason for why an engine didn't run.
// rule_based always runs given the request's input-presence
// precondition, so it has no corresponding key here. Each field is null
// when that engine ran normally.
export type SkipReasonCode = 'no_url_provided' | 'no_text_provided';

export interface SkipReasons {
  blacklist: SkipReasonCode | null;
  url_model: SkipReasonCode | null;
  text_model: SkipReasonCode | null;
}

// Confirmed from risk_scoring.py: per-engine wall-clock timing in
// milliseconds, plus a total covering the full aggregation (all
// applicable engine calls + the tier step). Engines that didn't apply are
// null, mirroring skip_reasons/engine_details. Presentation/explainability
// data only — not the same as the HTTP request's total latency.
export interface StageTimings {
  rule_based_ms: number;
  blacklist_ms: number | null;
  url_model_ms: number | null;
  text_model_ms: number | null;
  total_ms: number;
}

// Phase 3 explainability addition (2026-09), kept for API compatibility
// but no longer displayed (with one deciding engine, low "agreement" with
// engines that have nothing to say reads as doubt, which it is not). A
// heuristic "how much independent evidence backed this result" signal —
// NOT a calibrated statistical confidence. Computed server-side from the applicable
// engines' raw 0-100 scores, before point-architecture weighting and
// before the Egyptian brand allowlist override, so it reflects genuine
// engine signal rather than the final overridden verdict. Purely
// additive: never derived from, and never affects, final_score/verdict/
// engine_details. See risk_scoring.py's "Confidence score" docstring
// note for the full formula.
export interface Confidence {
  // 0-100 headline value: round(0.5 * coverage + 0.5 * agreement, 2).
  // The 50/50 weighting is a stated design choice, not a calibrated one.
  score: number;
  // 0-100. engines_considered / 4 * 100 — 75 (url-only), 50 (text-only)
  // or 100 (both provided).
  coverage: number;
  // 0-100. 100 - (max(applicable engine scores) - min(applicable engine
  // scores)). Lower when one engine flags strongly and another finds
  // nothing (e.g. a strong rule-based typosquat signal with no
  // corroborating blacklist match — see thesis Section 5.8.2).
  agreement: number;
  // How many engines' raw scores fed this calculation. Between 2 and 4.
  engines_considered: number;
}

export interface AnalyzeResponse {
  final_score: number;
  verdict: Verdict;
  engines_applied: string[];
  // Tier of each applicable engine, after the Egyptian-brand allowlist
  // override. The verdict is the highest tier any engine reaches.
  engine_tiers: Record<string, Verdict>;
  // The engine that produced the verdict; null when the verdict is LOW.
  decided_by: string | null;
  engine_details: EngineDetails;
  // Machine-readable reason for each engine that didn't apply to this
  // submission — lets the UI render a real explanation instead of a
  // generic "not applicable" label. See risk_scoring.py's "Skip reasons"
  // docstring note.
  skip_reasons: SkipReasons;
  // Per-engine and total processing time in milliseconds, for
  // explainability/presentation purposes. See risk_scoring.py's "Stage
  // timings" docstring note.
  stage_timings: StageTimings;
  // Heuristic "how much evidence backed this result" signal. See the
  // Confidence interface above and risk_scoring.py's docstring note.
  confidence: Confidence;
  // True when the submitted URL is a verified Egyptian brand domain (or a
  // genuine subdomain). The URL-derived engines' tiers are set to LOW;
  // evidence from the message text is NOT cancelled. engine_details still
  // reflects each engine's real, un-overridden output.
  allowlisted: boolean;
  saved: boolean;
  save_error: string | null;
}

// ---- GET /threats ----

export interface ThreatEntry {
  id: number;
  domain: string;
  source: string;
  brand_target: string | null;
  reputation_score: number;
  is_active: boolean;
  first_seen: string; // ISO date, e.g. "2026-07-11"
  last_seen: string;
}

export interface ThreatsQueryParams {
  source?: string;
  is_active?: boolean;
  limit?: number; // backend default 50, max 500
  offset?: number; // backend default 0
}

export interface ThreatsResponse {
  total_matching: number;
  limit: number;
  offset: number;
  results: ThreatEntry[];
}

// ---- GET /stats ----
// Confirmed against a real response — matches _blacklist_stats()/
// _scan_stats() exactly. Index signatures kept since stats.py's own
// docstring says this is deliberately meant to grow over time.

export interface BlacklistStats {
  total_entries: number;
  active_entries: number;
  by_source: Record<string, number>;
  [key: string]: unknown;
}

export interface ScanStats {
  total_scans: number;
  by_verdict: Record<Verdict, number>;
  average_final_score: number | null;
  [key: string]: unknown;
}

export interface StatsResponse {
  blacklist: BlacklistStats;
  scans: ScanStats;
  [key: string]: unknown;
}
