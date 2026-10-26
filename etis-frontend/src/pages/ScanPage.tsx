import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Box,
  TextField,
  Button,
  Typography,
  CircularProgress,
  Alert,
  Stack,
  Chip,
} from '@mui/material';
import FactCheckOutlined from '@mui/icons-material/FactCheckOutlined';
import StorageOutlined from '@mui/icons-material/StorageOutlined';
import PsychologyOutlined from '@mui/icons-material/PsychologyOutlined';
import LinkOutlined from '@mui/icons-material/LinkOutlined';

import { analyzeScan, ApiError } from '../api/client';
import { LocalizedError } from '../api/errorMessage';
import type { AnalyzeResponse, BlacklistDomainMatch, BlacklistExactMatch, SkipReasonCode } from '../api/types';
import { VerdictCard } from '../components/VerdictCard';
import { EngineBreakdownCard } from '../components/EngineBreakdownCard';
import { ModelScoreSummary } from '../components/ModelScoreSummary';
import { AutoResizingTextArea } from '../components/AutoResizingTextArea';
import { ErrorAlert } from '../components/ErrorAlert';

// Matches the first http(s) URL substring in free text. Used so a
// text-only paste (no URL entered separately) can still reach the URL
// engines (blacklist + learned URL model) -- text-only submissions only
// run rule-based + the text model. Auto-detecting a link in the pasted
// text and submitting it as the url moves the request into the full
// four-engine "both" case instead, and the verdict is the highest tier
// any engine reaches.
const URL_REGEX = /https?:\/\/[^\s<>"')]+/;

function extractFirstUrl(text: string): string | null {
  const match = text.match(URL_REGEX);
  return match ? match[0] : null;
}

/** Maps a backend skip-reason code to its i18n key. Null input -> null
 * (the engine ran, EngineBreakdownCard's `applicable` handles that case).
 * See risk_scoring.py's skip_reasons field. */
function skipReasonTranslationKey(code: SkipReasonCode | null): string | null {
  if (code === 'no_url_provided') return 'scan.skipReasonNoUrl';
  if (code === 'no_text_provided') return 'scan.skipReasonNoText';
  return null;
}

// Width of the input column -- also used as the hard ceiling for the
// message box's "unbreakable long content" overflow case, so it can
// never grow wide enough to overlap the report column beside it.
const INPUT_COLUMN_WIDTH = 520;
const FORM_ID = 'scan-form';

/**
 * Two-column layout: the input column (URL + message text) on one side,
 * and a sticky side column (Analyze button pinned at the top, then the
 * verdict/report below it) on the other -- so the report stays visible
 * and the Analyze button stays put even as the message box grows tall.
 * The Analyze button lives outside the <form> DOM-wise, associated via
 * the standard HTML `form` attribute instead, since it sits in the other
 * column.
 *
 * Which side the report column renders on isn't decided here -- the
 * input column is placed first in JSX order and the report column
 * second, and the browser's own flex layout resolves "first/second" into
 * "left/right" against the page's inherited direction: English (LTR)
 * pages put the report on the right, Arabic (RTL) pages put it on the
 * left, automatically.
 */
export function ScanPage() {
  const { t } = useTranslation();
  const [url, setUrl] = useState('');
  const [text, setText] = useState('');
  // True only while the URL field's current content was written by our own
  // auto-detection (or the browser extension's handoff), not typed by the
  // user. Lets a later submission with different text safely re-detect or
  // clear the field, without ever overwriting something the user
  // deliberately entered themselves.
  const [urlAutoFilled, setUrlAutoFilled] = useState(false);
  // Drives the URL field's label swap below: empty label at rest (so the
  // native placeholder "Paste URL" shows through), "URL" once focused.
  const [urlFocused, setUrlFocused] = useState(false);
  const [loading, setLoading] = useState(false);
  // Kept as the error object (not a string) so ErrorAlert translates it at render time.
  const [error, setError] = useState<unknown>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);

  // Shared by both the manual "Analyze" button and the extension handoff
  // auto-run below. Takes rawUrl/rawText/autoFilled as explicit arguments
  // rather than reading the url/text/urlAutoFilled state directly, so the
  // handoff effect can call this immediately after setting state without
  // racing React's asynchronous state updates.
  async function submitScan(rawUrl: string, rawText: string, autoFilled: boolean) {
    const trimmedUrl = rawUrl.trim();
    const trimmedText = rawText.trim();

    if (!trimmedUrl && !trimmedText) {
      setError(new LocalizedError('scan.errorEmptyInput'));
      return;
    }

    // Re-detect a URL from the pasted text whenever the field is empty, OR
    // it still holds a value WE auto-filled (text-regex or the extension's
    // own link extraction), not something the user typed -- otherwise a
    // stale link from an earlier message would silently carry over onto a
    // new, unrelated one.
    let effectiveUrl = trimmedUrl;
    if (!trimmedUrl || autoFilled) {
      const detected = trimmedText ? extractFirstUrl(trimmedText) : null;
      if (detected) {
        effectiveUrl = detected;
        setUrl(detected);
        setUrlAutoFilled(true);
      } else if (autoFilled) {
        effectiveUrl = '';
        setUrl('');
        setUrlAutoFilled(false);
      }
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await analyzeScan({
        url: effectiveUrl || undefined,
        text: trimmedText || undefined,
      });
      setResult(response);
    } catch (err) {
      if (!(err instanceof ApiError)) console.error(err);
      setError(err);
    } finally {
      setLoading(false);
    }
  }

  // Reads a handoff payload passed via query params by the ETIS browser
  // extension's "Open full report on website" button (?etis_text=...,
  // optional &etis_url=...). Pre-fills the form AND immediately runs the
  // analysis, then clears the params from the address bar so a page
  // refresh doesn't keep re-submitting and the URL bar doesn't sit there
  // showing raw email content.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const handoffText = params.get('etis_text') || '';
    const handoffUrl = params.get('etis_url') || '';

    if (handoffText || handoffUrl) {
      setText(handoffText);
      if (handoffUrl) {
        setUrl(handoffUrl);
        setUrlAutoFilled(true);
      }

      window.history.replaceState({}, '', window.location.pathname);

      // Use the raw handoff values directly rather than the state just
      // set above -- state updates aren't synchronous, so reading
      // url/text here could still see their pre-handoff (empty) values.
      void submitScan(handoffUrl, handoffText, Boolean(handoffUrl));
    }
    // Intentionally run once on mount only -- this reads the initial URL
    // the page was opened with, not something that should re-run later.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleUrlChange(e: React.ChangeEvent<HTMLInputElement>) {
    setUrl(e.target.value);
    setUrlAutoFilled(false); // any manual edit, including clearing it, is now the user's own input
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    await submitScan(url, text, urlAutoFilled);
  }

  return (
    <Box sx={{ py: 4 }}>
      <Typography variant="h5" gutterBottom>
        {t('nav.scan')}
      </Typography>

      <Box
        sx={{
          display: 'flex',
          flexDirection: { xs: 'column', md: 'row' },
          alignItems: 'flex-start',
          gap: 3,
          mt: 2,
        }}
      >
        {/* Input column -- URL + message text only. Placed first in JSX
            order; the browser's flex layout resolves this to the
            direction-appropriate side (see component doc comment). */}
        <Box
          component="form"
          id={FORM_ID}
          onSubmit={handleSubmit}
          sx={{ width: '100%', maxWidth: INPUT_COLUMN_WIDTH, flexShrink: 0 }}
        >
          <Stack spacing={2}>
            <TextField
              // MUI's label can't morph from one text to another mid-
              // animation -- it only animates position/size of a single
              // string. So at rest the label is empty (letting the native
              // placeholder "Paste URL" show through normally), and the
              // instant the field is focused, the label becomes "URL" and
              // is forced into its shrunk position -- MUI's border-notch +
              // label-fade transition still animates that swap.
              label={urlFocused || url ? t('scan.urlLabel') : ''}
              value={url}
              onChange={handleUrlChange}
              onFocus={() => setUrlFocused(true)}
              onBlur={() => setUrlFocused(false)}
              fullWidth
              placeholder={t('scan.urlPlaceholder')}
              // A URL is always left-to-right text. In the RTL page the browser's
              // bidi algorithm scrambled a bare "https://" into "//:https", so the
              // input itself is dir="ltr". Alignment still follows the page: "end"
              // is the right edge in an LTR input (RTL page), "start" the left. The
              // keywords, not left/right, because the RTL emotion cache would flip
              // a literal text-align: right back to the left.
              slotProps={{
                inputLabel: { shrink: urlFocused || Boolean(url) },
                htmlInput: { dir: 'ltr' },
              }}
              sx={(theme) => ({
                '& .MuiOutlinedInput-root': { backgroundColor: '#fff' },
                '& input': { textAlign: theme.direction === 'rtl' ? 'end' : 'start' },
              })}
            />
            <AutoResizingTextArea
              label={t('scan.textLabel')}
              value={text}
              onChange={setText}
              placeholder={t('scan.textPlaceholder')}
              minRows={3}
              maxWidthPx={INPUT_COLUMN_WIDTH}
            />
          </Stack>
        </Box>

        {/* Report column -- Analyze button pinned at the top, then the
            report below it. Sticky on desktop so it stays in view as the
            input column grows taller; a plain static block on narrow
            screens, where it naturally stacks below the inputs instead. */}
        <Box
          sx={{
            width: { xs: '100%', md: 'auto' },
            flex: { md: 1 },
            minWidth: 0,
            position: { xs: 'static', md: 'sticky' },
            top: { md: 24 },
          }}
        >
          <Button
            type="submit"
            form={FORM_ID}
            variant="contained"
            fullWidth
            disabled={loading}
          >
            {loading ? <CircularProgress size={20} /> : t('scan.submit')}
          </Button>

          {error != null && <ErrorAlert error={error} sx={{ mt: 2 }} />}

          {result && <ScanResultView result={result} />}
        </Box>
      </Box>
    </Box>
  );
}

function ScanResultView({ result }: { result: AnalyzeResponse }) {
  const { t } = useTranslation();
  const { rule_based, blacklist, url_model, text_model } = result.engine_details;
  const tiers = result.engine_tiers;
  const timings = result.stage_timings;
  const blacklistSkipReasonKey = skipReasonTranslationKey(result.skip_reasons.blacklist);
  const urlModelSkipReasonKey = skipReasonTranslationKey(result.skip_reasons.url_model);
  const textModelSkipReasonKey = skipReasonTranslationKey(result.skip_reasons.text_model);

  const keywordMatch = rule_based.engine_results.arabic_keyword;
  const englishMatch = rule_based.engine_results.english_keyword;
  // Both keyword engines run on every text submission (each simply finds
  // nothing in the other language), so chips from both are shown together.
  const keywordGroups = [
    { prefix: 'ar', match: keywordMatch },
    { prefix: 'en', match: englishMatch },
  ];
  const hasAnyKeywordMatch = keywordGroups.some(
    ({ match }) =>
      match &&
      match.matched_high.length + match.matched_medium.length + match.matched_low.length > 0,
  );
  const typosquat = rule_based.engine_results.domain_typosquatting;
  const urlPattern = rule_based.engine_results.url_pattern;
  // "Impersonates: <brand>" only makes sense for a look-alike host. The
  // detector reports matched_brand even for exact_match (the host IS the
  // brand's real domain, score 0), and the allowlist marks verified brand
  // hosts, so neither case may print an accusation next to a genuine site.
  const showImpersonation =
    Boolean(typosquat?.matched_brand) &&
    typosquat?.technique !== 'exact_match' &&
    typosquat?.technique !== 'brand_subdomain' &&
    !result.allowlisted;

  return (
    <Box
      sx={{
        mt: 2,
        display: 'grid',
        // Two columns whenever the report column is at least ~576px wide
        // (min track = the larger of 280px and half the width), otherwise
        // a single column. Never three, even on very wide screens. Each
        // row is as tall as its tallest block, and blocks stretch to fill
        // it, so pairs stay aligned whatever their text length.
        gridTemplateColumns: 'repeat(auto-fit, minmax(max(280px, calc((100% - 16px) / 2)), 1fr))',
        gap: 2,
        alignItems: 'stretch',
      }}
    >
      {!result.saved && (
        <Alert severity="warning" sx={{ gridColumn: '1 / -1' }}>
          {result.save_error}
        </Alert>
      )}

      <Box sx={{ '& > *': { height: '100%' } }}>
        <VerdictCard result={result} />
      </Box>

      <EngineBreakdownCard
        icon={<FactCheckOutlined fontSize="small" color="action" />}
        title={t('scan.engineRuleBased')}
        applicable
        score={rule_based.score}
        tier={tiers.rule_based}
        timingMs={timings.rule_based_ms}
      >
        <Stack spacing={1.5}>
          <Box>
            {hasAnyKeywordMatch ? (
              <Stack direction="row" spacing={0.5} sx={{ flexWrap: 'wrap', gap: 0.5 }}>
                {keywordGroups.flatMap(({ prefix, match }) =>
                  !match
                    ? []
                    : [
                        ...match.matched_high.map(([term]) => (
                          <Chip key={`${prefix}-h-${term}`} size="small" color="error" label={term} />
                        )),
                        ...match.matched_medium.map(([term]) => (
                          <Chip key={`${prefix}-m-${term}`} size="small" color="warning" label={term} />
                        )),
                        ...match.matched_low.map(([term]) => (
                          <Chip key={`${prefix}-l-${term}`} size="small" variant="outlined" label={term} />
                        )),
                      ],
                )}
              </Stack>
            ) : (
              <Typography variant="body2" color="text.secondary">
                {t('scan.noKeywordMatches')}
              </Typography>
            )}
          </Box>

          <Box>
            {showImpersonation ? (
              <Typography variant="body2">
                {t('scan.impersonates')}: <strong>{typosquat?.matched_brand}</strong>
              </Typography>
            ) : (
              <Typography variant="body2" color="text.secondary">
                {t('scan.noTyposquatting')}
              </Typography>
            )}
          </Box>

          <Box>
            {urlPattern && urlPattern.flags.length > 0 ? (
              <Stack direction="row" spacing={0.5} sx={{ flexWrap: 'wrap', gap: 0.5 }}>
                {urlPattern.flags.map(([flag, weight]) => (
                  <Chip
                    key={flag}
                    size="small"
                    variant="outlined"
                    // Unknown/new rule codes fall back to the raw code rather than a missing-key string.
                    label={`${t(`scan.urlFlags.${flag}`, { defaultValue: flag })} (+${weight})`}
                  />
                ))}
              </Stack>
            ) : (
              <Typography variant="body2" color="text.secondary">
                {t('scan.noUrlFlags')}
              </Typography>
            )}
          </Box>
        </Stack>
      </EngineBreakdownCard>

      <EngineBreakdownCard
        icon={<StorageOutlined fontSize="small" color="action" />}
        title={t('scan.engineBlacklist')}
        applicable={blacklist !== null}
        notApplicableReason={blacklistSkipReasonKey ? t(blacklistSkipReasonKey) : undefined}
        score={blacklist?.score}
        tier={tiers.blacklist}
        timingMs={timings.blacklist_ms}
      >
        {blacklist?.match_type === 'exact' && blacklist.matched_entry && (
          <Stack spacing={0.5}>
            <Typography variant="body2">{t('scan.exactMatch')}</Typography>
            <Typography variant="caption" color="text.secondary">
              {(blacklist.matched_entry as BlacklistExactMatch).source} ·{' '}
              {t('scan.lastSeen')}: {(blacklist.matched_entry as BlacklistExactMatch).last_seen}
            </Typography>
          </Stack>
        )}
        {blacklist?.match_type === 'domain' && blacklist.matched_entry && (
          <Stack spacing={0.5}>
            <Typography variant="body2">
              {t('scan.seenBefore', { count: (blacklist.matched_entry as BlacklistDomainMatch).hit_count })}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {(blacklist.matched_entry as BlacklistDomainMatch).most_recent_source} ·{' '}
              {t('scan.lastSeen')}: {(blacklist.matched_entry as BlacklistDomainMatch).most_recent_last_seen}
            </Typography>
          </Stack>
        )}
        {blacklist?.match_type === null && (
          <Typography variant="body2" color="text.secondary">
            {t('scan.notInBlacklist')}
          </Typography>
        )}
      </EngineBreakdownCard>

      <EngineBreakdownCard
        icon={<LinkOutlined fontSize="small" color="action" />}
        title={t('scan.engineUrlModel')}
        applicable={url_model !== null}
        notApplicableReason={urlModelSkipReasonKey ? t(urlModelSkipReasonKey) : undefined}
        score={url_model?.score}
        tier={tiers.url_model}
        timingMs={timings.url_model_ms}
      >
        {url_model && (
          <ModelScoreSummary
            kind="url"
            probability={url_model.probability}
            rawTier={url_model.tier}
            shownTier={tiers.url_model ?? url_model.tier}
            thresholds={url_model.thresholds}
          />
        )}
      </EngineBreakdownCard>

      <EngineBreakdownCard
        icon={<PsychologyOutlined fontSize="small" color="action" />}
        title={t('scan.engineTextModel')}
        applicable={text_model !== null}
        notApplicableReason={textModelSkipReasonKey ? t(textModelSkipReasonKey) : undefined}
        score={text_model?.score}
        tier={tiers.text_model}
        timingMs={timings.text_model_ms}
      >
        {text_model && (
          <ModelScoreSummary
            kind="text"
            probability={text_model.probability}
            rawTier={text_model.tier}
            shownTier={tiers.text_model ?? text_model.tier}
            thresholds={text_model.thresholds}
          />
        )}
      </EngineBreakdownCard>

      <Typography
        variant="caption"
        color="text.secondary"
        sx={{ gridColumn: '1 / -1', textAlign: 'center' }}
      >
        {t('scan.totalTimeLabel', { ms: timings.total_ms })}
      </Typography>
    </Box>
  );
}
