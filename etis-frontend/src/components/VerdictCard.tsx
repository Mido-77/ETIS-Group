import { Box, Typography, Chip, Stack } from '@mui/material';
import VerifiedOutlined from '@mui/icons-material/VerifiedOutlined';
import { useTranslation } from 'react-i18next';

import type { AnalyzeResponse, Verdict } from '../api/types';
import { explainVerdict } from '../api/explainVerdict';
import { VERDICT_STYLE } from '../theme/verdictPalette';

// No rotation -- the seal renders flat/balanced for every verdict.
// (An earlier version tilted this per-verdict to read like a rubber
// stamp; kept as a lookup table rather than removed entirely, in case a
// balanced-but-still-stamp-like treatment is wanted later -- e.g. a
// shared, non-zero rotation applied once to the whole card rather than
// varying per verdict.)
const SEAL_ROTATION: Record<Verdict, number> = {
  LOW: 0,
  MEDIUM: 0,
  HIGH: 0,
};

function verdictLabel(t: ReturnType<typeof useTranslation>['t'], verdict: Verdict) {
  if (verdict === 'LOW') return t('scan.verdictLow');
  if (verdict === 'MEDIUM') return t('scan.verdictMedium');
  return t('scan.verdictHigh');
}

/**
 * The verdict "seal" -- a rotated double-ring circular stamp containing
 * the verdict word and an ETIS mark, replacing the flat colored chip
 * used previously. This is the app's signature visual element: it
 * deliberately evokes the stamped-seal language of Egyptian official
 * documents (bank paperwork, government certificates) to reinforce that
 * a scan result is an authenticated finding, not just a colored label --
 * directly reinforcing user trust exactly where it matters most.
 *
 * No letter-spacing applied to the verdict word (unlike a typical
 * badge) since this label renders in both Latin ("LOW") and Arabic
 * ("منخفض") depending on language -- added letter-spacing distorts
 * Arabic ligatures, so the styling here is intentionally script-agnostic.
 */
function VerdictSeal({ verdict, color }: { verdict: Verdict; color: string }) {
  const { t } = useTranslation();
  return (
    <Box
      sx={{
        width: 76,
        height: 76,
        borderRadius: '50%',
        border: `2.5px solid ${color}`,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        position: 'relative',
        flexShrink: 0,
        transform: `rotate(${SEAL_ROTATION[verdict]}deg)`,
      }}
    >
      <Box
        sx={{
          position: 'absolute',
          inset: 5,
          border: `1px solid ${color}`,
          borderRadius: '50%',
        }}
      />
      <Typography
        sx={{
          fontSize: 10,
          fontWeight: 700,
          color,
          textAlign: 'center',
          lineHeight: 1.15,
          px: 0.5,
        }}
      >
        {verdictLabel(t, verdict)}
      </Typography>
      <Typography sx={{ fontSize: 8, color, fontFamily: "'IBM Plex Mono', monospace", mt: 0.25 }}>
        ETIS
      </Typography>
    </Box>
  );
}

const ENGINE_LABEL_KEY: Record<string, string> = {
  rule_based: 'scan.engineRuleBased',
  blacklist: 'scan.engineBlacklist',
  url_model: 'scan.engineUrlModel',
  text_model: 'scan.engineTextModel',
};

/**
 * The score is never shown as a bare number -- always paired with the
 * verdict seal AND a one-line reason built from the actual engine
 * breakdown (see explainVerdict). The score itself renders in IBM Plex Mono,
 * matching the mono treatment used for every other numeric readout in
 * the dashboard (domains, dates, counts -- see ThreatFeedPage/StatsPage),
 * so this card doesn't read as a one-off exception to the type system.
 *
 * Decision-tier scorer (2026-10): the verdict is the highest tier any
 * engine reaches, so the card names that engine ("Decided by ...") instead
 * of the old heuristic confidence chip. The confidence field is still in
 * the API but is no longer shown: with one deciding engine, low
 * "agreement" with engines that simply have nothing to say read as doubt
 * next to a clear HIGH verdict, which is not what it meant.
 */
export function VerdictCard({ result }: { result: AnalyzeResponse }) {
  const { t } = useTranslation();
  const style = VERDICT_STYLE[result.verdict];
  const decidedByKey = result.decided_by ? ENGINE_LABEL_KEY[result.decided_by] : undefined;

  return (
    <Box
      sx={{
        bgcolor: style.bg,
        border: '1px solid',
        borderColor: `${style.accent}33`,
        borderRadius: 3,
        p: 3,
      }}
    >
      <Stack direction="row" spacing={2.5} sx={{ alignItems: 'center', flexWrap: 'wrap' }}>
        <VerdictSeal verdict={result.verdict} color={style.accent} />

        <Box sx={{ flex: 1, minWidth: 160 }}>
          <Stack direction="row" spacing={1} sx={{ alignItems: 'baseline', flexWrap: 'wrap', rowGap: 0.5 }}>
            <Typography
              sx={{
                fontFamily: "'IBM Plex Mono', monospace",
                fontSize: '2rem',
                fontWeight: 600,
                color: style.accent,
                lineHeight: 1,
              }}
            >
              {result.final_score.toFixed(2)}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              / 100
            </Typography>
            {result.allowlisted && (
              <Chip
                icon={<VerifiedOutlined fontSize="small" />}
                label={t('scan.allowlistedBadge')}
                size="small"
                color="info"
                variant="outlined"
                sx={{ fontWeight: 600 }}
              />
            )}
          </Stack>

          <Typography variant="body1" sx={{ mt: 1 }}>
            {explainVerdict(t, result)}
          </Typography>

          {decidedByKey && !result.allowlisted && (
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
              {t('scan.decidedBy', { engine: t(decidedByKey) })}
            </Typography>
          )}
        </Box>
      </Stack>
    </Box>
  );
}
