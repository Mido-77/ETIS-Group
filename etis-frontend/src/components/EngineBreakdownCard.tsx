import type { ReactNode } from 'react';
import { Paper, Stack, Typography, Chip, Box } from '@mui/material';
import { useTranslation } from 'react-i18next';

import type { Verdict } from '../api/types';
import { VERDICT_STYLE } from '../theme/verdictPalette';

function tierLabel(t: ReturnType<typeof useTranslation>['t'], tier: Verdict) {
  if (tier === 'LOW') return t('scan.verdictLow');
  if (tier === 'MEDIUM') return t('scan.verdictMedium');
  return t('scan.verdictHigh');
}

interface EngineBreakdownCardProps {
  icon: ReactNode;
  title: string;
  /** Null when the engine didn't run for this input (e.g. blacklist on a text-only submission). */
  applicable: boolean;
  /**
   * Localized reason the engine didn't run, e.g. "No URL was submitted."
   * Shown instead of the generic "not applicable" line when provided.
   * Ignored when `applicable` is true. Callers derive this from the
   * API's `skip_reasons` field (see risk_scoring.py) via
   * scan.skipReasonNoUrl / scan.skipReasonNoText.
   */
  notApplicableReason?: string;
  /** Engine's own 0-100 internal score. */
  score?: number;
  /**
   * The tier (LOW/MEDIUM/HIGH) this engine reached on its own, after the
   * Egyptian-brand allowlist override -- the verdict is the highest tier
   * any engine reaches (see risk_scoring.py's engine_tiers). Replaces the
   * old additive points chip.
   */
  tier?: Verdict;
  /**
   * Wall-clock time this engine took, in milliseconds (see
   * risk_scoring.py's stage_timings field). Shown as a chip alongside
   * score/contribution when provided -- for presentation/explainability
   * (e.g. showing LSTM inference dominates latency), not a performance
   * diagnostic tool. Omitted (undefined/null) when not applicable, since
   * the engine didn't run.
   */
  timingMs?: number | null;
  children?: ReactNode;
}

/**
 * Shows the engine's own score and the tier it reached. Under the
 * decision-tier scorer engines are not added together: the highest tier
 * wins, so there is no per-engine points share to display.
 */
export function EngineBreakdownCard({
  icon,
  title,
  applicable,
  notApplicableReason,
  score,
  tier,
  timingMs,
  children,
}: EngineBreakdownCardProps) {
  const { t } = useTranslation();

  return (
    <Paper
      variant="outlined"
      sx={{
        p: 2,
        borderRadius: 3,
        height: '100%',
        opacity: applicable ? 1 : 0.55,
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <Stack direction="row" spacing={1} sx={{ alignItems: 'center', mb: 1 }}>
        {icon}
        <Typography variant="subtitle2" sx={{ fontWeight: 600, flexGrow: 1 }}>
          {title}
        </Typography>
      </Stack>

      {applicable ? (
        <>
          <Stack direction="row" spacing={1} sx={{ mb: 1.5, flexWrap: 'wrap' }}>
            {score !== undefined && (
              <Chip size="small" variant="outlined" label={`${t('scan.score')}: ${score}/100`} />
            )}
            {tier !== undefined && (
              <Chip
                size="small"
                color={VERDICT_STYLE[tier].chipColor}
                label={`${t('scan.tierLabel')}: ${tierLabel(t, tier)}`}
              />
            )}
            {timingMs !== undefined && timingMs !== null && (
              <Chip size="small" variant="outlined" label={`${t('scan.timingLabel')}: ${t('scan.timingValue', { ms: timingMs })}`} />
            )}
          </Stack>
          <Box sx={{ flexGrow: 1 }}>{children}</Box>
        </>
      ) : (
        <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic' }}>
          {notApplicableReason ?? t('scan.notApplicable')}
        </Typography>
      )}
    </Paper>
  );
}
