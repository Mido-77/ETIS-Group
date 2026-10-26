import { LinearProgress, Stack, Typography } from '@mui/material';
import { useTranslation } from 'react-i18next';

import type { ModelThresholds, Verdict } from '../api/types';
import { VERDICT_STYLE } from '../theme/verdictPalette';

interface ModelScoreSummaryProps {
  kind: 'url' | 'text';
  /** The model's raw probability, 0-1. Always the real, un-overridden value. */
  probability: number;
  /** The tier the model reached on its own (engine_details.<model>.tier). */
  rawTier: Verdict;
  /** The tier shown on the card, after the Egyptian-brand allowlist override (engine_tiers). */
  shownTier: Verdict;
  thresholds: ModelThresholds;
}

/**
 * Tier-aware explanation of a learned model's score, replacing the old
 * "N% match to phishing patterns" line. That wording sat next to a LOW tier
 * for scores like 60% (the MEDIUM threshold is ~67%) and read like a bug.
 * The number is never hidden -- it is always in the sentence and in the
 * card's Score chip -- but the sentence now says what the number meant:
 *
 *   - flagged (MEDIUM/HIGH): resembles / strongly matches phishing patterns
 *   - LOW on its own: below the flag threshold (so 60% + LOW is explained)
 *   - LOW only because of the allowlist: the raw score is stated as not counted
 *
 * In the below-threshold sentence the percentages are rounded in opposite
 * directions (score down, threshold up) so a score just under the
 * threshold can never print as "67% is below the 67% threshold".
 */
export function ModelScoreSummary({ kind, probability, rawTier, shownTier, thresholds }: ModelScoreSummaryProps) {
  const { t } = useTranslation();
  const percent = Math.round(probability * 100);

  let message: string;
  if (shownTier !== 'LOW') {
    const key =
      kind === 'url'
        ? shownTier === 'HIGH'
          ? 'scan.urlModelFlaggedHigh'
          : 'scan.urlModelFlaggedMedium'
        : shownTier === 'HIGH'
          ? 'scan.textModelFlaggedHigh'
          : 'scan.textModelFlaggedMedium';
    message = t(key, { percent });
  } else if (rawTier !== 'LOW') {
    message = t('scan.modelOverridden', { percent });
  } else {
    message = t('scan.modelBelowThreshold', {
      percent: Math.floor(probability * 100),
      threshold: Math.ceil(thresholds.medium * 100),
    });
  }

  return (
    <Stack spacing={1}>
      <LinearProgress
        variant="determinate"
        value={probability * 100}
        color={VERDICT_STYLE[shownTier].chipColor}
        sx={{ height: 6, borderRadius: 3 }}
      />
      <Typography variant="body2">{message}</Typography>
    </Stack>
  );
}
