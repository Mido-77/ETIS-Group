import type { Verdict } from '../api/types';

/**
 * Single source of truth for verdict colors, shared by VerdictCard and
 * anywhere else a verdict needs consistent color treatment (StatsPage's
 * by_verdict breakdown, e.g.) -- avoids the LOW/MEDIUM/HIGH palette
 * silently drifting apart between components.
 */
export const VERDICT_STYLE: Record<Verdict, { bg: string; accent: string; chipColor: 'success' | 'warning' | 'error' }> = {
  LOW: { bg: '#eafbf1', accent: '#2e7d4f', chipColor: 'success' },
  MEDIUM: { bg: '#fdf3e3', accent: '#c4881a', chipColor: 'warning' },
  HIGH: { bg: '#fdecea', accent: '#c0392b', chipColor: 'error' },
};
