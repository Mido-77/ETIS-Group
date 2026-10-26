import type { ReactNode } from 'react';
import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Box, Typography, CircularProgress, Grid, Paper, Stack, Chip } from '@mui/material';
import StorageOutlined from '@mui/icons-material/StorageOutlined';
import CheckCircleOutlineOutlined from '@mui/icons-material/CheckCircleOutlineOutlined';
import FactCheckOutlined from '@mui/icons-material/FactCheckOutlined';
import SpeedOutlined from '@mui/icons-material/SpeedOutlined';

import { getStats, ApiError } from '../api/client';
import { ErrorAlert } from '../components/ErrorAlert';
import type { StatsResponse, Verdict } from '../api/types';
import { VERDICT_STYLE } from '../theme/verdictPalette';

const VERDICT_LABEL_KEY: Record<Verdict, string> = {
  LOW: 'scan.verdictLow',
  MEDIUM: 'scan.verdictMedium',
  HIGH: 'scan.verdictHigh',
};

export function StatsPage() {
  const { t } = useTranslation();
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      setError(null);
      try {
        const response = await getStats();
        if (!cancelled) setStats(response);
      } catch (err) {
        if (cancelled) return;
        if (!(err instanceof ApiError)) console.error(err);
        setError(err);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <Box sx={{ py: 4, maxWidth: 900, mx: 'auto' }}>
      <Typography variant="h5" gutterBottom>
        {t('nav.stats')}
      </Typography>

      {loading && <CircularProgress size={24} />}
      {error != null && <ErrorAlert error={error} />}

      {stats && (
        <Grid container spacing={2} sx={{ mt: 1 }}>
          <Grid size={{ xs: 6, sm: 3 }}>
            <StatCard
              icon={<StorageOutlined fontSize="small" />}
              label={t('stats.totalEntries')}
              value={stats.blacklist.total_entries.toLocaleString()}
            />
          </Grid>
          <Grid size={{ xs: 6, sm: 3 }}>
            <StatCard
              icon={<CheckCircleOutlineOutlined fontSize="small" />}
              label={t('stats.activeEntries')}
              value={stats.blacklist.active_entries.toLocaleString()}
            />
          </Grid>
          <Grid size={{ xs: 6, sm: 3 }}>
            <StatCard
              icon={<FactCheckOutlined fontSize="small" />}
              label={t('stats.totalScans')}
              value={stats.scans.total_scans.toLocaleString()}
            />
          </Grid>
          <Grid size={{ xs: 6, sm: 3 }}>
            <StatCard
              icon={<SpeedOutlined fontSize="small" />}
              label={t('stats.averageScore')}
              value={stats.scans.average_final_score !== null ? stats.scans.average_final_score.toFixed(2) : '—'}
            />
          </Grid>

          <Grid size={{ xs: 12, sm: 6 }}>
            <Paper variant="outlined" sx={{ p: 2, borderRadius: 3 }}>
              <Typography variant="subtitle2" gutterBottom>
                {t('stats.bySourceHeading')}
              </Typography>
              <Stack spacing={1} sx={{ mt: 1 }}>
                {Object.entries(stats.blacklist.by_source).map(([source, count]) => (
                  <Stack key={source} direction="row" sx={{ justifyContent: 'space-between' }}>
                    <Typography variant="body2">{t(`sources.${source}`, { defaultValue: source })}</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 600, fontFamily: "'IBM Plex Mono', monospace" }}>
                      {count.toLocaleString()}
                    </Typography>
                  </Stack>
                ))}
              </Stack>
            </Paper>
          </Grid>

          <Grid size={{ xs: 12, sm: 6 }}>
            <Paper variant="outlined" sx={{ p: 2, borderRadius: 3 }}>
              <Typography variant="subtitle2" gutterBottom>
                {t('stats.byVerdictHeading')}
              </Typography>
              <Stack spacing={1} sx={{ mt: 1 }}>
                {(Object.entries(stats.scans.by_verdict) as [Verdict, number][]).map(([verdict, count]) => (
                  <Stack key={verdict} direction="row" spacing={1.5} sx={{ alignItems: 'center' }}>
                    <Chip
                      size="small"
                      label={t(VERDICT_LABEL_KEY[verdict])}
                      color={VERDICT_STYLE[verdict].chipColor}
                      sx={{ fontWeight: 600, minWidth: 72 }}
                    />
                    <Typography variant="body2" sx={{ fontWeight: 600, fontFamily: "'IBM Plex Mono', monospace" }}>
                      {count.toLocaleString()}
                    </Typography>
                  </Stack>
                ))}
              </Stack>
            </Paper>
          </Grid>
        </Grid>
      )}
    </Box>
  );
}

function StatCard({ icon, label, value }: { icon: ReactNode; label: string; value: string }) {
  return (
    <Paper variant="outlined" sx={{ p: 2, borderRadius: 3 }}>
      <Stack direction="row" sx={{ justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <Typography sx={{ fontFamily: "'IBM Plex Mono', monospace", fontWeight: 600, fontSize: '1.5rem' }}>
          {value}
        </Typography>
        <Box sx={{ color: 'text.secondary' }}>{icon}</Box>
      </Stack>
      <Typography variant="caption" color="text.secondary">
        {label}
      </Typography>
    </Paper>
  );
}
