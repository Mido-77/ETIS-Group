import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Box,
  Typography,
  CircularProgress,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  Chip,
  Stack,
  Button,
  TextField,
  MenuItem,
  Paper,
} from '@mui/material';

import { getThreats, ApiError } from '../api/client';
import { ErrorAlert } from '../components/ErrorAlert';
import type { ThreatEntry } from '../api/types';

const PAGE_SIZE = 20;
const MONO_FONT = "'IBM Plex Mono', monospace";

export function ThreatFeedPage() {
  const { t } = useTranslation();
  const [entries, setEntries] = useState<ThreatEntry[]>([]);
  const [totalMatching, setTotalMatching] = useState(0);
  const [offset, setOffset] = useState(0);
  const [source, setSource] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      setError(null);
      try {
        const response = await getThreats({
          limit: PAGE_SIZE,
          offset,
          source: source || undefined,
        });
        if (!cancelled) {
          setEntries(response.results);
          setTotalMatching(response.total_matching);
        }
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
  }, [offset, source]);

  const canGoPrev = offset > 0;
  const canGoNext = offset + PAGE_SIZE < totalMatching;
  const rangeStart = totalMatching === 0 ? 0 : offset + 1;
  const rangeEnd = Math.min(offset + PAGE_SIZE, totalMatching);

  return (
    <Box sx={{ py: 4, maxWidth: 900, mx: 'auto' }}>
      <Typography variant="h5" gutterBottom>
        {t('nav.threats')}
      </Typography>

      <Stack direction="row" spacing={2} sx={{ my: 2, alignItems: 'center' }}>
        <TextField
          select
          label={t('threats.source')}
          size="small"
          value={source}
          onChange={(e) => {
            setOffset(0);
            setSource(e.target.value);
          }}
          sx={{ minWidth: 160 }}
        >
          <MenuItem value="">{t('threats.allSources')}</MenuItem>
          <MenuItem value="phishtank">{t('sources.phishtank')}</MenuItem>
          <MenuItem value="openphish">{t('sources.openphish')}</MenuItem>
          <MenuItem value="manual">{t('sources.manual')}</MenuItem>
        </TextField>

        <Typography variant="body2" color="text.secondary">
          <Box component="span" sx={{ fontFamily: MONO_FONT }}>
            {totalMatching.toLocaleString()}
          </Box>{' '}
          {t('threats.totalMatching')}
        </Typography>
      </Stack>

      {loading && <CircularProgress size={24} />}
      {error != null && <ErrorAlert error={error} />}

      {!loading && !error && entries.length === 0 && (
        <Paper variant="outlined" sx={{ p: 3, borderRadius: 3, textAlign: 'center' }}>
          <Typography variant="body2" color="text.secondary">
            {t('threats.noResults')}
          </Typography>
        </Paper>
      )}

      {!loading && !error && entries.length > 0 && (
        <>
          <Paper variant="outlined" sx={{ borderRadius: 3, overflow: 'hidden' }}>
            <Table size="small">
              <TableHead>
                <TableRow sx={{ bgcolor: 'action.hover' }}>
                  <TableCell sx={{ fontWeight: 600 }}>{t('threats.domain')}</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>{t('threats.source')}</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>{t('threats.brandTarget')}</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>{t('threats.reputationScore')}</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>{t('threats.lastSeen')}</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>{t('threats.active')}</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {entries.map((entry) => (
                  <TableRow key={entry.id} hover>
                    <TableCell sx={{ fontFamily: MONO_FONT, fontSize: '0.8rem' }}>{entry.domain}</TableCell>
                    <TableCell>{t(`sources.${entry.source}`, { defaultValue: entry.source })}</TableCell>
                    <TableCell>{entry.brand_target ?? '—'}</TableCell>
                    <TableCell sx={{ fontFamily: MONO_FONT, fontSize: '0.8rem' }}>{entry.reputation_score}</TableCell>
                    <TableCell sx={{ fontFamily: MONO_FONT, fontSize: '0.8rem' }}>{entry.last_seen}</TableCell>
                    <TableCell>
                      <Chip
                        size="small"
                        label={entry.is_active ? t('threats.active') : t('threats.inactive')}
                        color={entry.is_active ? 'success' : 'default'}
                      />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Paper>

          <Stack direction="row" spacing={2} sx={{ mt: 2, alignItems: 'center' }}>
            <Button size="small" disabled={!canGoPrev} onClick={() => setOffset((o) => Math.max(0, o - PAGE_SIZE))}>
              {t('threats.previous')}
            </Button>
            <Button size="small" disabled={!canGoNext} onClick={() => setOffset((o) => o + PAGE_SIZE)}>
              {t('threats.next')}
            </Button>
            <Typography
              variant="caption"
              color="text.secondary"
              component="span"
              dir="ltr"
              sx={{ display: 'inline-block', fontFamily: MONO_FONT }}
            >
              {rangeStart.toLocaleString()}–{rangeEnd.toLocaleString()} / {totalMatching.toLocaleString()}
            </Typography>
          </Stack>
        </>
      )}
    </Box>
  );
}
