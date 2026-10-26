import { Alert, Typography } from '@mui/material';
import type { SxProps, Theme } from '@mui/material/styles';
import { useTranslation } from 'react-i18next';

import { describeError } from '../api/errorMessage';

/**
 * Error box shared by the three pages. It takes the error object and
 * translates it here, so the message follows the language toggle. The
 * server's English detail (400 only) is shown as a small left-to-right
 * caption; `end`/`start` rather than left/right so the RTL emotion cache
 * does not flip the alignment.
 */
export function ErrorAlert({ error, sx }: { error: unknown; sx?: SxProps<Theme> }) {
  const { t } = useTranslation();
  const { message, detail } = describeError(t, error);

  return (
    <Alert severity="error" sx={sx}>
      {message}
      {detail && (
        <Typography
          variant="caption"
          component="div"
          dir="ltr"
          sx={(theme) => ({ mt: 0.5, opacity: 0.75, textAlign: theme.direction === 'rtl' ? 'end' : 'start' })}
        >
          {detail}
        </Typography>
      )}
    </Alert>
  );
}
