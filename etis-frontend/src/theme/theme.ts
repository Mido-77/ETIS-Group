import { createTheme, type Direction } from '@mui/material/styles';

/**
 * ETIS visual identity ("letterhead" theme, redesigned Aug 2026).
 *
 * Token system:
 * - Sadaf ink   #101C2C -- primary text
 * - Nile blue   #1B4B85 -- primary/brand (evolved from the original
 *   steel-blue #1d4e89, close enough to preserve continuity while giving
 *   the palette a deliberate name and a slightly richer navy)
 * - Qasr paper  #F6F5F1 -- page background, warm off-white instead of
 *   clinical pure white -- reads closer to official document stock
 * - Khatm gold  #A9761F -- SPARING accent only: the AppBar's letterhead
 *   rule and each card's top border. Never a primary action color.
 *   Evokes the seal/stamp language used throughout Egyptian official
 *   documents (bank paperwork, government certificates), reinforcing
 *   the "verified finding" idea behind the verdict stamp in
 *   VerdictCard.tsx -- that's the single signature element this theme
 *   is built around.
 * - Verdict colors (LOW #2E7D4F / MEDIUM #C4881A / HIGH #C0392B) are
 *   UNCHANGED from the original theme -- these are functional signal
 *   colors carrying real meaning (verdictPalette.ts), not up for
 *   redesign.
 *
 * A single theme factory taking direction as a parameter, rather than two
 * separate theme objects, so LOW/MEDIUM/HIGH verdict colors and spacing
 * stay in one place as the app grows past this scaffold.
 */
export function createEtisTheme(direction: Direction) {
  return createTheme({
    direction,
    palette: {
      mode: 'light',
      primary: { main: '#1B4B85' }, // Nile
      background: {
        default: '#F6F5F1', // Qasr
        paper: '#FFFFFF',
      },
      text: {
        primary: '#101C2C', // Sadaf
      },
      error: { main: '#C0392B' }, // HIGH verdict
      warning: { main: '#C4881A' }, // MEDIUM verdict
      success: { main: '#2E7D4F' }, // LOW verdict
    },
    typography: {
      // Cairo covers Arabic + Latin in one family, avoiding a visible
      // font swap when the language toggle flips direction. IBM Plex
      // Mono is loaded separately (index.css) and applied inline
      // wherever a numeric/data readout appears (scores, domains,
      // dates, counts) -- see VerdictCard.tsx, ThreatFeedPage.tsx,
      // StatsPage.tsx.
      fontFamily: ['Cairo', 'Inter', 'system-ui', 'sans-serif'].join(','),
      // MUI's default base is 14px; every variant (h1-h6, body1/2,
      // button, caption, etc.) is derived proportionally from this via
      // pxToRem, so bumping it here scales the whole app's type a
      // little larger without needing per-component overrides.
      fontSize: 15,
    },
    shape: {
      borderRadius: 8,
    },
    components: {
      // Letterhead rule: every AppBar gets a thin gold hairline under
      // the navy bar, evoking an official document header.
      MuiAppBar: {
        styleOverrides: {
          root: {
            borderBottom: '2px solid #A9761F',
          },
        },
      },
      // "Seal of record" cue: every outlined card (StatCard,
      // EngineBreakdownCard, ThreatFeedPage's table wrapper) gets a
      // thin gold top border instead of a heavier shadow-based
      // elevation -- keeps the flat, document-like surface language
      // consistent across the whole dashboard.
      MuiPaper: {
        styleOverrides: {
          outlined: {
            borderTop: '3px solid #A9761F',
          },
        },
      },
    },
  });
}
