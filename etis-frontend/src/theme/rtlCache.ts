import createCache from '@emotion/cache';
import rtlPlugin from 'stylis-plugin-rtl';
import { prefixer } from 'stylis';

/**
 * MUI's `direction: 'rtl'` on the theme flips layout logic (margins,
 * flex order, etc.) but does NOT flip the actual CSS properties emotion
 * generates — that needs a separate stylis plugin in the cache. Without
 * this, Arabic mode gets RTL text but LTR-positioned icons/spacing.
 */
export function createRtlCache() {
  return createCache({
    key: 'mui-rtl',
    stylisPlugins: [prefixer, rtlPlugin],
  });
}

export function createLtrCache() {
  return createCache({
    key: 'mui-ltr',
  });
}
