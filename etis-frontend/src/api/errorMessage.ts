import type { TFunction } from 'i18next';

import { ApiError } from './client';

/**
 * An error that is only a translation key (e.g. the empty-input message).
 * Pages keep the error OBJECT in state and translate it at render time, so
 * an error already on screen follows the language toggle.
 */
export class LocalizedError extends Error {
  readonly key: string;

  constructor(key: string) {
    super(key);
    this.name = 'LocalizedError';
    this.key = key;
  }
}

export interface ErrorDescription {
  /** Friendly, translated message. */
  message: string;
  /** The server's own English detail string, only when it adds information (400). */
  detail?: string;
}

function serverDetail(body: unknown): string | undefined {
  if (typeof body === 'object' && body !== null && 'detail' in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === 'string') return detail;
  }
  return undefined;
}

/**
 * Turns anything a page can catch into a translated message. Never shows raw
 * JSON or a URL: the technical text (ApiError.message / body) stays available
 * to the console, not to the user.
 */
export function describeError(t: TFunction, error: unknown): ErrorDescription {
  if (error instanceof LocalizedError) return { message: t(error.key) };

  if (error instanceof ApiError) {
    const { status } = error;
    if (status === 0) return { message: t('errors.network') };
    if (status === 400) return { message: t('errors.invalidInput'), detail: serverDetail(error.body) };
    if (status === 422) return { message: t('errors.invalidInput') };
    if (status === 401) return { message: t('errors.unauthorized') };
    if (status === 429) return { message: t('errors.rateLimited') };
    if (status === 502) return { message: t('errors.upstream') };
    if (status >= 500) return { message: t('errors.server') };
    return { message: t('errors.generic', { status }) };
  }

  return { message: t('common.unexpectedError') };
}
