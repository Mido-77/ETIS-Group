import type {
  AnalyzeRequest,
  AnalyzeResponse,
  ThreatsQueryParams,
  ThreatsResponse,
  StatsResponse,
} from './types';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const API_KEY = import.meta.env.VITE_ETIS_API_KEY;

/**
 * Header name for verify_api_key. Checked against app/core/security.py as it
 * stood in the project files: the dependency parameter is `x_api_key`, which
 * FastAPI maps to the header `X-API-Key` (header names are case-insensitive).
 * Still NOT exercised end to end: VITE_ETIS_API_KEY is empty locally and the
 * backend skips the check when ETIS_API_KEY is unset, so no request has yet
 * been made with a key set. Do that once before the hosted deployment.
 */
const API_KEY_HEADER_NAME = 'X-API-Key';

export class ApiError extends Error {
  readonly status: number;
  readonly body: unknown;

  constructor(status: number, body: unknown, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.body = body;
  }
}

function buildHeaders(hasJsonBody: boolean): HeadersInit {
  const headers: Record<string, string> = {};
  if (hasJsonBody) headers['Content-Type'] = 'application/json';
  if (API_KEY) headers[API_KEY_HEADER_NAME] = API_KEY;
  return headers;
}

/**
 * Shared request handler. Error body shape confirmed from main.py's global
 * exception handler: { error: "<ExceptionClassName>", detail: "<message>" }
 * for any ETISException subclass (InvalidScanInputError -> 400,
 * BlacklistLookupError -> 502, EngineFailureError -> 500, anything else
 * unmapped -> 500). Uncaught non-ETISException errors would fall through
 * to FastAPI's default handler instead, which has a different shape
 * ({"detail": ...} with no "error" key) -- ApiError.body is left as
 * `unknown` rather than typed as EtisErrorResponse for that reason, but
 * EtisErrorResponse is exported from types.ts so callers can safely check
 * for it (e.g. `if (err.body && 'error' in err.body)`).
 */
async function request<TResponse>(
  path: string,
  options: RequestInit & { hasJsonBody?: boolean } = {},
): Promise<TResponse> {
  const { hasJsonBody = false, ...init } = options;
  const url = `${BASE_URL}${path}`;

  let response: Response;
  try {
    response = await fetch(url, {
      ...init,
      headers: { ...buildHeaders(hasJsonBody), ...init.headers },
    });
  } catch (networkError) {
    throw new ApiError(
      0,
      networkError,
      `Network error reaching ${url} — is the backend running and is VITE_API_BASE_URL correct?`,
    );
  }

  const rawText = await response.text();
  const body = rawText ? safeJsonParse(rawText) : null;

  if (!response.ok) {
    throw new ApiError(
      response.status,
      body,
      `ETIS API ${response.status} on ${path}`,
    );
  }

  return body as TResponse;
}

function safeJsonParse(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text; // non-JSON error body (e.g. a plain-text 500) — don't crash on it
  }
}

function buildQueryString(params: object): string {
  const searchParams = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null) {
      searchParams.set(key, String(value));
    }
  }
  const query = searchParams.toString();
  return query ? `?${query}` : '';
}

// ---- Public API ----

export function analyzeScan(payload: AnalyzeRequest): Promise<AnalyzeResponse> {
  return request<AnalyzeResponse>('/analyze', {
    method: 'POST',
    body: JSON.stringify(payload),
    hasJsonBody: true,
  });
}

export function getThreats(params: ThreatsQueryParams = {}): Promise<ThreatsResponse> {
  return request<ThreatsResponse>(`/threats${buildQueryString(params)}`);
}

export function getStats(): Promise<StatsResponse> {
  return request<StatsResponse>('/stats');
}
