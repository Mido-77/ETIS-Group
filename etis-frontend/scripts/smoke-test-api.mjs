/**
 * Run this with your backend running locally (uvicorn app.main:app --reload,
 * from backend/) to confirm the real response shapes match src/api/types.ts.
 *
 * Usage:
 *   node scripts/smoke-test-api.mjs
 *   node scripts/smoke-test-api.mjs --base-url https://your-hosted-backend
 *
 * Requires Node 18+ (built-in fetch). No relation to the Vite app — this
 * is a plain script, doesn't import from src/, doesn't need env vars set.
 */

const baseUrlArg = process.argv.indexOf('--base-url');
const BASE_URL =
  baseUrlArg !== -1 ? process.argv[baseUrlArg + 1] : 'http://localhost:8000';

function printSection(title) {
  console.log(`\n${'='.repeat(60)}\n${title}\n${'='.repeat(60)}`);
}

async function callEndpoint(label, path, options = {}) {
  printSection(label);
  const url = `${BASE_URL}${path}`;
  console.log(`${options.method || 'GET'} ${url}`);
  if (options.body) console.log(`Body: ${options.body}`);

  try {
    const response = await fetch(url, options);
    const text = await response.text();
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = text;
    }
    console.log(`Status: ${response.status}`);
    console.log(JSON.stringify(parsed, null, 2));
    return { status: response.status, body: parsed };
  } catch (err) {
    console.error(`FAILED — is the backend running at ${BASE_URL}?`);
    console.error(err.message);
    return null;
  }
}

async function main() {
  console.log(`ETIS API smoke test against ${BASE_URL}`);

  // 1. GET /stats — confirms BlacklistStats / ScanStats shape
  await callEndpoint('GET /stats', '/stats');

  // 2. GET /threats — confirms ThreatEntry shape, small page
  await callEndpoint('GET /threats?limit=3', '/threats?limit=3');

  // 3. POST /analyze — text-only, should exercise rule_based + lstm,
  //    blacklist should come back null (no URL submitted)
  await callEndpoint('POST /analyze (text only)', '/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text: 'يرجى تحديث بياناتكم البنكية فوراً لتجنب إيقاف حسابكم',
    }),
  });

  // 4. POST /analyze — url-only, should exercise rule_based + blacklist,
  //    lstm should come back null (no text submitted)
  await callEndpoint('POST /analyze (url only)', '/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      url: 'https://inst4pay-secure.com/login',
    }),
  });

  console.log(
    '\nCompare the printed shapes above against src/api/types.ts — ' +
      'especially engine_details (rule_based/blacklist/lstm inner fields ' +
      'beyond "score") and any error responses, which are both marked ' +
      'UNVERIFIED in client.ts / types.ts until checked against this.',
  );
}

main();
