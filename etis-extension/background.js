const API_BASE_URL = "http://localhost:8000";
const MAX_RECENT_SCANS = 10;

// Colors chosen to match the LOW/MEDIUM/HIGH palette already used on the
// website's VerdictCard (see etis-frontend/src/theme/verdictPalette.ts).
const BADGE_COLORS = {
  LOW: "#2e7d4f",
  MEDIUM: "#c4881a",
  HIGH: "#c0392b",
};

function setBadgeForVerdict(verdict) {
  const color = BADGE_COLORS[verdict] || "#888888";
  const text = verdict ? verdict.charAt(0) : "?";
  chrome.action.setBadgeBackgroundColor({ color });
  chrome.action.setBadgeText({ text });
}

function clearBadge() {
  chrome.action.setBadgeText({ text: "" });
}

chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local" && "autoScanEnabled" in changes) {
    if (!changes.autoScanEnabled.newValue) {
      clearBadge();
    }
  }
});

// --- Content-hash caching --------------------------------------------------
// SHA-256 hash of the extracted text + url, used as a cache key. Lets
// re-opening an already-scanned email (very common when browsing an
// inbox) reuse the stored verdict instead of calling /analyze again.
// Hashing both fields (not just text) means an email whose body stays the
// same but whose detected link differs is still treated as a distinct scan.
async function hashScanInput(text, url) {
  const encoded = new TextEncoder().encode(`${text}||${url || ""}`);
  const digest = await crypto.subtle.digest("SHA-256", encoded);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

// A short, human-readable label for the recent-scans list. content.js
// extracts emails as "<subject>\n\n<body>", so the first line is usually
// the subject; falls back to a truncated snippet for anything else.
function deriveLabel(text) {
  const firstLine = text.split("\n")[0].trim();
  return (firstLine || text).slice(0, 60);
}

async function getRecentScans() {
  const result = await chrome.storage.local.get(["recentScans"]);
  return result.recentScans || [];
}

async function findCachedScan(hash) {
  const scans = await getRecentScans();
  return scans.find((s) => s.hash === hash) || null;
}

async function recordScan(hash, text, data) {
  const scans = await getRecentScans();
  const entry = {
    hash,
    label: deriveLabel(text),
    verdict: data.verdict,
    score: data.final_score,
    timestamp: Date.now(),
  };
  // Newest first; drop any older entry for the same hash so re-scanning
  // doesn't create duplicate rows, then cap the list length.
  const updated = [entry, ...scans.filter((s) => s.hash !== hash)].slice(0, MAX_RECENT_SCANS);
  await chrome.storage.local.set({ recentScans: updated });
}

async function handleScanEmail(text, url) {
  const hash = await hashScanInput(text, url);
  const cached = await findCachedScan(hash);

  if (cached) {
    setBadgeForVerdict(cached.verdict);
    return {
      ok: true,
      fromCache: true,
      data: { verdict: cached.verdict, final_score: cached.score },
    };
  }

  const res = await fetch(`${API_BASE_URL}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, url: url || undefined }),
  });
  const data = await res.json();

  await recordScan(hash, text, data);
  setBadgeForVerdict(data.verdict);

  return { ok: true, fromCache: false, data };
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type !== "SCAN_EMAIL") return;

  handleScanEmail(message.text, message.url)
    .then(sendResponse)
    .catch((err) => {
      setBadgeForVerdict(null);
      sendResponse({ ok: false, error: String(err) });
    });

  return true; // keep the message channel open for the async response
});
