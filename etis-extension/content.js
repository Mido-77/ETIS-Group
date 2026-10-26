// --- Manual extraction (used by the popup's "Scan open email" button) ---
// Gmail-specific selectors -- verified working against live Gmail in
// Checkpoint 1, may need revisiting if Google changes their markup.
function extractOpenEmail() {
  const subjectEl = document.querySelector("h2.hP");
  const bodyEl = document.querySelector("div.a3s.aiL");

  if (!bodyEl) return null;

  const subject = subjectEl ? subjectEl.innerText : "";
  const body = bodyEl.innerText;
  const text = `${subject}\n\n${body}`;
  const url = extractPrimaryLink(bodyEl);

  return { text, url };
}

// Picks the first external http(s) link in the email body. Gmail routes
// external links through a google.com/url?q=...&sa=... redirect for
// click-safety/tracking -- that wrapper is unwrapped here so the real
// destination domain gets checked by the backend's blacklist/typosquat
// engines, not google.com itself.
//
// Known simplification (v1, deliberate): only the FIRST qualifying link
// is used, even if an email contains several. Scoring every link
// individually is a reasonable future enhancement, not built here.
function extractPrimaryLink(bodyEl) {
  const anchors = bodyEl.querySelectorAll("a[href]");

  for (const anchor of anchors) {
    const rawHref = anchor.getAttribute("href");
    if (!rawHref || rawHref.startsWith("mailto:")) continue;

    let href = rawHref;
    try {
      const parsed = new URL(rawHref);
      if (parsed.hostname.includes("google.com") && parsed.pathname === "/url") {
        const real = parsed.searchParams.get("q");
        if (real) href = real;
      }
    } catch {
      continue; // malformed href, skip
    }

    if (href.startsWith("http://") || href.startsWith("https://")) {
      return href;
    }
  }

  return null;
}

// If this content script was injected while its extension context was
// ALREADY invalidated (e.g. the extension was reloaded before this tab
// got refreshed), chrome.runtime itself can be undefined here -- not just
// chrome.runtime.id, which is what tryAutoScan() below checks for.
// Registering a listener unconditionally in that state throws
// synchronously, which would otherwise abort the rest of this script
// (including the auto-scan observer setup further down). Wrapping just
// this registration means a stale tab logs one clear warning and quietly
// does nothing, instead of silently breaking everything after it.
try {
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type !== "EXTRACT_EMAIL") return;

    const extraction = extractOpenEmail();
    if (!extraction) {
      sendResponse({ ok: false, error: "No open email detected." });
      return;
    }
    sendResponse({ ok: true, text: extraction.text, url: extraction.url });
  });
} catch (err) {
  console.warn(
    "[ETIS] This tab's extension context is stale (the extension was reloaded after this tab opened) -- refresh this page to restore ETIS."
  );
}

// --- Automatic scanning ---------------------------------------------------
// Only runs when the user has switched "Automatic scanning" on in the
// popup. Watches Gmail's page for the email body appearing/changing
// (Gmail is a single-page app -- there's no normal page navigation to
// hook into) and triggers the same scan flow as the manual button.

let autoScanEnabled = false;
let lastScannedText = null; // avoids re-scanning the same open email repeatedly
let contextInvalidated = false; // true once this tab's connection to the
                                 // extension runtime has been severed --
                                 // happens whenever the extension is
                                 // reloaded while this Gmail tab stays open.

chrome.storage.local.get(["autoScanEnabled"], (result) => {
  autoScanEnabled = Boolean(result.autoScanEnabled);
});

// Keep in sync if the user flips the toggle while this Gmail tab is open.
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local" && "autoScanEnabled" in changes) {
    autoScanEnabled = Boolean(changes.autoScanEnabled.newValue);
  }
});

function tryAutoScan() {
  if (!autoScanEnabled || contextInvalidated) return;

  // chrome.runtime.id becomes undefined once this content script's
  // extension context has been invalidated (e.g. the extension was
  // reloaded from chrome://extensions while this tab stayed open).
  // Checking it here avoids ever calling sendMessage into a dead runtime.
  if (!chrome.runtime?.id) {
    contextInvalidated = true;
    console.warn(
      "[ETIS] Extension was reloaded -- refresh this Gmail tab to resume auto-scanning."
    );
    observer.disconnect();
    return;
  }

  const extraction = extractOpenEmail();
  if (!extraction || !extraction.text || extraction.text === lastScannedText) return;

  lastScannedText = extraction.text;

  try {
    chrome.runtime.sendMessage(
      { type: "SCAN_EMAIL", text: extraction.text, url: extraction.url },
      (response) => {
        if (chrome.runtime.lastError) {
          // Context died between the check above and this call returning --
          // a narrow race window, handled the same way as the check above.
          contextInvalidated = true;
          observer.disconnect();
          return;
        }
        if (!response || !response.ok) {
          console.warn("[ETIS] Auto-scan failed:", response?.error);
          return;
        }
        console.log(
          "[ETIS] Auto-scan result:",
          response.data.verdict,
          response.data.final_score,
          response.fromCache ? "(cached)" : "(fresh)"
        );
      }
    );
  } catch (err) {
    // Belt-and-braces: covers any other "Extension context invalidated"
    // throw that slips past the checks above.
    contextInvalidated = true;
    observer.disconnect();
  }
}

// Gmail re-renders its content area in place (no full page reloads), so a
// MutationObserver on the main content container is how we detect "an
// email was opened" instead of listening for navigation events.
//
// Gmail paints an email's body in stages (initial text, then images,
// signatures, quoted-thread content filling in shortly after) -- reacting
// to every single mutation can catch the body mid-render and extract
// incomplete text, which hashes differently from the final, settled
// version and shows up as a spurious near-duplicate scan. Debouncing
// waits for mutations to stop for DEBOUNCE_MS before extracting, so we
// scan the settled, final content instead of a partial snapshot.
const DEBOUNCE_MS = 600;
let debounceTimer = null;

const observer = new MutationObserver(() => {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(tryAutoScan, DEBOUNCE_MS);
});

observer.observe(document.body, { childList: true, subtree: true });
