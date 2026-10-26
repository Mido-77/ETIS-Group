# ETIS Email Scanner — Browser Extension (Experimental)

A Chrome extension that brings ETIS's phishing detection directly into Gmail,
instead of requiring a manual copy-paste into the website. Built as an
experimental side-branch of the main ETIS project — the website
(`etis-frontend/`) and backend (`backend/`) remain the primary, thesis-stable
deliverable. This extension is additive: it calls the same `/analyze` API
and doesn't touch the detection engines at all.

**Status:** functional, manually tested end-to-end against live Gmail.
No automated test suite — this is a prototype, not production code.

---

## What it does

- **Manual scan** — click the extension icon, hit "Scan open email" to
  score whatever email is currently open in Gmail.
- **Automatic scanning** (opt-in toggle, default off) — watches Gmail for
  an opened email and scans it on its own, debounced so it doesn't fire
  mid-render while Gmail is still painting the message.
- **Badge icon** — the toolbar icon shows a colored letter (`L`/`M`/`H`)
  reflecting the verdict of the last scan; clears itself when automatic
  scanning is switched off.
- **Content-hash caching** — re-opening an already-scanned email reuses
  the stored result instead of calling the API again.
- **Recent scans list** — the last 10 scans, persisted locally, shown in
  the popup.
- **Link extraction** — pulls the primary link out of the email body
  (unwrapping Gmail's `google.com/url?q=...` click-tracking redirect) and
  submits it as the `url` alongside the email text, so scans use all four
  engines (rule-based, threat-intelligence blacklist, URL model and text
  model) instead of being capped at the text-only path. The verdict is the
  highest risk tier any engine reaches.
- **Website handoff** — "Open full report on website" opens the ETIS
  website pre-filled with the current email's text and detected link, and
  automatically runs the analysis — no need to click Analyze again.

---

## Requirements

- Google Chrome (or any Chromium-based browser supporting Manifest V3)
- The ETIS backend running locally: `uvicorn app.main:app --reload` from
  `backend/`, listening on `http://localhost:8000`
- For the website handoff feature specifically: the frontend dev server
  running too (`npm run dev` from `etis-frontend/`), on
  `http://localhost:5173`
- A Gmail account to test against (`mail.google.com` — this extension
  does not support Outlook or other webmail clients)

---

## Loading it unpacked

1. Make sure the backend is running (see above)
2. Open `chrome://extensions`
3. Enable **Developer mode** (top-right toggle)
4. Click **Load unpacked**
5. Select the `etis-extension/` folder
6. The ETIS shield icon should appear in your toolbar — pin it if it
   lands in the overflow menu instead (badges don't render on unpinned,
   overflowed icons)

---

## File overview

| File | Role |
|---|---|
| `manifest.json` | Extension config: permissions, icons, entry points |
| `content.js` | Injected into Gmail. Extracts the open email's subject/body/primary link; runs the automatic-scan `MutationObserver` |
| `background.js` | Service worker. Owns the actual `/analyze` call, badge-setting, content-hash cache, and recent-scans storage — the single source of truth both the manual and automatic scan paths funnel through |
| `popup.html` / `popup.js` | The toolbar popup UI: toggle, manual scan button, website handoff button, recent-scans list |
| `icons/` | Toolbar/extension-listing icon, four sizes (16/32/48/128px) |

---

## Known limitations

- **DOM-based extraction, not an official API.** `content.js` reads
  Gmail's live page structure (`h2.hP` for the subject, `div.a3s.aiL` for
  the body). This is inherently fragile — if Google changes Gmail's
  markup, extraction breaks until the selectors are updated. This was a
  known, accepted trade-off from the start (see the project's wider
  discussion of this vs. using the Gmail API with OAuth).
- **Gmail only.** No Outlook, no other webmail client.
- **Single primary link only.** If an email contains multiple links, only
  the first qualifying one is extracted and scored; the rest are ignored.
  Scoring every link individually is a reasonable future enhancement, not
  built here.
- **No API key handling.** `background.js` calls `/analyze` with no
  `X-API-Key` header. This is fine for local development, since
  `verify_api_key` no-ops when `ETIS_API_KEY` is unset on the backend —
  but if a deployment ever sets that env var, this extension will need a
  header added (and a safe way to store the key) before it will keep
  working against it.
- **Hardcoded localhost URLs.** Both `background.js`'s `API_BASE_URL` and
  `popup.js`'s `WEBSITE_BASE_URL` point at `localhost`. Pointing this at
  a hosted deployment means updating both constants.
- **New tab every handoff click.** "Open full report on website" always
  opens a new tab rather than reusing an existing ETIS tab.

---

## Troubleshooting

**Popup says "Could not extract email" / console shows `Extension context
invalidated` or `Could not establish connection`:**
This happens whenever the extension is reloaded from `chrome://extensions`
while a Gmail tab stays open — that tab's content script becomes
orphaned. **Refresh the Gmail tab** after every extension reload. This
isn't a bug to fix so much as a standard MV3 development habit:
reload the extension → refresh any already-open Gmail tabs → then test.

**Badge (`L`/`M`/`H`) doesn't show on the toolbar icon:**
Check the icon is actually **pinned** to the toolbar (click the puzzle-piece
overflow menu → pin icon next to ETIS Email Scanner). Badges don't render
on icons sitting in the overflow dropdown.

**Changes to `background.js` don't seem to take effect:**
Unlike `content.js` (refreshed by reloading the Gmail tab), edits to
`background.js` require an explicit extension reload at
`chrome://extensions` — saving the file alone doesn't do anything until
you click reload there.

**Lots of unrelated-looking errors in the console
(`ERR_BLOCKED_BY_CLIENT`, `Permissions-Policy` warnings,
`play.google.com/log` failures):**
This is Gmail's own internal telemetry being blocked by an ad blocker, and
unrelated Chrome/Permissions-Policy noise — not from this extension.
Safe to ignore; none of it references `content.js` or `background.js`.

**Website handoff opens a blank/unreachable page:**
The frontend dev server isn't running. Start it with `npm run dev` from
`etis-frontend/`, or update `WEBSITE_BASE_URL` in `popup.js` if it's
running on a different port.
