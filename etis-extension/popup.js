// Dev-server default per the frontend's .env.example (VITE_API_BASE_URL
// pattern) -- update this if the website runs on a different port, or
// once a hosted URL exists.
const WEBSITE_BASE_URL = "http://localhost:5173";

const toggleEl = document.getElementById("autoScanToggle");
const resultEl = document.getElementById("result");
const recentScansEl = document.getElementById("recentScans");

chrome.storage.local.get(["autoScanEnabled"], (result) => {
  toggleEl.checked = Boolean(result.autoScanEnabled);
});

toggleEl.addEventListener("change", () => {
  chrome.storage.local.set({ autoScanEnabled: toggleEl.checked });
});

function verdictColor(verdict) {
  if (verdict === "HIGH") return "#c0392b";
  if (verdict === "MEDIUM") return "#c4881a";
  if (verdict === "LOW") return "#2e7d4f";
  return "#888888";
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function renderRecentScans(scans) {
  if (!scans || scans.length === 0) {
    recentScansEl.innerHTML = '<div class="empty-state">No scans yet.</div>';
    return;
  }

  recentScansEl.innerHTML = scans
    .map((s) => {
      const time = new Date(s.timestamp).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      });
      return `
        <div class="scan-row">
          <span class="verdict-dot" style="background:${verdictColor(s.verdict)}"></span>
          <span class="scan-label">${escapeHtml(s.label)}</span>
          <span class="scan-meta">${s.verdict} · ${time}</span>
        </div>
      `;
    })
    .join("");
}

chrome.storage.local.get(["recentScans"], (result) => {
  renderRecentScans(result.recentScans || []);
});

chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local" && "recentScans" in changes) {
    renderRecentScans(changes.recentScans.newValue || []);
  }
});

document.getElementById("scanBtn").addEventListener("click", async () => {
  resultEl.textContent = "Scanning...";

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  chrome.tabs.sendMessage(tab.id, { type: "EXTRACT_EMAIL" }, (extraction) => {
    if (!extraction || !extraction.ok) {
      resultEl.textContent = extraction?.error || "Could not extract email.";
      return;
    }

    chrome.runtime.sendMessage(
      { type: "SCAN_EMAIL", text: extraction.text, url: extraction.url },
      (response) => {
        if (!response || !response.ok) {
          resultEl.textContent = "Scan failed: " + (response?.error || "unknown error");
          return;
        }
        const r = response.data;
        const cacheNote = response.fromCache ? " (cached)" : "";
        resultEl.textContent = `Verdict: ${r.verdict}\nScore: ${r.final_score}/100${cacheNote}`;
      }
    );
  });
});

// Re-extracts the currently open email (independent of whether "Scan open
// email" has already been clicked) and opens the website with it
// pre-filled, instead of a cold, empty form.
document.getElementById("openWebsiteBtn").addEventListener("click", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  chrome.tabs.sendMessage(tab.id, { type: "EXTRACT_EMAIL" }, (extraction) => {
    if (!extraction || !extraction.ok) {
      resultEl.textContent = extraction?.error || "Could not extract email.";
      return;
    }

    const params = new URLSearchParams();
    params.set("etis_text", extraction.text);
    if (extraction.url) {
      params.set("etis_url", extraction.url);
    }

    chrome.tabs.create({ url: `${WEBSITE_BASE_URL}/?${params.toString()}` });
  });
});
