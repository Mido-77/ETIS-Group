# ETIS — Setup Guide

Follow these steps in order. Each one builds on the last — don't skip ahead.

This covers the **full project**: backend API, database (seeded with real threat data),
and the React frontend. Skipping the seeding step or the frontend section will leave you
with a technically-running-but-empty/incomplete system.

---

## 1. Install the required tools

| Tool | Version | Windows | Mac |
|---|---|---|---|
| Git | any recent | https://git-scm.com/download/win | `brew install git` |
| Python | 3.11 or newer | https://www.python.org/downloads/ | `brew install python@3.12` |
| Node.js | LTS | https://nodejs.org/ | `brew install node` |
| PostgreSQL | 15.x | https://www.postgresql.org/download/windows/ | `brew install postgresql@15` |

**Note:** Docker isn't required for local setup — the project runs the backend and
frontend directly. Docker only becomes relevant later, for containerized deployment
(not part of this guide).

**Windows users:** if `winget install <package>` fails partway through with a download
error, just download the installer manually from the links above instead — that always
works.

**PostgreSQL install notes:**
- When the installer asks for a superuser password, set one and **write it down**. You'll
  need it in step 4.
- Leave the port as the default `5432`.
- Make sure "Command Line Tools" stays checked during install — this gives you the `psql`
  command.

### Check everything installed correctly

Open a terminal (PowerShell on Windows, Terminal on Mac) and run:

```
git --version
python --version
node --version
psql --version
```

Every command should print a version number. If any of them say "not recognized" /
"command not found," close and reopen your terminal first — if it still fails, the tool
didn't install correctly, ask in the group chat.

---

## 2. Clone the project

```
git clone https://github.com/Mido-77/ETIS.git
cd ETIS
```

**Everything below assumes your terminal stays at this project root (`ETIS/`) unless a
step explicitly tells you to `cd` somewhere else.** This matters — see the `.env` note in
step 4.

---

## 3. Set up the backend

```
cd backend
python -m venv venv
```

**Activate the virtual environment:**

- Windows (PowerShell):
  ```
  venv\Scripts\Activate.ps1
  ```
  If you get an error about "running scripts is disabled," run this once, then try again:
  ```
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

- Mac:
  ```
  source venv/bin/activate
  ```

Your terminal prompt should now start with `(venv)`. Then go back to the project root and
install all required packages from the **root** `requirements.txt` (not
`backend/requirements.txt` — the root one is the complete, current list, including the
TensorFlow/camel-tools/scikit-learn stack the LSTM engine needs):

```
cd ..
pip install -r requirements.txt
cd backend
```

This installs a lot (TensorFlow, PyTorch, transformers, etc.) and can take several
minutes and a few GB of disk — that's expected, not a sign anything's wrong.

---

## 4. Set up environment variables

**Important:** the `.env` file must live at the **project root** (`ETIS/.env`), NOT inside
`backend/`. `app/core/database.py` specifically loads it from the project root regardless
of which folder you run commands from — if `.env` ends up in `backend/.env` instead, the
app won't error, it'll just silently fall back to a default database URL and you'll get a
confusing connection failure later.

From the project root:

- Windows: `copy backend\.env.example .env`
- Mac: `cp backend/.env.example .env`

Open the new `.env` file (at the project root) in a text editor and fill in your own
PostgreSQL password (the one you set in step 1):

```
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD_HERE@localhost:5432/etis
```

---

## 5. Create and seed the local database

**Create the empty database.** With PostgreSQL running, open a terminal and run:

```
psql -U postgres
```

Enter your password when prompted, then inside the `psql` prompt run:

```sql
CREATE DATABASE etis;
\q
```

**Seed it with real data — don't skip this.** An empty database means the threat
intelligence blacklist engine and Egyptian brand typosquatting checks have nothing to
match against; scans will look like they're "working" but every URL will silently score
as if nothing is known about it. From the project root, with the backend venv still
activated:

```
python data/scripts/seed_database.py
```

Run this **from the project root** (`ETIS/`), not from inside `backend/` — if you're
still in `backend/` from step 3, run `cd ..` first, then `cd backend` afterward to
continue to step 6.

This reads the already-included `data/processed/dataset.csv` (it comes with the clone —
no need to regenerate it) and populates `blacklist_entries` from the PhishTank/OpenPhish
rows, plus `egyptian_brands`/`brand_domains` from the built-in brand registry. It prints
progress as it goes and is safe to re-run (duplicates are skipped automatically).

Don't run any other script in `data/scripts/` as part of setup — the rest of that folder
(`add_egyptian_domains.py`, `fetch_tranco.py`, `merge_arabic_sources.py`, etc.) are
one-off scripts used to originally build `dataset.csv`, not setup steps. Re-running them
would duplicate or corrupt data already in the file.

---

## 6. Run the backend

From inside `backend/` with the venv still activated:

```
uvicorn app.main:app --reload
```

If it starts without errors, open your browser to:

```
http://localhost:8000/docs
```

You should see the FastAPI interactive docs page. If that loads, your backend setup is
working correctly. **Leave this terminal running** — you'll need it alongside the
frontend in the next step.

**Optional but recommended — verify everything end-to-end:** open a *second* terminal,
activate the venv again (`venv\Scripts\Activate.ps1` / `source venv/bin/activate`), and
from `backend/` run:

```
python -m pytest
```

**This takes several minutes (5–6 minutes is normal)** — most of that time is the LSTM
model loading and running inference on CPU during the risk-scoring tests, not a hang.
You should see `59 passed`. If tests get skipped with a message like "Database not
seeded," go back and re-run step 5.

---

## 7. Set up the frontend

Open a **new terminal** (keep the backend running in the other one). From the project
root:

```
cd etis-frontend
```

Copy the example env file:

- Windows: `copy .env.example .env.local`
- Mac: `cp .env.example .env.local`

You shouldn't need to edit `.env.local` for local development — `VITE_API_BASE_URL`
already points at `http://localhost:8000` (your local backend from step 6), and
`VITE_ETIS_API_KEY` should stay empty (the backend skips the API key check entirely when
`ETIS_API_KEY` isn't set in its own `.env`, which it won't be for local dev).

Install dependencies and start the dev server:

```
npm install
npm run dev
```

Vite will print a local URL (typically `http://localhost:5173`). Open it in your browser
— you should see the ETIS dashboard in Arabic by default, with a toggle to switch to
English.

---

## 8. Verify the full stack together

With **both** the backend (`uvicorn`, step 6) and frontend (`npm run dev`, step 7) running
at the same time:

1. Open `http://localhost:5173`.
2. Go to the scan page and submit a test URL or piece of text.
3. You should get back a real LOW/MEDIUM/HIGH verdict with a per-engine breakdown — not an
   error, and not everything scoring 0.
4. Check the Threat Feed and Stats tabs show non-zero numbers (confirms step 5's seeding
   worked).

If any of this fails, see "Common issues" below.

---

## Common issues

- **Frontend shows a network error / can't reach the backend** → make sure `uvicorn` is
  still running in its own terminal, and that `VITE_API_BASE_URL` in
  `etis-frontend/.env.local` matches the address `uvicorn` printed (default
  `http://localhost:8000`).
- **Everything scores 0 / Threat Feed and Stats are empty** → the database wasn't seeded.
  Go back to step 5 and run the seeding scripts.
- **Backend can't connect to the database, or connects to the wrong one** → check that
  `.env` is at the project root (`ETIS/.env`), not inside `backend/`. See step 4.
- **`docker: command not found` or Docker errors** → Docker isn't required for this guide;
  ignore unless you're specifically trying to deploy the containerized version.
- **`psql` not recognized** → close and reopen your terminal, or re-check that "Command
  Line Tools" was included in the PostgreSQL install.
- **`pip install -r requirements.txt` fails** → make sure your venv is activated first
  (your prompt should show `(venv)`).
- **Arabic text looks garbled or errors out when querying `psql` directly** → on Windows,
  run `$env:PGCLIENTENCODING = "UTF8"` in that PowerShell session before running Arabic
  queries. This is a `psql` client-encoding issue, not an app bug — `chcp 65001` alone does
  NOT fix it.
- **`npm install` or `npm run dev` fails** → confirm `node --version` shows an LTS version
  (from step 1), and that you're running the command from inside `etis-frontend/`, not the
  project root.
- **Still stuck** → post the exact error message in the group chat, don't just say "it
  doesn't work."
