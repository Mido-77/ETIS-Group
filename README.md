# ETIS — Egypt-Specific Threat Intelligence System

ETIS is a hybrid phishing-detection system designed to identify phishing attempts targeting Egyptian users and businesses.

## Features

- Hybrid phishing detection using rules, threat-intelligence data, and machine-learning models.
- URL and text analysis.
- Egyptian-brand impersonation detection.
- Arabic-aware text processing.
- React dashboard with Arabic and English interfaces.
- FastAPI backend and PostgreSQL database.

## Prerequisites

- Git
- Python 3.11 or newer
- Node.js LTS and npm
- PostgreSQL 15.x

## Installation

Clone the repository:

```powershell
git clone https://github.com/Mido-77/ETIS-Group.git
cd ETIS-Group
```

Follow [SETUP.md](SETUP.md) for the complete installation instructions, including Python dependencies, database configuration, and frontend setup.

## Project Structure

- `backend/` — FastAPI application and tests
- data/ — Dataset and data-processing scripts
- docs/ — Documentation
- etis-extension/ — Browser extension
- etis-frontend/ — React dashboard
- evaluation/ — Evaluation scripts
- ml/ — Trained models and machine-learning assets

## Security

- Keep private environment files and credentials out of Git.
- Never put database passwords in frontend environment variables.

## License

No license has been specified yet.


