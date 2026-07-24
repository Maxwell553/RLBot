# MarketTrainer frontend

React + TypeScript + Tailwind CSS frontend for MarketTrainer. It has three
surfaces:

- `/` — public product and methodology overview.
- `/portal` — investor portal for mandate creation, build requests, status
  tracking, and released reports.
- `/ops` — Research Operations console with run artifacts, provenance, preflight, YAML
  export, and CLI handoff.

The source is tracked in this repository and tested in CI
(`.github/workflows/frontend.yml`: lint, typecheck, unit tests, build).

## Run locally

```bash
cd frontend
npm install       # Node >= 22 (see .nvmrc)
npm run dev       # starts missing local APIs, then http://localhost:5173
```

`npm run dev` starts (or reuses a healthy) research API on `:8787` and workflow
API on `:8790`, waits until they respond, then serves the UI at
http://localhost:5173. Use `npm run dev:ui` only when both APIs are managed
separately.

Verification commands (all must pass before committing):

```bash
npm run lint
npm run typecheck
npm test
npm run build
```

## Data connectivity

The two application surfaces use separate services:

- `scripts/frontend_api.py` is a local, read-only adapter over research
  artifacts for Research Operations.
- `scripts/workflow_api.py` owns tenant-scoped mandate workflow records in
  SQLite. It never stores product records under `Runs/`.

```bash
# repo root, in the project venv
pip install -e ".[api]"
python scripts/frontend_api.py --port 8787

# separate terminal; example local actor tokens
export MARKETTRAINER_WORKFLOW_TOKENS='{
  "investor-local":{"user_id":"investor_1","org_id":"org_1","role":"investor"},
  "operator-local":{"user_id":"operator_1","org_id":"internal","role":"operator"}
}'
export MARKETTRAINER_PAYMENT_WEBHOOK_SECRET='replace-me'
python scripts/workflow_api.py --port 8790

cp frontend/.env.example frontend/.env.local
```

The `/ops` dashboard, runs, and results pages then read live `Runs/` artifacts
(manifests, audit records, `backtest_summary.json`,
`cohort_vs_benchmark.json`) with full provenance. If the configured API fails,
pages show an error state with retry. Optional shared-secret auth:
set `MARKETTRAINER_API_TOKEN` for the API process and `VITE_API_TOKEN` in
`.env.local`.

The operator dashboard uses one narrow `/api/dashboard` startup request.
`/api/runs` supports server-side status/search filtering and pagination (25
rows per frontend page).

## Investor portal

- Mandate builder supports searching the instrument catalog and adding custom
  symbols to the universe (5–55 instruments).
- Symbol lookup aids request entry; it is not proof of training eligibility.
- The workflow server owns IDs, timestamps, state, quotes, payment state, and
  immutable mandate versions. Browser-created lifecycle fields are rejected.
- Records are scoped to the authenticated organization; there is no
  `localStorage` or shared `Runs/frontend_build_requests.json` fallback.
- The enforced lifecycle is draft → preflight passed → quote issued → checkout
  → verified payment → queued → training → validation → governed OOS
  evaluation → released.
- Payment verification is webhook-only. OOS authorization is a controlled,
  audited workflow transition and does not execute a backtest from the UI.

## Mandate builder guarantees (ops)

- Only engine-enforced fields (universe, costs, capital, global asset cap,
  stop loss, budget, reproducibility, benchmark) are written into active YAML.
- Per-asset annual holding costs are preserved from the base config.
- Validation enforces 5–55 unique asset ids and tickers before export.
- Engine preflight via `POST /api/preflight` when the research API is connected.

## Deployment

SPA fallbacks are included for Netlify (`public/_redirects`) and Vercel
(`vercel.json`). Fonts are self-hosted via Latin-only `@fontsource` weight
files — no third-party font CDN at runtime. `/ops` and its visual mode switch
are disabled in production unless explicitly enabled for an internal build.
Those build flags are not authorization; both APIs still enforce their own
credentials.
