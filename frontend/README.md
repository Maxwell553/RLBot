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

One command from the repo root (installs frontend deps if needed, then starts
the research API on `:8787`, workflow API on `:8790`, and Vite on `:5173`):

```bash
bash scripts/start_ui.sh
```

Or from `frontend/`:

```bash
npm install       # Node >= 22 (see .nvmrc); first time only
npm run dev       # same stack as start_ui.sh → http://127.0.0.1:5173
```

`npm run dev` publishes `frontend/public/data/*.json` snapshots, starts (or
reuses) an optional research API on `:8787` and workflow API on `:8790`, then
serves the UI at http://127.0.0.1:5173. **Ops page GETs load those static JSON
files** (milliseconds) — they do not wait on `:8787`. The Vite `/api` proxy is
only needed for preflight and forced forward refresh. Use `npm run dev:ui`
when APIs are managed separately (still run `python3 scripts/publish_frontend_data.py` first).

On iCloud Desktop, reading `frontend/node_modules` can stall Vite for minutes.
`npm run dev` therefore runs **`npm ci` into `/tmp/markettrainer-frontend`**,
symlinks `frontend/node_modules` → that install (moving any real tree aside to
`frontend/.node_modules.icloud`), and caches Vite prebundles under
`/tmp/markettrainer-vite-cache`. First start after a lockfile change installs
once; later starts should open the UI in a few seconds. Set
`MARKETTRAINER_SKIP_NM_CLONE=1` to leave in-tree `node_modules` alone.

Verification commands (all must pass before committing):

```bash
npm run lint
npm run typecheck
npm test
npm run build
```

## Data connectivity

Ops pages default to **static snapshots** (`VITE_DATA_SOURCE=static`):

```bash
# repo root — rebuilds frontend/public/data/{dashboard,runs,results,forward,...}.json
python3 scripts/publish_frontend_data.py
```

`dev.mjs` runs this on boot (~100ms from `execution/api_*_cache.json`). The SPA
fetches `/data/*.json` through Vite (typically 1–5ms warm) and keeps an
in-memory cache across navigations. Filtering/pagination for runs happens in
the browser.

Optional live API (can stall on iCloud Desktop when scanning `Runs/`):

- `VITE_DATA_SOURCE=api` + `scripts/frontend_api_lite.py` (default) or
  `frontend_api.py` on `:8787`
- `scripts/workflow_api.py` owns tenant-scoped mandate workflow records in
  SQLite (portal). It never stores product records under `Runs/`.

```bash
# optional research API (preflight / force-forward only when using static data)
python3 scripts/frontend_api_lite.py --port 8787

# separate terminal; example local actor tokens
export MARKETTRAINER_WORKFLOW_TOKENS='{
  "investor-local":{"user_id":"investor_1","org_id":"org_1","role":"investor"},
  "operator-local":{"user_id":"operator_1","org_id":"internal","role":"operator"}
}'
export MARKETTRAINER_PAYMENT_WEBHOOK_SECRET='replace-me'
python scripts/workflow_api.py --port 8790

cp frontend/.env.example frontend/.env.local
```

Set `VITE_DATA_SOURCE=offline` for the labeled synthetic sandbox. Optional
shared-secret auth for API mode: `MARKETTRAINER_API_TOKEN` + `VITE_API_TOKEN`.

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
