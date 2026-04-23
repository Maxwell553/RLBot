# IBKR paper trading (TWS) — RL portfolio bot

This folder runs the same **yfinance + 98-d observation + VecNormalize + RecurrentPPO** path as [paper_trade/paper_trade.py](../paper_trade/paper_trade.py), then (optionally) sends **MKT** orders to **Interactive Brokers** via a local **TWS** session and the **[ib_insync](https://github.com/erdewit/ib_insync)** client.

**Default model:** `65M_4_20_26_a` — `models/65M_4_20_26_a/best/best_model.zip` and `.../vec_normalize.pkl`.

## Security

- **Do not** put IB passwords, 2FA codes, or API secrets in the repo, env files that you commit, or in chat. Log in to **TWS** yourself; the script only opens a local API socket to an already-authenticated session.
- Connection settings use optional env vars: `IB_HOST` (default `127.0.0.1`), `IB_PORT` (default `7497` for paper TWS), `IB_CLIENT_ID` (default `7`), `IB_CONNECT_TIMEOUT` (default `25` seconds). Pick a **unique client ID** if multiple apps connect at once. With **many** paper sub-accounts, raise `IB_CONNECT_TIMEOUT` (e.g. `40`) if you see *positions / account updates … request timed out* at connect.

## One-time: TWS (paper) + API

1. Log in to **TWS** with your **paper** account (not live unless you know what you are doing).
2. **Configure → Settings → API → Settings**
   - Enable **ActiveX and Socket Clients**.
   - **Socket port:** paper often uses **7497**; live is often **7496** (this script prints a **warning** if you use 7496).
   - **Trusted IPs only:** add `127.0.0.1` and ensure your client runs on the same machine.
3. Leave TWS running while the script runs.

## Install (extra dependency)

From the **repository root** (with the same venv you use for training):

```bash
pip install -r ibkr_paper/requirements.txt
```

(Only adds `ib_insync`.)

## Execution vs. training (proxy caveat)

**Features** are still built from the same yfinance series as training (^N225, ^FTSE, HG=F, etc.). **Orders** by default go to **liquid US-style proxies** defined in [contracts.json](contracts.json) (e.g. **EWJ** for Japan, **EWU** for UK, **CPER** for copper). There is **tracking error** between the model’s notional for an index/continuous future and the ETF you hold; adjust `contracts.json` only if you understand the trade-offs.

## Commands

**Dry run (no TWS, no orders, no `state.json` write):** uses default `NetLiquidation` stand-in **$5,000,000** unless you set `--use-fixed-nlv`.

```bash
.venv/bin/python ibkr_paper/run_ibkr_paper.py --dry-run
.venv/bin/python ibkr_paper/run_ibkr_paper.py --dry-run --use-fixed-nlv 5000000
```

**Connect, show account + model, but do not place orders:**

```bash
.venv/bin/python ibkr_paper/run_ibkr_paper.py --no-orders
```

**Connect and place market orders (paper):**

1. Start **TWS (paper)** with API on **7497** (or set `IB_PORT`).
2. Run:

```bash
.venv/bin/python ibkr_paper/run_ibkr_paper.py
```

- First run (or new month of data) may be slow while **yfinance** and **inference** run.
- The script appends a short block to [trade_journal.txt](trade_journal.txt) and updates [state.json](state.json) (see [.gitignore](.gitignore)) so you do not double-submit the same **last yfinance bar date** (override with `--force`).

**Common flags**

| Flag | Purpose |
|------|---------|
| `--run-id` | Which trained run to load (default `65M_4_20_26_a`) |
| `--model` / `--vec-normalize` | Override `.zip` and `vec_normalize.pkl` paths |
| `--use-fixed-nlv USD` | Size `target = weight * fixed` (still use IB for **obs** unless `--dry-run`) |
| `--min-leg-dollars` | Skip tiny deltas (default 100) |
| `--forex-lot` | Round FX rebalances to this base-currency step (default 1000) |
| `--contracts FILE` | Alternate [contracts.json](contracts.json) |

## Timing

Run **once per decision** (e.g. after the U.S. close) so the last daily bar in yfinance is stable. The script does not auto-schedule; use **cron** or a manual habit.

## Files

| File | Role |
|------|------|
| [run_ibkr_paper.py](run_ibkr_paper.py) | CLI, loads `paper_trade` inference, then IB rebalance |
| [ibkr_rebalance.py](ibkr_rebalance.py) | TWS: NLV, portfolio USD, MKT order placement |
| [contract_map.py](contract_map.py) + [contracts.json](contracts.json) | Logical asset → `ib_insync` `Stock` / `Forex` |
| [requirements.txt](requirements.txt) | `ib_insync` |
| [state.json](state.json) | Last bar + peak (local; gitignored) |
| [trade_journal.txt](trade_journal.txt) | Appended run summary (gitignored) |

## Troubleshooting

- **Cannot connect / refused:** TWS not running, API disabled, wrong port, or firewall. Confirm **Socket port** in TWS matches `IB_PORT`.
- **Error 321: API is in Read-Only mode; NetLiquidation $0; all targets $0:** In TWS, open **File → Global Configuration → API → Settings** and **uncheck Read-Only API** (wording may vary). Click **Apply**, restart TWS, run again. Read-only is for data-only clients; for account balances and (later) orders, turn it off. If NLV is still $0, add or reset **paper** cash in Account Management, or use `--use-fixed-nlv` for notional-only sizing.
- **NLV read as $0 (but TWS shows equity):** `ib_insync` puts **NetLiquidation** in ``ib.accountSummary()`` (from `reqAccountSummary`), **not** in ``ib.accountValues()`` until account-updates have streamed. The script now reads the **merged** summary + values. If it is still $0, check API session, paper funding, and **managed account** selection.
- **NLV ~1M but TWS shows ~5M (or connect timeouts, many `DU…` / `DF…` lines):** IB can send **one** **NetLiquidation** row per sub-account; taking a single *USD* line across the whole set often yields **~$1M**. The code **picks the best currency per account (BASE before USD) and sums** sub-accounts, and still prefers **sum of `ib.portfolio()` `marketValue`** (incl. CASH lines) when that beats the tags. The script no longer re-requests `reqAccountUpdates` for every account in `nlv_usd` (that duplicated connect and could trigger timeouts). If connect still times out, raise `IB_CONNECT_TIMEOUT` (e.g. `40`) and avoid TWS flapping (1100/1102) while testing.
- **Event loop / asyncio errors on Python 3.12+:** this package sets a default main-thread loop before importing `ib_insync` (see `contract_map.py` / `ibkr_rebalance.py`).
- **Error 10168 / `No market price` (IB):** no **real-time** (or enabled **delayed**) market data for that contract. Subscribe in **Account Management** / **Market data**, or use **TWS** settings to allow **delayed** quotes where free. The rebalancer uses **yfinance** last close as a **fallback to size** share/FX orders when IB returns no quote; you’ll see a one-line note when that happens. Illiquid RTH or missing yfinance history can still block a leg.

## Disclaimer

This is **educational / infrastructure** code, not financial advice. Paper trading and backtests are not live guarantees. You are responsible for order risk, size, and compliance with IBKR terms.
