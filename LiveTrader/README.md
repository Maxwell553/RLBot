# LiveTrader

GeneralEquity1 (prod_return_alpha_v3) wired to Interactive Brokers.

The locked pack `GeneralEquity1/data/bars.db` still ends **2026-07-29**. LiveTrader computes the same formulas on the **Yahoo daily panel** (always refreshed unless `--no-refresh-data`) and sizes/fills at IBKR. `python LiveTrader/trader.py verify-data` prints both.

Default mode is `dry_run`. Live capital stays off until `allow_live`, port 7496/4001, `LIVE_TRADER_CONFIRM=GE1`, and `--arm-live`.

## Book

- 58% sleeve A, **Friday close**: TQQQ + QQQ
- 42% sleeve B, **month-end close**: GLD vs TLT (else cash)
- First empty GE1 book: **seed both sleeves** (`seed_if_flat`)
- Residual stays **cash** (not BIL)
- Fractional qty → **MKT**; whole shares → **MOC** (IBKR rejects fractional MOC)

## Safety wired in

- Idempotency: `last_trade_date` + journal `(account, asof)` + working IB orders
- Size on IB last when connected; cap buys at cash/buying power; `whatIf` before place
- Wait for MKT fills (MOC only waits until accepted); `reconcile` vs targets
- `flatten --confirm-flatten` kill switch; `cancel-open`
- `run-if-due` no-ops unless Friday / month-end / seed
- Submit requires `IBKR_ACCOUNT`

## Commands

```bash
pip install -e ".[live]"
cp LiveTrader/.env.example LiveTrader/.env   # set IBKR_ACCOUNT

python LiveTrader/trader.py verify-data
python LiveTrader/trader.py plan
python LiveTrader/trader.py dry-run
python LiveTrader/trader.py preflight --offline

# TWS/Gateway paper (7497), API on, fractionals on, GE1 names only:
python LiveTrader/trader.py snapshot
python LiveTrader/trader.py preflight
python LiveTrader/trader.py dry-run --connect
# execution.mode: paper
python LiveTrader/trader.py paper-submit --preview-only
python LiveTrader/trader.py paper-submit --if-due

python LiveTrader/trader.py reconcile
python LiveTrader/trader.py flatten              # preview
python LiveTrader/trader.py flatten --confirm-flatten
python LiveTrader/trader.py run-if-due --preview-only
bash LiveTrader/install_run_if_due_launchd.sh    # weekdays 15:45 local
```

## Live (after paper fills match)

1. Flatten leftover names
2. `execution.mode: live`, port **7496** / **4001**, `allow_live: true`
3. `LIVE_TRADER_CONFIRM=GE1`
4. `python LiveTrader/trader.py live-submit --arm-live --if-due`

Launchd will not arm live unless `LIVE_TRADER_LAUNCHD_LIVE=1` is also set.

Do not retune `ge1_strategy.P`. Do not reshape overnight or intraday.
