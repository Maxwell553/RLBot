# Strategy specification (source of truth)

**ID:** `pit_cross_sectional_momentum_v1`  
**Account:** cash, long-only  
**Config:** `config/strategy.locked.json`  
**Membership:** `data/pit/sp500_historical_components.csv`  
**Reference code:** `reference/signal_engine.py`

This document is enough to implement the strategy in any paper-trading stack. Do not invent new parameters.

---

## 1. Goal

Each month, hold an equal-weighted basket of the **top 30** S&P names by intermediate-term momentum, sized to **90%** of equity, with **10% cash**. Membership is **point-in-time** (historical S&P list as of the signal date), not today’s constituents.

---

## 2. Locked parameters

| Param | Value |
|---|---|
| `top_n` | 30 |
| `lookback_trading_days` | 189 |
| `skip_trading_days` | 21 |
| `execution_lag_trading_days` | 1 |
| `portfolio_gross_weight` | 0.9 |
| `cash_weight` | 0.1 |
| `crash_mode` | none (always allow risk-on book) |
| `min_price` | 5.0 |
| `rebalance` | calendar month-end |

---

## 3. Calendar

1. **Signal day** = last trading day of each month (exchange calendar).
2. On signal day **after the close**, compute target weights (no orders yet).
3. **Trade day** = next trading day (`execution_lag_trading_days = 1`).
4. On trade day, place orders so positions match target weights.

---

## 4. Universe (PIT)

File format: CSV with columns `date,tickers` where `tickers` is a comma-separated list.

```text
For signal_date D:
  membership = tickers from the latest row with date <= D
  Drop any ticker matching /^.+-\d{6,}$/ (annotated delists)
  Normalize symbols to the paper broker’s symbology (e.g. BRK.B → BRK.B or BRK-B)
```

A name is **eligible** for scoring only if:
- it is in `membership`, and
- it has a valid adjusted close ≥ `min_price` on both formation start and formation end dates.

---

## 5. Momentum score

Using the trading-day price series (adjusted close preferred):

```text
Let i = index of signal_date in the trading calendar.
formation_end_index   = i - 1 - skip_trading_days
formation_start_index = formation_end_index - lookback_trading_days

score(symbol) = P[formation_end] / P[formation_start] - 1

Discard scores outside (-0.95, 5.0) or non-finite.
Rank eligible symbols by score descending.
picks = top_n symbols (or fewer if not enough).
```

**Fallback:** if `len(valid_scores) < max(5, top_n // 2)`, set target to `{ "SPY": 0.9 }` (10% cash), or all cash if SPY missing.

---

## 6. Target portfolio weights

```text
N = len(picks)
w_i = portfolio_gross_weight / N   # 0.9 / N each
cash = 1 - sum(w_i)               # 0.1
```

No shorts. Sum of long weights ≤ 0.9. Never use margin.

---

## 7. Order generation (paper)

On **trade day**, given account equity `E` and current positions:

1. Compute target notional `E * w_i` for each symbol; `E * cash` stays cash.
2. **Sell / trim first** names not in target or overweight.
3. **Buy / add** names underweight, only with available cash after sells.
4. Prefer marketable orders consistent with the paper broker (market or market-on-open).
5. If a held name has **no quote**, exit using last good price / last trade; never silently drop shares.

Between month-end rebalances, hold the names. Optional: light drift rebalance to keep ~90% invested — do **not** change the pick list mid-month.

---

## 8. Data requirements from the paper environment

| Need | Notes |
|---|---|
| Daily adjusted closes | For all tradable US equities you might hold + SPY |
| Trading calendar | NYSE-style session dates |
| Account equity & positions | For sizing |
| Order API | Submit buy/sell to reach targets |
| PIT file | Ship with this folder; keep updated if you extend live beyond CSV end |

You do **not** need ALTrade’s `statarb.db` to go live in paper — use the paper env’s market data.

---

## 9. Acceptance checks

After integration, verify:

1. No orders on signal day; orders on the next session.
2. Only PIT members as of signal date appear in picks (spot-check a few months).
3. Gross long exposure ≈ 90% after fills (within lot-size tolerance).
4. No short positions; cash ≥ 0.
5. Month with thin eligibility falls back to SPY 90% / cash 10%.

---

## 10. Research context (not required to implement)

Backtest reference (research DB, costs approximated): OOS excess vs SPY ≈ **+61%**, full max DD ≈ **−34%**. Paper will differ. See `SUMMARY.json` / `AUDIT.md`.

Parameters were frozen on 2010–2017 only; do not retune when implementing.
