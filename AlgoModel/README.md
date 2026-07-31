# FINALMODEL — handoff pack for paper trading

Give **this entire folder** to another LLM (or engineer) with:

> See `PROMPT_FOR_IMPLEMENTER.md` — implement this locked strategy into my existing paper environment.

That is the intended use. You do **not** need the rest of the ALTrade repo or `statarb.db` for paper integration.

---

## Start here

| File | Role |
|---|---|
| **`PROMPT_FOR_IMPLEMENTER.md`** | Copy/paste prompt for the implementing LLM |
| **`STRATEGY_SPEC.md`** | Full deterministic strategy rules (source of truth) |
| **`config/strategy.locked.json`** | Frozen parameters |
| **`data/pit/sp500_historical_components.csv`** | Point-in-time S&P membership |
| **`reference/signal_engine.py`** | Standalone Python reference (stdlib only) |
| `SUMMARY.json` / `AUDIT.md` | Research metrics & integrity notes |
| `logs/equity_curve.csv` | Research backtest equity (optional reference) |

`_archive/` holds superseded research artifacts — **ignore for implementation**.

---

## Strategy in one paragraph

Cash, long-only. At each **month-end**, among names in the S&P **as of that date** (PIT CSV), rank by 189-trading-day return skipping the most recent 21 days, take the **top 30**, assign **equal weight totaling 90%** of equity (**10% cash**). Place orders on the **next trading day** (lag 1). No shorts, no margin.

---

## What the implementer must wire

1. Load PIT membership + daily prices from **your** paper data feed  
2. Call `compute_target_weights(...)` (or reimplement per `STRATEGY_SPEC.md`)  
3. On the session after month-end, send orders so the account matches those weights  

Do not retune parameters.
