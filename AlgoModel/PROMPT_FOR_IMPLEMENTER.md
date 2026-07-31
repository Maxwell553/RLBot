# Paste this to another LLM (with this whole FINALMODEL folder attached)

---

Implement the locked trading strategy defined in this folder into my **existing paper-trading environment**.

## Instructions for the implementer
1. Read `STRATEGY_SPEC.md` end-to-end — that is the source of truth.
2. Read `config/strategy.locked.json` for frozen parameters.
3. Use `reference/signal_engine.py` as the reference implementation of signal → target weights (adapt to my codebase; do not invent new parameters).
4. Use `data/pit/sp500_historical_components.csv` for point-in-time S&P membership.
5. Wire signals into **my existing paper broker / order loop**. Do not build a new backtest stack unless required for smoke tests.
6. Do **not** change locked knobs (`top_n`, `lookback`, `skip`, `blend_weight`, `exec_lag_days`).
7. Constraints: **long-only, cash account, no shorts, no margin, no futures**.
8. On each new trading day after a month-end signal, submit orders so the account matches target weights at the next session (execution lag = 1 trading day).
9. If something in my paper env conflicts with the spec, prefer the spec for strategy logic and adapt only the broker adapter.

## Deliverables I want from you
- Code that computes monthly target weights from prices + PIT file
- Paper-env integration that places/cancels orders to hit those weights
- A short note: which files you changed and how to run one paper day

## Do not
- Retrain or re-grid-search parameters
- Use today’s S&P list instead of the PIT CSV
- Trade the same bar the signal is computed on
- Add leverage or shorting

---
