# Audit notes (research integrity)

## Current deliverable
This folder is a **paper-implementation handoff pack** for `pit_cross_sectional_momentum_v1`.

Implement from: `PROMPT_FOR_IMPLEMENTER.md` → `STRATEGY_SPEC.md` → `reference/signal_engine.py`.

## Integrity controls baked into the strategy
- Point-in-time S&P membership (not today’s list)
- Parameters frozen on train (2010–2017); OOS not used to pick knobs
- Signal on month-end close; orders next session (lag 1)
- Long-only cash; 90% invested / 10% cash
- Missing-price policy: exit at last good mark

## Residual research limitation
Backtest prices came from a Yahoo-style DB that still under-represents true delists. Membership look-ahead is fixed; some delist wipeouts may still be missing from historical research metrics in `SUMMARY.json`.

## Not for implementers
`_archive/` — old Gatev cash-RV / survivor-momentum experiments. Ignore.
