# Research Notes

This document is the tracked, text-only research ledger for MarketTrainer. Raw
run trees live under `Runs/`, which is gitignored, so result tables here must be
self-contained. Do not rely on links or embedded images from `Runs/<run_id>/`
for durable reporting.

For implementation and operations, see [README.md](../README.md),
[TRAINING.md](TRAINING.md), and [MODAL.md](MODAL.md). This file focuses on the
walk-forward protocol, OOS results, and interpretation caveats.

## Protocol

- **Model:** RecurrentPPO `MlpLstmPolicy`.
- **Universe:** config-driven tradeable assets, default `N = 10`.
- **Observation/action:** `obs_dim = 10N + 28`; action is cash plus `N` risky
  logits, projected to a long-only capped simplex.
- **Data split:** chronological OOS holdout is reserved before any train/eval
  split. Default feature mode is `independent`.
- **Checkpoint selection:** `models/best/best_model.zip` is selected after the
  fee/churn ramp by robust benchmark-relative eval score, not holdout
  performance.
- **Eval benchmark:** `training.best_model_benchmark`, currently
  `equal_weight_daily`. This is separate from `universe.benchmark`, which is
  only the reporting sleeve used for benchmark-only buy-and-hold / 60-40 plots.
- **Reward benchmark:** `reward.benchmark_cap_weights`, currently an equal
  feasible passive book.
- **Backtest:** `scripts/backtest.py --run-id <RUN_ID> --checkpoint best` loads
  the run-local config, data snapshot, model, and matched `VecNormalize` state.
- **OOS burn:** manual and research-launched holdout reads are recorded in
  `Runs/oos_ledger.jsonl`; because that ledger is local and gitignored, copied
  tables here should still describe the trial context.

## Walk-Forward Windows

| Window | Train through | OOS holdout | Status |
| --- | --- | --- | --- |
| W1 | 2015-12-31 | 2016-01-01 to 2017-12-31 | Active research |
| W2 | 2017-12-31 | 2018-01-01 to 2019-12-31 | Active research |
| W3 | 2019-12-31 | 2020-01-01 to 2021-12-31 | Active research |
| W4 | 2021-12-31 | 2022-01-01 to 2023-12-31 | Active research |
| W5 | 2023-12-31 | 2024-01-01 to 2025-12-31 | Active research |
| W6 | 2025-12-31 | 2026-01-01 to 2027-12-31 | Embargoed terminal validation |

## Current Published Cohorts

The current complete cohort is `W*_615`: one seed (42) per window, 50M steps,
`feature_split_mode: independent`, `obs_lag: 1`, `max_single_asset_weight: 0.20`,
turnover penalty `0.007`, equal passive reward benchmark, robust eval selection
after `fee_ramp_end`, and `reward.exposure_risk_penalty_scale: 90`. A follow-on
cohort `W*_616` (`exposure_risk_penalty_scale: 80`, same protocol) is in progress
(W1-W3 complete; W4 training, W5 pending) and is not yet a published result.

Both cohorts run under the reworked reward structure introduced after the earlier
`W*_612/613/614` exposure-scale sweep. Those older cohorts are superseded and
their numbers are not comparable to the tables below.

| Cohort | Exposure risk scale | Status | Chained return | Mean Sharpe | Mean max DD | Beat equal-weight |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `W*_615` | 90 | Complete (5/5) | +135.2% (W1-W5) | 0.84 | -15.0% | 3/5 |
| `W*_616` | 80 | In progress (3/5) | +56.4% (W1-W3 only) | 0.80 (W1-W3) | -13.1% (W1-W3) | 2/3 |

The `W*_616` row covers only W1-W3 and must not be read as a cohort result until
W4-W5 complete. Numbers below are copied from each run's
`Runs/<run_id>/backtest_summary.json` (`--checkpoint best`).

## Per-Window OOS Results

### Cohort `W*_615` (`exposure_risk_penalty_scale: 90`, seed 42) — complete

| Window | Agent return | Sharpe | Max DD | DSR | Equal-weight | SP500 sleeve | 60/40 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W1 | +26.7% | 1.42 | -6.7% | 0.70 | +28.4% | +46.4% | +25.7% |
| W2 | +10.9% | 0.49 | -15.1% | 0.22 | +3.0% | +18.7% | +16.9% |
| W3 | +17.8% | 0.60 | -21.6% | 0.33 | +17.1% | +52.9% | +31.9% |
| W4 | +1.6% | 0.07 | -17.9% | 0.11 | +5.3% | +7.3% | +1.3% |
| W5 | +39.9% | 1.60 | -13.6% | 0.80 | +33.2% | +45.7% | +29.6% |

### Cohort `W*_616` (`exposure_risk_penalty_scale: 80`, seed 42) — in progress

| Window | Agent return | Sharpe | Max DD | DSR | Equal-weight | SP500 sleeve | 60/40 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W1 | +28.4% | 1.57 | -5.1% | 0.73 | +28.4% | +46.4% | +25.7% |
| W2 | +3.3% | 0.17 | -16.1% | 0.09 | +2.7% | +18.0% | +16.4% |
| W3 | +17.9% | 0.66 | -18.2% | 0.32 | +17.3% | +52.5% | +31.9% |
| W4 | training | — | — | — | — | — | — |
| W5 | pending | — | — | — | — | — | — |

## Interpretation

The current evidence suggests the environment can learn useful allocation
behavior: in `W*_615` the agent beats equal-weight on chained return
(+135.2% vs the per-window equal-weight book) and on 3/5 windows, with strong
risk-adjusted profiles in W1 and W5 (Sharpe 1.42 and 1.60). It is not yet
publication-ready or tradeable evidence on its own.

Main caveats:

- Single seed (42) per window is too thin for a stochastic RL claim.
- W1-W5 have now been used repeatedly; per-window DSR (shown above) stays well
  below the usual 0.95 bar, and trial counts per window are already 6-10.
- The agent still lags the SP500 sleeve in strong-equity windows (W1, W3, W5)
  and carries large drawdown in stress regimes (W3 -21.6%, W4 -17.9%).
- W4 (2022-2023) remains the hardest window: near-flat return and the weakest
  risk-adjusted profile in both cohorts.
- yfinance daily bars, simple transaction-cost modeling, and no capacity model
  are not sufficient for live trading claims.

Practical next steps:

1. Run at least 3 seeds for the most promising configurations before changing
   the universe.
2. Compare cohort distributions, not single-window winners.
3. Keep W6 embargoed until the method and seed protocol are fixed.
4. Consider a small tracked `docs/results/` artifact only after the result set is
   intentionally frozen.

## Training Plot Interpretation

Training plots are diagnostics, not OOS performance estimates. The eval panels
come from validation blocks inside the training period; they are useful for
checkpoint selection and failure detection, but not as proof of generalization.

The robust score is:

```text
score =
  (1 - blend) * mean(segment excess NAV)
  + blend * stitched_excess_nav
  - std_coef * std(segment excess NAV)
  - dd_coef * p75(max drawdown NAV)
```

With current defaults:

```text
blend = 0.5
std_coef = 0.75
dd_coef = 2.0
benchmark = equal_weight_daily
```

Negative robust scores are expected when the agent is behind the benchmark after
dispersion and drawdown penalties. A downward robust-score line means later
checkpoints are less attractive under the selection rule, even if training
reward or raw episode NAV remains high.