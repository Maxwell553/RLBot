# Research Notes

This document is the tracked, text-only research ledger for MarketTrainer. Raw
run trees live under `Runs/`, which is gitignored, so result tables here are self contained. 

For implementation and operations, see [README.md](../README.md),
[TRAINING.md](TRAINING.md), and [MODAL.md](MODAL.md). This file focuses on the walk-forward protocol, OOS results, and interpretations.

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


| Window | Train through | OOS holdout              | Status                        |
| ------ | ------------- | ------------------------ | ----------------------------- |
| W1     | 2015-12-31    | 2016-01-01 to 2017-12-31 | Active research               |
| W2     | 2017-12-31    | 2018-01-01 to 2019-12-31 | Active research               |
| W3     | 2019-12-31    | 2020-01-01 to 2021-12-31 | Active research               |
| W4     | 2021-12-31    | 2022-01-01 to 2023-12-31 | Active research               |
| W5     | 2023-12-31    | 2024-01-01 to 2025-12-31 | Active research               |
| W6     | 2025-12-31    | 2026-01-01 to 2027-12-31 | Embargoed terminal validation |


## Current Published Cohorts

Six complete walk-forward cohorts (`W*_612` … `W*_617`) form a **3 × 2 grid** over
`reward.exposure_risk_penalty_scale` (80 / 90 / 100) and training seed (0 / 42).
All runs use 50M steps, `feature_split_mode: independent`, `obs_lag: 1`,
`max_single_asset_weight: 0.20`, turnover penalty `0.007`, equal passive reward
benchmark, and robust eval selection after `fee_ramp_end`.

Shared protocol except where noted:


| Cohort   | Exposure scale | Seed | Status         | Chained return | Mean Sharpe | Mean max DD | Beat equal-weight |
| -------- | -------------- | ---- | -------------- | -------------- | ----------- | ----------- | ----------------- |
| `W*_612` | 80             | 0    | Complete (5/5) | +140.7%        | 1.12        | -10.7%      | 3/5               |
| `W*_616` | 80             | 42   | Complete (5/5) | +82.1%         | 0.81        | -11.1%      | 2/5               |
| `W*_614` | 90             | 0    | Complete (5/5) | +110.7%        | 0.66        | -16.0%      | 2/5               |
| `W*_615` | 90             | 42   | Complete (5/5) | +135.2%        | 0.84        | -15.0%      | 3/5               |
| `W*_613` | 100            | 0    | Complete (5/5) | +186.2%        | 1.05        | -11.2%      | 4/5               |
| `W*_617` | 100            | 42   | Complete (5/5) | +135.5%        | 0.86        | -15.5%      | 2/5               |


**Cohort caveats (read before comparing rows):**

- **`W*_612` is mixed-era:** W1–W2 trained on pre-rebalance code (`fe6d923`, NAV-based
best checkpoint, cap-weighted reward benchmark); W3–W5 on post-rebalance code
(`076137e`). Treat 612 as exploratory, not a clean replication cell.
- **`W*_615` is split mid-cohort:** W1–W2 on `076137e`; W3–W5 on `a2cc773` (eval
cadence fields added to the run snapshot). Exposure scale stayed at 90 throughout.
- **612–614** were an exposure-scale sweep at seed 0; **615–617** repeat the
90 / 80 / 100 grid at seed 42. **`W*_617`** completes the seed-42 row (100 was
missing until Jun 2026).

Numbers below are copied from each run's `Runs/<run_id>/backtest_summary.json`
(`--checkpoint best`). Equal-weight and SP500 sleeve columns are window-specific
reporting benchmarks (identical across cohorts for a given window).

### Exposure × seed grid (chained W1–W5 return)


|                  | Seed 0        | Seed 42       |
| ---------------- | ------------- | ------------- |
| **Exposure 80**  | 612 (+140.7%) | 616 (+82.1%)  |
| **Exposure 90**  | 614 (+110.7%) | 615 (+135.2%) |
| **Exposure 100** | 613 (+186.2%) | 617 (+135.5%) |


At seed 42, exposure 100 (`617`) ties exposure 90 (`615`) on chained return and
edges it on mean Sharpe; exposure 80 (`616`) is clearly weaker (W4 negative).
At seed 0, exposure 100 (`613`) is the strongest cell overall.

## Per-Window OOS Results

### Cohort `W*_615` (`exposure_risk_penalty_scale: 90`, seed 42)


| Window | Agent return | Sharpe | Max DD | DSR  | Equal-weight | SP500 sleeve |
| ------ | ------------ | ------ | ------ | ---- | ------------ | ------------ |
| W1     | +26.7%       | 1.42   | -6.7%  | 0.70 | +28.4%       | +46.4%       |
| W2     | +10.9%       | 0.49   | -15.1% | 0.22 | +3.0%        | +18.7%       |
| W3     | +17.8%       | 0.60   | -21.6% | 0.33 | +17.1%       | +52.9%       |
| W4     | +1.6%        | 0.07   | -17.9% | 0.11 | +5.3%        | +7.3%        |
| W5     | +39.9%       | 1.60   | -13.6% | 0.80 | +33.2%       | +45.7%       |


### Cohort `W*_616` (`exposure_risk_penalty_scale: 80`, seed 42)


| Window | Agent return | Sharpe | Max DD | DSR  | Equal-weight | SP500 sleeve |
| ------ | ------------ | ------ | ------ | ---- | ------------ | ------------ |
| W1     | +28.4%       | 1.57   | -5.1%  | 0.73 | +28.4%       | +46.4%       |
| W2     | +3.3%        | 0.17   | -16.1% | 0.09 | +2.7%        | +18.0%       |
| W3     | +17.9%       | 0.66   | -18.2% | 0.32 | +17.3%       | +52.5%       |
| W4     | -5.7%        | -0.37  | -12.4% | 0.04 | +5.3%        | +7.3%        |
| W5     | +23.5%       | 1.99   | -3.9%  | 0.94 | +33.2%       | +45.7%       |


### Cohort `W*_617` (`exposure_risk_penalty_scale: 100`, seed 42)


| Window | Agent return | Sharpe | Max DD | DSR  | Equal-weight | SP500 sleeve |
| ------ | ------------ | ------ | ------ | ---- | ------------ | ------------ |
| W1     | +26.4%       | 1.60   | -6.1%  | 0.70 | +28.4%       | +46.4%       |
| W2     | +10.7%       | 0.48   | -17.2% | 0.15 | +2.7%        | +18.0%       |
| W3     | +38.8%       | 1.09   | -23.6% | 0.49 | +17.3%       | +52.5%       |
| W4     | +2.0%        | 0.11   | -17.5% | 0.08 | +5.3%        | +7.3%        |
| W5     | +18.9%       | 1.02   | -13.2% | 0.45 | +33.2%       | +45.7%       |


### Seed 42 head-to-head (return by window)


| Window | 615 (90)   | 616 (80)   | 617 (100)  |
| ------ | ---------- | ---------- | ---------- |
| W1     | +26.7%     | **+28.4%** | +26.4%     |
| W2     | **+10.9%** | +3.3%      | +10.7%     |
| W3     | +17.8%     | +17.9%     | **+38.8%** |
| W4     | +1.6%      | -5.7%      | **+2.0%**  |
| W5     | **+39.9%** | +23.5%     | +18.9%     |


## Interpretation

The six cohorts show the environment can learn useful allocation behavior, but
results are **highly scale-dependent**. No single exposure setting wins
every window at seed 42; chained return ranges from +82% (616) to +136% (615/617)
under the same seed with only the exposure knob changed.

Main caveats:

- **Two seeds per scale is still thin** for a stochastic RL claim; treat the grid
as directional, not definitive.
- W1–W5 have been read many times; per-window DSR stays well below the usual 0.95
bar, and ledger trial counts are roughly **10-13 distinct models per window**
after the 617 backtests.
- The agent still lags the SP500 sleeve in strong-equity windows (W1, W3, W5) and
carries large drawdown in stress regimes (e.g. W3 -23.6% for 617).
- W4 (2022–2023) remains hard: near-flat returns except 617 (+2.0%) and 616
(-5.7% at exposure 80).
- yfinance daily bars, simple transaction-cost modeling, and no capacity model
are not sufficient for live trading claims.

Practical next steps:

1. Add **third seeds** for the best scales (90 and/or 100 at seed 42) before
  changing the universe.
2. Compare **cohort distributions**, not single-window winners.
3. Keep W6 embargoed until the method and seed protocol are fixed.
4. Archive `Runs/oos_ledger.jsonl` and frozen config snapshots when freezing a
  recipe for external reporting.

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
