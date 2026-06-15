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

The latest copied cohort comparison covers `W*_612`, `W*_613`, and `W*_614`.
Each cohort is one seed per window, 50M steps, `feature_split_mode:
independent`, `max_single_asset_weight: 0.20`, turnover penalty `0.007`, equal
passive reward benchmark, and robust eval selection after `fee_ramp_end`.

The intended single changed knob was `reward.exposure_risk_penalty_scale`:

| Cohort | Exposure risk scale | Chained W1-W5 return | Mean Sharpe | Mean max DD | Beat equal-weight |
| --- | ---: | ---: | ---: | ---: | ---: |
| `W*_612` | 80 | +140.7% | 1.12 | -10.7% | 3/5 |
| `W*_613` | 100 | +186.2% | 1.05 | -11.2% | 4/5 |
| `W*_614` | 90 | +110.7% | 0.66 | -16.0% | 2/5 |

The exposure-scale sweep is exploratory. The middle setting underperformed both
80 and 100, so the stronger 612/613 outcomes should be treated as promising but
not yet stable evidence.

## Per-Window OOS Results

### Cohort `W*_612` (`exposure_risk_penalty_scale: 80`)

| Window | Agent return | Sharpe | Max DD | DSR | Equal-weight | SP500 sleeve | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| W1 | +22.0% | 1.41 | -5.3% | 0.78 | +28.4% | +46.4% | Trailed strong equity regime |
| W2 | +9.6% | 0.58 | -10.8% | 0.35 | +3.0% | +18.7% | Beat equal-weight, lagged 60/40/SP500 |
| W3 | +33.6% | 0.88 | -26.3% | 0.88 | +17.1% | +52.9% | Strong return, drawdown too high |
| W4 | +8.1% | 0.42 | -8.9% | 0.72 | +5.3% | +7.3% | Beat listed benchmarks on return |
| W5 | +24.7% | 2.30 | -2.3% | 0.999 | +33.2% | +45.7% | Excellent risk-adjusted profile, lower raw return |

### Cohort `W*_613` (`exposure_risk_penalty_scale: 100`)

| Window | Agent return | Sharpe | Max DD | DSR | Equal-weight | SP500 sleeve | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| W1 | +29.5% | 1.51 | -6.7% | 0.79 | +28.4% | +46.4% | Beat equal-weight, lagged SP500 |
| W2 | +1.0% | 0.07 | -12.3% | 0.12 | +3.0% | +18.7% | Weak window |
| W3 | +54.4% | 1.93 | -6.7% | 0.96 | +17.1% | +52.9% | Best cohort/window result |
| W4 | +5.8% | 0.23 | -17.5% | 0.30 | +5.3% | +7.3% | Higher drawdown than desired |
| W5 | +34.0% | 1.52 | -12.8% | 0.928 | +33.2% | +45.7% | Beat equal-weight on return, worse risk than 612 |

### Cohort `W*_614` (`exposure_risk_penalty_scale: 90`)

| Window | Agent return | Sharpe | Max DD | DSR | Equal-weight | SP500 sleeve | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| W1 | +28.4% | 1.38 | -7.1% | 0.70 | +28.4% | +46.4% | Near equal-weight return |
| W2 | +0.1% | 0.01 | -18.0% | 0.08 | +3.0% | +18.7% | Poor |
| W3 | +20.7% | 0.53 | -26.4% | 0.33 | +17.1% | +52.9% | Return acceptable, drawdown poor |
| W4 | +7.9% | 0.30 | -12.8% | 0.22 | +5.3% | +7.3% | Beat listed benchmarks on return |
| W5 | +25.9% | 1.06 | -15.7% | 0.61 | +33.2% | +45.7% | Lagged equity-heavy benchmarks |

## Interpretation

The current evidence suggests the environment can learn useful allocation
behavior: across these cohorts it often beats equal-weight, sometimes beats the
SP500 sleeve, and can materially reduce drawdown in some regimes. It is not yet
publication-ready or tradeable evidence on its own.

Main caveats:

- Single seed per cohort/window is too thin for a stochastic RL claim.
- W1-W5 have now been used repeatedly; DSR helps, but local trial accounting is
  incomplete for informal experiments before the ledger existed.
- The best cohort is not monotonic in the tested exposure-risk scale, which
  weakens the case that the result is a clean knob effect.
- The strategy can still underperform simple passive books in strong equity
  regimes and can carry unacceptable drawdown in stress regimes.
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

## Suggested Plot Improvements

Concise changes that would make `training.png` more useful:

1. Add a vertical marker at the current `best_eval_step` on every eval panel,
   with a small label like `best @ 31.0M`.
2. Plot the running maximum robust score as a faint line, so the selected
   checkpoint is visually obvious even when the raw robust score trends down.
3. Rename the mean-NAV eval panel to `validation diagnostics` and avoid implying
   it estimates OOS return.
4. Replace or de-emphasize stitched validation NAV unless it is used directly in
   the selection score; show `stitched excess vs eval benchmark` as a compact
   subplot or annotation instead.
5. Add an optional small text box with the current score formula and benchmark:
   `0.5 segment excess + 0.5 stitched excess - 0.75 std - 2.0 p75 DD`.
6. Show post-gate region shading (`fee_ramp_end` onward), because only that
   region can update `models/best/`.
7. Add eval diagnostics that have explained failures in practice: mean cash,
   effective N, top-3 concentration, cap-hit fraction, and turnover.

## Result Reporting Rules

- Copy headline metrics into this tracked document; do not point readers at
  `Runs/` as the only source.
- If a plot is needed in a durable report, export it to a tracked report
  directory, not `Runs/`.
- Always state checkpoint (`best`), stochastic path count if used, cohort
  changes, and trial/seed context.
- Never compare runs without checking their snapshotted configs.
