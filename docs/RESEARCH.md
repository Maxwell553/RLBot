# Research Notes

This document is the tracked, text-only research ledger for MarketTrainer. Raw
run trees live under `Runs/`, which is gitignored, so result tables here are self contained. 

For implementation and operations, see [README.md](../README.md),
[TRAINING.md](TRAINING.md), and [MODAL.md](MODAL.md). This file focuses on the walk-forward protocol, OOS results, and interpretations.

## Known Issues — Contaminated Cohorts (read first)

> **Cohorts `W*_622` through `W*_625` were trained with a poorly scaled volatility
> penalty and should be discarded / re-run.** (Run snapshots confirm `622` already
> carried `vol_penalty_scale: 300.0`; `626`+ trained after the fix.)
>
> The `reward.vol_penalty_scale` term was changed (commit `12b940f`) to multiply by
> `reward_scale` (2000) **without** lowering the scale value (`300.0`), so the effective
> coefficient became `300 × 2000 = 600,000`. On typical excess downside vol this produced
> per-step penalties of roughly **600–2400 reward units**, hundreds to thousands of times
> larger than the return/benchmark/drawdown terms, so the volatility penalty dominated the
> reward signal (visible as per-step rewards spiking to ≈ -1200 in training plots).
>
> **Fix (Jun 2026):** keep the `× reward_scale` formula and set `vol_penalty_scale: 0.15`
> (`0.15 × 2000 = 300`, the pre-bug effective magnitude). Any cohort trained with a non-zero
> `vol_penalty_scale` before this fix (`W*_622`–`W*_625`) is invalid and must not be compared
> against the `W*_612`–`W*_617` published grid (which predates the volatility penalty). Re-run
> those cohorts with the rescaled penalty before drawing conclusions.

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

## Extended Cohorts (`W*_618`–`W*_621`, `W*_626`–`W*_627`)

Four exposure-extreme cohorts (60 / 120 at seeds 101 and 42, cap 0.20) and a
post-vol-fix cap experiment (`max_single_asset_weight: 0.60`,
`vol_penalty_scale: 0.15`) completed after the published grid. Numbers from each
run's `backtest_summary.json` (`--checkpoint best`):


| Cohort   | Exposure scale | Seed | Cap  | Chained return | Mean Sharpe | Mean max DD |
| -------- | -------------- | ---- | ---- | -------------- | ----------- | ----------- |
| `W*_618` | 60             | 101  | 0.20 | **+205.0%**    | 0.99        | -16.6%      |
| `W*_619` | 120            | 101  | 0.20 | +163.5%        | 0.87        | -15.6%      |
| `W*_620` | 60             | 42   | 0.20 | +102.7%        | 0.68        | -15.4%      |
| `W*_621` | 120            | 42   | 0.20 | +131.0%        | 0.78        | -16.1%      |
| `W*_626` | 80             | 0    | 0.60 | +174.7%        | 0.82        | -16.5%      |
| `W*_627` | 80             | 42   | 0.60 | +125.4%        | 0.55        | -18.5%      |


Findings across the full clean set (`612`–`621`, cap 0.20, n=50 window
backtests):

- **Exposure scale is not a stable linear lever.** Pearson correlation of
`exposure_risk_penalty_scale` vs OOS Sharpe ≈ **0.02**. `618` (+205%) and `620`
(+103%) share exposure 60 and differ only by seed — **seed variance dominates
knob variance** (deltas of tens of pp chained at fixed settings).
- **Exposure 100 is the best cell when averaged across seeds:** mean chained
return ≈ +161% (613/617) and mean Sharpe ≈ 0.96, vs ≈ +154% / 0.84 for
exposure 60 and lower for 80/90/120. It also has the best beat-equal-weight
rate (613: 4/5, mean excess +7.5%).
- **Raising the cap to 0.60 bought return at the cost of risk:** mean Sharpe
fell (0.82/0.55 vs 1.12/0.81 for matched exposure-80 cells), drawdowns
deepened, and cash collapsed to ~0% with effective N drifting to ~3.5–4.
**Cap 0.20 remains the research default.**

**Default recipe update (Jul 2026):** `config/config.yaml` now ships
`exposure_risk_penalty_scale: 100.0` (was 80.0), reflecting the strongest
seed-averaged cell in the grid above. Cap 0.20 and `vol_penalty_scale: 0.15`
are unchanged. Given the seed instability, treat this as the best available
default, not a validated edge; new claims still need ≥3 seeds across W1–W5.

**Method changes (Jul 2026, untested on OOS yet — first cohort pending):**
targeting the failure modes documented above (eval↔OOS misalignment, no
risk-off optionality, reward outlier saturation, blind self-state):

1. `training.best_model_score_mode: excess_sharpe` — checkpoint selection on
annualized daily-excess Sharpe instead of excess-NAV dollars.
2. `reward.cash_daily_yield: 0.00015` + `reward.inactivity_vix_relief: 1.0` —
cash earns ~3.8% ann. and the inactivity penalty is waived progressively
above VIX 18 (half at 27, zero at ≥ 36).
3. `reward.drawdown_amp_max: 4.0` — caps the `(1 + 12 × dd)` downside
amplification so worst-day rewards stay inside the VecNormalize clip band.
4. `environment.self_state_features: true` — 4 new obs features (realized
vol, downside vol, rolling excess vs benchmark, near-cap fraction);
`obs_dim` 128 → **132** for N=10. Old run snapshots keep their layout.

Runs trained before these changes remain backtestable unchanged (all new knobs
parse to legacy behavior when absent from a run's config snapshot), but new
cohorts must not be grid-compared against `612`–`627` — the environment and
selection rule differ.

**Cohort `W*_720` (first cohort on the Jul 2026 method changes; seed 0, cap
0.20, `vol_penalty_scale: 0.15`, `exposure_risk_penalty_scale: 100`):** chained
W1–W5 **+168.9%**, mean Sharpe **0.98**, mean max DD −13.4% — 4th/13 valid
cohorts on chained return, beats equal-weight on 4/5 windows (+51.8 pp chained
excess). W4 is the best of any cohort (Sharpe 0.55 / +11.9% in 2022–23); W5 is
near-top (1.85 / +42.8%). Versus the SPY sleeve per window it wins W4/W5, ties
W1 (1.46 vs 1.47), and loses W2 (0.30 vs 0.74) and W3 (0.73 vs 0.81, −26.3%
DD). Reward decomposition at end of training: raw return ≈ 42% of reward mass,
Sortino ≈ 20%, drawdown-level penalty ≈ 5%, vol penalty ≈ 0% (never binds —
agent downside vol sits below the equal-weight book's).

**Sharpe-focused recipe update (`721`, Jul 2026):** targeting the W2/W3 losses
above — vol-shock windows where the agent held risk through the shock:

1. `reward.risk_bonus_scale: 2.5 → 4.0` — reweight reward mass from raw return
toward the benchmark-relative Sortino differential.
2. `reward.drawdown_level_penalty: 3.0 → 6.0`, `drawdown_level_floor: 0.08 →
0.05` — sustained drawdowns bite earlier and ~2× harder (720's W3 sat at
−26% OOS with only ~5% of reward mass on this term).
3. `reward.participation_vix_relief: 1.0` (new knob, parser default 0) — the
participation bonus fades with the same VIX multiplier as the inactivity
relief, so at VIX ≥ 36 cash-vs-invested shaping is neutral and returns decide.
4. `training.best_model_score_dd_coef: 2.0 → 3.0` — checkpoint selection
penalizes p75(max_dd_frac) harder in `excess_sharpe` mode.

**Cohort `W*_721` results (721 recipe, 50M, seed 0):** chained **+146.1%**,
mean Sharpe **0.92**, mean max DD −13.8%. Per window: W1 1.80/+37.1% (best W1
of any cohort), W2 0.27/+4.4%, W3 0.78/+31.9% (**DD −29.4%**, worse than
720's −26.3%), W4 0.41/+9.5%, W5 1.34/+19.0% (72% mean cash OOS). Net vs 720:
W1 improved sharply, W3–W5 degraded; overall below 720.

**Cohort `W*_722` (same recipe at a 20M budget):** chained **+102.3%**, mean
Sharpe **0.72**. Fee ramp lands at 11.7M and best-model selection has only
~8M post-gate steps; treat 722 as a budget datapoint (50M ≫ 20M), not new
reward-shaping evidence.

**721/722 postmortem (Aug via eval logs — drove the 723 recipe):**

- **The `drawdown_level_penalty` raise was a no-op.** In
`drawdown_penalty_from_nav`, the dd-*increase* term is × `reward_scale`
(2000) but the dd-*level* term is raw `dd_excess × coefficient`. At 3–6 that
is ~0.1–1.2/step against a ±10–40/step return term; the decomp share stayed
~4–5% and W3 OOS DD worsened. All three cohorts rode W3 at ~100% gross.
**Fix: `drawdown_level_penalty: 60.0`** (≈6/step at 15% off peak — a real,
persistent de-risking gradient; the term is per-step so it integrates over
the whole time spent under water).
- **`risk_bonus_scale: 4.0` degraded training.** The Sortino diff is clipped
at ±3, so the extra weight is a constant "stay ahead" bonus (achieved on
trending train data by staying fully invested), not extra gradient. Under
4.0 the in-training eval excess-Sharpe **declined monotonically** past ~20M
steps on W3 (return signal +0.2 → −0.95, p75 dd_frac 0.047 → 0.127) and W5
(−0.07 → −0.65); under 2.5 (720) the same trajectories were flat.
**Reverted to 2.5.**
- Kept from 721: `drawdown_level_floor: 0.05`, `participation_vix_relief:
1.0`, `best_model_score_dd_coef: 3.0`.
- Open structural note: with `dr_widen_span_fraction: 0.65`, curriculum end
= min(50M, 29.25M + 32.5M) = 50M, so `early_stop_patience` can never fire
on a 50M run — declining-eval runs train to the full budget (best/ still
protects the checkpoint).

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
bar. As of Jul 2026 the ledger records roughly **16-19 distinct models
(53-76 total reads) per window** — no clean run clears DSR > 0.95, so single-window
Sharpes in the 1.5-2.0 range are not statistically decisive.
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

The robust score (legacy `best_model_score_mode: excess_nav`, used by all runs
through the `627` cohorts) is:

```text
score =
  (1 - blend) * mean(segment excess NAV)
  + blend * stitched_excess_nav
  - std_coef * std(segment excess NAV)
  - dd_coef * p75(max drawdown NAV)
```

**As of Jul 2026 the config default is `best_model_score_mode: excess_sharpe`**,
motivated by the negative correlation (~−0.13) between the NAV-dollar score and
OOS Sharpe across the clean cohorts: the return signal becomes the annualized
Sharpe of daily excess returns vs the eval benchmark (segment mean blended with
the pooled series), and the drawdown penalty uses the unitless
`p75(max_dd_frac)`:

```text
score =
  (1 - blend) * mean(segment excess Sharpe)
  + blend * pooled excess Sharpe
  - std_coef * std(segment excess Sharpe)
  - dd_coef * p75(max drawdown frac)
```

With current defaults:

```text
blend = 0.5
std_coef = 0.75
dd_coef = 2.0
benchmark = equal_weight_daily
```

Scores from the two modes are not comparable across eras (dollars vs Sharpe
units); compare ranks within a run only.

Negative robust scores are expected when the agent is behind the benchmark after
dispersion and drawdown penalties. A downward robust-score line means later
checkpoints are less attractive under the selection rule, even if training
reward or raw episode NAV remains high.
