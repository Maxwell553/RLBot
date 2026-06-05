# Empirical Research Report: Multi-Asset Portfolio Management via Recurrent PPO

Methodology and results for walk-forward **RecurrentPPO** runs on a config-driven tradeable universe (**5–55** assets via `config/config.yaml` → `universe.assets`).

Each window trains on data through a fixed **train-end** date; a chronological **OOS holdout** never appears in training or in-training validation. Published OOS metrics use **`Runs/<run_id>/models/best/best_model.zip`** (maximum mean in-training eval NAV), not holdout-tuned weights.

**Operations:** [TRAINING.md](TRAINING.md) · **Modal GPU:** [MODAL.md](MODAL.md) · **Implementation:** [README.md](../README.md)

---

## Executive summary

### Design

| Item | Value |
|------|--------|
| Policy | RecurrentPPO `MlpLstmPolicy`, 2×64 LSTM, MLP [128,128] |
| Tradeable **N** | `len(universe.assets)` (default example **N = 10**) |
| Observation | `obs_dim = 10N + 28` (includes per-asset **live mask**) |
| Action | `N + 1` (cash + risky), mean-centered softmax + per-asset cap; pre-IPO weights zeroed |
| Training episodes | `max_episode_steps = 252` (train); eval = full walk-forward segment |
| Training | 16 envs (local), `n_steps` 4096, `batch_size` 16384, `n_epochs` 3 → **12** backprop loops/pause (4 mini-batches × 3 epochs on 65,536-step rollout); VecNormalize, fee/churn/DR curriculum |
| Modal training | Optional H100/A100 via `scripts/modal_app.py`; broker overrides `n_envs` + `batch_size` at launch ([MODAL.md](MODAL.md)) |
| Eval / viz cadence | Every **500k** global steps (not tied to `n_steps`); plot refresh `viz_freq: 500_000` |
| Checkpoints | Every **1M** global steps under `models/checkpoints/` |
| In-training eval | One deterministic rollout **per eval segment** (~# of 126-bar eval blocks) |
| OOS backtest | `obs_lag = 1`, full fees, `churn_scale = 1`, action smoothing 0.15, full holdout length |
| Checkpoint rule | Eval-NAV-best only (`models/best/best_model.zip`) |

### Hyperparameter protocol

Core hyperparameters live in `config/config.yaml` and are **copied to `Runs/<run_id>/config.yaml`** at train start. Walk-forward windows differ by **calendar flags** and **run id** only, not per-window YAML sweeps, unless a new study is intentional.

After any change to universe, `asset_live` panel, `obs_dim`, or reward coefficients, run `--refresh-data` (if data/universe changed) and train with a **new** run id — old checkpoints and VecNormalize stats are incompatible. OOS backtest execution uses the **current** global config for env mechanics (fees, smoothing); only the policy weights come from the run artifact.

---

## Completed runs (results)

Record OOS metrics after:

```bash
python scripts/backtest.py --run-id <RUN_ID> --detailed --stochastic-paths 30 --plot-tag best
```

Plots: `Runs/<run_id>/plots/backtest_best.png` · Training: `Runs/<run_id>/plots/training.png`

### Walk-forward registry

Use `--window N` on `train.py` (or `modal run scripts/modal_app.py -- …`) for ids like `W{N}_<month><day>` (e.g. `W1_605` on June 5); duplicate folders get `_a`, `_b`, …

**Cohorts:** `W*_604` = earlier local runs (some under pre-refactor reward/cost settings). **`W*_605`** = current `config/config.yaml` cohort (churn 8.5, batch 16384, `n_epochs` 3, linear inactivity penalty) — primary walk-forward batch on Modal.

| Window | `--train-end` | OOS (`--holdout-start` … `--holdout-end`) | `--until` | Example `run_id` | Training | OOS backtest |
|--------|---------------|-------------------------------------------|-----------|------------------|----------|--------------|
| 1 | 2015-12-31 | 2016-01-01 … 2017-12-31 | 2017-12-31 | `W1_604` / `W1_605` | W1_604 complete (local, legacy config); W1_605 Modal | W1_604: **+8.4%** / Sh 0.50 / DD −8.3% |
| 2 | 2017-12-31 | 2018-01-01 … 2019-12-31 | 2019-12-31 | `W2_605` | Modal (current config) | — |
| 3 | 2019-12-31 | 2020-01-01 … 2021-06-30 | 2021-06-30 | `W3_605` | Pending | — |
| 4 | 2021-06-30 | 2021-07-01 … 2022-12-31 | 2022-12-31 | `W4_605` | Pending | — |
| 5 | 2022-12-31 | 2023-01-01 … 2024-12-31 | 2024-12-31 | `W5_605` | Pending | — |
| 6 | 2024-12-31 | 2025-01-01 … latest | (omit / latest bar) | `W6_605` | Pending | — |

**Local train example (window 1):**

```bash
python scripts/train.py --window 1 --timesteps 65000000 \
  --since 2006-01-01 --train-end 2015-12-31 \
  --holdout-start 2016-01-01 --holdout-end 2017-12-31 --until 2017-12-31
```

**Modal train example (window 2, H100):**

```bash
modal run scripts/modal_app.py -- \
  --modal-gpu H100 --window 2 --run-id W2_605 --timesteps 65000000 \
  --refresh-data --since 2006-01-01 --train-end 2017-12-31 \
  --holdout-start 2018-01-01 --holdout-end 2019-12-31 --until 2019-12-31
python scripts/modal_app.py sync --run-id W2_605 --watch
```

When advancing to a later window on Modal, pass `--refresh-data` with that window’s `--until` (or upload a full local cache once) so the shared `rlbot-cache` volume covers the new holdout dates.

### OOS performance (fill from backtest CLI)

| `run_id` | Agent total return | Agent Sharpe | Max DD | SPY B&H | Equal-weight | 60/40 | Risk parity |
|----------|-------------------|--------------|--------|---------|--------------|-------|-------------|
| `W1_604` | **+8.4%** | **0.50** | **−8.3%** | +40.6% | +27.0% | +21.1% | +15.6% |
| `W1_605` | — | — | — | — | — | — | — |
| `W2_605` | — | — | — | — | — | — | — |
| `W3_605` | — | — | — | — | — | — | — |
| `W4_605` | — | — | — | — | — | — | — |
| `W5_605` | — | — | — | — | — | — | — |
| `W6_605` | — | — | — | — | — | — | — |

*Agent columns use **`best_model.zip`** (eval-NAV-best). Benchmarks from `scripts/backtest.py --detailed` on the same OOS window. **`W1_604`** was trained under `Runs/W1_604/config.yaml` (`churn_penalty: 37.5`, return clip +0.03/−0.15, step inactivity above 50% cash). **`W*_605`** runs snapshot the **current** [reward & cost](#environment--reward) settings at train start.*

### Informal cross-window check (optional)

Loading window *N* weights on window *N+1* holdout tests regime shift without retraining. Override holdout dates on `backtest.py` **and** pass `--until` through the new holdout end (manifest `until` clips the cache otherwise).

**W1_604 → W2 holdout (2018–2019), 65M latest checkpoint** (not a registry entry for `W2_604`):

| Metric | Agent | SPY B&H |
|--------|-------|---------|
| Total return | +2.0% | +13.7% |
| Ann. Sharpe | 0.14 | 0.43 |
| Max DD | −12.6% | — |

Stochastic ensemble (30 paths): median return −1.4%, Sharpe mean −0.09.

---

## Passive benchmark methodology

Implemented in `rlbot/baselines.py`; plotted by `scripts/backtest.py`. Multi-asset books aggregate **simple returns** cross-sectionally each day, then compound.

| Benchmark | Allocation | Rebalance |
|-----------|------------|-----------|
| Benchmark B&H | 100% `universe.benchmark` sleeve (default SP500/SPY) | — |
| Equal-weight | 1/N per asset | Daily |
| 60/40 | 60% benchmark / 40% BOND10Y (IEF) | Calendar month-start |
| Naive risk parity | ∝ 1/σ (20d vol) | Daily |

---

## Asset universe

**Tradeable:** `config/config.yaml` → `universe.assets` (default ten global proxies: SPY, GLD, USO, FX, indices, IEF, copper, EEM).

**Macro only (4 series):** DXY, TNX, VIX, HY OAS — observation features, not in the action space.

**IPO / listing:** Panel rows are **not** dropped for missing early history. Missing OHLCV is filled to a tiny positive placeholder; `asset_live` marks real prints. The policy cannot allocate to names with `asset_live = 0` at the decision bar.

Ticker order: `Runs/<run_id>/manifest.json` → `universe.tickers` and `.cache/data_cache.npz`.

---

## Data engineering & anti-leakage

1. **Fractional differentiation** (default d = 0.4) on log prices.
2. **Walk-forward blocks:** `train_test_split_alternating`, 126-bar blocks, every 4th block eval; features precomputed on the full trainable timeline, sliced per segment (`WalkforwardEnvPack` in `data_utils.py`) — no cross-block leakage.
3. **Causal execution:** features at `t` use `close[t−obs_lag]`; fill `open[t+1]`; MTM `close[t+1]`; holding costs on pre-rebalance units at `close[t]`.
4. **Chronological holdout:** removed before train/eval; only `scripts/backtest.py` uses OOS bars.
5. **HY OAS macro:** causal expanding OLS calibration (no per-bar `polyfit` look-ahead).
6. **Risk-parity baseline:** inverse-vol weights use only past returns; IPO names borrow mean peer vol during warmup.

---

## Environment & reward

**Action:** ℝ^(N+1) → EMA on logits (`action_smoothing_alpha: 0.15`, train + backtest) → mean-centered softmax → cap + redistribute; `asset_live` zeroes pre-IPO risky weights.

### Reward decomposition

All terms are in **reward units** (VecNormalize scales the sum during training). Per-step `info` exposes `rew_decomp/*`.

| Component | Implementation | Default coefficients |
|-----------|----------------|-------------------|
| **Return** | `clip(log_ret, max_step_log_return_downside, max_step_log_return) × reward_scale` | clip **−0.12 / +0.06**; scale **2000** |
| **Sortino differential** | Agent vs cap-weighted benchmark Sortino over `risk_window` (min `sortino_min_steps` warmup), clipped ±3 | `risk_bonus_scale: 25` |
| **Participation** | `gross_exposure × participation_bonus × participation_reward_scale` | `0.05 × 20` |
| **Inactivity** | Linear in `cash_frac`: `cash_frac × inactivity_penalty_over_50`; extra ramp from 90%→100% cash | **10.0** base + **15.0** tail; **no 50% step cliff** |
| **Churn** | `turnover_frac × churn_penalty × VIX_mult × curriculum_churn_scale` | `churn_penalty: 8.5` |
| **Drawdown** | `(dd_frac)² × drawdown_penalty_scale × drawdown_quadratic_multiplier` | `25 × 12` |

**Churn detail:** `turnover_frac` = dollar turnover ÷ NAV (10% rebalance → `0.10`). `VIX_mult = clip(VIX/18, 0.75, 1.5)`. Training `curriculum_churn_scale` is **0** until ~20% of the run, then linear **0→1** over 10M steps. At full scale, effective coefficient is **6.4–12.8** (75–150% of 8.5). Eval/backtest keep `churn_scale = 1`.

**Inactivity detail:** Training envs use scale **1.0**. In-training eval envs use `eval_inactivity_penalty_scale: 0.05` so defensive cash is not over-penalized during segment rollouts.

**Costs:** per-asset **slippage**, **tx_fee**, and **annual_holding_cost** (length-N lists in `transaction_costs`, keyed like `universe.assets`). Costs multiply by `fee_scale` each step; training curriculum runs `fee_scale` from **0** (frictionless) through a linear ramp to **1.0**, then domain-randomizes fee/lag bounds. OOS backtest always uses full configured costs (`fee_scale = 1`).

| Asset (example) | Slippage | Tx fee | Annual holding |
|-----------------|----------|--------|----------------|
| SP500 (SPY) | 1 bp | 1 bp | 9 bp |
| GOLD (GLD) | 2 bp | 2 bp | 40 bp |
| OIL (USO) | 3 bp | 2 bp | 83 bp |
| BOND10Y (IEF) | 1 bp | 1 bp | 15 bp |
| EM (EEM) | 2 bp | 2 bp | 67 bp |

(FX sleeves: zero holding cost in config.)

---

## Cloud training (Modal)

Optional GPU path; artifacts use the same `Runs/<run_id>/` layout as local training.

| Step | Command |
|------|---------|
| Setup | `pip install -e ".[modal]"` · `modal setup` |
| Train | `modal run scripts/modal_app.py -- --modal-gpu H100 --window N --run-id WN_605 …` |
| Watch plot | `python scripts/modal_app.py sync --run-id WN_605 --watch` → `Runs/<id>/plots/training.png` |
| Pull all artifacts | `python scripts/modal_app.py sync --run-id WN_605 --pull-all` |
| Backtest locally | `python scripts/backtest.py --run-id WN_605 --checkpoint best --plot-tag best` |

**Volumes:** `rlbot-runs` (per-run tree: models, logs, `config.yaml`, `data_cache.npz`, plots) · `rlbot-cache` (shared OHLCV panel). `--watch` syncs plots/eval only; models and logs require `--pull-all`.

**GPU broker:** `--modal-gpu` selects vCPUs, `n_envs`, and `batch_size` at launch (e.g. H100 → 64 envs, batch 65536). Cannot change `n_envs` mid-run.

**Data:** Each window’s `--until` must be covered by the cache on `rlbot-cache`. Use `--refresh-data` per window or `modal run scripts/modal_app.py::upload_cache` once from a full local `.cache/data_cache.npz`.

---

## Training loop & curriculum (65M timesteps)

RecurrentPPO + VecNormalize + `TradingCurriculumCallback` + `EvalNavBestModelCallback` + `AdaptiveEntropyCallback` + `TrainingVizCallback`. Milestones use `curriculum.budget_short` fractions in `config/config.yaml`:

| Phase | Approx. step (65M run) | Effect |
|-------|------------------------|--------|
| Fee-free | 0 – 6.5M (`fee_free_fraction` 0.10) | `fee_scale = 0` |
| Fee ramp | 6.5M – 29.25M (`fee_ramp_fraction` 0.45) | fees → full |
| Churn off → on | churn starts ~13M (`churn_start_fraction` 0.20), full by ~23M | `curriculum_churn_scale` 0 → 1 over 10M steps |
| DR widen | through ~42.25M (`dr_widen_span_fraction` 0.65) | fee/lag domain-randomization bounds widen |
| Entropy | cosine decay from `decay_start_fraction` 0.45 (~29.25M) | exploration → `final_ent` |
| Eval / plot | every **500k** global steps | ~130 eval rollouts per 65M run (decoupled from `n_steps`) |
| Checkpoints | every **1M** global steps | `models/checkpoints/ppo_*_steps.zip` |

```bash
python scripts/train.py --refresh-data --timesteps 1000 --run-id _data_refresh --no-viz
python scripts/train.py --window 1 --timesteps 65000000 \
  --train-end 2015-12-31 --holdout-start 2016-01-01 --holdout-end 2017-12-31 --until 2017-12-31
python scripts/backtest.py --run-id W1_604 --checkpoint best --detailed --stochastic-paths 30 --plot-tag best

# Modal equivalent (pull artifacts before backtest)
modal run scripts/modal_app.py -- --modal-gpu H100 --window 1 --run-id W1_605 --timesteps 65000000 \
  --since 2006-01-01 --train-end 2015-12-31 \
  --holdout-start 2016-01-01 --holdout-end 2017-12-31 --until 2017-12-31
python scripts/modal_app.py sync --run-id W1_605 --pull-all
python scripts/backtest.py --run-id W1_605 --checkpoint best --detailed --stochastic-paths 30 --plot-tag best
```

**Do not load** checkpoints when `manifest.universe.obs_dim` or `universe.tickers` ≠ current config/cache. Reward/cost/config edits require a **new** `--run-id`; weights were optimized under the snapshotted `Runs/<id>/config.yaml`. Compare cohorts (`W1_604` vs `W1_605`) only with this config drift in mind.

---

## Seed robustness

```bash
./scripts/run_seed_ensemble.sh --cohort my_cohort -- --train-end 2019-12-31 \
  --holdout-start 2020-01-01 --holdout-end 2021-06-30 --until 2021-06-30 --timesteps 65000000
python scripts/backtest.py --ensemble-prefix my_cohort --ensemble-checkpoint best --detailed
```

---

## Assumptions & limitations

- **yfinance** (+ FRED HY OAS / HYG–IEF proxy); not institutional point-in-time data.
- **Universe** fixed per run; listing dates approximated via first valid print mask, not corporate actions database.
- **VecNormalize** must pair with the same `run_id` checkpoint.
- **In-training eval** is segment rollouts, not i.i.d. episode sampling; mean eval NAV is a monitoring signal, not an unbiased estimator of OOS Sharpe.

---

## What this report does not claim

No guarantee of live performance or absence of multiple-testing bias across windows. Results are evidence **within the documented pipeline** only.
