# Empirical Research Report: Multi-Asset Portfolio Management via Recurrent PPO

Methodology and results for walk-forward **RecurrentPPO** runs on a config-driven tradeable universe (**5–55** assets via `config/config.yaml` → `universe.assets`).

Each window trains on data through a fixed **train-end** date; a chronological **OOS holdout** never appears in training or in-training validation. Published OOS metrics use **`models/<run_id>/best/best_model.zip`** (maximum mean in-training eval ending NAV), not holdout-tuned weights.

**Operations:** [docs/TRAINING.md](docs/TRAINING.md) · **Implementation:** [README.md](README.md)

---

## Executive summary

### Design

| Item | Value |
|------|--------|
| Policy | RecurrentPPO `MlpLstmPolicy`, 2×64 LSTM, MLP [128,128] |
| Tradeable **N** | `len(universe.assets)` (default example **N = 10**) |
| Observation | `obs_dim = 9N + 28` |
| Action | `N + 1` (cash + risky), softmax + per-asset cap |
| Training | 8 envs, `n_steps` 32768, batch 16384, VecNormalize, fee/churn curriculum |
| OOS backtest | `obs_lag = 1`, full fees, no curriculum |
| Checkpoint rule | Eval-NAV-best only |

### Hyperparameter protocol

Core hyperparameters are frozen in `config/config.yaml` and snapshotted per run. Walk-forward windows differ by **calendar flags** and **`--run-id`** only, not per-window YAML sweeps, unless a new study is intentional.

---

## Completed runs (results)

Record OOS metrics after:

```bash
python scripts/backtest.py --run-id <RUN_ID> --detailed --stochastic-paths 30 --plot-tag best
```

Plots: `plots/<run_id>/backtest_best.png` · Training: `plots/<run_id>/training.png`

### Walk-forward registry

| Window | Train through | OOS holdout | `run_id` | Training | OOS backtest |
|--------|---------------|-------------|----------|----------|--------------|
| 1 | 2015-12-31 | 2016-01-01 … 2017-12-31 | `wf_window1_001` | In progress / artifacts under `models/wf_window1_001/` | Pending |
| 2 | 2017-12-31 | 2018-01-01 … 2019-12-31 | `wf_window2_001` | Manifest + partial artifacts | Pending |
| 3 | 2019-12-31 | 2020-01-01 … 2021-06-30 | — | Not started | — |
| 4 | 2020-12-31 | 2021-07-01 … 2022-12-31 | — | Not started | — |
| 5 | 2022-12-31 | 2023-01-01 … 2024-12-31 | — | Not started | — |
| 6 | 2024-12-31 | 2025-01-01 … latest | — | Not started | — |

### OOS performance (fill from backtest CLI)

| `run_id` | Agent total return | Agent Sharpe | Max DD | SPY B&H | Equal-weight | 60/40 | Risk parity |
|----------|-------------------|--------------|--------|---------|--------------|-------|-------------|
| `wf_window1_001` | — | — | — | — | — | — | — |
| `wf_window2_001` | — | — | — | — | — | — | — |

*Replace em dashes after running `scripts/backtest.py --detailed` for each completed training run.*

### Manifest snapshots (completed / in-flight)

**`wf_window1_001`** — 65M timesteps, seed 0, N=10, obs_dim=118, 521 OOS bars (2016-01-01 … 2017-12-29).

**`wf_window2_001`** — same hyperparameters, trainable through 2017-12-31, OOS 2018-01-01 … 2019-12-31, 521 OOS bars.

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

Ticker order: `runs/<run_id>/manifest.json` → `universe.tickers` and `.cache/data_cache.npz`.

---

## Data engineering & anti-leakage

1. **Fractional differentiation** (default d = 0.4) on log prices.
2. **Per-block feature isolation:** `train_test_split_alternating`, 126-bar blocks, every 4th block eval; features computed per segment; 25-bar join purge.
3. **Causal execution:** features at `t` use `close[t−obs_lag]`; fill `open[t+1]`; MTM `close[t+1]`.
4. **Chronological holdout:** removed before train/eval; only `scripts/backtest.py` uses OOS bars.

---

## Environment & reward

**Action:** ℝ^(N+1) → softmax → cap + redistribute.

**Reward:** scaled log return + Sortino vs `benchmark_cap_weights` + participation − inactivity − VIX-scaled normalized churn − quadratic drawdown.

**Costs:** per-asset slippage, fees, holding cost (length N); `fee_scale` curriculum in training; full costs in OOS backtest.

---

## Training loop

RecurrentPPO + VecNormalize + `TradingCurriculumCallback` + `EvalNavBestModelCallback` + `AdaptiveEntropyCallback`. See [README.md](README.md) for callback milestones and artifact paths.

```bash
python scripts/train.py --refresh-data --timesteps 1000 --run-id _data_refresh --no-viz  # after universe edits
RUN_ID=wf_window1_001 ./scripts/walkforward/window1_train.sh
python scripts/backtest.py --run-id wf_window1_001 --detailed --stochastic-paths 30 --plot-tag best
```

**Do not load** checkpoints when `manifest.universe.obs_dim` or `universe.tickers` ≠ current config/cache.

---

## Seed robustness

```bash
./scripts/run_seed_ensemble.sh --window 3 --cohort my_cohort
python scripts/backtest.py --ensemble-prefix my_cohort --ensemble-checkpoint best --detailed
```

---

## Assumptions & limitations

- **yfinance** (+ FRED HY OAS / HYG–IEF proxy); not institutional point-in-time data.
- **Universe** fixed per run; no dynamic listing/delisting.
- **VecNormalize** must pair with the same `run_id` checkpoint.

---

## What this report does not claim

No guarantee of live performance or absence of multiple-testing bias across windows. Results are evidence **within the documented pipeline** only.
