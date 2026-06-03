# MarketTrainer (RLBot)

Production research stack for training **RecurrentPPO** (LSTM) agents on a multi-asset daily portfolio environment, with strict chronological out-of-sample (OOS) holdouts and walk-forward in-training evaluation.

| Topic | Location |
|-------|----------|
| Hyperparameters, rewards, costs, curriculum | `config/config.yaml` → `rlbot/rl_config.py` |
| Universe size (5–55 assets), restart checklist | [docs/TRAINING.md](docs/TRAINING.md) |
| Config field reference | [config/README.md](config/README.md) |
| Methodology & run results | [RESEARCH.md](RESEARCH.md) |

Each training run snapshots effective config to `runs/<run_id>/config.yaml` and metadata to `runs/<run_id>/manifest.json`.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# After universe or fetch changes (once)
python scripts/train.py --refresh-data --timesteps 1000 --run-id _data_refresh --no-viz

# Train (new --run-id per experiment)
python scripts/train.py --timesteps 65000000 --run-id my_run_001

# OOS backtest on chronological holdout (never seen in training)
python scripts/backtest.py --run-id my_run_001 --detailed --stochastic-paths 30 --plot-tag best
```

**CLI entry points** (after `pip install -e .`): `market-trainer-train`, `market-trainer-backtest`.

**Walk-forward example:**

```bash
RUN_ID=wf_window1_001 ./scripts/walkforward/window1_train.sh
./scripts/walkforward/window1_backtest.sh wf_window1_001
```

**Universe:** `N = len(universe.assets)` in config, or `python scripts/train.py --n-assets N` (first N YAML keys). Override dates and `--run-id` per window; do not change core hyperparameters across walk-forward windows unless starting a new study.

Artifacts (gitignored): `models/`, `runs/`, `logs/`, `plots/`, `tb_logs/`, `.cache/data_cache.npz`.

First launch in a new terminal can take several minutes before PPO progress appears; see [docs/TRAINING.md#startup-time-first-run-in-a-session](docs/TRAINING.md#startup-time-first-run-in-a-session).

---

## Architecture

```mermaid
flowchart TB
  subgraph cfg [config.yaml + rl_config]
    YAML[RLConfig]
  end
  subgraph data [data_utils]
    FETCH[fetch_aligned_daily]
    HOLD[reserve_chronological_holdout]
    SPLIT[train_test_split_alternating]
    FETCH --> HOLD --> SPLIT
  end
  subgraph train [scripts/train.py]
    PPO[RecurrentPPO]
    VN[VecNormalize]
    CB[Curriculum + EvalNavBest + Entropy]
    PPO --> VN --> CB
  end
  subgraph env [trading_env]
    OBS[9N+28 obs]
    ACT[softmax N+1]
    EXEC[open t+1]
  end
  cfg --> train
  cfg --> env
  data --> train
  train --> env
  env --> BT[scripts/backtest.py]
```

---

## Core design

### Data (`rlbot/data_utils.py`)

1. Fetch aligned daily OHLCV for `universe.assets` (yfinance).
2. Cache panel + `tickers` in `.cache/data_cache.npz`.
3. Reserve chronological OOS holdout before any in-training split.
4. Walk-forward alternating split (126-bar blocks; every 4th block eval).
5. Per-segment features (RSI, MACD, fracdiff, trend) with join purge — no cross-block leakage.

### Environment (`rlbot/trading_env.py`)

- **Action:** `N+1` logits → softmax → long-only risky weights, per-asset cap (`max_single_asset_weight`).
- **Observation:** `obs_dim = 9×N + 28` (four fixed macro series).
- **Execution:** decide after `close[t−obs_lag]`, fill `open[t+1]`, MTM `close[t+1]`.
- **Reward:** return + Sortino vs cap-weighted benchmark − inactivity − VIX-scaled churn − drawdown² (training curricula on fees/churn/DR).

### Training (`scripts/train.py`)

- **RecurrentPPO** `MlpLstmPolicy` (2×64 LSTM, MLP [128,128]), 8 parallel envs, 65M default timesteps.
- **EvalNavBestModelCallback** → `models/<run_id>/best/best_model.zip` (max mean in-training eval NAV).
- **TradingCurriculumCallback** — frictionless phase, fee ramp, progressive domain randomization.
- Checkpoint selection for published OOS: **eval-NAV-best only** (holdout never used to pick weights).

### Evaluation

| Script | Purpose |
|--------|---------|
| `scripts/backtest.py` | OOS rollout, passive benchmarks, bootstrap Sharpe CI, plots |
| `scripts/backtest_sweep.py` | Calendar/regime slices with leakage labels |
| `scripts/run_seed_ensemble.sh` | Multi-seed training + ensemble backtest |
| `rlbot/baselines.py` | SPY B&H, equal-weight, 60/40, naive risk parity |

Passive benchmarks use **simple-return** cross-sectional aggregation, then compound (see `rlbot/baselines.py`).

---

## Walk-forward windows

| Window | Train through | OOS holdout | Train script |
|--------|---------------|-------------|--------------|
| 1 | 2015-12-31 | 2016–2017 | `scripts/walkforward/window1_train.sh` |
| 2 | 2017-12-31 | 2018–2019 | `window2_train.sh` |
| 3 | 2019-12-31 | 2020–H1 2021 | `window3_train.sh` |
| 4 | 2020-12-31 | 2021 H2–2022 | `window4_train.sh` |
| 5 | 2022-12-31 | 2023–2024 | `window5_train.sh` |
| 6 | 2024-12-31 | 2025–latest | `window6_train.sh` |

Validate bar counts: `python scripts/walkforward/validate_split.py --window 1`

---

## Project layout

| Path | Role |
|------|------|
| `config/config.yaml` | Universe, PPO, reward, costs |
| `rlbot/` | Library: data, env, config, artifacts, visualize, baselines |
| `scripts/` | `train.py`, `backtest.py`, `backtest_sweep.py`, walk-forward shells |
| `docs/TRAINING.md` | Operations guide |
| `RESEARCH.md` | Methodology + completed-run results |
| `tests/` | `pytest` |
| `execution/` | Reserved for future broker integration |

---

## Dependencies

`gymnasium`, `stable-baselines3`, `sb3-contrib`, `torch`, `pandas`, `numpy`, `yfinance`, `matplotlib`, `tensorboard`, `PyYAML` — see `requirements.txt` / `pyproject.toml`.
