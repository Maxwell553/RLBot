# Training guide (multi-asset universe)

## Where the number of assets is set

**Default:** tradeable count **N** is the number of entries under `universe.assets` in `config/config.yaml`.

**CLI shortcut:** `scripts/train.py --n-assets N` uses the **first N keys** from that block (YAML order), slices `benchmark_cap_weights` and `transaction_costs.*`, and renormalizes cap weights. You cannot set **N** larger than the number of keys in the file — add symbols to the YAML first. The benchmark label (`universe.benchmark`, usually `SP500`) must stay among those first N keys (it is first in the default config).

To change **which** assets are in the panel (not just how many), edit the YAML and run `--refresh-data`.

```yaml
# config/config.yaml — full list; --n-assets 7 keeps SP500 … BOND10Y
universe:
  benchmark: SP500   # must be one of the keys in assets
  assets:
    SP500: SPY
    GOLD: GLD
    # ... add or remove labels (yfinance symbols) here
```

Supported range: **5 ≤ N ≤ 55** (enforced at config load and again when training validates the data panel).

For each asset key you must provide **one float** in each of (length **N**, same key order):

- `reward.benchmark_cap_weights`
- `transaction_costs.slippage`
- `transaction_costs.tx_fee`
- `transaction_costs.annual_holding_cost`

Observation size: **`obs_dim = 10 × N + 28`** (includes per-asset live mask in the vector). Action size: **`N + 1`**.

See also [config/README.md](../config/README.md).

---

## Startup time (first run in a session)

`scripts/train.py` prints `[train] …` status lines with `flush=True` so you see progress before PPO starts. A **new shell** can sit quietly for several minutes while:

1. **PyTorch / Stable-Baselines3 import** — often 1–5 minutes the first time (SymPy and related deps).
2. **Walk-forward feature panels** — per-block RSI/MACD/fracdiff on the trainable window (~1–3 minutes).
3. **`SubprocVecEnv` spawn** — each of the `n_envs` workers reloads the stack on macOS (~1–3 minutes).

Later runs in the **same** terminal are usually faster (imports already cached). Use `PYTHONUNBUFFERED=1` if your terminal still buffers stdout.

If you pass `--refresh-data`, add extra time for yfinance downloads before step 2.

---

## Starting training again (clean run)

Use this checklist after changing the universe, reward/cost vectors, or pulling the latest code (dynamic universe refactor).

### 1. Stop any running training

Kill active `scripts/train.py` processes so checkpoints are not written mid-step.

### 2. Edit `config/config.yaml`

- Set `universe.assets` to your target **N** labels (5–55).
- Align `benchmark_cap_weights` and all `transaction_costs.*` lists to **N** values in the **same key order**.
- Keep `universe.benchmark` as one of the asset keys (typically `SP500`).

### 3. Refresh the data cache

```bash
source .venv/bin/activate
python scripts/train.py --refresh-data --timesteps 1000 --run-id _data_refresh --no-viz
```

This rebuilds `.cache/data_cache.npz` with a `tickers` array matching your config.

### 4. Launch training with a **new** run id

Do not reuse run ids from checkpoints trained with a different **N** or `obs_dim`.

```bash
# Walk-forward sample 1 (dates stored in manifest for backtest)
python scripts/train.py \
  --since 2006-01-01 \
  --until 2017-12-31 \
  --train-end 2015-12-31 \
  --holdout-start 2016-01-01 \
  --holdout-end 2017-12-31 \
  --timesteps 65000000 \
  --window 1
```

Omit `--run-id` and pass `--window N` to auto-name the run `W{N}_<month><day>` (e.g. `W1_604` on June 4). If that folder already exists, the next id is `W1_604_a`, then `W1_604_b`, etc. You can still set `--run-id` explicitly.

Training will:

1. Load config → **N** from `universe.assets`
2. Fetch or load cache → validate `validate_config_for_universe(cfg, ohlcv.shape[1])`
3. Write `Runs/<run-id>/manifest.json` with `universe.tickers`, `n_assets`, `obs_dim`
4. Build envs with dynamic observation/action spaces

### 5. OOS backtest (after training)

```bash
python scripts/backtest.py --run-id <RUN_ID> --detailed --stochastic-paths 30 --plot-tag best
```

Backtest reads `manifest.universe.tickers` and checks observation dimension against the saved model / VecNormalize stats.

---

## Walk-forward windows

Calendar presets are documented in [RESEARCH.md](RESEARCH.md). Pass `--train-end`, `--holdout-start`, `--holdout-end`, and `--until` on `train.py`; backtest reads them from `Runs/<run-id>/manifest.json`.

```bash
python scripts/backtest.py --run-ids W1,W2,W3,W4,W5,W6 --checkpoint both
python scripts/backtest.py --run-id W1 --checkpoint both --detailed --stochastic-paths 30 --plot-tag best
```

---

## What not to reuse

- **Checkpoints** (`Runs/<run_id>/models/`) trained with a different **N**, `obs_dim`, or pre-`asset_live` cache
- **VecNormalize** pickles from another universe size or observation layout
- **Old caches** without `tickers` / `asset_live` in the npz — always `--refresh-data` after universe or data-pipeline edits

## Run artifact layout

Each run lives under `Runs/<run_id>/` (see `rlbot/run_artifacts.py`):

| Path | Contents |
|------|----------|
| `manifest.json` | Dates, tickers, `n_assets`, `obs_dim`, universe metadata |
| `config.yaml` | Snapshot of training config |
| `models/` | `ppo_portfolio_final.zip`, `vec_normalize.pkl`, `best/best_model.zip` |
| `plots/`, `logs/`, `tb_logs/`, `eval_logs/` | Training visuals, text logs, TensorBoard, eval NAV history |

Migrate legacy scattered dirs once: `python scripts/migrate_runs_layout.py`.

---

## Training on Modal (optional)

For long runs on a cloud GPU with the same `Runs/<run_id>/` layout, see [MODAL.md](MODAL.md).

Quick flow:

1. `pip install -e ".[modal]"` and `modal setup`
2. `modal run scripts/modal_app.py -- --window 2 --timesteps 65000000 ...` (same date/universe flags as local)
3. In another terminal: `python scripts/modal_app.py sync --run-id <RUN_ID> --watch` (open `Runs/<RUN_ID>/plots/training.png` in the IDE)
4. After the job: `python scripts/modal_app.py sync --run-id <RUN_ID> --pull-all` then backtest locally

Use `--run-id` explicitly if you want a fixed id for sync before the job prints logs.
