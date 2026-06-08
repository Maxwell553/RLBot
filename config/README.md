# Configuration

`config.yaml` is the single source of truth for the **tradeable universe**, environment, reward, transaction costs, curriculum, and PPO defaults. Loaded by `rlbot.rl_config.load_config()`; each training run copies the file to `Runs/<run_id>/config.yaml`.

Override at runtime:

```bash
python scripts/train.py --config /path/to/config.yaml
```

## Where to set the number of assets

**N = number of keys** under `universe.assets` (not a separate setting). Supported range: **5–55** (`rlbot.rl_config.UNIVERSE_MIN_ASSETS` / `UNIVERSE_MAX_ASSETS`).

Training CLI: `--n-assets N` keeps the first **N** keys in YAML order and slices per-asset lists (cannot exceed keys defined in this file). After changing **N**, run `--refresh-data` so the global cache matches; each training run also writes an **N**-wide `Runs/<run_id>/data_cache.npz` snapshot for backtest reproducibility.

Example — default **N = 10**:

```yaml
universe:
  benchmark: SP500
  assets:
    SP500: SPY
    GOLD: GLD
    # ... eight more keys → N = 10
```

To use **N = 20**, add ten more `LABEL: yfinance_symbol` pairs and extend every per-asset list below to **20** floats in the **same key order**.

## `universe` (required)

| Key | Purpose |
|-----|---------|
| `benchmark` | Label of the Sortino / SPY buy-and-hold sleeve (must be a key in `assets`) |
| `assets` | Ordered map `LABEL → yfinance_symbol` (YAML key order = OHLCV axis order) |

## Per-asset lists (must match N)

| Section | Keys |
|---------|------|
| `reward` | `benchmark_cap_weights` |
| `transaction_costs` | `slippage`, `tx_fee`, `annual_holding_cost` |

## Changing the universe

1. Edit `universe.assets` and all per-asset lists (length **N**, same order).
2. Run `python scripts/train.py --refresh-data --timesteps 1000 --run-id _data_refresh --no-viz`.
3. Train with a **new** `--run-id` or `--window` (new `obs_dim = 10N + 28`; new VecNormalize + LSTM weights).

Full checklist: [docs/TRAINING.md](../docs/TRAINING.md).

## `data` (feature pipeline)

| Key | Purpose |
|-----|---------|
| `feature_split_mode` | `continuous` (default) or `independent` — how walk-forward blocks get RSI/MACD/fracdiff/trend/vol features |
| `feature_purge_warmup` | Bars neutralized at segment starts in `independent` mode (default 25) |

## `reward` (key knobs)

| Key | Purpose |
|-----|---------|
| `inactivity_penalty_over_50` / `over_90` | Linear cash penalty (default 1.5 + 1.0 tail above 90% cash) |
| `drawdown_downside_gamma` | Amplifies negative step returns when already in drawdown (default 5.0) |
| `churn_penalty` | Multiplier on `tx_cost_frac × reward_scale` (default 1.0) |
| `eval_inactivity_penalty_scale` | Eval env inactivity scale (default 1.0) |

## `curriculum`

| Key | Purpose |
|-----|---------|
| `churn_ramp_floor` | Churn scale at fee-ramp start; ramps to 1.0 by `fee_ramp_fraction` (default 0.1) |

## Other sections

See the reward/cost tables in [README.md](../README.md) and [docs/RESEARCH.md](../docs/RESEARCH.md).
