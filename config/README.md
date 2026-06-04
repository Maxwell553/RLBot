# Configuration

`config.yaml` is the single source of truth for the **tradeable universe**, environment, reward, transaction costs, curriculum, and PPO defaults. Loaded by `rlbot.rl_config.load_config()`; each training run copies the file to `Runs/<run_id>/config.yaml`.

Override at runtime:

```bash
python scripts/train.py --config /path/to/config.yaml
```

## Where to set the number of assets

**N = number of keys** under `universe.assets` (not a separate setting). Supported range: **5–55** (`rlbot.rl_config.UNIVERSE_MIN_ASSETS` / `UNIVERSE_MAX_ASSETS`).

Training CLI: `--n-assets N` keeps the first **N** keys in YAML order and slices per-asset lists (cannot exceed keys defined in this file).

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
3. Train with a **new** `--run-id` (new `obs_dim = 9N + 28`; new VecNormalize + LSTM weights).

Full checklist: [docs/TRAINING.md](../docs/TRAINING.md).

## Other sections

See the table in [README.md](../README.md#configuration-configconfigyaml-rlbotrl_configpy).
