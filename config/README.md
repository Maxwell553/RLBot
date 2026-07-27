# Configuration Reference

`config.yaml` is the single source of truth for the **tradeable universe**, environment, reward, transaction costs, curriculum, and PPO defaults. Loaded by `rlbot.rl_config.load_config()`; each training run copies the file to `Runs/<run_id>/config.yaml`.

This file is intentionally a compact field reference. Use [docs/TRAINING.md](../docs/TRAINING.md) for operations/checklists and [../README.md](../README.md) for the full environment and reward explanation.

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
| `benchmark` | Reporting sleeve for benchmark-only buy-and-hold and 60/40 diagnostics (default `SP500`; normally a key in `assets`). It does **not** define reward benchmark excess or robust eval selection. |
| `assets` | Ordered map `LABEL → yfinance_symbol` (YAML key order = OHLCV axis order) |

## Per-asset lists (must match N)

| Section | Keys |
|---------|------|
| `reward` | `benchmark_cap_weights` (reward/Sortino passive book; sliced and renormalized by `--n-assets`) |
| `transaction_costs` | `slippage`, `tx_fee`, `annual_holding_cost` |

## Changing the universe

Edit `universe.assets` and all per-asset lists (length **N**, same order), refresh data, and train under a new run id. Full checklist: [docs/TRAINING.md](../docs/TRAINING.md).

## `data` (feature pipeline)

| Key | Purpose |
|-----|---------|
| `feature_split_mode` | `independent` (default) or `continuous` — how walk-forward blocks get RSI/MACD/fracdiff/trend/vol features |
| `feature_purge_warmup` | Bars neutralized at segment starts in `independent` mode (default 25) |
| `feature_preroll_bars` | Causal warmup bars for independent segment features (default 252) |

## `reward` (key knobs)

| Key | Purpose |
|-----|---------|
| `risk_bonus_scale` | Sortino differential multiplier (default **2.5**; the 721/722 cohorts tested 4.0 and the eval excess-Sharpe trajectory degraded — the ±3 clip makes extra weight a constant risk-on bonus, not gradient) |
| `benchmark_cap_weights` | Passive book used by reward benchmark excess and Sortino diff (default equal 1/N; feasible under the 20% asset cap) |
| `benchmark_excess_scale` / `benchmark_excess_clip` | Per-step excess return vs the friction-aware passive book above |
| `benchmark_combined_abs_cap` | Constant cap on combined \|sortino+benchmark\| per step in reward units (default **24.0**; `0` disables both; never relative to the other terms) |
| `turnover_penalty` | Direct `turnover_frac × turnover_penalty × reward_scale × VIX_mult × curriculum_churn_scale` (default **0.007**; ramps with churn, off during fee-free) |
| `exposure_risk_mode` / `exposure_risk_penalty_scale` | Cut gross exposure in high-vol regimes (`realized_vol` or `vix_positive`; default scale **5.0** for `vix_positive`) |
| `vol_penalty_mode` / `vol_penalty_scale` | `absolute` taxes agent downside vol; `excess` taxes only vs the passive book (default mode **absolute**, scale **0.60**; `0` scale disables) |
| `drawdown_downside_gamma` | Amplifies negative step returns when already in drawdown (default 12.0) |
| `drawdown_amp_max` | Cap on the `(1 + gamma × dd)` amplification factor (default **4.0**, saturates at dd = 25%; `0` = uncapped legacy behavior). Bounds worst-day reward outliers so VecNormalize clipping keeps gradient on bad days |
| `drawdown_increase_penalty` / `drawdown_level_penalty` / `drawdown_level_floor` | Direct drawdown penalty on expansion + while sitting above floor |
| `drawdown_level_times_reward_scale` | When **true** (731+), level multiplies `reward_scale` like the increase term (default **true** with penalty **0.10**; parser default false preserves old raw-unit snapshots where level was ~0.4% abs_share) |
| `drawdown_level_exposure_coupling` | Scale level term by gross exposure: `level × ((1−c) + c×gross)` (default **0.5**; parser default 0 = legacy uncoupled level) |
| `concentration_penalty` / `concentration_target_eff_assets` | Penalize under-diversification of risky weights (default **1.25** → **5.5** eff assets; parser default penalty 0) |
| `cash_daily_yield` | Risk-free accrual on cash before MTM (default **0.00012**/day ≈ 3.0% ann.; `0` disables; parser default 0 preserves old snapshots) |
| `churn_penalty` | Multiplier on `tx_cost_frac × reward_scale` (default 4.0) |

Omitted from the active config (parser default **0**, still loadable from old `Runs/*/config.yaml`): `inactivity_penalty_*`, `participation_*`, and their VIX/drawdown relief knobs.

## `environment` (selected)

| Key | Purpose |
|-----|---------|
| `self_state_features` | Append 4 causal portfolio self-state obs features: realized vol, downside vol, rolling excess vs the reward benchmark, near-cap fraction (default **true** → `obs_dim = 10N + 32`; parser default false keeps old snapshots at `10N + 28`) |

## `training`

| Key | Purpose |
|-----|---------|
| `timesteps` | Default PPO budget (default **50M**) |
| `early_stop_patience` | Stop after K evals with no new best robust score once curriculum completes (default **12**; `0` disables) |
| `best_model_trailing_evals` / `best_model_trailing_agg` | Selection score = median/mean of the last K **post-gate** eval scores (default **3** / `median`; `0`/`1` = legacy single-eval) |
| `best_model_stress_suite` / `best_model_stress_weight` | Blend a fixed 1.5×-fee / lag-2 stress eval into selection (default **true** / **0.3**; benchmark priced at the same stress fees) |
| `best_model_score_std_coef` / `best_model_score_dd_coef` | Eval selection penalties on std(excess) and drawdown (defaults **0.75**, **8.0**; dd uses **max** dd_frac in `defensive_sharpe`, p75 in `excess_sharpe`) |
| `best_model_score_stitched_blend` | Weight on **stitched** excess in the eval return signal (default **0.5** → 50/50 stitched/segment blend; `0` = segment mean only, `1` = stitched only) |
| `best_model_benchmark` | Passive book for eval excess: **`equal_weight_daily`** (default) or `balanced_6040` |
| `best_model_score_mode` | **`defensive_sharpe`** (728 default): excess-Sharpe signal with **max**(max_dd_frac) penalty. `excess_sharpe` uses p75. `excess_nav` (legacy, parser default): excess-NAV dollars |
| `best_model_max_dd_reject` | Reject best saves when continuous/multi-regime eval max DD exceeds this fraction (default **0.15**; parser **0** = off) |
| `oos_aligned_eval` / `oos_aligned_eval_bars` / `oos_aligned_eval_weight` | Continuous validation for selection (default **true** / **252** / **1.0**; parser defaults off/504/0). With `multi_regime_eval` off: chronological tail held out of train. With it on (727+): overlays spanning train history |
| `multi_regime_eval` / `multi_regime_eval_slices` / `multi_regime_eval_agg` | Multi-regime continuous selection (default **true** / **5** / **p25**; parser defaults off/5/p25). Aggregates per-slice scores; does not carve train data |
| `eval_score_burn_in_bars` | Drop first N bars of each eval episode before scoring / portfolio diagnostics (default **21**; parser default **0**) |
| `viz_freq` / eval cadence | Two-speed eval: every **3M** steps before the fee-ramp gate (`eval_freq_pre_gate_steps`), every **500k** after (`eval_freq_steps`) — ~50 evals per 50M run; plots every 500k (`viz_freq`) |

## `curriculum`

| Key | Purpose |
|-----|---------|
| `budget_short` | Fraction-of-run schedule anchor (default **50M**; must match `training.timesteps` for standard runs) |
| `fee_free_fraction` / `fee_ramp_fraction` | Fee-free then linear ramp (defaults **0.10** / **0.45** → ~5M / ~22.5M on a 50M run) |
| `churn_ramp_floor` | Churn scale at fee-ramp start; ramps to 1.0 by `fee_ramp_fraction` (default 0.1) |
| `dr_widen_span_fraction` | Progressive DR widening span after fee ramp (default **0.25** × learn budget → full DR at ~35M, ~15M stationary full-DR on a 50M run) |
| `best_model_min_step` | Gate `models/best/` saves until this step (`null` → `fee_ramp_end`; `0` → disable gate). Eval + portfolio diagnostics always logged. |

Entropy schedule (`entropy_schedule.decay_start_fraction` / `early_floor_fraction`, default **0.45**) aligns with `fee_ramp_fraction`. LR schedule (`hyperparameters.lr_schedule`, default **`phase_aware`**) holds the initial LR through DR widening (`dr_widen_end`) and cosine-decays to the floor over the settled remainder; `cosine` (parser default, preserves old snapshots) is the legacy global curve.

## Other sections

See the reward/cost tables in [README.md](../README.md) and [docs/RESEARCH.md](../docs/RESEARCH.md).
