# AGENTS.md

This file provides guidance to coding agents (e.g. OpenAI Codex) when working with code in this repository. It mirrors `CLAUDE.md`.

MarketTrainer (RLBot) trains a RecurrentPPO (LSTM) agent on a multi-asset daily portfolio environment, with strict chronological out-of-sample holdouts. `README.md` is the canonical reference for the asset universe, observation layout, reward formula, and walk-forward windows — read it for those details. This file covers commands and the invariants that span files. **No published OOS results** under the current pipeline yet; use `<RUN_ID>` placeholders in docs and commands until fresh backtests complete.

Library code lives in the `rlbot/` package (`data_utils.py`, `trading_env.py`, `rl_config.py`, `baselines.py`, `run_artifacts.py`, `inference_load.py`, `inference_output.py`, `vecnorm_utils.py`, `modal_cloud.py`, `stats.py`, `reward_logging.py`, `research/`). CLIs live in `scripts/` (`train.py`, `backtest.py`, `modal_app.py`, `infer_weights.py`, `research.py`, `migrate_runs_layout.py`, `run_seed_ensemble.sh`). There is **no** top-level `train.py`/`backtest.py`, **no** `windows/` directory, and **no** `paper_trade/` tree.

## Commands

```bash
# Setup (a fresh clone has no deps installed)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # runtime deps + pytest; add ".[modal]" for cloud training

# Tests (no network, no training loop). Full suite needs the editable install (pulls torch).
# A minimal math/env-only run needs just gymnasium + pandas + numpy.
pytest                            # all
pytest tests/test_environment.py  # one file
pytest tests/test_core.py::test_fracdiff_weights_start_at_one  # one test
pytest -k cap                     # by keyword

# Train (defaults from config/config.yaml; first run needs --refresh-data to fetch yfinance data and build cache)
python scripts/train.py --refresh-data
python scripts/train.py --config path/to/config.yaml   # override hyperparameters
python scripts/train.py --window 1 --run-id <RUN_ID>   # walk-forward window via date flags
python scripts/train.py --n-assets 5                    # slice universe to first N keys (5–55)

# Seed ensemble (sequential runs, shared holdout) — the only shell helper in-tree
scripts/run_seed_ensemble.sh --cohort my_cohort --seeds "42 101 777" -- --window 1

# OOS backtest (window flags default from the run manifest; prefer --run-id over hand-passing dates)
python scripts/backtest.py --run-id <RUN_ID> --checkpoint best --detailed
python scripts/backtest.py --run-ids <RUN_ID_1>,<RUN_ID_2> --checkpoint best  # batch, one process
python scripts/backtest.py --ensemble-prefix my_cohort --detailed        # writes ensemble_summary.json

# Cloud training (see docs/MODAL.md)
modal run scripts/modal_app.py -- --modal-gpu H100 --window 1 --run-id <RUN_ID>

# Auto-research loop (spec → train/backtest → JSONL registry → report; OOS-gated)
python scripts/research.py plan   specs/feature_split_ab.yaml   # materialize variant configs
python scripts/research.py launch specs/feature_split_ab.yaml   # tiers 1–3; tier ≥4 needs --promote
python scripts/research.py report feature_split_ab

# Audited target weights for a run (provenance-rich; measurement only, no broker)
python scripts/infer_weights.py --run-id <RUN_ID> --checkpoint best --as-of 2022-12-31
```

CLI entry points after `pip install -e .`: `market-trainer-train`, `market-trainer-backtest` (see `[project.scripts]` in `pyproject.toml`). There is no linter configured; the code targets Python ≥ 3.10.

## Configuration is the single source of truth — install it before using env behavior

`config/config.yaml` → `rlbot/rl_config.py` (`load_config` / `_parse_config`) parses it into frozen `RLConfig` dataclasses. Non-obvious consequences:

- **`set_config(cfg)` installs a global singleton read via `get_config()`.** The environment captures the relevant config objects at construction (`self._env_cfg`, `self._reward_cfg`, and per-asset cost arrays in `MultiAssetPortfolioEnv.__init__`) and reads those — there are **no** synced module-level constants (no `sync_trading_env_aliases`, no `REWARD_SCALE`/`MAX_SINGLE_ASSET_WEIGHT` globals). **So a changed reward/cost/env field in `config.yaml` only takes effect for envs built after `set_config(load_config(...))` runs.** Tests rely on `tests/conftest.py`, which loads and installs the config in an autouse session fixture; standalone scripts must call `set_config(load_config(...))` before constructing envs. `scripts/train.py` does this in a two-pass argparse so `--config` and `--n-assets` are applied before any other default.
- Each training run snapshots `config/config.yaml` into `Runs/<run_id>/config.yaml` and writes `Runs/<run_id>/manifest.json` for reproducibility. Treat config + run_id as a pair. `scripts/backtest.py` and `scripts/infer_weights.py` load that snapshot by default (see "Train/backtest must agree").

`config.yaml` lists per-asset arrays (slippage, tx_fee, holding cost, benchmark cap weights) that **must have exactly N entries** in `universe.assets` key order, where `N` = number of `universe.assets` keys (default 10: SP500, GOLD, OIL, EURUSD, USDJPY, NIKKEI, FTSE, BOND10Y, COPPER, EM; supported range 5–55). `rl_config._float_list` / `validate_config_for_universe` enforce the length match; `--n-assets N` slices the first N keys and renormalizes benchmark weights.

## Data-leakage prevention is the core design constraint

The pipeline in `rlbot/data_utils.py` is built so indicators never see the future. When touching data handling, preserve these:

- **Chronological holdout** (`reserve_chronological_holdout`) strips the OOS tail *before* any train/eval split. Only `scripts/backtest.py` ever sees it.
- **Feature split mode** (`data.feature_split_mode`, default `continuous`) controls how `train_test_split_alternating()` builds train/eval block features (all features are strictly causal and the holdout is reserved first, so neither mode leaks OOS data):
  - `continuous` (default): RSI/MACD/fracdiff/trend/realized-vol are computed by `compute_feature_panel()` on the contiguous panel (cache or built once) and **sliced** into blocks. The in-training **eval** signal carries indicator memory continuous with adjacent train blocks (intentional — "matches continuous backtest memory"); treat eval NAV as a model-*selection* signal, not a fully independent estimate. `feature_purge_warmup` is **not** applied here.
  - `independent`: features are recomputed per contiguous segment and the first `feature_purge_warmup` (config `data.feature_purge_warmup: 25`) bars of each segment are neutralized via `_neutralize_feature_warmup()`, so eval blocks are not feature-state-contaminated by adjacent train blocks.
- **Causal execution** (`rlbot/trading_env.py`): market features at bar `t` use data through `close[t - obs_lag]`; holding cost is deducted on pre-rebalance units at `close[t]`; rebalance fills at `open[t+1]`; mark-to-market at `close[t+1]`. `obs_lag` is randomized over {0,1,2} during training (`min_obs_lag`=0, `max_obs_lag`=2, after the fee curriculum releases) and fixed to **1** in OOS backtest (`backtest.py --obs-lag`, default 1).

## Train/backtest must agree

A backtest is only valid if its window flags match training. `--until`, `--train-end`, `--holdout-start`, `--holdout-end`, `--holdout-days`, and `--obs-lag` must reproduce the training split. `scripts/backtest.py` defaults most of these from `Runs/<run_id>/manifest.json` when `--run-id` is given, so prefer `--run-id` over hand-passing dates.

Backtest binds the run's own snapshots by default for reproducibility:

- `backtest.py` loads `Runs/<run_id>/config.yaml` for costs/cap/env mechanics (override with `--use-current-config`) and prefers the run-local `Runs/<run_id>/data_cache.npz` over the global cache (override with `--data-cache PATH`), so editing config or refreshing the cache does not silently change an old run's OOS numbers.
- `--checkpoint` defaults to **`best`** (eval-NAV-selected; holdout not used to pick weights); `latest`/`both` print an OOS-touch warning.
- Each single-run backtest writes `Runs/<run_id>/backtest_summary.json` (override `--summary-json`) with metrics plus config/data-cache hashes for drift detection.

## Inference freezes normalization

`VecNormalize` updates running obs/reward statistics during training. For any out-of-sample rollout (`scripts/backtest.py`, `scripts/infer_weights.py`), freeze it first via `rlbot.vecnorm_utils.freeze_vec_normalize_for_inference()` — it sets `training=False`, disables reward norm, keeps obs norm. `rlbot/inference_load.py` (`load_recurrent_ppo_inference`, `load_vec_normalize_for_inference`) wraps this for backtest/inference. Never let OOS data update the running stats.

## Action space vs asset count

`N_ACTIONS = N + 1` (cash + N risky assets); for the default N=10 that is 11 actions. The policy outputs `Box(-3, 3)^(N+1)` → optional EMA smoothing on the logits → softmax (cash competes) → live-mask → per-risky-asset clip-and-redistribute cap (`max_single_asset_weight`, default **0.35**) → long-only simplex (`portfolio_weights_from_action`). Observation dimension is `observation_dim_for_universe(N) = 10*N + 28` (**128** for N=10). Macro series (DXY, TNX, VIX, HY OAS; 4 series) feed the observation **only** — they are not tradeable. `N` is dynamic (5–55); derive dims from `N`, never hard-code 11/118.

## Determinism

`apply_deterministic_seeds()` (in `rlbot/rl_config.py`, called by `scripts/train.py`) seeds Python/NumPy/Torch, sets `PYTHONHASHSEED`/`CUBLAS_WORKSPACE_CONFIG`, enables cuDNN-deterministic and `torch.use_deterministic_algorithms`. Keep this path intact when adding randomness — route new RNG through seeded sources. By default this is **not** full bit-reproducibility: training envs use `reseed_on_reset=True` (fresh OS entropy per episode for diversity), so two same-seed runs still diverge in episode starts / domain-randomization draws ("seeded framework + stochastic episode resets"). Set `training.reproducible: true` to use deterministic per-env seed streams (`seed + env index`) instead, so same-seed runs reproduce.

## Auto-research loop and inference

- **Experiment specs** (`specs/*.yaml`, parsed by `rlbot/research/spec.py`) declare a hypothesis + a config `patch`/`grid`. A patch may only target method knobs (`reward.*`, `curriculum.*`, `entropy_schedule.*`, `policy.*`, `hyperparameters.*`, `environment.*`, `data.feature_split_mode`, safe `training.*`). Patching the universe, transaction costs, holdout dates, or the split (`block_size`/`eval_stride`/`holdout_days`) is **rejected** — those change what OOS is.
- **`scripts/research.py`** (`plan`/`launch`/`collect`/`report`/`promote`) shells to the canonical `train.py`/`backtest.py`, writes `Runs/<cohort>/registry.jsonl`, and enforces the OOS firewall via `rlbot/research/gates.py`: tiers 1–3 train + in-training eval only; tier ≥ 4 reads the holdout **once per variant** and requires `--promote`.
- **Per-term reward logging**: training writes `Runs/<id>/eval_logs/reward_decomp.json` + TB scalars (`rew_decomp/*`) so the reward balance (e.g. inactivity vs participation) is observable.
- **`scripts/infer_weights.py`** emits audited target weights (long-only simplex ≤ cap) with full provenance (config/data/model hashes), reusing the backtest rollout. Measurement only — no broker adapter, no market-impact/capacity model.
- New env behavior knobs in `config.yaml`: `training.early_stop_patience` (>0 → patience early-stop after the curriculum completes) and `training.reproducible` (deterministic per-env seed streams).

## Artifacts and gitignored paths

The canonical run layout is `Runs/<run_id>/` (capital R): `manifest.json`, `config.yaml`, `data_cache.npz`, `models/{final,best,checkpoints}/`, `plots/`, `logs/`, `tb_logs/`, `eval_logs/`. Run paths/IDs/manifests are managed by `rlbot/run_artifacts.py` (`RunPaths`, `new_run_id`, `write_manifest`, `read_run_manifest`, `discover_run_ids_with_models`, `resolve_data_cache`, `snapshot_data_cache`). There is no `LATEST.txt`.

Gitignored (see `.gitignore`): `.cache/`, `data_cache.npz`, `Runs/`, the legacy artifact roots `models/ runs/ tb_logs/ logs/ plots/` (pre-`Runs/` layout, migratable via `scripts/migrate_runs_layout.py`), `ibkr_paper/`, and `execution/**`. Broker automation and local execution state stay out of the research tree.
