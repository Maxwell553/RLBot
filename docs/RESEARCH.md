# Empirical Research Report: Multi-Asset Portfolio Management via Recurrent PPO

Methodology and walk-forward protocol for **RecurrentPPO** on a config-driven tradeable universe (**5–55** assets via `config/config.yaml` → `universe.assets`).

> **No definitive published OOS results yet.** Current pipeline: **`feature_split_mode: independent`**, cap **`max_single_asset_weight: 0.25`**, benchmark excess + Sortino (`risk_bonus_scale: 2.5`, `benchmark_excess_scale: 600`, combined constant cap **`benchmark_combined_abs_cap: 24.0`**), aligned train/eval fee curriculum, post-`fee_ramp_end` best-model gate. Any interim probes must be interpreted through their snapshotted `Runs/<id>/config.yaml`.

Each window trains on data through a fixed **train-end** date; a chronological **OOS holdout** never appears in training or in-training validation. When reported, OOS metrics will use **`Runs/<run_id>/models/best/best_model.zip`** (maximum mean in-training eval NAV **after full fees/churn on eval**), not holdout-tuned weights.

**Operations:** [TRAINING.md](TRAINING.md) · **Modal GPU:** [MODAL.md](MODAL.md) · **Implementation:** [README.md](../README.md)

---

## Executive summary

### Design

| Item | Value |
|------|--------|
| Policy | RecurrentPPO `MlpLstmPolicy`, 2×64 LSTM, MLP [128,128] |
| Tradeable **N** | `len(universe.assets)` (default example **N = 10**) |
| Observation | `obs_dim = 10N + 28` (includes per-asset **live mask**) |
| Action | `Box(−3,3)^(N+1)` → EMA logits → softmax (cash competes) → live mask → cap **0.25** → long-only simplex |
| Training episodes | `max_episode_steps = 252` (train); eval = full walk-forward segment |
| Training | 16 envs (local), `n_steps` 4096, `batch_size` 16384, `n_epochs` 3, `gamma` 0.975 → **12** backprop loops/pause; VecNormalize (obs on, reward norm train-only); cosine LR; fee/churn/DR curriculum |
| Modal training | Optional H100/A100 via `scripts/modal_app.py`; broker overrides `n_envs` + `batch_size` at launch ([MODAL.md](MODAL.md)) |
| Eval / viz cadence | Every **500k** global steps (not tied to `n_steps`); plot refresh `viz_freq: 500_000` |
| Checkpoints | Every **1M** global steps under `models/checkpoints/` |
| In-training eval | One deterministic rollout **per eval segment** (~# of 126-bar eval blocks) |
| Reward logging | `RewardDecompCallback` → TB `rew_decomp/*` + `eval_logs/reward_decomp.json` |
| Early stop | Optional `training.early_stop_patience` (after curriculum completes) |
| OOS backtest | `obs_lag` from the run manifest (else run config default), full fees, `churn_scale = 1`, action smoothing 0.15; run-local config/cache binding |
| Checkpoint rule | Eval-NAV-best only (`models/best/best_model.zip` + matched `best/vec_normalize.pkl`) |

### Hyperparameter protocol

Core hyperparameters live in `config/config.yaml` and are **copied to `Runs/<run_id>/config.yaml`** at train start. Walk-forward windows differ by **calendar flags** and **run id** only, not per-window YAML sweeps, unless a new study is intentional.

After any change to universe, `asset_live` panel, `obs_dim`, or reward coefficients, run `--refresh-data` (if data/universe changed) and train with a **new** run id — old checkpoints and VecNormalize stats are incompatible. OOS backtest execution binds the **run-local snapshots** by default for reproducibility: `Runs/<run_id>/config.yaml` for env mechanics (fees, cap, smoothing) and `Runs/<run_id>/data_cache.npz` for the panel (`--use-current-config` / `--data-cache` override; the summary records config/data hashes and warns when they drift from the training manifest).

### Auto-research loop (`scripts/research.py` + `specs/*.yaml`)

Method experiments are pre-registered as specs (hypothesis + allow-listed config `patch`/`grid`; universe, costs, split, and holdout dates are not patchable, and `windows` must reference the canonical table below). The orchestrator shells out to the canonical `train.py`/`backtest.py` and records every run in `Runs/<cohort>/registry.jsonl`:

```bash
python scripts/research.py plan    specs/reward_ablation.yaml   # materialize variant configs
python scripts/research.py launch  specs/reward_ablation.yaml   # tiers 1–3: train + in-training eval only
python scripts/research.py report  reward_ablation              # registry → report.md (OOS shown for tier ≥ 4 only)
python scripts/research.py promote specs/reward_ablation.yaml --variant <id> --promote
```

**Backends & throughput:** `launch --backend modal --modal-gpu H100` trains each variant on Modal (variant config pushed to the runs volume first; run tree pulled back before collect); `spec.budget.max_modal_hours` is enforced as a per-variant wall-clock cap on both backends. `research.py run-queue` drains `Runs/queue/*.yaml` sequentially (launch → report → move to `done/`/`failed/`) and **refuses promotion-requiring specs (tiers 4 and 5)** — holdout reads and shadow starts stay human actions. `research.py screen <spec> --screen-timesteps 2000000 --keep-top 0.25` runs every grid combo at tier 1 with a tiny budget and writes `screen_ranking.json` naming the top fraction to advance to a full-tier launch under a new spec id (successive halving; never touches the holdout).

**Agent-proposed specs:** `research.py validate <spec> [--agent]` is the proposer's no-side-effect feedback loop (schema, firewall, canonical windows, gate keys, id collisions). `--agent` additionally requires `hypothesis`, `parent`, and `success_gates`, and refuses promotion-requiring tiers (4 and 5) — an autonomous proposer can iterate freely on tiers ≤ 3 via the queue, while promotion (and every holdout read) stays a human action. Tier ≥ 4 launches and promotes refuse a **dirty working tree** — and fail closed when git state cannot be determined at all (`--allow-dirty` overrides, recorded): an OOS number must be attributable to a commit.

**Tier 5 — shadow trading (`scripts/shadow_trade.py`):** forward evaluation that never burns a holdout. A daily `record --refresh-data` (after the close) refreshes the **global** cache (the run snapshot is frozen by design), runs the audited inference path (frozen VecNormalize, manifest-checked panel, OOS-ledger logged), and appends target weights + provenance + an **observation-drift alarm** (fraction of normalized features >5σ from frozen training stats; thresholds are heuristics, `--drift-sigma/--drift-frac`) to the gitignored `execution/shadow_ledger_<RUN_ID>.jsonl`. Rows are keyed by their **decision bar** — the bar whose observation actually produced the weights (the rollout env needs two later bars to execute a step, so the recorded book lags the cache tail; the ledger is honest about that). `reconcile` (torch-free) fills realized open→open returns **net of linear costs** (turnover vs the previous recorded book × slippage+fee, plus daily holding; the cap-weighted buy-and-hold benchmark pays holding only; no market-impact/capacity model); `report` summarizes the accumulating true walk-forward record. This is the natural terminal arbiter of the continuous loop.

**Cross-cohort memory:** `research.py report --all` aggregates every `Runs/*/registry.jsonl` into `Runs/research_report_all.md`: per-cohort summaries with `parent` lineage, holdout-read counts, and a **knob-sensitivity table** (median best-eval-NAV per patched config value, normalized as a delta vs its cohort's median). This is the registry-as-memory view a hypothesis proposer reads before writing the next spec.

The OOS firewall: tiers 1–3 never touch the holdout; tier ≥ 4 requires `--promote`, is budgeted (`--oos-budget`, default 1 read per launch), and every holdout read is written to the registry **before** it happens — a variant with a recorded tier-4 read cannot be re-scored (`--allow-failed-rescore` only retries crashed, never-scored reads). Published OOS numbers carry the cohort's variant count; interpret them with that multiplicity in mind.

---

## Walk-forward status (results pending)

**Run id convention:** `--window N` → `W{N}_MMDD` (month/day at launch); collisions get `_a`, `_b`, …; or pass `--run-id <RUN_ID>` explicitly. Each run snapshots the **current** `config/config.yaml` at train start.

Record OOS metrics here only after a run finishes training **and** you backtest with the run-local snapshot:

```bash
python scripts/backtest.py --run-id <RUN_ID> --checkpoint best --detailed --stochastic-paths 30 --plot-tag best
```

Outputs: `Runs/<run_id>/backtest_summary.json`, `Runs/<run_id>/plots/backtest_best.png`, `Runs/<run_id>/plots/training.png`.

### Walk-forward registry

Use `--window N` on `train.py` (or `modal run scripts/modal_app.py -- …`) for auto ids like `W{N}_MMDD`, or `--run-id <RUN_ID>` for a custom name.

| Window | `--train-end` | OOS (`--holdout-start` … `--holdout-end`) | `--until` | Sample `run_id` | Training | OOS backtest |
|--------|---------------|-------------------------------------------|-----------|-----------------|----------|--------------|
| 1 | 2015-12-31 | 2016-01-01 … 2017-12-31 | 2017-12-31 | `W1_MMDD` | Pending | Pending |
| 2 | 2017-12-31 | 2018-01-01 … 2019-12-31 | 2019-12-31 | `W2_MMDD` | Pending | Pending |
| 3 | 2019-12-31 | 2020-01-01 … 2021-12-31 | 2021-12-31 | `W3_MMDD` | Pending | Pending |
| 4 | 2021-12-31 | 2022-01-01 … 2023-12-31 | 2023-12-31 | `W4_MMDD` | Pending | Pending |
| 5 | 2023-12-31 | 2024-01-01 … 2025-12-31 | 2025-12-31 | `W5_MMDD` | Pending | Pending |
| 6 | 2025-12-31 | 2026-01-01 … 2027-12-31 | (omit / latest bar) | **embargoed** | — | reserved (terminal validation only) |

Window *N* trains through Dec-31 of `2013 + 2N` with a two-year holdout. This table is canonical: research specs may only reference these windows (`rlbot/research/spec.py:CANONICAL_WINDOWS` rejects anything else — a spec that placed its own holdout would change what OOS *is*). **W6 is embargoed** (`EMBARGOED_WINDOWS`): it is the reserved terminal validation window, untouched by the iterate-measure loop, usable only for a final human-run pre-deployment validation or the tier-5 shadow path.

### Holdout-burn accounting & selection-aware significance

- **Every** OOS backtest — research-launched or manual — appends a record to the global ledger `Runs/oos_ledger.jsonl` *before* the rollout starts (a crash still burns). Burn is counted in **distinct models per window** (re-scoring the same run adds no selection pressure).
- `research.py launch/promote` enforce a cumulative per-window budget (`--window-budget`, default `oos_ledger.DEFAULT_WINDOW_BUDGET = 10` distinct models). Past budget, further research reads are refused — iterate on tiers ≤ 3 instead.
- The ledger starts empty at its introduction: holdout reads that predate it are invisible, so early trial counts are a flattering undercount — treat early DSR values as upper bounds.
- Trial counts aggregate **same-key** reads only: canonical-window reads share one key, while tail-mode and `infer_weights`/shadow reads are keyed by their realized slices and never inflate (or deflate) a canonical window's count. In particular, daily shadow records overlapping W6's calendar range add no W6 trials — a future terminal W6 validation should reckon with that informal forward-selection feedback separately.
- `backtest_summary.json` reports `deflated_sharpe`: the probabilistic Sharpe ratio (Bailey & López de Prado, skew/kurtosis-adjusted) evaluated against the expected max Sharpe of *N* zero-skill strategies, with *N* = the ledger's distinct-model count for the window read. DSR > 0.95 is the conventional significance bar **after** selection. (Simplification: the null cross-trial SR variance is taken as `1/n_obs`; see `rlbot/stats.py`.)
- **Pre-registered decision rules**: spec `success_gates` (`min_seeds`, `eval_nav_mean_min`, `eval_nav_median_min`, `eval_nav_spread_max_frac`, `oos_sharpe_min`, `oos_max_drawdown_floor`, `deflated_sharpe_min`) are evaluated per seed-group at `collect`; verdicts (`pass`/`fail`/`inconclusive`) land in `Runs/<cohort>/verdicts.json`. `promote` refuses to spend a holdout read on a group whose verdict is not `pass` unless `--force-gates` is passed explicitly.

**Local train example (window 1):**

```bash
python scripts/train.py --window 1 --timesteps 65000000 \
  --since 2006-01-01 --train-end 2015-12-31 \
  --holdout-start 2016-01-01 --holdout-end 2017-12-31 --until 2017-12-31
```

**Modal train example (window 2, H100):**

```bash
modal run scripts/modal_app.py -- \
  --modal-gpu H100 --window 2 --run-id <RUN_ID> --timesteps 65000000 \
  --refresh-data --since 2006-01-01 --train-end 2017-12-31 \
  --holdout-start 2018-01-01 --holdout-end 2019-12-31 --until 2019-12-31
python scripts/modal_app.py sync --run-id <RUN_ID> --watch
```

When advancing to a later window on Modal, pass `--refresh-data` with that window’s `--until` (or upload a full local cache once) so the shared `rlbot-cache` volume covers the new holdout dates.

### OOS performance (to be filled)

All cells **TBD** until each window completes training and passes the backtest command above on the run-local `config.yaml` + `data_cache.npz`. Record the actual `<RUN_ID>` used in the `run_id` column when filling.

| Window | `run_id` | Agent total return | Agent Sharpe | Max DD | SPY B&H | Equal-weight | 60/40 | Risk parity |
|--------|----------|-------------------|--------------|--------|---------|--------------|-------|-------------|
| 1 | — | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 2 | — | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 3 | — | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 4 | — | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 5 | — | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 6 | — | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

*Reporting rule (when filled): agent columns from **`best_model.zip`** (eval-NAV-best) via `scripts/backtest.py --detailed --stochastic-paths 30`; benchmarks on the same OOS window. Each row must match the snapshotted `Runs/<run_id>/config.yaml` — do not mix runs trained under different configs.*

### Cross-window generalization (optional, TBD)

Loading window *N* weights on window *N+1* holdout tests regime shift without retraining. Override holdout dates on `backtest.py` **and** pass `--until` through the new holdout end (manifest `until` clips the cache otherwise). **No numbers recorded here yet** — run only after the primary window backtests are complete.

---

## Passive benchmark methodology

Implemented in `rlbot/baselines.py`; plotted by `scripts/backtest.py`. Multi-asset books aggregate **simple returns** cross-sectionally each day, then compound.

| Benchmark | Allocation | Rebalance |
|-----------|------------|-----------|
| Cash / no-trade | 100% cash (flat NAV) | — |
| Benchmark-only B&H | 100% `universe.benchmark` sleeve (default SP500/SPY) | Buy-and-hold (tx costs on entry) |
| Equal-weight (daily) | 1/N per live asset | Daily; slippage + fees + holding costs |
| Equal-weight (monthly) | 1/N per live asset | Calendar month-start; tx-cost-aware |
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

1. **Fractional differentiation** (`data.fracdiff_d: 0.4`) on log prices.
2. **Walk-forward blocks:** `train_test_split_alternating`, `block_size` 126, `eval_stride` 4 (`WalkforwardEnvPack` in `data_utils.py`).
3. **Feature split mode** (`data.feature_split_mode`): **`independent`** (default — recompute per segment over a causal preroll window, `feature_preroll_bars: 252`, neutralizing only panel-head bars without preroll history via `feature_purge_warmup: 25`) vs `continuous` (compute on full trainable panel, slice per block; eval carries indicator memory across adjacent blocks — a model-*selection* signal, not an independent estimate). Holdout reserved first in both modes.
4. **`asset_live` panel:** no global calendar `dropna`; missing pre-IPO bars filled; live mask gates allocation **and** per-asset features are neutralized on pre-live bars (no pre-IPO price level reaches the observation).
5. **Causal execution:** features at `t` use `close[t−obs_lag]`; holding cost at `close[t]`; fill `open[t+1]`; MTM `close[t+1]`.
6. **Chronological holdout:** removed before train/eval; only `scripts/backtest.py` uses OOS bars.
7. **HY OAS macro:** causal expanding OLS calibration (no per-bar `polyfit` look-ahead).
8. **Risk-parity baseline:** inverse-vol weights use only past returns; IPO names borrow mean peer vol during warmup.

---

## Environment & reward

**Policy:** `Box(−3, 3)^(N+1)` logits.

**Action → weights:** EMA on logits (`action_smoothing_alpha: 0.15`, train + backtest) → **softmax** (cash competes with risky assets) → `asset_live` mask → per-asset cap (`max_single_asset_weight: 0.25`) with redistribute + final projection to cash → long-only simplex.

**Observation:** `obs_dim = 10N + 28` — per-asset fracdiff/vol/RSI/MACD/trend, live mask, portfolio state, drawdown/progress, four macro series (not tradeable).

**Episodes:** `max_episode_steps: 252` (train); eval = full segment. Early stop if NAV ≤ `stop_loss_fraction` (0.45) × episode-start NAV.

**Domain randomization (training only):** after fee curriculum releases, per-episode `obs_lag` ∈ {0,1,2} and Beta-mapped `fee_scale`; bounds widen through DR phase. **Eval:** mirrors train fee/churn curriculum (linear ramp); fixed `obs_lag` (config default), no DR. OOS backtest: `obs_lag` from the run manifest, `fee_scale = 1`.

### Reward decomposition

Per-step (before VecNormalize during training):

`reward = return (with downside amp) + benchmark_excess + sortino_diff + participation − inactivity − churn`

| Component | Sign | Implementation | Default coefficients |
|-----------|------|----------------|-------------------|
| **Return** | + | `clip(log_ret, …) × reward_scale`; negative returns amplified by `(1 + drawdown_downside_gamma × dd_pre)` | clip **−0.12 / +0.06**; scale **2000**; `drawdown_downside_gamma: 5` |
| **Benchmark excess** | + | `clip(agent_log_ret − bench_log_ret, ±clip) × benchmark_excess_scale` (same friction model as Sortino bench) | `benchmark_excess_scale: 600`; `benchmark_excess_clip: 0.04` |
| **Sortino differential** | + | Agent vs cap-weighted benchmark Sortino over `risk_window`, clipped ±3; downside deviation floored at `sortino_downside_floor: 0.001` (10 bp/day) | `risk_bonus_scale: 2.5` |
| **Bench cap** | (meta) | Sortino + benchmark scaled so combined \|.\| ≤ `benchmark_combined_abs_cap` (a **constant** — a relative cap was reward-hackable) | **24.0** (`0` disables both) |
| **Participation** | + | `gross_exposure × participation_bonus × participation_reward_scale` | `0.05 × 20` |
| **Inactivity** | − | `cash_frac × inactivity_penalty_over_50` + ramp 90%→100% | **1.35** + **0.9** tail (max **~2.25** at 100% cash) |
| **Churn** | − | `tx_cost_frac × churn_penalty × reward_scale × VIX_mult × curriculum_churn_scale` | `churn_penalty: 1.0` |
| **Drawdown amp** | (in return) | Extra negative return when underwater; logged as `rew_decomp/drawdown` | `drawdown_downside_gamma: 5` |

`info` / TensorBoard: `rew_decomp/return`, `benchmark`, `sortino`, `participation`, `inactivity`, `churn`, `drawdown`, `vix_churn_mult`.

**Churn detail:** `tx_cost_frac` = realized slippage + fee dollars ÷ NAV (zero when `fee_scale = 0`). `VIX_mult = clip(VIX/18, 0.75, 1.5)`. Train **and eval** `curriculum_churn_scale` is **0** during fee-free phase, then **0.1 → 1.0** over the fee ramp window. OOS backtest: `churn_scale = 1`.

**Inactivity detail:** Training and eval envs use `eval_inactivity_penalty_scale: 1.0` (balanced with return term; cash can beat a −1% day).

**Costs:** per-asset **slippage**, **tx_fee**, and **annual_holding_cost** (length-N lists in `transaction_costs`, keyed like `universe.assets`). **Train + eval** curriculum: `fee_scale` **0** → linear ramp → **1.0**; DR widening on train only after ramp. OOS backtest: `fee_scale = 1`.

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
| Train | `modal run scripts/modal_app.py -- --modal-gpu H100 --window N --run-id <RUN_ID> …` |
| Watch plot | `python scripts/modal_app.py sync --run-id <RUN_ID> --watch` → `Runs/<run_id>/plots/training.png` |
| Pull all artifacts | `python scripts/modal_app.py sync --run-id <RUN_ID> --pull-all` |
| Backtest locally | `python scripts/backtest.py --run-id <RUN_ID> --checkpoint best --plot-tag best` |

**Volumes:** `rlbot-runs` (per-run tree: models, logs, `config.yaml`, `data_cache.npz`, plots) · `rlbot-cache` (shared OHLCV panel). `--watch` syncs plots/eval only; models and logs require `--pull-all`.

**GPU broker:** `--modal-gpu` selects vCPUs, `n_envs`, and `batch_size` at launch (e.g. H100 → 64 envs, batch 65536). Cannot change `n_envs` mid-run.

**Data:** Each window’s `--until` must be covered by the cache on `rlbot-cache`. Use `--refresh-data` per window or `modal run scripts/modal_app.py::upload_cache` once from a full local `.cache/data_cache.npz`.

---

## Evaluation & inference

| Script / module | Purpose |
|-----------------|---------|
| `scripts/backtest.py` | OOS rollout, benchmarks, stochastic-path fan plot; writes `backtest_summary.json`; binds run-local `config.yaml` + `data_cache.npz` by default |
| `scripts/infer_weights.py` | Audited target weights for a single `--as-of` date (provenance-rich JSON; no broker) |
| `scripts/research.py` | Auto-research: `plan`/`launch`/`report` over `specs/*.yaml` with OOS-gated tiers |
| `rlbot/inference_load.py` | VecNormalize + RecurrentPPO load helpers (frozen obs norm for OOS) |
| `rlbot/inference_output.py` | Torch-free weight-payload assembly/validation |
| `rlbot/stats.py` | Block-bootstrap helpers for `--detailed` backtest stats |

Default `--checkpoint` is **`best`** (eval-NAV-selected); `latest`/`both` print an OOS-touch warning.

---

## Training loop & curriculum (65M timesteps)

RecurrentPPO + VecNormalize (obs norm on; reward norm train-only) + `TradingCurriculumCallback` (fee/churn on **train + eval**) + `EvalNavBestModelCallback` (best saves gated until **`fee_ramp_end`**) + `AdaptiveEntropyCallback` + `RewardDecompCallback` + `TrainingVizCallback`. Cosine LR to `learning_rate_floor`. Optional patience early-stop via `training.early_stop_patience` (after `dr_widen_end` curriculum milestone). Milestones use `curriculum.budget_short` fractions in `config/config.yaml`:

| Phase | Approx. step (65M run) | Effect |
|-------|------------------------|--------|
| Fee-free | 0 – 6.5M (`fee_free_fraction` 0.10) | `fee_scale = 0` (train + eval) |
| Fee ramp | 6.5M – 29.25M (`fee_ramp_fraction` 0.45) | fees → full (train + eval, linear) |
| Churn ramp | 6.5M – 29.25M (aligned with fee ramp) | `curriculum_churn_scale` 0 → 0.1 → 1.0 (train + eval) |
| Best-model gate | opens at **29.25M** (`fee_ramp_end`; `best_model_min_step: null`) | eval NAV logged always; `models/best/` updates only after full eval fees + churn |
| DR widen | through ~42.25M (`dr_widen_span_fraction` 0.65) | fee/lag domain-randomization bounds widen (**train only**) |
| Entropy | cosine decay from `decay_start_fraction` 0.45 (~29.25M) | exploration → `final_ent` |
| Eval / plot | every **500k** global steps | ~130 eval rollouts per 65M run (decoupled from `n_steps`) |
| Checkpoints | every **1M** global steps | `models/checkpoints/ppo_*_steps.zip` |

```bash
python scripts/train.py --refresh-data --timesteps 1000 --run-id _data_refresh --no-viz
python scripts/train.py --window 1 --timesteps 65000000 \
  --train-end 2015-12-31 --holdout-start 2016-01-01 --holdout-end 2017-12-31 --until 2017-12-31
python scripts/backtest.py --run-id <RUN_ID> --checkpoint best --detailed --stochastic-paths 30 --plot-tag best

# Modal equivalent (pull artifacts before backtest)
modal run scripts/modal_app.py -- --modal-gpu H100 --window 1 --run-id <RUN_ID> --timesteps 65000000 \
  --since 2006-01-01 --train-end 2015-12-31 \
  --holdout-start 2016-01-01 --holdout-end 2017-12-31 --until 2017-12-31
python scripts/modal_app.py sync --run-id <RUN_ID> --pull-all
python scripts/backtest.py --run-id <RUN_ID> --checkpoint best --detailed --stochastic-paths 30 --plot-tag best
```

**Do not load** checkpoints when `manifest.universe.obs_dim` or `universe.tickers` ≠ current config/cache. Reward/cost/config edits require a **new** `--run-id`; weights were optimized under the snapshotted `Runs/<id>/config.yaml`. Do not compare numbers across runs trained under different config snapshots.

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

- **No published OOS edge** under the current pipeline until the tables above are filled from fresh backtests.
- **No guarantee** of live performance or absence of multiple-testing bias across windows.
- Pre-refactor artifacts and informal cross-window probes are **not** evidence for the current method — the harness and reward stack changed materially.
- When results are eventually recorded, they are evidence **within the documented pipeline** only, not forecasts of live trading performance.
