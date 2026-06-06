# Codex Repository Review - 2026-06-05

## Scope

This review is based on a static read of the current repository, including:

- `README.md`
- `docs/RESEARCH.md`
- `docs/TRAINING.md`
- `docs/MODAL.md`
- `docs/RLBot_Critical_Review.md`
- `docs/RLBot_Design_Review.md`
- `config/config.yaml`
- `config/README.md`
- `rlbot/data_utils.py`
- `rlbot/trading_env.py`
- `rlbot/rl_config.py`
- `rlbot/baselines.py`
- `rlbot/run_artifacts.py`
- `rlbot/inference_load.py`
- `rlbot/vecnorm_utils.py`
- `scripts/train.py`
- `scripts/backtest.py`
- `scripts/modal_app.py`
- `rlbot/modal_cloud.py`
- `scripts/run_seed_ensemble.sh`
- `tests/`

I did not run a training job or OOS backtest. The findings below are therefore a design and implementation review, not an empirical result.

## Executive Verdict

RLBot is a strong research prototype with unusually good bones for RL trading work: a typed config layer, dynamic asset-count support, strict chronological OOS holdouts, causal next-open execution, VecNormalize freezing for inference, run manifests, Modal support, seed-ensemble helpers, and a documented research process.

The repo is not yet ready for an unsupervised "auto research" loop that freely generates, trains, and ranks candidates. It is close enough to benefit from that pattern, but only after sealing a few methodological and reproducibility gaps. The biggest current issue is not model architecture; it is the integrity of the experiment harness.

The most important current findings:

1. The dynamic-universe refactor is real. The older "hard-coded 10 assets" critique is mostly obsolete, although some baselines and docs still assume default assets.
2. The older "eval only repeats first 63 bars" critique is also mostly obsolete. Current eval runs one full deterministic rollout per eval segment.
3. A new/current split behavior weakens the anti-leakage story: walk-forward train/eval blocks now slice a continuous, precomputed feature panel, and `feature_purge_warmup` is explicitly unused. This is not chronological OOS leakage, but it is train/eval feature-state contamination inside the model-selection signal.
4. Backtest reproducibility is weaker than the run layout implies: backtest uses the current global config and current global `.cache/data_cache.npz`, not the run's snapshotted `config.yaml` and `Runs/<run_id>/data_cache.npz`.
5. The docs and agent instructions are materially out of sync with the code in several places: `Runs/` vs `runs/`, dynamic obs/action dimensions vs stale 10-asset constants, cap 0.35 vs older 0.50, feature purge/per-segment claims, and missing/renamed execution directories.
6. The auto research pattern is feasible and likely valuable, but should be built as a gated experiment operating system, not as an unconstrained model-tuning agent.

## Comparison to Prior Assessments

The prior review documents in `docs/` are valuable, but parts are now stale or internally inconsistent with current code.

### Claims That Still Hold

- The repo's strongest asset is its research hygiene: explicit chronological holdouts, next-open execution, per-run artifacts, and candid research reporting.
- The policy-selection idea is right: choose `best_model.zip` by in-training eval ending NAV rather than shaped reward.
- Training still runs the full budget even when eval NAV peaks early. There is no automated early stopping.
- OOS evidence is still thin in the statistical sense. `scripts/backtest.py` now has better tooling, but each window is still one historical market path, and repeated research passes can easily become multiple testing.
- The deployment/inference story is incomplete. There are fast load helpers and backtest rollout code, but no small audited "produce today's target weights from run id" CLI or paper-trading path in-tree.
- Dependency reproducibility is still weak: requirements are `>=` ranges and there is no lock file.
- Same-seed reproducibility remains intentionally broken by `reseed_on_reset=True` plus fresh OS entropy on reset.

### Claims That Are Now Obsolete or Partly Fixed

- "Hard-coded 10 assets" is mostly fixed. `rl_config.py` supports 5-55 assets, `observation_dim_for_universe(n_assets)` is dynamic, the env derives `n_assets` from the OHLCV panel, and tests cover 5, 23, and 55 assets.
- "Eval repeats 75 fixed 63-bar episodes" is mostly fixed. Current config sets `eval_n_episodes: 1` as a fallback, `train.py` uses `len(eval_segments)`, and deterministic eval resets run full segments.
- "Backtest uses iid bootstrap" is fixed. `scripts/backtest.py` now implements stationary block bootstrap.
- "HY OAS uses non-causal polyfit" is fixed in the main path. The current implementation uses causal expanding affine calibration. Data-source and coefficient-persistence concerns remain.
- The old module-global alias concern is largely gone. Current `set_config()` only installs `_CONFIG`; there is no `sync_trading_env_aliases()` in the current `rl_config.py`.

### New or Current Regressions

- `train_test_split_alternating()` now documents that precomputed features are sliced from the trainable timeline and that `feature_purge_warmup` is retained but not applied. Tests now assert this continuous global-feature behavior. That directly contradicts the older reviews' strongest anti-leakage claim.
- The current README and `docs/RESEARCH.md` say "no cross-block leakage" while describing precomputed full-timeline feature slicing. That statement is too strong.
- Backtest now defaults to `--checkpoint both`, while `docs/RESEARCH.md` says published OOS metrics use eval-NAV-best only. Several docs commands omit `--checkpoint best`, which quietly evaluates latest on OOS too and disables single-run plotting in the `both` path.

### Comparison to `docs/grok-review-20260605.md`

An untracked `docs/grok-review-20260605.md` appeared during this review. It agrees with several current-state conclusions:

- dynamic universe support is real and materially improved
- eval segment mechanics are better than in the May 31 review
- inference loading and VecNormalize freezing are now real in-tree code
- block bootstrap and stochastic OOS paths are meaningful improvements
- early stopping, reward calibration, docs drift, and auto-research guardrails remain important

I disagree with, or would qualify, several Grok-review claims:

- It says the core invariant "per-segment feature computation with join purge" is intact. Current code contradicts that: `train_test_split_alternating()` slices precomputed full-timeline features and says `feature_purge_warmup` is not applied. This is the largest difference between the reviews.
- It says reward shaping remains "badly imbalanced" in nearly the same way as the older review. I agree reward instrumentation and scaling still need work, but the current coefficients are less inert than the earliest critique: participation is +1 at full exposure, ordinary churn is subtle, and inactivity can be large. The right answer is empirical reward-decomposition logging, not a purely static verdict.
- It reports bare test collection failures. I did not run pytest in this review, so I do not rely on that claim.
- It treats run artifacts as a strong reproducibility contract. I agree on the layout, but current backtest does not load the run-local config or run-local data snapshot by default, so the contract is incomplete.

## Current Design in Detail

### Configuration and Universe

`config/config.yaml` is parsed into frozen dataclasses in `rlbot/rl_config.py`. The tradeable universe is the ordered mapping under `universe.assets`; there is no separate asset-count knob. Supported `N` is 5-55.

The active observation size is:

```text
obs_dim = 10 * N + 28
```

For default `N = 10`, this is 128, not the stale 118 mentioned in older agent instructions.

`scripts/train.py --n-assets N` slices the first `N` YAML keys, slices transaction-cost vectors, and renormalizes benchmark cap weights. That is useful for quick scale experiments, but it means key order is semantically important.

Each run writes a `Runs/<run_id>/config.yaml` snapshot and a `manifest.json` with universe metadata. This is the right direction, but backtest does not yet actually load that config snapshot.

### Data Pipeline

`fetch_aligned_daily()` pulls tradeable assets from yfinance, macro context from yfinance, and HY OAS from FRED plus a HYG/IEF proxy. It outer-joins calendars, forward-fills short holiday gaps, records `asset_live`, and keeps pre-listing rows masked.

Current behavior keeps the full calendar and masks pre-live assets. Some docstrings still say pre-listing rows are dropped or that the panel starts when all assets have real quotes; those are stale.

The chronological OOS holdout is reserved by `reserve_chronological_holdout()` before the alternating train/eval split. That remains the most important leakage firewall in the repo.

### Feature Pipeline and Walk-Forward Split

The intended invariant in older docs was:

```text
raw trainable data -> alternating train/eval blocks -> per-segment features -> purge at joins
```

The current implementation is:

```text
cache/full trainable feature panel -> align to trainable dates -> slice features into train/eval blocks
```

Specifically:

- `scripts/train.py` aligns cached `rsi`, `macd`, `fracdiff`, `fracdiff_macro`, `trend`, `asset_vol`, and `macro_vol` to the trainable timeline, then passes them into `train_test_split_alternating()`.
- `train_test_split_alternating()` accepts the precomputed arrays, slices them by block, and says `feature_purge_warmup` is not applied.
- `block_boundaries` still prevent episodes from crossing non-contiguous joined ranges, so the env will not roll directly across stitched gaps.

This design matches continuous backtest memory, but it contaminates the in-training train/eval split: train blocks after eval blocks can contain RSI/MACD/fracdiff/trend state influenced by eval block prices, and eval blocks can contain state influenced by immediately prior train blocks. That makes in-training eval less independent as a checkpoint-selection signal.

This is not the same as leaking chronological OOS holdout data into training. OOS is still stripped first. But it does mean the phrase "no cross-block leakage" is not accurate for the alternating train/eval split.

### Environment

`MultiAssetPortfolioEnv` is dynamic in `n_assets`. The action space is `Box(-3, 3)^(N+1)` for cash plus risky sleeves.

Observation components are:

- 4 horizons of per-asset fracdiff plus cross-asset mean
- per-asset realized vol plus mean
- per-asset RSI
- per-asset MACD
- per-asset trend
- 4 horizons of macro fracdiff for 4 macro series
- macro realized vol
- per-asset live mask
- current portfolio weights, cash plus assets
- drawdown and episode progress

The action path is:

```text
raw logits -> optional EMA smoothing -> softmax -> zero non-live assets -> cap/redistribute risky legs -> long-only simplex
```

The current max risky asset cap is `0.35`.

Execution is causal:

```text
observe market features through close[t - obs_lag]
decide
execute at open[t + 1]
mark to close[t + 1]
```

The env deducts holding cost on pre-rebalance units at `close[t]`, rebalances at `open[t+1]`, and computes NAV at `close[t+1]`.

Reward components are:

- clipped log return times `reward_scale`
- benchmark-relative Sortino differential
- inactivity penalty based on cash fraction
- participation bonus
- VIX-scaled churn penalty
- quadratic drawdown penalty

Per-step `info` exposes `rew_decomp/*`. I did not find a dedicated callback that aggregates those terms to TensorBoard or a machine-readable reward-decomposition file, so the instrumentation exists but is underused.

### Training

`scripts/train.py` builds a RecurrentPPO `MlpLstmPolicy` with a 2-layer 64-unit LSTM and MLP heads from config. Defaults are 16 local envs, `n_steps=4096`, `batch_size=16384`, `n_epochs=3`, and 65M timesteps.

The training stack includes:

- `SubprocVecEnv`
- `VecNormalize` with reward normalization during training
- fee curriculum
- churn ramp
- progressive domain randomization for fee scale and observation lag
- mandatory entropy decay
- periodic checkpointing
- periodic deterministic eval
- best-by-mean-ending-NAV checkpointing
- training plots

Current eval is much better than the older review described: eval uses one deterministic full rollout per eval segment, and `eval_n_episodes` is only a fallback if segments are unavailable.

There is still no early stopping. The log explicitly prints `early_stop: off`.

### Backtest

`scripts/backtest.py` requires a run manifest for OOS dates. It loads model weights, locates VecNormalize stats, freezes VecNormalize for inference, runs a deterministic OOS rollout, and can optionally run stochastic policy paths.

Strengths:

- Manifest-derived holdout dates reduce manual date mismatch.
- VecNormalize is frozen for inference.
- `--checkpoint best|latest|both` supports comparing selected vs latest weights.
- Detailed stats include passive baselines, subperiod returns, stationary block bootstrap Sharpe intervals, and stochastic path summaries.
- Batch backtests reuse imports, cache, and policy shell for speed.

Current weaknesses:

- The parser default is `--checkpoint both`, not `best`.
- Backtest uses current global config for costs, cap, and env mechanics.
- Backtest uses current global `.cache/data_cache.npz`, not `Runs/<run_id>/data_cache.npz`.
- It verifies manifest ticker/order/obs-dim compatibility, but that is a guardrail, not full reproducibility.
- Dynamic universes below the default may break the 60/40 benchmark because `balanced_6040_nav()` requires `BOND10Y`.

### Artifacts and Modal

The consolidated `Runs/<run_id>/` layout is a strong improvement:

```text
Runs/<id>/
  manifest.json
  config.yaml
  data_cache.npz
  models/
  plots/
  logs/
  tb_logs/
  eval_logs/
```

`scripts/modal_app.py` and `rlbot/modal_cloud.py` provide a practical Modal training path with GPU-specific `n_envs` and batch sizing, live plot sync, volume commits, full artifact pull, cache upload, and status endpoints. This is a good substrate for automated experiment orchestration.

## Issues and Weaknesses

### P0 - Walk-Forward Feature-State Contamination

Current train/eval split does not recompute features per train/eval segment and does not apply join purge. It slices precomputed full-timeline features. This allows indicator memory from eval blocks to appear inside later train blocks and vice versa.

Why it matters:

- It weakens the independence of the eval NAV used for `best_model.zip`.
- It conflicts with the strongest claim in the prior assessments.
- It can make auto research overfit the validation surface faster.
- It creates doc/code confusion because some docs still say "no cross-block leakage."

Recommended fix:

- Add an explicit split feature mode:
  - `continuous_features`: current behavior, for matching continuous production/backtest memory.
  - `independent_split_features`: recompute features inside each contiguous train/eval block or segment, then apply purge.
- Use `independent_split_features` for checkpoint-selection eval until proven otherwise.
- Keep current continuous behavior available as an ablation.
- Add tests for both modes and make the mode visible in `manifest.json`.

### P0 - Backtest Does Not Load Run Config Snapshot

The run snapshot exists, but `scripts/backtest.py` never calls `load_config(Runs/<id>/config.yaml)` / `set_config(...)`. Current global `config/config.yaml` controls:

- transaction costs through `rlbot/baselines.py`
- max single asset cap through `portfolio_weights_from_action()`
- env cost arrays inside `MultiAssetPortfolioEnv`
- observation dimension formula and other env defaults

Why it matters:

- OOS metrics for an old run can change after editing config.
- Comparing cohorts can become accidental config drift.
- Auto research would produce unreliable leaderboards unless each result is tied to the exact config used for inference.

Recommended fix:

- By default, backtest should load `Runs/<run_id>/config.yaml` before building envs or baselines.
- Add `--use-current-config` only for deliberate stress tests.
- Record the effective backtest config path and SHA/hash in the backtest summary.

### P0 - Backtest Does Not Prefer Run-Local Data Snapshot

`train.py` writes `Runs/<run_id>/data_cache.npz`, but `backtest.py` uses the process-global `.cache/data_cache.npz`. If the current cache has a different universe or horizon, backtest can fail or silently depend on external state.

Why it matters:

- The run directory looks self-contained but is not actually the default source of truth for backtest data.
- Dynamic-universe experiments become brittle.
- Auto research needs old runs to remain evaluable after new data/cache refreshes.

Recommended fix:

- Backtest should prefer `Runs/<run_id>/data_cache.npz` when present.
- Add `--data-cache PATH` for explicit override.
- Store a cache content hash in the manifest and in backtest summaries.
- Consider writing a clipped run-local panel, not merely copying the current global cache.

### P0 - Docs and Instructions Drift

Examples of current drift:

- `AGENTS.md` / `CLAUDE.md` describe 11 actions, 10 assets, 118-d observations, 0.50 cap, `runs/`, and `windows/*.sh`; the current code uses dynamic `N`, 128 dims for N=10, 0.35 cap, `Runs/`, and no `windows/` directory.
- `docs/TRAINING.md` says walk-forward feature panels are per-block, but current code slices cached global features.
- `docs/RESEARCH.md` says full-trainable feature slicing has no cross-block leakage. That is not accurate for indicator state.
- `docs/RESEARCH.md` and `docs/TRAINING.md` include commands that omit `--checkpoint best` despite saying published metrics use best only.
- `rlbot/data_utils.py` module docstring says pre-listing rows are dropped and the panel starts when all configured assets have real quotes. Current code keeps rows and uses `asset_live`.
- `.gitignore` says keep `execution/README.md` tracked, but there is no `execution/` directory in this checkout.

Recommended fix:

- Make README/RESEARCH/TRAINING/AGENTS agree with current code.
- Add a short "Current invariants" section generated or checked by tests where possible.
- Treat docs drift as a testable surface for this repo because misuse can invalidate experiments.

### P1 - Checkpoint Policy Is Not Enforced by Defaults

The research docs say eval-NAV-best only, but `scripts/backtest.py` defaults to `--checkpoint both`. In single-run mode, `both` forces the batch path and sets `no_viz=True`, so commands with `--plot-tag best` may not do what the docs imply.

Recommended fix:

- Change default checkpoint to `best`.
- Keep `both` for explicit diagnostics only.
- Add a printed warning when `latest` or `both` touches OOS.
- Add `--summary-json` so diagnostic comparisons do not rely on copied terminal output.

### P1 - No Automated Early Stopping

Training explicitly runs the full budget. That may be defensible for fixed-budget research, but previous reports discuss validation NAV peaking before the end.

Recommended fix:

- Add optional patience-based early stopping after curriculum release.
- Make it default for exploratory runs, off only for pre-registered full-budget runs.
- Save the early-stop reason and best step in manifest.

### P1 - Determinism Is Partial and Potentially Misleading

`apply_deterministic_seeds()` seeds Python, NumPy, and Torch, and sets deterministic Torch flags. But training envs use `reseed_on_reset=True`, which replaces the env RNG with fresh OS entropy each episode.

Recommended fix:

- Rename the behavior in docs as "deterministic framework plus stochastic episode reseeding."
- Add `--reproducible-env-resets` or `--seed-stream deterministic`.
- Use deterministic per-env RNG streams when debugging or running ablations.
- Add a lock file or pinned container image for long studies.

### P1 - Dynamic Universe Support Has Baseline Edge Cases

The core env is dynamic, but some baseline assumptions still target the default universe. The 60/40 benchmark requires `BOND10Y`. With `--n-assets 5`, `BOND10Y` is not among the first 5 default assets, so detailed backtests or plots can fail.

Recommended fix:

- Make benchmark availability conditional.
- If `BOND10Y` is missing, skip 60/40 with a clear note.
- Add tests for backtest plotting/detailed stats with `N=5`.

### P1 - Reward Instrumentation Is Not Yet Research-Grade

The current reward coefficients are less obviously inert than the older review claimed, but the repo still needs empirical per-term distribution reports.

Approximate current magnitudes:

- 1% daily log return -> about +20 reward units.
- Sortino differential can reach +/-75.
- Full exposure participation -> +1.
- 10% turnover churn at full scale -> about -0.64 to -1.28 depending on VIX.
- 100% turnover churn -> about -6.4 to -12.8.
- 100% cash inactivity -> -25.
- 5% drawdown penalty -> about -0.75; 20% drawdown -> about -12.

This is not automatically wrong, but it means participation and ordinary churn are still subtle relative to return/Sortino. An auto researcher needs reward-decomposition metrics to avoid optimizing blind.

Recommended fix:

- Add a callback that logs per-term means, percentiles, and shares of absolute reward to TensorBoard and JSON.
- Include exposure, turnover, cash fraction, drawdown, and cap-hit frequency in every backtest summary.
- Run reward ablations before adding assets or model complexity.

### P1 - Research Results Are Still Too Manually Curated

`docs/RESEARCH.md` is honest but manual. It mixes completed, pending, legacy, current-cohort, cross-window, and command examples. This does not scale to an automated research loop.

Recommended fix:

- Store one machine-readable result record per backtest.
- Maintain a cohort-level `research_registry.jsonl`.
- Generate the human `RESEARCH.md` tables from that registry.

### P2 - Action Cap Is Tested, But Still Best-Effort

`portfolio_weights_from_action()` runs five clip/redistribute iterations, then normalizes. Current tests fuzz random actions and extreme logits under the 0.35 cap. That is good.

Remaining concern:

- The function has no final assert/projection that proves the cap post-condition for arbitrary caps and `N`.

Recommended fix:

- Add a final guarded projection or assertion.
- Expand fuzz tests across different caps and asset counts.

### P2 - External Data Limitations Remain

The repo correctly disclaims yfinance/FRED limitations. The current pipeline is reasonable for research, but not institutional point-in-time data.

Recommended fix:

- Persist data-source metadata in manifests.
- Persist HY OAS calibration metadata or coefficients.
- Add cache hashes.
- For serious research claims, graduate to point-in-time data with survivorship and corporate-action metadata.

### P2 - Missing Live/Paper Inference Path

There are inference load helpers, but no single audited path that says:

```text
given run_id and today's panel -> freeze VecNormalize -> warm recurrent state -> emit target weights + provenance
```

Recommended fix:

- Add `scripts/infer_weights.py`.
- Inputs: `--run-id`, `--checkpoint best`, `--as-of`, optional `--data-cache`.
- Outputs: JSON/CSV target weights, action logits, live mask, config hash, cache hash, model path, VecNormalize path.
- Keep broker adapters separate until this core path is tested.

## Feasibility of Applying an Auto Research Pattern

### Definition

By "auto research pattern," I mean a closed-loop experiment system:

```text
hypothesis -> experiment spec -> config/run materialization -> train -> eval/backtest -> metrics ingestion -> comparison/gating -> next hypothesis
```

This can be human-directed or agent-assisted. The important point is that the system pre-registers what it is testing, runs controlled comparisons, records results in machine-readable form, and prevents holdout leakage through the research process itself.

### Feasibility: High, After Guardrails

RLBot already has many of the pieces:

- Config-driven experiments.
- Per-run config snapshots.
- Run ids and manifests.
- Modal training backend.
- Batch backtest.
- Seed ensemble script.
- Best-by-eval-NAV checkpoint.
- OOS holdout date capture.
- Fast inference loading.
- Block bootstrap and stochastic-path diagnostics.

What is missing is the orchestration and firewall layer.

I would rate readiness as:

```text
Experiment substrate: 7/10
Reproducibility substrate: 5/10
Automated ranking safety: 4/10
Auto research readiness today: 5/10
Auto research readiness after P0 fixes: 8/10
```

### Expected Benefits

An auto research layer would be valuable because this project has many interacting knobs:

- feature split mode
- reward coefficients
- participation/churn/drawdown weighting
- inactivity scaling
- curriculum timing
- entropy schedule
- domain randomization bounds
- LSTM size
- asset universe size/order
- eval block size/stride
- checkpoint rule
- seed robustness

Manual iteration over these is slow and easy to bias. A structured loop would:

- make ablations cheaper and more honest
- prevent accidental config drift
- quantify seed variance
- surface failure modes faster
- turn `RESEARCH.md` from a hand-maintained notebook into a generated report
- reduce cloud waste by stopping weak candidates early
- keep promising changes tied to hypotheses rather than vibes

### Primary Risk

Auto research will amplify whatever objective and leakage structure it is given. If the current train/eval feature-state contamination remains, an automated loop may become very good at exploiting that validation surface. If OOS backtests are visible to the loop too early, it can overfit calendar windows through repeated attempts even without code-level leakage.

Therefore, auto research should start as "automated experiment bookkeeping and gated execution," not as "agent freely optimizes OOS metrics."

## Recommended Auto Research Design

### 1. Experiment Spec

Add a small spec format, initially YAML or JSON:

```yaml
id: reward_churn_grid_001
hypothesis: Higher churn penalty reduces turnover without hurting eval NAV.
parent: baseline_20260605
owner: human
status: proposed
base_config: config/config.yaml
patch:
  reward.churn_penalty: [4.0, 8.5, 15.0]
windows: [1, 2]
seeds: [42, 101, 777]
timesteps: 3000000
checkpoint_rule: eval_nav_best
evaluation_tier: dev
success_gates:
  max_turnover_p50: 0.15
  eval_nav_mean_min: 100000
  oos_access: false
budget:
  max_modal_hours: 6
```

This should materialize resolved configs under something like:

```text
Runs/<cohort>/spec.yaml
Runs/<cohort>/configs/<variant>.yaml
Runs/<cohort>/registry.jsonl
```

### 2. Run Registry

Every train/backtest should emit a JSON record. Minimum fields:

- cohort id
- variant id
- hypothesis id
- run id
- git commit
- dirty worktree flag
- config path/hash
- data cache path/hash
- feature split mode
- universe tickers
- train/eval/OOS dates
- seed
- training budget
- checkpoint path
- VecNormalize path
- best eval NAV and step
- final eval NAV
- OOS metrics if permitted
- benchmark metrics
- turnover/exposure/drawdown metrics
- reward decomposition summary
- status and failure reason

JSONL is enough at first. DuckDB or SQLite can come later.

### 3. Evaluation Tiers

Use gated tiers to avoid wasting compute and overusing OOS:

```text
Tier 0: static tests and split-leakage checks
Tier 1: smoke train, tiny budget, no OOS
Tier 2: short dev train on in-training eval only
Tier 3: medium train across multiple seeds/windows, still no final OOS
Tier 4: pre-registered full train, eval-NAV-best checkpoint, OOS read once
Tier 5: paper/shadow trading
```

Only Tier 4 should touch published OOS windows. The auto researcher can see Tier 1-3 results freely, but Tier 4 promotion should require a fixed spec and explicit human approval.

### 4. Holdout Firewall

Add policy to the code and docs:

- In-training eval is for model selection.
- Dev OOS can be used for method development only if labeled as such.
- Final OOS windows should be touched once per registered candidate.
- Do not let an agent iterate against final OOS metrics.

This is procedural as much as technical, but the tool can enforce it by default.

### 5. Orchestrator

Add `scripts/research.py` or `rlbot/research/` with subcommands:

```bash
python scripts/research.py plan specs/reward_churn_grid.yaml
python scripts/research.py launch specs/reward_churn_grid.yaml --backend modal
python scripts/research.py collect reward_churn_grid_001
python scripts/research.py report reward_churn_grid_001
python scripts/research.py promote reward_churn_grid_001 --variant churn_15 --tier 4
```

The first version can shell out to existing `scripts/train.py`, `scripts/backtest.py`, and `scripts/modal_app.py`. It does not need to rewrite the training stack.

### 6. Metrics Summary Output

Add `--summary-json PATH` to `scripts/backtest.py`. Add a training summary JSON on train exit. The orchestrator should not parse terminal text.

### 7. Safety and Cost Controls

The auto research runner should include:

- max concurrent jobs
- max total Modal hours
- max timesteps per tier
- explicit OOS access flag
- no automatic deletion of runs
- no automatic code changes during a registered experiment
- failure capture and resume behavior

### 8. Best Initial Research Questions

Start with questions that test harness assumptions, not alpha dreams:

1. Independent split features vs continuous split features.
2. Best checkpoint vs early stopping patience.
3. Reward decomposition ablation: return-only, return+Sortino, full reward.
4. Churn/participation scale grid.
5. Eval stride/block-size sensitivity.
6. Deterministic seed-stream vs OS-entropy reset variance.
7. RecurrentPPO vs non-recurrent PPO baseline on the same env.
8. Default 10 assets vs smaller/larger dynamic universes, after baseline fixes.

## Continued Evolution Roadmap

### Phase 0: Seal the Research Harness

Do these before trusting automated search:

1. Decide and implement explicit feature split modes.
2. Make backtest load run-local config by default.
3. Make backtest load run-local data snapshot by default.
4. Change default checkpoint to `best`.
5. Fix docs and agent instructions to match current code.
6. Add summary JSON outputs.

### Phase 1: Measurement Hardening

1. Add reward-decomposition aggregation.
2. Add exposure/turnover/cap-hit metrics.
3. Add baseline skipping for missing assets.
4. Add split-leakage regression tests.
5. Add deterministic env reset mode.
6. Add dependency lock or pinned Docker/Modal image.

### Phase 2: Auto Research MVP

1. Add experiment spec format.
2. Add local JSONL registry.
3. Add train/backtest collection commands.
4. Add tiered gates.
5. Generate cohort reports.
6. Keep OOS access human-gated.

### Phase 3: Method Development

Only after the harness is sound:

1. Run reward and curriculum ablations.
2. Re-test the validation NAV peak/cliff under independent split features.
3. Compare recurrent vs feed-forward policy baselines.
4. Compare feature sets and macro handling.
5. Scale universe size gradually.
6. Introduce better data sources if results survive.

### Phase 4: Deployment-Oriented Inference

Before live or paper claims:

1. Add audited target-weight inference CLI.
2. Add recurrent warmup rules.
3. Add provenance-rich output.
4. Add paper trading harness.
5. Add broker adapter only after the inference core is stable.

## Bottom Line

The repo is worth continuing. The useful asset here is not just a trained LSTM policy; it is the beginnings of a disciplined research operating system for portfolio RL.

The next best investment is not more assets or bigger GPUs. It is to make the experiment harness unambiguous:

```text
exact config
exact data snapshot
exact feature split semantics
exact checkpoint rule
exact metrics record
```

Once those are locked down, the auto research pattern is a very good fit. Without those locks, auto research will mostly automate overfitting and documentation drift.
