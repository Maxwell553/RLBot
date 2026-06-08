# Repository Analysis and Publication Readiness Audit

Date: 2026-06-08

Scope: this audit reviewed the repository code, tests, `README.md`, `config/README.md`, `docs/MODAL.md`, `docs/RESEARCH.md`, and `docs/TRAINING.md`. It intentionally did not use prior model review documents in `docs/`.

## Executive Summary

MarketTrainer/RLBot is a serious research codebase rather than a toy trading environment. Its strongest properties are the config-driven environment, explicit chronological OOS separation, run-local config/cache snapshots, dynamic universe sizing, asset live masks, frozen inference normalization, and a growing test suite around the most important invariants. The repository is already much closer to a paper-supporting experimental platform than most reinforcement-learning trading projects.

It is not publication ready yet. The most important blockers are not just missing OOS results. There are several validity and reproducibility issues that should be fixed before any results are trusted:

- The eval-selected `best_model.zip` is paired with end-of-training `VecNormalize` statistics rather than the observation statistics from the best-eval step.
- Backtest and inference weight collection ignore action smoothing, so plotted weights and `infer_weights.py` target weights can differ from the actual executed policy.
- `--n-assets` can train on a sliced universe while snapshotting an unsliced cache, causing backtest incompatibility unless data was refreshed under the sliced config.
- Resume behavior is closer to fine-tuning than crash-resume because curriculum and adaptive entropy callbacks are skipped when `--resume` is set.
- Verification could not be completed in the current shell because `gymnasium` is not installed.

The environment-design paper is plausible, but it should be framed around the environment, leakage controls, reproducibility protocol, and benchmarked empirical behavior only after a fresh, pre-registered run set is completed under the current pipeline.

## Major Strengths

### Clear Research Contract

The repository has a coherent methodological story:

- `README.md` defines the data pipeline, observation layout, action mapping, reward, execution timing, walk-forward windows, and artifact layout.
- `docs/RESEARCH.md` explicitly states that no current OOS results are published and keeps result tables as pending placeholders.
- `docs/TRAINING.md` and `docs/MODAL.md` describe local and cloud operation with the same `Runs/<run_id>/` artifact model.
- `AGENTS.md` and `CLAUDE.md` encode the most important invariants and have a tripwire test in `tests/test_docs_invariants.py`.

That discipline matters for publication because it reduces the risk of retrofitting methodology after results are known.

### Strong Anti-Leakage Architecture

The core data design is thoughtful:

- `reserve_chronological_holdout()` removes OOS data before in-training train/eval splitting.
- Backtest requires a run manifest and defaults to the run-local config and cache snapshot.
- Feature split modes are explicit: `continuous` for contiguous-memory eval and `independent` for per-segment recomputation plus warmup neutralization.
- The trading environment executes at `open[t+1]` after observing lagged market features, which avoids a common close-to-close look-ahead shortcut.
- `VecNormalize` is frozen for inference through `rlbot.vecnorm_utils` and `rlbot.inference_load`.

The anti-leakage logic is also partially tested in `tests/test_feature_split_modes.py`, `tests/test_eval_segments.py`, and `tests/test_backtest_config_binding.py`.

### Reproducible Artifact Model

The centralized `Runs/<run_id>/` layout is a major asset. A run stores:

- `manifest.json`
- `config.yaml`
- `data_cache.npz`
- models, checkpoints, logs, TensorBoard, plots, eval logs
- post-backtest summaries with config/cache hashes

This is the right foundation for paper artifacts and independent audit. The code also records git provenance through `rlbot/run_artifacts.py`, which is important for result traceability.

### Dynamic Universe and Live Masking

The environment supports 5 to 55 assets with computed observation/action dimensions. It avoids hard-coded 10-asset assumptions in most important paths, and tests exercise multiple synthetic universe sizes.

The `asset_live` mask is also a strong idea. It lets the panel keep earlier calendar history without letting the agent allocate to assets before listing.

### Useful Research Tooling

The auto-research loop has the right conceptual shape:

- pre-registered specs in `specs/`
- allow-list-restricted config patches
- registry JSONL
- OOS promotion gate
- report generation

This gives the project a path toward disciplined ablations and reduces the temptation to repeatedly read the final holdout.

### Measurement Beyond Mean Return

The backtest script includes:

- passive benchmarks
- max drawdown, CAGR, Calmar
- subperiod stats
- stationary block-bootstrap Sharpe intervals
- optional stochastic policy path fan

Reward decomposition is also logged during training, which will be valuable for diagnosing whether reward shaping dominates real returns.

## Critical Issues To Fix Before Trusting Results

### 1. Best Model Uses Mismatched VecNormalize Statistics

`EvalNavBestModelCallback` saves only the model weights when eval NAV improves. At training exit, `_persist_trade_artifacts()` saves the final `VecNormalize` state and copies that final state next to `best_model.zip`.

Relevant code:

- `scripts/train.py`: eval callback saves best weights around lines 1185-1197.
- `scripts/train.py`: final `VecNormalize` copied next to best around lines 1257-1262.

Why this matters: the selected best weights may come from much earlier than the final training step. Pairing those weights with end-of-run observation normalization can change the policy input distribution at inference. This breaks the meaning of "eval-NAV-best" and can contaminate OOS metrics.

Recommended fix:

- When a new best eval NAV is found, save both `best_model.zip` and a synchronized snapshot of the training `VecNormalize` stats at that exact step.
- In backtest, require that `best_model.zip` and `best/vec_normalize.pkl` are a matched pair.
- Add a test that a best-eval event writes both artifacts.

### 2. Target Weight Inference Ignores Action Smoothing

`rollout_policy_on_slice()` records weights by applying `portfolio_weights_from_action()` to the raw policy action before calling `raw_env.step(action)`. But `MultiAssetPortfolioEnv.step()` first applies EMA smoothing to the raw action and only then maps the smoothed logits to weights.

Relevant code:

- `scripts/backtest.py`: collected weights from raw action around lines 979-987.
- `rlbot/trading_env.py`: smoothing occurs inside `step()` before weight construction.

Why this matters: plotted weights and `scripts/infer_weights.py` target weights can be different from the actual weights traded by the environment. This is especially serious because `infer_weights.py` is presented as audited target-weight output.

Recommended fix:

- Add executed target weights to `info`, for example `info["target_weights"] = w.copy()`.
- In `rollout_policy_on_slice()`, collect `info["target_weights"]` after `step()`, not the raw pre-step mapping.
- Add a regression test with `action_smoothing_alpha > 0` showing collected weights match executed weights.

### 3. `--n-assets` Can Snapshot The Wrong Cache

Training can slice a larger cached panel down to the first N configured assets, but `snapshot_data_cache(data_cache, paths.data_snapshot)` copies the original cache file rather than the selected/sliced arrays.

Relevant code:

- `scripts/train.py`: cache snapshot happens at line 789 after possible in-memory slicing.
- `scripts/backtest.py`: backtest loads the run snapshot and checks `ohlcv.shape[1]` around lines 468-485.

Failure mode: if a 10-asset cache exists and training is launched with `--n-assets 5` without `--refresh-data`, training can proceed on a 5-asset in-memory panel, but the run-local `data_cache.npz` remains 10 assets. Backtest then sees manifest `n_assets=5` and cache width 10, causing incompatibility.

Recommended fix:

- Save the effective selected panel as the run-local cache snapshot, not a byte copy of the source cache.
- Alternatively, after loading a cache with a superset universe, backtest should select manifest tickers from cache tickers before compatibility checks.
- Add a test for `--n-assets` with an existing superset cache.

### 4. Crash Resume Skips Curriculum and Entropy Scheduling

When `--resume` is set, training loads a checkpoint and changes LR, entropy, and clip range for fine-tuning. It also skips `TradingCurriculumCallback` and `AdaptiveEntropyCallback`.

Relevant code:

- `scripts/train.py`: resume branch around lines 1100-1125.
- `scripts/train.py`: curriculum and adaptive entropy are only added when `not args.resume`, lines 1211-1231.

Why this matters: `docs/MODAL.md` describes resume as a way to continue interrupted training. The implementation behaves more like a different fine-tuning mode. A Modal preemption resume may therefore change the training distribution and schedule.

Recommended fix:

- Split the concepts into `--resume` and `--finetune`.
- For crash resume, restore timestep count, curriculum state, entropy schedule, and callback behavior.
- For fine-tuning, keep the current behavior but document it as a separate experimental regime.

### 5. Backtest Does Not Require VecNormalize By Default

Backtest will proceed without `VecNormalize` stats unless `--require-vec-normalize` is passed. Since training uses observation normalization by default, missing stats should be a hard error for publication-grade OOS metrics.

Recommended fix:

- Make VecNormalize required by default for run-id backtests.
- Provide an explicit escape hatch such as `--allow-missing-vec-normalize` only for debugging.

### 6. Manifest Is Overwritten Without Full Holdout Metadata

Training writes an initial manifest with `chronological_holdout` metadata before learning, then writes a final manifest after learning that omits that detailed holdout block. Backtest can still recover dates from `args`, but the manifest no longer contains all metadata advertised by the docs.

Recommended fix:

- Preserve the initial manifest fields when writing the final manifest.
- Include holdout bars, date start/end, trainable end, and the effective `until`.

## Other Weaknesses and Risks

### Data Semantics Need Publication-Level Tightening

The asset-live/pre-listing logic is sensible, but the implementation and comments drift in places:

- The top of `rlbot/data_utils.py` still says the panel starts when all configured assets have real quotes, but current code keeps earlier history with live masks.
- Pre-listing OHLC is back-filled with first listing prices after `asset_live` is computed.
- Macro leading values can become zero and are later clamped for logs.

These may be acceptable engineering choices, but the paper must define them precisely and justify them. The environment should also include tests that pre-listing features and live masks behave as intended across asset IPO boundaries.

### Evaluation Signal Has Limited Effective Sample Size

The eval-selected checkpoint is based on one deterministic rollout per eval segment. That is a clear rule, but the effective sample size is small and correlated. `continuous` feature mode also intentionally carries feature memory across adjacent train/eval blocks. This is acceptable as a model-selection signal only if reported honestly.

Recommended publication stance:

- Treat in-training eval as a checkpoint-selection metric, not an independent estimate.
- Publish final claims only from chronological OOS windows.
- Report sensitivity to `continuous` versus `independent` feature split mode.

### Reward Is Heavily Engineered

The reward includes clipped log return, drawdown downside amplification, Sortino differential, participation reward, inactivity penalty, and cost-linked churn. This is not a flaw by itself, but it creates interpretability burden.

For a paper, the repo needs:

- ablations for each major reward term
- reward-decomposition plots over training
- evidence that policy performance is not primarily reward hacking
- OOS results under passive and simple learned baselines

### Baselines Are Useful But Not Enough

Current baselines are SPY buy-and-hold, equal-weight, 60/40, and naive risk parity. For publication, add or compare against:

- transaction-cost-aware equal weight and monthly rebalanced equal weight
- volatility targeting
- no-trade/cash and benchmark-only policies
- simple momentum or trend-following allocation
- possibly a non-recurrent PPO or supervised allocation baseline

The environment paper does not need to claim state-of-the-art returns, but it does need to show that the environment produces meaningful and nontrivial behavior.

### Modal Integration Is Strong, But Research Backend Is Not Implemented

`docs/MODAL.md` and `scripts/modal_app.py` provide real cloud training. However, `scripts/research.py launch --backend modal` accepts the flag but still shells to local `scripts/train.py`.

Recommended fix:

- Implement Modal dispatch in `scripts/research.py`, or remove the backend flag until supported.

### Test Coverage Is Good But Uneven

Strengths:

- action simplex and cap behavior
- dynamic observation/action dimensions
- feature split behavior
- artifact resolver behavior
- target-weight payload validation
- doc invariant checks
- block bootstrap tests

Gaps:

- no end-to-end tiny train/backtest smoke test in an installed environment
- no test for best-model VecNormalize pairing
- no test for smoothed executed weights versus reported weights
- no test for `--n-assets` cache snapshot behavior
- no test for resume schedule continuity
- limited tests for data-cache migration and manifest finalization

## Publication Readiness Plan

### Phase 1: Fix Validity Blockers

1. Save matched `VecNormalize` stats whenever `best_model.zip` is saved.
2. Make inference/backtest collected weights use executed smoothed weights.
3. Fix run-local data snapshotting for sliced universes.
4. Separate crash resume from fine-tuning and keep curriculum continuity for true resume.
5. Require VecNormalize for publication backtests.
6. Preserve full holdout metadata in final manifests.

Exit criterion: tests cover all six issues, and a tiny train/backtest smoke run can complete in a clean environment.

### Phase 2: Lock The Experimental Protocol

1. Freeze a paper config and record its hash.
2. Freeze the asset universe, dates, transaction costs, observation layout, reward formula, and checkpoint rule.
3. Define walk-forward windows before training.
4. Define how many seeds per window will be run.
5. Define exactly which ablations are allowed before OOS reads.
6. Decide whether the paper uses `continuous` or `independent` feature split mode as default.

Exit criterion: a pre-registration document exists in `docs/` or `specs/`, with no result-dependent edits.

### Phase 3: Environment Validation

Before headline OOS results, validate the environment itself:

1. Unit-test and document causal timing with a synthetic price path.
2. Verify transaction costs and holding costs with hand-computable examples.
3. Verify live-mask behavior around asset listing dates.
4. Verify observation dimension and feature ordering for several N values.
5. Verify that OOS rollout never updates VecNormalize statistics.
6. Compare environment NAV accounting to `portfolio_step_nav()` on controlled weights.

Exit criterion: these tests pass and can be cited as environment validation.

### Phase 4: Run Main Experiments

Recommended minimum paper experiment:

- 6 walk-forward windows from `docs/RESEARCH.md`
- at least 3 seeds per window
- eval-NAV-best checkpoint only
- deterministic OOS rollout plus optional stochastic policy rollouts
- all metrics written to `backtest_summary.json`
- no repeated holdout reads for variant selection

Report:

- total return, CAGR, Sharpe, Sortino, max drawdown, Calmar
- turnover and transaction-cost burden
- benchmark-relative metrics
- bootstrap confidence intervals
- per-window and aggregate seed distributions
- subperiod behavior

### Phase 5: Ablations

Run ablations after fixing the blockers, preferably at dev tiers before OOS:

- feature split: `continuous` vs `independent`
- reward terms: no Sortino, no participation, no inactivity, no drawdown amp, no churn
- transaction costs: zero cost, configured cost, stressed cost
- policy memory: MLP PPO vs LSTM PPO
- observation lag: fixed 1 versus randomized 0/1/2 training
- action smoothing: off versus configured alpha
- universe size sensitivity: 5, 10, and larger N if data quality supports it

### Phase 6: Paper Framing

The strongest paper angle is not "this agent beats markets." A more defensible framing is:

> A reproducible, leakage-controlled, multi-asset daily portfolio RL environment with run-local provenance, live asset masking, realistic delayed execution, configurable transaction costs, frozen inference normalization, and walk-forward OOS evaluation.

Suggested paper sections:

1. Motivation: why portfolio RL benchmarks often leak or lack reproducibility.
2. Environment design: state, action, execution, costs, reward, live mask.
3. Data protocol: asset universe, macro features, feature causality, holdout design.
4. Training protocol: RecurrentPPO, VecNormalize, curriculum, checkpoint selection.
5. Validation: synthetic accounting tests and leakage checks.
6. Experiments: walk-forward OOS, baselines, seed robustness.
7. Ablations: reward, feature split, costs, memory, lag.
8. Limitations: daily bars, yfinance data quality, no capacity model, long-only simplex, engineered reward, short OOS windows.

## Current Verification Status

I attempted to run `pytest` in the current environment. Test collection failed because `gymnasium` is not installed:

```text
ModuleNotFoundError: No module named 'gymnasium'
```

No tests were executed successfully in this shell. Before publication work, run:

```bash
pip install -e ".[dev]"
pytest
```

Then run at least one tiny smoke training/backtest cycle after the validity blockers are fixed.

## Bottom Line

The repository has a strong skeleton: careful OOS intent, a coherent environment, good artifact discipline, useful docs, and many of the right tests. The main risk is that a few implementation details currently undermine the exact reproducibility and inference claims the project wants to make. Fix those first, then run a locked, pre-registered walk-forward study. After that, the project can support a credible environment-focused paper.
