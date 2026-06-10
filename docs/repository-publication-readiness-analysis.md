# Repository Analysis and Publication Readiness Audit

Date: 2026-06-08 (updated 2026-06-09 for current benchmark-relative reward pipeline)

Scope: this audit reviewed the repository code, tests, `README.md`, `config/README.md`, `docs/MODAL.md`, `docs/RESEARCH.md`, and `docs/TRAINING.md`. It intentionally did not use prior model review documents in `docs/`.

> **Pipeline update (2026-06-09):** Default `feature_split_mode: independent`; cap **`max_single_asset_weight: 0.25`**; benchmark excess + Sortino (`risk_bonus_scale: 2.5`, `benchmark_excess_scale: 600`, combined cap **`benchmark_combined_abs_cap: 24.0`**); train **and eval** share the fee/churn curriculum; `models/best/` saves are gated until `fee_ramp_end`. Runs are comparable only after checking their snapshotted `Runs/<id>/config.yaml`.

## Executive Summary

MarketTrainer/RLBot is a serious research codebase rather than a toy trading environment. Its strongest properties are the config-driven environment, explicit chronological OOS separation, run-local config/cache snapshots, dynamic universe sizing, asset live masks, frozen inference normalization, and a growing test suite around the most important invariants. The repository is already much closer to a paper-supporting experimental platform than most reinforcement-learning trading projects.

It is not publication ready yet, but the original validity blockers from this audit have mostly been fixed. Current blockers are now methodological rather than basic plumbing:

- no definitive, pre-registered OOS result table under the current config;
- no full seed-cohort distribution across all walk-forward windows;
- no paper-grade ablation table for benchmark excess, Sortino, feature split, action cap, transaction costs, and recurrent-vs-feed-forward policy;
- no dependency lockfile or pinned container image for exact long-run reproduction;
- no point-in-time institutional data or capacity/market-impact model.

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
- Feature split modes are explicit: **`independent`** (default — per-segment recompute + warmup purge) and `continuous` (contiguous-memory eval / backtest ablation).
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

## Publication-Critical Implementation Status

### 1. Best Model Uses Matched VecNormalize Statistics

**Status:** fixed. `EvalNavBestModelCallback` saves `best/vec_normalize.pkl` alongside `best_model.zip` whenever eval NAV improves after the best-model gate opens (`fee_ramp_end` by default). Backtest requires the matched pair by default.

Historical note: the original audit described mismatched VecNormalize pairing at final save time; that path now saves matched pairs at each best-eval step.

### 2. Target Weight Inference Uses Executed Smoothed Weights

**Status:** fixed. `MultiAssetPortfolioEnv.step()` emits `info["target_weights"]` after EMA smoothing, live masking, softmax, and cap projection. Backtest/inference collect those executed weights rather than mapping raw policy logits.

Regression coverage: `tests/test_publication_fixes.py` checks that reported weights differ from raw-logit mapping when smoothing is enabled and that backtest source collects `info.get("target_weights")`.

### 3. `--n-assets` Snapshots The Effective Panel

**Status:** fixed for the current pipeline. Training writes the effective selected **N**-wide panel to `Runs/<run_id>/data_cache.npz`; backtest binds that run-local cache by default and checks manifest/cache compatibility.

Operational caveat: after changing `--n-assets` or editing `universe.assets`, still run `--refresh-data` for a clean global cache and a new run id. Run-local snapshots protect backtests, but stale global caches make experimentation harder to reason about.

### 4. Crash Resume Is Separate From Fine-Tune

**Status:** fixed. `--resume` and `--finetune` are mutually exclusive. Crash resume restores weights + VecNormalize and continues curriculum/adaptive-entropy behavior from the checkpoint timestep; fine-tune is the explicit experimental mode with lower LR/entropy/clip and skipped curriculum callbacks.

### 5. VecNormalize Required For Publication Backtests

**Status:** fixed. Run-id OOS backtests require VecNormalize stats by default. `--allow-missing-vec-normalize` is an explicit debug escape hatch.

### 6. Manifest Holdout Metadata Is Preserved

**Status:** fixed. Final manifest writes merge with existing metadata, preserving `chronological_holdout` and run provenance.

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
- Report sensitivity to `continuous` versus `independent` feature split mode (default is now **`independent`**).

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

### Test Coverage Is Good But Still Needs Long-Run Validation

Strengths:

- action simplex and cap behavior
- dynamic observation/action dimensions
- feature split behavior
- artifact resolver behavior
- target-weight payload validation
- best-model VecNormalize pairing
- executed smoothed target weights
- effective run-local data snapshots
- manifest merge/preservation
- resume vs fine-tune argument separation
- doc invariant checks
- block bootstrap tests

Gaps:

- no full-budget train/backtest/infer E2E in a pinned environment
- no dependency lockfile or pinned Docker/Modal image digest for long studies
- no automated test that a resumed long run exactly matches uninterrupted schedule state beyond the CLI/callback-path coverage
- limited tests for data-cache migration and manifest finalization

## Publication Readiness Plan

### Phase 1: Freeze The Experimental Protocol

1. Freeze a paper config and record its hash.
2. Freeze the asset universe, dates, transaction costs, observation layout, reward formula, and checkpoint rule.
3. Define walk-forward windows before training.
4. Define how many seeds per window will be run.
5. Define exactly which ablations are allowed before OOS reads.
6. Decide whether the paper uses `independent` (current default) or `continuous` feature split mode as the primary result; keep the other as an ablation.

Exit criterion: a pre-registration document exists in `docs/` or `specs/`, with no result-dependent edits.

### Phase 2: Reproducibility Hardening

1. Add a dependency lockfile or pinned container image for long studies.
2. Run a tiny train/backtest/infer smoke in a clean installed environment.
3. Document CPU/GPU, Python, torch, SB3, and data-source versions for every reported cohort.
4. Keep `training.reproducible: false` for diversity studies unless exact replay is required; use `true` for deterministic debugging cohorts.

Exit criterion: another machine can reproduce a small run and verify artifact hashes/metadata.

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

Current local verification under the project `.venv`:

```text
110 passed
```

This is necessary but not sufficient for publication. A fresh full-budget walk-forward cohort and the ablations above still need to be run under the frozen protocol.

Next verification step: run at least one tiny train → backtest → infer cycle in a clean installed environment or pinned container, then proceed to full-budget walk-forward cohorts.

## Bottom Line

The repository has a strong skeleton: careful OOS intent, a coherent environment, good artifact discipline, useful docs, and many of the right tests. The main risk is that a few implementation details currently undermine the exact reproducibility and inference claims the project wants to make. Fix those first, then run a locked, pre-registered walk-forward study. After that, the project can support a credible environment-focused paper.
