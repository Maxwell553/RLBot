# Grok Review: RLBot / MarketTrainer — 2026-06-05

> **Status (2026-06-09):** Historical snapshot. Agent docs refreshed; `infer_weights.py` + auto-research loop shipped; `paper_trade/` removed. **Current pipeline:** `independent` features (default), benchmark excess + Sortino cap, cap **0.25**, aligned train/eval fee curriculum, post-`fee_ramp_end` best-model gate. See [README.md](../README.md) and [evolution-roadmap-progress-20260606.md](evolution-roadmap-progress-20260606.md).

**Reviewer:** Grok 4.3 (xAI)  
**Date:** 2026-06-05  
**Scope:** Full repository (code, config, data pipeline, training/eval harness, artifacts, docs, tests, Modal integration, prior self-assessments). Grounded in source + execution of key paths (imports, config load, test collection, logic inspection) rather than self-description.  
**Purpose:** Independent technical + methodological assessment; explicit comparison to the two prior assessments in `docs/`; feasibility/benefits/risks of applying an "auto research" pattern; detailed design notes; concrete issues/weaknesses; forward evolution advice. Output is this durable artifact.

---

## 1. Executive Summary

RLBot remains a **genuinely high-quality research prototype** for recurrent PPO on a daily multi-asset portfolio allocation task with unusually strong engineering hygiene around temporal leakage prevention, causal execution, reproducibility, and honest empirical reporting. The core invariants (chronological holdout before any split, per-segment feature computation with join purge, next-open execution + obs_lag, wealth-not-reward checkpointing, mandatory entropy decay, config-as-snapshot single source of truth) are intact and in some cases better instrumented than at the time of the prior reviews.

**Since the May 31 Design Review and subsequent Critical Review (both in `docs/`):**
- Clear forward progress on **inference loading** (`rlbot/inference_load.py` + `freeze_vec_normalize_for_inference` + backtest consumption) and **OOS statistics** (stationary block bootstrap wired into `--detailed`, `--stochastic-paths` support, dedicated `tests/test_block_bootstrap.py`).
- **Eval segment mechanics** are now explicitly tested (`tests/test_eval_segments.py` asserts full-segment coverage for deterministic eval, segment-boundary truncation, cycling behavior) and train.py now computes `n_validation_blocks` from `get_segments()` rather than blindly using the old `eval_n_episodes` fallback.
- Dynamic universe support (N=5–55 via `universe.assets` keys + `--n-assets` slicing + manifest stamping of `n_assets`/`obs_dim`/`tickers`) is more mature and documented.
- Artifact layout, manifest richness, and run management (`run_artifacts.py`) are solid and unified under `Runs/`.
- Some reward ergonomics improved (linear inactivity, separate `eval_inactivity_penalty_scale`, action smoothing consistent across train/backtest).

**Core problems identified in the prior assessments are only partially mitigated:**
- Reward shaping term magnitudes remain badly imbalanced relative to the primary return + Sortino terms (participation and churn contributions are still ~1–2 orders of magnitude smaller in practice). The config comment acknowledges the intent but the numbers have not moved enough.
- The in-training validation signal is still a narrow, deterministic, full-segment-per-block but single-rollout estimator (one deterministic policy rollout per eval segment, `deterministic=True`, no start jitter for the eval envs). The "validation NAV cliff" (early peak ~9–13M steps while training reward keeps rising) is still trained through in full; `EvalNavBestModelCallback` + manual `--checkpoint best` is the only mitigation. Train explicitly prints `early_stop: off`.
- OOS evidence per window remains thin in the canonical record (RESEARCH.md registry + published numbers still rest on best-of-{final,best} single deterministic paths); the stronger statistics (stochastic paths, block bootstrap CIs) are opt-in CLI features.
- No automated patience-based early stopping or "best is the default artifact" change.
- Architectural coupling to a fixed (though now dynamically sized) menu, observation layout versioning, and stale agent instruction files persist.
- Test collection currently fails under bare `python3 -m pytest` (likely env/editable-install/path issues); the targeted unit tests that *do* exist are directly relevant to prior critique points (segments, block bootstrap, fracdiff, weight mapping, cap math).

**Overall posture:** The methodology and leakage discipline are worth preserving and generalizing. The system is still best treated as a **methodology testbed and reference implementation**, not production portfolio machinery. The most valuable next investments are (a) making the validation signal and reward shaping trustworthy enough to decide whether the "cliff" is real overfitting or an artifact of the estimator, and (b) building safe automation around the existing excellent run/contract machinery.

---

## 2. Comparison to Prior Assessments in `docs/`

### 2.1 RLBot_Design_Review.md (2026-05-31) — the more detailed and prescriptive of the two

This document is the stronger prior reference. It correctly identified three highest-leverage issues (reward term imbalance, narrow validation estimator as a plausible *cause* of the observed cliff rather than pure model pathology, and thin OOS statistics + post-hoc checkpoint selection). It also gave a precise code-location appendix.

**Addressed or improved since then:**
- Inference path (3.8 / rec #8): substantial progress via `inference_load.py` (fast load without optimizer state, VecNormalize freeze helper, `load_recurrent_ppo_with_vecnorm`, weight-swap for batch). Backtest consumes it. This was a documentation-vs-reality gap; now there is audited in-tree loading logic.
- Bootstrap (3.3 / rec #3): `block_bootstrap_log_rets` + `block_bootstrap_sharpe_percentiles` (stationary/Politis–Romano style with geometric block breaks) are implemented and called from `_print_detailed_stats` under `--detailed`. CLI exposes `--bootstrap-resamples` / `--bootstrap-avg-block`. A dedicated test file exists. (Still opt-in for "fast" and not the default in RESEARCH registry tables.)
- Stochastic paths (3.3): `--stochastic-paths N` supported in backtest with fan plots.
- Vestigial alias coupling (3.6 / rec #6): `rl_config.py` no longer contains a visible `sync_trading_env_aliases`; env `__init__` captures `self._env_cfg`, `self._reward_cfg`, per-asset cost arrays etc. from `get_config()` and the live config objects. Hot paths read the captured values. (Some module globals for obs_lag bounds may remain for train.py convenience; the dual-source hazard is much reduced.)
- Eval segments (3.2 / rec #2): `test_eval_segments.py` directly tests the deterministic full-segment path (`test_deterministic_reset_covers_full_segment_not_63_cap` asserts `>63` and matches segment runway; `test_step_truncates_at_active_segment_end`; `test_eval_cycles_one_segment_per_reset`). Train now does `validation_segments = eval_env.env_method("get_segments")`; `n_validation_blocks = len(...)`; passes `n_eval_episodes=n_validation_blocks` and `deterministic=True`. The old "first ~63 bars of each block" description in the review is no longer accurate for the current deterministic eval path.
- Dynamic N / hard-coded 10-asset (3.7 / rec #7): much better. `observation_dim_for_universe`, `slice_config_to_n_assets`, manifest stamps `n_assets`/`obs_dim`/`tickers`, `--n-assets` CLI, TRAINING.md checklist, 5–55 enforced. Still no separate `universe.yaml` or explicit `obs_layout_version` hash.
- Cap enforcement nits (3.9): still 5-iteration clip-and-redistribute + final `_enforce_long_only_simplex` (no automatic re-clip on the final projection). Safe at current 0.35 cap; the review's stress-test recommendation (fuzz + assert post-condition) is still relevant if cap is ever lowered.
- HY-OAS (3.9): still regime-extrapolated proxy + causal expanding OLS calibration (good); calibrated series is in the per-run data snapshot. Coefficients themselves not separately manifested.

**Not addressed (or only cosmetically):**
- Reward magnitudes (3.1 / rec #1 — highest leverage): `config.yaml` still has `reward_scale: 2000`, `risk_bonus_scale: 25`, `participation_reward_scale: 20`, `churn_penalty: 8.5`, `drawdown_*` multipliers that only bite on large DD. The participation/churn/drawdown terms remain numerically small vs the clipped-return and Sortino-diff terms on realistic daily moves. `rew_decomp/*` is emitted in `info` (good instrumentation) but the relative weighting problem the review quantified is not solved. The config comment "Bring regularizers into the same order of magnitude..." exists but the values have not been adjusted aggressively.
- Early stopping + best-as-default (3.4 / rec #4): `train.py:1086` comment is explicit ("No StopTrainingOnNoModelImprovement: train for full..."). `EvalNavBestModelCallback` still only side-saves `best/`. Train banner says `early_stop: off`. No patience logic.
- Reproducibility switch + lockfile (3.5 / rec #5): `apply_deterministic_seeds` + cuDNN flags still present and called; training envs still use `reseed_on_reset=True` + fresh `np.random.default_rng()` per episode (intentional for diversity). No `--reproducible` pin of per-env streams. No `uv.lock` / pinned Docker image in tree.
- Ablation harness (rec #11): none.
- Property/fuzz tests on cap and leakage regression at block boundaries (rec #9, 3.10): unit tests for weight math and segments exist; no end-to-end "train 100k steps then assert feature values immediately after a recorded boundary equal a fresh per-segment recompute" or "fuzz logits and assert `max(w[1:]) <= cap + tol` after projection".
- Pre-registered checkpoint rule and multi-seed + distributional OOS as the *primary* reported number.

The Design Review's strategic advice ("Prove generalization on the existing 10-asset menu before scaling... Fix reward + validation first") is still the correct priority order.

### 2.2 RLBot_Critical_Review.md (2026-06)

Shorter, reaches similar conclusions, and notes it supersedes an even earlier one. Its high-priority list (close the validation-NAV loop with early stopping or at least hard stop, modularize universe, ship clean inference, version obs spec) overlaps heavily with Design.

**Delta since then:** inference shipping and some modularization progress (as above). Validation loop and early stopping: not closed. Universe modularization: partial (dynamic N + slicing) but not a first-class `universe.yaml` driving everything + schema version.

Both prior reviews praised the honest "validation cliff" disclosure in RESEARCH.md and the leakage discipline. Those strengths remain.

### 2.3 Other docs drift

- `AGENTS.md` at root is largely a stale copy of older Claude-style guidance (still references `windows/window1_train.sh`, hard-coded 10-asset lists in the "must have exactly 10 entries" paragraph, `sync_trading_env_aliases`, legacy `runs/` layout). It is inconsistent with current `README.md`, `TRAINING.md`, `rl_config.py`, and the actual `scripts/` layout. This is documentation debt that will mislead future agentic tooling or contributors.
- `Claude.md` (the one read at session start) is closer to current reality but still contains some legacy `windows/` references.
- `RESEARCH.md` and `TRAINING.md` are reasonably current on commands and walk-forward registry (some pending windows), but the OOS table is sparse and still reflects the single-path + best-of reality the reviews flagged.
- `config/README.md` is accurate on the per-list discipline.

**Finding:** Instruction files for agents (AGENTS.md, Claude.md) have not been kept in sync with the refactors that addressed some of the coupling the reviews complained about. This matters more if we pursue auto research.

---

## 3. Detailed Design (Current State)

### 3.1 Configuration — single source of truth with snapshotting

`config/config.yaml` → `rlbot/rl_config.py:load_config` / `_parse_config` → frozen `RLConfig` dataclass tree (EnvironmentConfig, RewardConfig, TransactionCostsConfig, ..., CurriculumConfig, DataConfig, UniverseConfig).

- Per-asset lists (`benchmark_cap_weights`, `slippage`, `tx_fee`, `annual_holding_cost`) are validated for length match to `len(universe.assets)` via `_float_list` and `validate_config_for_universe`.
- `slice_config_to_n_assets` supports `--n-assets` without editing YAML (truncates lists, renormalizes benchmark weights).
- `set_config` / `get_config` global singleton (tests use autouse session fixture in `conftest.py`).
- Every run writes `Runs/<run_id>/config.yaml` (full effective dict) + rich `manifest.json` (universe metadata, exact calendar flags, bar counts, artifact paths, finished_at, params counts).
- Observation dim is derived: `observation_dim_for_universe(n_assets) = 10*n + 8 + 5*4` (per-asset market features + live mask + portfolio state + 4 macros × 5 derived). Matches `noisy_market_feature_count` + live + port + meta in env.
- Curriculum and entropy schedules expressed as fractions of budget (short vs long anchors) so the same YAML works for 50M and 120M runs.

**Strength:** Changing reward/cost/DR without a new run-id is obviously wrong because the snapshot + manifest + VecNormalize + model obs layout are bound together. The two-pass argparse in train.py ensures `--config` / `--n-assets` are applied before other defaults.

**Weakness:** Still no first-class universe definition object that could carry a `layout_version` hash or drive a separate cost/beta file. Adding/removing assets or changing obs construction still requires coordinated understanding across config + data_utils + env + saved artifacts.

### 3.2 Data pipeline and anti-leakage (the strongest part of the system)

`rlbot/data_utils.py` is the heart of the methodological credibility.

Key functions and invariants:
- `fetch_aligned_daily` (or cache load): yfinance + FRED HY OAS (graph CSV recent + HYG/IEF proxy back-projected via causal expanding OLS `_calibrate_hy_proxy_expanding`). Pre-listing rows dropped or bfilled to tiny positive + `asset_live=0/1` mask. No global dropna on the calendar.
- `reserve_chronological_holdout`: applied **before** any train/eval split. Explicit date-based or `holdout_days` tail. Only backtest ever sees the holdout arrays.
- `train_test_split_alternating` (block_size=126, eval_stride=4 by default): divides the *trainable* timeline into blocks; every 4th to eval. **Features are either passed precomputed on the full trainable or recomputed inside the function via `compute_feature_panel` on the raw ohlcv+macro slices.** Never across the whole panel or across holdout. Contiguous same-label blocks are concatenated into *segments*; `block_boundaries` list records the join offsets.
- `compute_feature_panel` (RSI, MACD, trend=EMA20-EMA100 distance, fracdiff d=0.4 on log price via causal convolution `fracdiff_series_1d`, realized vol panels): runs on contiguous slices only.
- `_neutralize_feature_warmup` (purge=25): in-place neutralizes RSI→50, MACD/fracdiff/trend→0 at the head of each joined segment. (Note: the split docstring says "feature_purge_warmup is retained for API compatibility but is not applied" in the current `_concat_ranges` path; purge still happens upstream in some flows via `align_panel_to_timeline` + explicit call sites.)
- `WalkforwardEnvPack`: NamedTuple + `env_kwargs()` that forces keyword construction into `MultiAssetPortfolioEnv` (prevents positional misalignment of the many feature arrays).
- `clip_index_until`, `align_panel_to_timeline`, `select_tradeable_columns` for manifest-driven subsetting on backtest.

**Causal execution contract (env + data):**
- Features at decision use data through `close[t - obs_lag]`.
- Rebalance at `open[t+1]`.
- Holding cost on pre-rebalance units at `close[t]`; MTM + P&L at `close[t+1]`.
- `asset_live[t_mkt]` zeros pre-IPO weights.
- Episodes in training never cross segment boundaries (env `_build_segments` + `_current_seg_end` truncation).

This is best-in-class for this style of research. The prior reviews were correct to call it out.

### 3.3 Environment and reward (`rlbot/trading_env.py`)

`MultiAssetPortfolioEnv` (gym.Env, not a Vec wrapper itself):
- Action: Box(-3,3)^{N+1} → (optional EMA smoothing α=0.15 on logits, consistent train+backtest) → softmax (cash competes) → long-only + per-asset cap (`max_single_asset_weight`, current default 0.35) via 5-iter clip-and-redistribute + final simplex projection. `portfolio_weights_from_action` also applies `asset_live` mask.
- Observation construction (`_build_obs`): 10N+28 layout (RETURN_HORIZONS fracdiff per asset + mean, realized vol + mean, RSI scaled, tanh(MACD), clipped trend, macro fracdiffs + vol, asset_live mask, current portfolio weights, current DD + progress). Noise added only on the noisy subset during training.
- Domain randomization (training only): per-episode `obs_lag` ~ Discrete[min..max], `fee_scale` ~ Beta(5,5) mapped to widening [dr_min, dr_max] bounds after curriculum release.
- Curriculum hooks: `set_curriculum_state(fee_override, churn_scale)` and `set_randomization_bounds`.
- Reward (per-step, emitted decomposed in `info["rew_decomp/*"]`):
  - `clipped_log_ret * reward_scale`
  - Sortino(agent vs benchmark, risk_window=63, min_steps=20) diff clipped ±3 * risk_bonus_scale
  - participation = gross_exposure * participation_bonus * participation_reward_scale
  - inactivity = linear cash_frac * penalties (base over 50% + ramp over 90%); scaled differently for eval envs
  - churn = turnover_frac * VIX_mult (clip 0.75–1.5 around 18) * churn_penalty * curriculum_churn_scale
  - quadratic DD penalty (on peak-to-current within episode)
- Stop loss (decorative 0.45 fraction of episode start NAV) + segment runway + max_steps truncation.
- Benchmark for Sortino and visuals is cap-weighted (live-masked) with the *same* friction model as the agent (`portfolio_step_nav`).

**Cap note:** 5 iterations + final renormalize (no re-application of per-asset cap after the last redistribution). At 0.35 and N=10 this is practically safe; at tighter caps or higher N it would need more iterations or a proper projection.

**Reward note (recurring from priors):** The shaping terms are still small. `rew_decomp` logging is present, which is the right instrumentation to diagnose this.

### 3.4 Training harness (`scripts/train.py`)

- RecurrentPPO (sb3-contrib) MlpLstmPolicy (2×64 LSTM, [128,128] pi/vf nets).
- SubprocVecEnv (n_envs=16 local default; Modal broker overrides to 32/64), VecNormalize (obs always, reward only on train copy).
- Callbacks (in priority order):
  1. `TradingCurriculumCallback`: fee-free → linear fee ramp → churn ramp (0→1) → progressive DR bound widening. Milestones are fractions of budget so they scale.
  2. `EvalNavBestModelCallback` (subclasses SB3 EvalCallback): runs deterministic full-segment rollouts on the eval env pack; saves `best/best_model` (and copies VecNormalize) on *max mean ending NAV* across the episodes of that eval cycle. Persists `eval_nav_history.npz`. Syncs obs_rms from train before eval. Does *not* stop training.
  3. `AdaptiveEntropyCallback`: mandatory cosine decay from explore_ent → final_ent starting at `decay_start_fraction` (0.585, after fee ramp); floors early in run; *not* gated on eval improvement (the improvement counter is advisory only).
  4. `CheckpointCallback` (every 1M steps).
  5. `TrainingVizCallback` (periodic PNG of training curves + eval NAV history).
- LR cosine to floor. n_epochs=3, batch sized so ~12 backprop passes per PPO pause on the  n_steps*n_envs rollout.
- Deterministic seeds applied, but training envs reseed for diversity.
- On exit (even KeyboardInterrupt or error) always persists Vec + final + best-adjacent Vec so the run is "trade ready."

Explicit banner prints the entire contract (early_stop off, obs_lag DR, reward formula with coefficients, eval = one full rollout per segment, etc.). Excellent.

### 3.5 Backtest & inference (`scripts/backtest.py`, `rlbot/inference_load.py`, `vecnorm_utils.py`)

- Requires `--run-id`; most calendar/universe/obs_dim defaults come from `manifest.json` (or explicit overrides for cross-window checks).
- Loads via `load_recurrent_ppo_inference` (fast, no optimizer state via dummy AdamW swap at load time) + `load_vec_normalize_for_inference` (which calls `freeze...`).
- Deterministic full-holdout rollout (max_episode_steps = full OOS length) + optional N stochastic paths (policy sampling) with fan plot.
- Benchmarks: SPY B&H, equal-weight daily rebalance, 60/40 calendar month, naive risk-parity (inverse vol, peer vol for pre-IPO).
- `--detailed`: subperiod stats, block-bootstrap Sharpe (stationary, 2.5/50/97.5 percentiles, default 8k resamples), ensemble NAV stats if paths provided.
- `freeze_vec_normalize_for_inference`: sets `training=False`, `norm_reward=False`, keeps `norm_obs=True`. Correct and enforced.
- `inference_load` also provides `swap_recurrent_ppo_weights` for efficient batch backtests across checkpoints of the same run.

This is a material improvement over the "no in-tree audited inference path" state called out in both priors.

### 3.6 Artifacts, run management, Modal

`rlbot/run_artifacts.py` + `RunPaths`: canonical `Runs/<run_id>/` tree (manifest, config snapshot, models/{final,best,checkpoints}/, plots, logs, tb_logs, eval_logs/). Legacy roots still readable for migration. `new_run_id` (W{window}_MMDD with collision _a/_b). `snapshot_data_cache` per run.

Modal (`scripts/modal_app.py` + `rlbot/modal_cloud.py`): same layout on `rlbot-runs` volume; `rlbot-cache` shared data volume; `sync --watch` for live plots, `--pull-all` for models; GPU broker scales n_envs/batch; deployable web plot/status endpoints. Resume from checkpoint supported. `--refresh-data` / upload_cache for window advancement.

This is production-grade research infrastructure for the problem size.

### 3.7 Tests

- `tests/conftest.py`: autouse session fixture loads + `set_config`.
- `test_environment.py`: weight mapping (simplex, cap, live mask), reward decomp shapes, basic step/reset.
- `test_core.py`: fracdiff weights (start at 1, sum properties), other math.
- `test_eval_segments.py`: the new targeted coverage for the exact eval pathology discussed in priors (full segment runway, boundary truncation, cycling, random-start min runway).
- `test_block_bootstrap.py`: stationary block bootstrap (directly answers the autocorrelation-destroying i.i.d. critique).
- `test_run_artifacts.py`, `test_eval_segments.py` etc.

**Current practical state:** Collection errors under bare invocation (missing editable install or path setup in this shell). The *existence and focus* of the tests is the right signal — they were written against the pain points the reviews surfaced.

---

## 4. Issues and Weaknesses (Prioritized)

### 4.1 Methodological / Empirical (Highest Leverage — Decide Whether the Edge Is Real)

1. **Reward shaping still numerically inert (Design 3.1).** Participation ~ gross * 1.0 typical contribution; churn even smaller on low-turnover policies; quadratic DD only on large intra-episode drawdowns. Return term (2000 * ~0.01–0.02 clipped) and Sortino diff (up to ±75) dominate. VecNormalize scales the *sum*, not relative importance. This is almost certainly why participation incentives have been "invisible" in some windows. `rew_decomp` exists — use it to re-derive coefficients so that at target behaviors (e.g. 15–25% turnover day, 60–80% cash day) each term is within ~10–30% of a typical return step.

2. **Validation signal remains a narrow deterministic estimator (Design 3.2).** Even with full-segment episodes and segment-derived n_blocks, it is still one deterministic policy trajectory per eval segment per eval call (no action noise, fixed start per segment for the eval envs, `random_start=False`, `domain_randomize=False`). With ~7–10 eval segments on a typical window, the signal has low effective sample size and zero intra-segment start diversity. This is a very plausible mechanical contributor to noisy/early-peaking eval NAV. The segment tests verify mechanics; they do not prove the signal is now a good proxy for OOS generalization.

3. **OOS statistics and checkpoint rule still thin in the published record (Design 3.3/3.4).** RESEARCH.md tables and the walk-forward registry emphasize single best-of-{final,best} deterministic paths. Stronger tools (`--stochastic-paths`, block bootstrap in detailed mode, seed ensemble script) exist but are not the default reported numbers. No pre-registered "we will always publish X from the best checkpoint" rule that is enforced by the harness.

4. **Early stopping (default on).** Default `early_stop_patience: 8` stops after 8 evals with no new best once the curriculum completes (`0` disables). Still worth monitoring whether DR ends at budget cap.

### 4.2 Architectural & Coupling

5. **Universe / obs layout still implicit and coordinated by hand.** No `universe.yaml`, no persisted `obs_layout_version` or feature-schema hash stamped into models + manifest + Vec stats. Changing the menu or the 10N+28 construction invalidates prior artifacts in non-obvious ways. Dynamic N via slicing is a good tactical improvement but not the strategic modularization the reviews asked for.

6. **Cap projection is best-effort, not guaranteed.** 5 iterations + final simplex only. Safe today; add a post-projection re-clip loop + a property test that fuzzes logits and asserts the post-condition when the cap is material.

7. **Stop-loss fraction (0.45) is decorative** on a diversified book over 63-day (or full-segment) episodes. Either remove or make it a meaningful risk control.

8. **HY OAS calibration coefficients** are not separately recorded (only the final series in the data snapshot). Regime extrapolation risk is real for early years.

### 4.3 Operational, Docs, and Testing

9. **Stale agent instruction files.** `AGENTS.md` is the most out-of-date (old layout, hard 10-asset language, references to `sync_trading_env_aliases` and `windows/*.sh`). This will bite any future auto-research effort or new contributor using agent tooling. `Claude.md` is closer but not perfect.

10. **Test collection / runner friction.** Bare `python3 -m pytest` fails collection. A one-line smoke ("install -e .[dev] then pytest -q --tb=no") should be trivial and green in CI or a fresh shell. Missing: integration smoke that actually runs a few thousand training steps and asserts leakage invariants at recorded block boundaries + cap post-condition under fuzz + rew_decomp magnitudes are as documented.

11. **No ablation or experiment harness.** Manual config edits + new run-ids + backtest is the current workflow. Fine for small N of windows; scales poorly for the systematic work the reviews recommend (reward coefficient sweeps, curriculum phase ablations, eval-strategy variants, seed ensembles).

12. **Missing production inference surface.** Even with `inference_load`, there is still no small, documented, audited `rollout.py` / `emit_weights.py` that a paper-trading or broker layer would actually call (recurrent state warmup rules, provenance, audit fields). The old `paper_trade/` / `ibkr_paper/` references are still absent (as noted in priors).

13. **Determinism vs diversity tension** is intentional but under-documented for reproduction of long runs. Same master seed + reseed_on_reset=True means two "identical" runs diverge in episode starts/DR draws. The banner advertises determinism; reality is "seeded diversity."

---

## 5. Feasibility and Benefits of the "Auto Research" Pattern / Approach

**Definition in this context.** "Auto research" here means using agent/tooling loops (LLM with code-edit + terminal + file + search tools, sub-agents, todo tracking, structured output) to close the empirical iteration loop at higher velocity and with better bookkeeping than manual human effort:

- Propose a precise hypothesis + minimal config patch (e.g. "double participation_reward_scale and halve inactivity penalties; expect higher gross exposure on W1 without degrading Sortino diff or increasing turnover beyond X").
- Allocate a new `--run-id` (or short proxy run), launch training (local short or Modal), poll or wait.
- On completion: parse `eval_nav_history.npz`, `rew_decomp` logs or TensorBoard scalars, `manifest`, backtest the best + final with stochastic paths + block bootstrap, produce a structured diff vs prior runs on the same window.
- Synthesize (cliff timing vs curriculum milestones, term contribution histograms, OOS vs benchmarks + CIs, behavior metrics like avg gross / turnover / cash frac).
- Propose next experiment or "declare result + archive".
- Optionally run parallel ablations or seed ensembles.

The pattern can also drive literature synthesis, plot generation, RESEARCH.md updates, etc.

### 5.1 Feasibility Assessment — High (with guardrails)

**Enablers already present (unusually good for a research codebase):**
- Fully scriptable, manifest-driven contract: `train.py --run-id X ...` + `backtest.py --run-id X --checkpoint best --detailed --stochastic-paths 30` is machine-callable and self-describing.
- Rich, versioned artifacts per run (config snapshot, full manifest with bar counts + dates + universe, eval NAV history npz, rew_decomp in logs, plots, TB, models + paired VecNormalize).
- `RunPaths` + `new_run_id` + `discover_run_ids_with_models` give the automation a place to live and a way to enumerate prior work.
- Modal integration + `sync --watch` / `--pull-all` means heavy lifting can be offloaded while the agent watches local plots or polls status endpoints.
- Leakage hygiene is *structural* (reserve before split, per-segment features inside the split function or explicit recompute, block_boundaries + env truncation, asset_live). An agent that only edits `reward.*`, `curriculum.*`, `entropy_schedule.*`, `hyperparameters.*`, or `environment.*` (non-date) sections and always uses the documented flags is unlikely to create silent leaks *if* it follows the existing CLIs.
- `rew_decomp/*`, eval NAV history, and block bootstrap give quantitative signals the agent can parse without vision models.
- Tests for the exact invariants (segments, bootstrap, fracdiff, weights) can be part of the "accept a proposed change" gate.

**Challenges and required constraints (non-negotiable):**
- Wall time and cost: 50M-step runs are hours on H100 even with 64 envs. Auto loops must support cheap proxy signals (short timesteps for reward-shape ablations, curriculum-fraction runs, or even frozen-policy rollouts on eval packs) + human approval gates for full-budget runs.
- Narrow/noisy validation signal: any auto "improvement" detector will overfit to the current estimator unless the estimator itself is improved first (see §4.1). The agent should be able to propose *changes to the eval harness* as first-class experiments.
- Leakage surface: the agent must never be allowed to pass precomputed features across splits, touch holdout dates, or reuse old VecNormalize across obs_dim changes. The safest pattern is "agent proposes patch + hypothesis + short justification; human (or stronger verifier) approves; harness launches with fresh run-id and the canonical train/backtest CLIs only."
- Non-determinism: even with seeds, reseed_on_reset + DR means "identical" runs differ. The harness must treat single-run deltas as noisy; prefer seed ensembles or at least report variance.
- Cherry-picking and multiple-testing: the automation makes it *easier* to run 50 variants and only surface the flattering one. The pattern must log the full search tree and require pre-registration of success criteria for any "result" that will be written into RESEARCH.md.
- Statefulness and long context: the agent needs durable memory of prior runs, their hypotheses, and the current best understanding of the cliff. This review itself (and the prior two) should be part of the context or retrieved.
- Modal vs local: the agent needs to know which compute tier is appropriate and how to poll.

**Bottom line on feasibility:** The *infrastructure* is unusually ready (better than most RL research repos). The *methodological guardrails* are the real work — and they are exactly the same guardrails a careful human researcher would need. Implementing a small `scripts/research_harness.py` (or equivalent) that codifies "safe experiment" (only allowed patch locations, always new run-id, always manifest-driven backtest, structured result JSON + hypothesis log) would make the pattern immediately usable and self-auditing.

### 5.2 Benefits — Very High for the Current Bottlenecks

The highest-leverage open items from the priors are *exactly* the kind of systematic, quantitative, multi-run work that is tedious and error-prone for a human but natural for a well-instrumented auto loop:

- Reward coefficient sweep + term contribution measurement (participation 5×/10×/20×, churn, drawdown quadratic, inactivity) on one or two windows; measure effect on gross exposure, turnover, eval NAV trajectory, and OOS.
- Validation estimator variants (add a stochastic eval policy copy? random starts within segments for some fraction of eval episodes? report effective unique coverage not just "N episodes") and re-measure whether the cliff timing or severity changes.
- Curriculum phase ablations (turn churn on earlier/later; different fee ramp shapes; DR widen earlier) with quantitative "did this move the cliff or the participation behavior?" readout.
- Multi-seed ensembles + distributional OOS (median + 10–90% across seeds/paths) as the *primary* number for a window, with pre-registered checkpoint rule.
- Once the 10-asset base is on firmer ground: cheap exploration of N=15–25 menus (add a few more always-live proxies or single-name equities) with the same leakage harness, using the auto loop to keep the search honest.

Secondary benefits: faster filling of RESEARCH.md tables, automatic generation of "training dynamics vs curriculum milestones" plots and analysis, regression detection (a new code change that moves eval NAV or feature stats at boundaries should fail a gate), and extraction of reusable patterns into a small library.

**Risk if pursued naively:** accelerating the publication of still-unfalsifiable claims (the exact failure mode the priors warned against). The correct sequence is: use auto research *to fix the estimator and reward first*, then use it to scale or claim.

---

## 6. Advice on Continued Evolution of the System

### Short-term (do these before any N-scaling or "deployment" language)

1. **Rebalance the reward (highest ROI change).** Use the existing `rew_decomp` to set coefficients so shaping terms are visible at realistic behaviors. Log per-term histograms or quantiles during training. Re-run the affected windows (or at least W1) and re-examine the cliff and participation behavior. This directly tests one of the Design Review's central hypotheses.

2. **Make the validation signal trustworthy and close the early-stop loop.** Options (pick one or a cheap combination):
   - Allow a fraction of eval episodes to use `random_start=True` (or a separate "eval with jitter" env) so starts are not locked to segment heads.
   - Report effective sample size (unique segment coverage × diversity) alongside raw episode count.
   - Add patience-based early stopping (or at minimum a hard "stop if no new best mean_ending_nav for K evals after fee_ramp + churn_start") and make `best/` the default artifact that `backtest.py --checkpoint best` (and the registry) resolves to without extra flags.
   - After the above, re-evaluate whether the "validation NAV cliff while train reward rises" survives. If it does, treat it as a real finding about LSTM + alternating blocks + this reward, not an implementation bug.

3. **Promote the stronger OOS statistics to first-class.** Make `--detailed --stochastic-paths 30` (or a sensible default N) + block bootstrap the normal way a run is summarized. Update RESEARCH.md walk-forward tables to show median/interval numbers and to note the checkpoint rule explicitly. Add a small "pre-registered analysis" section or flag in the manifest.

4. **Clean the agent docs.** Update or replace `AGENTS.md` (and align `Claude.md`) with the current layout, the dynamic-N story, the inference_load path, the absence of `windows/` and `paper_trade/`, and the actual coupling (config objects captured in env, not module globals for most things). This is cheap and high-value if auto research is coming.

5. **Add a minimal integration smoke + property tests.** A fast "train 20k–50k steps on a tiny synthetic panel + assert no NaNs in features immediately after recorded block boundaries, cap respected under 1000 fuzzed actions, rew_decomp keys present and finite, eval NAV history written" would have caught several classes of future regression. The existing unit tests are good; this would be the glue.

### Medium-term (make experiments cheap and the harness reusable)

6. **Universe modularity.** A `config/universe.yaml` (or section) that is the single source for tickers, macros, per-asset costs, benchmark weights, and a `layout_version` or feature schema hash. Stamp the hash into manifest, model metadata, and VecNormalize sidecar. This makes N-experiments and feature-ablation experiments first-class and less error-prone.

7. **Small research harness layer.** Something like `scripts/research.py` or a `research/` package that the agent (or a human with a Makefile) can drive:
   - `propose_experiment(hypothesis, config_patch, budget_fraction=1.0, seeds=1)`
   - `launch(run_id, ...)` (local or modal)
   - `analyze(run_id)` → structured dict + human-readable card (cliff timing, term magnitudes, OOS table with CIs, behavior stats)
   - `compare(runs)` and "suggest next" heuristics.
   - Hard-coded allow-list of patchable sections + mandatory fresh run-id + manifest-driven backtest.
   - Writes a `research_log.jsonl` (hypothesis, patch, run_id, outcome, verdict) that becomes part of the durable record.

   This turns the auto research pattern from ad-hoc prompting into a reproducible, auditable tool.

8. **Ablation and sensitivity tooling.** Toggle individual reward terms, curriculum phases, DR bounds, etc., via CLI or patch, with automated comparison to a baseline run on the same window. Make claims like "the churn term helped Window 2 participation" falsifiable with numbers.

9. **Reproducibility story.** Add a `--reproducible` mode that derives per-env seed streams from the master seed (so same-seed runs are bit-identical when desired) while keeping the diversity mode the default. Add a lock file (uv/pip-tools) or a documented Docker image for the 50M-step regime.

### Longer-term / strategic

10. **Extract the reusable hygiene.** The leakage-safe walk-forward split + purge + segment boundary contract, the wealth-based `EvalNavBestModelCallback`, the mandatory (non-eval-gated) entropy schedule, the curriculum + DR hooks, the causal env + cost model, the frozen-Vec inference load, and the manifest+run-artifact discipline are the durable intellectual property here. Factor the non-asset-specific pieces into a small `market_rl_harness` (or similar) so that future efforts (larger equity universes, intraday, LLM-derived features + RL allocator, execution-aware costs) inherit the anti-leakage and reproducibility posture without re-inheriting the 10-asset coupling.

11. **Reconsider inductive bias once the 10-asset methodology is solid.** RecurrentPPO on engineered fracdiff/RSI/MACD + macro of ten always-live proxies is a reasonable bet. Single-name equities (earnings jumps, gaps, delistings, fat tails, news) have different statistics and will stress the continuous-rebalance + daily-cost assumptions. The right time to discover this is *after* reward balance and validation signal are fixed on the current menu.

12. **Capacity, market impact, and execution before any live framing.** The per-asset costs are static and small. There is no market-impact model, no capacity limit, no borrow/financing for the FX or futures legs, no spread-crossing realism beyond the configured slippage. Fine for research; load-bearing for any real capital. Treat "paper trade" hooks as measurement infrastructure, not a deployment claim.

13. **Keep the honest posture.** The authors' willingness to document the validation cliff and to not overclaim statistical significance is rare and valuable. Any automation (auto research or otherwise) must be constrained to preserve that posture — results are evidence *within the documented pipeline*, not guarantees.

---

## 7. Specific Code Locations Worth Attention (Current)

| Area | Location | Note |
|------|----------|------|
| Reward imbalance | `config/config.yaml:52-55`, `trading_env.py:884-897` (participation/churn/drawdown vs return 858 / sortino 871) | Still small; use rew_decomp to re-scale |
| Validation estimator | `train.py:1077-1097` (n_validation_blocks from segments, deterministic=True, n_eval_episodes), `trading_env.py:756-764` (deterministic reset) + `test_eval_segments.py` | Better than before but still narrow; add diversity |
| Early stop / best default | `train.py:1086` comment + `EvalNavBestModelCallback` (saves but does not stop), `backtest.py` main | "best" is opt-in; make default + add patience |
| Block bootstrap & stochastic | `backtest.py:810-861` (block_bootstrap_*), `1136` (call site), CLI args 1358-1368, 1341 | Good; promote to default detailed path |
| Inference loading | `rlbot/inference_load.py:56-128` (load_recurrent_ppo_inference + vecnorm), `vecnorm_utils.py:16` (freeze) | Real progress since priors |
| Cap projection | `trading_env.py:90-107` (5-iter clip + final simplex only) | Safe at 0.35; harden + fuzz test if tightening |
| Train/test split leakage core | `data_utils.py:827-969` (`train_test_split_alternating` + `_concat_ranges`), `898` (boundaries), `428` (neutralize) | Excellent; add regression assert at boundaries |
| Manifest & artifacts | `run_artifacts.py:83-156` (RunPaths), `train.py:816-839` + `1163-1190` (write) | Strong contract for automation |
| Stale agent docs | `AGENTS.md:1-74` (old windows/ refs, 10-asset literals, sync aliases) | Update or deprecate |
| Eval segments test (good example) | `tests/test_eval_segments.py:78-99` | Directly answers a prior critique; expand pattern |
| HY OAS calibration | `data_utils.py:249-281` (expanding OLS), `312-328` (attach) | Causal but coeffs not separately versioned |

---

## 8. Conclusion

RLBot's engineering discipline around leakage prevention, causal microstructure, config-driven reproducibility, wealth-based selection, and candid reporting of negative dynamics (the validation cliff) continues to set it apart from most public RL-for-trading artifacts. The prior reviews were fair: the hygiene is real and worth investing in; the headline claims rest on a still-thin statistical and methodological base.

Since those reviews, the project has shipped meaningful improvements (inference loading, block bootstrap, eval segment testing and dynamic-N support). The highest-leverage open items remain the ones the priors flagged: reward term balance and validation signal fidelity. Closing those (with the help of the instrumentation and tests that already exist) is the prerequisite for any credible scaling or deployment conversation.

The "auto research" pattern is **highly feasible here** precisely because the run contract, artifacts, and leakage invariants are already so well structured. The biggest risk is using faster iteration to amplify unfalsifiable claims rather than to falsify and improve the method. The recommended path is to first encode the guardrails (allowed patch surface, pre-registration, structured result cards, mandatory use of the canonical CLIs), then use the loop to drive the exact experiments the priors called for (reward rebalance, validation estimator variants, curriculum ablations, multi-seed distributional OOS). If that process makes the 10-asset results more credible, the same harness + extracted leakage library becomes a powerful foundation for larger or hybrid efforts.

The codebase is a strong foundation. Continued evolution should focus on *validating and hardening the method* on the current problem before *scaling the menu or the claims*.

---

*End of Grok review 2026-06-05. Prior assessments: docs/RLBot_Design_Review.md (2026-05-31), docs/RLBot_Critical_Review.md (2026-06). This document is intended to be read alongside RESEARCH.md, README.md, and the source.*