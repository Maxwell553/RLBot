# RLBot (MarketTrainer) — Design & Implementation Review

> **Status (2026-06-09):** Historical snapshot (pre-evolution-roadmap). OOS figures cited below are **obsolete**. **Current pipeline:** `independent` feature split, benchmark excess + Sortino cap, cap **0.25**, aligned train/eval fee curriculum, post-`fee_ramp_end` best-model gate. See [RESEARCH.md](RESEARCH.md).

**Date:** 2026-05-31
**Reviewer scope:** Full read of `trading_env.py`, `train.py`, `data_utils.py`, `backtest.py`, `backtest_sweep.py`, `rl_config.py`, `run_artifacts.py`, `vecnorm_utils.py`, `config.yaml`, `tests/`, `windows/`, `README.md`, `RESEARCH.md`.
**Goal:** Independent technical assessment grounded in the actual code (not the README's self-description), with concrete, prioritized recommendations.

> A prior `RLBot_Critical_Review.md` exists in the tree. This document supersedes it: it reaches the same broad conclusions but corrects two of its claims (clip-and-redistribute risk; HY-OAS persistence) and adds several findings that only surface from reading the numbers and the env reset logic.

---

## 1. Verdict

RLBot is a **genuinely above-average RL-for-trading research codebase**. The engineering discipline around leakage prevention, causal execution, config-as-source-of-truth, and honest reporting is better than the large majority of public work in this space. The authors have also done the rare and valuable thing of **documenting their own negative result** (the validation-NAV cliff) instead of hiding it.

The system is, however, a **research prototype, not portfolio machinery**, and its headline OOS numbers rest on a thin statistical base. The most important issues are not bugs — they are **methodological and reward-design** issues that determine whether the reported edge is real. My three highest-priority concerns:

1. **The auxiliary reward-shaping terms are numerically negligible** relative to the return/Sortino terms (off by ~2 orders of magnitude). The behaviors the team is trying to tune (participation, churn) are probably not being shaped at all.
2. **The in-training validation signal is far narrower than "75 episodes" implies** — it deterministically re-tests only the first ~63 bars of each eval segment from a fixed start. This plausibly *causes* the noisy, early-peaking validation NAV the team is fighting.
3. **OOS evidence is one deterministic path per window across three windows, with post-hoc best-of-{final,best} checkpoint selection.** This is honestly disclaimed in RESEARCH.md, but it means the standout result (W3 +37.7%) should be treated as a hypothesis, not a finding.

Everything below expands these and the rest.

---

## 2. What Is Done Well

**Leakage discipline (the strongest part of the codebase).**
- `train_test_split_alternating` (`data_utils.py:579`) takes **only raw `ohlcv`+`macro`** and recomputes RSI/MACD/fracdiff *per contiguous block* inside `_concat_ranges` (`data_utils.py:651-686`). EWM and fracdiff memory genuinely cannot cross a block boundary.
- Join purge (`_neutralize_feature_warmup`, `data_utils.py:276`) neutralizes the first 25 bars after each stitch.
- `reserve_chronological_holdout` (`data_utils.py:503`) is applied **before** any split; the OOS tail is structurally unreachable from training code.
- Causal execution is correct and explicit: market features at `t_mkt = t - obs_lag` (`trading_env.py:464`), rebalance at `open[t+1]` (`trading_env.py:648,658`), holding cost + MTM at `close[t+1]` (`trading_env.py:660-662`). Fracdiff is a causal convolution (`fracdiff_series_1d`, `data_utils.py:87`).
- Pre-IPO rows are dropped rather than back-filled (`fetch_aligned_daily`, `data_utils.py:454-457`) — avoids anchoring to a future listing price.

**Checkpoint selection by wealth, not reward.** `EvalNavBestModelCallback` (`train.py:134`) saves `best_model.zip` on **mean ending NAV**, collected via the `EpisodeEndNavRecorder` wrapper, not on episodic reward. This is the right objective and sidesteps the classic "low-churn cash policy farms shaped reward" trap.

**Mandatory entropy decay.** `AdaptiveEntropyCallback` (`train.py:218`) forces a cosine decay from `explore_ent → final_ent` starting at 45% of the run, *independent of eval performance* (`train.py:285-289`). The eval-improvement counter is advisory only. This avoids premature entropy collapse without letting a lucky eval freeze exploration.

**Inference-time normalization is frozen correctly.** `freeze_vec_normalize_for_inference` (`vecnorm_utils.py:10`) sets `training=False`, disables reward norm, keeps obs norm. Backtest uses it (`backtest.py:224`) and even raises a clear error on obs-dim mismatch from stale stats (`backtest.py:213-223`).

**Config as a typed, validated, snapshotted single source of truth.** `rl_config.py` parses `config.yaml` into frozen dataclasses with per-field presence checks and 10-element array validation (`_float_list`), and every run snapshots the resolved file to `runs/<id>/config.yaml`. The two-phase argparse (`train.py:461-464`) loads `--config` before computing other defaults — clean.

**Honest reporting.** RESEARCH.md shows the training-vs-validation divergence plots, states the survivorship/vendor/calibration limitations, and explicitly declines to claim statistical significance. This is the right posture.

---

## 3. Issues & Weaknesses

### 3.1 Critical — reward-term magnitudes are wildly imbalanced

With `reward_scale = 2000` and `max_step_log_return = 0.03`, the **return term spans ±60 per step** and a typical day is ~±20. The Sortino differential is `clip(±3) × 25 = ±75`. Now compare the shaping terms (`trading_env.py:690-702`, `config.yaml:19-31`):

| Term | Max magnitude / step | Typical |
|------|----------------------|---------|
| Scaled return | ±60 | ±10–20 |
| Sortino differential | ±75 | ±10–25 |
| Inactivity (cash>50% / >90%) | −5 / −0.1 | −5 |
| **Participation bonus** (`0.1 × gross`) | **+0.1** | +0.05 |
| **Churn** (`0.05 × |Δw|`, \|Δw\|≤2) | **−0.1** | −0.02 |
| Quadratic drawdown (`dd²×100`) | bites only on big DD: 0.25 @5%, 4 @20% | ~0.1 |

**The participation bonus and the churn penalty are ~100–600× smaller than the primary terms.** They cannot meaningfully steer a policy optimizing scaled return + Sortino. This is almost certainly *why*:
- **Window 2 underexposure persists** — the participation bonus meant to force equity beta is numerically invisible (RESEARCH.md attributes the miss to "insufficient equity beta"; the fix it cites, the churn term, is itself negligible).
- **The quadratic drawdown penalty only activates in regimes the 63-day clipped-return episodes rarely produce**, so it is mostly decorative alongside the per-step return clip.

VecNormalize divides reward by a running std, but that scales all terms together — it does **not** rebalance their *relative* weights. This is the single highest-leverage knob in the system and it is currently mis-set. Recommended: re-derive shaping coefficients so that, at the behaviors you care about (e.g. a 20% turnover day, a 70%-cash day), each shaping term is within ~10–30% of a typical return-term step, then tune from there. Consider logging the per-term reward decomposition to TensorBoard so this is visible.

### 3.2 Critical — the validation estimator is far narrower than it looks

`eval_n_episodes = 75` (`config.yaml:69`) suggests a robust validation sample. It isn't. The eval env is a **single** env (`train.py:770`), `random_start=False`, `reseed_on_reset=False`, `domain_randomize=False`. In `reset()` the deterministic branch (`trading_env.py:606-615`) sets `self._t = earliest` (the segment's first usable bar) and runs `max_episode_steps = 63` steps. Consequences:

- Each eval episode **always starts at the first bar of an eval segment** and covers only the **first ~63 bars** of that segment. The rest of every eval block is never scored.
- With `eval_stride=4` over ~3,700 trainable bars, there are only a handful of eval segments (~7). The 75 episodes simply **cycle the same ~7 deterministic rollouts ~10× each** (`seg_idx = _reset_count % len(segments)`), so the *effective* sample is ~7 fixed 3-month windows, not 75.
- A deterministic policy on a deterministic start yields identical NAVs each cycle — the extra episodes add zero information and ~10× wasted eval compute.

A validation metric built from ~7 fixed, non-overlapping 63-bar slices is a **high-variance, narrow estimator of generalization** — which is a very plausible mechanical cause of the "validation NAV peaks at 9–13M then oscillates" cliff the team treats as a model pathology. Before concluding the LSTM overfits, fix the estimator: sample eval start points across the *whole* eval segment (the `random_start and segments` branch already exists, lines 616-627), evaluate full segments, or at minimum stop pretending 75 ≫ 7.

### 3.3 Critical — OOS evidence is thin and checkpoint-selected

`backtest.py` runs **one deterministic rollout** over the entire holdout slice (`max_episode_steps = n_bars`, `deterministic=True`, `backtest.py:200,605`). So each window's OOS number is a single path of ~260–500 bars. The bootstrap Sharpe band (`backtest.py:358-373`) resamples days **i.i.d.**, destroying autocorrelation, and the code itself flags that positive skew inflates short-window Sharpe (`backtest.py:374-377`). RESEARCH.md then reports **best-of-{final, best} per window after seeing OOS**, and the W3 +37.7% rests on the early checkpoint whose validation peak (~9M) the report admits "coincides approximately with the end of the fee curriculum ramp."

None of this is hidden — RESEARCH.md's "What this report does not claim" section is admirably candid. But the implication should be loud: **n=3 windows × 1 path × post-hoc checkpoint pick is hypothesis-generating, not edge-confirming.** Strengthen with: multiple stochastic rollouts (non-deterministic actions) for a distribution, a stationary/block bootstrap instead of i.i.d., several seeds per window, and a pre-registered checkpoint rule (see 3.4).

### 3.4 High — no automated early stopping; "best" is not the default artifact

`train.py:869` explicitly disables `StopTrainingOnNoModelImprovement` and trains the full budget. Given the documented cliff, this **burns the majority of compute past the useful point** and risks shipping `ppo_portfolio_final.zip` (often strictly worse OOS) if someone forgets to pass `--model .../best/best_model.zip`. Make the best-by-validation-NAV checkpoint the default the backtest resolves to, and add patience-based early stopping (e.g. stop if no eval-NAV improvement for K evals after the curriculum completes). Pair with 3.2 so the patience signal is trustworthy.

### 3.5 High — stated determinism contradicts intentional env reseeding

`apply_deterministic_seeds` (`rl_config.py:379`) sets Python/NumPy/Torch seeds, `use_deterministic_algorithms(True)`, and cuDNN-deterministic flags, and the run banner advertises reproducibility. But training envs are created with `reseed_on_reset=True` (`train.py:753`), and `reset()` then does `self._rng = np.random.default_rng()` with **fresh OS entropy every episode** (`trading_env.py:573-574`). Episode start points, DR fee/lag draws, and observation noise are therefore **not reproducible across runs with the same seed**.

This is *intentional* ("seed shuffling" is listed as an anti-overfitting measure), and it's a defensible choice — but the determinism scaffolding then creates a false impression. Two same-seed runs will diverge, which complicates debugging, ablations, and any "bit-reproducible" claim. Recommendation: make this a deliberate, documented switch (`--reproducible` that pins a per-env seed stream derived from the master seed), so you can get exact reproduction when you need it and entropy diversity when you don't. Combined with the lack of a dependency lock file (`requirements.txt`/`pyproject.toml` use `>=` only), the system currently cannot reproduce a 65M run exactly.

### 3.6 Medium — `sync_trading_env_aliases` is largely vestigial and misleading

`rl_config.sync_trading_env_aliases` (`rl_config.py:353`) mutates ~18 module-level globals in `trading_env` (`REWARD_SCALE`, `CHURN_LAMBDA`, `ASSET_SLIPPAGE`, …). But the env's hot path reads from **config objects captured at `__init__`** (`self._reward_cfg`, `self._asset_slippage`, etc., `trading_env.py:197-202`) and from `get_config()` directly (`portfolio_weights_from_action`, `trading_env.py:97`). Most of those synced globals are **never read** — only `MIN_OBS_LAG`/`MAX_OBS_LAG` are still used (`trading_env.py:361`, imported by `train.py`). The result is a load-bearing-looking mechanism that is mostly dead, plus a genuine dual-source-of-truth hazard (`self._min_t` uses `env_cfg.max_obs_lag` at line 287 while `_build_segments` uses the module global at line 361 — consistent only because `set_config` happens to sync both). Delete the vestigial aliases; keep the few that are real as explicit constants or read them from config everywhere. The CLAUDE.md I wrote documents this coupling, but the right end state is to remove it.

### 3.7 Medium — hard-coded 10-asset / 4-macro universe

`N_ASSETS=10`, `N_ACTIONS=11`, `N_MACRO=4` are structural constants threaded through `data_utils`, `trading_env`, `config.yaml` (10-element cost/benchmark literals), `rl_config` validation, and the saved VecNormalize/obs layout. Adding or removing one instrument is a coordinated, multi-file, model-invalidating change, and the 118-d observation layout is implicit (versioned only by run-id folklore; note RESEARCH.md juggling 83/98/108/118-d). For a *research* testbed this is tolerable, but it blocks cheap experimentation. Recommendation: a `universe.yaml` (tickers, macros, cost vectors, benchmark weights) that drives observation dimensions, cost arrays, and a persisted `obs_layout_version`/schema hash stamped into each model + manifest.

### 3.8 Medium — documented inference/deployment path does not exist in-tree

`train.py`'s module docstring and the README reference `paper_trade/paper_trade.py` and `ibkr_paper/`, but both are gitignored (`.gitignore:31-32`) and absent from a fresh clone; `pyproject.toml` ships only `train`/`backtest` entry points. There is **no in-tree, audited rollout/inference module**. Beyond the doc-reality gap, this means the path most likely to be used live (recurrent-state warmup, weight emission, audit metadata) is the least tested. Ship a small `inference.py` that loads a run-id + best/final, freezes VecNormalize, handles LSTM state reset/warmup, and emits target weights with provenance — even if broker drivers stay out-of-tree.

### 3.9 Low/Medium — correctness & realism nits

- **Cap enforcement is best-effort, and the final projection doesn't re-clip.** `portfolio_weights_from_action` (`trading_env.py:100-117`) iterates clip-and-redistribute 5×, then ends with `_enforce_long_only_simplex`, which **only renormalizes to sum 1 — it does not re-impose the per-asset cap**. In practice this is safe (softmax means at most one leg can exceed a 50% cap initially, and the overflow shrinks geometrically), so the prior review's "not obviously correct under 6+ simultaneous cap hits" overstates the risk — 6 legs can't simultaneously exceed 50%. But the contract is "best-effort cap," not "guaranteed cap." If you lower the cap (e.g. to 20%, where many legs *can* exceed it), increase the iteration count and assert the post-condition. Add a property test that fuzzes logits and asserts `max(w[1:]) ≤ cap + tol`.
- **HY-OAS calibration regime, not persistence.** The prior review claimed the calibration "is not persisted with the cache." That's misleading: the *calibrated series* is baked into `data_cache.npz` and snapshotted per run, so a given run is reproducible. The real issue is upstream: `fetch_fred_daily_series` pulls the FRED **graph CSV (recent window only)** (`data_utils.py:142`), so the `np.polyfit` overlap calibration (`data_utils.py:209-217`) fits on recent data and then back-projects the HYG/IEF proxy across 2006–2018 — a **regime-extrapolated** credit feature. Persist the calibration coefficients in the manifest and prefer a full-history OAS source if this feature matters.
- **obs_lag inconsistencies.** Market features lag to `t-obs_lag` while NAV/drawdown meta use `close[t]` (`trading_env.py:493-499`). Both are known at decision time, so it's not a leak, but it's an asymmetry worth a comment. Separately, `_compute_noise_scale` is computed once at `__init__` with `self.obs_lag=0` (the factory passes `obs_lag=0`) even though training resamples obs_lag per episode — negligible, but technically the noise scale doesn't track the active lag.
- **Macro log-returns of spread/yield series.** `_macro_realized_vol` / fracdiff take `log`-returns of TNX (a yield) and HY_OAS (a % spread) (`trading_env.py:446-458`). Level changes are the conventional transform for these; log-returns are an unusual (though bounded) feature choice.
- **`stop_loss_fraction=0.45`** (55% DD over 63 days on a diversified book) is effectively never triggered — decorative, as RESEARCH.md concedes.

### 3.10 Low — test coverage

15 unit tests cover weight-mapping simplex/cap math and a few data helpers — good as far as they go. Missing: any **integration test** that runs a few hundred training steps and asserts (a) leakage invariants (features right after a block boundary equal a fresh per-segment recompute), (b) NAV/booking identities, (c) the cap post-condition under fuzzed actions. `windows/validate_split.py` and the RESEARCH.md command blocks also carry hard-coded dates/paths (note the stale `/Users/maxingargiola/...` path at RESEARCH.md:267) that can drift from `config.yaml`.

---

## 4. Recommendations (prioritized)

**Do first — these decide whether the edge is real:**
1. **Rebalance the reward (3.1).** Re-scale participation/churn/drawdown so they're within an order of magnitude of the return term at the behaviors you want to shape; log a per-term reward decomposition. This is the cheapest high-impact change.
2. **Fix the validation estimator (3.2),** then re-examine whether the "validation cliff" survives. Sample eval starts across whole eval segments; report effective sample size, not episode count.
3. **Harden OOS statistics (3.3):** multiple stochastic rollouts, block bootstrap, multiple seeds, and a **pre-registered** checkpoint-selection rule so headline numbers aren't best-of-N on the holdout.
4. **Automated early stopping + best-as-default artifact (3.4).**

**Do next — make experiments cheap and trustworthy:**
5. **Reproducibility switch + lock file (3.5).** A `--reproducible` per-env seed stream; pin dependencies (`uv.lock`/`pip-tools`) or a Docker image.
6. **Remove vestigial alias coupling (3.6);** read config everywhere.
7. **Modularize the universe (3.7)** behind `universe.yaml`; stamp an `obs_layout_version` into models + manifest.
8. **Ship `inference.py` (3.8)** with frozen VecNormalize, LSTM warmup, and audit metadata.

**Then — robustness & rigor:**
9. Property/fuzz test the cap post-condition (3.9); add leakage-regression integration tests (3.10).
10. Persist HY-OAS calibration coefficients; consider a full-history OAS source (3.9).
11. Add an ablation harness (toggle each reward term, each curriculum phase) so claims like "churn penalty helps Window 2" are measured, not asserted.

---

## 5. Evolution / strategic direction

- **Prove generalization on the existing 10-asset menu before scaling the universe.** Adding 50 or 500 names to a policy whose validation signal is a 7-window estimator and whose shaping terms are numerically inert just scales an unvalidated process. The order should be: fix reward + validation (§4.1–4.2) → confirm the cliff is real or an artifact → *then* modularize and expand.
- **Extract the reusable methodology.** The leakage-safe walk-forward split, the wealth-based checkpoint callback, the mandatory-entropy schedule, the curriculum, and the deterministic harness are the durable assets here. Factor them into a small library so future RL (or hybrid LLM-feature/RL) efforts inherit the hygiene without inheriting the 10-asset coupling.
- **Reconsider the inductive bias as the universe changes.** RecurrentPPO on engineered fracdiff/RSI/MACD features of ten always-live macro proxies is a reasonable bet; single-name equities (gaps, earnings, delistings, fat tails) have very different statistics and would stress both the env's continuous-rebalance assumptions and the cost model.
- **Treat costs and capacity seriously before any "deployment" framing.** The per-asset slippage/fee vectors are static and small; there's no market-impact, no capacity model, no borrow/financing for the FX/futures legs. Fine for research, load-bearing for live.

---

## 6. Code-location appendix

| Area | Location | Note |
|------|----------|------|
| Reward magnitude imbalance | `trading_env.py:676-702`, `config.yaml:19-31` | Shaping terms ~100–600× smaller than return/Sortino |
| Narrow eval estimator | `trading_env.py:606-615`; `train.py:770,876` | Deterministic, fixed-start, first-63-bars-only; 75 eps ≈ 7 unique |
| Single-path OOS + i.i.d. bootstrap | `backtest.py:200,358-377` | One deterministic rollout; autocorrelation discarded |
| No early stopping | `train.py:869` | Full budget despite documented cliff |
| Reseed vs determinism | `rl_config.py:379`; `trading_env.py:573-574`; `train.py:753` | Same-seed runs diverge |
| Vestigial aliases | `rl_config.py:353-376`; `trading_env.py:40-58` | Mostly unread; dual source for obs_lag |
| Hard-coded universe | `trading_env.py:36-37`; `data_utils.py:27-63`; `config.yaml:33-37` | 10/11/4 threaded everywhere |
| Missing inference path | `.gitignore:31-32`; `train.py:9`; README | `paper_trade/`, `ibkr_paper/` not in tree |
| Cap best-effort, no final re-clip | `trading_env.py:100-117` | Safe at 50%; fuzz + assert if cap lowered |
| HY-OAS regime extrapolation | `data_utils.py:142,209-217` | FRED recent-window fit back-projected; coeffs not in manifest |
| Test gaps | `tests/` | No integration/leakage/fuzz tests |

---

## 7. Bottom line

The hygiene here is real and worth preserving — leakage prevention, causal execution, wealth-based selection, honest reporting. But the **reported edge is currently unfalsifiable in the directions that matter**: the shaping terms that supposedly tune behavior are numerically inert, the validation signal that drives checkpoint selection is a narrow deterministic estimator, and the OOS headline is a single selected path per window. Fix the reward balance and the validation estimator first; re-test the "overfitting cliff" against that; only then decide whether the architecture deserves a larger universe. The codebase is a strong foundation — provided the next investment goes into *validating the method*, not *scaling the menu*.
