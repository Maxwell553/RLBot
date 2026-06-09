# RLBot Comprehensive Project Review — 2026-06-09

**Branch/commit reviewed:** `main` @ `9687454` ("rebalance rewards+add benchmarks")
**Method:** six parallel deep-read reviews (data pipeline, environment/reward, training/config, backtest/inference, auto-research loop, tests/docs), each tracing code line-by-line; findings marked **VERIFIED** were confirmed by tracing or execution, others marked *suspected*. Test suite executed on this machine (torch-free): **108 passed, 1 skipped, 1 failed** (the failure is `test_finetune_and_resume_mutually_exclusive`, which requires torch before reaching the behavior it tests — a test bug, not a code bug).

---

## Executive summary

The project is in substantially better shape than the historical reviews in `docs/` describe. The core anti-leakage architecture is sound and verified: chronological holdout is reserved before any split, every feature computation is strictly causal, execution timing (`close[t−obs_lag]` → fill `open[t+1]` → MTM `close[t+1]`) has no off-by-one, VecNormalize freeze discipline is enforced with defense in depth, and the run-snapshot/manifest/hash provenance system is genuinely strong. The recent fix sprint (438d5c8, 9687454) landed its headline fixes correctly — best-model/VecNormalize pairing, effective-N data snapshots, eval-side fee curriculum, and the `fee_ramp_end` best-model gate are all real and regression-tested.

However, the review found **one critical bug that invalidates the auto-research loop's core use case**, plus a cluster of high-severity issues in exactly the places the project wants to go next:

1. **(CRITICAL) Config does not propagate to SubprocVecEnv workers.** Training envs are constructed in spawned worker processes where the config singleton is unset, so they fall back to the default `config/config.yaml`. Every `--config` variant run — i.e., **every `research.py` reward/environment ablation** — trains under the *default* reward/cost/env config while backtesting under the *variant* config. The shipped `reward_ablation.yaml` spec would produce 18 identical training runs evaluated under 18 different configs.
2. **(HIGH) The new `benchmark_relative_max_share` reward cap creates a verified reward-hacking gradient** — when the capped benchmark term is positive, burning transaction costs *increases* total reward (net +2000·cost per unit burned, numerically verified).
3. **(HIGH) Prices are not dividend-adjusted** (`auto_adjust=False`, `Close` not `Adj Close`) — bonds/equity-ETF total returns are systematically understated; every absolute and cross-asset result is biased.
4. **(HIGH) The OOS firewall is an honor system in three places**: `promote` never records its holdout read (unlimited re-reads), spec `windows` can move the holdout freely, and `base_config` is unrestricted.

None of these are visible from the test suite, which is good at what it covers but has no train/backtest end-to-end execution. The forward plan (§5) sequences: fix correctness → make the firewall real → add statistical decision rules and a holdout-burn ledger → orchestrate compute → close the loop with agent-proposed experiments and a tier-5 shadow ledger.

---

## 1. Current state

### What exists and works

- **Library** (`rlbot/`, ~5,300 lines): data pipeline with cache + causal features (`data_utils.py`), gym env with curriculum/DR/reward decomposition (`trading_env.py`), frozen-dataclass config (`rl_config.py`), shared cost-model baselines (`baselines.py`), run artifacts/provenance (`run_artifacts.py`), inference loading with VecNormalize freezing (`inference_load.py`, `vecnorm_utils.py`), block-bootstrap stats (`stats.py`), research spec/gates/registry/report (`research/`).
- **CLIs** (`scripts/`, ~5,400 lines): `train.py` (RecurrentPPO + curriculum + eval-NAV best-model selection), `backtest.py` (manifest-defaulted OOS windows, run-local config/data binding, detailed baselines), `research.py` (plan/launch/collect/report/promote), `infer_weights.py` (audited target weights), `modal_app.py` (cloud training), `run_seed_ensemble.sh`.
- **Tests:** 110 collected; strong on action-mapping invariants, feature-split leakage semantics, reward mechanics, config binding, baseline accounting, research firewall units, and an unusual doc-drift tripwire (`test_docs_invariants.py`).
- **CI:** `.github/workflows/ci.yml` exists — single torch-free job, Python 3.11, hardcoded list of 12 test files (excludes `test_publication_fixes.py` entirely).
- **Docs:** README/RESEARCH/TRAINING/config-README numeric claims were spot-checked **clean** against code after the 9687454 doc sweep — the doc-drift problem of earlier cycles is largely fixed. Five historical external reviews carry maintained "historical snapshot" banners.
- **No published OOS results** under the current pipeline (honestly disclosed everywhere); no runs in `Runs/` on this machine.

### Recent history

`5facc99` merged the evolution roadmap (harness, measurement, auto-research, inference). `438d5c8` fixed the VecNormalize/best-model pairing, run-cache snapshots, and added `baselines.py` + publication-fix regression tests. `9687454` rebalanced the reward (added the `benchmark_relative_max_share` cap), extended the fee curriculum to eval envs, added the `fee_ramp_end` best-model gate, and swept the docs.

---

## 2. What is done well

### Data pipeline & leakage prevention (the core design constraint) — verified sound

- **Holdout reservation order is correct**: `reserve_chronological_holdout` runs on the raw panel (`scripts/train.py:881-895`) *before* `train_test_split_alternating` (`:934-948`); only `backtest.py` consumes the tail. Empty-set and ordering violations raise (`data_utils.py:719-741`); explicit-date mode supports an embargo gap excluded from both sets.
- **Every feature is strictly causal** — traced individually: RSI (trailing ewm, `data_utils.py:189-196`), MACD (`:199-202`), fracdiff (`np.convolve` full-mode prefix, `:155-162`), trend EMA distance (`:395-404`), realized vol on `[t−lookback, t]` (`:365-375`, regression-matched to the env's on-the-fly computation in `test_core.py:244-270`). Building the feature cache before the holdout cut therefore does not leak.
- **Feature-split modes are well designed and well tested**: `independent` recomputes per segment + neutralizes `feature_purge_warmup` bars; `continuous` slices one global panel; precomputed-feature passthrough is all-or-nothing and `independent` deliberately ignores precomputed continuous features (`data_utils.py:879-882`). `test_feature_split_modes.py` pins all of it.
- **Pre-IPO ordering is right**: live mask computed from `notna()` *before* bfill (`data_utils.py:632-640`), so pre-IPO bars are non-live; the env both observes and gates on the mask.
- **HY OAS proxy calibration is expanding/causal** (`_calibrate_hy_proxy_expanding`, `data_utils.py:249-281`) and directly tested.
- **Cache schema evolution handled defensively** (`load_cache`, `data_utils.py:1135-1195`): recompute-on-missing, width validation with actionable errors.

### Environment & execution — verified sound

- **Causal execution timing has no off-by-one**: NAV telescopes correctly across steps; `_min_t = lookback + max_obs_lag` plus segment-aware episode placement guarantees observations never cross block joins (`trading_env.py:372, 459-483`).
- **One shared friction model** (`portfolio_step_nav`, `baselines.py:58-127`) used by the env's reward benchmark, all passive baselines, and the backtest comparators — no drift between reward-time and report-time accounting. Benchmark friction follows the live `fee_scale`, keeping the excess signal fair across the curriculum.
- **Action→weights mapping is robust**: live-mask before cap, 5-iteration clip-and-redistribute with a guaranteed final projection (`trading_env.py:83-121`), property-tested across caps × N ∈ {5,10,23,55}. EMA smoothing inside the env means train/eval/backtest see identical executed weights, and `info["target_weights"]` is the executed target (regression-tested).
- **Reward decomposition observability**: per-term `rew_decomp/*` emitted every step, NaN-safe accumulation, terms sum exactly to the scalar reward.

### Training pipeline & provenance — strong design

- **Config as single source of truth**: frozen dataclasses, fail-fast required keys, per-asset list lengths validated against the *loaded panel* (`train.py:835`), careful `--n-assets` slicing with re-parse, correctly ordered two-pass argparse (in the main process).
- **Best-model/VecNormalize pairing is genuinely fixed**: `best/vec_normalize.pkl` saved in the same eval-callback branch as `best_model.zip` (`train.py:268-273`); the exit-time persister deliberately does not overwrite it; regression test pins it. Selection by eval ending NAV (not reward) resists passive reward hacking; the `fee_ramp_end` gate resets `best_mean_nav` when it opens so frictionless pre-gate NAVs can't suppress saves.
- **Manifests/provenance**: pre-train manifest with the `chronological_holdout` block, `merge_manifest` preserving it post-train (tested); config SHA-256 of the *effective* config, data-snapshot file hash, git commit + dirty flag; machine-readable `training_summary.json` for the registry.
- **Curriculum**: fraction-of-budget milestones with interpolation between budget anchors; fee and churn share one ramp; 9687454 correctly extended fee/churn to eval envs (DR stays train-only) so eval NAV is measured under the training friction regime.
- **Modal integration**: SDK-free sync path, GPU profiles deriving env/batch geometry, volumes mounted to the same relative layout so `RunPaths` works unchanged, crash-safe `finally` that persists weights/stats and commits volumes.

### Backtest & inference — right contracts

- **Manifest-defaulted windows**: a manifest is *required*; `train_end`/`holdout_start`/`holdout_end`/`holdout_days`/`--until` default from it. Run-local config snapshot and run-local `data_cache.npz` bind by default; `_assert_manifest_panel_compatible` (`backtest.py:240-266`) catches ticker/width drift; batch mode rebinds config per run to avoid cross-run bleed.
- **VecNormalize freeze with defense in depth**: single chokepoint (`vecnorm_utils.py:16-26`); the rollout steps the raw env and only calls `normalize_obs` (a pure transform), so OOS data cannot update stats even if the freeze were bypassed; OOS requires VecNormalize by default.
- **Drift-detectable summaries**: `backtest_summary.json` carries config hash, data-cache hash, split mode, git provenance — directly comparable to the training manifest.
- **Detailed baselines through the same cost model**: cash, benchmark, EW daily/monthly, 60/40 (gracefully degrades on small universes), risk parity, subperiods, block-bootstrap Sharpe CI (a correct Politis–Romano stationary bootstrap, seeded).

### Auto-research skeleton — right architecture

- **Deny-by-default patch allow-list** enforced at three layers (spec construction, variant build, eager round-trip through the real parser at `plan` time); no key injection possible (`set_nested` requires existing paths); fail-closed tier table; tier ≥4 hard-requires `--promote` checked before any subprocess; per-variant repeat-OOS guard; the orchestrator never hand-passes holdout dates (inherits the manifest machinery).
- **Registry as aggregator, not metric source** — flattens manifest/training_summary/backtest_summary with strong provenance; append-only JSONL; idempotent collect; `run_id == variant_id` makes provenance joinable.
- **Honest posture**: shipped specs are dev-tier with explicit hypotheses; RESEARCH.md names multiple-testing bias as a known unguarded risk.

### Tests & docs

- The doc-drift tripwire (`test_docs_invariants.py`) — asserting obs dims, cap values, layout claims, and byte-identity of CLAUDE.md/AGENTS.md bodies — is an unusually good guard. The 9687454 doc sweep left README/RESEARCH/TRAINING numerically consistent with code (every spot-check passed except items noted in §3).

---

## 3. Issue register (prioritized)

### CRITICAL

**C1. Config singleton does not propagate to SubprocVecEnv workers — `--config` and `--n-assets` build training envs from the default `config/config.yaml`.** VERIFIED (traced + factory inspected).
`MultiAssetPortfolioEnv.__init__` calls `get_config()` (`trading_env.py:243`), and construction happens inside SubprocVecEnv worker processes (`train.py:1140, 1166`). SB3 spawns/forkservers workers (never fork), so modules re-import fresh, `rlbot.rl_config._CONFIG` is `None`, and `get_config()` falls back to `DEFAULT_CONFIG_PATH` (`rl_config.py:522-526`). `_make_env_factory` (`train.py:124-161`) captures no config. Consequences:
- **The research A/B loop is silently invalid for `reward.*`/`environment.*`/cost-relevant patches**: `research.py` launches `train.py --config <variant.yaml>`, but worker envs train under the default config. `backtest.py` builds its env in-process *after* `set_config(load_config(snapshot))`, so the variant config *does* apply at backtest → a train/backtest mechanics mismatch dressed up as an experiment result. `specs/reward_ablation.yaml`'s variants would all train identically.
- `--n-assets N` (N≠10) crashes loudly at env spawn (default 10-wide cost arrays vs N-wide pack) — the advertised feature cannot work with subprocess training.
- Values that *do* propagate (passed explicitly through the factory): `max_episode_steps`, `obs_noise_std`, `obs_lag_default`, `inactivity_penalty_scale`. Values that don't: all `reward.*`, all `transaction_costs.*`, `max_single_asset_weight`, `action_smoothing_alpha`, `stop_loss_fraction`, DR fee bounds, `min/max_obs_lag`.
- Default-config runs (editing `config/config.yaml` in place) are unaffected — which is exactly why this went unnoticed.
**Fix:** capture the effective config (raw dict or snapshot path) in the `_init` closure and `set_config` inside the worker before env construction; add a regression test booting a 2-worker SubprocVecEnv with a patched `reward.reward_scale` and asserting the worker-side captured value.

### HIGH

**H1. `benchmark_relative_max_share` cap creates a reward-hacking gradient ("burn costs to raise the cap").** VERIFIED numerically. (`trading_env.py:124-152`, applied at `:968-976`; introduced in 9687454.)
The cap sets `max_bench_abs = share·other_abs/(1−share)` where `other_abs = |return|+|participation|+|inactivity|+|churn|`. When the benchmark+Sortino term is capped and positive (common: raw bench can reach +31.5 and the cap is ~1.5–4.5 on quiet days), every other term's *magnitude* feeds the reward with coefficient +1.5: burning tx cost `c` nets **+2000c reward** (−2000c return, −2000c churn, +6000c cap increase); holding penalized cash has net derivative +0.5; bigger losses locally increase reward. Over a 65M-step run PPO should be expected to find this. **Fix:** replace the relative cap with fixed absolute clips (which already exist) or cap against a constant budget, never a function of the other terms' magnitudes. Add a regression test that increasing churn at fixed everything-else never increases total reward.

**H2. Prices are split- but not dividend-adjusted — systematic total-return bias.** VERIFIED (code path). (`data_utils.py:483, 486` `auto_adjust=False`; `:579` uses `Close`.)
SPY (~1.5–2%/yr), IEF (coupon is most of its return), EEM etc. all understate returns; agent, benchmark, and baselines all train/evaluate on price returns mislabeled as investable returns, with asset-heterogeneous bias (ETFs vs price indices vs futures). First-order correctness problem for any published number. **Fix:** `auto_adjust=True` (or `Adj Close`) for distributing symbols; document per-asset return type; refresh caches.

**H3. The OOS firewall is bypassable in three ways.** VERIFIED.
- **`promote` never records its holdout read** (`research.py:231-246`): no registry append, and the variant's existing record says tier 3, so `assert_no_repeat_oos` passes on every subsequent promote — unlimited holdout re-reads; the tier-4 result never enters the registry. Collect can't fix it (skip-by-run_id).
- **Spec `windows` are free-form** (`spec.py:81`; passed straight to `train.py` at `research.py:117-121`): any spec can move/shrink/shift the holdout — the one thing the patch firewall exists to protect. No canonical window registry exists in code (the walk-forward table is prose in `docs/RESEARCH.md`).
- **`base_config` is unrestricted** (`spec.py:77`): the firewall validates the patch *relative to the base*, so a doctored base with different costs/universe/split sails through. No hash-match against the repo config.
Also: the firewall lives only in the orchestration layer — direct `backtest.py --holdout-start …` CLI overrides are invisible to any accounting.

**H4. `infer_weights.py` — the closest thing to a production path — is the least guarded.** VERIFIED.
(a) Runs silently without VecNormalize if the pkl is missing (`infer_weights.py:119`) — backtest treats this as a hard error; (b) for `--checkpoint best` with missing `best/vec_normalize.pkl` it silently substitutes end-of-run stats (`:42-44`) — exactly the mispairing the docs warn about; (c) no `_assert_manifest_panel_compatible` equivalent: `resolve_panel_tickers` prefers manifest tickers unconditionally, so a reordered/different cache of the same width emits **weights mapped to the wrong tickers** with no error.

**H5. Research report seed aggregation is structurally broken.** VERIFIED. `resolve_variants` bakes the seed into `variant_id` (`spec.py:139`) and the report groups by `variant_id` (`report.py:62-66`) — every group has exactly one record; "median across seeds" is a median of one. The loop's stated purpose (seed-robust comparison) cannot be delivered. Additionally, tier-3 reports show dashes in all metric columns because `best_eval_nav` — the one signal tier-3 runs have — is not a column.

**H6. DR `obs_lag` is randomized full-wide from step 0, then snaps to a cliff at `fee_ramp_end`.** VERIFIED. `_dr_bounds` (`train.py:527-530`) returns full bounds for `t < fee_ramp_end` (only the middle window is progressive), while `reset()` always samples DR lag for train envs. Net: full lag randomization during the frictionless phase, a hard narrowing discontinuity exactly at the step the best-model gate opens, then re-widening. Contradicts the documented "randomized after the fee curriculum releases."

**H7. `--resume` schedule incoherence.** VERIFIED against SB3 semantics. With `reset_num_timesteps=False`, SB3 adds the new budget to the absolute counter: resuming a 65M run from 30M with `--timesteps 65000000` trains to 95M; entropy decay and cosine LR are progress-based and shift (LR jumps back *up* at resume), while curriculum milestones are absolute — no value of `--timesteps` satisfies both. Also (M7 below) the resume VecNormalize fallback for `best_model.zip` silently picks end-of-run stats over the adjacent `best/vec_normalize.pkl` (`train.py:1226-1231`).

### MEDIUM

**M1. Crashed research runs are registered `status: "ok"` with null metrics, then block the corrected record.** VERIFIED. Manifest is written pre-training, collect defaults `status="ok"` (`registry.py:42`), the failures list is printed and discarded, and the run-id lands in `seen` so a successful re-train can never append the corrected record (`research.py:200-217`). No cohort resume either — re-running `launch` re-trains and overwrites every variant.

**M2. Episode-start benchmark asymmetry.** VERIFIED. At reset the benchmark starts already-invested (earns overnight return, zero entry cost) while the agent starts 100% cash and pays full entry friction (`trading_env.py:796-799`, `baselines.py:89-90`) — a systematic negative bias in the excess/Sortino signals for the first ~20–63 bars of *every* training episode.

**M3. Stale-cache config drift.** VERIFIED. The cache stores no `since`/`until` metadata; changing `data.since` or `data.fracdiff_d` without `--refresh-data` is silently ignored, and the run snapshot *labels* stale-d fracdiff arrays with the new d (`train.py:872`); in `independent` mode train features use the new d while the snapshot/OOS features carry the old d.

**M4. Default `independent` training vs `continuous` OOS backtest features = train/inference distribution shift.** VERIFIED semantics. Cold-started segment features (fracdiff is effectively expanding-window — see M5 — and trend uses EMA100) vs fully-warmed OOS features; `feature_purge_warmup=25` is far shorter than the indicators' memory. Not leakage, but model selection happens on a feature distribution that differs from inference, and it muddies the `feature_split_ab` experiment's attribution.

**M5. Fracdiff weight cap always binds.** VERIFIED empirically. For d=0.4 the 50,000-term cap binds long before `weight_eps=1e-14` (last |w| ≈ 7e-8) — fracdiff is effectively expanding-window, scale drifts with position-in-segment, `weight_eps` is dead code, and per-segment recompute is ~100× more expensive than a meaningful truncation (~1e-4 → a few hundred bars).

**M6. Curriculum/decomp callback cadence is in vector steps, not global timesteps.** VERIFIED. `n_calls % 50_000` (`train.py:565-568`) = 800k global steps at 16 envs, 3.2M at the H100 profile's 64 — stair-step ramps whose granularity varies by GPU profile (eval/checkpoint frequencies are correctly divided by `n_envs`; these aren't).

**M7. VecNormalize fallback chains pick end-of-run stats without warning** — both train-resume (`train.py:1226-1231`) and backtest legacy-run path (`backtest.py:867-874`). Mirror of the bug 438d5c8 fixed, surviving in the fallbacks.

**M8. Tail-based (`holdout_days`) runs can silently backtest a shifted OOS window.** VERIFIED. The window is recomputed from the current cache's `index[-1]` (`data_utils.py:727-732`); the manifest's realized `date_start`/`date_end` are recorded but never cross-checked (`backtest.py:363-422`). One assertion closes it. Related: partial CLI window overrides mix silently with manifest values; `--obs-lag` is *not* defaulted from the manifest (hard default 1) unlike every other window flag.

**M9. `best_eval_step` provenance is wrong under the gate.** VERIFIED. `best_eval_nav` is post-gate but `best_eval_step` is argmax over the full nav history including pre-gate (reduced-friction) evals (`train.py:1387-1396`) — it usually points at an eval that did not write `best_model.zip`.

**M10. `early_stop_patience` is dead under default config.** VERIFIED arithmetic. `curriculum_end_step = min(65M, 71.5M) = 65M` — the whole budget; the patience branch can only fire at the final eval. Same arithmetic means the advertised post-DR "full randomization" phase never occurs.

**M11. Run-id reuse (incl. Modal retry) poisons best-model tracking** via stale `eval_nav_history.npz` (`train.py:213-229`): the dead attempt's `best_mean_nav` suppresses saves and the gate-reset is skipped.

**M12. Pre-IPO/gap `bfill` injects future price levels into observations.** VERIFIED. Pre-IPO bars carry fracdiff of the future first-listing price (partial weight sums 0.06–0.18, computed); mid-history gaps >5 days get future prices visible cross-asset (`data_utils.py:639-641`); HY OAS bfills a 2007 level into 2005–2007 macro obs (`:328`). The live mask blocks trading but not observation. Also the `fetch_aligned_daily` docstring claims "no backward-fill from IPO levels" — contradicted by the code it documents.

**M13. Batch backtest policy-shell reuse swaps only the state dict** (`backtest.py:202-209`): parameter-free architecture differences (activation_fn, ortho_init) leak from run 1 to runs 2..N in a mixed-policy batch. Suspected (not observed); key the shell by a policy-kwargs hash.

**M14. No multiplicity correction anywhere in the research loop**, no cross-cohort holdout-reuse accounting (all three shipped specs share one holdout), and `success_gates`/`budget` are dead fields — "did the variant win?" is eyeballed. A tier-4 grid launch performs one holdout read per variant under a single `--promote`.

**M15. `--backend modal` is a dead flag** (`research.py:256`; never read) — every research launch is sequential local despite the Modal stack existing. `run_seed_ensemble.sh` is sequential and aborts the cohort on first failure.

### LOW (selected; full details in agent traces)

- `--stochastic-paths` truncates the deterministic NAV array before headline metrics if any path stops early (`backtest.py:613-620`) — same checkpoint, different reported numbers depending on a diagnostic flag. Stochastic ensemble torch sampling is unseeded (not reproducible).
- Reported `n_bars` overstates evaluated bars (lookback trim + possible early stop-loss termination not flagged) (`backtest.py:708`).
- Ensemble "latest" ≠ batch "latest" (final.zip vs newest step checkpoint), ensemble path skips the OOS-touch warning, and `ensemble_summary.json` has no provenance hashes.
- `freeze_vec_normalize_for_inference` forces `norm_obs=True` unconditionally (`vecnorm_utils.py:24`) — wrong if a run trained with `norm_obs: false`.
- `portfolio_weights_from_action` reads the live global config per call (`trading_env.py:90`), violating the captured-config invariant the docs state; DR bound sampling does too.
- Reward-decomp `abs_share` double-counts the drawdown term in its denominator (`reward_logging.py:15-23`) — the balance dashboard is biased exactly during drawdowns.
- Negative cash from holding costs flips the inactivity penalty into a tiny bonus (unclamped `cash_frac`, `trading_env.py:952-957`).
- First env segment double-pays the lookback margin (22 bars wasted); segment margin uses `lookback` rather than `max(lookback, max(RETURN_HORIZONS))` — a latent leak if anyone configures `lookback < 20`.
- README mis-states the cap semantics: code allows bench mass = `share/(1−share)` = **150%** of other mass at 0.6, not "≤ 60% of other mass" (`README.md` vs `trading_env.py:148`).
- CLAUDE.md/AGENTS.md still claim `.gitignore` covers the legacy artifact roots — 9687454 deleted those lines; "minimal test run needs gymnasium+pandas+numpy" omits PyYAML; the library list omits `rlbot/visualize.py`.
- CI runs a hardcoded 12-file list that excludes `test_publication_fixes.py` (7 of its 8 tests are torch-free, including the two headline regression guards); new test files are silently un-CI'd; `test_finetune_and_resume_mutually_exclusive` fails without torch because train.py imports torch before the check (move the check above the import, or skipif).
- Modal: `N_STEPS=4096` constant duplicates `hyperparameters.n_steps`; `extract_run_id --window` derives the id from *local* `Runs/` while the remote generates its own; `PYTHONHASHSEED` set at runtime doesn't affect the current process.
- Registry: no file locking, `read_records` bricks on a torn line, TOCTOU on concurrent tier-4 launches, `or`-coalescing treats 0/"" as missing, `promote` re-materializes and silently rewrites `cohort.json` from the (possibly edited) spec.
- `_short()` 16-char grid-value truncation can collide variant ids → silent run-dir overwrite.
- Dead code: `self._prev_target_w` (env), entropy floors that never bind, `warmup_improvements`, `--fast` in the `--run-ids` path.

### Test gaps (no coverage at all)

`reserve_chronological_holdout` (the single most load-bearing function — zero direct tests), `freeze_vec_normalize_for_inference`, any train smoke (all four callbacks, curriculum schedule math), any backtest end-to-end, obs_lag causal indexing, `training.reproducible` twin-run equality, resume/finetune behavior, Modal path, `research.py launch/collect` orchestration, `inference_load.py`.

---

## 3b. The stranded fix branch: `feat/evolution-roadmap` @ `c761886`

A prior fix sprint ("Seal the research/measurement harness and fix reward/split-mode correctness", +2,203/−196 across 34 files, 6 new test files) sits on `feat/evolution-roadmap`, **one commit ahead of a base that is three substantive commits behind main** (the branches diverged at `ca3c4a5`; main gained `20d92ba`, `438d5c8`, `9687454` independently). Fifteen files overlap between the two lines of development — including `trading_env.py`, `train.py`, `backtest.py`, `data_utils.py`, and `config.yaml` — and both sides reworked the reward, so a port will have real conflicts.

**Findings from §3 that `c761886` already fixes (verified in its diff):**
- **H3 — all three firewall holes**: `oos_read_attempt` registry records written *before* any tier-4 backtest (crashed reads still gate; `failed` records don't block corrections), spec windows pinned to a `CANONICAL_WINDOWS` table (W1–W6), `base_config` pinned to `config/config.yaml`, plus an `--oos-budget` cap (default 1) on tier-4 reads per launch — this also covers part of M14's multiplicity accounting.
- **H4 / M8 / M7 (backtest side)**: missing VecNormalize on `norm_obs` runs hard-fails with an explicit `--allow-raw-obs` escape hatch; `hash_drift` comparison vs the training manifest; `--obs-lag` defaulted from the manifest; loud warning on global-cache fallback; ensemble honors `--detailed`.
- **M12**: per-asset features neutralized on pre-live bars (pre-IPO fracdiff price leak closed); HY-OAS ffill-only; delist-safe liquidation.
- **M4**: `independent` split mode gains a causal preroll (`data.feature_preroll_bars: 252`; full preroll equals continuous bit-for-bit) — removing the train/inference feature distribution shift.
- **M3 (partial)**: `load_cache` validates `fracdiff_d`. **M11**: train.py refuses run-dir reuse without `--overwrite-run`. Sortino floor exploit closed (`reward.sortino_downside_floor: 0.001`).
- **Test gaps**: new `test_holdout_reservation.py`, `test_vecnorm_freeze.py`, `test_causal_execution.py`, `test_reproducible_seeding.py`, `test_reward_terms.py`, `test_research_cli_gates.py` (the suite there is 150 passed / 3 torch-gated skips), and CI switched to wholesale `pytest tests/`.

**Fixed nowhere (neither branch):** C1 (worker config propagation — the critical), H1 (the reward-cap hacking gradient was *introduced* by `9687454`, which postdates the branch), H2 (dividend adjustment), H5 (report seed grouping — partially improved on the branch), H6 (DR lag cliff), H7 (resume incoherence), M1, M2, M5, M6, M13, M15.

**Recommendation:** the fastest sound path through Phases 0–A below is **port `c761886` onto current main** (rebase or re-apply hunk-by-hunk, reconciling the two reward reworks — the branch's Sortino-floor fix and main's `benchmark_relative_max_share` cap both touch the same term math), then fix the remaining main-only items (C1, H1, H2) on top. Re-deriving the branch's fixes from scratch would waste ~2,200 lines of reviewed, tested work; merging it blindly would silently revert main's three commits' worth of changes in the 15 overlapping files.

---

## 4. Improvement opportunities

Beyond fixing §3, ranked by leverage:

### Measurement integrity
1. **Total-return data** (H2) — the highest-value single data change; it moves every number.
2. **Window cross-check assertion** (M8) — after `resolve_oos_holdout`, assert `test_idx[0]/[-1]` against the manifest's recorded dates; warn whenever any CLI window/obs-lag flag differs from the manifest.
3. **Benchmark-relative significance**: block bootstrap on *paired daily excess* log-returns vs EW/benchmark (the baselines are already computed in `_print_detailed_stats` — the data is in hand); add Sortino/Calmar/turnover/gross-exposure to `backtest_summary.json` so the registry can gate on them; vectorize the bootstrap (~100× speedup, makes benchmark-relative variants free).
4. **Deflated Sharpe / PSR** (Bailey–López de Prado) using trial counts from the registry — the statistically honest headline metric for a many-variants-one-holdout shop.

### Reward & env realism
5. **Replace the relative reward cap** (H1) with fixed absolute clips or a constant budget; consider a differential Sortino (Moody–Saffell) instead of the 63-bar trailing window (O(1), stationary, no per-episode dead zone).
6. **Benchmark replicability**: cap weights put 0.55 on SP500 while the agent's per-asset cap is 0.25 — the agent structurally cannot match its own benchmark in SP500-led rallies; project the benchmark onto the same cap or report both.
7. **Risk-free accrual on cash** (TNX is already in the panel; at 2022–24 rates a ~5% cash drag distorts the cash-vs-invest tradeoff and inactivity calibration) and **market impact/capacity** (volume is in the panel, unused: square-root impact `k·σ·√(trade/ADV)` + participation cap gives `infer_weights` a defensible capacity story).
8. **Expose DR context** (`fee_scale`, `obs_lag`) in the observation — standard contextual-RL practice; the LSTM currently has to infer episode-level latents.
9. **Fixed-width FFD fracdiff** (M5) + size `feature_purge_warmup` to actual indicator memory; mask per-asset features wherever `asset_live==0` (closes M12 cleanly).

### Engineering hygiene
10. **CI**: run `pytest -q` over the whole suite (drop the hardcoded list), add a torch CPU job with a tiny-budget smoke (`--timesteps 2048 --n-envs 2` synthetic train → backtest → infer_weights), add ruff, pin SB3/sb3-contrib/torch versions (the determinism and resume semantics are version-sensitive).
11. **Tests for the load-bearing invariants**: holdout reservation, VecNormalize freeze, obs_lag causality on a synthetic ramp, curriculum values at known steps, reproducible-mode twin runs, and the C1 worker-config regression.
12. **Doc generation**: the reward/curriculum tables duplicate `config.yaml` by hand in 3+ places — generate the numbers from `RLConfig`; extend `test_docs_invariants.py` to assert doc-quoted constants and the `.gitignore` claims.
13. **Archive the five historical review docs** to `docs/archive/` with pointer lines; keep the publication-readiness and roadmap-progress docs as the live status pair.

---

## 5. Forward plan: toward a robust continuous self-improvement environment

The auto-research skeleton (allow-list + tiers + append-only registry) is the right architecture. What exists today is a **single-shot sweep runner with an honor-system firewall**; the goal is a **closed loop with budgets, decision rules, and memory**. Sequenced phases, each independently shippable:

### Phase 0 — Correctness gate (days; everything else is built on this)
0. **Port `c761886` onto main** (§3b) — recovers the firewall sealing, backtest/infer guards, pre-live neutralization, causal preroll, and six test files in one move; reconcile the two reward reworks during the port.
1. **Fix C1** (worker config propagation) + regression test. *Prerequisite for trusting any research result.*
2. **Fix H1** (reward cap hacking gradient) + monotonicity regression test.
3. **Fix H2** (dividend adjustment) + cache refresh.
4. **Fix H4** (infer_weights guards: require VecNormalize, panel compatibility assert, hash model/vecnorm files into provenance).
5. Quick wins from M-list: window cross-check assertion (M8), `--obs-lag` from manifest, VecNormalize fallback-order fixes (M7), `best_eval_step` post-gate filter (M9), DR pre-ramp bounds (H6), CI whole-suite run.
**Exit criterion:** a fresh seed-ensemble training run + OOS backtest on windows 1–2 produces the project's first publishable baseline numbers under the fixed pipeline. Until then, do not run experiment cohorts — they would be invalidated by C1 anyway.

### Phase A — Make the OOS firewall real (days; items 6–8 are largely delivered by the §3b port — verify against this list after porting)
6. **Record every holdout read at the moment it happens**: `promote` and tier-4 `launch` append registry records (tier 4, status, metrics) immediately after the backtest returns; make `collect` upsert-by-(run_id, tier) instead of skip-by-run_id (also fixes M1's blocked corrections).
7. **Canonical window registry**: move the walk-forward table into `config/windows.yaml`; `load_spec` validates `spec.windows` against it (named references, e.g. `window: w4`); reject free-form holdout dates the same way forbidden patch keys are rejected.
8. **Pin `base_config`** by hash to the repo config or a registered parent cohort's materialized variant; record `base_config_hash` + `spec_sha256` in every record.
9. **Failure records + cohort resume**: append `status="failed"` inline; `launch --resume` skips variants with a completed `training_summary.json`; advisory flock on registry appends; skip-with-warning on torn lines.
10. Strip the seed from the report grouping key (H5) and add `best_eval_nav` columns so tier-3 reports are usable.

### Phase B — Statistical decision rules + holdout-burn ledger (the "when does a variant win?" layer)
11. **Global OOS ledger** (`Runs/oos_ledger.jsonl`) keyed by (holdout window, data-cache hash), written by **`backtest.py` itself** so even manual backtests are counted — closing the orchestration-layer-only hole. Each window gets a configurable read budget; `gates.assert_tier_allowed` consults the ledger and refuses when spent; every report header shows the burn count.
12. **Implement `success_gates` as a real gate engine** at collect time: thresholds on mean post-gate `best_eval_nav` across seeds, max-DD, seed dispersion, reward-decomp balance (aggregate `eval_logs/reward_decomp.json` into the registry). Cohort verdicts (`pass/fail/inconclusive` per grid combo) become registry records.
13. **Pre-registered promotion rule**: tier-4 promotion requires (a) K ≥ 3 seeds at tier 3, (b) a pre-declared one-sided comparison vs an explicit control (wire the currently-decorative `parent` field), (c) the OOS read reports deflated Sharpe using the ledger's trial count plus the existing bootstrap CI.
14. **Reserve an embargoed final window** (e.g. window 6) excluded from `config/windows.yaml` entirely — touchable only by the future tier-5 path.

### Phase C — Compute orchestration & throughput (unlocks experiment volume)
15. **Wire `--backend modal`**: route `_train_cmd` through the existing Modal broker; enforce `budget.max_modal_hours` as a hard cap; fix the run-id derivation (L: local-vs-remote id drift) and derive Modal batch geometry from the parsed config's `n_steps`.
16. **Experiment queue**: `Runs/queue/` of pending specs + `research.py run-queue` worker (pull → plan → launch → collect → report) with per-spec budgets and parallel variants (runs are already independent processes). Parallel multi-seed via Modal fan-out replaces the sequential shell script.
17. **Cheap-proxy laddering**: formalize tiers 1–2 as successive halving — grid specs auto-run short-budget tier 1 on all combos, promote the top fraction to full tier-3 seeds. The tier vocabulary already exists; only the scheduler is missing. (Cadence normalization M6 matters here: curriculum behavior must be invariant across GPU profiles for tier-1 results to predict tier-3.)

### Phase D — Memory & meta-learning (the registry becomes the system's memory)
18. **Global registry view**: `research.py report --all` aggregating every cohort, grouped by knob, with lineage from `parent`. Today each cohort is an island.
19. **Knob-sensitivity summaries**: per allowed config key, the distribution of eval-NAV deltas vs control across all historical cohorts — exactly the input a hypothesis-proposing agent needs.
20. **Registry-driven docs**: generate the RESEARCH.md results tables from the registry (the roadmap doc already proposes this).

### Phase E — Agent-driven research + the only leak-free evaluator
21. **Spec-proposer interface**: the spec YAML *is* the agent API — keep it that way. Add `research.py validate <spec>` (schema + firewall + window + budget checks, no side effects) as the agent's fast feedback loop; require non-empty `hypothesis`, `parent`, and `success_gates` for agent-submitted specs.
22. **Hard guardrails for autonomous mode**: `--promote` stays human-only; refuse launch when `git_dirty` (already recorded, never gated); the Phase-B ledger budget is the backstop against agent-driven holdout mining.
23. **Tier 5 — shadow trading**: a daily cron running `infer_weights.py` against fresh data, appending to a shadow ledger with next-day realized-vs-modeled reconciliation (under the gitignored `execution/` root per repo convention). This is the only evaluation that doesn't burn holdout, which makes it the natural terminal arbiter of a continuous loop — and it exercises the whole inference path (H4 fixes) daily. Add an observation-drift alarm (current obs z-scores vs frozen `obs_rms`) as a cheap regime/staleness check.

### Why this order

The single highest-leverage pairing is **Phase 0 item 1 + Phase B item 11**: until variant configs actually reach the training envs and every holdout read is recorded and budgeted, every other improvement just makes it faster to produce invalid results or overfit the holdout. Phases C–E are deliberately last: throughput and autonomy amplify whatever the measurement layer is — they must amplify a sound one.

---

## Appendix: verification status

- Test suite on this machine (torch-free): 108 passed, 1 skipped, 1 failed (`test_finetune_and_resume_mutually_exclusive` — requires torch before reaching its check; test-design bug).
- Findings labeled VERIFIED were confirmed by line-level tracing; H1's reward gradient and M5's fracdiff cap were additionally confirmed numerically; C1 was confirmed by direct inspection of `_make_env_factory` plus SB3 SubprocVecEnv start-method semantics.
- Six subsystem review transcripts (data, env/reward, training/config, backtest/inference, auto-research, tests/docs) underlie this synthesis; file:line references throughout point at `main` @ `9687454`.
