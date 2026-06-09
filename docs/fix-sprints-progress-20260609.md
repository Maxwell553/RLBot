# Fix-sprint implementation log — 2026-06-09

Implements §5 of `docs/claude-review-20260609.md` (all four sprints). Test suite: 94 → **150 passed / 3 skipped** (torch/SB3-gated; they run in the new manual CI torch job). Torch-free suite still ~2 s.

## Sprint 1 — research firewall (P0 + P1)

- **OOS reads are recorded at read time.** `research.py promote` and tier-4 `launch` append a `status: oos_read_attempt` registry record *before* the backtest subprocess starts, and a scored record after. A crash between read and scoring leaves the attempt record, which blocks re-reads (`gates.assert_no_repeat_oos`); `promote --allow-failed-rescore` retries only never-scored reads. `failed` records (e.g. a train crash before any read) are informational and never gate, so a transient failure cannot brick a cohort relaunch.
- **`promote` writes the registry** (the P0): full tier-4 result record with OOS metrics; `collect` dedups on `(run_id, evaluation_tier)` of scored records, so tier upgrades are recordable.
- **Windows + base config pinned.** `spec.windows` must match `CANONICAL_WINDOWS` (W1–W6: train through Dec-31 of 2013+2N, two-year holdout; reference by name). Unknown window keys are rejected (no more silently-dropped typo'd date flags). `base_config` must be `config/config.yaml` (sha256 recorded in `cohort.json`). Shipped specs moved from a non-canonical 2020/2021–22 window to canonical **W4**.
- **OOS budget.** Tier-4 launches refuse to read the holdout for more variants than `--oos-budget` (default 1); grid-sized reads are an explicit decision.
- **Report firewall is real.** `report.py` surfaces OOS medians only from tier ≥ 4 *scored* records (a hand-run backtest on a tier-3 variant can no longer leak numbers into the table), renders the captured Sharpe CI, and the header counts variants + actual holdout reads (attempt-record based) for multiplicity context.
- **Run-dir reuse refused.** `train.py` exits on an existing `Runs/<id>/manifest.json` unless `--overwrite-run` (which also clears the stale best-eval threshold + best-model artifacts that used to suppress `best_model` saves) or `--resume`. `launch` resumes cohorts by skipping variants already scored at the cohort tier. `--backend modal` (accepted-and-ignored) was removed; `success_gates`/`budget` now print an explicit "not enforced" note.

## Sprint 2 — measurement path (P1/P2)

- **`chronological_holdout` survives training.** The final manifest write updates the pre-training manifest instead of rebuilding it; `training_status: completed|interrupted` is stamped (Ctrl-C is no longer indistinguishable from a clean finish). Empty-holdout IndexError fixed.
- **No silent raw-obs rollouts.** `backtest.py` and `infer_weights.py` hard-fail when `vec_normalize.pkl` is missing on a `norm_obs: true` run; `--allow-raw-obs` is the explicit escape hatch.
- **Hashes are compared, not just recorded.** `backtest_summary.json` gains `hash_drift`; config/data-cache hash mismatches vs the training manifest print warnings. Missing run-local data snapshot triggers a loud reproducibility warning before the global-cache fallback.
- **Best checkpoint pairs with its stats.** `EvalNavBestModelCallback` saves `models/best/vec_normalize.pkl` at best-save time; the end-of-training copy is only a fallback.
- **Label integrity.** `checkpoint_label` is derived from the weights actually evaluated (`best`/`final`/`latest`), never `--plot-tag` (which once mislabeled final-model OOS as "best"). Ensemble mode honors `--detailed` and writes per-run summaries; plot-skip is announced. `--obs-lag` defaults from the run manifest (then run config), like every other window flag. Stochastic/deterministic length-mismatch truncation now warns.

## Sprint 3 — reward and split-mode correctness (P1/P2)

- **Sortino exploit closed.** `reward.sortino_downside_floor` (config 0.001 = 10 bp/day; parser default 1e-4 preserves old run snapshots) floors the downside deviation, so a no-loss ~cash book can no longer pin the clipped differential at ±3 (±75/step).
- **`independent` split mode now fit for purpose.** Per-segment features are computed over a causal preroll (`data.feature_preroll_bars: 252`) and sliced, so EMA-100/MACD/fracdiff get real warmup instead of truncation transients dominating 126-bar eval blocks; only panel-head bars without preroll are neutralized. With full preroll, independent == continuous bit-for-bit (tested) — `specs/feature_split_ab.yaml` now measures contamination, not warmup artifacts.
- **Data hygiene.** Pre-live bars are feature-neutralized via `compute_feature_panel(asset_live=…)` (the pre-IPO bfilled price level no longer reaches the obs through fracdiff); OHLC fill is unlimited-ffill-then-bfill, so delisted/halted assets liquidate at their last real close instead of a 1e-8 filler (and mid-history gaps are never backfilled with future resumption prices); HY-OAS is ffill-only; `load_cache(expected_fracdiff_d=…)` recomputes fracdiff when the cached `d` mismatches the active config.

## Sprint 4 — tests, docs, CI

- **New test files**: `test_holdout_reservation.py` (12), `test_causal_execution.py` (4: obs at `t−obs_lag`, fill at `open[t+1]`/MTM `close[t+1]` to 1e-12, holding cost on pre-rebalance units at `close[t]`), `test_reward_terms.py` (8: every `rew_decomp/*` term vs config coefficients; decomposition sums to the reward; Sortino floor), `test_research_cli_gates.py` (6: the orchestrator CLI actually enforces the gates, attempt-before-read verified from inside a fake subprocess), `test_reproducible_seeding.py` (5), `test_vecnorm_freeze.py` (3, SB3-gated). Extended `test_research_registry.py` (+17) and `test_feature_split_modes.py` (preroll semantics).
- **Docs**: README's "join purge — no cross-block leakage" overclaim replaced with the real split-mode semantics; README/RESEARCH window tables aligned to the canonical W1–W6 pattern; RESEARCH.md's stale "backtest uses the current global config" corrected to run-local snapshot binding and an auto-research-loop section added; published-metric examples use `--checkpoint best`; CLAUDE.md/AGENTS.md updated for all new behavior and the `paper_trade`/`execution` tracking claims fixed (with `.gitignore` comment); `data_utils.py` module docstring no longer claims pre-listing rows are dropped.
- **Doc tripwires extended** to README/RESEARCH/TRAINING (purge overclaim, run-local binding claim, canonical-window agreement, `--checkpoint best` examples, tracked-path claims).
- **CI** runs `pytest tests/` (new test files can't be silently excluded) and gains a `workflow_dispatch` torch job that installs the full stack so the torch-gated tests actually execute.
- **Modal fix (drive-by)**: user-passed `--n-envs`/`--batch-size` now override the GPU broker (they were silently ignored; `docs/MODAL.md`'s claim is true again).

## Not done here (by design)

Running `specs/reward_ablation.yaml` and the validation-cliff cohort is compute to launch on the sealed harness (review §6 Phase 2). Remaining review P2s not in the sprint plan: Modal unauthenticated endpoints, wrong run-id reporting with `--window` on Modal, lockfile, `lookback` decoupling, `obs_layout_version`.
