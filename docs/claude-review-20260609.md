# RLBot Comprehensive Review — 2026-06-09

> **Status (2026-06-09, post feat/continuous-research-hardening):** Historical snapshot. Written against an earlier tree; numbers and open findings herein were superseded by the hardening branch (see git log: C1/H1/H2 fixed, OOS burn ledger + deflated Sharpe + success gates + modal backend + shadow trading added; suite 226 passed / 3 skipped). Kept for provenance.

**Branch:** `feat/evolution-roadmap` (HEAD `ca3c4a5`) · **Test suite:** 92 passed / 1 skipped (torch-free, 1.6 s)
**Scope:** full repo — core methodology (`rlbot/trading_env.py`, `rlbot/data_utils.py`, `rlbot/rl_config.py`), harness (`scripts/train.py`, `scripts/backtest.py`, artifacts/inference), auto-research loop (`rlbot/research/`, `scripts/research.py`), Modal cloud, tests, and docs. Follows up the 2026-06-05 review cycle (`docs/claude-review-20260605.md`, codex/grok reviews, May-31 design/critical reviews) and audits what `feat/evolution-roadmap` actually delivered.

---

## 1. Executive summary

The evolution-roadmap branch **delivered what it claimed**: of the five prior P0/P1 harness findings, all five are verifiably fixed (run-local config + data-cache binding in backtest, `--checkpoint best` default with OOS-touch warnings, eval-NAV best-model selection with opt-in early stopping, reward-decomposition logging to TB + JSON, opt-in `training.reproducible` seed streams). The test suite nearly doubled (51 → 92) and is wired into CI. The infrastructure layer is now genuinely good.

Three things keep the project from being ready to *run* research on this substrate:

1. **The OOS firewall has a hole at its most important gate (P0).** `research.py promote` reads the holdout but never writes a registry record, so `assert_no_repeat_oos` never sees tier-4 reads on the canonical workflow — the "holdout read once per variant" guarantee is void. Related gaps (unconstrained spec `windows`/`base_config`, grid-sized tier-4 reads under one `--promote`, a report header that falsely claims tier filtering) mean the auto-research loop, if used today by an agent, would quietly do the multiple-testing the tier system exists to prevent.
2. **The reward function still has two terms that can dominate the economic signal (P1).** The Sortino bonus saturates at ±75/step via its `1e-4` downside-deviation floor (an exploit channel rewarding near-zero-variance books), and the inactivity penalty reaches −25/step at full cash (~2× the typical daily return term). The decomposition logging built for exactly this purpose exists but **the rebalance experiment has never been run** — as with every other open methodological question.
3. **Zero research has been executed on the new harness.** The validation-cliff investigation — the central empirical question of every prior review — remains open. The three shipped specs have never been launched. RESEARCH.md is still stale/placeholder.

**Bottom line:** the 06-05 verdict was "substrate ~5/10, → 8/10 after sealing the harness." The harness sealing happened; the substrate is now ~7.5/10. What's needed next is a short fix sprint (the P0 + the handful of P1 harness holes below, ~1–2 days of work), then a deliberate pivot from *building* infrastructure to *using* it.

---

## 2. Strengths

These are genuine and worth preserving through any refactor:

- **Causality discipline is unusually thorough.** Holdout reserved before any split; strictly causal features (trailing-window RSI/MACD/EMA/vol, one-sided fracdiff); expanding-window (not full-sample) OLS calibration of the HY-OAS proxy (`data_utils.py:249-281`); obs at `t − obs_lag`, fills at `open[t+1]`, MTM at `close[t+1]`.
- **Cost-fair comparisons by construction.** The agent, the in-reward benchmark, and all baselines share one execution model (`baselines.portfolio_step_nav`): same fill timing, same per-asset slippage/fee/holding arrays, same live-mask renormalization, including first-bar entry costs for baselines. The Sortino differential and baseline tables are genuinely apples-to-apples.
- **Run-local snapshot binding, done right.** Backtest binds `Runs/<id>/config.yaml` and `Runs/<id>/data_cache.npz` by default, with batch-bleed protection in `_maybe_load_run_config` (`backtest.py:148-164`) and explicit overrides — and it's unit-tested (`tests/test_backtest_config_binding.py`).
- **Fail-closed research firewall design.** Default-deny patch allowlist with prefix matching and typo-proof `set_nested` (refuses non-existent keys, `spec.py:59-69`); unknown tiers treated as OOS-touching; corrupt registry blocks rather than permits; `fee_scale_default=1.0` hard-coded in the OOS env as defense in depth (`backtest.py:932`).
- **Hard cap invariant.** The per-asset cap projection has a guaranteed post-condition (`trading_env.py:108-118`), fuzz-tested across caps × N.
- **Test architecture.** Torch-free core suite (1.6 s, runs in CI on every push), doc↔code tripwire tests (`test_docs_invariants.py`), torch-gated escape hatches for heavy paths.
- **Observability for the reward question.** Per-step `rew_decomp/*` info, TB scalars, and `Runs/<id>/eval_logs/reward_decomp.json` make the reward-balance debate measurable instead of rhetorical.
- **Honest self-documentation.** The continuous-mode eval contamination and "eval NAV is a selection signal, not an independent estimate" caveats are stated in code, config, and CLAUDE.md rather than hidden.
- **Clean Modal integration.** SDK-free local path, identical `Runs/<id>/` artifact layout in the cloud, mid-run volume commits gated by `RLBOT_MODAL=1`.

---

## 3. Status of the 2026-06-05 findings

| Prior finding | Status | Evidence |
|---|---|---|
| Backtest bound global config/cache (P0) | **Fixed** | `backtest.py:141-164`; tested |
| `--checkpoint` defaulted to `both` (P1) | **Fixed** | default `best` + OOS warning (`backtest.py:1234, 1302-1306`) |
| No early stopping / weak best-model selection (P1) | **Fixed (opt-in)** | `EvalNavBestModelCallback` + `training.early_stop_patience` (`train.py:162-249`) |
| `rew_decomp/*` never aggregated to disk (P1) | **Fixed** | `RewardDecompCallback` → TB + `eval_logs/reward_decomp.json`; tested |
| Same-seed runs diverge (P1) | **Fixed (opt-in)** | `training.reproducible` → per-env seed streams (`train.py:1033-1049`) |
| Feature-state contamination / dead purge (P0-A) | **Fixed (mechanism)** | `data.feature_split_mode` continuous/independent; 6 tests. But see §4.2 — the `independent` mode's purge is too short to achieve its goal. |
| Agent docs stale (P0-D) | **Fixed, drifting again** | CLAUDE/AGENTS accurate + tripwire; README/RESEARCH.md still carry the old errors (§4.5) |
| Machine-readable summaries (P1-E) | **Fixed** | `backtest_summary.json`, `training_summary.json`, hashes + git provenance |
| Reward rebalance experiment (P1-A) | **Not done** | spec shipped (`specs/reward_ablation.yaml`), never run |
| Eval-estimator diversity (P1-B) | **Not done** | coverage print only (`train.py:1171-1181`) |
| Multi-seed distributional OOS as primary number | **Not done** | ensemble plumbing exists; no cohort run |
| Validation-cliff investigation | **Not done** | the central open question, still open |
| Dependency lockfile; GPU/torch smoke in CI; feed-forward PPO baseline; HY-OAS coeff persistence; `lookback` decoupling (hardcoded 20 at `data_utils.py:381,411`); stop-loss 0.45 decorative; `obs_layout_version` | **Not done** | acknowledged deferrals or unclaimed |

---

## 4. Findings

Severity: **P0** = voids a stated guarantee; **P1** = serious, can corrupt results or conclusions; **P2** = real but bounded.

### 4.1 OOS firewall (auto-research loop)

- **P0 — `promote` never records the OOS read; "once per variant" is unenforced.** `cmd_promote` (`scripts/research.py:231-246`) runs the tier-4 backtest but never appends a registry record. The only registry writer is `cmd_collect`, which dedupes by `run_id` — and the variant already has a tier-3 record from the original launch, so even a manual `collect` skips it (and would stamp tier from `cohort.json` anyway). `assert_no_repeat_oos` (`gates.py:46-57`) only counts `evaluation_tier >= 4` records, so on the canonical workflow it passes forever: the same variant can re-read the holdout indefinitely.
- **P1 — OOS reads recorded after the fact, not at read time.** In a tier-4 launch the holdout is read at `research.py:173` but the record is appended only after the whole sweep via `cmd_collect`. A crash between backtest and collect leaves an unrecorded read; relaunch reads again. Record a `tier4_attempt` row *before* launching the backtest.
- **P1 — spec `windows` and `base_config` are outside the firewall.** The patch allowlist blocks `training.holdout_days`, but `windows` entries pass `--train-end/--holdout-start/--holdout-end` straight through (`research.py:117-120`) with no validation against a canonical window registry — a spec can place a tiny favorable holdout anywhere. `base_config` may point at any YAML (different universe/costs/split) with an empty patch.
- **P1 — one `--promote` authorizes grid-sized holdout reads.** A tier-4 spec with an 18-cell grid reads the holdout 18 times in one launch; picking the best cell on the holdout is exactly the multiple-comparisons problem the tiers exist to prevent. No cap, no warning, no per-cohort OOS budget.
- **P1 — report header claim is false.** `report.py:50-51` prints "OOS metrics shown only for promoted (tier ≥ 4) runs" but `_render` (`report.py:26-42`) shows `oos_*` from any record with no tier filter. Relatedly (P2), `cmd_collect` ingests any `backtest_summary.json` — a hand-run `backtest.py` on a tier-3 variant puts OOS metrics into a tier-3 record that the gate never counts.
- **P1 — relaunch overwrites runs while the registry keeps stale records.** `run_id == variant_id` deterministically; `train.py` has no collision check (`RunPaths.mkdirs` is `exist_ok=True`). Relaunching a cohort retrains into the same `Runs/<variant_id>/` (destroying the artifacts the registry describes), and bumping a spec tier 3→4 + relaunch re-reads the holdout for every variant without tripping the gate.
- **P2** — no file locking / re-read on `registry.jsonl` (concurrent launches race past the gate); `build_record` never sets `status`/`failure`; a tier-4 variant whose backtest crashed is blocked forever despite never being scored; a corrupt registry line bricks every subcommand (at least it fails closed); `success_gates` and `budget` in specs are parsed and never enforced; `--backend modal` is accepted and silently ignored (trains locally).

### 4.2 Core methodology

- **P1 — Sortino term saturates via the downside-deviation floor.** `_compute_sortino` (`trading_env.py:624-628, 873-885`) floors downside dev at `1e-4`. Any 20–63-bar window with no negative daily returns (easy for a ~90 %-cash book; cash returns are exactly 0) sends agent Sortino to `mean/1e-4` ≈ thousands; the diff clips at ±3 × `risk_bonus_scale=25` = **±75/step**, ~5× the typical return term. This is a quasi-binary exploit channel rewarding tiny-positive-variance books, and the spikes inflate VecNormalize's reward RMS, shrinking the effective weight of the real return signal. Fix: floor downside dev at an economically meaningful level (10–20 bp/day) or relative to total vol, and/or shrink the clip.
- **P1 — inactivity shaping dominates at the cash end.** At 100 % cash the penalty is −25/step (`trading_env.py:887-899`, `config.yaml:49-54`) — ~2× the typical daily return term, every step, with the +1 participation bonus pushing the same direction. The learned exposure level is substantially a reward-engineering artifact. The decomposition logging makes this measurable; `specs/reward_ablation.yaml` is the designed experiment — run it.
- **P1 — `independent` feature-split mode doesn't achieve its goal.** The 25-bar `feature_purge_warmup` is far too short: fracdiff(d=0.4)'s truncation transient decays as t^-0.4 (partial-weight sum 0.72 at t=25, 0.42 at t=100), so after the purge, *every* remaining bar of a 126-bar eval segment still has fracdiff dominated by the segment-start price level; EMA-100 trend never warms up inside a block at all (`data_utils.py:962-984, 395-404`). Independent-mode eval features come from a systematically different distribution than training (3-block ≈ 378-bar segments) and than the continuous backtest — so `specs/feature_split_ab.yaml` would partly measure this artifact, not contamination. Default `continuous` mode is unaffected. Fix direction: per-segment warmup ≥ ~250 bars (impractical at block=126), or compute features on segment + a causal pre-roll of train-side history (correct and cheap), and treat the current `independent` mode as not fit for purpose.
- **P2 — pre-IPO `bfill()` leaks the future first-listing price into the fracdiff channel** (`data_utils.py:639-641`): fracdiff of a constant `log(p_IPO)` is nonzero for hundreds of bars, so pre-IPO obs encode the future IPO price level. Live-masked (untradeable) so exploit value is low, but it's strictly future information and pollutes obs-norm stats.
- **P2 — delisting landmine**: after the 5-day ffill window, prices become the `1e-8` filler (`data_utils.py:607, 641`) and `_rebalance` force-sells held units at `1e-8` (`trading_env.py:678-686`) — near-total sleeve loss instead of liquidation at last real price. Latent for the advertised 5–55 universe range.
- **P2 — HY-OAS `bfill()` contradicts the no-backfill invariant** (`data_utils.py:328`): moot when FRED returns full history, but a truncated fetch backfills 2005-07 bars with a future calibrated value. Should be `ffill()` only.
- **P2 — fracdiff obs never includes lag 0** (`trading_env.py:570-574`): horizons (1,5,10,20) with `t0 = t_mkt − h` mean the freshest, most return-like feature is a day stale even at `obs_lag=0`, while RSI/MACD/trend/vol are sampled at `t_mkt`. If intentional, document; if not, off-by-one.
- **P2 — stale-cache desync**: `load_cache` doesn't validate cached `fracdiff_d` (or `lookback`-derived vol panels) against the active config (`data_utils.py:1170-1177`) — changing `data.fracdiff_d` without `--refresh-data` silently trains on stale-d features while the manifest records the new value.
- **Notes**: in-training eval blocks are interleaved with later train blocks → `best` selection is steered by an optimistic criterion (acknowledged; firewalled by the holdout). `obs_lag` applies to market features but portfolio meta (weights/drawdown/NAV) is valued at `close[t]` — defensible, consistent train/backtest, worth documenting. `portfolio_weights_from_action` reads `get_config()` at call time while the env captures config at construction — inconsistent with the documented invariant. `fracdiff_weights`' `weight_eps` threshold is unreachable (always returns 50k weights; O(n×50k) convolution). Dead guard at `trading_env.py:780`; first segment double-skips lookback (~22 wasted bars); `stats.py` Sharpe hard-codes 252 and ddof=0; no benchmark-relative (Sharpe-difference) bootstrap — the test you'd actually want for "beats 60/40".

### 4.3 Harness (train / backtest / inference)

- **P1 — the final manifest write drops `chronological_holdout`.** `train.py:1283-1314` rewrites `manifest.json` after `learn()` without the holdout block the first write (line 890) included. Backtest's `resolve_oos_holdout` falls back to `manifest["args"]`, which works for explicit dates — but the *computed* `holdout_end` is lost, so if the run-local `data_cache.npz` is ever deleted (large, gitignored) and the global cache has newer bars, the OOS window silently extends past what training reserved. This is the same failure class the run-local binding was built to close. The interrupted-run path also reaches this overwrite with no "interrupted" marker.
- **P1 — missing VecNormalize stats silently degrade to raw observations.** `backtest.py:546-550`: if `vec_normalize.pkl` is absent and `--require-vec-normalize` not passed, the rollout feeds unnormalized obs to a policy trained on normalized ones — plausible-looking but meaningless metrics, zero warning. Same exposure in `infer_weights.py:119`. For an OOS-measurement tool the default should be hard-fail (or loud warning) when the run config says `norm_obs: true`.
- **P2 — hash "drift detection" is record-only.** Train, backtest, and infer_weights all *record* config/data hashes; nothing ever *compares* backtest-time hashes against the training manifest. One `if mismatch: warn` closes the loop.
- **P2 — `best_model.zip` is paired with end-of-training VecNormalize stats**, not stats from the best-save moment (`train.py:106-117` vs `:235-236`) — a real train/inference mismatch on the published checkpoint. Save `vec_normalize.pkl` alongside the model in the best-model callback.
- **P2 — `--allow-latest-checkpoint` mislabels output**: it makes `_resolve_model_path_for_run` *prefer* `final` (`backtest.py:291-294`) while `plot_tag`/`ckpt_label`/summary still say `best` — OOS-provenance corruption in `backtest_summary.json`.
- **P2 — silent global-cache fallback** when the run snapshot is missing (info log only, `backtest.py:468`); combined with the manifest issue above, this is the concrete path to silently shifted OOS numbers.
- **P2 — ensemble mode silently ignores `--detailed`** (`backtest.py:674, 715, 793`) — CLAUDE.md's canonical ensemble command does nothing for that flag.
- **P2 — run-id reuse footgun**: re-training with an existing `--run-id` restores the previous run's `best_mean_nav` from `eval_nav_history.npz` (`train.py:192-202`), so the fresh model may never save `best_model.zip`. Training should refuse or warn on an existing run dir (also fixes the research-relaunch overwrite in §4.1).
- **P2 — `--obs-lag` is not defaulted from the manifest** (`backtest.py:1379`, hard default 1) while every other window flag is.
- **Notes**: `freeze_vec_normalize_for_inference` forces `norm_obs=True` regardless of training-time setting (latent for `norm_obs: false` runs). Import-time `resolve_data_cache()` side effect (`backtest.py:90`) moves legacy files on import. Deterministic OOS metric silently truncated if stochastic-path lengths mismatch (`backtest.py:606-611`). Hard-coded prints that can disagree with applied config (`train.py:982-985, 1136`). `--checkpoint both` silently disables plots; `--fast` resample counts differ between single (2000) and batch (500) modes. Empty-holdout IndexError at `train.py:899-908`. Structure: `run_oos_backtest` is a ~280-line monolith; the 12-tuple panel unpack is repeated ~5× across three scripts (use the existing `WalkforwardEnvPack`-style dataclass); load→clip→holdout pipeline duplicated between train and backtest.

### 4.4 Modal cloud

- **P2 — unauthenticated web endpoints**: `training_plot`/`run_status` (`modal_app.py:151-206`) serve plots and the full manifest (CLI args, git commit, paths, hashes) to anyone with the URL. Add `requires_proxy_auth`.
- **P2 — broker silently overrides user `--n-envs`/`--batch-size`** (overrides appended after forwarded argv, argparse last-wins; `modal_app.py:118-126`) — contradicts `docs/MODAL.md:163` and forces GPU-profile `n_envs` onto `--resume` runs where an env-count mismatch matters.
- **P2 — wrong run-id reported with `--window`**: `extract_run_id` recomputes `new_run_id` *after* training, when the real dir already exists, returning the next suffix (`W2_605_a` instead of the actual run) (`modal_cloud.py:131-146`, `modal_app.py:143, 236-241`).
- **Notes**: infra retry re-runs from scratch into the same run dir (no `--resume`), mixing artifacts; `list_runs` filter `^W\d+_` hides research cohorts and custom run-ids; image build ships the whole repo with a thin ignore list.

### 4.5 Documentation and tests

- **P1 — README still carries the leakage overclaim the 06-05 cycle flagged.** `README.md:111`: "Per-segment features … with join purge — no cross-block leakage" is wrong for the default `continuous` mode (features are sliced from the continuous panel; purge applies only in `independent` mode). README is self-described as the canonical reference, and the doc tripwire only covers CLAUDE/AGENTS.
- **P1 — `docs/RESEARCH.md` is stale in a dangerous direction**: claims backtest "uses the **current** global config for env mechanics" (now false — snapshot binding is the fix the whole cycle centered on), repeats the no-cross-block-leakage overclaim, and never mentions the auto-research loop it nominally governs. Tables are placeholders (acknowledged in commit `a2a055a`).
- **P1 — user docs lag the roadmap surface**: README/TRAINING/RESEARCH/MODAL contain zero mentions of `feature_split_mode`, `research.py`, `infer_weights.py`, `backtest_summary.json`, `training.reproducible`, or `early_stop_patience`. Only CLAUDE/AGENTS document the new machinery.
- **P1 — test gaps on the core invariants**: no direct test of `reserve_chronological_holdout` (the single most load-bearing function in the repo), none of env causal execution/`obs_lag`/fill timing, none of the per-term reward values against config coefficients, none (even torch-gated) of `freeze_vec_normalize_for_inference`, none of `apply_deterministic_seeds`/`training.reproducible`, and nothing exercises the `research.py` CLI actually calling the gates (a refactor could silently bypass `assert_tier_allowed`).
- **P2** — CI hardcodes the 12 test files (a new file silently never runs; use `pytest tests/`); CLAUDE.md self-contradiction on `paper_trade/` (it **is** tracked: `paper_trade/README.md`, `scripts/paper_trade.py`) and on `execution/README.md` (nothing under `execution/` is tracked); stale `data_utils.py:8-10` docstring ("pre-listing rows are dropped" — contradicts live-mask design); README/TRAINING examples still model `--checkpoint both`; the risk-parity decision-bar test is near-tautological; the only torch-gated test always skips in CI; no lockfile (acknowledged); `[tool.setuptools.data-files]` won't place `config.yaml` for non-editable installs, so console scripts only work from a checkout.

---

## 5. Remediation plan

### Sprint 1 — Seal the research firewall (before any cohort is launched) — ~1 day

1. **Record OOS reads at read time.** Append a `tier4_attempt` registry record *before* the promote/tier-4 backtest runs; make `cmd_promote` append the full tier-4 result record; key `collect` dedup on `(run_id, evaluation_tier)` instead of `run_id`. *(P0 + P1)*
2. **Constrain windows and base config.** Validate spec `windows` against a small canonical window table (the walk-forward windows in README); require `base_config` to be the repo config (or hash-pin it in `cohort.json` and the registry). *(P1)*
3. **Cap tier-4 reads.** Refuse `--promote` on a spec whose grid exceeds N variants (e.g. 3) without an explicit `--oos-budget`; print "K variants will read the holdout" and record K in the cohort. *(P1)*
4. **Fix the report**: filter `oos_*` rendering to tier ≥ 4 records (making the header claim true) and print "N variants tested in cohort" next to any OOS number; render the `oos_sharpe_ci` that's already captured. *(P1)*
5. **Refuse run-dir reuse**: `train.py` errors (or requires `--overwrite-run`) when `Runs/<run_id>/manifest.json` exists. Fixes both the research relaunch-overwrite and the stale `best_mean_nav` footgun. *(P1)*

### Sprint 2 — Seal the measurement path — ~1 day

6. **Preserve `chronological_holdout` in the final manifest write** (build the final manifest by updating the first, not rebuilding it); mark interrupted runs as such. *(P1)*
7. **Hard-fail on missing VecNormalize stats** when the run config has `norm_obs: true` (backtest + infer_weights), with `--allow-raw-obs` as the explicit escape hatch. *(P1)*
8. **Compare hashes, don't just record them**: backtest warns loudly when its config/data hashes differ from the training manifest's. Warn loudly on global-cache fallback. *(P2)*
9. **Save `vec_normalize.pkl` next to `best_model.zip` at best-save time** in `EvalNavBestModelCallback`. *(P2)*
10. **Fix `--allow-latest-checkpoint` labeling**, ensemble `--detailed`, and manifest-default `--obs-lag`. *(P2)*

### Sprint 3 — Reward and split-mode correctness — ~2 days

11. **Fix the Sortino floor**: economically meaningful downside-dev floor (10–20 bp/day or relative to total vol); consider shrinking the ±3 clip. Then **run `specs/reward_ablation.yaml`** (tiers 1–3) — the inactivity/participation rebalance is an empirical question the harness can now answer. *(P1)*
12. **Fix or retire `independent` split mode**: recompute per-segment features with a causal pre-roll of train-side history (correct, cheap) instead of the 25-bar purge; only then is `specs/feature_split_ab.yaml` measuring contamination rather than warmup artifacts. *(P1)*
13. **Data hygiene**: `ffill`-only for HY-OAS (`data_utils.py:328`); replace pre-IPO `bfill` with neutral fill + assert fracdiff ≈ 0 pre-listing; delist-safe forced liquidation at last real price; validate cached `fracdiff_d`/`lookback` against active config on load. *(P2)*

### Sprint 4 — Tests and docs — ~1–2 days, parallelizable with 3

14. **Tests for the load-bearing invariants**: `reserve_chronological_holdout` boundaries; env causal execution (obs at `t−obs_lag`, fill at `open[t+1]`, MTM at `close[t+1]`, holding cost on pre-rebalance units); per-term reward values at constructed states vs config coefficients; torch-gated vecnorm-freeze test; `training.reproducible` same-seed equality; a `research.py` CLI test proving the gates fire. *(P1)*
15. **Docs**: fix `README.md:111` and RESEARCH.md's stale config-binding claim; document the roadmap surface (feature_split_mode, research loop, infer_weights, summaries, reproducible, early-stop) in user docs; extend the doc tripwire to README/RESEARCH; reconcile the `paper_trade`/`execution` claims in CLAUDE.md/.gitignore; fix the `data_utils.py` docstring. *(P1/P2)*
16. **CI**: run `pytest tests/`, not a hardcoded list; add a small torch smoke job (300-step train + backtest on N=5, asserting manifest/holdout/summary invariants) even if only weekly/manual. *(P2)*

---

## 6. Roadmap for continued development

The defining fact of the project today: **world-class harness, zero executed research.** Every methodological question from four independent reviews is still open because no experiment has been run. After Sprints 1–2 (the firewall and measurement seals), the priority is to *stop building and start measuring*. Phases 2–3 are where the project's actual questions get answered.

### Phase 1 — Fix sprint (≈1 week)
Sprints 1–4 above. Exit criterion: research firewall enforces its stated guarantees under test; backtest cannot silently produce wrong OOS numbers; reward terms bounded sanely; docs/tests tripwires extended.

### Phase 2 — First real research campaigns (≈2–4 weeks, mostly compute time)
Run, in order, on the sealed harness — each is a shipped or trivially-written spec:
1. **Reward ablation** (`specs/reward_ablation.yaml`, post-Sortino-fix): does the agent learn defensible exposure when shaping is weakened? This is the prerequisite for trusting anything else the agent does.
2. **Validation-cliff investigation** — the central question. Multi-seed cohort across walk-forward windows; use `eval_nav_history.npz` + reward decomposition to characterize when/why eval NAV collapses (overfitting to train blocks vs regime non-stationarity vs estimator noise).
3. **Curriculum ablation** (`specs/curriculum_ablation.yaml`).
4. **Feature-split A/B** (`specs/feature_split_ab.yaml`, only after the independent-mode fix).
5. **Feed-forward PPO baseline** (deferred from the prior cycle): if a memoryless policy matches RecurrentPPO, the LSTM is cost without benefit.
Promote at most the single best method per campaign to tier 4. **Adopt multi-seed distributional OOS (median + IQR across ≥5 seeds) as the only published number** — single-seed OOS Sharpe stops being citable.

### Phase 3 — Statistical rigor (≈1 week, alongside Phase 2)
- Benchmark-relative inference: stationary-bootstrap CI on the *Sharpe difference* vs 60/40 (the machinery in `stats.py` is 90 % there) — "beats the benchmark" gets a p-value.
- Eval-estimator diversity (deferred P1-B): jittered eval starts / multiple eval episodes per block so `best`-selection isn't steered by one deterministic path.
- Report multiplicity honestly: every published OOS number carries "selected from N variants × M seeds."

### Phase 4 — Robustness and scale (after Phase 2 results justify it)
- Universe stress: N = 5 / 25 / 55 cohorts (exercises the delisting fix, cap behavior, live-masking at scale).
- Cost sensitivity: 2× and 5× slippage/fee tiers — if the edge dies at 2× costs, it isn't an edge.
- Regime-sliced OOS reporting (per-window attribution is already in the backtest; aggregate it in the report).
- `lookback` decoupling and `obs_layout_version` in the manifest (carried from the prior cycle).

### Phase 5 — Toward live measurement (only if Phase 2–4 produce a method that survives)
- `scripts/paper_trade.py` hardened into a scheduled daily measurement job (it stays measurement-only — no broker).
- Capacity/impact modeling before any real-money conversation (currently absent by design).
- Authenticated Modal endpoints + a `--backend modal` that actually works (or remove the flag) if research campaigns move to the cloud.

### Standing rules (cheap to keep, expensive to lose)
- The holdout is read only through `research.py promote`, never by hand, and every read lands in the registry.
- Every config/method change goes through a spec; no untracked `Runs/` experiments feeding conclusions.
- RESEARCH.md gets regenerated from the registry (the stated purpose of `report.py`) — retire the hand-maintained tables.
- The doc tripwire grows with every doc fix: a claim corrected twice gets a test.

---

*Methodology note: this review was produced by four parallel deep-read passes (core methodology; harness/reproducibility; research loop + Modal; tests/docs/prior-finding audit), with numerical verification of the fracdiff-transient and reward-magnitude claims, cross-checked against the local test suite (92 passed / 1 skipped).*
