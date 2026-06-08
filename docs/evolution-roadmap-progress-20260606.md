# Continued-Evolution Roadmap — Progress & Enhancements (2026-06-06)

This document captures the implementation of the roadmap in `docs/claude-review-20260605.md`
(§7), what shipped vs the prior version, how it was verified, what to do next with the
auto-research capability, and the issues still open. The full plan lives at
`.claude/plans/fuzzy-hatching-horizon.md`.

**Branch:** `feat/evolution-roadmap`. **Tests:** 51 → **92 passing** (+1 torch-gated skip),
all torch-free and wired into CI (`.github/workflows/ci.yml`).

> **Results status (2026-06-08):** Harness work shipped; **no definitive OOS numbers** under the
> current config yet. Prior `W*_604` backtests used a different reward/cost/env stack and must not
> be cited. See [RESEARCH.md](RESEARCH.md).

---

## 1. Context

Four independent reviews (the two May-31 priors + the same-day Codex/Grok/Claude reviews on
2026-06-05) converged: the RL methodology is strong, but a few **harness-integrity** gaps
made experiments ambiguous, and the open methodological questions (reward balance,
validation-signal fidelity, the "validation NAV cliff") could not be answered until those
gaps were sealed. The roadmap sequenced the work 0→4: seal the harness, harden measurement,
build a gated auto-research loop, use it for method development, then add an audited
inference path. This round implements all five phases (with two honestly-scoped deferrals,
§4).

---

## 2. What shipped, by phase

### Phase 0 — Seal the harness
- **`feature_split_mode`** (`config.yaml` → `DataConfig`): `continuous` (default, unchanged
  behavior) vs `independent`, which **recomputes features per contiguous segment and applies
  the `feature_purge_warmup`** (`rlbot/data_utils.py: train_test_split_alternating`). The
  purge — previously retained but dead — is now functional in `independent` mode.
- **Backtest binds run-local snapshots by default** (`scripts/backtest.py`): loads
  `Runs/<id>/config.yaml` (`--use-current-config` to opt out) and prefers
  `Runs/<id>/data_cache.npz` (`--data-cache` override) via `resolve_run_data_cache`.
- **`--checkpoint` default `both` → `best`**, with an OOS-touch warning for `latest`/`both`.
- **Machine-readable results**: `Runs/<id>/backtest_summary.json` (single runs;
  `_print_detailed_stats` now returns a dict) + `Runs/<id>/training_summary.json`, both with
  config/data SHA-256 + git provenance (`run_artifacts.sha256_file`, `config_sha256`,
  `git_provenance`).
- **Doc-vs-code tripwire** (`tests/test_docs_invariants.py`) + CI so `CLAUDE.md`/`AGENTS.md`
  can't silently drift again.

### Phase 1 — Measurement hardening
- **Reward-decomposition logging** (`rlbot/reward_logging.py` + `RewardDecompCallback`):
  windowed per-term means and **share-of-absolute-reward** to TensorBoard +
  `eval_logs/reward_decomp.json` — makes the review's reward asymmetry observable.
- **Patience early-stop** (`EvalNavBestModelCallback`, gated on curriculum completion via
  `dr_widen_end_milestone`; `training.early_stop_patience`, default 0 = off).
- **Eval effective-coverage reporting** (segments × scored bars) — the honest part of the
  narrow-estimator concern.
- **`rlbot/stats.py`**: bootstrap helpers moved out of torch-importing `backtest.py`, so the
  bootstrap unit test runs in CI without torch.
- **60/40 benchmark skips gracefully** when the universe lacks BOND10Y (no more `KeyError`
  on `--n-assets 5`).
- **Cap post-condition projection** in `portfolio_weights_from_action` (guarantees max risky
  ≤ cap for arbitrary cap/N) + fuzz tests across caps × N.
- **`training.reproducible`**: deterministic per-env seed streams (`seed + env index`) as an
  opt-in alternative to `reseed_on_reset`.

### Phase 2 — Auto-research MVP
- **`rlbot/research/`**: `spec.py` (experiment spec, **allow-list firewall**, grid
  expansion, config materialization), `registry.py` (append-only JSONL + record builder),
  `report.py` (Markdown tables from the registry), `gates.py` (tiers T0–T5 + OOS firewall).
- **`scripts/research.py`** (`plan`/`launch`/`collect`/`report`/`promote`): shells to the
  canonical `train.py`/`backtest.py`, writes `Runs/<cohort>/registry.jsonl`, resilient
  per-variant (failures logged + recorded, sweep continues), enforces "OOS read once per
  variant, tier ≥ 4 requires `--promote`".

### Phase 3 — Method development (tooling)
- Runnable cohort specs: `specs/feature_split_ab.yaml` (the continuous-vs-independent A/B),
  `specs/reward_ablation.yaml`, `specs/curriculum_ablation.yaml`, each validated by
  `tests/test_research_registry.py`. Universe scaling stays a `--n-assets` workflow by design
  (the firewall forbids patching the universe).

### Phase 4 — Deployment-oriented inference
- **`scripts/infer_weights.py`**: audited target weights from `--run-id --as-of`, reusing
  the proven backtest rollout (recurrent warmup, frozen VecNormalize), with full provenance;
  torch-free assembly/validation in `rlbot/inference_output.py`.
- *(Removed 2026-06-08)* `scripts/paper_trade.py` + `paper_trade/` — thin turnover-logging
  wrapper around `infer_weights.py`; superseded by calling `infer_weights` directly.

---

## 3. Enhancements vs the prior version

| Area | Before | After |
|---|---|---|
| Feature purge | Retained but **never applied** | Applied in `independent` split mode (opt-in) |
| Backtest config | Current **global** config only | Binds **run-local** `config.yaml` by default |
| Backtest data | Current **global** cache only | Prefers **run-local** `data_cache.npz` |
| Checkpoint default | `both` (touches OOS w/ non-best weights) | `best`, with OOS warning |
| Run results | stdout only | `backtest_summary.json` + `training_summary.json` + manifest hashes |
| Reward decomposition | Emitted in `info`, never aggregated | Windowed TB scalars + JSON (abs-share) |
| Early stopping | Off only (`early_stop: off`) | Optional patience after curriculum |
| Determinism | `reseed_on_reset` only (not reproducible) | Optional `reproducible` per-env streams |
| Bootstrap stats | Inside torch-importing `backtest.py` (untestable) | `rlbot/stats.py`, CI-tested |
| 60/40 on small N | Hard `KeyError` | Graceful skip + note |
| Per-asset cap | Best-effort 5-iter redistribute | Guaranteed final projection + fuzz tests |
| Auto-research | None | Spec → registry → report → gated orchestrator |
| Inference | No audited in-tree path | `infer_weights.py` + `inference_output.py` |
| Agent docs | Materially stale | Corrected + invariant test + CI |
| Tests | 51 | 92 (+ torch-gated) |

---

## 4. Verification status (honest)

- **Verified:** 92 torch-free tests; every torch-importing module byte-compiles; the
  orchestrator was run end-to-end (`research.py plan` materialized + validated variant
  configs; `launch --dry-run` built correct CLIs with window dates) on this checkout.
- **Not executed here (no torch / no yfinance data in this environment):** the actual
  train → backtest → infer_weights smoke E2E, the training callbacks (reward-decomp,
  patience early-stop), and the inference rollout. These were written to mirror existing
  proven code and pass an independent skeptical review, but **should get a GPU/data smoke run
  before being relied on.**
- **Deferred (scoped out this round):**
  1. **Dependency lockfile** — needs `uv`/network to resolve the torch stack offline; the
     Modal image already pins Python 3.11.
  2. **Feed-forward (non-recurrent) PPO architecture experiment** — needs a non-recurrent
     train+inference path that can't be verified without torch, so it was not shipped untested.

---

## 5. Hardening applied after self-review (this round)

An independent review of the diff drove these fixes: removed a **dead `eval_start_jitter`
config knob** that silently did nothing (config + allow-list + log line); fixed **batch-mode
config bleed** (a run without a snapshot now rebinds the fresh global default instead of
inheriting the prior run's config); narrowed two broad `except Exception` blocks
(`git_provenance`, train best-step) so real bugs aren't swallowed; `_read_json` now **warns**
on malformed files instead of silently returning `None`; the research `launch` loop is
**resilient with timings** (per-variant failures logged + recorded, cohort elapsed reported,
nonzero exit if any failed); added timing logs to `infer_weights` model-load + rollout;
made the reward-decomp JSON **windowed** (reset per interval); added an empty-block guard in
the split; removed dead code (`report.variant_table`, an unused bootstrap import, a redundant
Sharpe wrapper); and corrected the misleading 60/40 skip message.

---

## 6. Directions for continued development & use of auto-research

**Immediate (next session):**
1. **Run the first cohort.** `python scripts/research.py launch specs/feature_split_ab.yaml`
   on a GPU/data box (tier 3, no OOS), then `report feature_split_ab`. This answers the
   review's central open question: does `independent` split + the purge change the validation
   cliff? Read `reward_decomp.json` alongside to see the term balance.
2. **Reward rebalance.** `specs/reward_ablation.yaml` — use `rew_decomp/abs_share` to pick
   coefficients where participation/churn are non-trivial at target behaviors, then re-test
   the cliff.

**Near-term capability growth:**
3. **Modal backend for `research.py launch`** (`--backend modal`): submit variants to Modal,
   poll the existing `/status` endpoint, distinguish running-vs-crashed via
   `finished_at_utc` + a heartbeat, honor `budget.max_modal_hours`.
4. **Generate `docs/RESEARCH.md` from the registry** (wire `report.write_report` to the
   canonical doc) so the walk-forward tables stop being hand-maintained.
5. **Agent-driven loop:** a thin layer that proposes the next spec from registry results
   (e.g., bisect a reward grid, escalate a promising dev variant to a `--promote` tier-4
   pre-registration) — the firewall + JSONL search-tree make this safe to automate.
6. **Promotion workflow:** tighten `promote` to require the variant to already have tier-1–3
   evidence in the registry before its single OOS read.

**Longer-term:**
7. **Feed-forward PPO baseline** (the deferred architecture experiment) once a tested
   non-recurrent train+inference path exists — then it becomes one more spec cohort.
8. **Gradual universe scaling** via `--n-assets` cohorts (outside the spec firewall), only
   after the 10-asset method is credible.
9. **Live-readiness** (only after the above): point-in-time data, capacity/market-impact
   model, then a broker adapter consuming the `infer_weights` payload.

---

## 7. Remaining issues to address

**Should fix before heavy use:**
- **Smoke-run the torch paths** (training callbacks + inference rollout) on GPU/data — the
  one real verification gap.
- **Dependency lockfile** (`uv.lock` or pinned image) for reproducible long studies.

**Known latent issues (pre-existing, worth a follow-up):**
- **`lookback` decoupling:** `compute_feature_panel`/`compute_realized_vol_panels` hardcode a
  20-bar realized-vol window while the env's fallback `_realized_vol` uses
  `environment.lookback`. They agree at the default (20) — `continuous` and `independent`
  modes are consistent — but a non-default `environment.lookback` would desync the precomputed
  vol panels from the env. Thread `environment.lookback` through all feature call sites.
- **Eval estimator is still narrow:** coverage is now *reported*, but the deterministic
  full-segment eval remains a low-diversity selection signal. A proper jittered-start eval
  (a separate diagnostic env that doesn't drive selection) is the real follow-up — it was
  deliberately not faked as a dead config knob this round.

**Methodological (the reviews' standing priorities, now *answerable* with this tooling):**
- Whether the validation cliff is real overfitting or an estimator/contamination artifact
  (run `feature_split_ab` + eval-diversity work).
- Whether the reward shaping, once rebalanced, actually steers exposure/turnover.
- Multi-seed distributional OOS as the *primary* reported number (the registry supports it).
