# Claude Review: RLBot / MarketTrainer — 2026-06-05

> **Status (2026-06-08):** Roadmap items P0–P4 largely implemented per [evolution-roadmap-progress-20260606.md](evolution-roadmap-progress-20260606.md). Remaining: dependency lockfile, torch-path smoke runs, Modal research backend, registry-driven RESEARCH.md.

**Reviewer:** Claude (Opus 4.8, 1M context) via Claude Code
**Date:** 2026-06-05
**Scope:** Full repository — code (`rlbot/`, `scripts/`), `config/`, tests, docs, Modal integration — plus the two same-day peer reviews (`docs/codex-review-20260605.md`, `docs/grok-review-20260605.md`) and the two May-31 priors (`docs/RLBot_Design_Review.md`, `docs/RLBot_Critical_Review.md`).
**Method:** Static read of the source **plus execution**: I installed the light dependency subset and ran the torch-free test suite, ran the parser/CLI surface, and read the contested code paths line-by-line to adjudicate where the prior reviews disagree. I did **not** run a training job or an OOS backtest (no GPU / no data cache in this checkout). Findings are a design + harness-integrity review, not an empirical trading result.

---

## 1. Executive Verdict

RLBot remains the same thing all four prior reviews concluded: a **genuinely above-average RL-for-trading research prototype** whose durable value is its *methodology and harness discipline* — strict chronological holdout reserved before any split, causal next-open execution with `obs_lag`, wealth-based (not reward-based) checkpoint selection, frozen-VecNormalize inference, per-run config snapshots + manifests, dynamic universe support, Modal cloud training, and unusually honest negative reporting (the "validation NAV cliff"). That praise is real and I echo it.

My distinct contribution to the record is **adjudication**: the two same-day reviews directly contradict each other on the single most important methodological question, and I resolved it against the code.

1. **The Codex review is correct; the Grok review is wrong** on the leakage core. Grok's claim that "per-segment feature computation with join purge is intact" is **false**. The walk-forward split slices a *precomputed, continuous* feature panel and **does not apply the purge** — the code's own docstring and a runtime print say so explicitly. This is not an OOS leak, but it is real train/eval feature-state contamination in the model-selection signal, and several docs still claim the opposite.
2. **The test suite is not broken.** Grok reported "collection fails under bare `python3 -m pytest`." I reproduced that, then installed `gymnasium`+`pandas` and **51 tests pass**. The failure is purely a missing-dependency / onboarding artifact — with one real smell: the pure-numpy block-bootstrap unit test cannot even import, because its helper lives inside torch-importing `scripts/backtest.py`.
3. **A sharper reward finding than either prior made:** the shaping is not uniformly "too small." It is **asymmetric** — the inactivity penalty is *large* (up to **−25**/step at 100% cash in training) while the participation bonus (**+1**) and ordinary churn penalty (**−0.85** at 10% turnover) are tiny. The policy is pushed out of cash by a sledgehammer and toward exposure/low-churn by a feather. Both prior reviews said "shaping is inert"; the truer statement is "shaping is lopsided."

**On the auto-research question:** feasibility is **high**, benefit is **high**, but only *after* a small set of harness-integrity fixes. Run today, an automated loop would mostly automate two things this repo is already prone to: **overfitting the (contaminated, narrow) validation surface** and **propagating documentation drift** — and it would do the latter to *itself*, because the agent-facing instruction files (`CLAUDE.md`, `AGENTS.md`) are the most stale documents in the repo.

Readiness scorecard (my assessment; deliberately close to Codex's so the deltas are visible):

```
Experiment substrate (run ids, manifests, Modal, batch backtest):   7/10
Reproducibility substrate (config/data binding at inference time):   4/10
Metrics-as-data (machine-readable results without parsing stdout):   3/10
Automated-ranking safety (leakage firewall, OOS gating, pre-reg):    4/10
Auto-research readiness today:                                       5/10
Auto-research readiness after the P0 fixes below:                    8/10
```

---

## 2. Adjudicating the Prior Reviews

### 2.1 The central dispute: feature computation and purge — **Codex is right**

This is the one place the two same-day reviews are mutually exclusive, so it matters.

- **Grok** (§1, §3.2): "per-segment feature computation with join purge … intact"; "features … recomputed inside the function via `compute_feature_panel` on the raw ohlcv+macro slices"; lists `_neutralize_feature_warmup (purge=25)` as active.
- **Codex** (Executive #3, P0): the split "slice[s] a continuous, precomputed feature panel, and `feature_purge_warmup` is explicitly unused."

The code is unambiguous, and Codex matches it:

- `rlbot/data_utils.py:809-1043` — `train_test_split_alternating()` accepts precomputed `rsi/macd/fracdiff/fracdiff_macro/trend/asset_vol/macro_vol` (keyword args) and, in `_concat_ranges`, **slices them** (`rsi_g[orig_start:orig_end]`, etc.). There is no per-block `compute_feature_panel` call on the sliced ranges, and **`_neutralize_feature_warmup` is never called** in this path.
- `rlbot/data_utils.py:841` (docstring): *"`feature_purge_warmup` is retained for API compatibility but is not applied."*
- The fallback (`data_utils.py:883-885`) — when features are *not* passed — computes them once via `compute_feature_panel(ohlcv, macro)` on the **whole** passed array and then slices. Still not per-segment, still no purge.
- `scripts/train.py:757-801` — the live path: it reads `purge = cfg.data.feature_purge_warmup` (25), aligns the **cache** feature panel to the trainable timeline (`align_panel_to_timeline`), passes those precomputed arrays into the split, and then **prints**: `"features: cache panel → block slice (purge=25 unused); matches continuous backtest memory"`.

So this is a **deliberate design choice the author is aware of**, not a hidden bug. The rationale ("matches continuous backtest memory") is defensible: in production/backtest, features *are* computed continuously, so per-segment recompute+purge would make training perceive the world differently than inference does.

But two things follow, and Codex states them correctly:

1. It is **not** an OOS leak. The holdout is reserved first (`train.py:746`), holdout is the chronological *tail*, and all features (fracdiff, RSI, MACD, EMA, realized-vol) are strictly causal/backward-looking, so a trainable-region feature can only depend on trainable-region prices. **Verified by construction.**
2. It **is** train↔eval feature-state contamination *within* the trainable set. Because eval blocks are interleaved with train blocks (`eval_stride=4`) and features are sliced from one continuous panel, the head of each eval segment carries EMA/fracdiff/RSI memory seeded by the immediately preceding *train* block (and vice-versa). The 25-bar purge was designed to neutralize exactly this and is now off. The consequence is a less-independent eval-NAV signal — the very signal that selects `best_model.zip`.

**Verdict:** Codex's P0 stands. Grok's §1/§3.2 contains a factual error; everything Grok builds on "the invariants are intact" should be discounted accordingly. (Grok's instinct elsewhere — that the *agent docs* are the worst-drifted files — is, however, also correct; see §2.3.)

### 2.2 The test-collection dispute — **settled empirically**

Grok: "Test collection currently fails under bare `python3 -m pytest` (likely env/editable-install/path issues)." Codex declined to run pytest.

I ran it. In a bare checkout (no `.venv`, README setup never executed) collection fails with `ModuleNotFoundError: No module named 'gymnasium'`. After `pip install gymnasium pandas`:

```
tests/test_core.py tests/test_environment.py tests/test_eval_segments.py tests/test_run_artifacts.py
→ 51 passed in 0.40s
```

So **the test logic is sound**; Grok's failure was a missing-dependency artifact, exactly as Grok hedged. Two real findings survive, though:

- **Onboarding friction is real.** A fresh clone has *nothing* installed and the README's `.venv` flow hadn't been run here; "clone → pytest" is red until a multi-hundred-MB torch stack is installed. A documented light-test path (math/env tests need only `gymnasium`+`pandas`+`numpy`) would make CI and agents green in seconds.
- **A coupling smell:** `tests/test_block_bootstrap.py` imports `block_bootstrap_log_rets`/`block_bootstrap_sharpe_percentiles` from `scripts/backtest.py`, which imports `torch` at module load (`rlbot/inference_load.py:9`). So a **pure-numpy statistics test cannot run without a GPU-class dependency.** The bootstrap helpers should live in a light module (e.g. `rlbot/stats.py`) that both `backtest.py` and the test import.

### 2.3 Documentation drift — both priors are partly right; the worst is the *agent* docs

I checked every doc against verified code. The drift is **not uniform**; it splits cleanly:

- **`README.md`, `docs/RESEARCH.md`, `docs/TRAINING.md`, `docs/MODAL.md`, `config/README.md` are largely current.** They use the correct `obs_dim = 10N + 28`, `Runs/` casing, dynamic-N story, and even describe the precompute-then-slice feature behavior honestly (RESEARCH.md: "features precomputed on the full trainable timeline, sliced per segment"). Minor nit: examples sometimes show `--checkpoint best` while the parser default is `both`.
- **`CLAUDE.md` and `AGENTS.md` (the agent-facing instruction files) are badly stale** — and `CLAUDE.md` is the file loaded into every Claude Code session here. Verified-wrong claims in both:
  - `sync_trading_env_aliases()` "writes module-level constants" — **the function does not exist** (`rl_config.py` has no such symbol; the env captures config objects at construction, `trading_env.py:207-212`).
  - "118-d observation" — actual is **128** for N=10 (`observation_dim_for_universe`, `trading_env.py:135-137`).
  - "cap … default 0.50" — actual is **0.35** (`config.yaml:28`).
  - "RSI/MACD/fracdiff are computed … on each contiguous train/eval segment … features are recomputed after splitting" and "the first feature_purge_warmup (25) bars after each segment join are neutralized" — **both false** (see §2.1).
  - `windows/window1_train.sh`, `windows/window1_backtest.sh`, `windows/validate_split.py`, `backtest_sweep.py`, `models/<RUN_ID>/best/...` — **none exist** (no `windows/` dir; canonical path is `Runs/<id>/models/best/`).

This is the drift that matters most for the auto-research question (§5): if an LLM loop is bootstrapped from `CLAUDE.md`/`AGENTS.md`, it starts from a wrong mental model of the action space, the cap, the leakage guarantees, and the directory layout. **Codex caught this** (P0 "Docs and Instructions Drift"); **Grok caught it most sharply** ("`AGENTS.md` is the most out-of-date … this will bite any future auto-research effort"). Both correct; I add only that `CLAUDE.md` is *active context*, so this is not cosmetic debt — it is live misinformation.

### 2.4 Where each same-day review is slightly off

- **Codex** slightly overstates RESEARCH.md drift in one spot: RESEARCH.md *does* now describe precompute-then-slice. The residual problem is narrower — RESEARCH.md's phrase "no cross-block leakage" is *too strong* for sliced continuous features (the head-of-segment memory bleed is exactly cross-block state). So the doc is honest about the *mechanism* but overclaims the *guarantee*.
- **Grok** is wrong on §2.1 (purge/per-segment), and its reward verdict ("badly imbalanced … participation and churn ~1–2 orders of magnitude smaller") is half-right: true for participation/churn, but it misses that **inactivity is large**, which inverts the practical conclusion (§4.1).
- **Both** carried forward the May-31 "eval = first ~63 bars" critique only to (correctly) retire it: current eval is one full-segment deterministic rollout per eval segment (`train.py:1077-1095`, `test_eval_segments.py`). That fix is real.

### 2.5 Net vs the May-31 priors

The May-31 Design/Critical reviews' top-four — (1) reward-term balance, (2) validation-estimator fidelity / the cliff, (3) thin OOS statistics + post-hoc checkpoint pick, (4) no early stopping — **all remain open**. What *has* shipped since: in-tree inference loading + VecNormalize freeze, stationary block bootstrap + stochastic paths, eval-segment correctness + tests, and mature dynamic-N support. So the trajectory is good but the *methodological* core the priors flagged is essentially untouched. Their strategic order — *validate the method on the current 10-asset menu before scaling the menu or the claims* — is still correct.

---

## 3. Current Design, in Detail

### 3.1 Configuration (single source of truth, with one inference-time hole)

`config/config.yaml` → `rl_config.py` (`load_config`/`_parse_config`) → frozen `RLConfig` dataclass tree; `set_config`/`get_config` install a global singleton (tests use an autouse fixture in `tests/conftest.py`). Per-asset lists (`benchmark_cap_weights`, `slippage`, `tx_fee`, `annual_holding_cost`) are length-checked against `len(universe.assets)`; `slice_config_to_n_assets` powers `--n-assets` without editing YAML. Each run snapshots `Runs/<id>/config.yaml` + a rich `manifest.json`.

The single biggest gap: **inference does not re-bind to that snapshot** (§4, P0-B). The env reads config objects captured at construction (`trading_env.py:207-212`) — good, no stale module globals — but `scripts/backtest.py` never calls `load_config(Runs/<id>/config.yaml)`. So the snapshot is *written* but not *honored*.

### 3.2 Data pipeline & anti-leakage (the strongest part — with one overstated claim)

`rlbot/data_utils.py`. `fetch_aligned_daily` pulls tradeable assets + macro (DXY/TNX/VIX) from yfinance and HY OAS from FRED with a causal HYG/IEF expanding-OLS proxy (`_calibrate_hy_proxy_expanding`, `_attach_hy_oas_column`). It keeps the full calendar and live-masks pre-IPO rows (`asset_live`) — **note the module docstring (lines 8-11) still says "Pre-listing rows are dropped. The panel starts when all configured assets have real quotes," which is stale** (code: "live-masked pre-IPO; no global row drop", line 651).

Invariants that hold and are genuinely strong:
- `reserve_chronological_holdout` runs **before** any split (`train.py:746`); only backtest sees the holdout.
- All features are causal: fracdiff via `np.convolve` of López-de-Prado weights, RSI/MACD/EMA via backward EWMs, realized vol via trailing windows. Verified that trainable features cannot see holdout prices.
- `block_boundaries` + env `_build_segments` prevent episodes from spanning stitched gaps (`trading_env.py:407-431`).

The one weakness (§2.1): features are sliced from a continuous panel, the 25-bar purge is off, so the eval-selection signal is feature-state-contaminated; and "no cross-block leakage" in RESEARCH.md overstates this.

### 3.3 Environment & reward (`rlbot/trading_env.py`)

`MultiAssetPortfolioEnv` is dynamic in `n_assets`. Action `Box(-3,3)^(N+1)` → optional EMA logit smoothing (α=0.15, train+backtest) → softmax (cash competes) → live-mask → 5-iteration clip-and-redistribute per-asset cap (0.35) → long-only simplex (`portfolio_weights_from_action`, lines 60-107). Execution is causal: holding cost on pre-rebalance units at `close[t]`, rebalance at `open[t+1]`, MTM at `close[t+1]` (`step`, lines 802-930). Obs at `t_mkt = t - obs_lag`.

**Reward magnitudes (my computation from `config.yaml` + `step()` lines 857-898).** This is the sharper version of the priors' "shaping is inert":

| Term | Formula | Typical value | Extreme |
|---|---|---|---|
| Return | `clip(log_ret,−.12,.06) × 2000` | **±20** at a 1% day | +120 / −240 (clipped) |
| Sortino diff | `clip(Δsortino,−3,3) × 25` | a few → tens | **±75** |
| Inactivity | `cash_frac×10` (+`(cash−.9)/.1×15` over 90%); train scale 1.0 | **−5 at 50% cash** | **−25 at 100% cash** |
| Participation | `gross × 0.05 × 20` | **+1 at full investment** | +1 |
| Churn | `turnover × VIX_mult(.75–1.5) × 8.5 × churn_scale(0–1)` | **−0.85 at 10% turnover** | −12.75 at 100% turnover |
| Drawdown | `dd² × (25×12)=dd²×300` | −0.75 at 5% DD | −12 at 20%, −60 at 45% |

The story is **asymmetry, not smallness**: leaving cash is punished hard (−25 dominates a return step), but being invested is rewarded weakly (+1) and churning is penalized weakly (−0.85, and only after the churn curriculum ramps in). Sortino dominates everything when it fires. So:
- The config comment "bring regularizers into the same order of magnitude as return" was **achieved for inactivity, not for participation or churn.**
- Behaviorally this biases the policy toward "stay near fully invested, don't worry much about turnover" — which is plausibly *fine* as a prior but is not what the participation/churn knobs are nominally tuning, and it means those knobs are nearly dead while inactivity silently sets exposure.
- `rew_decomp/*` is emitted per step in `info` (lines 917-924) — excellent instrumentation — but **nothing aggregates it to TensorBoard or disk**, so the asymmetry is invisible unless you go looking.

Side note: training inactivity scale = 1.0, eval = 0.05 (`train.py:963,987`). Since checkpoint selection is by *ending NAV*, not reward, this asymmetry doesn't bias selection — but it makes eval-episode *reward* non-comparable to training reward, which is mildly confusing instrumentation.

The 0.45 stop-loss (`config.yaml:30`) is, as both priors note, effectively decorative on a diversified daily book.

### 3.4 Training (`scripts/train.py`)

RecurrentPPO (`MlpLstmPolicy`, 2×64 LSTM, [128,128] heads), `SubprocVecEnv` (16 local / 32–64 on Modal), `VecNormalize` (obs always; reward only on the train copy). Callbacks: `TradingCurriculumCallback` (fee-free → fee ramp → churn ramp → progressive DR widening, all as fractions of budget), `EvalNavBestModelCallback` (deterministic full-segment rollouts; saves `best/` on max mean ending NAV; **does not stop**), mandatory cosine `AdaptiveEntropyCallback`, periodic checkpoint + viz. Banner honestly prints `early_stop: off`. Training envs use `reseed_on_reset=True` with fresh OS entropy per episode (`trading_env.py:724-725`) — so "deterministic" is really "seeded framework + stochastic episode reseeding."

Eval estimator (`train.py:1077-1095`): `n_eval_episodes = #eval segments`, `deterministic=True`, `random_start=False`, `domain_randomize=False`. Correct and full-segment now — but a single fixed-start deterministic path per segment (~7–10 segments) is a **narrow, low-diversity estimator**, exactly as both same-day and both May-31 reviews note. Combined with the §2.1 feature contamination, the signal that selects `best_model.zip` is the weakest link in the whole method.

### 3.5 Backtest & inference (`scripts/backtest.py`, `rlbot/inference_load.py`, `rlbot/vecnorm_utils.py`)

Manifest-driven holdout dates, fast weight load (no optimizer state), `freeze_vec_normalize_for_inference()` (`training=False`, `norm_reward=False`, keep `norm_obs=True`) — all correct and real progress vs the May-31 "no in-tree inference" gap. `--detailed` adds subperiod stats, stationary block-bootstrap Sharpe CIs, and (with `--stochastic-paths N`) policy-sampled fan charts. Batch mode reuses cache + policy shell across run ids.

Verified weaknesses: default `--checkpoint both` (not `best`); **no run-local config load** (§3.1); **no run-local data snapshot use** (`DATA_CACHE = resolve_data_cache()` is global, no `--data-cache` flag, run-local `data_cache.npz` written by `train.py:739` is ignored); machine-readable output only for `--ensemble-prefix` (`ensemble_summary.json`) — a single `--run-id` backtest **prints to stdout only**; and `balanced_6040_nav` **raises `KeyError` if BOND10Y is absent** (`baselines.py:290-291`), called unguarded at `backtest.py:595,1103`, so `--detailed`/plots **hard-crash on `--n-assets 5`**.

### 3.6 Artifacts & Modal (the automation substrate)

`run_artifacts.py` (`RunPaths`, `new_run_id`, `resolve_data_cache`, `write_manifest`, `snapshot_data_cache`) gives a clean `Runs/<id>/` tree. The manifest records: `run_id`, `config_path`, all argparse `args`, `universe{tickers,n_assets,n_actions,obs_dim}`, `chronological_holdout{holdout_days,train_end,holdout_start,holdout_end,trainable_end,holdout_bars,date_start,date_end}`, `n_index/n_trainable_bars/n_train_bars/n_eval_bars`, `data_cache_snapshot`, and on completion `finished_at_utc`, param counts, and an `artifacts{}` map (final/best model, both VecNormalize copies, plots, tb, monitor logs, eval npz, eval_nav_history). This is a strong run-registry seed. `eval_logs/eval_nav_history.npz` makes the cliff machine-readable.

Modal (`scripts/modal_app.py`, `rlbot/modal_cloud.py`): per-GPU `n_envs`/batch profiles (T4→H100), `sync --watch` for live plots, `--pull-all`, `upload_cache`, `list_runs`, resume-from-checkpoint, and a FastAPI `/status?run_id=` returning manifest readiness + content. This is genuinely good substrate for an orchestrator — what's missing is a job/queue abstraction with crash-vs-running disambiguation (polling only).

---

## 4. Issues & Weaknesses (prioritized)

### P0 — harness integrity (do before any automated search)

- **P0-A · Walk-forward feature-state contamination + disabled purge + docs that deny it.** §2.1. Eval-selection signal is not independent of adjacent train blocks; `feature_purge_warmup` is dead; `CLAUDE.md`/`AGENTS.md` claim per-segment recompute + purge. *Fix:* add an explicit `feature_split_mode ∈ {continuous, independent}`; recompute-per-segment + apply purge for the **selection** eval; keep `continuous` as a labeled ablation; stamp the mode into the manifest; add a regression test that asserts purge-zeroing at recorded boundaries in `independent` mode.
- **P0-B · Backtest ignores the run-local config snapshot.** `scripts/backtest.py` uses the current global `config.yaml` for costs, cap, and env mechanics — so editing config silently changes old runs' OOS numbers. *Fix:* default to `load_config(Runs/<id>/config.yaml)`; add `--use-current-config` for deliberate stress tests; record the effective config hash in the summary.
- **P0-C · Backtest ignores the run-local data snapshot.** `train.py` writes `Runs/<id>/data_cache.npz` but backtest reads a global cache (`resolve_data_cache()`), with no `--data-cache` flag. Old runs become unevaluable after a cache refresh / universe change. *Fix:* prefer the run-local snapshot; add `--data-cache`; store a cache content hash in the manifest.
- **P0-D · Agent-instruction docs are live misinformation.** `CLAUDE.md` (loaded every session) + `AGENTS.md` assert a nonexistent `sync_trading_env_aliases`, 118-d obs (vs 128), 0.50 cap (vs 0.35), per-segment+purge leakage, and nonexistent `windows/`/`backtest_sweep.py` paths. *Fix:* rewrite both to current reality; ideally generate the invariants block (obs_dim, cap, action count, split mode) from code so it can't drift; add a doc-vs-code check to CI.

### P1 — measurement & defaults

- **P1-A · Reward asymmetry + no on-disk decomposition.** §3.3. *Fix:* add a callback aggregating `rew_decomp/*` means/percentiles/abs-shares to TB and JSON; re-derive participation/churn so they're non-trivial at target behaviors *or* explicitly accept inactivity-as-exposure-controller and demote the others; run a return-only / +Sortino / full-reward ablation.
- **P1-B · Narrow eval estimator; no early stopping; `best` not the default.** *Fix:* add jittered-start eval episodes (or report effective unique coverage); add patience-based early stopping after curriculum release; make `--checkpoint best` the default and warn when `latest`/`both` touches OOS; persist the stop reason + best step in the manifest.
- **P1-C · Determinism is partial and advertised as total.** `reseed_on_reset=True` + fresh entropy. *Fix:* add a `--reproducible` per-env seed-stream mode; rename the claim to "seeded framework + stochastic resets"; pin deps / image for long studies.
- **P1-D · 60/40 benchmark hard-crashes small universes.** *Fix:* skip `balanced_6040_nav` (with a printed note) when BOND10Y is absent; add an `--n-assets 5` detailed-backtest smoke test.
- **P1-E · No single-run machine-readable backtest output.** *Fix:* `--summary-json PATH`, always written, with metrics + config/data hashes + checkpoint label + benchmark stats.

### P2 — hardening & ergonomics

- **P2-A · Cap projection is best-effort.** 5 iterations + final simplex, no post-condition assert. Safe at 0.35/N=10; add a guarded final projection + a fuzz test across caps and N (extend the existing weight fuzz tests).
- **P2-B · Bootstrap stats coupled to torch.** Move `block_bootstrap_*` to a light `rlbot/stats.py` so the unit test (and any analysis tool) imports without torch.
- **P2-C · Reproducibility plumbing.** No lock file; `requirements.txt` is all `>=`. Add `uv.lock`/pinned image; document a light test-only install.
- **P2-D · External-data provenance.** Persist HY-OAS calibration coefficients + data-source metadata + cache hashes into the manifest. yfinance/FRED are fine for research, not point-in-time/survivorship-clean.
- **P2-E · No in-tree inference/paper path.** Add `scripts/infer_weights.py` (`--run-id --checkpoint best --as-of [--data-cache]` → JSON target weights + logits + live mask + config/cache/model hashes) before any broker adapter; the referenced `paper_trade/`/`ibkr_paper/` dirs do not exist in a fresh clone.

---

## 5. Feasibility & Benefits of the Auto-Research Pattern

**What "auto research" means here.** A closed loop — *hypothesis → config patch → materialize run → train → eval/backtest → ingest metrics → gate/compare → next hypothesis* — optionally driven by an LLM agent (this Claude Code harness can do it: `Agent`/`Workflow` fan-out, structured output, durable run registry). It can be in-repo tooling (`scripts/research.py`) and/or an agent loop on top.

**Feasibility: high, after the P0 fixes.** The substrate is unusually complete for a research repo: every run is a self-describing, manifest-driven, machine-callable unit (`train.py --run-id … ; backtest.py --run-id … --detailed --stochastic-paths N`); Modal offloads compute with a status endpoint; `eval_nav_history.npz`, `rew_decomp` (in-info), block-bootstrap CIs, and the manifest give quantitative signals an agent can parse without vision. A seed-ensemble script already exists.

**Benefits: high, because the open problems are exactly the tedious-but-mechanical work** an instrumented loop is good at: reward-coefficient sweeps with per-term measurement; eval-estimator variants (jittered starts, effective-coverage reporting) to test whether the "cliff" is overfitting or a measurement artifact; curriculum-phase ablations; multi-seed distributional OOS as the *primary* reported number; and — only once the 10-asset base is trustworthy — disciplined universe scaling. It would also turn `RESEARCH.md` from a hand-curated notebook into a generated report and cut Modal waste via early stopping.

**The primary risk, stated plainly:** *auto research amplifies whatever objective and leakage structure you hand it.* With P0 unfixed, a loop would (a) climb the **contaminated, narrow** validation surface faster than a human, (b) multiple-test its way to flattering OOS windows across many attempts, and (c) bootstrap from **wrong agent docs** and propagate that drift into new docs/configs. So the loop must begin as *automated bookkeeping + gated execution*, not *an agent free-optimizing OOS*.

**Three guardrails are non-negotiable before turning a loop loose:**
1. **Patch allow-list.** The agent may edit only `reward.*`, `curriculum.*`, `entropy_schedule.*`, `hyperparameters.*`, `policy.*`, and non-date `environment.*` — never holdout dates, the split, or feature plumbing — and must always go through the canonical CLIs with a fresh `--run-id`.
2. **Holdout firewall + evaluation tiers.** In-training eval for selection; dev OOS only when labeled; **final OOS read once per pre-registered candidate**, human-approved. Enforce by default in the tool, not just in prose.
3. **Pre-registration + full search-tree logging.** Every variant (including the unflattering ones) lands in a registry with its hypothesis and success criteria, so cherry-picking is visible.

---

## 6. Recommended Auto-Research Design (incremental, reuses what exists)

1. **Experiment spec** (`specs/*.yaml`): `id`, `hypothesis`, `parent`, `base_config`, `patch` (dotted keys + value grids, restricted to the allow-list), `windows`, `seeds`, `timesteps`, `checkpoint_rule`, `evaluation_tier`, `success_gates`, `budget{max_modal_hours}`. Materialize resolved configs under `Runs/<cohort>/configs/<variant>.yaml`.
2. **Run registry** (`research_registry.jsonl`): one record per train+backtest — cohort/variant/hypothesis ids, run id, git commit + dirty flag, config hash, data-cache hash, `feature_split_mode`, universe, train/eval/OOS dates, seed, budget, checkpoint path, VecNormalize path, best/final eval NAV + step, OOS metrics (if tier permits) with bootstrap CIs, benchmark metrics, turnover/exposure/drawdown, reward-decomposition summary, status + failure reason. Generate `RESEARCH.md` tables *from* this.
3. **Summary JSON everywhere** (P1-E): `train.py` writes a training summary on exit; `backtest.py --summary-json` always emits. The orchestrator never parses stdout.
4. **Evaluation tiers:** T0 static + leakage tests → T1 smoke (no OOS) → T2 short dev (in-training eval only) → T3 multi-seed/window (no final OOS) → T4 pre-registered full run, `best` checkpoint, OOS **read once**, human-gated → T5 paper/shadow. The agent sees T1–T3 freely.
5. **Orchestrator** (`scripts/research.py {plan,launch,collect,report,promote}`) that shells to the existing `train.py`/`backtest.py`/`modal_app.py` — no rewrite of the training stack. With this Claude Code harness, a `Workflow` can fan T1–T3 variants across Modal and a verifier sub-agent can adversarially check any "improvement" against seed variance before it's allowed to print a result.
6. **First questions (test the harness, not alpha):** independent vs continuous feature split; reward decomposition ablation; participation/churn/inactivity re-scaling grid; eval jitter & early-stop patience; deterministic vs entropy-reset seed variance; RecurrentPPO vs feed-forward PPO on the same env.

---

## 7. Continued Evolution Roadmap

**Phase 0 — Seal the harness (gate for everything else).** P0-A…D: feature-split mode + purge restored for selection; backtest binds run-local config + data; `best` default; rewrite `CLAUDE.md`/`AGENTS.md` from code; add summary-JSON outputs; add the doc-vs-code CI check.

**Phase 1 — Measurement hardening.** Reward-decomposition aggregation (P1-A); eval jitter + early stopping (P1-B); reproducible seed-stream mode + lock file (P1-C, P2-C); 60/40 graceful skip + small-N smoke (P1-D); move bootstrap to a light module (P2-B); cap post-assert + fuzz (P2-A).

**Phase 2 — Auto-research MVP.** Spec format + JSONL registry + collect/report/promote + tiered gates, OOS human-gated; generate `RESEARCH.md` from the registry.

**Phase 3 — Method development (only after the harness is sound).** Re-test the validation cliff under `independent` split + jittered eval; reward/curriculum ablations; recurrent-vs-feedforward; then *gradual* universe scaling with the same firewall.

**Phase 4 — Deployment-oriented inference.** Audited `infer_weights.py` with recurrent warmup + provenance; paper-trading harness; broker adapter last; persist data/HY-OAS provenance (P2-D/E).

---

## 8. Bottom Line

The asset worth protecting here is not the LSTM policy — it is the beginnings of a *disciplined research operating system* for portfolio RL, and four independent reviews now agree on that. My net read after reading the code rather than the docs:

- On the one question where the two same-day reviews collide, **Codex is right**: the walk-forward split slices continuous features and the purge is off. It is not an OOS leak, but it weakens the exact signal that picks the shipped model, and the agent docs still claim the opposite.
- The tests are fine; the *onboarding and a torch-coupling* are not.
- The reward isn't inert, it's **lopsided** — cash is bludgeoned, exposure and churn are barely touched.

None of this is fatal; all of it is mechanical to fix. And the sequencing the priors urged still holds, sharpened by the auto-research lens: **make the experiment unambiguous first** —

```
exact config bound at inference · exact data snapshot bound at inference
exact feature-split semantics · exact checkpoint rule · exact metrics record
exact, code-generated agent instructions
```

— *then* let an auto-research loop run. Built in that order, the pattern is an excellent fit and the single highest-leverage investment available. Built before it, the loop will mostly automate overfitting and documentation drift, faster and more convincingly than a human ever could.

---

*Companion to docs/codex-review-20260605.md and docs/grok-review-20260605.md (same day) and the May-31 priors. Read alongside README.md, docs/RESEARCH.md, and the source. Findings verified against code at commit-time of 2026-06-05; line references are to the files as read during this review.*
