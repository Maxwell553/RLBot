# RLBot (MarketTrainer) — Critical Review

**Date:** 2026-06 (review performed on current main branch checkout)  
**Scope:** Full repository: code, configuration, data pipeline, training loop, evaluation, documentation, and research claims.  
**Purpose:** Independent technical and methodological assessment to guide continued development.

---

## Executive Summary

RLBot is a **high-quality research prototype** for applying Recurrent PPO (LSTM policy) to daily multi-asset portfolio allocation. It demonstrates unusually strong engineering discipline around **temporal leakage prevention**, reproducibility, and realistic market microstructure (transaction costs, next-open execution, domain randomization).

**Strengths that stand out:**
- Meticulous anti-leakage architecture (per-segment features, purge windows, chronological holdouts, causal lags).
- Sophisticated training curriculum and callbacks (fee ramp + progressive DR, mandatory entropy decay, best-by-validation-NAV selection).
- Excellent documentation of both methodology and empirical limitations (the RESEARCH.md "validation NAV cliff" analysis is refreshingly honest).
- Config-driven everything with per-run snapshots.

**Critical weaknesses:**
- The core phenomenon the authors themselves document — **validation NAV peaks early (~9–13M steps) while training reward keeps rising** — is only partially mitigated by manual "best checkpoint" selection. This suggests the alternating-block walk-forward still permits overfitting that the LSTM exploits.
- Several **documented components are missing** from the repository (paper_trade, ibkr_paper).
- The system is **tightly coupled to a fixed 10-asset universe**. Almost nothing is modular with respect to N_assets.
- No production inference path exists in the tree despite architectural claims.

**Overall recommendation:** The *methodology and engineering patterns* are worth preserving and generalizing. The current 10-asset implementation should be treated as a **reference implementation and testbed**, not as production portfolio machinery. Significant refactoring is required before it can support larger or dynamic universes.

---

## 1. What Is Done Well

### 1.1 Leakage Discipline (Best-in-Class for This Style of Research)
- `train_test_split_alternating` (data_utils.py:579) computes RSI/MACD/fracdiff **per contiguous block** on raw OHLCV only. No EWM state or fracdiff memory leaks across train/eval boundaries.
- `feature_purge_warmup` (default 25 bars) neutralizes indicators at every join.
- `reserve_chronological_holdout` is called **before** any split; the OOS tail is never visible to training code.
- Execution is strictly causal: features at t use data through close[t-obs_lag], trades execute at open[t+1], P&L and costs realized at close[t+1].
- Fracdiff (d=0.4) is a thoughtful choice vs raw returns or integer differencing.

These choices are correctly emphasized in RESEARCH.md and make the OOS numbers more credible than most RL trading papers.

### 1.2 Training Mechanics and Anti-Overfitting
- `EvalNavBestModelCallback` saves on **mean ending NAV on validation episodes**, not episodic reward. This is the right objective for a portfolio agent.
- `AdaptiveEntropyCallback` performs **mandatory** cosine decay from explore_ent → final_ent starting at 45% of the run, independent of eval performance. Prevents the common "entropy collapses too early" pathology.
- `TradingCurriculumCallback` implements a principled multi-phase schedule (frictionless → fee ramp → progressive DR widening → full DR + churn).
- Domain randomization (Beta-centered fee_scale + discrete obs_lag) after curriculum release is sound.
- Deterministic seeding + cuDNN flags + VecNormalize snapshotting give strong reproducibility guarantees.

### 1.3 Reward and Environment Realism
- Reward includes **Sortino differential vs a cap-weighted benchmark** (not just raw return), participation bonus, inactivity penalties, quadratic drawdown penalty, and a **decoupled** churn penalty (CHURN_LAMBDA is not multiplied by REWARD_SCALE).
- Per-asset cost vectors (slippage, tx_fee, annual_holding_cost) are first-class in config.yaml.
- The clip-and-redistribute cap logic (50% per risky asset by default) + long-only simplex is correctly implemented and has reasonable test coverage (tests/test_environment.py).

### 1.4 Documentation and Research Hygiene
- RESEARCH.md is unusually good: it shows training dynamics plots that reveal the validation-collapse problem, states explicit assumptions and limitations, and does not overclaim statistical significance.
- config.yaml is the single source of truth and is snapshotted per run.
- Walk-forward window definitions are explicit and have a validator script.

---

## 2. Issues, Weaknesses, and Errors

### 2.1 Missing or Stale Components (Documentation vs Reality Gap)
The README and architecture diagram reference:
- `paper_trade/paper_trade.py`
- `ibkr_paper/` IBKR execution driver

**Neither directory exists** in the repository (confirmed via filesystem inspection and git status). The CLI entry points in pyproject.toml only cover train/backtest. This is a clear documentation debt that misleads readers about deployment readiness.

### 2.2 The Validation NAV Cliff — Partially Addressed, Still Dangerous
All three walk-forward windows exhibit the same pattern (documented in RESEARCH.md):
- Training reward and training episode NAV rise through 65M steps.
- Validation ending NAV peaks early (≈9–13M steps for W1/W3, later for W2) then stagnates or declines.
- The **final 65M checkpoint is frequently materially worse OOS** than the early "best" checkpoint.

The mitigation (EvalNavBestModelCallback + manual comparison of best vs final) is better than nothing, but:
- No automated early stopping on validation NAV.
- No analysis of *why* the LSTM hidden state overfits the alternating blocks (possible temporal correlation across segment boundaries despite purge).
- Continuing to 65M after the peak wastes compute and risks publishing the worse checkpoint if someone forgets the manual step.

This is the single most important empirical red flag in the current system.

### 2.3 Hard-Coded 10-Asset Coupling (Major Architectural Limitation)
`N_ASSETS = 10`, `N_ACTIONS = 11`, `N_MACRO = 4` are structural constants in multiple files. Observation construction, cost arrays, benchmark weights, and the action clipping loop all assume exactly these dimensions.

Consequences:
- Adding/removing one asset requires coordinated changes across data_utils, trading_env, config, rl_config, tests, and every saved VecNormalize/model.
- The observation vector layout (118-d) is implicit and versioned only by "run_id folklore."
- Impossible to experiment with different menus without forking the entire codebase.

### 2.4 Action Mapping Edge Cases
`portfolio_weights_from_action` (trading_env.py:83) runs a 5-iteration clip-and-redistribute loop, then falls back to dumping overflow into cash. While tests cover several cases, the logic is:
- Not obviously correct under simultaneous cap hits on 6+ assets.
- Has magic numbers (5 iterations, 1e-12 tolerances).
- The redistribution proportional to current underflow can create second-order concentration.

The 50% cap with 10 assets mathematically allows up to 5× cap in risky exposure before cash absorbs the rest — this is by design but should be more explicitly validated in stress tests.

### 2.5 Brittle External Data Handling
- HY OAS uses a one-time FRED fetch + HYG/IEF proxy calibration at `fetch_aligned_daily` time. The calibration coefficients are **not persisted** with the cache; re-running on different FRED data changes history.
- yfinance corporate-action quality is acknowledged as a limitation but not instrumented (no adjustment flags or survivorship metadata stored per bar).
- No point-in-time guarantee for the static 10-asset menu itself (delisted or low-liquidity periods are simply absent).

### 2.6 Test Coverage Gaps
- Strong unit tests for the portfolio weight mapping and basic math.
- **No integration tests** that run a short training loop and assert leakage invariants or NAV monotonicity under curriculum.
- No property-based or fuzz tests on long episodes with block boundaries.
- The `windows/validate_split.py` and RESEARCH.md examples can drift independently (hard-coded dates).

### 2.7 Other Medium Issues
- Many module-level "legacy aliases" in trading_env.py that are mutated by `sync_trading_env_aliases`. Easy source of stale imports.
- Backtest VecNormalize discovery has many fallback paths; easy to accidentally load training-time statistics for an OOS run if run_id inference is wrong.
- No hyperparameter search, ablation framework, or automated regime-sweep tooling beyond the manual calendar backtest_sweep.
- `stop_loss_fraction=0.45` (55% drawdown) is effectively never triggered on a diversified 10-asset book over 63-day episodes; it is decorative.

---

## 3. Recommendations for Continued Development

### High Priority (Do These First)
1. **Close the validation-NAV loop.** Implement patience-based early stopping on `mean_ending_nav` (or at minimum a hard "stop if no improvement for X evals after 15M steps"). Make the best checkpoint the *default* artifact.
2. **Modularize the asset universe.** Define assets, costs, and macros in config (or a separate `universe.yaml`). Generate observation dimensions, cost arrays, and benchmark weights dynamically. This is a prerequisite for any larger-universe work.
3. **Add a clean, documented inference path.** Even if paper/IBKR drivers stay out-of-tree for now, ship a `inference.py` or `rollout.py` that:
   - Loads a specific run_id + best/final choice
   - Freezes VecNormalize correctly
   - Handles recurrent state reset / warm-up rules
   - Emits target weights with audit metadata
4. **Version the observation specification.** Add an explicit `obs_layout_version` or schema hash to saved models and the config snapshot.

### Medium Priority
5. Harden the action clipping logic (more iterations, better redistribution math, or switch to a projected-gradient / softmax-with-temperature + post-hoc renormalization approach that is easier to reason about).
6. Add automated leakage regression tests (e.g., assert that feature values immediately after a block boundary match what a fresh per-segment computation would produce).
7. Persist the HY OAS calibration coefficients (and any other derived parameters) inside the data cache or manifest.
8. Improve the RESEARCH.md training-dynamics analysis: quantify the correlation between validation NAV peak timing and curriculum phase transitions.
9. Add a `uv.lock` or `poetry.lock` (or at least a reproducible Docker image) so that 65M-step runs are bit-reproducible across machines.

### Longer-Term / Strategic
10. Treat the current 10-asset system as a **methodology testbed**. Extract the valuable pieces (anti-leakage split, curriculum, best-by-wealth callback, cost-aware reward, deterministic training harness) into a small reusable library.
11. Before adding more assets, prove that the validation-collapse problem is solved on the existing menu. Otherwise you are simply scaling up an overfitter.
12. Consider whether a pure RecurrentPPO on engineered price features is the right inductive bias once you leave the "10 always-live macro proxies" regime. Single-name equities have very different statistical properties.

---

## 4. Specific Code Locations Worth Attention

| File | Lines | Issue |
|------|-------|-------|
| `trading_env.py` | 83–117 | Iterative clip-and-redistribute; magic 5-iteration limit |
| `trading_env.py` | 40–58 | Module-level aliases + `sync_trading_env_aliases` mutation pattern |
| `train.py` | 134–199 | `EvalNavBestModelCallback` — excellent, but no early stopping |
| `train.py` | 218–304 | `AdaptiveEntropyCallback` — mandatory decay is good; the improvement counter is only advisory |
| `data_utils.py` | 579–698 | `train_test_split_alternating` — the core anti-leakage logic; needs more regression tests |
| `rl_config.py` | 353–376 | `sync_trading_env_aliases` — central point of coupling |
| `README.md` + `RESEARCH.md` | many | References to non-existent `paper_trade/` and `ibkr_paper/` |
| `config.yaml` | 33–37 | Per-asset cost vectors are 10-element literals; will not scale |

---

## Conclusion

RLBot's **engineering hygiene and methodological caution** are materially better than the median RL-for-trading research artifact. The authors have done the hard, unglamorous work of making leakage difficult and reproducibility easy.

However, the system currently demonstrates that even with those safeguards, **the LSTM policy overfits the particular alternating-block structure** of the training data. That finding is valuable. It should be treated as a **result**, not a bug to be papered over while scaling the same architecture to 50 or 800 names.

The highest-leverage next step is **not** "add more assets." It is:
1. Fix the early-peak / late-collapse dynamic on the existing 10-asset problem.
2. Modularize the asset dimension so experiments become cheap.
3. Extract the good patterns into infrastructure that future, larger RL efforts (or hybrid LLM+RL efforts) can reuse.

The codebase is worth investing in, but only if the above structural issues are addressed rather than worked around.

---

*End of RLBot Critical Review*