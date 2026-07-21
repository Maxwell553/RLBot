# CONTEXT.md — Empirical Run Analysis for Environment Improvement

**Audience:** a separate agent tasked with improving MarketTrainer’s *environment / reward / observation* design (not just hyperparameter tuning).  
**Generated:** 2026-07-19 from local `Runs/` artifacts (`backtest_summary.json`, run `config.yaml`, `manifest.json`, plus `docs/RESEARCH.md` and `Runs/cohort_vs_benchmark.json`).  
**Scope:** 80 completed OOS backtests across cohorts `W*_612` … `W*_627` (5 walk-forward windows × 16 cohort IDs). One incomplete tree `W1_625_a` (training interrupted, no backtest) is excluded.

This document is evidence-first. Prefer the tables and statistics below over narrative memory. Canonical protocol details live in `README.md` / `AGENTS.md`; published cohort narrative lives in `docs/RESEARCH.md`.

---

## 1. How to use this document

1. Treat **OOS holdout metrics** (`total_return`, `sharpe`, `max_drawdown`, `deflated_sharpe`) as the only generalization signal.
2. Treat in-training eval NAV / robust score as **model-selection** diagnostics only (they do not validate OOS).
3. **Do not** optimize on contaminated cohorts (`622`–`625`; see §3).
4. When proposing env/reward changes, keep the OOS firewall: do not patch universe, costs, holdout dates, or split geometry via research specs (`rlbot/research/spec.py` rejects those).
5. Success criteria for an “improved environment” should be stated as **distributional** wins (median Sharpe / chained return / drawdown across W1–W5 and ≥2 seeds), not a single-window spike.

---

## 2. Protocol constants (shared by nearly all runs)

| Knob | Typical value |
| --- | --- |
| Model | RecurrentPPO `MlpLstmPolicy`, LSTM 64×2, MLP `[128,128]` |
| Universe | N=10 (SP500, GOLD, OIL, EURUSD, USDJPY, NIKKEI, FTSE, BOND10Y, COPPER, EM) |
| Timesteps | 50M |
| `feature_split_mode` | `independent` |
| `obs_lag` (OOS) | 1 |
| Action | cash + N logits → softmax → live-mask → per-asset cap → long-only simplex |
| Default `max_single_asset_weight` | **0.20** (cohorts 626/627 use **0.60**) |
| Reward benchmark | equal feasible weights `1/N` |
| Best-checkpoint selection | robust score vs `equal_weight_daily` after `fee_ramp_end` |
| Turnover penalty | `0.007` |
| `reward_scale` | 2000 |
| `benchmark_excess_scale` | 600 |
| `sortino_downside_floor` | 0.001 |
| Early-stop patience | 8 (rarely triggered; 77/80 runs finished without early stop) |

Walk-forward windows:

| Window | OOS | Passive EW (median) | SPY sleeve (median) |
| --- | --- | ---: | ---: |
| W1 | 2016–2017 | +28.4% | +46.4% |
| W2 | 2018–2019 | +2.7% | +18.0% |
| W3 | 2020–2021 | +17.3% | +52.5% |
| W4 | 2022–2023 | +5.3% | +7.3% |
| W5 | 2024–2025 | +33.2% | +45.7% |

Chained W1–W5 passive reference (compounding the median window returns): **EW ≈ +117%**, **SPY ≈ +312%**.

OOS burn: `oos_trials_for_window` in recent summaries is ~**46–49** distinct model reads per window. Deflated Sharpe is almost always low (see §5).

---

## 3. Validity filter — contaminated vs usable

### Contaminated (discard for method conclusions)

`docs/RESEARCH.md` documents a `vol_penalty_scale` scaling bug: the term is multiplied by `reward_scale` (2000). With `vol_penalty_scale: 300`, the effective coefficient is ~600,000 and **dominates** the reward (training rewards ≈ −1200/step).

| Cohort | `vol_penalty_scale` | Verdict |
| --- | ---: | --- |
| `622`–`625` | 300.0 | **Invalid** — do not use for env design decisions |
| `626`–`627` | 0.15 (fixed) | Usable, but also changed **cap to 0.60** (confounded) |
| `612`–`621` | absent / pre-term | Usable as pre-vol-penalty grid (see caveats) |

Raw contaminated chained returns still look “fine” (`622` +83%, `623` +149%, `624` +120%, `625` +176%) — **that is not evidence the penalty worked**; the reward signal was broken.

### Caveats on older “clean” cohorts

- `docs/RESEARCH.md` flags **`W*_612` as mixed-era** (W1–W2 pre-rebalance code / different best-checkpoint rule; W3–W5 post-rebalance). Treat 612 as exploratory.
- `W*_615` was split mid-cohort on commit metadata historically; current backtest summaries stamp a uniform commit hash — still treat seed/exposure cells as thin samples.
- Backtest `git_commit` on disk is often the *backtest-time* tree, not a perfect training-time pin. Prefer run-local `config.yaml` + `manifest.json` for settings.

### Usable analysis sets used below

- **Set A — Published exposure×seed grid:** cohorts `612`–`617` (n=30). Matches `docs/RESEARCH.md`.
- **Set B — Extended cap=0.20 grid:** `612`–`621` (n=50). Adds exposure 60/120 and seed 101.
- **Set C — Post-fix cap experiment:** `626`–`627` (n=10). `vol_penalty_scale=0.15`, `max_single_asset_weight=0.60`.
- **Excluded:** `622`–`625`, `W1_625_a`.

---

## 4. Cohort fingerprints (what actually varied)

Primary intentional axes across the tree:

| Cohort | Seed | `exposure_risk_penalty_scale` | `vol_penalty_scale` | Cap | Config recipe | Chained OOS | Mean Sharpe | Mean max DD | Mean cash |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 612 | 0 | 80 | — | 0.20 | `config.yaml` | **+140.7%** | 1.12 | −10.7% | 8% |
| 613 | 0 | 100 | — | 0.20 | `config.yaml` | **+186.2%** | 1.05 | −11.2% | 3% |
| 614 | 0 | 90 | — | 0.20 | `config.yaml` | +110.7% | 0.66 | −16.0% | 1% |
| 615 | 42 | 90 | — | 0.20 | `config.yaml` | +135.2% | 0.84 | −15.0% | 4% |
| 616 | 42 | 80 | — | 0.20 | `config.yaml` | +82.1% | 0.81 | −11.1% | 17% |
| 617 | 42 | 100 | — | 0.20 | `exp100_s42.yaml` | +135.5% | 0.86 | −15.5% | 9% |
| 618 | 101 | 60 | — | 0.20 | `exp60_s101.yaml` | **+205.0%** | 0.99 | −16.6% | 3% |
| 619 | 101 | 120 | — | 0.20 | `exp120_s101.yaml` | +163.5% | 0.87 | −15.6% | 1% |
| 620 | 42 | 60 | — | 0.20 | `exp60_s101.yaml` | +102.7% | 0.68 | −15.4% | 12% |
| 621 | 42 | 120 | — | 0.20 | `exp120_s101.yaml` | +131.0% | 0.78 | −16.1% | 4% |
| 622† | 42 | 80 | **300** | 0.20 | `config.yaml` | +82.8%† | 0.82† | −12.5%† | 25%† |
| 623† | 42 | 80 | **300** | 0.20 | `config.yaml` | +148.9%† | 0.88† | −13.4%† | 7%† |
| 624† | 101 | 60 | **300** | 0.20 | `exp60_vol_s101.yaml` | +119.8%† | 0.70† | −14.7%† | 4%† |
| 625† | 0 | 100 | **300** | 0.20 | `exp100_vol_s0.yaml` | +175.9%† | 0.96† | −12.5%† | 9%† |
| 626 | 0 | 80 | 0.15 | **0.60** | `exp80_cap60_s0.yaml` | +174.7% | 0.82 | −16.5% | **0%** |
| 627 | 42 | 80 | 0.15 | **0.60** | `exp80_cap60_s0.yaml` | +125.4% | 0.55 | −18.5% | 1% |

† Contaminated — listed only for inventory.

Almost everything else (costs, curriculum fractions, entropy schedule, network size, LR, turnover penalty, concentration target, inactivity/participation terms) was held constant. So historical “search” has mostly been **exposure scale × seed**, then a broken vol-penalty attempt, then a **cap relaxation**.

---

## 5. Statistical findings (usable runs)

### 5.1 Overall (Set B: cap=0.20, n=50)

| Metric | Value |
| --- | ---: |
| Mean OOS return / window | +19.7% |
| Mean Sharpe | 0.87 |
| Median Sharpe | ~0.78–0.88 (window-dependent) |
| Mean max DD | ≈ −14.3% |
| Fraction windows with return > 0 | 49/50 (only `W4_616` negative in this set) |
| Mean deflated Sharpe | **0.25** (median ~0.17; **0% > 0.95**) |

Window difficulty (Set B means):

| Window | Mean ret | Mean Sharpe | Mean max DD | Beat EW rate* |
| --- | ---: | ---: | ---: | ---: |
| W1 | +29.4% | 1.52 | −6.4% | 4/10 |
| W2 | +4.8% | 0.23 | −16.3% | 6/10 |
| W3 | +28.7% | 0.86 | −23.0% | **10/10** |
| W4 | +4.0% | 0.16 | −14.3% | 5/10 |
| W5 | +31.6% | 1.56 | −11.7% | 4/10 |

\*Beat-EW available for cohorts `612`–`621` via `Runs/cohort_vs_benchmark.json`.

**Interpretation:** absolute returns are often positive because several OOS windows are strong beta regimes. Risk-adjusted edge vs equal-weight is uneven; vs SPY sleeve the agent almost never wins (typical beat-SPY ≤ 2/5 per cohort).

### 5.2 Published grid (Set A: `612`–`617`)

Exposure × seed chained return:

| | Seed 0 | Seed 42 |
| --- | ---: | ---: |
| Exp 80 | 612 **+140.7%** | 616 +82.1% |
| Exp 90 | 614 +110.7% | 615 +135.2% |
| Exp 100 | 613 **+186.2%** | 617 +135.5% |

- At seed 0, **exp 100 (613)** is strongest.
- At seed 42, exp 90/100 tie on chained return (~+135%); exp 80 is weak (W4 −5.7%).
- Mean Sharpe range across the six cells: **0.66 – 1.12**.
- Seed deltas at fixed exposure are huge (**−24 to +59 pp** chained). Two seeds are not enough to crown a setting.

Beat equal-weight (from cohort_vs_benchmark):

| Cohort | Beat EW | Mean excess vs EW |
| ---: | ---: | ---: |
| 613 | 4/5 | **+7.5%** |
| 612 | 3/5 | +2.2% |
| 615 / 617 | 3/5 / 2/5 | +2.0% |
| 614 | 2/5 | −0.8% |
| 616 | 2/5 | **−3.9%** |

### 5.3 Extended exposure extremes (Set B extras)

| Cohort | Exp | Seed | Chained | Notes |
| ---: | ---: | ---: | ---: | --- |
| **618** | 60 | 101 | **+205.0%** | Best chained in the tree; W1 +48.7% / W5 +38.3%; still W3 DD −28.8% |
| 619 | 120 | 101 | +163.5% | Strong W5 (+52.7%) |
| 620 | 60 | 42 | +102.7% | Same exp as 618, much weaker → **seed dominates** |
| 621 | 120 | 42 | +131.0% | Mid-pack |

Pearson correlation of `exposure_risk_penalty_scale` vs OOS Sharpe on clean runs ≈ **0.02** (noise). Exposure is not a stable linear lever; interactions with seed/window dominate.

### 5.4 Cap 0.20 vs 0.60 (Set C vs matched exp-80)

At `exposure_risk_penalty_scale=80`:

| Cohort | Cap | Seed | Chained | Mean Sharpe | Mean DD | Mean cash |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 612 | 0.20 | 0 | +140.7% | **1.12** | −10.7% | 8% |
| 616 | 0.20 | 42 | +82.1% | 0.81 | −11.1% | 17% |
| 626 | 0.60 | 0 | +174.7% | 0.82 | −16.5% | ~0% |
| 627 | 0.60 | 42 | +125.4% | **0.55** | −18.5% | ~1% |

Raising the cap increased chained return for seed 0 but **worsened mean Sharpe and drawdowns**, and collapsed cash to ~0%. Behaviorally the policy concentrates harder (see §6). **Do not treat 626’s chained return as a clear env win** without risk-adjusted and multi-seed confirmation.

### 5.5 Selection quality: in-training score vs OOS

Across clean runs, correlation of `best_eval_score` with OOS Sharpe ≈ **−0.13** (with OOS return ≈ −0.14).

Implications for an env-improvement agent:

- The robust eval objective is poorly aligned with holdout Sharpe (or is dominated by scale/noise).
- Improving the **eval selection metric / benchmark / penalties** may matter as much as changing the step reward.
- Negative robust scores with large magnitude (means around −1.4e4 to −3.0e4 in stored units) are normal under current coefficients; compare ranks within a run, not absolute values across eras.

### 5.6 Deflated Sharpe / multiple testing

No clean run clears DSR > 0.95. With ~50 OOS reads per window, nominal Sharpes in the 1.5–2.0 range on single windows are **not** statistically decisive. Prefer:

- multi-window chained metrics,
- seed medians,
- pre-registered gates in `specs/*.yaml` + `research.py` tiers,

over celebrating one W1/W5 printout.

---

## 6. Portfolio behavior associated with outcomes

Diagnostics from `backtest_summary.json → portfolio_diagnostics` (clean runs):

| Behavior metric | Corr vs Sharpe | Corr vs return | Corr vs max DD |
| --- | ---: | ---: | ---: |
| Mean cash fraction | −0.10 | **−0.24** | +0.20 (more cash ↔ milder DD) |
| Mean gross exposure | +0.10 | **+0.24** | −0.20 |
| Mean effective N | +0.17 | +0.17 | +0.16 |
| Mean HHI | **−0.20** | −0.17 | −0.14 |
| Top-3 concentration | −0.18 | −0.17 | −0.15 |
| Mean turnover | −0.09 | −0.08 | ~0 |
| Cap-hit fraction | −0.03 | −0.14 | ~0 |

### Typical sleeves (mean weights across a cohort’s five windows)

High performers often overweight **NIKKEI / EM / GOLD** and underweight FX (EURUSD especially). Examples:

| Cohort | Cash | SP500 | GOLD | NIKKEI | EM | EURUSD | Notes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 613 (strong) | 3% | 14% | 11% | **18%** | 12% | 2% | Low cash, equity-tilted |
| 618 (best chain) | 3% | **17%** | 4% | 16% | **18%** | 1% | Aggressive risk-on |
| 616 (weak) | **17%** | 8% | 14% | 13% | 15% | 4% | Highest cash; W4 went to 50% cash |
| 626 (cap 0.60) | **0%** | 8% | 16% | **22%** | 17% | 0% | Near-full invested; concentrates |
| 627 (cap 0.60) | 1% | 3% | 5% | **25%** | **25%** | 0% | Extreme NIKKEI+EM; worst mean Sharpe |

### Training-plot pattern (example `W2_626`)

Post-fee-ramp eval shows high gross exposure, low turnover, and **effective N drifting toward ~3.5–4** (below `concentration_target_eff_assets: 6`). Episode-end NAV variance is extreme (≈ $100k → $600k). This matches OOS: the agent learns a concentrated, high-beta sleeve rather than a stable diversified allocator.

---

## 7. Hard problems the environment is not solving

1. **W4 (2022–2023) is the stress test.** Clean returns cluster near flat (−5.7% to +9.0%). Policies that “win” W4 often do so with luck/seed, not a repeatable defensive skill. High cash in W4 sometimes helps DD (`W4_616` 50% cash, −5.7% ret) but destroys chained performance.
2. **W2 is chronically low Sharpe** (mean ~0.23). Positive absolute returns can still be bad risk-adjusted years.
3. **W3 drawdowns are severe** (often −20% to −29%) even when return is strong — COVID rebound beta, not skillful hedging.
4. **SPY sleeve still dominates** in W1/W3/W5. The agent is a diversified multi-asset book fighting a pure-equity bull benchmark; reward excess is vs equal-weight, not vs SPY — agents can look good on EW excess while lagging the product users mentally compare to.
5. **Seed variance ≫ knob variance** for exposure. Any env change needs multi-seed evaluation or it will overfit noise.
6. **Eval↔OOS misalignment.** Best-checkpoint selection may be locking in policies that do not generalize.
7. **Concentration target is soft.** Penalty toward eff-N=6 is not delivering OOS diversification; HHI/top-3 still anti-correlate mildly with Sharpe.
8. **Inactivity vs participation balance** may be wrong-signed in stress: high cash helps DD but is heavily penalized, pushing agents to stay invested into drawdowns (especially with cap 0.60).

---

## 8. Full clean per-window return grid (% OOS)

Cohorts left-to-right: 612 613 614 615 616 617 618 619 620 621 626 627

| Win | 612 | 613 | 614 | 615 | 616 | 617 | 618 | 619 | 620 | 621 | 626 | 627 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W1 | +22.0 | +29.5 | +28.4 | +26.7 | +28.4 | +26.4 | **+48.7** | +28.8 | +26.5 | +28.4 | +38.3 | +40.1 |
| W2 | +9.6 | +1.0 | +0.1 | +10.9 | +3.3 | +10.7 | +3.4 | +1.2 | +2.2 | +5.0 | +5.9 | −0.0 |
| W3 | +33.6 | **+54.4** | +20.7 | +17.8 | +17.9 | +38.8 | +31.6 | +23.4 | +20.5 | +28.8 | +39.0 | +29.4 |
| W4 | +8.1 | +5.8 | +7.9 | +1.6 | **−5.7** | +2.0 | +9.0 | +7.4 | +2.3 | +1.3 | +6.6 | −4.2 |
| W5 | +24.7 | +34.0 | +25.9 | +39.9 | +23.5 | +18.9 | +38.3 | **+52.7** | +27.2 | +31.2 | +26.5 | +29.8 |

Mean Sharpe by same columns:

| Win | 612 | 613 | 614 | 615 | 616 | 617 | 618 | 619 | 620 | 621 | 626 | 627 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W1 | 1.41 | 1.51 | 1.38 | 1.42 | 1.57 | 1.60 | **2.05** | 1.26 | 1.41 | 1.57 | 1.69 | 1.35 |
| W2 | 0.58 | 0.07 | 0.01 | 0.49 | 0.17 | 0.48 | 0.15 | 0.06 | 0.10 | 0.24 | 0.24 | 0.00 |
| W3 | 0.88 | **1.93** | 0.53 | 0.60 | 0.66 | 1.09 | 0.78 | 0.62 | 0.68 | 0.78 | 0.95 | 0.84 |
| W4 | 0.42 | 0.23 | 0.30 | 0.07 | −0.37 | 0.11 | 0.34 | 0.28 | 0.13 | 0.08 | 0.29 | −0.12 |
| W5 | **2.30** | 1.52 | 1.06 | 1.60 | 1.99 | 1.02 | 1.63 | 2.11 | 1.10 | 1.25 | 0.94 | 0.66 |

---

## 9. Hypotheses worth testing (environment-focused)

Ranked for an agent that can change `trading_env.py` / reward terms / obs features / eval selection — not just `exposure_risk_penalty_scale`.

### H1 — Fix risk-off optionality without cash cliffs
**Problem:** inactivity penalties + zero cash yield push full investment; stress windows then show brutal DD.  
**Ideas:** state-dependent inactivity (weaker when realized vol / VIX / drawdown elevated); small `cash_daily_yield`; or replace cash penalties with a *target exposure band* reward.  
**Success:** improve median W2/W4 Sharpe and mean max DD without collapsing W1/W5 returns; keep seed-median chained ≥ current Set A median (~+135%).

### H2 — Make concentration penalty bite
**Problem:** eff-N drifts to ~3–4; HHI anti-correlates with Sharpe; cap 0.60 made this worse.  
**Ideas:** stronger `concentration_penalty`, higher target eff-N, or a hard softer-max on HHI in the action projection (env constraint vs reward). Prefer constraints over huge reward scales (see vol-penalty disaster).  
**Success:** raise mean effective N toward 5–7 in OOS diagnostics with non-worse median Sharpe.

### H3 — Align eval selection with OOS goals
**Problem:** `best_eval_score` ↔ OOS Sharpe correlation is negative.  
**Ideas:** include drawdown/turnover/cash-path terms in selection; evaluate on *independent* feature-split blocks more aggressively; reduce stitched-blend overweighting continuous memory; try selecting on Sortino excess rather than NAV dollars.  
**Success:** positive rank correlation between selection score and holdout Sharpe across a frozen seed grid.

### H4 — Regime-conditioned exposure penalty (replace scalar sweep)
**Problem:** scalar `exposure_risk_penalty_scale` ∈ {60…120} has ~zero linear correlation with OOS Sharpe; seed noise dominates.  
**Ideas:** penalty based on asset-level downside beta / corr to SPY; asymmetric penalty only when gross exposure is high *and* vol rising; remove the need for another exposure grid.  
**Success:** reduce cross-seed chained-return gap at fixed settings (currently tens of pp).

### H5 — Obs / feature gaps for defensive skill
**Problem:** agent fails W4 and deep W3 DD despite macro inputs (DXY/TNX/VIX/HY).  
**Ideas:** richer causal drawdown/vol-of-vol features; explicit distance-to-cap / turnover-budget features already partially present — audit whether the policy can “see” its own risk state; consider longer LSTM memory only if eval proves undercapacity (do not jump architecture first).  
**Success:** W4 median Sharpe > 0.3 across ≥3 seeds without sacrificing W5.

### H6 — Do not re-introduce large multiplicative reward terms carelessly
**Problem:** `vol_penalty_scale=300` with `× reward_scale` wiped the learning signal.  
**Rule:** any new penalty must be dimensionally checked in *reward units per step* against return/excess/drawdown terms (typical healthy per-step reward settles near mid-single-digits after curriculum; spikes to hundreds/thousands are bugs). Log `rew_decomp/*` before claiming a term is “small.”

### H7 — Cap 0.20 is probably still the right research default
**Evidence:** cap 0.60 raised concentration and DD; mean Sharpe fell (esp. seed 42).  
Unless the product goal explicitly allows 60% single-name risk, keep **0.20** and improve diversification/risk-off inside that simplex.

---

## 10. Suggested experiment protocol for the improving agent

1. **Freeze a baseline cell:** `exposure_risk_penalty_scale=100`, seeds `{0,42,101}`, cap `0.20`, `vol_penalty_scale=0.15`, 50M steps, windows W1–W5 (or a cheaper screen on W2+W4 first, then promote).  
2. Change **one env/reward mechanism per cohort ID**.  
3. Report: per-window return/Sharpe/DD, chained return, mean/median Sharpe, mean DD, mean cash, mean eff-N, beat-EW rate, and DSR context (ledger trial count).  
4. **Promotion bar (recommended):**  
   - median Sharpe across 5 windows × 3 seeds ≥ baseline median, **and**  
   - mean max DD not worse by >2 pp, **and**  
   - W4 median Sharpe improved, **and**  
   - no reward-decomp domination (per-step term magnitudes within design band).  
5. Keep **W6 embargoed**.  
6. Prefer `scripts/research.py` tiers 1–3 before any new OOS burns; budget holdout reads.

Baseline numbers to beat (Set A medians / strong cells):

- Chained return seed-median target: **≥ +135%** (615/617 level) with DD mean **≥ −15%** (less negative is better).  
- Stretch: approach **618’s +205%** *without* 618’s −28.8% W3 DD and with ≥2 seeds agreeing.

---

## 11. File map for follow-up analysis

| Path | Use |
| --- | --- |
| `Runs/<id>/backtest_summary.json` | OOS metrics + portfolio diagnostics |
| `Runs/<id>/config.yaml` | Frozen reward/env settings |
| `Runs/<id>/manifest.json` | Seeds, dates, training status, best-eval score |
| `Runs/<id>/plots/training.png` | Curriculum / eval / allocation diagnostics |
| `Runs/cohort_vs_benchmark.json` | EW/SPY comparisons for cohorts ≤621 |
| `Runs/oos_ledger.jsonl` | Multiple-testing / burn history |
| `docs/RESEARCH.md` | Published narrative + contamination warning |
| `config/config.yaml` | Current default recipe (`vol_penalty_scale: 0.15`, cap 0.20, exp 100 as of Jul 2026) |
| `config/cohorts/*.yaml` | Frozen cohort recipes |
| `.cache/runs_analysis.json` | Machine-readable extract used to build this doc (regenerate via `.cache/analyze_runs.py`) |

---

## 12. Bottom line for the improving agent

Historical search mostly swept **exposure penalty × seed** under a fixed env. That produced occasionally strong chained returns (best clean: **618 +205%**, **613 +186%**) but:

- **seed instability is first-order**,
- **W2/W4 risk-adjusted performance is weak**,
- **drawdowns in W3 remain large**,
- **eval selection does not predict OOS**,
- **policies concentrate and stay invested**,
- **deflated Sharpes never clear a high bar** given ~50 burns/window,
- the only major reward-term innovation attempted (`vol_penalty_scale=300`) **corrupted training** and must not be repeated without unit checks,
- relaxing the asset cap to 0.60 bought return with **worse Sharpe/DD**.

Highest-leverage env work is therefore: (1) healthier risk-off / cash optionality, (2) real diversification constraints, (3) eval–OOS alignment, (4) regime-aware risk penalties with safe magnitudes — validated multi-seed on W1–W5, with W4/W2 as gate metrics.
