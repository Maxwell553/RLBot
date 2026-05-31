# Empirical Research Report: Multi-Asset Portfolio Management via Recurrent PPO

This report documents the methodology, training mechanics, and **strict out-of-sample (OOS)** performance of the deep reinforcement learning framework in this repository across three different evaluation windows. The objective is an autonomous multi-asset portfolio manager that allocates dynamically across global markets using a recurrent policy rather than fixed rules.

Each walk-forward window trains a separate **65M-step** RecurrentPPO run with a chronological holdout that never appears in training or in-training validation. OOS metrics below are measured only on that held-out block and compared to SPY buy-and-hold and an equal-weight (10% per asset, daily rebalanced) buy-and-hold on the same dates. SPY uses `benchmark_ohlcv_index()` (SP500 sleeve).

### Asset universe

The agent allocates across **cash + 10 tradeable underlyings** (softmax, long-only risky legs, **50%** per-asset cap by default with clip-and-redistribute). **DXY** and **TNX** are macro observation inputs only.

**Tradeable (action space, in order):**

| Label | Yahoo | Represents |
|-------|-------|------------|
| SP500 | SPY | S&P 500 |
| GOLD | GLD | Gold |
| OIL | USO | Crude oil (WTI) |
| EURUSD | EURUSD=X | EUR/USD FX |
| USDJPY | USDJPY=X | USD/JPY FX |
| NIKKEI | ^N225 | Nikkei 225 |
| FTSE | ^FTSE | FTSE 100 |
| BOND10Y | IEF | 10-year Treasury (ETF proxy) |
| COPPER | HG=F | Copper futures |
| EM | EEM | Emerging markets |

**Macro only (observation, not traded):**

| Label | Symbol | Role |
|-------|--------|------|
| DXY | DX-Y.NYB | US dollar index |
| TNX | ^TNX | 10-year Treasury yield |

---

## Executive Summary

### Walk-forward design

| Window | Train (in-sample) | OOS holdout (never in training) | Run ID |
|--------|-------------------|----------------------------------|--------|
| **1** | 2006–2020 | 2021-01-01 → 2022-12-30 | `65M_W1_6_01_26` |
| **2** | 2006–2022 | 2023-01-02 → 2024-12-31 | `65M_W2_6_01_26` |
| **3** | 2006–2024 | 2025-05-27 → 2026-05-26 (~365d tail) | `65M_W3_6_01_26` |

Training used **8 parallel envs**, **32768 rollout steps**, **batch size 16384**, **63-day episodes**, and **VecNormalize** on observations and rewards. Hyperparameters and per-asset costs are defined in **`config.yaml`** (snapshotted per run under `runs/<run_id>/`).

**Checkpoints:** `ppo_portfolio_final.zip` is the terminal 65M weights. **`best/best_model.zip`** is selected by **`EvalNavBestModelCallback`** on **mean eval ending NAV** (not episodic reward). Best checkpoints occassionally align with early validation NAV peaks, not the final 65M weights.

### OOS results vs SPY (buy-and-hold benchmark)

> **Archival metrics:** The table below records completed walk-forward cohorts that used earlier env versions. Removed artifacts: `65M_*_5_27_26`, `65M_W*_5_28_26`, `65M_W*_5_30_26` (**98-d**). Figures for **`65M_W*_6_01_26`** use **108-d** observations and a **65%** per-asset cap (superseded). **Current `config.yaml`:** **118-d** observations (dual-EMA trend features), **50%** cap, fee DR **`[0.75, 1.3]`** — train under new run IDs (e.g. `65M_W*_5_31_26` / `6_02_26`); do not resume old checkpoints.

| Window | Regime (brief) | Checkpoint | OOS return | Ann. Sharpe | SPY return | Excess vs SPY |
|--------|----------------|------------|------------|-------------|------------|---------------|
| **1** | Fed tightening; flat/risk-off equities | Final (65M) | **+17.2%** | 0.55 | +0.5% | **+16.7 pp** |
| **1** | | Best (~13M eval) | **+19.7%** | 0.74 | +0.5% | **+19.2 pp** |
| **2** | Low-vol equity bull | Final (65M) | +14.2% | 0.64 | +43.2% | −29.0 pp |
| **2** | | Best (~12.5M eval) | +13.2% | 0.60 | +43.2% | −29.9 pp |
| **3** | High rotation / mixed macro | Final (65M) | +19.9% | 2.47 | +21.9% | −2.0 pp |
| **3** | | Best (~9M eval) | **+37.7%** | **3.71** | +21.9% | **+15.9 pp** |

### Headline findings

1. **Window 1 (2021–2022):** The agent materially beat SPY in a difficult macro period, with the best eval checkpoint **outperforming** the final 65M model. Allocation (gold, bonds, FX, EM) dominated over passive US equity exposure.
2. **Window 2 (2023–2024):** Both final and best checkpoints **underperformed SPY** in a strong bull market while still posting positive absolute returns.
3. **Window 3 (2025–2026):** The **best checkpoint (~9M steps)** was the standout result: **+37.7% OOS**, Sharpe **3.71**, and **+15.9 pp** over SPY with a **−4.6%** max drawdown. The final 65M model captured most of the trend but lagged the early checkpoint. The best checkpoint at ~9M steps coincides approximately with the end of the fee curriculum ramp in that run — whether early OOS performance reflects genuine policy learning or partially curriculum-phase behavior remains an open question.

---

### Structural anomalies

#### 1. Validation NAV cliff (~9M–15M steps)

All three windows show the same pattern: **training reward and training episode NAV improve through 65M steps**, while **validation ending NAV peaks early** then stagnates or falls. Continuing training past the peak **hurts OOS performance** (W1, W3: final < best; W3 gap is extreme).

**Interpretation:** Policy overfits idiosyncrasies of the alternating in-sample training blocks. Saving the **best-by-eval-NAV** checkpoint is appropriate; the **final** checkpoint is often strictly worse OOS.

**Mitigation in current stack:** `EvalNavBestModelCallback` + `eval_nav_history.npz` for analysis; future work: hard validation-NAV early stopping.

#### 2. Bull-market beta underexposure (Window 2)

Window 2’s underperformance is **not** primarily from deeper drawdowns — both model and SPY experienced **~−10% to −11%** troughs. The issue is **insufficient equity beta** during a +43% SPY rally: heavy cash and defensive sleeves from a reward mix that penalizes drawdown and idle capital.

**Mitigation in current stack:** Stronger decoupled **churn penalty** (`churn_lambda` in config, not multiplied by `REWARD_SCALE`); continued reward tuning (Sortino differential vs market, inactivity terms) is ongoing.

---

## OOS performance by window

### Window 1 — OOS 2021–2022 (bear / tightening)

**Macro context:** Multi-asset drawdown and rate-driven repricing; SPY was effectively flat over the holdout (+0.5%).

**Results:** Final model **+17.2%** (Sharpe 0.55); best eval checkpoint **+19.7%** (Sharpe 0.74). Both strongly beat SPY.

#### Training dynamics

<p align="center">
  <img src="plots/65M_W1_6_01_26/training.png" alt="Window 1 training log" width="550">
</p>

*Three-panel training log for `65M_W1_6_01_26`. **Top:** per-step training reward rises steadily to 65M steps. **Middle:** validation ending NAV peaks near **12–13M steps** ($102k), then oscillates near $100k–$101k. **Bottom:** training episode-end NAV mean climbs to ~$130k, diverging from validation.*

#### OOS vs SPY and backtest dashboard (best checkpoint)

<p align="center">
  <img src="plots/65M_W1_6_01_26/backtest_best.png" alt="Window 1 best checkpoint backtest" width="550">
</p>

*OOS equity and drawdown vs <strong>SPY</strong> and <strong>equal-weight 10-asset</strong> buy-and-hold, plus <strong>target portfolio weights</strong> for the best checkpoint (~13M validation peak). Archival table: model <strong>+19.7%</strong> vs SPY <strong>+0.5%</strong> (<strong>+19.2 pp</strong> excess).*

---

### Window 2 — OOS 2023–2024 (bull market)

**Macro context:** Strong US equity rally; SPY **+43.2%** over the holdout.

**Results:** Final **+14.2%** (Sharpe 0.64); best **+13.2%** (Sharpe 0.60). Positive absolute returns but large **beta underexposure** vs SPY (~**−29 pp**).

**Note:** Unlike windows 1 and 3, the saved **best** checkpoint here aligns with a **late** validation episode (~**12.5M** steps), not an early peak — both checkpoints behave similarly on OOS.

#### Training dynamics

<p align="center">
  <img src="plots/65M_W2_6_01_26/training.png" alt="Window 2 training log" width="550">
</p>

*Validation NAV drops to the **$100k floor near ~27M steps**, then slightly recovers into a broad plateau. Training rewards and training episode NAV still trend up through 65M steps.*

#### OOS vs SPY and backtest dashboard (best checkpoint)

<p align="center">
  <img src="plots/65M_W2_6_01_26/backtest_best.png" alt="Window 2 best checkpoint backtest" width="550">
</p>

*Best checkpoint <strong>+13.2%</strong> vs SPY <strong>+43.2%</strong>. Underperformance is driven by <strong>missed upside</strong> (persistent cash / non-equity sleeves), not uniformly shallower drawdowns.*

---

### Window 3 — OOS 2025–2026 (~365-day tail)

**Macro context:** Shorter holdout (**260 daily bars**); higher realized vol and sector rotation. SPY **+21.9%**.

**Results:** Final **+19.9%** (Sharpe 2.47); best **+37.7%** (Sharpe **3.71**, max DD **−4.6%**). Best beats SPY by **+15.9 pp** — the strongest risk-adjusted OOS result in the study.

#### Training dynamics

<p align="center">
  <img src="plots/65M_W3_6_01_26/training.png" alt="Window 3 training log" width="550">
</p>

*Validation NAV peaks near **~9M steps**, collapses toward baseline by ~12M, then slowly recovers after ~40M. Training metrics continue improving throughout — separating **training fit** from **validation generalization**.*

#### OOS vs SPY and backtest dashboard (best checkpoint)

<p align="center">
  <img src="plots/65M_W3_6_01_26/backtest_best.png" alt="Window 3 best checkpoint backtest" width="550">
</p>

*Best (~9M) <strong>+37.7%</strong> vs SPY <strong>+21.9%</strong> (<strong>+15.9 pp</strong>). Model drawdown near <strong>−5%</strong> vs SPY <strong>~−9%</strong> in the April 2026 dip.*

---

## 1. Data engineering & anti-leakage architecture

Ensuring the environment was leakage-free was critical for credible OOS results.

### Fractional differentiation (d = 0.4)

Raw prices are non-stationary; integer differencing (d = 1) is stationary but destroys long memory. Fractional differentiation applies:

$$\Delta^d x_t = \sum_{k=0}^{\infty} w_k x_{t-k}, \quad w_0 = 1, \quad w_k = w_{k-1}\frac{k - 1 - d}{k}$$

With **d = 0.4**, log-price features retain regime memory while remaining usable for policy gradients. Implementation: `fracdiff_weights`, vectorized `fracdiff_series_1d`, `compute_fracdiff_panel` in `data_utils.py`.

### Per-block feature isolation (current)

`train_test_split_alternating()` receives **only** aligned `ohlcv` and `macro`. For each contiguous train or eval segment it calls `compute_feature_panel()` on that slice alone. RSI/MACD/fracdiff therefore **never** use prices from a different regime block or from the OOS holdout.

At segment joins, the first **`feature_purge_warmup`** bars (default **25**) are neutralized so stitched panels do not inherit spurious indicator state.

### Causal execution pipeline

1. **Feature lag (`obs_lag`):** At bar *t*, market features use data through `close[t − obs_lag]`. OOS backtests fix **`obs_lag = 1`**.
2. **Next-open fills:** Weights from the overnight decision execute at **`open[t+1]`**.
3. **Holding costs & MTM:** Applied and marked at **`close[t+1]`** after rebalance.

No same-bar close execution after observing that close.

### Walk-forward splits & holdout

- **Alternating blocks:** Trainable timeline split into **126-bar** blocks; every **4th** block is in-training eval (both train and eval span full pre-holdout history).
- **Chronological holdout:** Trailing calendar segment (explicit dates or `--holdout-days`) is **removed before** any split and used **only** for `backtest.py`.

---

## 2. Environment dynamics & state matrix

Custom Gymnasium env: **10 global tradeable assets + cash**, long-only, softmax over **11** logits. Macro **DXY** and **TNX** appear in observations only. Parameters are loaded from **`config.yaml`** via `get_config()` in `trading_env.py`.

### Action mapping

Policy outputs **a ∈ ℝ¹¹** → softmax → weights. Per-asset risky weights are capped at **`max_single_asset_weight`** (default **50%**); overflow is **clip-and-redistributed** across other active risky assets (not dumped to cash). **Σ wᵢ = 1**.

### Observation (118 dimensions with regime macros)

| Block | Dims | Content |
|-------|-----:|---------|
| Fracdiff increments | 40 | Horizons 1, 5, 10, 20 days × 10 assets |
| Market mean fracdiff | 4 | Cross-sectional mean per horizon |
| Realized volatility | 11 | 20-day vol per asset + market mean |
| RSI + MACD | 20 | 10 + 10, scaled |
| EMA trend distance | 10 | (EMA20 − EMA100) / EMA100 per asset, clipped |
| Macro (DXY, TNX, VIX, HY OAS) | 20 | Fracdiff horizons + vol |
| Portfolio + meta | 13 | 11 weights + drawdown + episode progress |

### Reward structure

The step reward combines (scalars in `config.yaml` → `reward` section):

- **Scaled log return** (primary PnL term, clipped per step)
- **Sortino differential** vs **cap-weighted** passive benchmark (`benchmark_cap_weights` in `config.yaml`) over a **21-day** window
- **Participation bonus** on gross risky exposure
- **Inactivity penalties** when cash > 50% and > 90%
- **Churn penalty:** `churn_scale × churn_lambda × |Δw|` (λ **not** multiplied by `REWARD_SCALE`)
- **Quadratic drawdown penalty** from episode peak NAV: `(dd_frac²) × (drawdown_penalty_scale × 10)`

**Transaction costs:** Per-asset slippage, fees, and annual holding-cost vectors in `config.yaml` (`transaction_costs`).

---

## 3. Training loop mechanics

### RecurrentPPO architecture

- **MlpLstmPolicy:** 2×64 LSTM → actor/critic MLP heads **[128, 128]**
- **VecNormalize** on observations and training rewards (frozen at inference via `vecnorm_utils.freeze_vec_normalize_for_inference`)
- **8× SubprocVecEnv**, **63-step** episodes (~3 months of daily bars)
- **AdamW**, cosine LR decay to **1e−6** floor
- **PyTorch determinism** flags set in `train.py` for reproducibility

### Curriculum & callbacks (65M budget)

Milestones from `trade_curriculum_milestones(learn_budget)` in `train.py` / `config.yaml` `curriculum` section:

| Phase | Approx. step (65M) | Behavior |
|-------|-------------------|----------|
| Frictionless | 0 – **5.2M** (8%) | `fee_scale = 0` |
| Fee ramp | **5.2M – 22.75M** (35%) | Linear ramp to full fees |
| Progressive DR | **22.75M – 55.25M** | Widen `fee_scale` / `obs_lag` bounds |
| Full DR | **≥ 55.25M** | `fee_scale ∈ [0.75, 1.3]`, `obs_lag ∈ {0,1,2}` |
| Churn on | from **~9.75M** (15%) | `churn_scale` 0 → 1 |

1. **`TradingCurriculumCallback`:** Applies the schedule above on training envs only (eval envs use fixed fees/lag).
2. **`EvalNavBestModelCallback`:** Periodic eval; persists `best_model.zip` on **max mean ending NAV**; logs `eval_nav_history.npz` and `evaluations.npz`.
3. **`AdaptiveEntropyCallback`:** High entropy during exploration; **mandatory** cosine decay from `explore_ent` to `final_ent` starting at **`decay_start_fraction`** of the run (default **45%**), independent of eval NAV. Early floors still apply before decay during fee curriculum.

---

## 4. Training walk-forward windows (current architecture)

Rebuild `data_cache.npz` once after any change to `data_utils.py` or macro/benchmark logic (**118-d** observations: VIX + HY OAS macros, dual-EMA trend features, cap-weight Sortino). Legacy **98-d** / **108-d** checkpoints are incompatible with the current env; do not resume them.

From the repo root with `.venv` activated:

```bash
cd /Users/maxingargiola/Desktop/Lockin/ModelsProj/RLBot

# 1) Refresh aligned panel + features (run once; ~few minutes)
.venv/bin/python train.py --refresh-data --timesteps 1000 --run-id _data_refresh --no-viz

# 2) Window 1 — train 2006–2020, OOS 2021–2022
.venv/bin/python train.py \
  --since 2006-01-01 \
  --until 2022-12-31 \
  --train-end 2020-12-31 \
  --holdout-start 2021-01-01 \
  --holdout-end 2022-12-31 \
  --timesteps 65000000 \
  --run-id 65M_W1_6_01_26

# 3) Window 2 — train through 2022, OOS 2023–2024 (omit --refresh-data)
.venv/bin/python train.py \
  --since 2006-01-01 \
  --until 2024-12-31 \
  --train-end 2022-12-31 \
  --holdout-start 2023-01-01 \
  --holdout-end 2024-12-31 \
  --timesteps 65000000 \
  --run-id 65M_W2_6_01_26

# 4) Window 3 — train through 2024, OOS = last 365 calendar days
.venv/bin/python train.py \
  --since 2006-01-01 \
  --holdout-days 365 \
  --timesteps 65000000 \
  --run-id 65M_W3_6_01_26
```

Optional: `./windows/window{1,2,3}_train.sh` if kept in sync; validate splits with  
`.venv/bin/python windows/validate_split.py --window 1`.

After each run, OOS backtest (example W1):

```bash
.venv/bin/python backtest.py --run-id 65M_W1_6_01_26 --detailed
.venv/bin/python backtest.py --run-id 65M_W1_6_01_26 \
  --model models/65M_W1_6_01_26/best/best_model.zip --detailed
```

### Current stack (`65M_W*_5_31_26` and later)

`config.yaml` sets **`max_single_asset_weight: 0.50`** (50% per risky asset), **118-d** observations, quadratic drawdown penalty, and fee DR **`[0.75, 1.3]`**. Use new run IDs after any architecture change (do not resume `6_01_26` or older checkpoints — those used a **65%** cap and smaller observation vectors).

```bash
# Refresh only if data cache is stale; required on first run after macro changes
.venv/bin/python train.py --refresh-data --timesteps 1000 --run-id _data_refresh --no-viz

.venv/bin/python train.py \
  --since 2006-01-01 --until 2022-12-31 \
  --train-end 2020-12-31 --holdout-start 2021-01-01 --holdout-end 2022-12-31 \
  --timesteps 65000000 --run-id 65M_W1_5_31_26

.venv/bin/python train.py \
  --since 2006-01-01 --until 2024-12-31 \
  --train-end 2022-12-31 --holdout-start 2023-01-01 --holdout-end 2024-12-31 \
  --timesteps 65000000 --run-id 65M_W2_5_31_26

.venv/bin/python train.py \
  --since 2006-01-01 --holdout-days 365 \
  --timesteps 65000000 --run-id 65M_W3_5_31_26
```

---

## Assumptions & limitations

This section states explicit boundaries for interpreting results. OOS metrics above are useful for comparing checkpoints and windows **within** this framework, **not** as guarantees of live performance.

### Data vendor and corporate actions

Daily OHLCV and macro series are pulled from **yfinance** (and a limited FRED graph export for recent HY OAS). Yahoo’s adjusted history does **not** provide institutional-grade, **point-in-time** corporate-action correction. Splits and dividends are handled via Yahoo’s end-of-day adjustment rules, which can differ from CRSP/Compustat-style backtests. That introduces **small historical tracking error** versus a fully corrected tape—usually minor for liquid ETF/FX proxies at daily frequency, but non-zero for precise attribution and for any single-name event study.

### Static universe and survivorship

The policy trades a **fixed list** of ten liquid proxies plus cash (see asset tables). The universe is a **static macro allocation menu**, not a dynamic screener. We do **not** model:

- ETFs that did not exist in 2006 (the panel starts only when all ten assets have real quotes; there is no backfilled “what if” listing),
- Delisted or merged products removed from today’s menu,
- **Survivorship selection bias** from choosing instruments that remained tradable and data-available through the backtest end date.

Reported OOS performance is therefore conditional on “these ten sleeves, as implemented today, over the aligned history,” not on an unbiased draw of all funds that could have been held historically.

### Macro and benchmark modeling choices

- **HY OAS:** FRED `BAMLH0A0HYM2` is available only over a recent window via the public graph CSV; earlier history uses an **HYG/IEF proxy** calibrated to FRED on overlap. Regime features are informative but not identical to exchange-traded OAS levels for the full sample.
- **Sortino differential:** The passive benchmark in the **reward** is **cap-weighted** across the ten tradeable assets (`benchmark_cap_weights` in `config.yaml`), not SPY or equal-weight. **Backtest plots** additionally show **SPY** and **equal-weight (10% each, daily rebalanced)** buy-and-hold for interpretability.
- **In-training validation:** Alternating walk-forward blocks can peak early while training loss improves; **best-by-eval-NAV** and **final** checkpoints often diverge on true holdout.

### Structural stop-loss (wide barrier)

The environment terminates an episode if NAV falls below **`stop_loss_fraction`** of episode-start capital (default **0.45**, i.e. a **55%** drawdown from the episode open). Over a **63-day** (~3-month) window on a diversified basket of cash, rates, gold, and global equity proxies, that threshold is rarely (if ever) hit outside a systemic crash.

That is intentional: risk control is meant to come from rolling Sortino differentials, drawdown penalties, and inactivity/churn terms, not from frequent artificial episode truncations that distort the value function. The stop-loss flag is a safety rail, not the primary training signal.

### Execution and live deployment

Training assumes next-open fills, configurable lags, and per-asset cost vectors in config. VecNormalize statistics and observation scaling must match the training run; mixing checkpoints from different `run_id`s or observation dimensions is invalid. At inference, VecNormalize is frozen via `training=False` (see `vecnorm_utils.freeze_vec_normalize_for_inference`).

### What this report does not claim

We do not claim statistical significance of OOS Sharpe ratios, absence of multiple-testing bias across windows, or robustness to every macro regime without retraining. Walk-forward holdouts **reduce** calendar leakage but do **not** remove all researcher degrees of freedom (hyperparameters, checkpoint choice, window design). Treat headline OOS numbers as **evidence within a documented pipeline**, subject to the limits above.
