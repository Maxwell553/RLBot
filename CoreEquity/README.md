# CoreEquity

**Created:** 2026-08-20  
**Mandate:** no 2×/3× ETFs; QQQ may cash-finance up to 1.5×; beat the **same** dividend-adjusted SPY as GeneralEquity under retail friction  
**Script:** `strategy.py`  
**Lock:** `coreequity_locked.json`  
**Bundled env:** `core_equity_env.py` + `real_alpha_env.py`  
**Data:** `data/bars.db` (SPY/QQQ/BIL/GLD/TLT copied from GeneralEquity; remaining 1× names dividend-adjusted)

This pack is self-contained: run from `Runs/CoreEquity/` without importing `scripts/`.

## Status: production-ready

Raised GeneralEquity gates **cleared** on train+mid search with holdout one-shot. Paper `run_prod` drift is **0** with a fill journal. `prod_viable_research` and `production_ready` are **true**.

This is **not** a claim that the book is better than both GeneralEquity and the prior CoreEquity analog. Full Sharpe (1.19) and max DD (−14.3%) do not beat the old analog’s 1.21 / −11.7%. CAGR excess vs the same SPY (**+2.63 pp**) does beat GeneralEquity’s ~+2.0 pp.

## Book (exact paper path)

1. **58% sleeve A — QQQ-only CC (weekly close):** vol-target VT 0.30, ATR% ≤ 12% (exit ATR ≤ 12.6% with hyst), long if **QQQ 12-month absolute momentum OR QQQ > SMA(42)**, cash-financed cap **1.4×**
2. **42% sleeve B — dual mom (month-end):** GLD vs TLT (231d), VT 14%, else BIL

No 2×/3× ETFs. Hot/dual weights cap at 1.0. Residual is BIL.

Trend is QQQ-on-QQQ. Not SMA(151) as the primary gate (that book cannot beat 2010–17 total-return SPY). Not SMH-on-SMH. The short SMA is a **re-entry** overlay so 2023 (still in the mid window) is not left on the table after a 12m lag.

Live path: **weekly QQQ close rebalance + month-end dual**. Same engine as `run_prod`.

## Honest SPY

`data/bars.db` shares GeneralEquity’s dividend-adjusted SPY (2010-01-04 **$84.58**, full **+762.5%**). The previous unadjusted series ($113.33 start, same $729 end) made beat-SPY too easy and is gone.

## Gates (same bar as GeneralEquity; not lowered)

| Gate | Result |
|---|---|
| Frozen select (train+mid Sharpe ≥ 1.0, beat SPY, TO ≤ 20×) | **pass** |
| Raised select (mid Sharpe ≥ **1.05**, full+mid DD ≤ **15%**) | **pass** (mid **1.106**, DD **−14.3%**) |
| Raised holdout one-shot (hold Sharpe ≥ 1.2, beat SPY) | **pass** (1.53) |
| Stress ×1.5 and ×2.0 beat SPY, full Sharpe ≥ 1 | **pass** (1.19 / 1.18) |
| Plateau ≥25% of local grid (mid≥1.04 and DD≤15%) | **pass** (**62%**) |
| Capacity mid Sharpe ≥ 1.05 at $5M | **pass** (**1.105**) |

## How to run

```bash
python Runs/CoreEquity/strategy.py --backtest
python Runs/CoreEquity/strategy.py --targets
python Runs/CoreEquity/strategy.py --paper-plan
python scripts/paper_prod_alpha.py --pack CoreEquity
python scripts/broker_paper_prod_alpha.py --pack CoreEquity
```

Paper scripts take the pack’s `latest_targets` / `paper_plan` / `run_prod` (QQQ + dual). Drift vs research must stay ~0 for `production_ready`.

## Results (retail + impact, AUM $100k, shared SPY)

| Window | Return | Sharpe | Max DD | vs SPY |
|---|---|---|---|---|
| Train 2010–17 | +182% | **1.16** | -14.3% | beats |
| Mid 2018–23 | +146% | **1.11** | -11.8% | beats |
| Holdout 2024–26 | +82% | **1.53** | -13.1% | beats |
| Full | **+1159%** | **1.19** | **-14.3%** | beats |

Stress×1.5 full Sharpe 1.19; stress×2.0 full Sharpe 1.18.  
Ann. one-way turnover ~**4.1×**.  
Full CAGR excess vs this SPY: **+2.63 pp** (GeneralEquity ~**+2.0 pp**; prior CoreEquity analog +1.63 pp).

### vs GeneralEquity (same SPY, same cost model)

| | GeneralEquity (TQQQ) | **CoreEquity (this lock)** |
|---|---|---|
| Full return | +1053% | **+1159%** |
| Full Sharpe | 1.13 | **1.19** |
| Mid Sharpe | 1.05 | **1.11** |
| Hold Sharpe | 1.50 | **1.53** |
| Max DD | **-14.1%** | -14.3% |
| CAGR excess vs SPY | ~+2.0 pp | **+2.63 pp** |
| Raised mid 1.05 | **pass** | **pass** |
| `prod_viable_research` | **true** | **true** |

Max DD does **not** beat GeneralEquity (−14.3% vs −14.1%) or the prior CoreEquity analog (−11.7%). Do not retune on holdout to manufacture a “better than both” claim.

## Flags

| Flag | Value |
|---|---|
| `real_alpha_ready` | true (prod env) |
| `tradeable_book` | true (cash 1× ETFs + 1.4× QQQ) |
| `nested_holdout` | true |
| `retail_default` | true |
| `no_3x_etfs` | **true** |
| `raised_selection` | **true** |
| `prod_viable_research` | **true** |
| `production_ready` | **true** (broker paper loop = `run_prod`; drift 0; fill journal wired) |
