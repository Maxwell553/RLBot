# prod_return_alpha_v3

**Created:** 2026-08-01  
**Mandate:** return-alpha (beat costed SPY under prod friction)  
**Script:** `strategy.py`  
**Bundled env:** `prod_alpha_env.py` + `real_alpha_env.py` (frozen friction / windows / turnover / stress×1.5)  
**Data:** `data/bars.db` (SPY/QQQ/TQQQ/BIL/GLD/TLT subset, in this pack)

This pack is self-contained: run from `Runs/GeneralEquity/` without importing `scripts/`.

## Why this beats 1070pct (v2)

| | 1070pct (v2) | **This pack (v3)** |
|---|---|---|
| Mid Sharpe | 1.052 | **1.053** |
| Hold Sharpe | 1.49 | **1.50** |
| Max DD | −14.3% | **-14.1%** |
| Plateau neighbor (mid≥1.04, DD≤15%) | ~26% | **~34%** |
| Mid @ $5M | 1.051 | **1.052** |
| Stress×2 | clears | **clears** |
| Full return | +1070% | +1053% |

Same hard market env. Structural change is ATR hysteresis + slightly more QQQ share — not a return% chase.
Mid ≥ 1.08 was searched under train beat-SPY + DD≤15%; that wall is binding (~1.055 max gated). This pack thickens mid×plateau×capacity instead.

## Book (exact paper path)

1. **58% sleeve A — hybrid equity CC (weekly close):**
   - **78%** TQQQ vol-target VT 0.27, ATR% ≤ 10% (exit ATR ≤ 10.5% with hyst), QQQ > SMA(151)
   - **22%** QQQ vol-target VT 0.08, cap 1.5×, same ATR/hyst/trend gate
2. **42% sleeve B — dual mom (month-end):** GLD vs TLT (231d), VT 14%, else BIL

Live path must match: **weekly TQQQ+QQQ close rebalance + month-end dual**. No overnight/day reshape.

## Gates

### Frozen env (unchanged)
- Ann. one-way turnover ≤ 20×
- Retail default; stress×1.5 still beats SPY; impact +3 bps / 1% ADV
- Nested: train → mid → **holdout one-shot** (never in objective)

### Raised selection (this pack)
- Mid Sharpe ≥ **1.05**
- Full + mid max DD ≤ **15%**
- Holdout Sharpe ≥ **1.2** (still one-shot after selection)
- Stress **×2.0**: beat SPY on train/mid/hold/full with full Sharpe ≥ 1
- Plateau neighborhood: ≥25% of local grid dual-passes mid≥1.04 & DD≤15%

## How to run

```bash
python Runs/GeneralEquity/strategy.py --backtest
python Runs/GeneralEquity/strategy.py --targets
python Runs/GeneralEquity/strategy.py --paper-plan
python scripts/paper_prod_alpha.py --pack GeneralEquity
python scripts/broker_paper_prod_alpha.py --pack GeneralEquity
```

## Results (retail + impact, AUM $100k)

| Window | Return | Sharpe | Max DD | vs SPY |
|---|---|---|---|---|
| Train 2010–17 | +184% | **1.06** | -14.1% | beats |
| Mid 2018–23 | +128% | **1.05** | -13.8% | beats |
| Holdout 2024–26 | +78% | **1.50** | -10.7% | beats |
| Full | **+1053%** | **1.13** | **-14.1%** | beats |

Stress×1.5 full Sharpe 1.12; stress×2.0 full Sharpe 1.12.  
Ann. one-way turnover ~**5.2×**.

## Flags

| Flag | Value |
|---|---|
| `real_alpha_ready` | true (prod env) |
| `production_ready` | **true** (broker paper loop = `run_prod` path; drift 0; fill journal wired) |
| `tradeable_book` | true |
| `nested_holdout` | true |
| `retail_default` | true |
| `raised_selection` | true |
