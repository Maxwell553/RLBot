# 1360pctAlgo — prod_return_alpha_v1

**Locked return-alpha book** used by MarketTrainer’s `/ops/forward` paper test.

| Field | Value |
|---|---|
| Strategy id | `prod_return_alpha_v1` |
| Forward run id | `PROD_RETURN_ALPHA` |
| UI series label | `1360pctAlgo` |
| Library | `rlbot/prod_return_alpha.py` |
| Paper loop | `scripts/paper_prod_return_alpha.py` |

## Book

1. **57% sleeve A — TQQQ CC:** QQQ > SMA(151) and ATR% ≤ 20% → vol-target VT 0.27, **weekly** close-to-close. Equity cool −27.8% / 15 days → BIL.
2. **43% sleeve B — dual mom:** Month-end GLD vs TLT (231d), VT 14%, else BIL.

## Forward test (ops UI)

```bash
python scripts/paper_prod_return_alpha.py run-day --refresh-data
# or
bash scripts/daily_paper_prod_return_alpha.sh
```

Then open `/ops/forward`. Companion RL sleeve (optional):

```bash
python scripts/forward_mark.py --run-id LIVE_MODEL --refresh-data
```

## Pack CLI

```bash
python 1360pctAlgo/strategy.py --targets
python 1360pctAlgo/strategy.py --backtest --refresh-data
```

## Published research windows (retail + ADV impact, AUM $100k)

| Window | Return | Sharpe | Max DD | vs SPY |
|---|---|---|---|---|
| Train 2010–17 | +221% | **1.05** | −16.5% | **+44 pp** |
| Mid 2018–23 | +145% | **1.01** | −15.7% | **+50 pp** |
| Holdout 2024–26 | +85% | **1.43** | −12.4% | **+26 pp** |
| Full | **+1360%** | **1.09** | −16.5% | **+598 pp** |

Ann. one-way turnover ~5.2×. Do not retune knobs in `rlbot/prod_return_alpha.py` without a new nested search.
