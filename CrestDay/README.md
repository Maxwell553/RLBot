# CrestDay

**Created:** 2026-08-03  
**Mandate:** Honest crypto day trader ($500–$1k), Gate USDT-M perps, lev ≤ 4×, DD ≤ 25%  
**Script:** `strategy.py`  
**Bundled engines:** `crestday_engine.py` → `pulseday_engine.py` / `forgeday_engine.py` / `trueday_engine.py` / `soliday_engine.py`  
**Lock:** `crestday_locked.json` (in this pack)  
**Data:** `data/crypto_intraday/breakout/` (in this pack)  
**Venue:** `gate_usdt_perp` — INTX banned  
**Pick:** `crest_polish_332`

This pack is self-contained: run from `Runs/CrestDay/` without importing `scripts/`.

## Why this exists (vs PulseDay)

PulseDay is strong, but on the restored ~2y FULL_HIST panel its win−lose gap sits under +5pp. CrestDay keeps the same honesty stack and locks a successor where:

1. **tpd ≥ 1**
2. **calendar wins ≥ losses + 5pp**
3. **full return ≥ PulseDay** on the same panel

| Honesty knob | Setting |
|---|---|
| Universe | FULL_HIST only (~17.5k 1h bars, 2024-08 → 2026-08) |
| Pyramid | ON, **next open only** |
| Scale-out | partial TP → BE runner |
| Day protect | `flatten_green` + `one_green` + `stop_after_loss` |
| Costs | 7 bps + funding; stress ×1.5 |
| Nested | 50/25/25; hold one-shot |
| Same-bar pyramid | banned |

## Locked results (AUM $1000)

| Window | Return | Notes |
|---|---|---|
| Train | **+64.5%** | select |
| Mid | **+37.3%** | select; beats BTC mid |
| Hold | **+40.6%** | one-shot |
| Full | **+215.9%** | DD **−17.1%** |
| Stress ×1.5 | **+101.3%** | DD −19.5% |

Day mix (calendar): **win 28.8% / flat 47.7% / lose 23.5%** — gap **+5.3pp**.  
Active WR ≈ **55.1%**. Trades/day ≈ **1.21**.

Mechanics: coil + sniper, next-open pyramid (0.75×), scale-out, `flatten_green`, `one_green=1.2%`, `protect_scale=0.25`, `max_names=2`, `stop_after_loss`.

## vs PulseDay (same panel / honesty)

| | PulseDay | CrestDay |
|---|---|---|
| Full | +158.4% | **+215.9%** |
| Stress ×1.5 | ~+73% | **+101%** |
| Calendar w/f/l | 27.6 / 49.2 / 23.2 | **28.8 / 47.7 / 23.5** |
| Win−lose | +4.4pp | **+5.3pp** |
| tpd | 1.27 | **1.21** |

## Usage

```bash
python Runs/CrestDay/strategy.py --backtest --aum 1000
python Runs/CrestDay/strategy.py --targets
python Runs/CrestDay/strategy.py --live-intents --aum 1000
python scripts/search_crestday.py 600
```
