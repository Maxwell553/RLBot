#!/usr/bin/env python3
"""
PulseDay — TrueDay-honest book with higher returns + improved day mix.

Honesty (same bar as TrueDay / ForgeDay audit fixes):
  - FULL_HIST universe only
  - Pyramid adds: next-open only (no same-bar open-after-close)
  - 7 bps + funding + gap-through adverse stops; stress ×1.5
  - train+mid select / one-shot hold; anti-fantasy caps

Mechanics beyond TrueDay:
  - Honest pyramid into winners (scale 0.75, max 2)
  - Scale-out partial TP → BE runner (locks more green days)
  - stop_after_loss: no new entries after first fully-closed red of day
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np

from forgeday_engine import (
    COST,
    DEFAULT_AUM,
    FUNDING_PER_8H,
    HARD,
    P,
    VENUE,
    dd25,
    run as _run_raw,
)
from trueday_engine import day_stats_from_eq as day_stats_from_eq  # re-export
from trueday_engine import latest_intents as _td_intents

PACK = Path(__file__).resolve().parent
LOCK_PATH = PACK / "pulseday_locked.json"

# Fallback lock (overwritten when pulseday_locked.json exists in this pack)
P_LOCK = P(
    top_k=25,
    adv_lb_days=21,
    rebalance_days=7,
    min_hist_bars=720,
    compress_lb=48,
    near_lb=18,
    near_buf=0.04,
    atr_pct_max=0.25,
    rs_lb=18,
    rs_min=0.03,
    vol_lb=24,
    vol_min=0.85,
    break_confirm=True,
    stop_atr=1.5,
    target_atr=5.0,
    trail_atr=0.0,
    atr_lb=14,
    risk_frac=0.007,
    lev_cap=4.0,
    max_names=2,
    max_trades_day=8,
    cool_bars=3,
    day_loss=0.025,
    day_bank=0.05,
    entry_start=1,
    entry_end=20,
    size_dd=0.10,
    hard=HARD,
    roll_hwm=60,
    side_mode="both",
    btc_gate=False,
    stop_adverse_bps=5.0,
    funding_per_8h=FUNDING_PER_8H,
    require_full_hist=True,
    family="coil",
    sniper_on=True,
    sniper_top_k=15,
    sniper_vol_min=1.8,
    sniper_atr_pct_max=0.2,
    sniper_near_buf=0.02,
    sniper_stop_atr=1.5,
    sniper_target_atr=8.0,
    sniper_risk_frac=0.015,
    sniper_cool=12,
    short_btc_max=-0.025,
    short_rs_min=0.05,
    short_vol_min=2.0,
    short_risk_frac=0.008,
    short_stop_atr=1.5,
    short_target_atr=5.0,
    pyramid_on=True,
    pyramid_r=1.0,
    pyramid_scale=0.75,
    pyramid_max=2,
    one_green=0.0,
    flatten_green=False,
    be_r=0.0,
    time_stop_bars=0,
    protect_scale=1.0,
    scale_out_r=0.7,
    scale_out_frac=0.65,
    stop_after_loss=True,
)


def _load_lock() -> P:
    if not LOCK_PATH.exists():
        return P_LOCK
    raw = json.loads(LOCK_PATH.read_text())
    params = raw.get("params", raw)
    # Keep only known P fields
    allowed = {f.name for f in fields(P)}
    clean = {k: v for k, v in params.items() if k in allowed}
    return replace(P_LOCK, **clean)


P_LOCK = _load_lock()


def asdict_p(p: P = P_LOCK) -> dict:
    return asdict(p)


def run(p: P = P_LOCK, aum: float = DEFAULT_AUM, cost_mult: float = 1.0, fund_mult: float = 1.0):
    p = replace(
        p,
        require_full_hist=True,
        hard=HARD,
        lev_cap=min(p.lev_cap, 4.0),
        # honesty: never allow same-bar pyramid optimism (engine is next-open)
        pyramid_on=bool(p.pyramid_on),
        pyramid_scale=float(p.pyramid_scale if p.pyramid_on else 0.0),
    )
    return _run_raw(p, aum=aum, cost_mult=cost_mult, fund_mult=fund_mult)


def passes_selection(nest: dict) -> bool:
    if not dd25(nest, keys=("train", "mid")):
        return False
    if nest["train"]["total_return"] <= 0 or nest["mid"]["total_return"] <= 0:
        return False
    if nest["mid"]["total_return"] <= nest["btc_mid"]["total_return"]:
        return False
    if nest["train"]["sharpe_ratio"] < 0.35 or nest["mid"]["sharpe_ratio"] < 0.7:
        return False
    if nest["full"]["trades_per_year"] < 100:
        return False
    avg_sel = 0.5 * (nest["train"]["avg_daily_return"] + nest["mid"]["avg_daily_return"])
    if avg_sel > 0.008:
        return False
    if nest["train"]["total_return"] > 5.0 or nest["mid"]["total_return"] > 5.0:
        return False
    return True


def passes_holdout(nest: dict) -> bool:
    if not dd25(nest, keys=("hold", "full")):
        return False
    if nest["hold"]["total_return"] <= 0:
        return False
    if nest["hold"]["total_return"] <= nest["btc_hold"]["total_return"]:
        return False
    if nest["hold"]["sharpe_ratio"] < 0.7:
        return False
    if nest["hold"]["total_return"] > 1.2:
        return False
    return True


def latest_intents(p: P = P_LOCK, aum: float = DEFAULT_AUM, equity: float | None = None) -> dict:
    """Reuse TrueDay causal intent builder with PulseDay params."""
    return _td_intents(p, aum=aum, equity=equity)


def set_lock(p: P) -> None:
    global P_LOCK
    P_LOCK = p


def day_mix(eq: np.ndarray) -> dict:
    d = day_stats_from_eq(eq)
    r = np.diff(eq) / eq[:-1]
    r = r[np.isfinite(r)]
    return {
        **d,
        "day_lose_rate": float((r < -1e-12).mean()) if len(r) else 0.0,
    }
