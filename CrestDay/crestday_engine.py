#!/usr/bin/env python3
"""
CrestDay — PulseDay successor under TrueDay honesty.

Goals vs PulseDay (same panel):
  - tpd >= 1
  - calendar wins >= losses + 5pp
  - full return ~even or better

Honesty (unchanged):
  - FULL_HIST universe only
  - Pyramid adds: next-open only
  - 7 bps + funding + gap-through adverse stops; stress ×1.5
  - train+mid select / one-shot hold; anti-fantasy caps
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
from pulseday_engine import passes_holdout, passes_selection  # same gates
from trueday_engine import day_stats_from_eq as day_stats_from_eq
from trueday_engine import latest_intents as _td_intents

PACK = Path(__file__).resolve().parent
LOCK_PATH = PACK / "crestday_locked.json"

# Fallback if lock missing (crest_polish_332)
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
    rs_min=0.014,
    vol_lb=24,
    vol_min=1.1,
    break_confirm=True,
    impulse_lb=3,
    impulse_min=0.012,
    impulse_vol=1.3,
    stop_atr=1.2,
    target_atr=5.5,
    trail_atr=0.0,
    atr_lb=14,
    risk_frac=0.004,
    lev_cap=4.0,
    max_names=2,
    max_trades_day=24,
    cool_bars=5,
    day_loss=0.015,
    day_bank=0.05,
    entry_start=1,
    entry_end=21,
    size_dd=0.10,
    hard=HARD,
    roll_hwm=60,
    side_mode="long",
    btc_gate=False,
    stop_adverse_bps=5.0,
    funding_per_8h=FUNDING_PER_8H,
    require_full_hist=True,
    family="coil",
    sniper_on=True,
    sniper_top_k=15,
    sniper_vol_min=1.3,
    sniper_atr_pct_max=0.2,
    sniper_near_buf=0.02,
    sniper_stop_atr=1.5,
    sniper_target_atr=7.5,
    sniper_risk_frac=0.02,
    sniper_cool=3,
    short_btc_max=-0.025,
    short_rs_min=0.05,
    short_vol_min=2.0,
    short_risk_frac=0.008,
    short_stop_atr=1.5,
    short_target_atr=5.0,
    pyramid_on=True,
    pyramid_r=1.0,
    pyramid_scale=0.75,
    pyramid_max=1,
    one_green=0.012,
    flatten_green=True,
    be_r=0.6,
    time_stop_bars=0,
    protect_scale=0.25,
    scale_out_r=0.45,
    scale_out_frac=0.7,
    stop_after_loss=True,
)


def _load_lock() -> P:
    if not LOCK_PATH.exists():
        return P_LOCK
    raw = json.loads(LOCK_PATH.read_text())
    params = raw.get("params", raw)
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
        pyramid_on=bool(p.pyramid_on),
        pyramid_scale=float(p.pyramid_scale if p.pyramid_on else 0.0),
    )
    return _run_raw(p, aum=aum, cost_mult=cost_mult, fund_mult=fund_mult)


def latest_intents(p: P = P_LOCK, aum: float = DEFAULT_AUM, equity: float | None = None) -> dict:
    return _td_intents(p, aum=aum, equity=equity)


def day_mix(eq: np.ndarray) -> dict:
    d = day_stats_from_eq(eq)
    r = np.diff(eq) / eq[:-1]
    r = r[np.isfinite(r)]
    return {
        **d,
        "day_lose_rate": float((r < -1e-12).mean()) if len(r) else 0.0,
    }


__all__ = [
    "COST",
    "DEFAULT_AUM",
    "HARD",
    "P",
    "P_LOCK",
    "VENUE",
    "asdict_p",
    "day_mix",
    "day_stats_from_eq",
    "dd25",
    "latest_intents",
    "passes_holdout",
    "passes_selection",
    "run",
]
