#!/usr/bin/env python3
"""
TrueDay — deflated, prod-oriented Gate-perp day book (ForgeDay audit fix).

Honesty vs ForgeDay:
  - pyramid OFF (ForgeDay same-bar open-after-close add removed)
  - require_full_hist=True (FULL_HIST universe only)
  - anti-fantasy avg daily ≤ 0.8%; hold return cap 120%
  - 7 bps + funding + gap-through adverse stops; stress ×1.5
  - train+mid select / one-shot hold; no hold in score

Mandate: $500–$1k, Gate USDT-M perps, INTX banned, lev ≤ 4×, DD ≤ 25%, selective shorts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from forgeday_engine import (
    COST,
    DEFAULT_AUM,
    FUNDING_PER_8H,
    HARD,
    P,
    VENUE,
    build_membership,
    dd25,
    precompute,
    run as _run_raw,
)
from soliday_engine import FULL_HIST, load_panel

PACK = Path(__file__).resolve().parent

# Locked: results/trueday_locked.json — full_hist, no pyramid, selective shorts
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
    vol_min=1.0,
    break_confirm=True,
    stop_atr=1.5,
    target_atr=4.0,
    trail_atr=0.0,
    atr_lb=14,
    risk_frac=0.005,
    lev_cap=4.0,
    max_names=2,
    max_trades_day=4,
    cool_bars=6,
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
    sniper_target_atr=6.0,
    sniper_risk_frac=0.01,
    sniper_cool=12,
    short_btc_max=-0.025,
    short_rs_min=0.05,
    short_vol_min=2.0,
    short_risk_frac=0.008,
    short_stop_atr=1.5,
    short_target_atr=5.0,
    pyramid_on=False,
    pyramid_r=1.0,
    pyramid_scale=0.0,
    pyramid_max=0,
)


def asdict_p(p: P = P_LOCK) -> dict:
    return asdict(p)


def run(p: P = P_LOCK, aum: float = DEFAULT_AUM, cost_mult: float = 1.0, fund_mult: float = 1.0):
    # Force honesty knobs even if caller drifts
    p = replace(p, require_full_hist=True, pyramid_on=False, pyramid_max=0, hard=HARD, lev_cap=min(p.lev_cap, 4.0))
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
    # Reject ForgeDay-style monster hold (audit: +238% / ~6m)
    if nest["hold"]["total_return"] > 1.2:
        return False
    return True


def day_stats_from_eq(eq: np.ndarray) -> dict:
    r = np.diff(eq) / eq[:-1]
    r = r[np.isfinite(r)]
    active = r[np.abs(r) > 1e-12]
    return {
        "avg_daily": float(r.mean()) if len(r) else 0.0,
        "avg_active": float(active.mean()) if len(active) else 0.0,
        "day_win_rate": float((r > 0).mean()) if len(r) else 0.0,
        "active_win_rate": float((active > 0).mean()) if len(active) else 0.0,
        "pct_ge_5pct": float((r >= 0.05).mean()) if len(r) else 0.0,
        "pct_active_ge_5pct": float((active >= 0.05).mean()) if len(active) else 0.0,
        "pct_flat": float((np.abs(r) <= 1e-12).mean()) if len(r) else 0.0,
        "pct_nonnegative": float((r >= -1e-12).mean()) if len(r) else 0.0,
        "best_day": float(r.max()) if len(r) else 0.0,
        "worst_day": float(r.min()) if len(r) else 0.0,
    }


def latest_intents(p: P = P_LOCK, aum: float = DEFAULT_AUM, equity: float | None = None) -> dict:
    """Causal live intents from the latest completed bar → next-open entries.

    Returns Gate USDT-M style order intents (not signed API calls).
    INTX banned. Max lev 4×. EOD-flat day book — caller must flatten by UTC day end.
    """
    equity = float(equity if equity is not None else aum)
    panel = load_panel(min_bars=2000, require_full=True)
    feat = precompute(panel, p)
    mem = build_membership(feat, p)
    mem_s = build_membership(feat, replace(p, top_k=p.sniper_top_k)) if p.sniper_on else {}
    idx = feat["_index"]
    i = len(idx) - 1  # last completed bar
    d = idx[i].date()
    members = set(mem.get(d, set())) | set(mem_s.get(d, set()))
    btc_r = feat["_btc_ret_24"][i]
    hour = int(idx[i].hour)
    intents = []
    if hour < p.entry_start or hour > p.entry_end:
        return {
            "asof": str(idx[i]),
            "venue": VENUE,
            "intx": False,
            "equity": equity,
            "intents": [],
            "risk": _risk_limits(p, equity),
            "note": "outside entry window — flat / no new entries",
        }

    scored = []
    sniper_m = mem_s.get(d, set())
    for a in members:
        f = feat[a]
        if not f["valid"][i]:
            continue
        ar, nhi, nlo, rs, vr = f["atr_rank"][i], f["near_hi"][i], f["near_lo"][i], f["rs"][i], f["vol_r"][i]
        hh, ll, c = f["hh"][i], f["ll"][i], f["c"][i]
        if not all(np.isfinite(x) for x in (ar, nhi, nlo, rs, hh, ll, c)):
            continue
        allow_long = p.side_mode in ("long", "both")
        allow_short = p.side_mode in ("short", "both")
        if p.sniper_on and a in sniper_m and ar <= p.sniper_atr_pct_max and np.isfinite(vr) and vr >= p.sniper_vol_min:
            long_ok = allow_long and nhi >= -p.sniper_near_buf and rs >= p.rs_min and c > hh
            if long_ok and p.btc_gate and (not np.isfinite(btc_r) or btc_r <= 0):
                long_ok = False
            if long_ok:
                scored.append((100 + rs * 5, a, 1, p.sniper_stop_atr, p.sniper_target_atr, p.sniper_risk_frac, "sniper"))
            short_ok = (
                allow_short
                and nlo <= p.sniper_near_buf
                and rs <= -abs(p.short_rs_min)
                and c < ll
                and np.isfinite(btc_r)
                and btc_r <= p.short_btc_max
                and vr >= p.short_vol_min
            )
            if short_ok:
                scored.append((100 - rs * 5, a, -1, p.short_stop_atr, p.short_target_atr, p.short_risk_frac, "sniper_short"))
        if ar <= p.atr_pct_max and not (p.vol_min > 0 and np.isfinite(vr) and vr < p.vol_min):
            long_ok = allow_long and nhi >= -p.near_buf and rs >= p.rs_min and c > hh
            if long_ok and p.btc_gate and (not np.isfinite(btc_r) or btc_r <= 0):
                long_ok = False
            if long_ok:
                scored.append((50 + rs * 5, a, 1, p.stop_atr, p.target_atr, p.risk_frac, "coil"))
            short_ok = (
                allow_short
                and nlo <= p.near_buf
                and rs <= -abs(p.short_rs_min)
                and c < ll
                and np.isfinite(btc_r)
                and btc_r <= p.short_btc_max
                and np.isfinite(vr)
                and vr >= p.short_vol_min
            )
            if short_ok:
                scored.append((50 - rs * 5, a, -1, p.stop_atr, p.target_atr, p.risk_frac * 0.8, "coil_short"))

    scored.sort(reverse=True)
    used = set()
    for row in scored:
        if len(intents) >= p.max_names:
            break
        _, a, side, sa, ta, rf, tag = row
        if a in used:
            continue
        f = feat[a]
        # next-open fill assumption; size from last ATR
        a_tr = f["atr"][i]
        px = float(f["c"][i])  # mark; live fill = next open
        if not (np.isfinite(a_tr) and a_tr > 0 and px > 0):
            continue
        sd = sa * a_tr
        nf = min(p.lev_cap, rf * px / max(sd, px * 1e-4))
        nf = min(nf, p.lev_cap / max(p.max_names, 1))
        if nf < 0.05:
            continue
        stop = px - sd if side > 0 else px + sd
        target = px + ta * a_tr if side > 0 else px - ta * a_tr
        notional_usd = equity * nf
        intents.append(
            {
                "contract": f"{a}_USDT",
                "settle": "usdt",
                "side": "buy" if side > 0 else "sell",
                "reduce_only": False,
                "order_type": "market",  # research: next open ≈ market on new bar
                "tif": "ioc",
                "target_notional_usd": round(notional_usd, 2),
                "leverage": round(float(nf), 4),
                "stop_price": round(float(stop), 8),
                "take_profit_price": round(float(target), 8),
                "signal": tag,
                "mark_close": round(px, 8),
                "fill_rule": "next_bar_open",
            }
        )
        used.add(a)

    return {
        "asof": str(idx[i]),
        "venue": VENUE,
        "intx": False,
        "universe": "FULL_HIST",
        "eligible": sorted(FULL_HIST),
        "equity": equity,
        "btc_ret_24": float(btc_r) if np.isfinite(btc_r) else None,
        "intents": intents,
        "risk": _risk_limits(p, equity),
        "flatten_utc_eod": True,
        "cost_model_bps_one_way": COST * 1e4,
    }


def _risk_limits(p: P, equity: float) -> dict:
    return {
        "lev_cap": p.lev_cap,
        "hard_dd": p.hard,
        "day_loss": p.day_loss,
        "day_bank": p.day_bank,
        "max_names": p.max_names,
        "max_trades_day": p.max_trades_day,
        "equity": equity,
        "max_gross_notional_usd": round(equity * p.lev_cap, 2),
        "intx_banned": True,
        "pyramid": False,
    }


def set_lock(p: P) -> None:
    global P_LOCK
    P_LOCK = p
