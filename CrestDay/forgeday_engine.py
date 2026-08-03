#!/usr/bin/env python3
"""
ForgeDay — long/short Gate-perp day book under SolidDay honesty protocol.

Mandate:
  - $500–$1k, Gate USDT-M perps, INTX banned, lev ≤ 4×, hard DD ≤ 25%
  - Train+mid selection only; hold one-shot after lock
  - Gap-through adverse stops + 5 bps; 7 bps one-way; funding proxy
  - PIT lagged-ADV membership; EOD flat UTC
  - Can short (regime-gated)

Sleeves (shared equity / DD):
  1) Coil breakout / breakdown (compression + Donchian + RS)
  2) Impulse continuation (short-horizon volume thrust)
  3) Optional residual fade (cross-sectional RS z-score mean-revert)
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from soliday_engine import (
    COST,
    DEFAULT_AUM,
    FUNDING_PER_8H,
    VENUE,
    _stop_fill,
    _target_fill,
    atr_series,
    btc_buyhold_eq,
    build_membership,
    list_symbols,
    load_panel,
    metrics_from_eq,
    nested_with_btc,
    roll_pctile,
)

PACK = Path(__file__).resolve().parent
HARD = 0.25  # user mandate (stricter packs use 0.15)


@dataclass(frozen=True)
class P:
    top_k: int = 25
    adv_lb_days: int = 21
    rebalance_days: int = 7
    min_hist_bars: int = 720
    # coil
    compress_lb: int = 48
    near_lb: int = 18
    near_buf: float = 0.04
    atr_pct_max: float = 0.25
    rs_lb: int = 18
    rs_min: float = 0.03
    vol_lb: int = 24
    vol_min: float = 0.8
    break_confirm: bool = True
    # impulse
    impulse_lb: int = 3
    impulse_min: float = 0.012
    impulse_vol: float = 1.3
    # residual fade
    fade_z: float = 2.2
    fade_vol_max: float = 1.2
    # risk / exits
    stop_atr: float = 1.2
    target_atr: float = 4.0
    trail_atr: float = 0.0
    atr_lb: int = 14
    risk_frac: float = 0.008
    lev_cap: float = 4.0
    max_names: int = 2
    max_trades_day: int = 6
    cool_bars: int = 6
    day_loss: float = 0.03
    day_bank: float = 0.05
    entry_start: int = 1
    entry_end: int = 20
    size_dd: float = 0.12
    hard: float = HARD
    roll_hwm: int = 60
    side_mode: str = "both"
    btc_gate: bool = True
    stop_adverse_bps: float = 5.0
    funding_per_8h: float = FUNDING_PER_8H
    require_full_hist: bool = False
    family: str = "coil"  # coil | impulse | fade | coil_impulse | all
    # sniper overlay (stricter coil, preferred)
    sniper_on: bool = True
    sniper_top_k: int = 15
    sniper_vol_min: float = 1.8
    sniper_atr_pct_max: float = 0.20
    sniper_near_buf: float = 0.02
    sniper_stop_atr: float = 1.5
    sniper_target_atr: float = 6.0
    sniper_risk_frac: float = 0.012
    sniper_cool: int = 12
    # Selective shorts (asymmetric): only fire under strong BTC down + sniper breakdown.
    # side_mode "both"/"short" enables; "long" disables regardless.
    short_btc_max: float = -0.02
    short_rs_min: float = 0.05
    short_vol_min: float = 2.0
    short_risk_frac: float = 0.01
    short_stop_atr: float = 1.5
    short_target_atr: float = 5.0
    # Pyramid into winners toward day bank (up to pyramid_max adds, never above lev_cap)
    pyramid_on: bool = False
    pyramid_r: float = 1.0
    pyramid_scale: float = 0.75
    pyramid_max: int = 1
    # Day-protect / trade management (defaults preserve legacy TrueDay/ForgeDay behavior)
    one_green: float = 0.0  # stop new entries once day_pnl >= this; 0 = off
    flatten_green: bool = False  # if True, flatten all when one_green hits
    be_r: float = 0.0  # move stop to cost-aware BE after this R multiple; 0 = off
    time_stop_bars: int = 0  # flatten after N bars in trade if >0
    protect_scale: float = 1.0  # risk scale for new entries after one_green (if not flatten)
    # Partial take-profit (same-bar target fill rules as full TP; remainder BE + runner)
    scale_out_r: float = 0.0  # R-multiple for first partial; 0 = off
    scale_out_frac: float = 0.5  # fraction of notional closed at scale-out
    stop_after_loss: bool = False  # no new entries after first fully-closed red trade of day


# Locked from results/forgeday_search.json (train+mid select, one-shot hold+stress)
P_LOCK = P(
    top_k=25,
    near_lb=18,
    compress_lb=48,
    rs_lb=18,
    rs_min=0.03,
    vol_min=0.8,
    atr_pct_max=0.25,
    near_buf=0.04,
    stop_atr=1.2,
    target_atr=5.0,
    risk_frac=0.008,
    family="coil",
    break_confirm=True,
    day_bank=0.05,
    day_loss=0.03,
    max_names=2,
    max_trades_day=6,
    cool_bars=6,
    side_mode="both",
    btc_gate=False,
    lev_cap=4.0,
    size_dd=0.12,
    hard=HARD,
    stop_adverse_bps=5.0,
    funding_per_8h=FUNDING_PER_8H,
    sniper_on=True,
    sniper_vol_min=1.8,
    sniper_risk_frac=0.014,
    sniper_stop_atr=1.5,
    sniper_target_atr=6.0,
    short_btc_max=-0.025,
    short_rs_min=0.05,
    short_vol_min=2.0,
    short_risk_frac=0.008,
    short_stop_atr=1.5,
    short_target_atr=5.0,
    pyramid_on=True,
    pyramid_r=0.75,
    pyramid_scale=1.25,
    pyramid_max=1,
)


def asdict_p(p: P) -> dict:
    return asdict(p)


def precompute(panel: dict, p: P) -> dict:
    feat = {}
    btc_c = panel["BTC"]["c"]
    idx = panel["BTC"]["index"]
    n = len(idx)
    rs_mat = []
    syms = panel["_symbols"]
    for a in syms:
        d = panel[a]
        h, l, c, v = d["h"], d["l"], d["c"], d["v"]
        valid = d["valid"]
        atr = atr_series(h, l, c, p.atr_lb)
        atr_pct = atr / np.maximum(c, 1e-12)
        atr_rank = roll_pctile(np.where(valid, atr_pct, np.nan), p.compress_lb)
        hh = pd.Series(h).rolling(p.near_lb, min_periods=p.near_lb).max().shift(1).to_numpy(float)
        ll = pd.Series(l).rolling(p.near_lb, min_periods=p.near_lb).min().shift(1).to_numpy(float)
        near_hi = c / np.maximum(hh, 1e-12) - 1.0
        near_lo = c / np.maximum(ll, 1e-12) - 1.0
        alt_ret = pd.Series(c).pct_change(p.rs_lb).to_numpy(float)
        btc_ret = pd.Series(btc_c).pct_change(p.rs_lb).to_numpy(float)
        rs = alt_ret - btc_ret
        rs_mat.append(rs)
        ret_imp = pd.Series(c).pct_change(p.impulse_lb).to_numpy(float)
        vsma = pd.Series(v).rolling(p.vol_lb, min_periods=p.vol_lb).mean().to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            vol_r = np.where(vsma > 0, v / vsma, np.nan)
        adv_bars = max(24, p.adv_lb_days * 24)
        adv = pd.Series(v).rolling(adv_bars, min_periods=adv_bars).sum().to_numpy(float)
        adv_lag = np.roll(adv, 24)
        adv_lag[:24] = np.nan
        first = int(np.argmax(valid)) if valid.any() else n
        age = np.arange(n) - first
        feat[a] = {
            "atr": atr,
            "atr_rank": atr_rank,
            "hh": hh,
            "ll": ll,
            "near_hi": near_hi,
            "near_lo": near_lo,
            "rs": rs,
            "vol_r": vol_r,
            "adv_lag": adv_lag,
            "age": age,
            "valid": valid,
            "ret_imp": ret_imp,
            **{k: d[k] for k in ("o", "h", "l", "c", "v")},
        }
    # cross-sectional RS z for fade sleeve (causal: uses only current bar RS vs peers)
    R = np.vstack(rs_mat)  # (n_sym, n_bars)
    mu = np.nanmean(R, axis=0)
    sd = np.nanstd(R, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = (R - mu) / np.where(sd > 1e-12, sd, np.nan)
    for j, a in enumerate(syms):
        feat[a]["rs_z"] = Z[j]
    feat["_index"] = idx
    feat["_symbols"] = syms
    feat["_btc_c"] = btc_c
    feat["_btc_ret_24"] = pd.Series(btc_c).pct_change(24).to_numpy(float)
    return feat


def _dd25(nest: dict, keys=("train", "mid", "hold", "full")) -> bool:
    return all(nest[k]["max_drawdown"] > -HARD - 1e-12 for k in keys)


def simulate(feat: dict, p: P, cost: float = COST, aum: float = DEFAULT_AUM, funding_mult: float = 1.0):
    mem = build_membership(feat, p)
    p_snipe = replace(
        p,
        top_k=p.sniper_top_k,
        vol_min=p.sniper_vol_min,
        atr_pct_max=p.sniper_atr_pct_max,
        near_buf=p.sniper_near_buf,
        stop_atr=p.sniper_stop_atr,
        target_atr=p.sniper_target_atr,
        risk_frac=p.sniper_risk_frac,
        cool_bars=p.sniper_cool,
        max_names=1,
        max_trades_day=2,
    )
    mem_s = build_membership(feat, p_snipe) if p.sniper_on else {}

    idx = feat["_index"]
    symbols = feat["_symbols"]
    days = np.array([t.date() for t in idx])
    day_ix: dict = defaultdict(list)
    day_list = []
    for i, d in enumerate(days):
        day_ix[d].append(i)
        if not day_list or day_list[-1] != d:
            day_list.append(d)

    equity = float(aum)
    eod, eq_hist = [], []
    ntr = 0
    traded_names: set[str] = set()
    funding_paid = 0.0
    fund_h = float(p.funding_per_8h) / 8.0 * funding_mult
    day_wins = 0
    day_count = 0
    bank_hits = 0

    def peak_now() -> float:
        if not eq_hist:
            return equity
        w = eq_hist[-p.roll_hwm :] if p.roll_hwm > 0 else eq_hist
        return max(max(w), equity)

    fam_coil = p.family in ("coil", "coil_impulse", "all")
    fam_imp = p.family in ("impulse", "coil_impulse", "all")
    fam_fade = p.family in ("fade", "all")

    for d in day_list:
        idxs = day_ix[d]
        members = set(mem.get(d, set()))
        if p.sniper_on:
            members |= set(mem_s.get(d, set()))
        day_start_eq = equity
        if len(idxs) < max(p.entry_start + 2, 4) or not members:
            eod.append(equity)
            eq_hist.append(equity)
            day_count += 1
            continue

        peak = peak_now()
        cur_dd = 1.0 - equity / peak if peak > 0 else 0.0
        if cur_dd >= p.hard - 0.002:
            scale = 0.05
        elif cur_dd <= p.size_dd:
            scale = 1.0
        else:
            scale = max(0.05, (p.hard - cur_dd) / max(p.hard - p.size_dd, 1e-9))

        positions: dict[str, dict] = {}
        pending: dict[str, tuple] = {}  # a -> (side, stop_atr, target_atr, risk_frac, cool)
        cool: dict[str, int] = {a: 0 for a in symbols}
        trades_today = 0
        day_pnl = 0.0
        banked = False
        greened = False
        entry_scale = 1.0
        red_stop = False

        def flatten(asset: str, px: float, frac: float = 1.0) -> None:
            nonlocal equity, day_pnl, ntr, red_stop
            pos = positions.get(asset)
            if not pos:
                return
            side = pos["side"]
            frac = float(min(max(frac, 0.0), 1.0))
            if frac <= 0:
                return
            fill = px * (1 - cost) if side > 0 else px * (1 + cost)
            notion_close = abs(pos["notion"]) * frac
            pnl = notion_close * side * (fill / pos["entry"] - 1.0)
            equity *= 1.0 + pnl
            day_pnl += pnl
            ntr += 1
            fully = frac >= 0.999
            if frac >= 0.999:
                del positions[asset]
            else:
                pos["notion"] = abs(pos["notion"]) - notion_close
                if pos["notion"] <= 0.05:
                    del positions[asset]
                    fully = True
            if fully and p.stop_after_loss and pnl < 0:
                red_stop = True

        for k, i in enumerate(idxs):
            peak = peak_now()
            cur_dd = 1.0 - equity / peak if peak > 0 else 0.0
            if cur_dd >= p.hard - 0.002:
                scale = 0.05
                if positions and cur_dd >= p.hard:
                    for a in list(positions):
                        flatten(a, feat[a]["c"][i])
                    break
            elif cur_dd <= p.size_dd:
                scale = 1.0
            else:
                scale = max(0.05, (p.hard - cur_dd) / max(p.hard - p.size_dd, 1e-9))

            if positions and fund_h > 0:
                gross = sum(abs(pos["notion"]) for pos in positions.values())
                if gross > 0:
                    fee = gross * fund_h
                    equity *= 1.0 - fee
                    day_pnl -= fee
                    funding_paid += fee

            for a in list(positions):
                pos = positions[a]
                f = feat[a]
                if not f["valid"][i] or not np.isfinite(f["c"][i]):
                    continue
                a_tr = f["atr"][i]
                c, h, l, o = f["c"][i], f["h"][i], f["l"][i], f["o"][i]
                pos["bars"] = int(pos.get("bars", 0)) + 1
                # Cost-aware break-even after be_r (uses initial stop distance)
                if p.be_r > 0 and not pos.get("be") and np.isfinite(pos.get("init_stop", np.nan)):
                    stop_dist = abs(pos["entry"] - pos["init_stop"])
                    move = pos["side"] * (c - pos["entry"])
                    if stop_dist > 1e-12 and move / stop_dist >= p.be_r:
                        be_px = pos["entry"] * (1 + cost) if pos["side"] > 0 else pos["entry"] * (1 - cost)
                        if pos["side"] > 0:
                            pos["stop"] = max(pos["stop"], be_px)
                        else:
                            pos["stop"] = min(pos["stop"], be_px)
                        pos["be"] = True
                if p.trail_atr > 0 and np.isfinite(a_tr):
                    if pos["side"] > 0:
                        pos["trail"] = max(pos["trail"], c - p.trail_atr * a_tr)
                        pos["stop"] = max(pos["stop"], pos["trail"])
                    else:
                        t = c + p.trail_atr * a_tr
                        pos["trail"] = min(pos["trail"], t) if pos["trail"] else t
                        pos["stop"] = min(pos["stop"], pos["trail"])
                if p.time_stop_bars > 0 and pos["bars"] >= p.time_stop_bars:
                    cool_b = int(pos.get("cool", p.cool_bars))
                    flatten(a, c)
                    cool[a] = cool_b
                    continue
                # Partial scale-out at R, then BE remainder
                if (
                    p.scale_out_r > 0
                    and not pos.get("scaled")
                    and np.isfinite(pos.get("init_stop", np.nan))
                    and a in positions
                ):
                    sd0 = abs(pos["entry"] - pos["init_stop"])
                    if sd0 > 1e-12:
                        so = pos["entry"] + pos["side"] * p.scale_out_r * sd0
                        hit = (pos["side"] > 0 and h >= so) or (pos["side"] < 0 and l <= so)
                        if hit:
                            cool_b = int(pos.get("cool", p.cool_bars))
                            fill_px = _target_fill(pos["side"], so, o)
                            flatten(a, fill_px, frac=p.scale_out_frac)
                            if a in positions:
                                be_px = pos["entry"] * (1 + cost) if pos["side"] > 0 else pos["entry"] * (1 - cost)
                                if pos["side"] > 0:
                                    pos["stop"] = max(pos["stop"], be_px)
                                else:
                                    pos["stop"] = min(pos["stop"], be_px)
                                pos["scaled"] = True
                                pos["be"] = True
                            else:
                                cool[a] = cool_b
                            continue
                if a not in positions:
                    continue
                pos = positions[a]
                if pos["side"] > 0:
                    if l <= pos["stop"]:
                        flatten(a, _stop_fill(1, pos["stop"], o, h, l, p.stop_adverse_bps))
                        cool[a] = pos.get("cool", p.cool_bars)
                    elif h >= pos["target"]:
                        flatten(a, _target_fill(1, pos["target"], o))
                else:
                    if h >= pos["stop"]:
                        flatten(a, _stop_fill(-1, pos["stop"], o, h, l, p.stop_adverse_bps))
                        cool[a] = pos.get("cool", p.cool_bars)
                    elif l <= pos["target"]:
                        flatten(a, _target_fill(-1, pos["target"], o))

                # Pyramid: signal on close → fill NEXT bar open (no same-bar open optimism)
                if p.pyramid_on and a in positions:
                    pos = positions[a]
                    n_pyr = int(pos.get("n_pyr", 0))
                    if n_pyr < p.pyramid_max and not pos.get("pyr_pending"):
                        stop_dist = abs(pos["entry"] - pos["init_stop"]) / max(pos["entry"], 1e-12)
                        move = pos["side"] * (c / pos["entry"] - 1.0)
                        need_r = p.pyramid_r * (1.0 + 0.5 * n_pyr)
                        if stop_dist > 1e-8 and move / stop_dist >= need_r and scale > 0.05:
                            pos["pyr_pending"] = True

            if p.one_green > 0 and day_pnl >= p.one_green:
                greened = True
                entry_scale = float(p.protect_scale)
                if p.flatten_green:
                    for a in list(positions):
                        flatten(a, feat[a]["c"][i])
                    break

            if day_pnl <= -p.day_loss or day_pnl >= p.day_bank:
                for a in list(positions):
                    flatten(a, feat[a]["c"][i])
                if day_pnl >= p.day_bank:
                    banked = True
                    bank_hits += 1
                break

            # Fill pyramid adds queued on prior close (next-open only)
            for a in list(positions):
                pos = positions[a]
                if not pos.get("pyr_pending"):
                    continue
                f = feat[a]
                if not f["valid"][i] or not np.isfinite(f["o"][i]):
                    pos["pyr_pending"] = False
                    continue
                add_px = f["o"][i] * (1 + cost) if pos["side"] > 0 else f["o"][i] * (1 - cost)
                add_n = abs(pos["notion"]) * p.pyramid_scale
                cap = p.lev_cap / max(p.max_names, 1)
                new_n = min(abs(pos["notion"]) + add_n, cap)
                added = new_n - abs(pos["notion"])
                if added > 0.05 and np.isfinite(add_px) and add_px > 0:
                    pos["entry"] = (pos["entry"] * abs(pos["notion"]) + add_px * added) / new_n
                    pos["notion"] = new_n
                    pos["n_pyr"] = int(pos.get("n_pyr", 0)) + 1
                pos["pyr_pending"] = False

            for a, meta in list(pending.items()):
                if red_stop:
                    continue
                if greened and p.flatten_green:
                    continue
                if greened and entry_scale <= 0.05:
                    continue
                if a in positions or cool[a] > 0 or scale <= 0.05:
                    continue
                if len(positions) >= p.max_names or trades_today >= p.max_trades_day:
                    continue
                side, stop_atr, target_atr, risk_frac, cool_b = meta
                f = feat[a]
                if not f["valid"][i] or not np.isfinite(f["o"][i]):
                    continue
                px = f["o"][i] * (1 + cost) if side > 0 else f["o"][i] * (1 - cost)
                a_tr = f["atr"][i - 1] if i > 0 and np.isfinite(f["atr"][i - 1]) else f["atr"][i]
                if not (np.isfinite(a_tr) and a_tr > 0 and px > 0):
                    continue
                sd = stop_atr * a_tr
                nf = min(p.lev_cap, risk_frac * px / max(sd, px * 1e-4)) * scale * entry_scale
                nf = min(nf, p.lev_cap / max(p.max_names, 1))
                if nf <= 0.05:
                    continue
                stop = px - sd if side > 0 else px + sd
                target = px + target_atr * a_tr if side > 0 else px - target_atr * a_tr
                positions[a] = {
                    "side": side,
                    "entry": px,
                    "stop": stop,
                    "init_stop": stop,
                    "target": target,
                    "trail": stop,
                    "notion": nf,
                    "cool": cool_b,
                    "n_pyr": 0,
                    "pyr_pending": False,
                    "be": False,
                    "bars": 0,
                    "scaled": False,
                }
                trades_today += 1
                traded_names.add(a)
            pending.clear()

            for a in symbols:
                if cool[a] > 0:
                    cool[a] -= 1

            if i == idxs[-1]:
                for a in list(positions):
                    flatten(a, feat[a]["c"][i])
                break

            if banked or red_stop or (greened and p.flatten_green) or k < p.entry_start or k > p.entry_end:
                continue
            if len(positions) >= p.max_names or trades_today >= p.max_trades_day or scale <= 0.05:
                continue
            if greened and entry_scale <= 0.05:
                continue

            scored = []
            btc_r = feat["_btc_ret_24"][i]
            sniper_members = mem_s.get(d, set()) if p.sniper_on else set()

            for a in members:
                if a in positions or cool[a] > 0:
                    continue
                f = feat[a]
                if not f["valid"][i]:
                    continue
                ar, nhi, nlo, rs, vr = (
                    f["atr_rank"][i],
                    f["near_hi"][i],
                    f["near_lo"][i],
                    f["rs"][i],
                    f["vol_r"][i],
                )
                hh, ll, c = f["hh"][i], f["ll"][i], f["c"][i]
                ri = f["ret_imp"][i]
                rz = f["rs_z"][i]
                if not all(np.isfinite(x) for x in (ar, nhi, nlo, rs, hh, ll, c)):
                    continue

                allow_long = p.side_mode in ("long", "both")
                allow_short = p.side_mode in ("short", "both")

                # --- sniper coil longs (preferred) ---
                if p.sniper_on and a in sniper_members and ar <= p.sniper_atr_pct_max:
                    if not (np.isfinite(vr) and vr < p.sniper_vol_min):
                        long_ok = allow_long and nhi >= -p.sniper_near_buf and rs >= p.rs_min
                        if long_ok and p.break_confirm:
                            long_ok = c > hh
                        if long_ok and p.btc_gate and (not np.isfinite(btc_r) or btc_r <= 0):
                            long_ok = False
                        if long_ok:
                            score = 100.0 + (1.0 - ar) * 2.0 + rs * 5.0 + min(vr if np.isfinite(vr) else 0, 4) * 0.2
                            scored.append(
                                (
                                    score,
                                    a,
                                    1,
                                    p.sniper_stop_atr,
                                    p.sniper_target_atr,
                                    p.sniper_risk_frac,
                                    p.sniper_cool,
                                )
                            )

                # --- selective shorts (asymmetric: BTC down + high-vol breakdown only) ---
                if allow_short and np.isfinite(btc_r) and btc_r <= p.short_btc_max:
                    short_ok = (
                        nlo <= p.sniper_near_buf
                        and rs <= -abs(p.short_rs_min)
                        and np.isfinite(vr)
                        and vr >= p.short_vol_min
                        and ar <= p.sniper_atr_pct_max
                    )
                    if short_ok and p.break_confirm:
                        short_ok = c < ll
                    if short_ok:
                        score = 110.0 + (-rs) * 5.0 + min(vr, 4.0) * 0.25 + max(-btc_r, 0) * 5.0
                        scored.append(
                            (
                                score,
                                a,
                                -1,
                                p.short_stop_atr,
                                p.short_target_atr,
                                p.short_risk_frac,
                                p.sniper_cool,
                            )
                        )

                # --- coil longs ---
                if fam_coil and ar <= p.atr_pct_max:
                    if not (p.vol_min > 0 and np.isfinite(vr) and vr < p.vol_min):
                        long_ok = allow_long and nhi >= -p.near_buf and rs >= p.rs_min
                        if long_ok and p.break_confirm:
                            long_ok = c > hh
                        if long_ok and p.btc_gate and (not np.isfinite(btc_r) or btc_r <= 0):
                            long_ok = False
                        if long_ok:
                            score = (1.0 - ar) * 2.0 + rs * 5.0 + max(nhi + p.near_buf, 0.0) * 3.0
                            if np.isfinite(vr):
                                score += min(vr, 3.0) * 0.15
                            scored.append((score, a, 1, p.stop_atr, p.target_atr, p.risk_frac, p.cool_bars))

                # --- impulse longs only (shorts reserved for selective path) ---
                if fam_imp and np.isfinite(ri) and np.isfinite(vr):
                    if ri >= p.impulse_min and vr >= p.impulse_vol and allow_long:
                        ok = not (p.btc_gate and (not np.isfinite(btc_r) or btc_r <= 0))
                        if ok:
                            score = 10.0 + abs(ri) * 20.0 + min(vr, 4.0) * 0.3
                            scored.append((score, a, 1, p.stop_atr, p.target_atr, p.risk_frac, p.cool_bars))

                # --- residual fade longs only ---
                if fam_fade and np.isfinite(rz) and np.isfinite(vr):
                    if rz <= -p.fade_z and vr <= p.fade_vol_max and ar <= 0.6 and allow_long:
                        ok = not (p.btc_gate and np.isfinite(btc_r) and btc_r < -0.02)
                        if ok:
                            score = 5.0 + abs(rz)
                            scored.append(
                                (score, a, 1, p.stop_atr, p.target_atr * 0.7, p.risk_frac * 0.7, p.cool_bars)
                            )

            scored.sort(reverse=True)
            slots = p.max_names - len(positions)
            used = set(positions) | set(pending)
            picked = 0
            for row in scored:
                if picked >= slots or trades_today + len(pending) >= p.max_trades_day:
                    break
                _, a, side, sa, ta, rf, cb = row
                if a in used:
                    continue
                pending[a] = (side, sa, ta, rf, cb)
                used.add(a)
                picked += 1

        eod.append(equity)
        eq_hist.append(equity)
        day_count += 1
        if equity > day_start_eq:
            day_wins += 1

    eq = np.asarray(eod, dtype=float)
    m = metrics_from_eq(eq, ntr)
    m["n_names_traded"] = int(len(traded_names))
    m["names_traded"] = sorted(traded_names)
    m["funding_drag_frac"] = float(funding_paid)
    m["venue"] = VENUE
    m["day_win_rate"] = float(day_wins / day_count) if day_count else 0.0
    m["bank_hits"] = int(bank_hits)
    m["bank_hit_rate"] = float(bank_hits / day_count) if day_count else 0.0
    return m, eq


def run(p: P = P_LOCK, aum: float = DEFAULT_AUM, cost_mult: float = 1.0, fund_mult: float = 1.0):
    panel = load_panel(min_bars=2000, require_full=bool(p.require_full_hist))
    feat = precompute(panel, p)
    m, eq = simulate(feat, p, cost=COST * cost_mult, aum=aum, funding_mult=fund_mult)
    btc = btc_buyhold_eq(feat, aum=aum)
    return nested_with_btc(eq, m, btc), m, eq, panel


def passes_selection(nest: dict) -> bool:
    if not _dd25(nest, keys=("train", "mid")):
        return False
    if nest["train"]["total_return"] <= 0 or nest["mid"]["total_return"] <= 0:
        return False
    if nest["mid"]["total_return"] <= nest["btc_mid"]["total_return"]:
        return False
    if nest["train"]["sharpe_ratio"] < 0.5 or nest["mid"]["sharpe_ratio"] < 0.9:
        return False
    if nest["full"]["trades_per_year"] < 80:
        return False
    # anti-fantasy: calendar avg daily > 1.5% on train+mid is near-certain leakage under 4×
    avg_sel = 0.5 * (nest["train"]["avg_daily_return"] + nest["mid"]["avg_daily_return"])
    if avg_sel > 0.015:
        return False
    if nest["train"]["total_return"] > 20.0 or nest["mid"]["total_return"] > 20.0:
        return False
    return True


def passes_holdout(nest: dict) -> bool:
    if not _dd25(nest, keys=("hold", "full")):
        return False
    if nest["hold"]["total_return"] <= 0:
        return False
    if nest["hold"]["total_return"] <= nest["btc_hold"]["total_return"]:
        return False
    if nest["hold"]["sharpe_ratio"] < 0.8:
        return False
    return True


def selection_score(nest: dict) -> float:
    """HOLD ABSENT. Prefer mid edge, consistency, active-day punch, DD cushion."""
    mid_edge = nest["mid"]["total_return"] - nest["btc_mid"]["total_return"]
    avg_act = 0.5 * (
        nest["train"].get("avg_active_day_return", 0) + nest["mid"].get("avg_active_day_return", 0)
    )
    avg_day = 0.5 * (nest["train"]["avg_daily_return"] + nest["mid"]["avg_daily_return"])
    return (
        nest["mid"]["total_return"] * 3.0
        + nest["train"]["total_return"] * 2.0
        + mid_edge * 2.5
        + nest["mid"]["sharpe_ratio"] * 0.6
        + nest["train"]["sharpe_ratio"] * 0.3
        + 8.0 * max(0.0, HARD + nest["train"]["max_drawdown"])
        + 8.0 * max(0.0, HARD + nest["mid"]["max_drawdown"])
        + 40.0 * avg_day
        + 25.0 * avg_act
        + 0.02 * nest["full"].get("n_names_traded", 0)
    )


def set_lock(p: P) -> None:
    global P_LOCK
    P_LOCK = p


dd25 = _dd25
