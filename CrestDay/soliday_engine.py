#!/usr/bin/env python3
"""
SolidDay — honest Gate-perp day trader research engine.

Fixes vs QuadDay critique:
  1. Train+mid selection only; hold is one-shot after lock (never in score/ASCEND)
  2. Adverse stop fills (gap-through open) + stop_adverse_bps; targets not gifted
  3. Survivorship: PIT lagged-ADV membership + min_hist_bars (no hand-picked sleeve).
     Optional require_full_hist=True stress; residual listing bias documented.
  4. Derivatives costs: 10 bps one-way + per-hour funding proxy on gross notional
  5. Venue: Gate USDT-M perps style; INTX explicitly banned

Mandate: $500–$1k, lev ≤ 4×, DD ≤ 15%, multi-trade days, EOD flat UTC.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PACK = Path(__file__).resolve().parent
DATA = PACK / "data" / "crypto_intraday" / "breakout"

# Retail Gate-perp style friction (stricter than QuadDay's optimistic 5 bps exact-stop)
COST = 0.0007  # 7 bps one-way
# Funding proxy: ~0.01% per 8h; charged per hour held on |notional|
FUNDING_PER_8H = 0.0001
DEFAULT_AUM = 1000.0
HARD = 0.15

# Survivorship control: only names with ~full panel history (see universe.json)
FULL_HIST = frozenset(
    {
        "AAVE",
        "ADA",
        "AKT",
        "AVAX",
        "BNB",
        "CFX",
        "COTI",
        "DEXE",
        "DOGE",
        "ENA",
        "EPIC",
        "ETH",
        "GRAM",
        "INJ",
        "KAS",
        "LTC",
        "NEAR",
        "ONDO",
        "SHIB",
        "SOL",
        "TRX",
        "WLD",
        "XLM",
        "XRP",
        "ZEC",
    }
)

# Never trade / never route
EXCLUDE = {
    "INTX",  # Coinbase International — venue ban
    "USDP",
    "USDC",
    "DAI",
    "TUSD",
    "FDUSD",
    "USDE",
    "XAUT",
    "PAXG",
    "QQQON",
    "NVDAON",
    "SPYON",
    "TSLAON",
    "AAPLON",
}

VENUE = "gate_usdt_perp"  # not coinbase_intx


@dataclass(frozen=True)
class P:
    top_k: int = 12
    adv_lb_days: int = 21
    rebalance_days: int = 10
    min_hist_bars: int = 720
    compress_lb: int = 48
    near_lb: int = 18
    near_buf: float = 0.04
    atr_pct_max: float = 0.25
    rs_lb: int = 18
    rs_min: float = 0.03
    vol_lb: int = 24
    vol_min: float = 1.0
    break_confirm: bool = True
    stop_atr: float = 1.2
    target_atr: float = 3.0
    trail_atr: float = 0.0
    atr_lb: int = 14
    risk_frac: float = 0.008
    lev_cap: float = 4.0
    max_names: int = 2
    max_trades_day: int = 4
    cool_bars: int = 6
    day_loss: float = 0.02
    entry_start: int = 2
    entry_end: int = 18
    size_dd: float = 0.10
    hard: float = HARD
    roll_hwm: int = 60
    side_mode: str = "long"
    btc_gate: bool = False  # if True, longs need BTC 24h ret > 0; shorts need < 0
    # Honest fill / funding knobs
    stop_adverse_bps: float = 5.0  # extra adverse on stop fills
    funding_per_8h: float = FUNDING_PER_8H
    require_full_hist: bool = False


# Locked SolidDay (results/soliday_search.json) — train+mid select, one-shot hold pass
P_LOCK = P(
    top_k=25,
    adv_lb_days=21,
    rebalance_days=10,
    min_hist_bars=720,
    compress_lb=48,
    near_lb=18,
    near_buf=0.04,
    atr_pct_max=0.25,
    rs_lb=18,
    rs_min=0.03,
    vol_lb=24,
    vol_min=0.8,
    break_confirm=True,
    stop_atr=1.2,
    target_atr=4.0,
    trail_atr=0.0,
    atr_lb=14,
    risk_frac=0.005,
    lev_cap=4.0,
    max_names=2,
    max_trades_day=4,
    cool_bars=6,
    day_loss=0.02,
    entry_start=2,
    entry_end=18,
    size_dd=0.10,
    hard=0.15,
    roll_hwm=60,
    side_mode="long",
    btc_gate=False,
    stop_adverse_bps=5.0,
    funding_per_8h=FUNDING_PER_8H,
    require_full_hist=False,
)


def asdict_p(p: P) -> dict:
    return asdict(p)


def _tradeable(sym: str, require_full: bool) -> bool:
    s = sym.upper()
    if s in EXCLUDE or s == "BTC" or s == "INTX":
        return False
    if require_full and s not in FULL_HIST:
        return False
    return True


def list_symbols(require_full: bool = True) -> list[str]:
    uni = DATA / "universe.json"
    if uni.exists():
        return [s for s in json.loads(uni.read_text())["symbols"] if _tradeable(s, require_full)]
    out = []
    for p in sorted(DATA.glob("*_1h.parquet")):
        sym = p.name.replace("_1h.parquet", "").upper()
        if _tradeable(sym, require_full):
            out.append(sym)
    return out


def load_bars(asset: str) -> pd.DataFrame:
    path = DATA / f"{asset.lower()}_1h.parquet"
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df.sort_index().dropna(subset=["Open", "High", "Low", "Close"])
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    return df


def atr_series(h, l, c, lb: int) -> np.ndarray:
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    return pd.Series(tr).rolling(lb, min_periods=lb).mean().to_numpy(dtype=float)


def roll_pctile(x: np.ndarray, lb: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)
    if n < lb or lb < 2:
        return out
    from numpy.lib.stride_tricks import sliding_window_view

    x2 = np.asarray(x, dtype=float).copy()
    w = sliding_window_view(x2, lb)
    last = w[:, -1]
    ok = np.isfinite(last)
    cmp = w <= last[:, None]
    pct = np.nanmean(np.where(np.isfinite(w), cmp, np.nan), axis=1)
    out[lb - 1 :] = np.where(ok, pct, np.nan)
    return out


def load_panel(min_bars: int = 2000, require_full: bool = True) -> dict:
    btc = load_bars("BTC")
    symbols = list_symbols(require_full=require_full)
    panels = {"BTC": btc}
    for a in symbols:
        try:
            df = load_bars(a)
        except Exception:
            continue
        if len(df) < min_bars:
            continue
        panels[a] = df
    idx = btc.index.sort_values().unique()
    out = {}
    for a, df in panels.items():
        d = df.reindex(idx)
        out[a] = {
            "o": d["Open"].to_numpy(float),
            "h": d["High"].to_numpy(float),
            "l": d["Low"].to_numpy(float),
            "c": d["Close"].to_numpy(float),
            "v": d["Volume"].fillna(0.0).to_numpy(float),
            "valid": d["Close"].notna().to_numpy(bool),
            "index": idx,
        }
    out["_symbols"] = [a for a in out if a not in ("BTC", "_symbols")]
    out["_venue"] = VENUE
    return out


def precompute(panel: dict, p: P) -> dict:
    feat = {}
    btc_c = panel["BTC"]["c"]
    idx = panel["BTC"]["index"]
    n = len(idx)
    for a in panel["_symbols"]:
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
            **{k: d[k] for k in ("o", "h", "l", "c", "v")},
        }
    btc_ret_24 = pd.Series(btc_c).pct_change(24).to_numpy(float)
    feat["_index"] = idx
    feat["_symbols"] = panel["_symbols"]
    feat["_btc_c"] = btc_c
    feat["_btc_ret_24"] = btc_ret_24
    return feat


def build_membership(feat: dict, p: P) -> dict:
    idx = feat["_index"]
    symbols = feat["_symbols"]
    days = np.array([t.date() for t in idx])
    day_list = []
    day_end_i = {}
    for i, d in enumerate(days):
        if not day_list or day_list[-1] != d:
            day_list.append(d)
        day_end_i[d] = i
    membership: dict = {}
    last_set: set[str] = set()
    last_rebal = None
    for di, d in enumerate(day_list):
        i = day_end_i[d]
        need = last_rebal is None or (di - last_rebal) >= p.rebalance_days or not last_set
        if need:
            ranked = []
            for a in symbols:
                adv = feat[a]["adv_lag"][i]
                age = feat[a]["age"][i]
                valid = feat[a]["valid"][i]
                if not valid or not np.isfinite(adv) or adv <= 0 or age < p.min_hist_bars:
                    continue
                ranked.append((adv, a))
            ranked.sort(reverse=True)
            last_set = {a for _, a in ranked[: p.top_k]}
            last_rebal = di
        membership[d] = set(last_set)
    return membership


def metrics_from_eq(eq: np.ndarray, n_trades: int) -> dict:
    r = np.diff(eq) / eq[:-1]
    r = r[np.isfinite(r)]
    years = max(len(r) / 365.25, 1e-9)
    sh = float(r.mean() / r.std() * math.sqrt(365)) if len(r) and r.std() > 0 else 0.0
    dd = float(np.min(eq / np.maximum.accumulate(eq) - 1.0)) if len(eq) else 0.0
    active = r[np.abs(r) > 1e-12]
    return {
        "total_return": float(eq[-1] / eq[0] - 1.0) if len(eq) and eq[0] > 0 else 0.0,
        "cagr": float((eq[-1] / eq[0]) ** (1 / years) - 1.0) if len(eq) and eq[0] > 0 and eq[-1] > 0 else 0.0,
        "sharpe_ratio": sh,
        "max_drawdown": dd,
        "avg_daily_return": float(r.mean()) if len(r) else 0.0,
        "avg_active_day_return": float(active.mean()) if len(active) else 0.0,
        "n_days": int(len(r)),
        "n_trades": int(n_trades),
        "trades_per_year": float(n_trades / years),
        "ending_equity": float(eq[-1]) if len(eq) else 0.0,
        "win_rate": float(np.mean(active > 0)) if len(active) else 0.0,
    }


def _stop_fill(side: int, stop: float, o: float, h: float, l: float, adverse_bps: float) -> float:
    """Pessimistic stop: gap-through open, else stop, plus adverse bps.

    adverse_bps < 0 → exact stop (parity / ablation only).
    """
    if adverse_bps < 0:
        return stop
    if side > 0:
        raw = o if (np.isfinite(o) and o < stop) else stop
        raw = min(raw, stop)
        return raw * (1.0 - adverse_bps / 1e4)
    raw = o if (np.isfinite(o) and o > stop) else stop
    raw = max(raw, stop)
    return raw * (1.0 + adverse_bps / 1e4)


def _target_fill(side: int, target: float, o: float) -> float:
    """Target fill: exact level unless open already through (then open)."""
    if side > 0:
        return o if (np.isfinite(o) and o > target) else target
    return o if (np.isfinite(o) and o < target) else target


def simulate(
    feat: dict,
    p: P,
    cost: float = COST,
    aum: float = DEFAULT_AUM,
    membership: dict | None = None,
    funding_mult: float = 1.0,
):
    if membership is None:
        membership = build_membership(feat, p)

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
    eod = []
    ntr = 0
    eq_hist: list[float] = []
    traded_names: set[str] = set()
    member_counts: list[int] = []
    funding_paid = 0.0
    fund_h = float(p.funding_per_8h) / 8.0 * funding_mult

    def peak_now() -> float:
        if not eq_hist:
            return equity
        w = eq_hist[-p.roll_hwm :] if p.roll_hwm > 0 else eq_hist
        return max(max(w), equity)

    for d in day_list:
        idxs = day_ix[d]
        members = membership.get(d, set())
        member_counts.append(len(members))
        if len(idxs) < max(p.entry_start + 2, 4) or not members:
            eod.append(equity)
            eq_hist.append(equity)
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
        pending: dict[str, int] = {}
        cool: dict[str, int] = {a: 0 for a in symbols}
        trades_today = 0
        day_pnl = 0.0

        def flatten(asset: str, px: float) -> None:
            nonlocal equity, day_pnl, ntr
            pos = positions.get(asset)
            if not pos:
                return
            side = pos["side"]
            fill = px * (1 - cost) if side > 0 else px * (1 + cost)
            pnl = abs(pos["notion"]) * side * (fill / pos["entry"] - 1.0)
            equity *= 1.0 + pnl
            day_pnl += pnl
            ntr += 1
            del positions[asset]

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

            # funding on open notionals each hour
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
                if p.trail_atr > 0 and np.isfinite(a_tr):
                    if pos["side"] > 0:
                        pos["trail"] = max(pos["trail"], c - p.trail_atr * a_tr)
                        pos["stop"] = max(pos["stop"], pos["trail"])
                    else:
                        t = c + p.trail_atr * a_tr
                        pos["trail"] = min(pos["trail"], t) if pos["trail"] else t
                        pos["stop"] = min(pos["stop"], pos["trail"])
                # Adverse stop first, then target (no gift)
                if pos["side"] > 0:
                    if l <= pos["stop"]:
                        flatten(a, _stop_fill(1, pos["stop"], o, h, l, p.stop_adverse_bps))
                        cool[a] = p.cool_bars
                    elif h >= pos["target"]:
                        flatten(a, _target_fill(1, pos["target"], o))
                else:
                    if h >= pos["stop"]:
                        flatten(a, _stop_fill(-1, pos["stop"], o, h, l, p.stop_adverse_bps))
                        cool[a] = p.cool_bars
                    elif l <= pos["target"]:
                        flatten(a, _target_fill(-1, pos["target"], o))

            if day_pnl <= -p.day_loss:
                for a in list(positions):
                    flatten(a, feat[a]["c"][i])
                break

            for a, side in list(pending.items()):
                if a in positions or cool[a] > 0 or scale <= 0.05:
                    continue
                if len(positions) >= p.max_names or trades_today >= p.max_trades_day:
                    continue
                f = feat[a]
                if not f["valid"][i] or not np.isfinite(f["o"][i]):
                    continue
                px = f["o"][i] * (1 + cost) if side > 0 else f["o"][i] * (1 - cost)
                a_tr = f["atr"][i - 1] if i > 0 and np.isfinite(f["atr"][i - 1]) else f["atr"][i]
                if not (np.isfinite(a_tr) and a_tr > 0 and px > 0):
                    continue
                sd = p.stop_atr * a_tr
                nf = min(p.lev_cap, p.risk_frac * px / max(sd, px * 1e-4)) * scale
                nf = min(nf, p.lev_cap / max(p.max_names, 1))
                if nf <= 0.05:
                    continue
                stop = px - sd if side > 0 else px + sd
                target = px + p.target_atr * a_tr if side > 0 else px - p.target_atr * a_tr
                positions[a] = {"side": side, "entry": px, "stop": stop, "target": target, "trail": stop, "notion": nf}
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

            if k < p.entry_start or k > p.entry_end:
                continue
            if len(positions) >= p.max_names or trades_today >= p.max_trades_day or scale <= 0.05:
                continue

            scored = []
            for a in members:
                if a in positions or cool[a] > 0:
                    continue
                f = feat[a]
                if not f["valid"][i]:
                    continue
                ar, nhi, nlo, rs, vr = f["atr_rank"][i], f["near_hi"][i], f["near_lo"][i], f["rs"][i], f["vol_r"][i]
                hh, ll, c = f["hh"][i], f["ll"][i], f["c"][i]
                if not all(np.isfinite(x) for x in (ar, nhi, nlo, rs, hh, ll, c)):
                    continue
                if ar > p.atr_pct_max:
                    continue
                if p.vol_min > 0 and np.isfinite(vr) and vr < p.vol_min:
                    continue

                btc_r = feat["_btc_ret_24"][i]
                long_ok = p.side_mode in ("long", "both") and nhi >= -p.near_buf and rs >= p.rs_min
                if long_ok and p.break_confirm:
                    long_ok = c > hh
                if long_ok and p.btc_gate and (not np.isfinite(btc_r) or btc_r <= 0):
                    long_ok = False
                short_ok = p.side_mode in ("short", "both") and nlo <= p.near_buf and rs <= -abs(p.rs_min)
                if short_ok and p.break_confirm:
                    short_ok = c < ll
                if short_ok and p.btc_gate and (not np.isfinite(btc_r) or btc_r >= 0):
                    short_ok = False

                if long_ok:
                    score = (1.0 - ar) * 2.0 + rs * 5.0 + max(nhi + p.near_buf, 0.0) * 3.0
                    if np.isfinite(vr):
                        score += min(vr, 3.0) * 0.15
                    scored.append((score, a, 1))
                if short_ok:
                    score = (1.0 - ar) * 2.0 + (-rs) * 5.0 + max(p.near_buf - nlo, 0.0) * 3.0
                    if np.isfinite(vr):
                        score += min(vr, 3.0) * 0.15
                    scored.append((score, a, -1))

            scored.sort(reverse=True)
            slots = p.max_names - len(positions)
            used = set(positions) | set(pending)
            picked = 0
            for _, a, side in scored:
                if picked >= slots or trades_today + len(pending) >= p.max_trades_day:
                    break
                if a in used:
                    continue
                pending[a] = side
                used.add(a)
                picked += 1

        eod.append(equity)
        eq_hist.append(equity)

    eq = np.asarray(eod, dtype=float)
    m = metrics_from_eq(eq, ntr)
    m["n_names_traded"] = int(len(traded_names))
    m["avg_universe_size"] = float(np.mean(member_counts)) if member_counts else 0.0
    m["names_traded"] = sorted(traded_names)
    m["funding_drag_frac"] = float(funding_paid)
    m["venue"] = VENUE
    return m, eq


def btc_buyhold_eq(feat: dict, aum: float = DEFAULT_AUM) -> np.ndarray:
    """Daily EOD BTC buy-hold on same calendar (costed entry once)."""
    idx = feat["_index"]
    c = feat["_btc_c"]
    days = np.array([t.date() for t in idx])
    day_list = []
    day_end = {}
    for i, d in enumerate(days):
        if not day_list or day_list[-1] != d:
            day_list.append(d)
        day_end[d] = i
    # enter at first valid close + cost
    eq = []
    units = None
    cash = aum
    for d in day_list:
        i = day_end[d]
        px = c[i]
        if not np.isfinite(px) or px <= 0:
            eq.append(eq[-1] if eq else aum)
            continue
        if units is None:
            units = (cash * (1 - COST)) / px
            cash = 0.0
        eq.append(units * px)
    return np.asarray(eq, dtype=float)


def nested(eq: np.ndarray, m: dict) -> dict:
    """50/25/25 nested. Hold is the last 25% — selection must ignore it."""
    n = len(eq)
    i1 = n // 2
    i2 = int(n * 0.75)

    def win(a, b, n_tr_est=None):
        path = eq[a : b + 1] / eq[a]
        r = np.diff(path) / path[:-1]
        # approximate trades by active days in window (reported separately on full)
        return metrics_from_eq(path, int(np.sum(np.abs(r) > 1e-12)))

    out = {"train": win(0, i1 - 1), "mid": win(i1, i2 - 1), "hold": win(i2, n - 1), "full": m}
    return out


def nested_with_btc(eq: np.ndarray, m: dict, btc_eq: np.ndarray) -> dict:
    nest = nested(eq, m)
    bm = metrics_from_eq(btc_eq, 0)
    bn = nested(btc_eq, bm)
    for k in ("train", "mid", "hold", "full"):
        nest[f"btc_{k}"] = bn[k]
    return nest


def dd15(nest: dict, hard: float = HARD, keys=("train", "mid", "hold", "full")) -> bool:
    return all(nest[k]["max_drawdown"] >= -hard - 1e-12 for k in keys)


def passes_selection(nest: dict, min_tpy: float = 100, min_names: int = 6) -> bool:
    """Train+mid only. Hold must NOT be consulted here (incl. no full-sample DD).

    Beat-BTC is required on *mid* (where BH is weak / negative), not train.
    An EOD-flat day book cannot fairly beat a roaring BTC bull on train; that
    bar would force beta-chasing, not day alpha.
    """
    f = nest["full"]
    if not dd15(nest, keys=("train", "mid")):
        return False
    if nest["train"]["total_return"] <= 0 or nest["mid"]["total_return"] <= 0:
        return False
    if nest["train"]["sharpe_ratio"] < 0.5 or nest["mid"]["sharpe_ratio"] < 1.0:
        return False
    # Mid must beat costed BTC (genuine relative alpha in that window)
    if nest["mid"]["total_return"] <= nest["btc_mid"]["total_return"]:
        return False
    if f["trades_per_year"] < min_tpy:
        return False
    if f.get("n_names_traded", 0) < min_names:
        return False
    sel_ret = (1 + nest["train"]["total_return"]) * (1 + nest["mid"]["total_return"]) - 1
    if sel_ret > 20.0:
        return False
    avg_sel = 0.5 * (nest["train"]["avg_daily_return"] + nest["mid"]["avg_daily_return"])
    if avg_sel > 0.012:
        return False
    # Absolute edge floors (train softer: EOD-flat vs BTC bull is hard)
    if nest["train"]["total_return"] < 0.08:
        return False
    if nest["mid"]["total_return"] < 0.10:
        return False
    return True


def passes_holdout(nest: dict, min_tpy_hold: float = 60) -> bool:
    """One-shot after lock. Never used in search score."""
    if not dd15(nest, keys=("hold", "full")):
        return False
    if nest["hold"]["total_return"] <= 0:
        return False
    if nest["hold"]["sharpe_ratio"] < 1.0:
        return False
    # Hold must beat costed BTC (one-shot relative alpha)
    if nest["hold"]["total_return"] <= nest["btc_hold"]["total_return"]:
        return False
    if nest["hold"]["trades_per_year"] < min_tpy_hold:
        return False
    if nest["hold"]["total_return"] < 0.15:
        return False
    # collapse check vs mid (gate only — not a score term)
    if nest["hold"]["total_return"] < 0.25 * nest["mid"]["total_return"] and nest["hold"]["total_return"] < 0.15:
        return False
    return True


def selection_score(nest: dict) -> float:
    """HOLD IS ABSENT. Only train + mid (+ their DD cushion + mid vs BTC edge)."""
    mid_edge = nest["mid"]["total_return"] - nest["btc_mid"]["total_return"]
    return (
        nest["mid"]["total_return"] * 3.0
        + nest["train"]["total_return"] * 2.0
        + mid_edge * 2.0
        + nest["mid"]["sharpe_ratio"] * 0.5
        + nest["train"]["sharpe_ratio"] * 0.3
        + 8.0 * max(0.0, HARD + nest["train"]["max_drawdown"])
        + 8.0 * max(0.0, HARD + nest["mid"]["max_drawdown"])
        + 0.02 * nest["full"].get("n_names_traded", 0)
    )


def set_lock(p: P) -> None:
    global P_LOCK
    P_LOCK = p
