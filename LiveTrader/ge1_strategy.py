"""Locked GeneralEquity1 (prod_return_alpha_v3) signal copy.

Params locked 2026-08-01 in ``GeneralEquity1/strategy.py``. Indicators match
``GeneralEquity1/real_alpha_env.py`` without numba. Do not retune here.

Live path uses a Yahoo (or IB) daily panel, never the frozen pack ``bars.db``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np

BOOK_SYMBOLS = ("TQQQ", "QQQ", "GLD", "TLT", "BIL")
PANEL_SYMBOLS = ("SPY", "QQQ", "TQQQ", "BIL", "GLD", "TLT")
STRATEGY_ID = "prod_return_alpha_v3"


@dataclass(frozen=True)
class ProdParams:
    ma: int = 151
    vt: float = 0.27
    atr_max: float = 0.10
    es: float = 0.278
    cool: int = 15
    w_a: float = 0.58
    dual_lb: int = 231
    dual_vt: float = 0.14
    dual_b: str = "TLT"
    rebal: str = "wk"
    vol_lb: int = 21
    w_tqqq: float = 0.78
    q_vt: float = 0.08
    q_cap: float = 1.5
    q_atr_max: float = 0.10
    vol_spike: float = 9.0
    atr_hyst: float = 0.05


# Locked 2026-08-01 (train+mid search; holdout one-shot)
P = ProdParams()


def rets(arr: np.ndarray) -> np.ndarray:
    r = np.zeros(len(arr), dtype=np.float64)
    for i in range(1, len(arr)):
        if arr[i - 1] > 0 and np.isfinite(arr[i]) and np.isfinite(arr[i - 1]):
            r[i] = arr[i] / arr[i - 1] - 1.0
    return r


def _roll_vol(rr: np.ndarray, lb: int) -> np.ndarray:
    n = len(rr)
    v = np.full(n, np.nan)
    for i in range(lb - 1, n):
        m = 0.0
        ok = True
        for k in range(i - lb + 1, i + 1):
            if not np.isfinite(rr[k]):
                ok = False
                break
            m += rr[k]
        if not ok:
            continue
        m /= lb
        var = 0.0
        for k in range(i - lb + 1, i + 1):
            d = rr[k] - m
            var += d * d
        v[i] = math.sqrt(var / (lb - 1)) * math.sqrt(252.0)
    return v


def _sma_ok(q: np.ndarray, ma: int) -> np.ndarray:
    n = len(q)
    ok = np.zeros(n, dtype=np.bool_)
    for i in range(ma - 1, n):
        m = 0.0
        good = True
        for k in range(i - ma + 1, i + 1):
            if not np.isfinite(q[k]):
                good = False
                break
            m += q[k]
        if good and q[i] >= m / ma:
            ok[i] = True
    return ok


def _atrp(h: np.ndarray, l: np.ndarray, c: np.ndarray, lb: int = 14) -> np.ndarray:
    n = len(c)
    out = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        if np.isfinite(h[i]) and np.isfinite(l[i]) and np.isfinite(c[i - 1]):
            tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    for i in range(lb - 1, n):
        if c[i] > 0:
            m = 0.0
            for k in range(i - lb + 1, i + 1):
                m += tr[k]
            out[i] = (m / lb) / c[i]
    return out


def _sleeve_weight(ok, vol, atr, vt, atr_max, w_cap, i, atr_hyst=0.0) -> float:
    atr_gate = atr_max * (1.0 + atr_hyst) if atr_hyst > 0.0 else atr_max
    if ok[i] and (atr_gate >= 9 or not np.isfinite(atr[i]) or atr[i] <= atr_gate):
        if np.isfinite(vol[i]) and vol[i] > 1e-8:
            return float(min(w_cap, max(0.0, vt / vol[i])))
    return 0.0


def latest_targets(
    dates: list[date],
    px: dict[str, np.ndarray],
    tqqq_ohlc: tuple[np.ndarray, ...],
    qqq_ohlc: tuple[np.ndarray, ...],
    p: ProdParams = P,
) -> dict[str, Any]:
    i = len(dates) - 1
    _o, h, l, c = tqqq_ohlc
    vol = _roll_vol(rets(c), p.vol_lb)
    atr = _atrp(h, l, c)
    ok = _sma_ok(px["QQQ"], p.ma)
    w_t = _sleeve_weight(ok, vol, atr, p.vt, p.atr_max, 1.0, i, p.atr_hyst)

    qo, qh, ql, qc = qqq_ohlc
    qvol = _roll_vol(rets(qc), p.vol_lb)
    qatr = _atrp(qh, ql, qc)
    w_q = _sleeve_weight(ok, qvol, qatr, p.q_vt, p.q_atr_max, p.q_cap, i, p.atr_hyst)

    gld, alt = px["GLD"], px[p.dual_b]
    v1 = _roll_vol(rets(gld), p.vol_lb)
    v2 = _roll_vol(rets(alt), p.vol_lb)
    dual_sym, dual_w = "BIL", 0.0
    if i >= p.dual_lb and gld[i] > 0 and gld[i - p.dual_lb] > 0 and alt[i] > 0 and alt[i - p.dual_lb] > 0:
        tr1 = gld[i] / gld[i - p.dual_lb] - 1.0
        tr2 = alt[i] / alt[i - p.dual_lb] - 1.0
        if tr1 > 0 or tr2 > 0:
            if tr1 >= tr2 and np.isfinite(v1[i]) and v1[i] > 1e-8:
                dual_sym, dual_w = "GLD", float(min(1.0, p.dual_vt / v1[i]))
            elif np.isfinite(v2[i]) and v2[i] > 1e-8:
                dual_sym, dual_w = p.dual_b, float(min(1.0, p.dual_vt / v2[i]))

    return {
        "asof": str(dates[i]),
        "rebalance_mode": "weekly close-to-close (TQQQ+QQQ hybrid) + month-end (dual)",
        "sleeve_A_capital": p.w_a,
        "sleeve_A_TQQQ_share": p.w_tqqq,
        "sleeve_A_QQQ_share": 1.0 - p.w_tqqq,
        "TQQQ_cc_weight": w_t,
        "QQQ_cc_weight": w_q,
        "portfolio_TQQQ": p.w_a * p.w_tqqq * w_t,
        "portfolio_QQQ": p.w_a * (1.0 - p.w_tqqq) * w_q,
        "sleeve_B_capital": 1.0 - p.w_a,
        "dual_asset": dual_sym,
        "dual_weight": dual_w,
        "portfolio_dual": (1.0 - p.w_a) * dual_w,
        "params": asdict(p),
    }


def portfolio_weights(targets: dict[str, Any], p: ProdParams = P) -> dict[str, float]:
    """Lag-1 book weights; BIL residual folded into CASH (matches pack adapter)."""
    w_tqqq = float(targets["portfolio_TQQQ"])
    w_qqq = float(targets["portfolio_QQQ"])
    dual_sym = str(targets["dual_asset"]).upper()
    w_dual = float(targets["portfolio_dual"]) if dual_sym != "BIL" else 0.0
    w_cash = max(0.0, 1.0 - w_tqqq - min(w_qqq, p.w_a * (1.0 - p.w_tqqq)) - w_dual)
    out: dict[str, float] = {"TQQQ": w_tqqq, "QQQ": w_qqq, "CASH": w_cash}
    if dual_sym != "BIL" and w_dual > 0:
        out[dual_sym] = w_dual
    out = {k: float(v) for k, v in out.items() if k == "CASH" or v > 1e-12}
    s = sum(out.values())
    if s <= 1e-12:
        return {"CASH": 1.0}
    if abs(s - 1.0) > 1e-6:
        return {k: v / s for k, v in out.items()}
    return out
