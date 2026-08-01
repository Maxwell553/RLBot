"""prod_return_alpha_v1 — TQQQ weekly CC + GLD/TLT dual momentum (locked).

Paper / forward identity for ``/ops/forward``. Knobs match ``1360pctAlgo/``
(nested train/mid/holdout pack). Do not retune here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any  # noqa: TC003 — used in np.savez payload

import numpy as np
import pandas as pd

from rlbot.run_artifacts import PROJECT_ROOT

STRATEGY_ID = "prod_return_alpha_v1"
PAPER_RUN_ID = "PROD_RETURN_ALPHA"
PACK_DIR = PROJECT_ROOT / "1360pctAlgo"

SYMBOLS = ("SPY", "QQQ", "TQQQ", "BIL", "GLD", "TLT")


@dataclass(frozen=True)
class ProdParams:
    ma: int = 151
    vt: float = 0.27
    atr_max: float = 0.20
    es: float = 0.278
    cool: int = 15
    w_a: float = 0.57
    dual_lb: int = 231
    dual_vt: float = 0.14
    dual_b: str = "TLT"
    rebal: str = "wk"
    vol_lb: int = 21


P = ProdParams()


def to_yahoo_symbol(sym: str) -> str:
    """Map broker dots to Yahoo (BRK.B → BRK-B). ETFs are identity."""
    s = str(sym).strip().upper()
    return s.replace(".", "-") if s else s


def _rets(arr: np.ndarray) -> np.ndarray:
    r = np.zeros(len(arr), dtype=np.float64)
    for i in range(1, len(arr)):
        if arr[i - 1] > 0 and np.isfinite(arr[i]) and np.isfinite(arr[i - 1]):
            r[i] = arr[i] / arr[i - 1] - 1.0
    return r


def _roll_vol(rr: np.ndarray, lb: int) -> np.ndarray:
    n = len(rr)
    v = np.full(n, np.nan)
    for i in range(lb - 1, n):
        w = rr[i - lb + 1 : i + 1]
        if np.all(np.isfinite(w)):
            sd = float(np.std(w))
            if sd > 0:
                v[i] = sd * np.sqrt(252.0)
    return v


def _atrp(h: np.ndarray, l: np.ndarray, c: np.ndarray, lb: int = 14) -> np.ndarray:
    n = len(c)
    tr = np.full(n, np.nan)
    for i in range(1, n):
        if not (np.isfinite(h[i]) and np.isfinite(l[i]) and np.isfinite(c[i - 1])):
            continue
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    out = np.full(n, np.nan)
    for i in range(lb, n):
        w = tr[i - lb + 1 : i + 1]
        if np.all(np.isfinite(w)) and c[i] > 0:
            out[i] = float(np.mean(w) / c[i])
    return out


def _sma_ok(px: np.ndarray, ma: int) -> np.ndarray:
    n = len(px)
    ok = np.zeros(n, dtype=np.bool_)
    for i in range(ma - 1, n):
        w = px[i - ma + 1 : i + 1]
        if np.all(np.isfinite(w)) and w[-1] > 0:
            ok[i] = bool(w[-1] > float(np.mean(w)))
    return ok


def week_end_mask(dates: list[date]) -> np.ndarray:
    n = len(dates)
    m = np.zeros(n, dtype=np.bool_)
    for i in range(1, n):
        if dates[i].isocalendar()[1] != dates[i - 1].isocalendar()[1]:
            m[i - 1] = True
    if n:
        m[-1] = True
    return m


def month_end_mask(dates: list[date]) -> np.ndarray:
    n = len(dates)
    m = np.zeros(n, dtype=np.bool_)
    for i in range(n - 1):
        if dates[i + 1].month != dates[i].month:
            m[i] = True
    if n:
        m[-1] = True
    return m


def fetch_daily_ohlc(
    symbols: list[str] = list(SYMBOLS),
    *,
    start: date | None = None,
    end: date | None = None,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
) -> tuple[list[date], dict[str, np.ndarray], dict[str, tuple[np.ndarray, ...]]]:
    """Daily adjusted OHLC via yfinance. Returns (dates, close_panel, ohlc_by_sym)."""
    import yfinance as yf

    start_d = start or date(2009, 1, 1)
    end_d = end or date.today()
    cache_dir = cache_dir or (PROJECT_ROOT / "execution" / "paper_prod_return_alpha")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "daily_ohlc.npz"

    want = [s.upper() for s in symbols]
    cached_dates: list[date] | None = None
    cached_closes: dict[str, np.ndarray] = {}
    cached_ohlc: dict[str, tuple[np.ndarray, ...]] = {}
    if cache_path.is_file() and not force_refresh:
        try:
            blob = np.load(cache_path, allow_pickle=True)
            cached_dates = [date.fromisoformat(str(x)[:10]) for x in blob["dates"].tolist()]
            for sym in want:
                if f"{sym}_close" in blob.files:
                    o = np.asarray(blob[f"{sym}_open"], dtype=np.float64)
                    h = np.asarray(blob[f"{sym}_high"], dtype=np.float64)
                    l = np.asarray(blob[f"{sym}_low"], dtype=np.float64)
                    c = np.asarray(blob[f"{sym}_close"], dtype=np.float64)
                    cached_closes[sym] = c
                    cached_ohlc[sym] = (o, h, l, c)
        except Exception:  # noqa: BLE001
            cached_dates = None

    tip_stale = True
    if cached_dates:
        tip_stale = cached_dates[-1] < (end_d - timedelta(days=3))

    if not cached_dates or "SPY" not in cached_closes or tip_stale or force_refresh:
        frames: dict[str, pd.DataFrame] = {}
        for sym in want:
            raw = yf.download(
                to_yahoo_symbol(sym),
                start=str(start_d),
                end=str(end_d + timedelta(days=1)),
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if raw is None or raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            frames[sym] = raw[["Open", "High", "Low", "Close"]].copy()
        if "SPY" not in frames:
            raise RuntimeError("yfinance returned no SPY bars for prod_return_alpha")
        idx = frames["SPY"].dropna(subset=["Close"]).index
        dates = [pd.Timestamp(t).date() for t in idx]
        closes = {}
        ohlc = {}
        payload: dict[str, Any] = {"dates": np.asarray([str(d) for d in dates])}
        for sym in want:
            if sym not in frames:
                continue
            block = frames[sym].reindex(idx).ffill()
            o = block["Open"].to_numpy(dtype=np.float64)
            h = block["High"].to_numpy(dtype=np.float64)
            l = block["Low"].to_numpy(dtype=np.float64)
            c = block["Close"].to_numpy(dtype=np.float64)
            closes[sym] = c
            ohlc[sym] = (o, h, l, c)
            payload[f"{sym}_open"] = o
            payload[f"{sym}_high"] = h
            payload[f"{sym}_low"] = l
            payload[f"{sym}_close"] = c
        np.savez_compressed(cache_path, **payload)
        return dates, closes, ohlc

    assert cached_dates is not None
    return cached_dates, cached_closes, cached_ohlc


def latest_sleeve_targets(
    dates: list[date],
    closes: dict[str, np.ndarray],
    tqqq_ohlc: tuple[np.ndarray, ...],
    *,
    p: ProdParams = P,
    flat: bool = False,
) -> dict[str, Any]:
    """Lag-1 style sleeve targets at the last bar (pack ``latest_targets``)."""
    _o, h, l, c = tqqq_ohlc
    i = len(dates) - 1
    vol = _roll_vol(_rets(c), p.vol_lb)
    atr = _atrp(h, l, c)
    ok = _sma_ok(closes["QQQ"], p.ma)
    w = 0.0
    if (not flat) and ok[i] and (p.atr_max >= 9 or not np.isfinite(atr[i]) or atr[i] <= p.atr_max):
        if np.isfinite(vol[i]) and vol[i] > 1e-8:
            w = float(min(1.0, p.vt / vol[i]))
    gld, alt = closes["GLD"], closes[p.dual_b]
    v1 = _roll_vol(_rets(gld), p.vol_lb)
    v2 = _roll_vol(_rets(alt), p.vol_lb)
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
        "rebalance_mode": "weekly close-to-close (TQQQ) + month-end (dual)",
        "sleeve_A_capital": p.w_a,
        "TQQQ_cc_weight": w,
        "sleeve_B_capital": 1.0 - p.w_a,
        "dual_asset": dual_sym,
        "dual_weight": dual_w,
        "flat_a": bool(flat),
    }


def portfolio_weights_from_sleeves(
    *,
    tqqq_w: float,
    dual_asset: str,
    dual_w: float,
    p: ProdParams = P,
) -> dict[str, float]:
    """Map sleeve weights → long-only portfolio weights (cash residual in BIL)."""
    w_a = float(p.w_a)
    w_b = 1.0 - w_a
    tqqq_port = w_a * max(0.0, float(tqqq_w))
    dual_port = w_b * max(0.0, float(dual_w))
    out: dict[str, float] = {}
    if tqqq_port > 1e-12:
        out["TQQQ"] = tqqq_port
    dual_asset_u = str(dual_asset).upper()
    if dual_port > 1e-12 and dual_asset_u not in ("", "BIL", "CASH"):
        out[dual_asset_u] = out.get(dual_asset_u, 0.0) + dual_port
    bil = max(0.0, 1.0 - sum(out.values()))
    if bil > 1e-12:
        out["BIL"] = bil
    tot = sum(out.values())
    if tot <= 1e-12:
        return {"BIL": 1.0}
    return {k: float(v) / tot for k, v in out.items()}


def compute_target_weights(
    *,
    as_of: date | None = None,
    force_refresh: bool = False,
    flat_a: bool = False,
    p: ProdParams = P,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Fetch panel and return (portfolio_weights, sleeve_meta) as-of last bar ≤ as_of."""
    dates, closes, ohlc = fetch_daily_ohlc(force_refresh=force_refresh)
    if as_of is not None:
        keep = [i for i, d in enumerate(dates) if d <= as_of]
        if not keep:
            raise ValueError(f"no bars on or before {as_of}")
        last = keep[-1] + 1
        dates = dates[:last]
        closes = {k: v[:last] for k, v in closes.items()}
        ohlc = {k: tuple(a[:last] for a in v) for k, v in ohlc.items()}
    meta = latest_sleeve_targets(dates, closes, ohlc["TQQQ"], p=p, flat=flat_a)
    weights = portfolio_weights_from_sleeves(
        tqqq_w=float(meta["TQQQ_cc_weight"]),
        dual_asset=str(meta["dual_asset"]),
        dual_w=float(meta["dual_weight"]),
        p=p,
    )
    return weights, meta


def weights_with_cash(weights: dict[str, float]) -> dict[str, float]:
    """Ensure a CASH key (BIL counts as invested cash-like; residual → CASH)."""
    out = {str(k).upper(): float(v) for k, v in weights.items() if float(v) > 0}
    invested = sum(v for k, v in out.items() if k != "CASH")
    cash = max(0.0, 1.0 - invested)
    if cash > 1e-12:
        out["CASH"] = out.get("CASH", 0.0) + cash
    tot = sum(out.values())
    if tot <= 1e-12:
        return {"CASH": 1.0}
    return {k: v / tot for k, v in out.items()}
