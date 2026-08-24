"""Read-only adapter for the locked ``CoreEquity/`` pack.

Adds the pack directory to ``sys.path`` and imports pack modules as-is — never
edits pack files. Paper / forward / LiveTrader feed a live Yahoo daily panel
into the pack's ``latest_targets`` / ``paper_plan``; frozen ``bars.db`` is only
the fallback when no panel is provided.

Pack locals (``real_alpha_env``, ``core_equity_env``) are loaded under unique
module names so a same-process GeneralEquity1 import cannot shadow them.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from rlbot.run_artifacts import PROJECT_ROOT

PACK_DIR = PROJECT_ROOT / "CoreEquity"
STRATEGY_ID = "core_equity"
PAPER_RUN_ID = "CORE_EQUITY"
DEFAULT_INITIAL_CASH = 100_000.0
_STRATEGY_MOD_NAME = "_pack_core_equity_strategy"
_ENV_MOD_NAME = "_pack_core_equity_core_equity_env"
_REAL_MOD_NAME = "_pack_core_equity_real_alpha_env"
_PACK_ALIASES = ("real_alpha_env", "core_equity_env")


def _ensure_pack_on_path() -> Path:
    pack = PACK_DIR.resolve()
    if not pack.is_dir():
        raise FileNotFoundError(f"CoreEquity pack missing at {pack}")
    root = str(pack)
    if root not in sys.path:
        sys.path.insert(0, root)
    return pack


def _load_module(mod_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_strategy() -> Any:
    """Import ``CoreEquity/strategy.py`` without mutating the pack or GE1 modules."""
    if _STRATEGY_MOD_NAME in sys.modules:
        return sys.modules[_STRATEGY_MOD_NAME]
    pack = _ensure_pack_on_path()
    saved = {name: sys.modules[name] for name in _PACK_ALIASES if name in sys.modules}
    try:
        real = _load_module(_REAL_MOD_NAME, pack / "real_alpha_env.py")
        sys.modules["real_alpha_env"] = real
        env = _load_module(_ENV_MOD_NAME, pack / "core_equity_env.py")
        sys.modules["core_equity_env"] = env
        return _load_module(_STRATEGY_MOD_NAME, pack / "strategy.py")
    finally:
        for name in _PACK_ALIASES:
            prev = saved.get(name)
            if prev is not None:
                sys.modules[name] = prev
            else:
                sys.modules.pop(name, None)


def locked_params() -> Any:
    return load_strategy().P


def panel_symbols() -> tuple[str, ...]:
    """Yahoo names the locked book needs (SPY for the calendar + sleeve ETFs)."""
    p = locked_params()
    out: list[str] = []
    for raw in ("SPY", "QQQ", "BIL", "GLD", p.dual_b, p.eq_sym):
        su = str(raw).strip().upper()
        if su and su not in out:
            out.append(su)
    return tuple(out)


def book_symbols() -> tuple[str, ...]:
    """Tradeable names at IBKR / paper (no SPY, no 2x/3x ETFs)."""
    return tuple(s for s in panel_symbols() if s != "SPY")


def _need_symbols(p: Any) -> list[str]:
    need = ["SPY", "QQQ", "BIL", "GLD", str(p.dual_b).upper(), str(p.eq_sym).upper()]
    out: list[str] = []
    for s in need:
        if s not in out:
            out.append(s)
    return out


def _ohlc_tuple(ohlc: dict[str, tuple[np.ndarray, ...]], sym: str) -> tuple[np.ndarray, ...]:
    if sym not in ohlc:
        raise ValueError(f"live panel missing {sym} OHLC")
    return tuple(np.asarray(x, dtype=np.float64) for x in ohlc[sym])


def paper_plan(
    *,
    aum: float = DEFAULT_INITIAL_CASH,
    dates: list[date] | None = None,
    closes: dict[str, np.ndarray] | None = None,
    ohlc: dict[str, tuple[np.ndarray, ...]] | None = None,
) -> dict[str, Any]:
    """Pack ``paper_plan`` on a live Yahoo panel when provided, else frozen bars.db."""
    ce = load_strategy()
    p = ce.P
    hot = str(p.eq_sym).upper()
    need = _need_symbols(p)
    if dates is None or closes is None or ohlc is None:
        dates, px = ce.load_panel(need)
        hot_ohlc = ce.load_ohlc(hot, dates)
        qqq_ohlc = ce.load_ohlc("QQQ", dates)
        source = "bars.db"
    else:
        px = {}
        for sym in need:
            if sym not in closes:
                raise ValueError(f"live panel missing {sym}")
            px[sym] = np.asarray(closes[sym], dtype=np.float64)
        hot_ohlc = _ohlc_tuple(ohlc, hot)
        qqq_ohlc = _ohlc_tuple(ohlc, "QQQ")
        source = "yahoo"
    plan = ce.paper_plan(dates, px, hot_ohlc, qqq_ohlc, p, aum=float(aum))
    plan["data_source"] = source
    return plan


def latest_targets(
    dates: list[date],
    px: dict[str, np.ndarray],
    hot_ohlc: tuple[np.ndarray, ...],
    qqq_ohlc: tuple[np.ndarray, ...],
    p: Any | None = None,
) -> dict[str, Any]:
    """Pack ``latest_targets`` on the caller-supplied panel (Yahoo for live)."""
    ce = load_strategy()
    params = p if p is not None else ce.P
    return ce.latest_targets(dates, px, hot_ohlc, qqq_ohlc, params)


def weights_from_targets(targets: dict[str, Any], p: Any | None = None) -> dict[str, float]:
    """Lag-1 book from pack targets. Do not renormalize cash-financed QQQ > 1.0 sleeve.

    Residual parks in BIL (pack book). When sleeve-A QQQ is cash-financed above
    100% of AUM together with dual, there is no residual and gross can exceed 1.
    """
    params = p if p is not None else locked_params()
    hot = str(params.eq_sym).upper()
    w_hot = float(targets.get(f"portfolio_{hot}") or 0.0)
    w_qqq = float(targets.get("portfolio_QQQ") or 0.0)
    dual_sym = str(targets.get("dual_asset") or "BIL").upper()
    w_dual = float(targets.get("portfolio_dual") or 0.0) if dual_sym != "BIL" else 0.0
    tw: dict[str, float] = {}
    if hot == "QQQ":
        if w_hot > 1e-12:
            tw["QQQ"] = w_hot
    else:
        if w_hot > 1e-12:
            tw[hot] = w_hot
        if w_qqq > 1e-12:
            tw["QQQ"] = w_qqq
    if dual_sym != "BIL" and w_dual > 1e-12:
        tw[dual_sym] = w_dual
    invested = float(sum(tw.values()))
    residual = 1.0 - invested
    if residual > 1e-12:
        tw["BIL"] = residual
    elif abs(residual) <= 1e-12:
        tw["CASH"] = 0.0
    return {k: float(v) for k, v in tw.items() if abs(v) > 1e-12 or k == "CASH"}


def latest_portfolio_weights(
    *,
    aum: float = DEFAULT_INITIAL_CASH,
    dates: list[date] | None = None,
    closes: dict[str, np.ndarray] | None = None,
    ohlc: dict[str, tuple[np.ndarray, ...]] | None = None,
) -> dict[str, float]:
    plan = paper_plan(aum=aum, dates=dates, closes=closes, ohlc=ohlc)
    targets = plan.get("targets") or {}
    if not targets:
        return {"CASH": 1.0}
    return weights_from_targets(targets)


def to_yahoo_symbol(sym: str) -> str:
    s = str(sym).strip().upper()
    return s.replace(".", "-") if s else s


def _pack_env() -> Any:
    load_strategy()
    return sys.modules[_ENV_MOD_NAME]


def _pack_real() -> Any:
    load_strategy()
    return sys.modules[_REAL_MOD_NAME]


@dataclass(frozen=True)
class SleeveAState:
    """Sleeve-A overlay at the panel tip, matching ``equity_cc_with_turnover``.

    Emergency stop / cool-down run on the sleeve-A path (not account NLV).
    Re-entry after cool requires the trend gate; the next long weight waits
    for a weekly rebalance, but flatten is immediate.
    """

    equity: float
    peak: float
    flat: bool
    cool_remaining: int
    held_w: float
    ok_now: bool
    next_cc_weight: float

    @property
    def was_long(self) -> bool:
        return float(self.held_w) > 1e-8


def cc_sleeve_weight(
    *,
    ok: bool,
    vol: float,
    atr: float,
    vt: float,
    atr_max: float,
    w_cap: float,
    was_long: bool,
    atr_hyst: float = 0.0,
    vol_spike: float = 9.0,
    vol_avg: float = float("nan"),
) -> float:
    """Vol-target weight with research ATR hysteresis (hyst only if already long)."""
    atr_exit = atr_max * (1.0 + atr_hyst) if atr_hyst > 0.0 else atr_max
    atr_lim = atr_exit if was_long else atr_max
    spike = False
    if (
        vol_spike < 8.0
        and np.isfinite(vol)
        and np.isfinite(vol_avg)
        and vol_avg > 1e-8
        and vol > vol_spike * vol_avg
    ):
        spike = True
    if (
        (not spike)
        and ok
        and (atr_max >= 9.0 or (not np.isfinite(atr)) or atr <= atr_lim)
    ):
        if np.isfinite(vol) and vol > 1e-8:
            cap = w_cap if w_cap > 0.0 else 1.0
            return float(min(cap, max(0.0, vt / vol)))
    return 0.0


def sleeve_a_live_state(
    dates: list[date],
    px: dict[str, np.ndarray],
    hot_ohlc: tuple[np.ndarray, ...],
    p: Any | None = None,
) -> SleeveAState:
    """Replay sleeve A from the pack start date through the panel tip."""
    env = _pack_env()
    real = _pack_real()
    params = p if p is not None else locked_params()
    n = len(dates)
    if n < 2:
        return SleeveAState(
            equity=1.0,
            peak=1.0,
            flat=False,
            cool_remaining=0,
            held_w=0.0,
            ok_now=False,
            next_cc_weight=0.0,
        )
    bt_start = getattr(env, "BT_START", date(2010, 1, 4))
    i0 = next((i for i, d in enumerate(dates) if d >= bt_start), 0)
    i1 = n - 1
    ok = env.trend_ok(px, params)
    _o, h, l, c = hot_ohlc
    vol = real._roll_vol(real.rets(c), params.vol_lb)
    atr = real._atrp(h, l, c)
    tr = real.rets(c)
    bil = real.rets(px["BIL"])
    mask = env.week_end_mask(dates) if params.rebal == "wk" else real.month_end_mask(dates)
    q_cap = float(min(env.Q_CAP_MAX, max(0.0, getattr(params, "q_cap", 1.0))))
    hot_is_qqq = str(params.eq_sym).upper() == "QQQ"
    w_cap = q_cap if hot_is_qqq else float(env.W_CAP)
    _daily, _turn, w_path = env.equity_cc_with_turnover(
        ok,
        vol,
        atr,
        tr,
        bil,
        i0,
        i1,
        params.vt,
        params.atr_max,
        params.es,
        params.cool,
        mask,
        float(params.vol_spike),
        float(params.atr_hyst),
        w_cap,
    )
    eq = 1.0
    peak = 1.0
    flat = False
    cd = 0
    es = float(params.es)
    cool = int(params.cool)
    for i in range(i0 + 1, i1 + 1):
        g = float(_daily[i])
        if not np.isfinite(g):
            g = 0.0
        eq *= 1.0 + g
        if eq > peak:
            peak = eq
        if (not flat) and peak > 0 and (eq / peak - 1.0) <= -es:
            flat = True
            cd = cool
        elif flat:
            if cd > 0:
                cd -= 1
            elif bool(ok[i]):
                flat = False
                peak = eq
    held_w = float(w_path[i1]) if np.isfinite(w_path[i1]) else 0.0
    j = i1
    next_w = 0.0
    if not flat:
        next_w = cc_sleeve_weight(
            ok=bool(ok[j]),
            vol=float(vol[j]),
            atr=float(atr[j]),
            vt=float(params.vt),
            atr_max=float(params.atr_max),
            w_cap=w_cap,
            was_long=held_w > 1e-8,
            atr_hyst=float(params.atr_hyst),
            vol_spike=float(params.vol_spike),
        )
    return SleeveAState(
        equity=float(eq),
        peak=float(peak),
        flat=bool(flat),
        cool_remaining=int(cd),
        held_w=held_w,
        ok_now=bool(ok[i1]),
        next_cc_weight=float(next_w),
    )


def apply_sleeve_a_to_targets(
    targets: dict[str, Any],
    sleeve: SleeveAState,
    p: Any | None = None,
) -> dict[str, Any]:
    """Replace sleeve-A CC weights with the run_prod overlay (ATR hyst + ES)."""
    params = p if p is not None else locked_params()
    env = _pack_env()
    out = dict(targets)
    hot = str(params.eq_sym).upper()
    q_cap = float(min(env.Q_CAP_MAX, max(0.0, getattr(params, "q_cap", 1.0))))
    w_h = 0.0 if sleeve.flat else float(sleeve.next_cc_weight)
    hot_is_qqq = hot == "QQQ"
    cap = q_cap if hot_is_qqq else float(env.W_CAP)
    w_h = float(min(cap, max(0.0, w_h)))
    w_hot_share = 1.0 if hot_is_qqq else float(min(1.0, max(0.0, params.w_hot)))
    out["hot_cc_weight"] = w_h
    if hot_is_qqq:
        out["QQQ_cc_weight"] = w_h
        port = float(params.w_a) * w_h
        out["portfolio_QQQ"] = port
        out[f"portfolio_{hot}"] = port
        out["sleeve_A_hot_share"] = 1.0
        out["sleeve_A_QQQ_share"] = 0.0
    else:
        out["QQQ_cc_weight"] = float(out.get("QQQ_cc_weight") or 0.0)
        out[f"portfolio_{hot}"] = float(params.w_a) * w_hot_share * w_h
        # Calm QQQ leg keeps pack latest_targets (same ATR bug class); locked book is QQQ-only.
    out["sleeve_a_flat"] = bool(sleeve.flat)
    out["sleeve_a_cool_remaining"] = int(sleeve.cool_remaining)
    out["sleeve_a_equity"] = float(sleeve.equity)
    return out


def live_session_rebalance_flags(
    dates: list[date],
    i: int,
    *,
    calendar_today: date | None = None,
) -> tuple[bool, bool]:
    """Week-end / month-end without forcing the series tip.

    If Yahoo has not published today's bar yet (same ISO week, ≤3 calendar days),
    evaluate Friday / month-end on ``calendar_today`` so a 15:45 MOC is not skipped.
    """
    n = len(dates)
    if i < 0 or i >= n:
        return False, False
    d = dates[i]
    if i + 1 < n:
        week_end = dates[i].isocalendar()[1] != dates[i + 1].isocalendar()[1]
        month_end = dates[i].month != dates[i + 1].month
        return bool(week_end), bool(month_end)
    week_end = d.weekday() == 4
    nxt = d + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    month_end = nxt.month != d.month
    if calendar_today is None or calendar_today <= d:
        return bool(week_end), bool(month_end)
    lag = (calendar_today - d).days
    if lag <= 0 or lag > 3 or calendar_today.weekday() >= 5:
        return bool(week_end), bool(month_end)
    same_week = (
        d.isocalendar()[0] == calendar_today.isocalendar()[0]
        and d.isocalendar()[1] == calendar_today.isocalendar()[1]
    )
    if same_week and calendar_today.weekday() == 4:
        week_end = True
    nxt2 = calendar_today + timedelta(days=1)
    while nxt2.weekday() >= 5:
        nxt2 += timedelta(days=1)
    if nxt2.month != calendar_today.month:
        month_end = True
    return bool(week_end), bool(month_end)
