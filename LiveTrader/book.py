"""Rebalance calendar, share diffs, sleeves, and cool-down — no broker calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from ge1_strategy import BOOK_SYMBOLS, P, ProdParams


PAPER_PORTS = frozenset({7497, 4002})
LIVE_PORTS = frozenset({7496, 4001})
SLEEVE_A = frozenset({"TQQQ", "QQQ"})
SLEEVE_B = frozenset({"GLD", "TLT", "BIL"})
CURRENCY = frozenset({"CASH", "USD", "EUR", "GBP", "JPY"})


def session_rebalance_flags(dates: list[date], i: int) -> tuple[bool, bool]:
    """Live week-end / month-end. Does not force the series tip True."""
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
    return week_end, month_end


def active_sleeve_symbols(*, week_end: bool, month_end: bool, seed: bool) -> frozenset[str]:
    """Locked book: weekly A, month-end B, first seed both."""
    if seed:
        return SLEEVE_A | SLEEVE_B
    allow: set[str] = set()
    if week_end:
        allow |= set(SLEEVE_A)
    if month_end:
        allow |= set(SLEEVE_B)
    return frozenset(allow)


def sleeve_for(symbol: str) -> str:
    su = str(symbol).upper()
    if su in SLEEVE_A:
        return "A"
    if su in SLEEVE_B:
        return "B"
    return "other"


@dataclass
class OrderIntent:
    symbol: str
    side: str
    qty: float
    target_weight: float
    mark: float
    notional: float
    order_type: str = "MKT"
    sleeve: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "target_weight": self.target_weight,
            "mark": self.mark,
            "notional": self.notional,
            "order_type": self.order_type,
            "sleeve": self.sleeve,
        }


def is_whole_share(qty: float) -> bool:
    q = float(qty)
    if q <= 1e-9:
        return False
    return abs(q - round(q)) <= 1e-6


def needs_fractional(orders: list[OrderIntent]) -> list[str]:
    return [o.symbol for o in orders if not is_whole_share(o.qty)]


def assign_order_types(
    orders: list[OrderIntent],
    *,
    whole_share_type: str = "MOC",
    fractional_type: str = "MKT",
) -> list[OrderIntent]:
    """MOC only for whole shares — IBKR rejects fractional MOC."""
    whole = str(whole_share_type or "MOC").upper()
    frac = str(fractional_type or "MKT").upper()
    out: list[OrderIntent] = []
    for o in orders:
        o.order_type = whole if is_whole_share(o.qty) else frac
        o.sleeve = o.sleeve or sleeve_for(o.symbol)
        out.append(o)
    return out


def orders_to_targets(
    equity: float,
    positions: dict[str, float],
    marks: dict[str, float],
    targets: dict[str, float],
    *,
    min_notional: float = 1.0,
    allow_symbols: frozenset[str] | None = None,
) -> list[OrderIntent]:
    """Share deltas to hit target weights. Sells first. Skips CASH."""
    eq = max(float(equity), 1e-9)
    want: dict[str, float] = {}
    for sym, w in targets.items():
        su = str(sym).upper()
        if su in ("CASH",) or float(w) <= 0:
            continue
        if allow_symbols is not None and su not in allow_symbols:
            continue
        px = marks.get(su)
        if px is None or px <= 0:
            continue
        want[su] = (float(w) * eq) / px
    have = {str(k).upper(): float(v) for k, v in positions.items() if abs(float(v)) > 1e-12}
    if allow_symbols is not None:
        have = {k: v for k, v in have.items() if k in allow_symbols}
    orders: list[OrderIntent] = []
    for sym in sorted(set(have) | set(want)):
        px = float(marks.get(sym) or 0.0)
        if px <= 0:
            continue
        delta = want.get(sym, 0.0) - have.get(sym, 0.0)
        notional = abs(delta) * px
        if notional < float(min_notional):
            continue
        side = "buy" if delta > 0 else "sell"
        orders.append(
            OrderIntent(
                symbol=sym,
                side=side,
                qty=abs(delta),
                target_weight=float(targets.get(sym, 0.0)),
                mark=px,
                notional=notional,
                sleeve=sleeve_for(sym),
            )
        )
    orders.sort(key=lambda o: (0 if o.side == "sell" else 1, o.symbol))
    return orders


def flatten_intents(
    positions: dict[str, float],
    marks: dict[str, float],
    *,
    include_foreign: bool = False,
    min_notional: float = 1.0,
) -> list[OrderIntent]:
    allowed = {s.upper() for s in BOOK_SYMBOLS}
    orders: list[OrderIntent] = []
    for sym, qty in positions.items():
        su = str(sym).upper()
        q = float(qty)
        if q <= 1e-12 or su in CURRENCY:
            continue
        if (not include_foreign) and su not in allowed:
            continue
        px = float(marks.get(su) or 0.0)
        if px <= 0:
            continue
        notional = q * px
        if notional < float(min_notional):
            continue
        orders.append(
            OrderIntent(
                symbol=su,
                side="sell",
                qty=q,
                target_weight=0.0,
                mark=px,
                notional=notional,
                sleeve="flatten",
            )
        )
    orders.sort(key=lambda o: o.symbol)
    return orders


def clamp_buys_to_cash(
    orders: list[OrderIntent],
    cash: float,
    *,
    min_notional: float = 1.0,
) -> list[OrderIntent]:
    """Apply sells first, then shrink/drop buys that exceed spendable cash."""
    avail = max(0.0, float(cash))
    out: list[OrderIntent] = []
    for o in orders:
        if o.side == "sell":
            avail += float(o.notional)
            out.append(o)
            continue
        need = float(o.notional)
        if need <= avail + 1e-9:
            avail -= need
            out.append(o)
            continue
        if o.mark <= 0:
            continue
        qty = avail / float(o.mark)
        notional = qty * float(o.mark)
        if notional < float(min_notional) or qty <= 0:
            continue
        o.qty = qty
        o.notional = notional
        avail = 0.0
        out.append(o)
    return out


def merge_marks(yahoo: dict[str, float], ib_last: dict[str, float] | None) -> dict[str, float]:
    out = {str(k).upper(): float(v) for k, v in yahoo.items() if float(v) > 0}
    for k, v in (ib_last or {}).items():
        if v is None:
            continue
        px = float(v)
        if px > 0:
            out[str(k).upper()] = px
    return out


def spendable_cash(cash: float, buying_power: float) -> float:
    c = max(0.0, float(cash or 0.0))
    bp = max(0.0, float(buying_power or 0.0))
    if c > 0 and bp > 0:
        return min(c, bp)
    return c or bp


def update_cool_state(
    peak_equity: float,
    equity: float,
    flat_a: bool,
    cool_remaining: int,
    p: ProdParams = P,
) -> tuple[float, bool, int]:
    peak = float(peak_equity or equity)
    if equity > peak:
        peak = equity
    flat = bool(flat_a)
    cool = int(cool_remaining or 0)
    if (not flat) and peak > 0 and (equity / peak - 1.0) <= -float(p.es):
        return peak, True, int(p.cool)
    if flat:
        if cool > 0:
            return peak, True, cool - 1
        return float(equity), False, 0
    return peak, False, cool


def park_sleeve_a(targets: dict[str, float], dual_asset: str) -> dict[str, float]:
    dual = {str(dual_asset).upper()}
    parked: dict[str, float] = {}
    for k, v in targets.items():
        ku = str(k).upper()
        if ku in dual or ku == "CASH":
            parked[ku] = parked.get(ku, 0.0) + float(v)
        else:
            parked["CASH"] = parked.get("CASH", 0.0) + float(v)
    s = sum(parked.values())
    if s <= 1e-12:
        return {"CASH": 1.0}
    return {k: v / s for k, v in parked.items()}


def foreign_symbols(positions: dict[str, float]) -> list[str]:
    allowed = {s.upper() for s in BOOK_SYMBOLS}
    out: list[str] = []
    for k, v in positions.items():
        ku = str(k).upper()
        if ku in CURRENCY:
            continue
        if abs(float(v)) > 1e-8 and ku not in allowed:
            out.append(ku)
    return sorted(out)


def journal_key(account: str, asof: str, symbol: str) -> str:
    return f"{account}|{asof}|{str(symbol).upper()}"


def weight_drift(
    equity: float,
    positions: dict[str, float],
    marks: dict[str, float],
    targets: dict[str, float],
    *,
    tol_weight: float = 0.02,
    tol_usd: float = 5.0,
) -> list[dict[str, Any]]:
    """Target vs held weight gaps larger than both tolerances."""
    eq = max(float(equity), 1e-9)
    diffs: list[dict[str, Any]] = []
    keys = {str(k).upper() for k in list(targets) + list(positions) if str(k).upper() != "CASH"}
    for sym in sorted(keys):
        px = float(marks.get(sym) or 0.0)
        have_qty = float(positions.get(sym) or 0.0)
        have_w = (have_qty * px) / eq if px > 0 else 0.0
        want_w = float(targets.get(sym) or 0.0)
        gap_w = have_w - want_w
        gap_usd = gap_w * eq
        if abs(gap_w) > float(tol_weight) and abs(gap_usd) > float(tol_usd):
            diffs.append(
                {
                    "symbol": sym,
                    "have_weight": have_w,
                    "want_weight": want_w,
                    "gap_usd": gap_usd,
                }
            )
    return diffs


@dataclass
class CoolState:
    peak_equity: float = 0.0
    flat_a: bool = False
    cool_remaining: int = 0
    last_trade_date: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
