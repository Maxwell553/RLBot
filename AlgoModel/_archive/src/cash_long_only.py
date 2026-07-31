"""
Cash-account long-only strategy (no margin, no shorts, no futures).

Audited design:
  - Candidates must be supplied point-in-time (no full-sample durable leak).
  - Formation uses only prices before cycle start.
  - Signals on day t execute on day t+execution_lag (default 1).
  - Cash never goes negative; no short sales.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Callable, Optional

import numpy as np
import pandas as pd

from statarb.backtest.costs import CostModel
from statarb.config_models import Bar, Fill, OrderIntent
from statarb.strategy.portable_alpha import (
    DUAL_CLASS_BAN,
    _bars_to_frames,
    _series_before,
    load_durable_candidates,
)

log = logging.getLogger("cash_long_only")

# asof date -> list of (a,b)
CandidateFn = Callable[[date], list[tuple[str, str]]]


@dataclass
class CashLongConfig:
    starting_equity: float = 100_000.0
    formation_days: int = 252
    trading_days: int = 63
    top_n: int = 8
    open_z: float = 1.5
    min_edge_cost_multiple: float = 2.0
    position_pct: float = 0.10
    min_spy_pct: float = 0.40
    max_concurrent_longs: int = 6
    max_sector_longs: int = 2
    min_adv_usd: float = 5_000_000.0
    max_participation_pct_adv: float = 0.02
    adv_window: int = 20
    ban_dual_class: bool = True
    commission_per_share: float = 0.005
    slippage_bps: float = 2.0
    bid_ask_spread_bps: float = 5.0
    beta_symbol: str = "SPY"
    cash_buffer: float = 100.0
    # Anti look-ahead / risk
    execution_lag_days: int = 1  # signal day t → trade day t+lag
    spy_ma_filter: int = 0  # 0=off; e.g. 200 = only open RV if SPY > MA


@dataclass
class _Book:
    a: str
    b: str
    sector: str
    ssd: float
    sigma: float
    pa0: float
    pb0: float
    long_sym: Optional[str] = None
    qty: float = 0.0
    pending_long: Optional[str] = None  # awaiting lag execution
    pending_flat: bool = False


@dataclass
class CashLongResult:
    fills: list[Fill] = field(default_factory=list)
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    cycles: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    daily: list[dict] = field(default_factory=list)


def load_pit_schedule(path) -> dict[date, list[tuple[str, str]]]:
    df = pd.read_csv(path)
    out: dict[date, list[tuple[str, str]]] = {}
    for asof, g in df.groupby("asof"):
        d = date.fromisoformat(str(asof)[:10])
        out[d] = [(str(r.symbol_a), str(r.symbol_b)) for r in g.itertuples()]
    return out


def make_pit_candidate_fn(schedule: dict[date, list[tuple[str, str]]]) -> CandidateFn:
    keys = sorted(schedule)

    def fn(asof: date) -> list[tuple[str, str]]:
        # latest schedule date <= asof
        avail = [k for k in keys if k <= asof]
        if not avail:
            return []
        return schedule[avail[-1]]

    return fn


def make_fixed_candidate_fn(pairs: list[tuple[str, str]]) -> CandidateFn:
    frozen = list(pairs)

    def fn(asof: date) -> list[tuple[str, str]]:
        return frozen

    return fn


class CashLongOnlyEngine:
    """Fully cash-funded: SPY core + long-only relative-value satellite."""

    def __init__(self, cfg: CashLongConfig) -> None:
        self.cfg = cfg
        self.costs = CostModel(
            commission_per_share=cfg.commission_per_share,
            slippage_bps=cfg.slippage_bps,
            bid_ask_spread_bps=cfg.bid_ask_spread_bps,
        )
        self.cash = float(cfg.starting_equity)
        self.positions: dict[str, float] = {}
        self.fills: list[Fill] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self.cycles: list[dict] = []
        self.daily: list[dict] = []
        self._books: list[_Book] = []
        self.equity = float(cfg.starting_equity)

    def run(
        self,
        candidate_fn: CandidateFn,
        bars_by_symbol: dict[str, list[Bar]],
        start: date,
        end: date,
        sector_of: dict[str, str] | None = None,
    ) -> CashLongResult:
        sector_of = sector_of or {}
        closes, volumes, by_sym_date = _bars_to_frames(bars_by_symbol)
        beta = self.cfg.beta_symbol
        if beta not in closes:
            return CashLongResult(summary={"error": f"missing {beta}"})

        days = sorted(d for d in by_sym_date[beta] if start <= d <= end)
        if not days:
            return CashLongResult(summary={"error": "no days"})

        m0 = self._marks(days[0], by_sym_date)
        self._buy(
            beta,
            max(self.cfg.starting_equity - self.cfg.cash_buffer, 0) / m0[beta],
            m0[beta],
            days[0],
            "seed_spy",
        )

        i = 0
        while i < len(days):
            j = min(i + self.cfg.trading_days - 1, len(days) - 1)
            c0, c1 = days[i], days[j]
            trade_days = days[i : j + 1]
            cands = candidate_fn(c0)
            selected = self._form(cands, closes, volumes, c0, sector_of)
            self._books = selected
            self.cycles.append(
                {
                    "trade_start": str(c0),
                    "trade_end": str(c1),
                    "n_candidates": len(cands),
                    "n_pairs": len(selected),
                    "pairs": [f"{p.a}-{p.b}" for p in selected],
                }
            )

            m_roll = self._marks(c0, by_sym_date)
            self._flatten_rv(m_roll, c0, reason="cycle_roll")
            self._rebalance_spy(m_roll, c0)

            for di, d in enumerate(trade_days):
                next_d = trade_days[di + 1] if di + 1 < len(trade_days) else None
                self._on_day(
                    d,
                    next_d,
                    by_sym_date,
                    closes,
                    is_period_end=(d == c1),
                )
            i = j + 1

        if days:
            m = self._marks(days[-1], by_sym_date)
            for sym, q in list(self.positions.items()):
                if q > 0 and sym in m:
                    self._sell(sym, q, m[sym], days[-1], "final_flatten")
            self.equity = self._mtm(m)

        return CashLongResult(
            fills=list(self.fills),
            equity_curve=list(self.equity_curve),
            cycles=list(self.cycles),
            summary=self._summary(start, end),
            daily=list(self.daily),
        )

    def _form(
        self,
        candidates: list[tuple[str, str]],
        closes: dict[str, pd.Series],
        volumes: dict[str, pd.Series],
        form_end: date,
        sector_of: dict[str, str],
    ) -> list[_Book]:
        scored: list[_Book] = []
        for a, b in candidates:
            if self.cfg.ban_dual_class and frozenset({a, b}) in DUAL_CLASS_BAN:
                continue
            if a not in closes or b not in closes:
                continue
            sa = _series_before(closes[a], form_end).tail(self.cfg.formation_days)
            sb = _series_before(closes[b], form_end).tail(self.cfg.formation_days)
            al = pd.concat([sa, sb], axis=1, join="inner").dropna()
            if len(al) < int(self.cfg.formation_days * 0.8):
                continue
            aa, bb = al.iloc[:, 0].astype(float), al.iloc[:, 1].astype(float)
            rets = pd.concat([aa.pct_change(), bb.pct_change()], axis=1).dropna()
            if len(rets) > 30:
                corr = float(rets.iloc[:, 0].corr(rets.iloc[:, 1]))
                if np.isfinite(corr) and corr > 0.995:
                    continue
            if not self._adv_ok(a, form_end, closes, volumes):
                continue
            if not self._adv_ok(b, form_end, closes, volumes):
                continue
            na, nb = aa / float(aa.iloc[0]), bb / float(bb.iloc[0])
            spr = na - nb
            ssd = float(((na - nb) ** 2).sum())
            sig = float(spr.std(ddof=1))
            if not np.isfinite(ssd) or sig <= 0:
                continue
            rt = self.costs.round_trip_cost_bps() / 10_000.0
            if self.cfg.open_z * sig < self.cfg.min_edge_cost_multiple * rt * 2:
                continue
            scored.append(
                _Book(
                    a=a,
                    b=b,
                    sector=sector_of.get(a) or sector_of.get(b) or "?",
                    ssd=ssd,
                    sigma=sig,
                    pa0=float(aa.iloc[-1]),
                    pb0=float(bb.iloc[-1]),
                )
            )
        scored.sort(key=lambda x: x.ssd)
        books: list[_Book] = []
        sec_n: dict[str, int] = {}
        for p in scored:
            if sec_n.get(p.sector, 0) >= self.cfg.max_sector_longs:
                continue
            books.append(p)
            sec_n[p.sector] = sec_n.get(p.sector, 0) + 1
            if len(books) >= self.cfg.top_n:
                break
        return books

    def _adv_ok(
        self,
        sym: str,
        asof: date,
        closes: dict[str, pd.Series],
        volumes: dict[str, pd.Series],
    ) -> bool:
        c = _series_before(closes[sym], asof).tail(self.cfg.adv_window)
        v = _series_before(volumes[sym], asof).tail(self.cfg.adv_window)
        al = pd.concat([c, v], axis=1, join="inner").dropna()
        if len(al) < max(5, self.cfg.adv_window // 2):
            return False
        return float((al.iloc[:, 0] * al.iloc[:, 1]).mean()) >= self.cfg.min_adv_usd

    def _spy_trend_ok(self, d: date, closes: dict[str, pd.Series]) -> bool:
        ma_n = self.cfg.spy_ma_filter
        if ma_n <= 0:
            return True
        s = _series_before(closes[self.cfg.beta_symbol], d)
        # include today? use < d only for MA to avoid same-bar peek on filter
        if len(s) < ma_n:
            return True
        return float(s.iloc[-1]) >= float(s.tail(ma_n).mean())

    def _marks(self, d: date, by_sym_date: dict[str, dict[date, Bar]]) -> dict[str, float]:
        return {sym: idx[d].close for sym, idx in by_sym_date.items() if d in idx}

    def _on_day(
        self,
        d: date,
        next_d: date | None,
        by_sym_date: dict[str, dict[date, Bar]],
        closes: dict[str, pd.Series],
        is_period_end: bool,
    ) -> None:
        m = self._marks(d, by_sym_date)
        beta = self.cfg.beta_symbol
        if beta not in m:
            return

        # Execute pending from prior signal day
        for bk in self._books:
            if bk.pending_flat and bk.long_sym and bk.qty > 0:
                self._close_long(bk, m, d)
                bk.pending_flat = False
            if bk.pending_long and bk.long_sym is None:
                if self._spy_trend_ok(d, closes):
                    self._open_long(bk, bk.pending_long, m, d, by_sym_date)
                bk.pending_long = None

        lag = max(int(self.cfg.execution_lag_days), 0)
        for bk in self._books:
            if bk.a not in m or bk.b not in m or bk.pa0 <= 0 or bk.pb0 <= 0:
                continue
            spr = m[bk.a] / bk.pa0 - m[bk.b] / bk.pb0

            if bk.long_sym is None and bk.pending_long is None:
                long_sym = None
                if spr <= -self.cfg.open_z * bk.sigma:
                    long_sym = bk.a
                elif spr >= self.cfg.open_z * bk.sigma:
                    long_sym = bk.b
                if long_sym is None:
                    continue
                open_n = sum(1 for x in self._books if x.long_sym or x.pending_long)
                if open_n >= self.cfg.max_concurrent_longs:
                    continue
                if not self._spy_trend_ok(d, closes):
                    continue
                if lag == 0:
                    self._open_long(bk, long_sym, m, d, by_sym_date)
                else:
                    # queue for next session (if period ends today, skip new entry)
                    if not is_period_end and next_d is not None:
                        bk.pending_long = long_sym
            else:
                flat = is_period_end
                if bk.long_sym == bk.a and spr >= 0:
                    flat = True
                if bk.long_sym == bk.b and spr <= 0:
                    flat = True
                if flat and bk.long_sym:
                    if lag == 0 or is_period_end:
                        self._close_long(bk, m, d)
                    else:
                        bk.pending_flat = True

        self._rebalance_spy(m, d)
        eq = self._mtm(m)
        self.equity = eq
        ts = datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc)
        self.equity_curve.append((ts, eq))
        spy_val = self.positions.get(beta, 0.0) * m.get(beta, 0.0)
        self.daily.append(
            {
                "date": str(d),
                "equity": eq,
                "cash": self.cash,
                "spy_value": spy_val,
                "n_rv_longs": sum(1 for b in self._books if b.long_sym),
                "spy_trend_ok": self._spy_trend_ok(d, closes),
            }
        )
        assert all(q >= -1e-9 for q in self.positions.values())
        assert self.cash >= -1.0, f"margin used: cash={self.cash}"

    def _open_long(
        self,
        bk: _Book,
        long_sym: str,
        m: dict[str, float],
        d: date,
        by_sym_date: dict[str, dict[date, Bar]],
    ) -> bool:
        px = m.get(long_sym)
        if not px or px <= 0:
            return False
        eq = self._mtm(m)
        target = eq * self.cfg.position_pct
        need = target + self.cfg.cash_buffer
        if self.cash < need:
            self._free_cash(m, d, need - self.cash)
        spend = min(target, max(self.cash - self.cfg.cash_buffer, 0.0))
        if spend < 200:
            return False
        qty = spend / px
        adv = self._adv_shares(long_sym, d, by_sym_date)
        if adv <= 0 or qty / adv > self.cfg.max_participation_pct_adv:
            return False
        if not self._buy(long_sym, qty, px, d, f"enter_long_rv:{bk.a}-{bk.b}:{long_sym}"):
            return False
        bk.long_sym, bk.qty = long_sym, qty
        return True

    def _close_long(self, bk: _Book, m: dict[str, float], d: date) -> None:
        if not bk.long_sym or bk.qty <= 0:
            bk.long_sym = None
            bk.qty = 0.0
            return
        sym = bk.long_sym
        if sym in m:
            self._sell(sym, bk.qty, m[sym], d, f"exit_rv:{bk.a}-{bk.b}:{sym}")
        bk.long_sym = None
        bk.qty = 0.0

    def _flatten_rv(self, m: dict[str, float], d: date, reason: str) -> None:
        beta = self.cfg.beta_symbol
        for bk in self._books:
            bk.pending_long = None
            bk.pending_flat = False
            if bk.long_sym and bk.qty > 0:
                self._close_long(bk, m, d)
        for sym, q in list(self.positions.items()):
            if sym == beta or q <= 0:
                continue
            if sym in m:
                self._sell(sym, q, m[sym], d, f"flatten_{reason}")

    def _free_cash(self, m: dict[str, float], d: date, amount: float) -> None:
        beta = self.cfg.beta_symbol
        if beta not in m or amount <= 0:
            return
        eq = self._mtm(m)
        spy_qty = self.positions.get(beta, 0.0)
        spy_val = spy_qty * m[beta]
        min_spy = eq * self.cfg.min_spy_pct
        sellable = max(spy_val - min_spy, 0.0)
        sell_val = min(amount, sellable)
        if sell_val < 50:
            return
        self._sell(beta, sell_val / m[beta], m[beta], d, "trim_spy_for_rv")

    def _rebalance_spy(self, m: dict[str, float], d: date) -> None:
        beta = self.cfg.beta_symbol
        if beta not in m:
            return
        excess = self.cash - self.cfg.cash_buffer
        if excess > 200:
            self._buy(beta, excess / m[beta], m[beta], d, "park_cash_spy")

    def _adv_shares(
        self, sym: str, d: date, by_sym_date: dict[str, dict[date, Bar]]
    ) -> float:
        days = sorted(x for x in by_sym_date.get(sym, {}) if x < d)[-self.cfg.adv_window :]
        if not days:
            return 0.0
        return float(np.mean([by_sym_date[sym][x].volume for x in days]))

    def _buy(self, sym: str, qty: float, mid: float, d: date, reason: str) -> bool:
        if qty <= 0 or mid <= 0:
            return False
        intent = OrderIntent(
            symbol=sym, side="buy", quantity=qty, order_type="market", reason=reason
        )
        price, commission, slip = self.costs.fill_price(intent, mid)
        cost = price * qty + commission
        if cost > self.cash + 1e-6:
            if self.cash <= self.cfg.cash_buffer:
                return False
            qty = max((self.cash - self.cfg.cash_buffer) / (price + 1e-9), 0.0)
            if qty * mid < 50:
                return False
            intent = OrderIntent(
                symbol=sym, side="buy", quantity=qty, order_type="market", reason=reason
            )
            price, commission, slip = self.costs.fill_price(intent, mid)
            cost = price * qty + commission
            if cost > self.cash + 1e-6:
                return False
        self.cash -= cost
        self.positions[sym] = self.positions.get(sym, 0.0) + qty
        self.fills.append(
            Fill(
                symbol=sym,
                side="buy",
                quantity=qty,
                price=price,
                timestamp=datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc),
                commission=commission,
                slippage=slip,
                reason=reason,
            )
        )
        return True

    def _sell(self, sym: str, qty: float, mid: float, d: date, reason: str) -> None:
        held = self.positions.get(sym, 0.0)
        qty = min(qty, held)
        if qty <= 0 or mid <= 0:
            return
        intent = OrderIntent(
            symbol=sym, side="sell", quantity=qty, order_type="market", reason=reason
        )
        price, commission, slip = self.costs.fill_price(intent, mid)
        self.cash += price * qty - commission
        new = held - qty
        if new <= 1e-10:
            self.positions.pop(sym, None)
        else:
            self.positions[sym] = new
        self.fills.append(
            Fill(
                symbol=sym,
                side="sell",
                quantity=qty,
                price=price,
                timestamp=datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc),
                commission=commission,
                slippage=slip,
                reason=reason,
            )
        )

    def _mtm(self, m: dict[str, float]) -> float:
        eq = self.cash
        for sym, q in self.positions.items():
            if q > 0 and sym in m:
                eq += q * m[sym]
        return eq

    def _summary(self, start: date, end: date) -> dict:
        start_eq = float(self.cfg.starting_equity)
        if not self.equity_curve:
            return {"error": "empty", "uses_margin": False}
        eq = pd.Series(
            [e for _, e in self.equity_curve],
            index=pd.to_datetime([t for t, _ in self.equity_curve]),
        )
        rets = eq.pct_change().dropna()
        years = max((end - start).days / 365.25, 1e-6)
        total = (float(eq.iloc[-1]) - start_eq) / start_eq
        cagr = (float(eq.iloc[-1]) / start_eq) ** (1 / years) - 1
        sharpe = (
            float(rets.mean() / rets.std() * math.sqrt(252))
            if len(rets) > 1 and rets.std() > 0
            else 0.0
        )
        dd = float(((eq - eq.cummax()) / eq.cummax()).min())
        return {
            "strategy": "cash_long_only_spy_core_rv_satellite",
            "uses_margin": False,
            "allows_shorts": False,
            "allows_futures": False,
            "execution_lag_days": self.cfg.execution_lag_days,
            "spy_ma_filter": self.cfg.spy_ma_filter,
            "total_return": total,
            "cagr": cagr,
            "sharpe_ratio": sharpe,
            "max_drawdown": dd,
            "ending_equity": float(eq.iloc[-1]),
            "starting_equity": start_eq,
            "number_of_fills": len(self.fills),
            "n_cycles": len(self.cycles),
            "params": {
                "top_n": self.cfg.top_n,
                "open_z": self.cfg.open_z,
                "position_pct": self.cfg.position_pct,
                "min_spy_pct": self.cfg.min_spy_pct,
                "max_concurrent_longs": self.cfg.max_concurrent_longs,
                "min_edge_cost_multiple": self.cfg.min_edge_cost_multiple,
                "execution_lag_days": self.cfg.execution_lag_days,
                "spy_ma_filter": self.cfg.spy_ma_filter,
            },
        }
