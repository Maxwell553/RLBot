"""GeneralEquity1 pack adapter must load even when numba is iCloud-evicted."""

from __future__ import annotations

from pathlib import Path

from rlbot.pack_general_equity1 import (
    PACK_DIR,
    ensure_numba_for_pack,
    load_strategy,
    njit_passthrough,
    paper_plan,
)


def test_njit_passthrough_bare_and_kwargs() -> None:
    @njit_passthrough
    def add(a: int, b: int) -> int:
        return a + b

    @njit_passthrough(cache=True, nopython=True)
    def mul(a: int, b: int) -> int:
        return a * b

    assert add(2, 3) == 5
    assert mul(2, 3) == 6


def test_ensure_numba_for_pack_exposes_njit() -> None:
    ensure_numba_for_pack()
    from numba import njit

    @njit
    def ident(x: int) -> int:
        return x

    assert ident(7) == 7


def test_load_strategy_and_paper_plan() -> None:
    assert PACK_DIR.is_dir()
    assert (PACK_DIR / "data" / "bars.db").is_file()
    ge = load_strategy()
    assert ge.P.dual_b == "TLT"
    plan = paper_plan(aum=100_000.0)
    assert "portfolio_targets" in plan
    assert "asof" in plan
    tw = plan["portfolio_targets"]
    assert "TQQQ" in tw or "QQQ" in tw or "GLD" in tw
    assert abs(sum(float(v) for v in tw.values()) - 1.0) < 1e-6


def test_paper_plan_live_panel_sets_yahoo_asof() -> None:
    from datetime import date, timedelta

    import numpy as np

    from rlbot.pack_general_equity1 import paper_plan

    dates: list[date] = []
    d = date(2025, 1, 2)
    while len(dates) < 300:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    n = len(dates)
    rng = np.random.default_rng(0)

    def series(start: float) -> np.ndarray:
        r = rng.normal(0.0003, 0.012, n)
        return start * np.cumprod(1.0 + r)

    closes = {s: series(100.0) for s in ("SPY", "QQQ", "TQQQ", "BIL", "GLD", "TLT")}

    def ohlc(c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return c, c * 1.01, c * 0.99, c

    plan = paper_plan(
        aum=100_000.0,
        dates=dates,
        closes=closes,
        ohlc={"TQQQ": ohlc(closes["TQQQ"]), "QQQ": ohlc(closes["QQQ"])},
    )
    assert plan["data_source"] == "yahoo"
    assert plan["asof"] == str(dates[-1])
    assert "portfolio_targets" in plan
