"""Locked prod_return_alpha signal helpers (no network)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from rlbot.prod_return_alpha import (
    ProdParams,
    month_end_mask,
    portfolio_weights_from_sleeves,
    to_yahoo_symbol,
    week_end_mask,
    weights_with_cash,
)


def test_to_yahoo_symbol() -> None:
    assert to_yahoo_symbol("BRK.B") == "BRK-B"
    assert to_yahoo_symbol("TQQQ") == "TQQQ"


def test_portfolio_weights_blend_sleeves() -> None:
    p = ProdParams(w_a=0.57)
    w = portfolio_weights_from_sleeves(
        tqqq_w=0.5, dual_asset="GLD", dual_w=0.25, p=p
    )
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert abs(w["TQQQ"] - 0.57 * 0.5) < 1e-9
    assert abs(w["GLD"] - 0.43 * 0.25) < 1e-9
    assert "BIL" in w


def test_weights_with_cash() -> None:
    tw = weights_with_cash({"TQQQ": 0.4, "GLD": 0.3})
    assert abs(sum(tw.values()) - 1.0) < 1e-9
    assert abs(tw["CASH"] - 0.3) < 1e-9


def test_week_and_month_end_masks() -> None:
    cal: list[date] = []
    d = date(2024, 1, 2)
    while d < date(2024, 3, 1):
        if d.weekday() < 5:
            cal.append(d)
        d += timedelta(days=1)
    wk = week_end_mask(cal)
    me = month_end_mask(cal)
    assert wk.dtype == np.bool_
    assert me.any()
    assert wk.any()
