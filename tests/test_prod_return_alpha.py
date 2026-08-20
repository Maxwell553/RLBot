"""GeneralEquity1 forward MTM helpers (no network)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from rlbot.prod_return_alpha import (
    PAPER_RUN_ID,
    STRATEGY_ID,
    month_end_mask,
    to_yahoo_symbol,
    week_end_mask,
    weights_with_cash,
)


def test_identity() -> None:
    assert STRATEGY_ID == "prod_return_alpha_v3"
    assert PAPER_RUN_ID == "GENERAL_EQUITY1"


def test_to_yahoo_symbol() -> None:
    assert to_yahoo_symbol("BRK.B") == "BRK-B"
    assert to_yahoo_symbol("TQQQ") == "TQQQ"


def test_weights_with_cash() -> None:
    tw = weights_with_cash({"TQQQ": 0.4, "GLD": 0.3})
    assert abs(sum(tw.values()) - 1.0) < 1e-9
    assert abs(tw["CASH"] - 0.3) < 1e-9
    # Existing CASH must not be double-counted.
    tw2 = weights_with_cash({"TQQQ": 0.2, "CASH": 0.8})
    assert abs(tw2["TQQQ"] - 0.2) < 1e-9
    assert abs(tw2["CASH"] - 0.8) < 1e-9


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


def test_session_rebalance_flags_live_tip() -> None:
    from rlbot.prod_return_alpha import session_rebalance_flags

    days = [date(2026, 8, 13), date(2026, 8, 14), date(2026, 8, 17), date(2026, 8, 18)]
    # Thursday → Friday same week
    assert session_rebalance_flags(days, 0) == (False, False)
    # Friday Aug 14 is week-end (next bar is Monday)
    assert session_rebalance_flags(days, 1) == (True, False)
    # Tuesday live tip is not week/month end
    assert session_rebalance_flags(days, 3) == (False, False)
    friday_month_end = [date(2026, 7, 30), date(2026, 7, 31)]
    assert session_rebalance_flags(friday_month_end, 1) == (True, True)
