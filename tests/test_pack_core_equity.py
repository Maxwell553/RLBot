"""CoreEquity pack adapter must load the locked pack without editing it."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from rlbot.pack_core_equity import (
    PACK_DIR,
    PAPER_RUN_ID,
    STRATEGY_ID,
    book_symbols,
    load_strategy,
    locked_params,
    paper_plan,
    panel_symbols,
    weights_from_targets,
)


def test_pack_layout_and_identity() -> None:
    assert PACK_DIR.is_dir()
    assert (PACK_DIR / "data" / "bars.db").is_file()
    assert STRATEGY_ID == "core_equity"
    assert PAPER_RUN_ID == "CORE_EQUITY"
    assert "TQQQ" not in book_symbols()
    assert "TQQQ" not in panel_symbols()
    assert "QQQ" in book_symbols()
    assert "SPY" in panel_symbols()


def test_load_strategy_uses_locked_params() -> None:
    ce = load_strategy()
    p = locked_params()
    assert ce.P.eq_sym == "QQQ"
    assert p.eq_sym == "QQQ"
    assert p.q_cap == 1.4
    assert p.trend_mode == "abs_or_sma"
    assert p.dual_b == "TLT"
    assert p.w_a == 0.58


def test_paper_plan_from_frozen_bars() -> None:
    plan = paper_plan(aum=100_000.0)
    assert "portfolio_targets" in plan
    assert "asof" in plan
    assert plan["data_source"] == "bars.db"
    tw = plan["portfolio_targets"]
    assert "TQQQ" not in tw
    assert abs(sum(float(v) for v in tw.values()) - 1.0) < 0.5 or "QQQ" in tw


def _synth_panel(n: int = 300, *, seed: int = 0):
    dates: list[date] = []
    d = date(2025, 1, 2)
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    rng = np.random.default_rng(seed)

    def series(start: float, vol: float) -> np.ndarray:
        r = rng.normal(0.0003, vol, n)
        return start * np.cumprod(1.0 + r)

    closes = {
        "SPY": series(500.0, 0.01),
        "QQQ": series(400.0, 0.012),
        "BIL": series(91.0, 0.0002),
        "GLD": series(180.0, 0.009),
        "TLT": series(90.0, 0.008),
    }

    def ohlc(c: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return c, c * 1.01, c * 0.99, c

    px = {k: np.asarray(v, dtype=np.float64) for k, v in closes.items()}
    oh = {k: ohlc(px[k]) for k in px}
    return dates, px, oh


def test_paper_plan_live_panel_sets_yahoo_asof() -> None:
    from rlbot.pack_core_equity import latest_portfolio_weights

    dates, px, oh = _synth_panel()
    plan = paper_plan(aum=100_000.0, dates=dates, closes=px, ohlc=oh)
    assert plan["data_source"] == "yahoo"
    assert plan["asof"] == str(dates[-1])
    assert "TQQQ" not in (plan.get("portfolio_targets") or {})
    tw = latest_portfolio_weights(aum=100_000.0, dates=dates, closes=px, ohlc=oh)
    assert "TQQQ" not in tw
    assert "QQQ" in tw or "GLD" in tw or "TLT" in tw or "BIL" in tw or "CASH" in tw


def test_weights_from_targets_keeps_qqq_leverage() -> None:
    tw = weights_from_targets(
        {
            "portfolio_QQQ": 0.812,
            "dual_asset": "GLD",
            "portfolio_dual": 0.42,
        }
    )
    assert abs(tw["QQQ"] - 0.812) < 1e-12
    assert abs(tw["GLD"] - 0.42) < 1e-12
    assert "BIL" not in tw
    assert "TQQQ" not in tw
    assert abs(sum(tw.values()) - 1.232) < 1e-9


def test_cc_sleeve_weight_atr_hyst_only_when_long() -> None:
    from rlbot.pack_core_equity import cc_sleeve_weight

    kwargs = dict(
        ok=True,
        vol=0.20,
        atr=0.123,
        vt=0.30,
        atr_max=0.12,
        w_cap=1.4,
        atr_hyst=0.05,
    )
    assert cc_sleeve_weight(was_long=False, **kwargs) == 0.0
    w_long = cc_sleeve_weight(was_long=True, **kwargs)
    assert w_long > 0.0
    assert abs(w_long - min(1.4, 0.30 / 0.20)) < 1e-12


def test_apply_sleeve_a_zeros_qqq_when_flat() -> None:
    from rlbot.pack_core_equity import SleeveAState, apply_sleeve_a_to_targets, weights_from_targets

    sleeve = SleeveAState(
        equity=0.8,
        peak=1.0,
        flat=True,
        cool_remaining=10,
        held_w=1.2,
        ok_now=False,
        next_cc_weight=1.4,
    )
    targets = apply_sleeve_a_to_targets(
        {
            "portfolio_QQQ": 0.812,
            "dual_asset": "GLD",
            "portfolio_dual": 0.42,
        },
        sleeve,
    )
    assert targets["portfolio_QQQ"] == 0.0
    tw = weights_from_targets(targets)
    assert "QQQ" not in tw
    assert abs(tw["GLD"] - 0.42) < 1e-12
    assert abs(tw["BIL"] - 0.58) < 1e-12


def test_live_session_flags_do_not_force_tip() -> None:
    from rlbot.pack_core_equity import live_session_rebalance_flags

    days = [date(2026, 8, 13), date(2026, 8, 14), date(2026, 8, 17), date(2026, 8, 18)]
    assert live_session_rebalance_flags(days, 3) == (False, False)
    assert live_session_rebalance_flags(days, 1) == (True, False)


def test_reset_paper_book_is_flat_cash(tmp_path, monkeypatch) -> None:
    import rlbot.paper_core_equity as paper

    monkeypatch.setattr(paper, "PAPER_DIR", tmp_path)
    monkeypatch.setattr(paper, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(paper, "ORDERS_PATH", tmp_path / "order_intents.jsonl")
    monkeypatch.setattr(paper, "EXECUTION_DIR", tmp_path)
    monkeypatch.setattr(paper, "flatten_core_equity_companion", lambda **kwargs: {"n_bars": 3})

    out = paper.reset_paper_book(initial_cash=100_000.0, hold_until=date(2026, 8, 20))
    assert out["equity"] == 100_000.0
    assert out["hold_until"] == "2026-08-20"
    st = paper.load_state()
    assert abs(float(st["cash"]) - 100_000.0) < 1e-9
    assert st["positions"] == {}
    assert st["target_weights"] == {"CASH": 1.0}
    assert st["last_trade_date"] == "2026-08-20"
    ledger = (tmp_path / "shadow_ledger_CORE_EQUITY.jsonl").read_text(encoding="utf-8")
    assert "Reset to 100k cash" in ledger


def test_paper_session_vs_last_bar_already_traded() -> None:
    from rlbot.paper_core_equity import (
        core_paper_already_traded,
        paper_book_needs_reopen,
        resolve_paper_session_day,
    )

    bar = date(2026, 8, 20)
    friday = date(2026, 8, 21)
    assert resolve_paper_session_day(friday, bar) == friday
    assert resolve_paper_session_day(None, bar, today=friday) == friday
    assert resolve_paper_session_day(bar, bar) == bar
    assert core_paper_already_traded("2026-08-20", bar)
    assert not core_paper_already_traded("2026-08-20", friday)

    reset = {"positions": {}, "last_trade_date": "2026-08-20"}
    filled = {"positions": {"QQQ": 10.0}, "last_trade_date": "2026-08-20"}
    assert paper_book_needs_reopen(reset, session="2026-08-21")
    assert not paper_book_needs_reopen(reset, session="2026-08-20")
    assert not paper_book_needs_reopen(filled, session="2026-08-21")
