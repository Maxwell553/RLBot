"""LiveTrader CoreEquity pack + IBKR gates (no TWS required)."""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

LIVE = Path(__file__).resolve().parents[1] / "LiveTrader"
if str(LIVE) not in sys.path:
    sys.path.insert(0, str(LIVE))

from book import (  # noqa: E402
    foreign_symbols,
    needs_fractional,
    orders_to_targets,
    session_rebalance_flags,
)
from config import load_config  # noqa: E402
from ce_strategy import P, latest_targets, portfolio_weights  # noqa: E402
from ibkr_client import AccountSnapshot  # noqa: E402
from preflight import blocking_failures, evaluate  # noqa: E402


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
    oh = {"QQQ": ohlc(px["QQQ"])}
    return dates, px, oh


def test_copy_matches_pack_on_synthetic_panel() -> None:
    dates, px, oh = _synth_panel()
    from rlbot.pack_core_equity import load_strategy, paper_plan

    ce = load_strategy()
    pack_t = ce.latest_targets(dates, px, oh["QQQ"], oh["QQQ"], ce.P)
    live_t = latest_targets(dates, px, oh["QQQ"], oh["QQQ"], P)
    for key in (
        "asof",
        "portfolio_QQQ",
        "portfolio_dual",
        "dual_asset",
        "QQQ_cc_weight",
    ):
        pv, lv = pack_t[key], live_t[key]
        if isinstance(pv, float):
            assert abs(float(pv) - float(lv)) < 1e-12, key
        else:
            assert pv == lv, key
    pack_plan = paper_plan(aum=1_000.0, dates=dates, closes=px, ohlc=oh)
    live_w = portfolio_weights(live_t, P)
    from rlbot.pack_core_equity import latest_portfolio_weights

    pack_w = latest_portfolio_weights(aum=1_000.0, dates=dates, closes=px, ohlc=oh)
    assert pack_plan["data_source"] == "yahoo"
    assert pack_w.keys() == live_w.keys()
    for k in live_w:
        assert abs(live_w[k] - pack_w[k]) < 1e-12


def test_params_match_locked_pack() -> None:
    from rlbot.pack_core_equity import locked_params

    lp = locked_params()
    assert lp.w_a == P.w_a
    assert lp.eq_sym == P.eq_sym
    assert lp.vt == P.vt
    assert lp.q_cap == P.q_cap
    assert lp.dual_b == P.dual_b
    assert lp.es == P.es
    assert lp.cool == P.cool
    assert lp.atr_hyst == P.atr_hyst
    assert lp.trend_mode == "abs_or_sma"


def test_default_data_source_is_ibkr() -> None:
    cfg = load_config()
    assert cfg.data_source == "ibkr"
    assert cfg.yahoo_fallback is True


def test_panel_from_ohlc_frames_aligns_to_spy() -> None:
    from data import panel_from_ohlc_frames

    d0, d1, d2 = date(2026, 8, 18), date(2026, 8, 19), date(2026, 8, 20)
    frames = {
        "SPY": {d0: (1, 1, 1, 100.0), d1: (1, 1, 1, 101.0), d2: (1, 1, 1, 102.0)},
        "QQQ": {d0: (1, 1, 1, 50.0), d2: (1, 1, 1, 52.0)},
    }
    dates, closes, ohlc = panel_from_ohlc_frames(frames, symbols=("SPY", "QQQ"))
    assert dates == [d0, d1, d2]
    assert closes["QQQ"][1] == 50.0  # ffill Thursday gap
    assert closes["QQQ"][2] == 52.0
    assert ohlc["SPY"][3][-1] == 102.0


def test_ib_bar_date_parses_yyyymmdd() -> None:
    from ibkr_client import _ib_bar_date

    assert _ib_bar_date("20260820") == date(2026, 8, 20)
    assert _ib_bar_date("2026-08-20 16:00:00") == date(2026, 8, 20)


def test_book_has_no_levered_etfs() -> None:
    from ce_strategy import BOOK_SYMBOLS, PANEL_SYMBOLS

    assert "TQQQ" not in BOOK_SYMBOLS
    assert "TQQQ" not in PANEL_SYMBOLS
    assert "QQQ" in BOOK_SYMBOLS
    assert "GLD" in BOOK_SYMBOLS
    assert "TLT" in BOOK_SYMBOLS
    assert "BIL" in BOOK_SYMBOLS


def test_orders_sells_first_and_skips_cash() -> None:
    orders = orders_to_targets(
        1000.0,
        {"QQQ": 10.0, "AAPL": 2.0},
        {"QQQ": 50.0, "GLD": 400.0, "AAPL": 200.0},
        {"QQQ": 0.2, "GLD": 0.3, "CASH": 0.5},
        min_notional=1.0,
    )
    assert [o.side for o in orders[:2]] == ["sell", "sell"]
    assert {o.symbol for o in orders if o.side == "sell"} == {"QQQ", "AAPL"}
    assert all(o.symbol != "CASH" for o in orders)


def test_session_rebalance_flags_do_not_force_tip() -> None:
    days = [date(2026, 8, 13), date(2026, 8, 14), date(2026, 8, 17), date(2026, 8, 18)]
    assert session_rebalance_flags(days, 3) == (False, False)
    assert session_rebalance_flags(days, 1) == (True, False)


def test_foreign_and_fractional_helpers() -> None:
    assert foreign_symbols({"QQQ": 3.0, "NVDA": 1.0, "USD": 12.0}) == ["NVDA"]
    from book import OrderIntent

    o = OrderIntent("QQQ", "buy", 0.07, 0.05, 700.0, 49.0)
    assert needs_fractional([o]) == ["QQQ"]
    whole = OrderIntent("QQQ", "buy", 3.0, 0.2, 70.0, 210.0)
    assert needs_fractional([whole]) == []


def test_round_to_whole_shares_floors_and_drops() -> None:
    from book import OrderIntent, assign_order_types, round_to_whole_shares

    orders = [
        OrderIntent("QQQ", "buy", 2.53, 0.18, 71.0, 179.63),
        OrderIntent("GLD", "buy", 0.06, 0.04, 713.0, 42.78),
        OrderIntent("TLT", "buy", 0.56, 0.22, 411.0, 230.16),
    ]
    out = round_to_whole_shares(orders, min_notional=1.0)
    assert [o.symbol for o in out] == ["QQQ"]
    assert out[0].qty == 2.0
    assert out[0].notional == 142.0
    typed = assign_order_types(out, whole_share_type="MOC", fractional_type="MKT")
    assert typed[0].order_type == "MOC"


def test_journal_ignores_failed_fills(tmp_path, monkeypatch) -> None:
    import trader as lt

    path = tmp_path / "order_intents.jsonl"
    monkeypatch.setattr(lt, "_journal_path", lambda: path)
    lt._append_jsonl(
        path,
        {
            "asof": "2026-08-20",
            "account": "U1",
            "submitted": True,
            "fills_ok": False,
        },
    )
    assert lt._journal_submitted("U1", "2026-08-20") is False
    lt._append_jsonl(
        path,
        {
            "asof": "2026-08-20",
            "account": "U1",
            "submitted": True,
            "fills_ok": True,
        },
    )
    assert lt._journal_submitted("U1", "2026-08-20") is True


def _cfg(**kwargs):
    os.environ.pop("LIVE_TRADER_MODE", None)
    os.environ.pop("IBKR_PORT", None)
    cfg = load_config()
    data = cfg.__dict__.copy()
    data.update(kwargs)
    return type(cfg)(**data)


def test_preflight_blocks_foreign_and_live_arm() -> None:
    dates, px, oh = _synth_panel()
    marks = {k: float(v[-1]) for k, v in px.items()}
    weights = {"QQQ": 0.2, "GLD": 0.3, "CASH": 0.5}
    snap = AccountSnapshot(
        account="U12345",
        net_liquidation=1000.0,
        cash=200.0,
        buying_power=200.0,
        positions={"NVDA": 4.0},
        port=7496,
    )
    cfg = _cfg(mode="live", port=7496, allow_live=False, allow_foreign_positions=False)
    checks = evaluate(
        cfg=cfg,
        dates=dates,
        px=px,
        ohlc=oh,
        marks=marks,
        weights=weights,
        snapshot=snap,
        orders=[],
        want_connect=True,
        want_submit=True,
        arm_live=False,
        confirm_env="",
    )
    names = {c.name: c for c in checks}
    assert names["foreign_positions"].ok is False
    assert names["live_arm"].ok is False
    assert blocking_failures(checks)


def test_preflight_rejects_live_mode_on_paper_port() -> None:
    dates, px, oh = _synth_panel()
    cfg = _cfg(mode="live", port=7497, allow_live=True)
    checks = evaluate(
        cfg=cfg,
        dates=dates,
        px=px,
        ohlc=oh,
        marks={},
        weights={"CASH": 1.0},
        snapshot=None,
        want_submit=True,
        arm_live=True,
        confirm_env="CORE",
    )
    names = {c.name: c for c in checks}
    assert names["port_mode"].ok is False


def test_build_plan_seeds_flat_account(monkeypatch) -> None:
    from book import CoolState
    import trader as lt

    cfg = load_config()
    snap = AccountSnapshot(
        account="DU123",
        net_liquidation=1000.0,
        cash=1000.0,
        buying_power=1000.0,
        positions={},
        port=7497,
    )
    dates, px, oh = _synth_panel()

    def fake_load(**kwargs):
        del kwargs
        ohlc = dict(oh)
        for extra in ("GLD", "TLT", "SPY", "BIL"):
            ohlc[extra] = (px[extra], px[extra] * 1.01, px[extra] * 0.99, px[extra])
        return dates, px, ohlc

    monkeypatch.setattr(lt, "load_live_panel", fake_load)
    monkeypatch.setattr(lt, "load_cool", lambda: CoolState())
    plan = lt.build_plan(cfg=cfg, snapshot=snap, force_refresh=False)
    assert plan["data_source"] == "yahoo"
    assert plan["asof"] == str(dates[-1])
    gross = sum(plan["target_weights"].values())
    assert 0.5 < gross < 1.6  # pack may cash-finance QQQ up to 1.4× on sleeve A
    assert "TQQQ" not in plan["target_weights"]
    assert plan["rebalance"] is True  # seed_if_flat
    assert plan["n_orders"] >= 1
    assert all(o["order_type"] in {"MKT", "MOC"} for o in plan["orders"])


def test_assign_order_types_moc_only_for_whole_shares() -> None:
    from book import OrderIntent, assign_order_types, is_whole_share, needs_fractional

    orders = [
        OrderIntent("QQQ", "buy", 3.0, 0.2, 70.0, 210.0),
        OrderIntent("GLD", "buy", 0.06, 0.04, 700.0, 42.0),
        OrderIntent("TLT", "buy", 2.34, 0.25, 400.0, 936.0),
    ]
    typed = assign_order_types(orders, whole_share_type="MOC", fractional_type="MKT")
    by = {o.symbol: o.order_type for o in typed}
    assert is_whole_share(3.0)
    assert not is_whole_share(2.34)
    assert by["QQQ"] == "MOC"
    assert by["GLD"] == "MKT"
    assert by["TLT"] == "MKT"
    assert needs_fractional(typed) == ["GLD", "TLT"]


def test_sleeve_split_week_end_skips_gld() -> None:
    from book import active_sleeve_symbols, orders_to_targets

    allow = active_sleeve_symbols(week_end=True, month_end=False, seed=False)
    assert "QQQ" in allow
    assert "GLD" not in allow
    assert "TLT" not in allow
    orders = orders_to_targets(
        1000.0,
        {"QQQ": 0.0, "GLD": 1.0},
        {"QQQ": 700.0, "GLD": 400.0},
        {"QQQ": 0.25, "GLD": 0.3, "CASH": 0.45},
        allow_symbols=allow,
    )
    assert {o.symbol for o in orders} <= {"QQQ", "BIL"}
    assert all(o.symbol != "GLD" for o in orders)


def test_clamp_buys_to_cash_uses_sell_proceeds() -> None:
    from book import OrderIntent, clamp_buys_to_cash

    orders = [
        OrderIntent("QQQ", "sell", 2.0, 0.0, 50.0, 100.0),
        OrderIntent("GLD", "buy", 1.0, 0.4, 400.0, 400.0),
    ]
    out = clamp_buys_to_cash(orders, cash=50.0, min_notional=1.0)
    buy = next(o for o in out if o.side == "buy")
    assert buy.symbol == "GLD"
    assert buy.notional <= 150.0 + 1e-9
    assert abs(buy.qty * buy.mark - buy.notional) < 1e-6


def test_already_traded_skips_orders(monkeypatch) -> None:
    from book import CoolState
    import trader as lt

    cfg = load_config()
    dates, px, oh = _synth_panel()
    snap = AccountSnapshot(
        account="DU123",
        net_liquidation=1000.0,
        cash=1000.0,
        buying_power=1000.0,
        positions={},
        port=7497,
    )

    def fake_load(**kwargs):
        del kwargs
        ohlc = dict(oh)
        for extra in ("GLD", "TLT", "SPY", "BIL"):
            ohlc[extra] = (px[extra], px[extra] * 1.01, px[extra] * 0.99, px[extra])
        return dates, px, ohlc

    monkeypatch.setattr(lt, "load_live_panel", fake_load)
    monkeypatch.setattr(
        lt,
        "load_cool",
        lambda: CoolState(last_trade_date=str(dates[-1])),
    )
    monkeypatch.setattr(lt, "_journal_submitted", lambda *a, **k: False)
    plan = lt.build_plan(cfg=cfg, snapshot=snap, force_refresh=False)
    assert plan["already_traded"] is True
    assert plan["rebalance"] is False
    assert plan["n_orders"] == 0
    assert plan["skip_reason"] == "already_traded"


def test_open_orders_block_submit() -> None:
    dates, px, oh = _synth_panel()
    from book import OrderIntent

    snap = AccountSnapshot(
        account="DU1",
        net_liquidation=1000.0,
        cash=1000.0,
        buying_power=1000.0,
        positions={},
        open_order_symbols=["QQQ"],
        port=7497,
    )
    cfg = _cfg(mode="paper", port=7497, account="DU1", require_account=True)
    checks = evaluate(
        cfg=cfg,
        dates=dates,
        px=px,
        ohlc=oh,
        marks={},
        weights={"CASH": 1.0},
        snapshot=snap,
        orders=[OrderIntent("QQQ", "buy", 1.0, 0.2, 70.0, 70.0)],
        want_submit=True,
    )
    names = {c.name: c for c in checks}
    assert names["open_orders"].ok is False


def test_flatten_and_fake_fill_reconcile() -> None:
    from book import flatten_intents, weight_drift
    from ibkr_client import FakeBroker

    snap = AccountSnapshot(
        account="DU1",
        net_liquidation=1000.0,
        cash=100.0,
        buying_power=100.0,
        positions={"QQQ": 2.0, "GLD": 1.0},
        last_prices={"QQQ": 50.0, "GLD": 400.0},
        port=7497,
    )
    orders = flatten_intents(snap.positions, snap.last_prices)
    assert {o.symbol for o in orders} == {"QQQ", "GLD"}
    assert all(o.side == "sell" for o in orders)
    broker = FakeBroker(snap).connect()
    fills = broker.submit(orders, order_type="MKT", tif="DAY")
    assert all(f["status"] == "Filled" for f in fills)
    after = broker.snapshot()
    assert after.positions == {}
    drift = weight_drift(1000.0, after.positions, snap.last_prices, {"CASH": 1.0})
    assert drift == []


def test_merge_marks_prefers_ib() -> None:
    from book import merge_marks

    out = merge_marks({"QQQ": 50.0, "GLD": 700.0}, {"QQQ": 51.5})
    assert out["QQQ"] == 51.5
    assert out["GLD"] == 700.0


def test_append_jsonl_and_default_refresh(tmp_path) -> None:
    import argparse
    import trader as lt

    path = tmp_path / "order_intents.jsonl"
    lt._append_jsonl(path, {"asof": "2026-08-19", "submitted": False})
    lt._append_jsonl(path, {"asof": "2026-08-19", "submitted": True})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert '"submitted": false' in lines[0]
    assert '"submitted": true' in lines[1]
    assert lt._force_refresh(argparse.Namespace()) is True
    assert lt._force_refresh(argparse.Namespace(no_refresh_data=False)) is True
    assert lt._force_refresh(argparse.Namespace(no_refresh_data=True)) is False


def test_spendable_cash_uses_buying_power() -> None:
    from book import spendable_cash

    assert spendable_cash(1000.0, 2000.0) == 2000.0
    assert spendable_cash(1000.0, 0.0) == 1000.0
    assert spendable_cash(0.0, 1500.0) == 1500.0


def test_park_sleeve_a_moves_qqq_to_bil() -> None:
    from book import park_sleeve_a

    out = park_sleeve_a({"QQQ": 0.58, "GLD": 0.42}, "GLD")
    assert abs(out["BIL"] - 0.58) < 1e-12
    assert abs(out["GLD"] - 0.42) < 1e-12
    assert "QQQ" not in out
    assert "CASH" not in out


def test_session_rebalance_flags_yahoo_lag_friday() -> None:
    thu = [date(2026, 8, 12), date(2026, 8, 13)]  # Wed, Thu
    assert session_rebalance_flags(thu, 1) == (False, False)
    wk, me = session_rebalance_flags(thu, 1, calendar_today=date(2026, 8, 14))
    assert wk is True
    assert me is False


def test_cool_state_ticks_once_and_needs_trend() -> None:
    from book import update_cool_state

    same = update_cool_state(
        1.0, 0.8, True, 10, asof="2026-08-14", last_asof="2026-08-14"
    )
    assert same == (1.0, True, 10)
    stepped = update_cool_state(
        1.0, 0.8, True, 10, asof="2026-08-15", last_asof="2026-08-14"
    )
    assert stepped == (1.0, True, 9)
    waiting = update_cool_state(1.0, 0.8, True, 0, ok_now=False)
    assert waiting == (1.0, True, 0)
    reenter = update_cool_state(1.0, 0.8, True, 0, ok_now=True)
    assert reenter[1] is False
    assert reenter[2] == 0


def test_es_park_allows_qqq_and_bil_midweek() -> None:
    from book import active_sleeve_symbols

    allow = active_sleeve_symbols(week_end=False, month_end=False, seed=False, es_park=True)
    assert "QQQ" in allow
    assert "BIL" in allow
    assert "GLD" not in allow
    assert "TLT" not in allow
