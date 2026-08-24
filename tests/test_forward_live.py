"""Torch-free live forward MTM helpers (30m candles + legacy close path)."""

from __future__ import annotations

import numpy as np

import pandas as pd

from rlbot.forward_live import (
    _ew_nav,
    _fill_ohlc_nans,
    _merge_price_history,
    _nav_from_weights,
    _nav_series_from_ohlc,
    _resolve_book_start,
    _spy_nav,
    _weight_vector,
    equal_weight_ohlc_candles,
    offhours_extend_until,
    portfolio_ohlc_candles,
    prices_are_stale,
    spy_ohlc_candles,
)


def test_weight_vector_normalizes_cash_alias() -> None:
    labels = ["Cash", "SP500", "GOLD"]
    w = _weight_vector({"CASH": 0.5, "SP500": 0.25, "GOLD": 0.25}, labels)
    assert abs(w.sum() - 1.0) < 1e-9
    assert abs(w[0] - 0.5) < 1e-9


def test_nav_from_weights_cash_only_flat_without_yield() -> None:
    closes = np.array([[100.0, 50.0], [110.0, 55.0], [121.0, 60.5]], dtype=np.float64)
    w = [_weight_vector({"Cash": 1.0}, ["Cash", "A", "B"])] * 3
    nav = _nav_from_weights(closes, w, initial_cash=100_000.0, cash_daily_yield=0.0)
    assert nav.shape == (3,)
    assert abs(nav[0] - 100_000.0) < 1e-6
    assert abs(nav[-1] - 100_000.0) < 1e-6


def test_nav_from_weights_full_invest_tracks_asset() -> None:
    closes = np.array([[100.0], [110.0], [121.0]], dtype=np.float64)
    w = [_weight_vector({"Cash": 0.0, "A": 1.0}, ["Cash", "A"])] * 3
    nav = _nav_from_weights(closes, w, initial_cash=100_000.0)
    assert abs(nav[-1] / nav[0] - 1.21) < 1e-9


def test_benchmark_navs() -> None:
    closes = np.array([[100.0, 200.0], [110.0, 180.0]], dtype=np.float64)
    ew = _ew_nav(closes, 100_000.0)
    # mean return = 0.5*(0.10 + -0.10) = 0
    assert abs(ew[-1] - 100_000.0) < 1e-6
    spy = _spy_nav(np.array([50.0, 55.0]), 100_000.0)
    assert abs(spy[-1] - 110_000.0) < 1e-6


def test_portfolio_ohlc_full_invest_tracks_asset_bar() -> None:
    # One asset: open 100 → close 110 on bar 0; bar 1 open 110 → close 121.
    open_ = np.array([[100.0], [110.0]], dtype=np.float64)
    high = np.array([[112.0], [122.0]], dtype=np.float64)
    low = np.array([[99.0], [109.0]], dtype=np.float64)
    close = np.array([[110.0], [121.0]], dtype=np.float64)
    w = _weight_vector({"Cash": 0.0, "A": 1.0}, ["Cash", "A"])
    candles = portfolio_ohlc_candles(
        open_, high, low, close, w, initial_cash=100_000.0, cash_yield_per_bar=0.0
    )
    assert candles.shape == (2, 4)
    assert abs(candles[0, 0] - 100_000.0) < 1e-6
    assert abs(candles[0, 3] - 110_000.0) < 1e-6
    assert abs(candles[1, 3] - 121_000.0) < 1e-6
    assert candles[0, 1] >= candles[0, 0] and candles[0, 1] >= candles[0, 3]
    assert candles[0, 2] <= candles[0, 0] and candles[0, 2] <= candles[0, 3]


def test_cash_park_candles_earn_yield() -> None:
    open_ = np.array([[100.0], [100.0]], dtype=np.float64)
    high = open_.copy()
    low = open_.copy()
    close = open_.copy()
    w = _weight_vector({"Cash": 1.0}, ["Cash", "A"])
    candles = portfolio_ohlc_candles(
        open_, high, low, close, w, initial_cash=100_000.0, cash_yield_per_bar=0.001
    )
    # Bar 0: open=initial; close earns one bar of yield from the open base.
    assert abs(candles[0, 0] - 100_000.0) < 1e-6
    assert abs(candles[0, 3] - 100_100.0) < 1e-6


def test_spy_and_ew_candle_helpers() -> None:
    o = np.array([50.0, 55.0])
    h = np.array([51.0, 56.0])
    l = np.array([49.0, 54.0])
    c = np.array([50.5, 55.5])
    spy = spy_ohlc_candles(o, h, l, c, initial_cash=100_000.0)
    assert abs(spy[0, 0] - 100_000.0) < 1e-6
    assert abs(spy[1, 3] / spy[0, 0] - 55.5 / 50.0) < 1e-9

    open_ = np.array([[100.0, 200.0], [110.0, 180.0]])
    high = open_ + 1
    low = open_ - 1
    close = np.array([[110.0, 180.0], [121.0, 162.0]])
    ew = equal_weight_ohlc_candles(open_, high, low, close, initial_cash=100_000.0)
    assert ew.shape == (2, 4)
    assert abs(ew[0, 0] - 100_000.0) < 1e-6


def test_resolve_book_start_persists_unless_reset() -> None:
    existing = {"holdout_start": "2026-07-28", "book_start": "2026-07-28"}
    stamp = {"holdout_start": "2026-07-31"}
    assert _resolve_book_start(existing, stamp) == "2026-07-28"
    assert _resolve_book_start(existing, stamp, reset_book=True) == str(
        pd.Timestamp.now(tz="America/New_York").date()
    )


def test_nav_series_starts_at_initial_cash() -> None:
    ohlc = np.array([[100_000.0, 101_000.0, 99_000.0, 100_500.0]], dtype=np.float64)
    nav = _nav_series_from_ohlc(ohlc, initial_cash=100_000.0)
    assert abs(nav[0] - 100_000.0) < 1e-6


def test_merge_price_history_unions_and_prefers_fresh() -> None:
    old_t = pd.DatetimeIndex(["2026-07-30 09:30", "2026-07-30 09:35"])
    new_t = pd.DatetimeIndex(["2026-07-30 09:35", "2026-07-31 09:30"])
    old_o = np.array([[1.0], [2.0]])
    new_o = np.array([[9.0], [3.0]])  # overlapping 09:35 → fresh wins
    old_s = np.array([10.0, 20.0])
    new_s = np.array([29.0, 30.0])
    times, o, _h, _l, _c, so, _sh, _sl, _sc = _merge_price_history(
        cached_times=old_t,
        cached_o=old_o,
        cached_h=old_o,
        cached_l=old_o,
        cached_c=old_o,
        cached_so=old_s,
        cached_sh=old_s,
        cached_sl=old_s,
        cached_sc=old_s,
        times=new_t,
        o=new_o,
        h=new_o,
        l=new_o,
        c=new_o,
        so=new_s,
        sh=new_s,
        sl=new_s,
        sc=new_s,
    )
    assert len(times) == 3
    assert abs(float(o[0, 0]) - 1.0) < 1e-9
    assert abs(float(o[1, 0]) - 9.0) < 1e-9  # fresh overlap
    assert abs(float(o[2, 0]) - 3.0) < 1e-9
    assert abs(float(so[1]) - 29.0) < 1e-9


def test_offhours_extend_refuses_mid_session_gap() -> None:
    """A Thursday 10:32 last print must not invent bars through later sessions."""
    last = pd.Timestamp("2026-08-06 10:32")  # Thursday, mid-RTH
    # Same evening
    assert offhours_extend_until(last, pd.Timestamp("2026-08-06 22:00")) is None
    # Following week
    assert offhours_extend_until(last, pd.Timestamp("2026-08-12 10:32")) is None


def test_offhours_extend_allows_same_evening_after_close() -> None:
    last = pd.Timestamp("2026-08-06 15:55")  # Thursday close
    until = offhours_extend_until(last, pd.Timestamp("2026-08-06 22:00"))
    assert until is not None
    assert until == pd.Timestamp("2026-08-06 22:00")


def test_offhours_extend_weekend_from_friday_close() -> None:
    last = pd.Timestamp("2026-08-07 15:55")  # Friday close
    until = offhours_extend_until(last, pd.Timestamp("2026-08-09 18:00"))  # Sunday
    assert until is not None
    assert until == pd.Timestamp("2026-08-09 18:00")


def test_prices_stale_after_missed_session() -> None:
    last = pd.Timestamp("2026-08-06 10:32")
    assert prices_are_stale(last, pd.Timestamp("2026-08-06 22:00"))
    assert prices_are_stale(last, pd.Timestamp("2026-08-12 10:32"))
    friday_close = pd.Timestamp("2026-08-07 15:55")
    assert not prices_are_stale(friday_close, pd.Timestamp("2026-08-09 18:00"))


def test_fill_ohlc_nans_uses_cache_then_flat() -> None:
    o = np.array([[1.0, np.nan], [1.1, np.nan]], dtype=np.float64)
    spy = np.array([10.0, np.nan], dtype=np.float64)
    filled_o, filled_spy = _fill_ohlc_nans(o, spy)
    assert filled_o.shape == (2, 2)
    assert abs(float(filled_o[0, 1]) - 1.0) < 1e-9  # empty col → 1.0
    assert abs(float(filled_spy[1]) - 10.0) < 1e-9
    assert filled_spy.ndim == 1


def test_last_invested_shadow_weights_skips_reset_stub(tmp_path) -> None:
    from rlbot.forward_live import last_invested_shadow_weights

    path = tmp_path / "shadow_ledger_RLModel.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"target_weights": {"CASH": 0.1, "GOLD": 0.9}, "note": null}',
                '{"target_weights": {"CASH": 1.0}, "note": "Reset to 100k cash (flat paper book)."}',
                '{"target_weights": {"CASH": 0.06, "GOLD": 0.2, "OIL": 0.2}, "note": null}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    w = last_invested_shadow_weights(path)
    assert w is not None
    assert abs(w["CASH"] - 0.06) < 1e-9
    assert abs(w["GOLD"] - 0.2) < 1e-9

    path.write_text(
        '{"target_weights": {"CASH": 1.0}, "note": "Reset to 100k cash (flat paper book)."}\n',
        encoding="utf-8",
    )
    assert last_invested_shadow_weights(path) is None


def test_portfolio_ohlc_switches_weights_mid_series() -> None:
    open_ = np.array([[100.0], [100.0], [110.0]], dtype=np.float64)
    high = open_.copy()
    low = open_.copy()
    close = np.array([[100.0], [110.0], [121.0]], dtype=np.float64)
    cash = _weight_vector({"Cash": 1.0}, ["Cash", "A"])
    long_a = _weight_vector({"Cash": 0.0, "A": 1.0}, ["Cash", "A"])
    W = np.vstack([cash, long_a, long_a])
    candles = portfolio_ohlc_candles(
        open_, high, low, close, W, initial_cash=100_000.0, cash_yield_per_bar=0.0
    )
    assert abs(candles[0, 3] - 100_000.0) < 1e-6
    # Rebalance to 100% A at bar 1 open; close 110 vs prior close 100.
    assert abs(candles[1, 3] - 110_000.0) < 1e-6
    assert abs(candles[2, 3] / candles[1, 3] - 121.0 / 110.0) < 1e-9


def test_mtm_from_start_is_flat_cash_until_start() -> None:
    from rlbot.forward_live import mtm_ohlc_from_start

    open_ = np.array([[100.0], [100.0], [110.0]], dtype=np.float64)
    high = open_.copy()
    low = open_.copy()
    close = np.array([[100.0], [100.0], [110.0]], dtype=np.float64)
    long_a = _weight_vector({"Cash": 0.0, "A": 1.0}, ["Cash", "A"])
    W = np.vstack([long_a, long_a, long_a])
    times = pd.DatetimeIndex(["2026-08-14 15:55", "2026-08-17 09:30", "2026-08-17 09:35"])
    candles = mtm_ohlc_from_start(
        open_,
        high,
        low,
        close,
        W,
        times=times,
        start=pd.Timestamp("2026-08-17 09:30"),
        initial_cash=100_000.0,
        cash_yield_per_bar=0.001,
    )
    assert abs(candles[0, 3] - 100_000.0) < 1e-6
    assert abs(candles[1, 0] - 100_000.0) < 1e-6


def test_lots_ohlc_marks_held_shares_not_rebalanced_weights() -> None:
    from rlbot.forward_live import lots_ohlc_candles

    # One share of A bought at the second bar; cash leftover 50k.
    open_ = np.array([[100.0], [100.0], [110.0]], dtype=np.float64)
    high = open_.copy()
    low = open_.copy()
    close = np.array([[100.0], [100.0], [110.0]], dtype=np.float64)
    times = pd.DatetimeIndex(["2026-08-03 09:30", "2026-08-03 12:05", "2026-08-04 09:30"])
    candles = lots_ohlc_candles(
        open_,
        high,
        low,
        close,
        cash=50_000.0,
        quantities=np.array([500.0]),
        times=times,
        start=pd.Timestamp("2026-08-03 12:05"),
        initial_cash=100_000.0,
    )
    assert abs(candles[0, 3] - 100_000.0) < 1e-6
    assert abs(candles[1, 3] - 100_000.0) < 1e-6  # 50k cash + 500*100
    assert abs(candles[2, 3] - 105_000.0) < 1e-6  # 50k + 500*110


def test_weight_matrix_from_events_cash_until_start_then_switches() -> None:
    from rlbot.forward_live import weight_matrix_from_events

    times = pd.DatetimeIndex(
        [
            "2026-08-14 15:55",
            "2026-08-17 09:30",
            "2026-08-18 15:37",
            "2026-08-18 15:40",
            "2026-08-18 15:45",
        ]
    )
    events = [
        (pd.Timestamp("2026-08-14 16:00"), {"CASH": 1.0, "GOLD": 0.0, "OIL": 0.0}),
        (pd.Timestamp("2026-08-18 15:37"), {"CASH": 0.0, "GOLD": 1.0, "OIL": 0.0}),
        (pd.Timestamp("2026-08-18 15:40"), {"CASH": 0.0, "GOLD": 0.0, "OIL": 1.0}),
    ]
    W = weight_matrix_from_events(
        times,
        events,
        ["Cash", "GOLD", "OIL"],
        start=pd.Timestamp("2026-08-17 09:30"),
    )
    assert abs(W[0, 0] - 1.0) < 1e-9  # before start: cash
    assert abs(W[1, 0] - 1.0) < 1e-9  # after start, before first record: cash
    assert abs(W[2, 1] - 1.0) < 1e-9  # recorded GOLD book
    assert abs(W[3, 2] - 1.0) < 1e-9  # later ledger row switches the book
    assert abs(W[4, 2] - 1.0) < 1e-9


def test_resolve_live_model_start_after_close() -> None:
    from rlbot.forward_live import resolve_live_model_start

    existing = {"live_model_start": "2026-08-17T09:30"}
    ts = resolve_live_model_start(existing, None)
    assert str(ts.date()) == "2026-08-17"
    assert ts.hour == 9 and ts.minute == 30


def test_paper_lots_start_ignores_later_mark_timestamp() -> None:
    from rlbot.forward_live import paper_lots_start_from_state

    fill = paper_lots_start_from_state(
        {
            "last_trade_date": "2026-08-03",
            "updated_at_utc": "2026-08-03T16:04:26+00:00",
        }
    )
    assert fill is not None
    assert str(fill.date()) == "2026-08-03"
    assert fill.hour == 12 and fill.minute == 4

    later_mark = paper_lots_start_from_state(
        {
            "last_trade_date": "2026-08-03",
            "updated_at_utc": "2026-08-18T20:30:00+00:00",
        }
    )
    assert later_mark is not None
    assert str(later_mark.date()) == "2026-08-03"
    assert later_mark.hour == 9 and later_mark.minute == 30

    after_hours_same_day = paper_lots_start_from_state(
        {
            "last_trade_date": "2026-08-20",
            "updated_at_utc": "2026-08-20T21:24:00+00:00",
        }
    )
    assert after_hours_same_day is not None
    assert str(after_hours_same_day.date()) == "2026-08-20"
    assert after_hours_same_day.hour == 9 and after_hours_same_day.minute == 30
