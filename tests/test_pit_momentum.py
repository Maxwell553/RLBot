"""Locked PIT momentum signal engine + paper helpers (no network)."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from rlbot.pit_momentum import (
    Params,
    compute_target_weights,
    execution_date,
    last_actionable_signal,
    load_pit_snapshots,
    membership_asof,
    month_end_signals,
    orders_to_targets,
    to_yahoo_symbol,
    weights_with_cash,
)


def _synth_cal(n: int = 300, start: date = date(2020, 1, 2)) -> list[date]:
    cal: list[date] = []
    d = start
    while len(cal) < n:
        if d.weekday() < 5:
            cal.append(d)
        d += timedelta(days=1)
    return cal


def _synth_prices(cal: list[date], drift: float) -> list[tuple[date, float]]:
    px = 100.0
    out: list[tuple[date, float]] = []
    for i, day in enumerate(cal):
        px *= 1.0 + drift + (0.001 if (i % 17 == 0) else 0.0)
        out.append((day, px))
    return out


def test_to_yahoo_symbol_maps_dots() -> None:
    assert to_yahoo_symbol("BRK.B") == "BRK-B"
    assert to_yahoo_symbol("AAPL") == "AAPL"


def test_month_end_and_execution_lag() -> None:
    cal = _synth_cal(60)
    signals = month_end_signals(cal)
    assert signals
    # Last day of each month in the calendar.
    for sig in signals:
        i = cal.index(sig)
        assert i + 1 == len(cal) or cal[i + 1].month != sig.month
    trade = execution_date(signals[0], cal, lag=1)
    assert trade is not None
    assert cal.index(trade) == cal.index(signals[0]) + 1


def test_compute_target_weights_picks_top_momentum(tmp_path: Path) -> None:
    cal = _synth_cal(250)
    prices = {
        "SPY": _synth_prices(cal, 0.0003),
        "AAA": _synth_prices(cal, 0.0015),
        "BBB": _synth_prices(cal, 0.0001),
        "CCC": _synth_prices(cal, 0.0010),
        "DDD": _synth_prices(cal, 0.0005),
        "EEE": _synth_prices(cal, 0.0008),
    }
    pit = tmp_path / "pit.csv"
    pit.write_text(
        'date,tickers\n2020-01-01,"SPY,AAA,BBB,CCC,DDD,EEE"\n', encoding="utf-8"
    )
    sig = month_end_signals(cal)[-1]
    params = Params(top_n=2, lookback_trading_days=60, skip_trading_days=5)
    w = compute_target_weights(sig, prices, pit, params=params)
    assert len(w) == 2
    assert abs(sum(w.values()) - 0.9) < 1e-9
    # Highest drifts should win.
    assert "AAA" in w
    assert "BBB" not in w
    cash = weights_with_cash(w)
    assert abs(cash["CASH"] - 0.1) < 1e-9


def test_fallback_when_thin_eligibility(tmp_path: Path) -> None:
    cal = _synth_cal(250)
    prices = {"SPY": _synth_prices(cal, 0.0004), "AAA": _synth_prices(cal, 0.001)}
    pit = tmp_path / "pit.csv"
    pit.write_text('date,tickers\n2020-01-01,"AAA"\n', encoding="utf-8")
    sig = month_end_signals(cal)[-1]
    # top_n=30 → need max(5, 15)=15 valid scores; only 1 eligible → SPY fallback
    w = compute_target_weights(sig, prices, pit, params=Params(top_n=30))
    assert w == {"SPY": 0.9}


def test_orders_sells_before_buys() -> None:
    intents = orders_to_targets(
        equity=100_000.0,
        positions={"AAA": 100.0, "BBB": 50.0},
        marks={"AAA": 100.0, "BBB": 50.0, "CCC": 25.0},
        targets={"CCC": 0.9},
        min_notional=1.0,
    )
    sides = [i["side"] for i in intents]
    assert sides.index("sell") < sides.index("buy")
    assert any(i["symbol"] == "AAA" and i["side"] == "sell" for i in intents)


def test_pit_csv_loads_and_membership(tmp_path: Path) -> None:
    # Tiny fixture — avoid parsing the full multi-MB PIT CSV in unit tests.
    pit = tmp_path / "pit.csv"
    pit.write_text(
        'date,tickers\n'
        '2020-01-01,"AAPL,MSFT,DEAD-202006"\n'
        '2020-06-01,"AAPL,MSFT,TSLA"\n',
        encoding="utf-8",
    )
    snaps = load_pit_snapshots(pit)
    assert len(snaps) == 2
    mem = membership_asof(snaps, date(2020, 6, 30))
    assert "AAPL" in mem and "TSLA" in mem
    assert "DEAD-202006" not in mem


def test_last_actionable_signal_respects_lag() -> None:
    cal = _synth_cal(120)
    signals = month_end_signals(cal)
    # Prefer a signal that still has a lag+1 trade session in-calendar.
    sig = next(s for s in reversed(signals) if execution_date(s, cal, lag=1) is not None)
    trade = execution_date(sig, cal, lag=1)
    assert trade is not None
    # as_of = signal day → that month's trade is not yet due
    pair_on_signal = last_actionable_signal(cal, sig, lag=1)
    if pair_on_signal is not None:
        assert pair_on_signal[1] <= sig
        assert pair_on_signal[0] < sig
    pair2 = last_actionable_signal(cal, trade, lag=1)
    assert pair2 == (sig, trade)
