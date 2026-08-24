"""Lite API allocation panel must follow the live shadow ledger, not a cash stub."""

from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path


def _load_lite() -> object:
    spec = importlib.util.spec_from_file_location(
        "frontend_api_lite_test",
        Path(__file__).resolve().parents[1] / "scripts" / "frontend_api_lite.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_attach_live_allocations_follows_ledger_not_cash_stub(tmp_path: Path) -> None:
    exec_dir = tmp_path / "execution"
    exec_dir.mkdir()
    (exec_dir / "shadow_ledger_RLModel.jsonl").write_text(
        "\n".join(
            [
                '{"target_weights": {"CASH": 1.0}, "note": "Reset to 100k cash (flat paper book)."}',
                '{"target_weights": {"CASH": 0.063, "GOLD": 0.2, "OIL": 0.2, "COPPER": 0.2, "EM": 0.2, "EURUSD": 0.056, "SP500": 0.081}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    mod = _load_lite()
    mod.ROOT = tmp_path
    mod.EXEC = exec_dir
    payload = {
        "initial_cash": 100_000.0,
        "stats": {"live_model": {"nav": 103_899.0}, "model": {"nav": 103_000.0}},
        "nav": {
            "live_model": [100_000.0, 103_899.0],
            "model": [100_000.0, 103_000.0],
        },
        "latest_weights": {"Cash": 1.0},
        "allocations": {
            "live_model": {
                "nav": 100_052.0,
                "positions": [
                    {"label": "Cash", "ticker": "CASH", "weight": 1.0, "value_usd": 100_052.0}
                ],
                "latest_weights": {"CASH": 1.0},
            }
        },
        "live": {"as_of_utc": "2026-08-18T19:54:00Z"},
    }
    out = mod._attach_live_allocations(payload)
    book = out["allocations"]["live_model"]
    assert abs(float(book["nav"]) - 103_899.0) < 1e-6
    cash = next(p for p in book["positions"] if str(p["label"]).lower() == "cash")
    assert float(cash["weight"]) < 0.2
    gold = next(p for p in book["positions"] if p["ticker"] == "GOLD")
    assert abs(float(gold["weight"]) - 0.2) < 1e-9


def test_jsonl_last_weights_skips_reset_stub(tmp_path: Path) -> None:
    exec_dir = tmp_path / "execution"
    exec_dir.mkdir()
    (exec_dir / "shadow_ledger_RLModel.jsonl").write_text(
        '{"target_weights": {"CASH": 0.1, "GOLD": 0.9}, "note": null}\n'
        '{"target_weights": {"CASH": 1.0}, "note": "Reset to 100k cash (flat paper book)."}\n',
        encoding="utf-8",
    )
    mod = _load_lite()
    mod.ROOT = tmp_path
    mod.EXEC = exec_dir
    w = mod._jsonl_last_weights("RLModel")
    assert w is not None
    assert abs(w["GOLD"] - 0.9) < 1e-9


def test_weight_rows_on_clock_starts_cash_then_follows_ledger(tmp_path: Path) -> None:
    mod = _load_lite()
    clock = [
        datetime(2026, 8, 14, 15, 55),
        datetime(2026, 8, 17, 9, 30),
        datetime(2026, 8, 18, 15, 37),
        datetime(2026, 8, 18, 15, 40),
        datetime(2026, 8, 18, 15, 45),
    ]
    events = [
        (datetime(2026, 8, 18, 15, 37), {"CASH": 0.0, "GOLD": 1.0}),
        (datetime(2026, 8, 18, 15, 40), {"CASH": 0.0, "OIL": 1.0}),
    ]
    rows = mod._weight_rows_on_clock(
        clock,
        events,
        ["Cash", "GOLD", "OIL"],
        start=datetime(2026, 8, 17, 9, 30),
    )
    assert abs(rows[0][0] - 1.0) < 1e-9
    assert abs(rows[1][0] - 1.0) < 1e-9
    assert abs(rows[2][1] - 1.0) < 1e-9
    assert abs(rows[3][2] - 1.0) < 1e-9
    assert abs(rows[4][2] - 1.0) < 1e-9


def test_attach_live_allocations_keeps_ge1_lot_weights() -> None:
    mod = _load_lite()
    payload = {
        "initial_cash": 100_000.0,
        "stats": {"model": {"nav": 101_500.0}, "live_model": {"nav": 100_000.0}},
        "nav": {"model": [100_000.0, 101_500.0], "live_model": [100_000.0, 100_000.0]},
        "latest_weights": {"Cash": 0.47, "GLD": 0.29, "QQQ": 0.05, "TQQQ": 0.19},
        "positions": [
            {"label": "Cash", "ticker": "CASH", "weight": 0.47, "value_usd": 47_705.0, "price": 1.0},
            {"label": "GLD", "ticker": "GLD", "weight": 0.29, "value_usd": 29_435.0, "price": 385.0},
        ],
        "live": {"as_of_utc": "2026-08-18T20:00:00Z"},
    }
    out = mod._attach_live_allocations(payload)
    book = out["allocations"]["model"]
    assert book["label"] == "GeneralEquity1"
    assert book["run_id"] == "GENERAL_EQUITY1"
    assert abs(float(book["latest_weights"]["GLD"]) - 0.29) < 1e-9
    assert abs(float(book["nav"]) - 101_500.0) < 1e-6
    assert book["positions"][1]["ticker"] == "GLD"


def test_attach_live_allocations_core_equity_does_not_reuse_ge1_lots() -> None:
    mod = _load_lite()
    payload = {
        "initial_cash": 100_000.0,
        "stats": {
            "model": {"nav": 101_500.0},
            "core_equity": {"nav": 100_800.0},
            "live_model": {"nav": 100_000.0},
        },
        "nav": {
            "model": [100_000.0, 101_500.0],
            "core_equity": [100_000.0, 100_800.0],
            "live_model": [100_000.0, 100_000.0],
        },
        "latest_weights": {"Cash": 0.47, "GLD": 0.29, "QQQ": 0.05, "TQQQ": 0.19},
        "positions": [
            {"label": "Cash", "ticker": "CASH", "weight": 0.47, "value_usd": 47_705.0, "price": 1.0},
            {"label": "TQQQ", "ticker": "TQQQ", "weight": 0.19, "value_usd": 19_285.0, "price": 70.0},
        ],
        "core_equity_weights": {"CASH": 0.42, "QQQ": 0.38, "GLD": 0.12, "TLT": 0.08},
        "core_equity_positions": [
            {"label": "Cash", "ticker": "CASH", "weight": 0.42, "value_usd": 42_336.0, "price": 1.0},
            {"label": "QQQ", "ticker": "QQQ", "weight": 0.38, "value_usd": 38_304.0, "price": 510.0},
            {"label": "GLD", "ticker": "GLD", "weight": 0.12, "value_usd": 12_096.0, "price": 385.0},
            {"label": "TLT", "ticker": "TLT", "weight": 0.08, "value_usd": 8_064.0, "price": 90.0},
        ],
        "live": {"as_of_utc": "2026-08-18T20:00:00Z"},
    }
    out = mod._attach_live_allocations(payload)
    ge1 = out["allocations"]["model"]
    ce = out["allocations"]["core_equity"]
    assert ge1["run_id"] == "GENERAL_EQUITY1"
    assert any(str(p.get("ticker")).upper() == "TQQQ" for p in ge1["positions"])
    assert ce["label"] == "CoreEquity"
    assert ce["run_id"] == "CORE_EQUITY"
    assert abs(float(ce["nav"]) - 100_800.0) < 1e-6
    tickers = {str(p.get("ticker")).upper() for p in ce["positions"]}
    assert "TQQQ" not in tickers
    assert "QQQ" in tickers
    assert "TLT" in tickers


def test_paper_share_book_ignores_later_mark_timestamp(tmp_path: Path) -> None:
    exec_dir = tmp_path / "execution"
    paper = exec_dir / "paper_general_equity1"
    paper.mkdir(parents=True)
    (paper / "state.json").write_text(
        """{
          "cash": 47843.51,
          "positions": {"GLD": 76.34, "QQQ": 6.81, "TQQQ": 284.19},
          "last_trade_date": "2026-08-03",
          "updated_at_utc": "2026-08-18T20:30:00+00:00"
        }
        """,
        encoding="utf-8",
    )
    mod = _load_lite()
    mod.EXEC = exec_dir
    cash, lots, start = mod._paper_share_book("GENERAL_EQUITY1")
    assert abs(cash - 47843.51) < 1e-6
    assert abs(lots["GLD"] - 76.34) < 1e-9
    assert start is not None
    assert start.date().isoformat() == "2026-08-03"
    assert start.hour == 9 and start.minute == 30


def test_paper_share_book_same_day_after_hours_uses_session_open(tmp_path: Path) -> None:
    exec_dir = tmp_path / "execution"
    paper = exec_dir / "paper_core_equity"
    paper.mkdir(parents=True)
    (paper / "state.json").write_text(
        """{
          "cash": 100.0,
          "positions": {"QQQ": 10.0, "GLD": 5.0, "BIL": 3.0},
          "last_trade_date": "2026-08-20",
          "updated_at_utc": "2026-08-20T21:24:00+00:00"
        }
        """,
        encoding="utf-8",
    )
    mod = _load_lite()
    mod.EXEC = exec_dir
    cash, lots, start = mod._paper_share_book("CORE_EQUITY")
    assert abs(cash - 100.0) < 1e-6
    assert lots["QQQ"] == 10.0
    assert start is not None
    assert start.date().isoformat() == "2026-08-20"
    assert start.hour == 9 and start.minute == 30


def test_ensure_core_equity_nav_fills_missing_series() -> None:
    mod = _load_lite()
    payload = {
        "initial_cash": 100_000.0,
        "nav": {
            "model": [100_000.0, 101_000.0, 102_000.0],
            "spy": [100_000.0, 100_500.0, 101_000.0],
        },
        "timestamps": [
            "2026-08-18T09:30",
            "2026-08-18T09:35",
            "2026-08-18T09:40",
        ],
        "stats": {"model": {"nav": 102_000.0}},
        "positions": [
            {"label": "QQQ", "ticker": "QQQ", "weight": 0.5, "price": 500.0},
        ],
    }
    out = mod._ensure_core_equity_nav(payload)
    series = out["nav"]["core_equity"]
    assert len(series) == 3
    assert all(abs(v - 100_000.0) < 1e-6 for v in series)
    assert out["stats"]["core_equity"]["nav"] == 100_000.0
    assert len(out["candles"]["core_equity"]) == 3
    again = mod._ensure_core_equity_nav(out)
    assert again["nav"]["core_equity"] is series


def test_attach_live_allocations_injects_core_equity_nav() -> None:
    mod = _load_lite()
    payload = {
        "initial_cash": 100_000.0,
        "nav": {"model": [100_000.0, 101_500.0], "live_model": [100_000.0, 100_000.0]},
        "timestamps": ["2026-08-18T09:30", "2026-08-18T15:55"],
        "stats": {"model": {"nav": 101_500.0}, "live_model": {"nav": 100_000.0}},
        "latest_weights": {"Cash": 0.47, "GLD": 0.29, "QQQ": 0.05, "TQQQ": 0.19},
        "positions": [
            {"label": "Cash", "ticker": "CASH", "weight": 0.47, "value_usd": 47_705.0, "price": 1.0},
            {"label": "TQQQ", "ticker": "TQQQ", "weight": 0.19, "value_usd": 19_285.0, "price": 70.0},
        ],
        "live": {"as_of_utc": "2026-08-18T20:00:00Z"},
    }
    out = mod._attach_live_allocations(payload)
    assert len(out["nav"]["core_equity"]) == 2
    assert out["allocations"]["core_equity"]["run_id"] == "CORE_EQUITY"
    assert out["allocations"]["model"]["run_id"] == "GENERAL_EQUITY1"
