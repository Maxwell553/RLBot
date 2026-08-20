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
    assert abs(float(book["latest_weights"]["GLD"]) - 0.29) < 1e-9
    assert abs(float(book["nav"]) - 101_500.0) < 1e-6
    assert book["positions"][1]["ticker"] == "GLD"


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
