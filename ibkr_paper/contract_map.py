"""Build ib_insync Contract objects from contracts.json (logical ticker -> tradable)."""

from __future__ import annotations

# ib_insync/eventkit: ensure main-thread loop (Python 3.10+)
import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import json
from pathlib import Path
from typing import Any

from ib_insync import Forex, Stock

HERE = Path(__file__).resolve().parent
DEFAULT_PATH = HERE / "contracts.json"


def _load_leg_list(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or DEFAULT_PATH
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return data["legs"]


def build_contract(leg: dict[str, Any]) -> Contract:
    k = leg["kind"]
    if k == "STK":
        return Stock(
            leg["symbol"],
            leg.get("exchange", "SMART"),
            leg.get("currency", "USD"),
        )
    if k == "CASH":
        b = leg["base"]
        q = leg["quote"]
        pair = f"{b}{q}"
        if len(pair) != 6:
            raise ValueError(f"Forex pair must be 6 chars (e.g. EURUSD), got {pair!r}")
        return Forex(pair)
    raise ValueError(f"Unknown leg kind: {k}")


def load_contracts(
    path: Path | None = None,
) -> tuple[tuple[str, ...], list[Contract]]:
    """Return (TICKER order, contracts) in the same order as data_utils.TICKERS."""
    from data_utils import TICKERS

    legs = _load_leg_list(path)
    by_logical: dict[str, dict[str, Any]] = {x["logical"]: x for x in legs}
    order: list[str] = []
    contracts: list[Contract] = []
    for name in TICKERS:
        if name not in by_logical:
            raise KeyError(
                f"contracts.json missing logical leg {name!r} (data_utils has {TICKERS})"
            )
        order.append(name)
        contracts.append(build_contract(by_logical[name]))
    return (tuple(order), contracts)
