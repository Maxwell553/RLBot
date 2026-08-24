"""Locked CoreEquity signals via the pack (read-only). Do not retune here.

LiveTrader feeds a Yahoo (or IB) daily panel into ``CoreEquity/strategy.py``.
The pack is not copied; frozen ``bars.db`` is never used for live targets.
"""

from __future__ import annotations

from typing import Any

from rlbot.pack_core_equity import (  # noqa: F401
    STRATEGY_ID,
    apply_sleeve_a_to_targets,
    book_symbols,
    latest_targets,
    locked_params,
    panel_symbols,
    sleeve_a_live_state,
    weights_from_targets,
)

P = locked_params()
ProdParams = type(P)
BOOK_SYMBOLS = book_symbols()
PANEL_SYMBOLS = panel_symbols()


def portfolio_weights(targets: dict[str, Any], p: Any = P) -> dict[str, float]:
    """Lag-1 book weights from pack ``latest_targets`` (BIL residual; no TQQQ)."""
    return weights_from_targets(targets, p)
