"""Read-only adapter for the locked ``GeneralEquity1/`` pack.

Adds the pack directory to ``sys.path`` and imports pack modules as-is — never
edits pack files. Paper / forward marks use the pack's ``paper_plan`` book.
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any

from rlbot.run_artifacts import PROJECT_ROOT

PACK_DIR = PROJECT_ROOT / "GeneralEquity1"
STRATEGY_ID = "prod_return_alpha_v3"
PAPER_RUN_ID = "GENERAL_EQUITY1"
DEFAULT_INITIAL_CASH = 100_000.0
LEGACY_PAPER_RUN_IDS = frozenset(
    {"GENERAL_EQUITY", "PROD_RETURN_ALPHA", "FINALMODEL"}
)
_STRATEGY_MOD_NAME = "_pack_general_equity1_strategy"


def _ensure_pack_on_path() -> None:
    pack = PACK_DIR.resolve()
    if not pack.is_dir():
        raise FileNotFoundError(f"GeneralEquity1 pack missing at {pack}")
    root = str(pack)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_strategy() -> Any:
    if _STRATEGY_MOD_NAME in sys.modules:
        return sys.modules[_STRATEGY_MOD_NAME]
    _ensure_pack_on_path()
    path = PACK_DIR.resolve() / "strategy.py"
    spec = importlib.util.spec_from_file_location(_STRATEGY_MOD_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_STRATEGY_MOD_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


def locked_params() -> Any:
    return load_strategy().P


def paper_plan(*, aum: float = DEFAULT_INITIAL_CASH) -> dict[str, Any]:
    """Exact pack paper plan (JSON-serializable dict)."""
    ge = load_strategy()
    dates, px = ge.load_panel(["SPY", "QQQ", "TQQQ", "BIL", "GLD", ge.P.dual_b])
    tqqq_ohlc = ge.load_ohlc("TQQQ", dates)
    qqq_ohlc = ge.load_ohlc("QQQ", dates)
    return ge.paper_plan(dates, px, tqqq_ohlc, qqq_ohlc, ge.P, aum=float(aum))


def latest_portfolio_weights(*, aum: float = DEFAULT_INITIAL_CASH) -> dict[str, float]:
    """Lag-1 portfolio weights + residual cash (BIL slot folded into Cash)."""
    plan = paper_plan(aum=aum)
    pt = plan.get("portfolio_targets") or {}
    tw: dict[str, float] = {}
    cash = 0.0
    for k, v in pt.items():
        key = str(k)
        w = float(v)
        if key.startswith("BIL") or key.upper() == "CASH":
            cash += max(0.0, w)
            continue
        if w > 0:
            tw[key.upper()] = w
    # Pack already reports BIL residual; only top-up if weights undershoot 1.
    cash = max(cash, max(0.0, 1.0 - sum(tw.values())))
    tw["CASH"] = cash
    s = sum(tw.values())
    if s <= 1e-12:
        return {"CASH": 1.0}
    if abs(s - 1.0) > 1e-6:
        return {k: float(v) / s for k, v in tw.items()}
    return {k: float(v) for k, v in tw.items()}


def to_yahoo_symbol(sym: str) -> str:
    s = str(sym).strip().upper()
    return s.replace(".", "-") if s else s
