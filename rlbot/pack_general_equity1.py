"""Read-only adapter for the locked ``GeneralEquity1/`` pack.

Adds the pack directory to ``sys.path`` and imports pack modules as-is — never
edits pack files. Paper / forward marks use the pack's ``paper_plan`` book.

The pack does ``from numba import njit``. On iCloud Desktop the venv's numba
tree is often ``SF_DATALESS`` (evicted); opening those files hangs until
``TimeoutError: [Errno 60]``. When numba is missing or unreadable, this adapter
installs a passthrough ``njit`` so the pack loads and the Python bodies run.
"""

from __future__ import annotations

import importlib.util
import os
import stat
import sys
import types
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from rlbot.run_artifacts import PROJECT_ROOT

PACK_DIR = PROJECT_ROOT / "GeneralEquity1"
STRATEGY_ID = "prod_return_alpha_v3"
PAPER_RUN_ID = "GENERAL_EQUITY1"
DEFAULT_INITIAL_CASH = 100_000.0
LEGACY_PAPER_RUN_IDS = frozenset(
    {"GENERAL_EQUITY", "PROD_RETURN_ALPHA", "FINALMODEL"}
)
_STRATEGY_MOD_NAME = "_pack_general_equity1_strategy"
_NUMBA_STUB_FILE = "<markettrainer-numba-stub>"
_SF_DATALESS = int(getattr(stat, "SF_DATALESS", 0))


def _path_is_dataless(path: Path | str) -> bool:
    """True when macOS iCloud has evicted the file (open() would hang)."""
    if not _SF_DATALESS:
        return False
    try:
        flags = int(getattr(os.stat(path), "st_flags", 0) or 0)
    except OSError:
        return False
    return bool(flags & _SF_DATALESS)


def _purge_dataless_pyc(root: Path) -> int:
    """Drop iCloud-evicted bytecode so import reads the hydrated ``.py`` files."""
    n = 0
    cache = root / "__pycache__"
    if not cache.is_dir():
        return 0
    for path in cache.glob("*.pyc"):
        if not _path_is_dataless(path):
            continue
        try:
            path.unlink()
            n += 1
        except OSError:
            continue
    return n


def njit_passthrough(*args: Any, **kwargs: Any) -> Any:
    """Stand-in for ``numba.njit``: leave the function as ordinary Python."""
    del kwargs
    if args and callable(args[0]):
        return args[0]

    def deco(fn: Any) -> Any:
        return fn

    return deco


def _install_numba_stub() -> types.ModuleType:
    mod = sys.modules.get("numba")
    if isinstance(mod, types.ModuleType) and getattr(mod, "__file__", None) == _NUMBA_STUB_FILE:
        return mod
    stub = types.ModuleType("numba")
    stub.__file__ = _NUMBA_STUB_FILE
    stub.njit = njit_passthrough  # type: ignore[attr-defined]
    sys.modules["numba"] = stub
    return stub


def _numba_usable() -> bool:
    existing = sys.modules.get("numba")
    if existing is not None and getattr(existing, "__file__", None) == _NUMBA_STUB_FILE:
        return False
    if existing is not None and hasattr(existing, "njit"):
        origin = getattr(existing, "__file__", None)
        if origin and _path_is_dataless(origin):
            return False
        return True
    try:
        spec = importlib.util.find_spec("numba")
    except (ImportError, ValueError, TimeoutError, OSError):
        return False
    if spec is None or not spec.origin:
        return False
    origin = Path(spec.origin)
    try:
        if not origin.is_file() or _path_is_dataless(origin):
            return False
    except OSError:
        return False
    return True


def ensure_numba_for_pack() -> str:
    """Make ``from numba import njit`` succeed without touching evicted files.

    Returns ``"numba"`` or ``"stub"``.
    """
    if _numba_usable():
        return "numba"
    _install_numba_stub()
    return "stub"


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
    ensure_numba_for_pack()
    _purge_dataless_pyc(PACK_DIR.resolve())
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


def paper_plan(
    *,
    aum: float = DEFAULT_INITIAL_CASH,
    dates: list[date] | None = None,
    closes: dict[str, np.ndarray] | None = None,
    ohlc: dict[str, tuple[np.ndarray, ...]] | None = None,
) -> dict[str, Any]:
    """Pack ``paper_plan`` on a live Yahoo panel when provided, else frozen bars.db."""
    ge = load_strategy()
    dual = str(ge.P.dual_b)
    need = ["SPY", "QQQ", "TQQQ", "BIL", "GLD", dual]
    if dates is None or closes is None or ohlc is None:
        dates, px = ge.load_panel(need)
        tqqq_ohlc = ge.load_ohlc("TQQQ", dates)
        qqq_ohlc = ge.load_ohlc("QQQ", dates)
        source = "bars.db"
    else:
        px = {}
        for sym in need:
            if sym not in closes:
                raise ValueError(f"live panel missing {sym}")
            px[sym] = np.asarray(closes[sym], dtype=np.float64)
        tqqq_ohlc = tuple(np.asarray(x, dtype=np.float64) for x in ohlc["TQQQ"])
        qqq_ohlc = tuple(np.asarray(x, dtype=np.float64) for x in ohlc["QQQ"])
        source = "yahoo"
    plan = ge.paper_plan(dates, px, tqqq_ohlc, qqq_ohlc, ge.P, aum=float(aum))
    plan["data_source"] = source
    return plan


def latest_portfolio_weights(
    *,
    aum: float = DEFAULT_INITIAL_CASH,
    dates: list[date] | None = None,
    closes: dict[str, np.ndarray] | None = None,
    ohlc: dict[str, tuple[np.ndarray, ...]] | None = None,
) -> dict[str, float]:
    """Lag-1 portfolio weights + residual cash (BIL slot folded into Cash)."""
    plan = paper_plan(aum=aum, dates=dates, closes=closes, ohlc=ohlc)
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
