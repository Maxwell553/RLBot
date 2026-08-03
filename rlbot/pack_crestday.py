"""Read-only adapter for the locked ``CrestDay/`` pack.

Adds the pack directory to ``sys.path`` and imports pack modules as-is — never
edits pack files. Soft forward companion NAV uses the pack ``run()`` equity curve.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rlbot.run_artifacts import PROJECT_ROOT

PACK_DIR = PROJECT_ROOT / "CrestDay"
STRATEGY_ID = "crestday"
PAPER_RUN_ID = "CREST_DAY"
DEFAULT_INITIAL_CASH = 100_000.0
CACHE_DIR = PROJECT_ROOT / "execution" / "paper_crest_day"
NAV_CACHE_PATH = CACHE_DIR / "nav_series.npz"
NAV_CACHE_META = CACHE_DIR / "nav_series_meta.json"
NAV_CACHE_TTL_S = 300.0


def _ensure_pack_on_path() -> None:
    pack = PACK_DIR.resolve()
    if not pack.is_dir():
        raise FileNotFoundError(f"CrestDay pack missing at {pack}")
    root = str(pack)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_engine() -> Any:
    _ensure_pack_on_path()
    import crestday_engine  # type: ignore[import-not-found]

    return crestday_engine


def latest_intents(
    *, aum: float = DEFAULT_INITIAL_CASH, equity: float | None = None
) -> dict[str, Any]:
    eng = load_engine()
    return eng.latest_intents(eng.P_LOCK, aum=float(aum), equity=equity)


def simulate_nav_series(
    *,
    force_refresh: bool = False,
    initial_cash: float = DEFAULT_INITIAL_CASH,
    since: Any | None = None,
    cache_ttl_s: float = NAV_CACHE_TTL_S,
) -> dict[str, Any] | None:
    """Pack backtest equity on bar timestamps (for soft ``nav.crypto`` alignment)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    now = time.time()
    if not force_refresh and NAV_CACHE_PATH.is_file() and NAV_CACHE_META.is_file():
        try:
            meta = json.loads(NAV_CACHE_META.read_text(encoding="utf-8"))
            age = now - float(meta.get("cached_at_unix") or 0.0)
            if age <= float(cache_ttl_s) and abs(
                float(meta.get("aum") or 0.0) - float(initial_cash)
            ) < 1.0:
                z = np.load(NAV_CACHE_PATH, allow_pickle=False)
                nav = np.asarray(z["nav"], dtype=np.float64)
                times = pd.DatetimeIndex(pd.to_datetime(z["times"]))
                if since is not None:
                    cut = pd.Timestamp(since)
                    mask = times >= cut
                    nav, times = nav[mask], times[mask]
                if nav.size >= 2:
                    nav = nav / float(nav[0]) * float(initial_cash)
                    return {"nav": nav, "times": times, "aum": float(initial_cash)}
        except Exception:  # noqa: BLE001
            pass

    eng = load_engine()
    _nest, _m, eq, panel = eng.run(eng.P_LOCK, aum=float(initial_cash))
    idx = panel["BTC"]["index"]
    times = pd.DatetimeIndex(pd.to_datetime(idx)).tz_localize(None)
    nav = np.asarray(eq, dtype=np.float64)
    if nav.size != len(times):
        n = min(nav.size, len(times))
        nav, times = nav[-n:], times[-n:]
    if nav.size < 2 or not np.isfinite(nav[0]) or nav[0] <= 0:
        return None

    try:
        np.savez(
            NAV_CACHE_PATH,
            nav=nav.astype(np.float64),
            times=np.asarray(times.astype("datetime64[ns]")),
        )
        NAV_CACHE_META.write_text(
            json.dumps(
                {
                    "cached_at_unix": now,
                    "aum": float(initial_cash),
                    "n": int(nav.size),
                    "strategy_id": STRATEGY_ID,
                    "pack": "CrestDay",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError:
        pass

    if since is not None:
        cut = pd.Timestamp(since)
        mask = times >= cut
        nav, times = nav[mask], times[mask]
    if nav.size < 2:
        return None
    return {"nav": nav, "times": times, "aum": float(initial_cash)}
