"""Live forward-mark refresh (torch-free): 5m NAV candles for model / EW / SPY.

The ops dashboard polls ``/api/forward``; that endpoint calls
``refresh_forward_mark_live`` so a new 5-minute candle (and the forming bar)
updates from a throttled yfinance pull without re-running RecurrentPPO.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rlbot.forward_mark import (
    _series_stats,
    build_forward_mark_payload,
    load_forward_mark,
    resolve_active_forward_run_id,
    set_active_forward_run,
    write_forward_mark,
)
from rlbot.rl_config import get_config, load_config, set_config
from rlbot.run_artifacts import PROJECT_ROOT

LIVE_STAMP_NAME = "forward_live_stamp.json"
# Companion RL deploy shown alongside GENERAL_EQUITY1 on /ops/forward.
RL_LIVE_RUN_ID = "LIVE_MODEL"
# Locked GeneralEquity1 pack paper book (TQQQ+QQQ hybrid weekly + GLD/TLT dual).
ALGO_LIVE_RUN_ID = "GENERAL_EQUITY1"
# Soft companion CrestDay series (pack NAV; never blocks equity forward).
CRYPTO_LIVE_RUN_ID = "CREST_DAY"
# Soft companion Durable.v1 (CDE FCM long/short; never blocks equity forward).
DURABLE_LIVE_RUN_ID = "DURABLE_V1"
# Keep CrestDay attach short — a hung pack sim previously pinned the whole
# Yahoo refresh past the lite-API subprocess timeout (~50s) and left prices stale.
CRYPTO_SOFT_TIMEOUT_S = 8.0
# Candle grid is 5m; refresh Yahoo about once per new bar.
DEFAULT_MIN_REFRESH_SECONDS = 300
BAR_INTERVAL = "5m"
# Approx US cash-session 5-minute bars (09:30–16:00 ET → 78).
BARS_PER_TRADING_DAY = 78
# yfinance 5m history is capped (~60d); fetch this much lookback.
INTRADAY_LOOKBACK_DAYS = 59


def _session_date_today() -> str:
    """US/Eastern trading calendar date (naive YYYY-MM-DD)."""
    return str(pd.Timestamp.now(tz="America/New_York").date())


def _resolve_book_start(
    existing: dict[str, Any] | None,
    stamp: dict[str, Any] | None,
    *,
    reset_book: bool = False,
) -> str:
    """Stable live-book start date (YYYY-MM-DD).

    Persists across refreshes so 1W/1M toggles keep prior days. Only resets to
    today's US session when ``reset_book`` is set or no prior start exists.
    On weekends/holidays a reset seeds a flat $initial_cash baseline (no
    replay of the prior session) until the next cash open.
    """
    if reset_book:
        return _session_date_today()
    for src in (existing, stamp):
        if not isinstance(src, dict):
            continue
        for key in ("book_start", "holdout_start"):
            raw = src.get(key)
            if raw:
                try:
                    return str(pd.Timestamp(raw).date())
                except (TypeError, ValueError):
                    continue
    return _session_date_today()


def seed_flat_forward_baseline(
    run_id: str,
    *,
    initial_cash: float | None = None,
    root: Path | None = None,
    include_live_model: bool = True,
) -> dict[str, Any]:
    """Write a single-point flat mark at ``initial_cash`` (all series).

    Used when ``reset_book`` lands on a session with no 5m prints yet (weekend /
    holiday / pre-open) so the UI shows a true $100k baseline instead of
    replaying the prior trading day.
    """
    from rlbot.rl_config import get_config, load_config, set_config

    rid = str(run_id).strip()
    exec_root = (root or PROJECT_ROOT) / "execution"
    set_config(load_config(PROJECT_ROOT / "config" / "config.yaml"))
    cfg = get_config()
    cash = float(
        initial_cash
        if initial_cash is not None
        else cfg.environment.initial_cash
    )
    book_start = _session_date_today()
    # One synthetic bar at session open (US/Eastern); chart reads as flat @ cash.
    ts = pd.Timestamp(f"{book_start} 09:30:00")
    flat = np.asarray([cash], dtype=np.float64)
    flat_ohlc = np.asarray([[cash, cash, cash, cash]], dtype=np.float64)
    candles = {
        "model": _candles_to_payload(pd.DatetimeIndex([ts]), flat_ohlc),
        "equal_weight": _candles_to_payload(pd.DatetimeIndex([ts]), flat_ohlc),
        "spy": _candles_to_payload(pd.DatetimeIndex([ts]), flat_ohlc),
        "crypto": _candles_to_payload(pd.DatetimeIndex([ts]), flat_ohlc),
    }
    nav_live = flat.copy() if include_live_model else None
    if include_live_model:
        candles["live_model"] = _candles_to_payload(pd.DatetimeIndex([ts]), flat_ohlc)

    latest_w = _latest_shadow_weights(rid) or {"Cash": 1.0}
    if not isinstance(latest_w, dict) or not latest_w:
        latest_w = {"Cash": 1.0}
    labels = ["Cash"] + [
        str(k)
        for k, v in latest_w.items()
        if str(k).upper() != "CASH" and float(v) > 0
    ]
    if len(labels) == 1:
        labels = ["Cash"]
    live_vec = _weight_vector(latest_w, labels)
    w_mat = np.asarray([live_vec], dtype=np.float64)

    payload = build_forward_mark_payload(
        run_id=rid,
        checkpoint_label="locked",
        dates=[ts.date()],
        nav_model=flat,
        nav_spy=flat,
        nav_ew=flat,
        weights=w_mat,
        asset_labels=labels,
        initial_cash=cash,
        holdout_start=book_start,
        holdout_end=None,
        note=(
            f"Flat baseline reset at {cash:,.0f} on {book_start} "
            f"(no session bars yet). Live 5m MTM resumes at the next cash open. "
            f"GENERAL_EQUITY1 + CrestDay + LIVE_MODEL companion."
        ),
        bar_interval=BAR_INTERVAL,
        timestamps=[ts.isoformat(timespec="minutes")],
        candles=candles,
        bars_per_year=float(BARS_PER_TRADING_DAY * 252),
        nav_live_model=nav_live,
        nav_crypto=flat,
    )
    payload["book_start"] = book_start
    payload["latest_weights"] = {
        labels[i]: float(live_vec[i]) for i in range(len(labels))
    }
    payload["positions"] = _positions_snapshot(
        latest_w,
        nav=cash,
        labels=labels,
        last_closes=None,
        tickers=[str(x) for x in labels[1:]],
    )
    payload["companion_run_id"] = RL_LIVE_RUN_ID if include_live_model else None
    payload["companion_crypto_run_id"] = CRYPTO_LIVE_RUN_ID
    payload["live"] = {
        "prices_refreshed": False,
        "as_of_bar": ts.isoformat(timespec="minutes"),
        "as_of_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "min_refresh_seconds": int(DEFAULT_MIN_REFRESH_SECONDS),
        "bar_interval": BAR_INTERVAL,
        "source": "flat_baseline",
        "book_start": book_start,
        "session_start": book_start,
        "reset_book": True,
    }
    # Wipe intraday caches so soft polls cannot rebuild Friday's path.
    for path in (
        exec_root / f"forward_prices_5m_{rid}.npz",
        exec_root / f"forward_prices_5m_{RL_LIVE_RUN_ID}.npz",
        exec_root / LIVE_STAMP_NAME,
    ):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    payload = _finalize_forward_mark(
        payload,
        model_positions=payload.get("positions"),
        model_price_source="flat",
        write=True,
    )
    if include_live_model:
        live_payload = dict(payload)
        live_payload["run_id"] = RL_LIVE_RUN_ID
        live_payload["checkpoint_label"] = "best"
        live_payload["companion_run_id"] = None
        live_payload.pop("companion_crypto_run_id", None)
        write_forward_mark(live_payload)
    # CrestDay tip mark at the same flat cash baseline.
    crest_payload = dict(payload)
    crest_payload["run_id"] = CRYPTO_LIVE_RUN_ID
    crest_payload["checkpoint_label"] = "locked"
    crest_payload["companion_run_id"] = None
    crest_payload["companion_crypto_run_id"] = CRYPTO_LIVE_RUN_ID
    crest_payload["note"] = (
        f"CrestDay flat baseline reset at {cash:,.0f} on {book_start}."
    )
    write_forward_mark(crest_payload)
    _write_stamp(
        {
            "run_id": rid,
            "bar_interval": BAR_INTERVAL,
            "book_start": book_start,
            "holdout_start": book_start,
            "n_bars": 1,
            "last_bar": ts.isoformat(timespec="minutes"),
            "prices_fetched_at_unix": time.time(),
            "flat_baseline": True,
        },
        root,
    )
    set_active_forward_run(rid)
    return payload


def _merge_price_history(
    *,
    cached_times: pd.DatetimeIndex | None,
    cached_o: np.ndarray | None,
    cached_h: np.ndarray | None,
    cached_l: np.ndarray | None,
    cached_c: np.ndarray | None,
    cached_so: np.ndarray | None,
    cached_sh: np.ndarray | None,
    cached_sl: np.ndarray | None,
    cached_sc: np.ndarray | None,
    times: pd.DatetimeIndex,
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    so: np.ndarray,
    sh: np.ndarray,
    sl: np.ndarray,
    sc: np.ndarray,
) -> tuple[
    pd.DatetimeIndex,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Union cached + fresh bars; fresh wins on timestamp collisions."""
    if cached_times is None or len(cached_times) < 1 or cached_o is None:
        return times, o, h, l, c, so, sh, sl, sc
    if cached_o.shape[1] != o.shape[1]:
        # Label / universe change — drop incompatible cache.
        return times, o, h, l, c, so, sh, sl, sc

    old_idx = pd.DatetimeIndex(cached_times)
    new_idx = pd.DatetimeIndex(times)
    union = old_idx.union(new_idx).sort_values()
    n_assets = int(o.shape[1])

    def _reindex_2d(idx: pd.DatetimeIndex, arr: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(arr, index=idx)

    def _combine_2d(old: np.ndarray, new: np.ndarray) -> np.ndarray:
        old_df = _reindex_2d(old_idx, old)
        new_df = _reindex_2d(new_idx, new)
        # Align columns by position (0..N-1).
        old_df.columns = list(range(n_assets))
        new_df.columns = list(range(n_assets))
        merged = new_df.combine_first(old_df).reindex(union)
        return merged.to_numpy(dtype=np.float64)

    def _combine_1d(old: np.ndarray, new: np.ndarray) -> np.ndarray:
        s_old = pd.Series(np.asarray(old, dtype=np.float64), index=old_idx)
        s_new = pd.Series(np.asarray(new, dtype=np.float64), index=new_idx)
        return s_new.combine_first(s_old).reindex(union).to_numpy(dtype=np.float64)

    return (
        pd.DatetimeIndex(union),
        _combine_2d(cached_o, o),
        _combine_2d(cached_h, h),
        _combine_2d(cached_l, l),
        _combine_2d(cached_c, c),
        _combine_1d(np.asarray(cached_so), so),
        _combine_1d(np.asarray(cached_sh), sh),
        _combine_1d(np.asarray(cached_sl), sl),
        _combine_1d(np.asarray(cached_sc), sc),
    )


def _nav_series_from_ohlc(ohlc: np.ndarray, *, initial_cash: float) -> np.ndarray:
    """Close-marked NAV path that starts at ``initial_cash`` (first open)."""
    arr = np.asarray(ohlc, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 1:
        return np.asarray([], dtype=np.float64)
    nav = arr[:, 3].copy()
    # First plotted point = book cash (candle open), not the first bar's close —
    # otherwise a hot open makes GeneralEquity/SPY appear to start ≠ 100k.
    nav[0] = float(arr[0, 0]) if np.isfinite(arr[0, 0]) else float(initial_cash)
    return nav


def _stamp_path(root: Path | None = None) -> Path:
    base = root or PROJECT_ROOT
    return base / "execution" / LIVE_STAMP_NAME


def _read_stamp(root: Path | None = None) -> dict[str, Any]:
    path = _stamp_path(root)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_stamp(payload: dict[str, Any], root: Path | None = None) -> None:
    path = _stamp_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _normalize_weight_key(label: str) -> str:
    return str(label).strip().upper().replace(" ", "")


def _weight_vector(
    weights: dict[str, float] | None,
    asset_labels: list[str],
) -> np.ndarray:
    """Map a weight dict onto ``asset_labels`` (Cash + risky). Missing → 0."""
    n = len(asset_labels)
    out = np.zeros(n, dtype=np.float64)
    if not weights:
        out[0] = 1.0
        return out
    by_key = {_normalize_weight_key(k): float(v) for k, v in weights.items()}
    for i, lab in enumerate(asset_labels):
        out[i] = by_key.get(_normalize_weight_key(lab), 0.0)
    s = float(out.sum())
    if s <= 1e-12:
        out[0] = 1.0
        return out
    return out / s


def _align_nav_list_on_grid(
    nav: list[Any],
    stamps: list[Any],
    times: pd.DatetimeIndex,
    *,
    initial_cash: float,
) -> np.ndarray | None:
    """Forward-fill a stamped NAV list onto ``times`` and rebase to ``initial_cash``."""
    if not isinstance(nav, list) or len(nav) < 1 or not stamps:
        return None
    try:
        s = pd.Series(
            np.asarray(nav, dtype=np.float64),
            index=pd.DatetimeIndex([pd.Timestamp(x) for x in stamps]).tz_localize(None),
        )
        aligned = s.reindex(pd.DatetimeIndex(times).tz_localize(None), method="ffill")
        arr = aligned.to_numpy(dtype=np.float64)
        if arr.size < 1 or not np.isfinite(arr[0]) or arr[0] <= 0:
            return None
        return arr / float(arr[0]) * float(initial_cash)
    except Exception:  # noqa: BLE001
        return None


def _soft_pack_nav_on_grid(
    times: pd.DatetimeIndex,
    *,
    initial_cash: float,
    simulate_fn: Any,
    disk_run_id: str,
) -> np.ndarray | None:
    """Align a pack NAV series onto the equity 5m grid (soft-fail).

    Prefers the on-disk companion mark (ms) so equity Yahoo refreshes are not
    blocked by a cold CrestDay simulate. Pack simulate is best-effort with a
    short timeout and non-blocking worker shutdown.
    """
    from concurrent.futures import ThreadPoolExecutor

    mark = load_forward_mark(disk_run_id)
    if isinstance(mark, dict):
        disk = _align_nav_list_on_grid(
            (mark.get("nav") or {}).get("model") or [],
            mark.get("timestamps") or mark.get("dates") or [],
            times,
            initial_cash=initial_cash,
        )
        if disk is not None:
            return disk

    def _sim() -> dict[str, Any] | None:
        try:
            since = None
            if len(times):
                since = pd.Timestamp(times[0]).date()
            return simulate_fn(
                force_refresh=False,
                initial_cash=float(initial_cash),
                since=since,
            )
        except Exception:  # noqa: BLE001
            return None

    sim: dict[str, Any] | None = None
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        sim = pool.submit(_sim).result(timeout=CRYPTO_SOFT_TIMEOUT_S)
    except Exception:  # noqa: BLE001
        sim = None
    finally:
        # Never wait on a hung pack worker — same pattern as call_with_timeout.
        pool.shutdown(wait=False, cancel_futures=True)

    if not isinstance(sim, dict):
        return None

    try:
        s = pd.Series(
            np.asarray(sim["nav"], dtype=np.float64),
            index=pd.DatetimeIndex(sim["times"]).tz_localize(None),
        )
        aligned = s.reindex(pd.DatetimeIndex(times).tz_localize(None), method="ffill")
        aligned = aligned.fillna(float(initial_cash))
        arr = aligned.to_numpy(dtype=np.float64)
        if arr.size < 1 or not np.isfinite(arr[0]) or arr[0] <= 0:
            return None
        return arr / float(arr[0]) * float(initial_cash)
    except Exception:  # noqa: BLE001
        return None


def _soft_crypto_nav_on_grid(
    times: pd.DatetimeIndex,
    *,
    initial_cash: float,
) -> np.ndarray | None:
    """CrestDay pack NAV on the equity session grid (soft-fail)."""
    from rlbot.pack_crestday import simulate_nav_series

    return _soft_pack_nav_on_grid(
        times,
        initial_cash=initial_cash,
        simulate_fn=simulate_nav_series,
        disk_run_id=CRYPTO_LIVE_RUN_ID,
    )


def _soft_durable_nav_on_grid(
    times: pd.DatetimeIndex,
    *,
    initial_cash: float,
) -> np.ndarray | None:
    """Durable.v1 pack NAV on the equity session grid (soft-fail)."""
    from rlbot.pack_durable import simulate_nav_series

    return _soft_pack_nav_on_grid(
        times,
        initial_cash=initial_cash,
        simulate_fn=simulate_nav_series,
        disk_run_id=DURABLE_LIVE_RUN_ID,
    )


def _latest_shadow_weights(run_id: str) -> dict[str, float] | None:
    path = PROJECT_ROOT / "execution" / f"shadow_ledger_{run_id}.jsonl"
    if not path.is_file():
        return None
    last: dict[str, float] | None = None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tw = rec.get("target_weights")
            if isinstance(tw, dict) and tw:
                last = {str(k): float(v) for k, v in tw.items()}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return last
    return last


def _is_external_weight_book(
    weights: dict[str, float] | None,
    *,
    run_id: str,
    cfg_asset_keys: list[str],
) -> bool:
    """True when the live book is not the RL research sleeve (algo ETF / stock picks)."""
    rid = str(run_id).upper()
    if rid in {
        ALGO_LIVE_RUN_ID,
        "GENERAL_EQUITY",
        "PROD_RETURN_ALPHA",
        "FINALMODEL",
    }:
        return True
    if not weights:
        return False
    cfg = {_normalize_weight_key(k) for k in cfg_asset_keys}
    cfg.add("CASH")
    keys = {_normalize_weight_key(k) for k in weights}
    risky = keys - {"CASH"}
    if not risky:
        return False
    overlap = len(risky & cfg) / len(risky)
    return overlap < 0.5


def _stock_symbols_from_weights(weights: dict[str, float]) -> dict[str, str]:
    """Map display label → Yahoo symbol for external books (BRK.B → BRK-B)."""
    from rlbot.pack_general_equity1 import to_yahoo_symbol

    out: dict[str, str] = {}
    for k, v in weights.items():
        lab = str(k).strip()
        if not lab or lab.upper() == "CASH" or float(v) <= 0:
            continue
        out[lab.upper() if "." not in lab else lab] = to_yahoo_symbol(lab)
    # Prefer uppercase keys consistently.
    return {str(k).upper(): v for k, v in out.items()}


def _cash_yield_per_bar(cash_daily_yield: float) -> float:
    """Scale config daily cash yield onto one 5m RTH bar."""
    return float(cash_daily_yield) / float(BARS_PER_TRADING_DAY)


def _nav_from_weights(
    closes: np.ndarray,
    weight_rows: list[np.ndarray],
    *,
    initial_cash: float,
    cash_daily_yield: float = 0.0,
) -> np.ndarray:
    """Close-to-close MTM (legacy daily helper; kept for unit tests)."""
    t_bars, n_assets = closes.shape
    if t_bars < 1:
        return np.asarray([initial_cash], dtype=np.float64)
    # Interpret cash_daily_yield as per-bar when used with intraday closes.
    y = float(cash_daily_yield)
    nav = np.empty(t_bars, dtype=np.float64)
    nav[0] = float(initial_cash)
    for t in range(t_bars - 1):
        w = weight_rows[min(t, len(weight_rows) - 1)]
        if w.shape[0] != n_assets + 1:
            ww = np.zeros(n_assets + 1, dtype=np.float64)
            m = min(w.shape[0], ww.shape[0])
            ww[:m] = w[:m]
            w = ww
        cash_w = float(w[0])
        risky = w[1:]
        rets = closes[t + 1] / np.maximum(closes[t], 1e-12) - 1.0
        port = float(np.dot(risky, rets)) + cash_w * y
        nav[t + 1] = nav[t] * (1.0 + port)
    return nav


def _ew_nav(closes: np.ndarray, initial_cash: float) -> np.ndarray:
    t_bars, n = closes.shape
    nav = np.empty(t_bars, dtype=np.float64)
    nav[0] = float(initial_cash)
    if n <= 0:
        nav[:] = float(initial_cash)
        return nav
    w = np.full(n, 1.0 / n, dtype=np.float64)
    for t in range(t_bars - 1):
        rets = closes[t + 1] / np.maximum(closes[t], 1e-12) - 1.0
        nav[t + 1] = nav[t] * (1.0 + float(np.dot(w, rets)))
    return nav


def _spy_nav(spy_close: np.ndarray, initial_cash: float) -> np.ndarray:
    s0 = max(float(spy_close[0]), 1e-12)
    return (np.asarray(spy_close, dtype=np.float64) / s0) * float(initial_cash)


def portfolio_ohlc_candles(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    weight: np.ndarray,
    *,
    initial_cash: float,
    cash_yield_per_bar: float = 0.0,
) -> np.ndarray:
    """Build NAV OHLC candles from aligned asset OHLC and a fixed weight vector.

    ``weight`` is Cash + N risky (sums to 1). Within each bar, O/H/L/C returns are
    taken vs the previous bar's asset closes (bar 0 vs that bar's opens). Highs/lows
    are simultaneous-price approximations (standard for multi-asset candles).
    Returns ``[T, 4]`` with columns open, high, low, close.
    """
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    if open_.ndim != 2:
        raise ValueError("open/high/low/close must be [T, N]")
    t_bars, n_assets = close.shape
    w = np.asarray(weight, dtype=np.float64).reshape(-1)
    if w.shape[0] != n_assets + 1:
        ww = np.zeros(n_assets + 1, dtype=np.float64)
        m = min(w.shape[0], ww.shape[0])
        ww[:m] = w[:m]
        s = float(ww.sum())
        w = ww / s if s > 1e-12 else np.array([1.0] + [0.0] * n_assets)
    cash_w = float(w[0])
    risky = w[1:]
    out = np.empty((t_bars, 4), dtype=np.float64)
    if t_bars < 1:
        return out

    nav_close = float(initial_cash)
    prev_px = np.asarray(open_[0], dtype=np.float64)
    y = float(cash_yield_per_bar)

    for t in range(t_bars):
        # Cash yield accrues once per bar (on H/L/C); open is mark-to-market only.
        if t == 0:
            base = np.maximum(open_[t], 1e-12)
            r_h = high[t] / base - 1.0
            r_l = low[t] / base - 1.0
            r_c = close[t] / base - 1.0
            o_nav = float(initial_cash)
            h_nav = o_nav * (1.0 + float(np.dot(risky, r_h)) + cash_w * y)
            l_nav = o_nav * (1.0 + float(np.dot(risky, r_l)) + cash_w * y)
            c_nav = o_nav * (1.0 + float(np.dot(risky, r_c)) + cash_w * y)
        else:
            base = np.maximum(prev_px, 1e-12)
            r_o = open_[t] / base - 1.0
            r_h = high[t] / base - 1.0
            r_l = low[t] / base - 1.0
            r_c = close[t] / base - 1.0
            o_nav = nav_close * (1.0 + float(np.dot(risky, r_o)))
            h_nav = nav_close * (1.0 + float(np.dot(risky, r_h)) + cash_w * y)
            l_nav = nav_close * (1.0 + float(np.dot(risky, r_l)) + cash_w * y)
            c_nav = nav_close * (1.0 + float(np.dot(risky, r_c)) + cash_w * y)
        hi = max(o_nav, h_nav, l_nav, c_nav)
        lo = min(o_nav, h_nav, l_nav, c_nav)
        out[t] = (o_nav, hi, lo, c_nav)
        nav_close = c_nav
        prev_px = close[t]
    return out


def equal_weight_ohlc_candles(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    initial_cash: float,
) -> np.ndarray:
    n = int(close.shape[1]) if close.ndim == 2 else 0
    if n <= 0:
        t = int(close.shape[0]) if close.ndim >= 1 else 0
        return np.full((t, 4), float(initial_cash), dtype=np.float64)
    w = np.concatenate([[0.0], np.full(n, 1.0 / n, dtype=np.float64)])
    return portfolio_ohlc_candles(
        open_, high, low, close, w, initial_cash=initial_cash, cash_yield_per_bar=0.0
    )


def spy_ohlc_candles(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    initial_cash: float,
) -> np.ndarray:
    """Scale a single-asset OHLC series so the first open equals ``initial_cash``."""
    o = np.asarray(open_, dtype=np.float64).reshape(-1)
    h = np.asarray(high, dtype=np.float64).reshape(-1)
    l = np.asarray(low, dtype=np.float64).reshape(-1)
    c = np.asarray(close, dtype=np.float64).reshape(-1)
    if o.size < 1:
        return np.zeros((0, 4), dtype=np.float64)
    scale = float(initial_cash) / max(float(o[0]), 1e-12)
    out = np.column_stack([o * scale, h * scale, l * scale, c * scale])
    # Enforce OHLC consistency after float scaling.
    out[:, 1] = np.maximum(out[:, 1], np.maximum(out[:, 0], out[:, 3]))
    out[:, 2] = np.minimum(out[:, 2], np.minimum(out[:, 0], out[:, 3]))
    return out


def _candles_to_payload(ts: pd.DatetimeIndex, ohlc: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, t in enumerate(ts):
        stamp = pd.Timestamp(t)
        if stamp.tzinfo is None:
            # Yahoo intraday is exchange-local; keep naive ISO without claiming UTC.
            iso = stamp.isoformat(timespec="minutes")
        else:
            iso = stamp.isoformat(timespec="minutes")
        o, h, l, c = (float(ohlc[i, j]) for j in range(4))
        rows.append({"t": iso, "o": o, "h": h, "l": l, "c": c})
    return rows


def _positions_snapshot(
    weights: dict[str, float],
    *,
    nav: float,
    labels: list[str],
    last_closes: np.ndarray | None,
    tickers: list[str],
) -> list[dict[str, Any]]:
    w = _weight_vector(weights, labels)
    rows: list[dict[str, Any]] = []
    for i, lab in enumerate(labels):
        value = float(w[i] * nav)
        if not np.isfinite(value):
            value = 0.0
        price: float | None = None
        if i == 0:
            price = 1.0
        elif last_closes is not None and i - 1 < last_closes.shape[0]:
            raw = float(last_closes[i - 1])
            price = raw if np.isfinite(raw) and raw > 0 else None
        rows.append(
            {
                "label": lab,
                "ticker": "CASH" if i == 0 else (tickers[i - 1] if i - 1 < len(tickers) else lab),
                "weight": float(w[i]) if np.isfinite(w[i]) else 0.0,
                "value_usd": value,
                "price": price,
            }
        )
    rows.sort(
        key=lambda r: (-1 if str(r["label"]).lower() == "cash" else 0, -float(r["weight"]))
    )
    return rows


# Forward allocation panels (nav key → UI label + paper/shadow run id).
_ALLOCATION_BOOKS: tuple[tuple[str, str, str], ...] = (
    ("model", "GeneralEquity1", ALGO_LIVE_RUN_ID),
    ("live_model", "RLModel", RL_LIVE_RUN_ID),
    ("crypto", "CrestDay", CRYPTO_LIVE_RUN_ID),
)

_PAPER_STATE_BY_RUN: dict[str, Path] = {
    ALGO_LIVE_RUN_ID: PROJECT_ROOT / "execution" / "paper_general_equity1" / "state.json",
    CRYPTO_LIVE_RUN_ID: PROJECT_ROOT / "execution" / "paper_crest_day" / "state.json",
}


def _equity_rth_open_now() -> bool:
    """True during Mon–Fri US cash session (09:30–16:00 America/New_York)."""
    now = pd.Timestamp.now(tz="America/New_York")
    if int(now.dayofweek) >= 5:
        return False
    minutes = int(now.hour) * 60 + int(now.minute)
    return (9 * 60 + 30) <= minutes < (16 * 60)


def _now_et_floor_5m() -> pd.Timestamp:
    return pd.Timestamp.now(tz="America/New_York").tz_localize(None).floor("5min")


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _weights_from_paper_state(run_id: str) -> dict[str, float] | None:
    path = _PAPER_STATE_BY_RUN.get(str(run_id).upper())
    if path is None:
        return None
    st = _read_json_file(path)
    tw = st.get("target_weights")
    if not isinstance(tw, dict) or not tw:
        return None
    return {str(k): float(v) for k, v in tw.items()}


def _resolve_strategy_weights(run_id: str, *, aum: float) -> dict[str, float]:
    """Prefer shadow ledger / paper state (fast); avoid pack engine on soft polls."""
    del aum  # reserved for future tip-mark AUM scaling
    for getter in (
        lambda: _latest_shadow_weights(run_id),
        lambda: _weights_from_paper_state(run_id),
    ):
        try:
            w = getter()
        except Exception:  # noqa: BLE001
            w = None
        if isinstance(w, dict) and w:
            return {str(k): float(v) for k, v in w.items()}
    tip = load_forward_mark(run_id)
    if isinstance(tip, dict):
        tw = tip.get("latest_weights")
        if isinstance(tw, dict) and tw:
            return {str(k): float(v) for k, v in tw.items()}
    return {"Cash": 1.0}


def _positions_from_weight_map(
    weights: dict[str, float],
    *,
    nav: float,
    price_by_ticker: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Build allocation rows from an arbitrary weight dict (Cash + names)."""
    price_by_ticker = price_by_ticker or {}
    by_key = {_normalize_weight_key(k): float(v) for k, v in weights.items()}
    cash_w = float(by_key.get("CASH", 0.0))
    risky_keys = [k for k in by_key if k != "CASH" and abs(by_key[k]) > 1e-12]
    # Preserve insertion order from original weights where possible.
    ordered: list[str] = []
    seen: set[str] = set()
    for k in weights:
        nk = _normalize_weight_key(k)
        if nk == "CASH" or nk in seen or nk not in by_key:
            continue
        if abs(by_key[nk]) <= 1e-12:
            continue
        ordered.append(nk)
        seen.add(nk)
    for nk in risky_keys:
        if nk not in seen:
            ordered.append(nk)
            seen.add(nk)
    labels = ["Cash"] + ordered
    tickers = ["CASH"] + ordered
    closes = None
    if ordered and price_by_ticker:
        vals = []
        any_px = False
        for lab in ordered:
            px = price_by_ticker.get(lab, price_by_ticker.get(lab.upper()))
            try:
                fpx = float(px) if px is not None else float("nan")
            except (TypeError, ValueError):
                fpx = float("nan")
            if np.isfinite(fpx) and fpx > 0:
                any_px = True
                vals.append(fpx)
            else:
                vals.append(float("nan"))
        closes = np.asarray(vals, dtype=np.float64) if any_px else None
    return _positions_snapshot(
        {**{lab: by_key.get(_normalize_weight_key(lab), 0.0) for lab in labels}, "Cash": cash_w},
        nav=float(nav),
        labels=labels,
        last_closes=closes,
        tickers=tickers[1:],
    )


def _coinbase_prices_for_weights(weights: dict[str, float]) -> dict[str, float]:
    syms = [
        str(k)
        for k, v in weights.items()
        if str(k).upper() not in {"CASH", "USD"} and abs(float(v)) > 1e-12
    ]
    if not syms:
        return {}
    try:
        from rlbot.coinbase_market import fetch_last_prices

        return fetch_last_prices(syms)
    except Exception:  # noqa: BLE001
        return {}


def _build_allocations_payload(
    mark: dict[str, Any],
    *,
    model_positions: list[dict[str, Any]] | None = None,
    model_price_source: str = "yahoo",
) -> dict[str, Any]:
    """Snapshot current positions for every live strategy sleeve."""
    nav_map = mark.get("nav") if isinstance(mark.get("nav"), dict) else {}
    stats = mark.get("stats") if isinstance(mark.get("stats"), dict) else {}
    initial_cash = float(mark.get("initial_cash") or 100_000.0)
    as_of = (mark.get("live") or {}).get("as_of_utc") or mark.get("generated_at_utc")
    out: dict[str, Any] = {}

    for key, label, run_id in _ALLOCATION_BOOKS:
        tip_nav = None
        st = stats.get(key) if isinstance(stats.get(key), dict) else None
        if st and st.get("nav") is not None:
            tip_nav = float(st["nav"])
        elif isinstance(nav_map.get(key), list) and nav_map[key]:
            tip_nav = float(nav_map[key][-1])
        nav = float(tip_nav) if tip_nav is not None else initial_cash

        if key == "model" and model_positions is not None:
            positions = model_positions
            price_source = model_price_source
            weights = mark.get("latest_weights") if isinstance(mark.get("latest_weights"), dict) else {}
        else:
            weights = _resolve_strategy_weights(run_id, aum=nav)
            if key == "crypto":
                prices = _coinbase_prices_for_weights(weights)
                price_source = "coinbase" if prices else "weights"
            else:
                prices = {}
                price_source = "weights"
                # Avoid per-poll Yahoo fan-out for the RL sleeve (model panel
                # already carries Yahoo-marked ETF prices from the live refresh).
            positions = _positions_from_weight_map(weights, nav=nav, price_by_ticker=prices)

        out[key] = {
            "key": key,
            "label": label,
            "run_id": run_id,
            "nav": nav,
            "as_of": as_of,
            "price_source": price_source,
            "positions": positions,
            "latest_weights": {
                str(k): float(v) for k, v in (weights or {}).items()
            },
        }
    return out


def _payload_times(mark: dict[str, Any]) -> pd.DatetimeIndex:
    stamps = mark.get("timestamps") or mark.get("dates") or []
    if not stamps:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex([pd.Timestamp(x).tz_localize(None) for x in stamps])


def _extend_nav_series(series: list[Any] | None, n_add: int) -> list[float]:
    if not series:
        return []
    last = float(series[-1])
    return [float(x) for x in series] + [last] * int(n_add)


def _extend_payload_clock_24_7(mark: dict[str, Any]) -> dict[str, Any]:
    """Append 5m bars past the last equity print so crypto sleeves keep updating.

    Only extends outside the US cash session. During RTH, inventing flat equity
    bars after a failed Yahoo pull made the chart look live while NAV was frozen.
    """
    times = _payload_times(mark)
    if len(times) < 1:
        return mark
    now = _now_et_floor_5m()
    last = pd.Timestamp(times[-1])
    if now <= last:
        return mark
    # During RTH wait for real Yahoo bars — do not ffill invent.
    if _equity_rth_open_now():
        return mark
    extra = pd.date_range(last + pd.Timedelta(minutes=5), now, freq="5min")
    # Cap weekend growth (~3 calendar days of 5m bars).
    max_extra = 3 * 24 * 12
    if len(extra) > max_extra:
        extra = extra[-max_extra:]
    if len(extra) < 1:
        return mark

    n_add = int(len(extra))
    new_times = times.append(extra)
    iso = [pd.Timestamp(t).isoformat(timespec="minutes") for t in new_times]
    mark = dict(mark)
    mark["timestamps"] = iso
    mark["dates"] = iso
    mark["n_bars"] = int(len(iso))

    nav = dict(mark.get("nav") or {})
    for key in ("model", "spy", "equal_weight", "live_model", "crypto"):
        if key in nav and isinstance(nav[key], list) and nav[key]:
            nav[key] = _extend_nav_series(nav[key], n_add)
    # Drop retired Durable.v1 series so 24/7 extension does not keep them alive.
    nav.pop("durable", None)
    mark["nav"] = nav

    stats = dict(mark.get("stats") or {})
    stats.pop("durable", None)
    bpy = float(BARS_PER_TRADING_DAY * 252)
    for key, series in nav.items():
        if isinstance(series, list) and series:
            stats[key] = _series_stats(series, bars_per_year=bpy, timestamps=iso)
    mark["stats"] = stats

    candles = mark.get("candles")
    if isinstance(candles, dict):
        new_candles = dict(candles)
        new_candles.pop("durable", None)
        for key, series in nav.items():
            if not isinstance(series, list) or len(series) != len(new_times):
                continue
            arr = np.asarray(series, dtype=np.float64)
            ohlc = np.column_stack([arr, arr, arr, arr])
            new_candles[key] = _candles_to_payload(new_times, ohlc)
        mark["candles"] = new_candles

    live = dict(mark.get("live") or {})
    live["as_of_bar"] = iso[-1]
    live["as_of_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    live["crypto_clock"] = "24_7"
    if not _equity_rth_open_now():
        live["equity_session"] = "closed"
    mark["live"] = live
    mark.pop("companion_durable_run_id", None)
    note = str(mark.get("note") or "")
    if "24/7 crypto clock" not in note:
        mark["note"] = (
            note
            + (" " if note and not note.endswith(".") else "")
            + " After the cash close, equity sleeves hold last print; "
            "CrestDay keeps a 24/7 5m clock."
        ).strip()
    return mark


def _attach_soft_companions_to_mark(payload: dict[str, Any]) -> dict[str, Any]:
    """Align CrestDay NAV onto the mark's timestamp grid (soft-fail)."""
    times = _payload_times(payload)
    if len(times) < 1:
        return payload
    initial_cash = float(payload.get("initial_cash") or 100_000.0)
    date_strs = [pd.Timestamp(t).isoformat(timespec="minutes") for t in times]
    candles = payload.get("candles") if isinstance(payload.get("candles"), dict) else None
    nav = dict(payload.get("nav") or {})
    stats = dict(payload.get("stats") or {})
    bpy = float(BARS_PER_TRADING_DAY * 252)
    # Strip retired Durable.v1 series from older marks.
    nav.pop("durable", None)
    stats.pop("durable", None)
    if isinstance(candles, dict):
        candles = dict(candles)
        candles.pop("durable", None)
    payload.pop("companion_durable_run_id", None)
    allocations = payload.get("allocations")
    if isinstance(allocations, dict) and "durable" in allocations:
        allocations = dict(allocations)
        allocations.pop("durable", None)
        payload["allocations"] = allocations

    def _attach(nav_key: str, arr: np.ndarray | None, companion_field: str, run_id: str) -> None:
        if arr is None or arr.size < len(times):
            return
        series = np.asarray(arr[: len(times)], dtype=np.float64)
        if np.isfinite(series[0]) and series[0] > 0:
            series = series / float(series[0]) * float(initial_cash)
        nav[nav_key] = series.tolist()
        stats[nav_key] = _series_stats(
            series.tolist(), bars_per_year=bpy, timestamps=date_strs
        )
        payload[companion_field] = run_id
        if candles is not None:
            ohlc = np.column_stack([series, series, series, series])
            candles[nav_key] = _candles_to_payload(times, ohlc)

    try:
        _attach(
            "crypto",
            _soft_crypto_nav_on_grid(times, initial_cash=initial_cash),
            "companion_crypto_run_id",
            CRYPTO_LIVE_RUN_ID,
        )
    except Exception:  # noqa: BLE001
        pass

    payload["nav"] = nav
    payload["stats"] = stats
    if candles is not None:
        payload["candles"] = candles
    if "crypto" not in nav:
        try:
            from rlbot.forward_mark import merge_crypto_companion

            payload = merge_crypto_companion(payload)
        except Exception:  # noqa: BLE001
            pass
    return payload


def _finalize_forward_mark(
    payload: dict[str, Any],
    *,
    model_positions: list[dict[str, Any]] | None = None,
    model_price_source: str = "yahoo",
    write: bool = True,
) -> dict[str, Any]:
    """Extend 24/7 crypto clock, soft-attach companions, stamp all allocations."""
    payload = _extend_payload_clock_24_7(dict(payload))
    payload = _attach_soft_companions_to_mark(payload)
    positions = model_positions
    if positions is None and isinstance(payload.get("positions"), list):
        positions = payload["positions"]
    allocations = _build_allocations_payload(
        payload,
        model_positions=positions,
        model_price_source=model_price_source,
    )
    payload["allocations"] = allocations
    model_alloc = allocations.get("model") or {}
    if isinstance(model_alloc.get("positions"), list):
        payload["positions"] = model_alloc["positions"]
    live = dict(payload.get("live") or {})
    live["as_of_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload["live"] = live
    payload["generated_at_utc"] = live["as_of_utc"]
    if write:
        write_forward_mark(payload)
    return payload


def _yahoo_chart_ohlc(
    symbol: str,
    *,
    range_key: str = "5d",
    interval: str = BAR_INTERVAL,
    timeout_s: float = 12.0,
) -> pd.DataFrame:
    """Fetch one symbol's intraday OHLC via Yahoo's chart API (no yfinance)."""
    import urllib.parse

    params = urllib.parse.urlencode(
        {"range": range_key, "interval": interval, "events": "history"}
    )
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{urllib.parse.quote(symbol, safe='')}?{params}"
    )
    try:
        from curl_cffi import requests as curl_requests

        response = curl_requests.get(url, impersonate="chrome", timeout=float(timeout_s))
    except Exception:  # noqa: BLE001 - fall back to stdlib urllib
        try:
            import urllib.request

            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            response = None
        except Exception:  # noqa: BLE001
            return pd.DataFrame()
        result = (payload.get("chart", {}).get("result") or [None])[0]
        if not isinstance(result, dict):
            return pd.DataFrame()
        timestamps = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [{}])[0] or {}
        if not timestamps:
            return pd.DataFrame()
        idx = pd.DatetimeIndex(pd.to_datetime(timestamps, unit="s", utc=True))
        idx = idx.tz_convert("America/New_York").tz_localize(None)
        frame = pd.DataFrame(
            {
                "Open": quote.get("open") or [None] * len(timestamps),
                "High": quote.get("high") or [None] * len(timestamps),
                "Low": quote.get("low") or [None] * len(timestamps),
                "Close": quote.get("close") or [None] * len(timestamps),
            },
            index=idx,
        )
        frame = frame.apply(pd.to_numeric, errors="coerce")
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        return frame.dropna(subset=["Close"], how="any")

    if getattr(response, "status_code", 200) != 200:
        return pd.DataFrame()
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    result = (payload.get("chart", {}).get("result") or [None])[0]
    if not isinstance(result, dict):
        return pd.DataFrame()
    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0] or {}
    if not timestamps:
        return pd.DataFrame()
    idx = pd.DatetimeIndex(pd.to_datetime(timestamps, unit="s", utc=True))
    idx = idx.tz_convert("America/New_York").tz_localize(None)
    frame = pd.DataFrame(
        {
            "Open": quote.get("open") or [None] * len(timestamps),
            "High": quote.get("high") or [None] * len(timestamps),
            "Low": quote.get("low") or [None] * len(timestamps),
            "Close": quote.get("close") or [None] * len(timestamps),
        },
        index=idx,
    )
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame.dropna(subset=["Close"], how="any")


def _fetch_intraday_ohlc(
    symbols: dict[str, str],
    *,
    since: str,
    spy_symbol: str = "SPY",
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aligned intraday OHLC for universe labels + SPY on the SPY bar clock.

    Uses Yahoo's chart HTTP API (curl_cffi) instead of yfinance — the latter
    routinely hangs under Desktop iCloud load and left the forward mark stuck
    on a stale ``execution/`` price cache.

    Assets that trade different sessions (FX, Nikkei, FTSE, futures) are
    reindexed onto SPY timestamps and forward-filled so a missing print carries
    the last close as a flat candle (o=h=l=c). Returns
    ``times, open[T,N], high[T,N], low[T,N], close[T,N], spy_o/h/l/c[T]``.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    labels = list(symbols.keys())
    ticker_list = list(dict.fromkeys([*symbols.values(), spy_symbol]))
    # 5d of 5m bars covers the live forward window without the monthly
    # downgrade Yahoo applies to range=max.
    range_key = "5d"
    since_ts = pd.Timestamp(since)

    frames_by_sym: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=min(8, len(ticker_list))) as pool:
        futs = {
            pool.submit(_yahoo_chart_ohlc, sym, range_key=range_key, interval=BAR_INTERVAL): sym
            for sym in ticker_list
        }
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                frames_by_sym[sym] = fut.result()
            except Exception:  # noqa: BLE001
                frames_by_sym[sym] = pd.DataFrame()

    spy_df = frames_by_sym.get(spy_symbol, pd.DataFrame())
    if spy_df.empty:
        raise RuntimeError(f"Yahoo chart returned no {BAR_INTERVAL} SPY rows ({spy_symbol})")
    spy_df = spy_df[spy_df.index >= since_ts]
    if spy_df.empty:
        raise RuntimeError(
            f"no {BAR_INTERVAL} SPY bars on/after {since_ts.date()} ({spy_symbol})"
        )

    clock = spy_df.index
    frames: dict[str, pd.DataFrame] = {}
    for lab, sym in symbols.items():
        df = frames_by_sym.get(sym, pd.DataFrame())
        if df.empty:
            raise RuntimeError(f"missing {BAR_INTERVAL} history for {lab} ({sym})")
        df = df[df.index >= since_ts] if len(df) else df
        close_ff = df["Close"].reindex(clock, method="ffill")
        # Prepend a short bfill only at the head if the asset lists after SPY open.
        if close_ff.isna().any():
            close_ff = close_ff.bfill()
        native = df.reindex(clock)
        aligned = pd.DataFrame(index=clock)
        aligned["Close"] = close_ff
        for col in ("Open", "High", "Low"):
            aligned[col] = native[col].where(native[col].notna(), close_ff)
        if aligned["Close"].isna().any():
            raise RuntimeError(
                f"could not align {BAR_INTERVAL} bars for {lab} ({sym}) onto SPY clock"
            )
        frames[lab] = aligned

    o = np.column_stack([frames[lab]["Open"].to_numpy(dtype=np.float64) for lab in labels])
    h = np.column_stack([frames[lab]["High"].to_numpy(dtype=np.float64) for lab in labels])
    l = np.column_stack([frames[lab]["Low"].to_numpy(dtype=np.float64) for lab in labels])
    c = np.column_stack([frames[lab]["Close"].to_numpy(dtype=np.float64) for lab in labels])
    so = spy_df["Open"].to_numpy(dtype=np.float64)
    sh = spy_df["High"].to_numpy(dtype=np.float64)
    sl = spy_df["Low"].to_numpy(dtype=np.float64)
    sc = spy_df["Close"].to_numpy(dtype=np.float64)
    return pd.DatetimeIndex(clock), o, h, l, c, so, sh, sl, sc


_fetch_30m_ohlc = _fetch_intraday_ohlc  # back-compat


def refresh_forward_mark_live(
    run_id: str | None = None,
    *,
    min_refresh_seconds: int = DEFAULT_MIN_REFRESH_SECONDS,
    force_price_refresh: bool = False,
    reset_book: bool = False,
    root: Path | None = None,
) -> dict[str, Any] | None:
    """Refresh 5m NAV candles/positions for the active (or given) LIVE run.

    - Throttled Yahoo pull for 5-minute OHLC (universe + SPY).
    - Model book: last shadow / mark weights, MTM'd into OHLC candles.
    - Benchmarks: equal-weight and SPY buy-and-hold on the same 5m grid.
    - History accumulates from a persistent ``book_start`` (not reset each day).
    - ``reset_book=True`` wipes the price cache and restarts all sleeves at
      ``initial_cash`` from today's US session.
    """
    rid = (run_id or "").strip() or (resolve_active_forward_run_id(root) or "")
    if not rid:
        return None

    existing = load_forward_mark(rid)
    stamp = _read_stamp(root)
    now = time.time()
    last = float(stamp.get("prices_fetched_at_unix") or 0.0)
    last_attempt = float(stamp.get("prices_attempt_at_unix") or 0.0)
    need_fetch = force_price_refresh or reset_book or (now - last >= float(min_refresh_seconds))
    # After a failed/slow Yahoo pull, cool down briefly so /api/forward polls
    # (every ~30s from the UI) do not each block on a fresh download.
    exec_root = (root or PROJECT_ROOT) / "execution"
    price_cache = exec_root / f"forward_prices_5m_{rid}.npz"
    if reset_book:
        for path in (
            price_cache,
            exec_root / f"forward_mark_{rid}.json",
            exec_root / f"forward_prices_5m_{RL_LIVE_RUN_ID}.npz",
        ):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        existing = None
        stamp = {k: v for k, v in stamp.items() if k not in ("holdout_start", "book_start", "n_bars", "last_bar", "flat_baseline")}
        # Weekend / holiday: seed flat $100k immediately (no prior-day replay).
        today = pd.Timestamp.now(tz="America/New_York")
        if int(today.dayofweek) >= 5:
            return seed_flat_forward_baseline(rid, root=root, include_live_model=True)
    if (
        need_fetch
        and not force_price_refresh
        and not reset_book
        and price_cache.is_file()
        and (now - last_attempt) < 60.0
    ):
        need_fetch = False
    # Keep equity flat baseline until a real session prints, but still extend the
    # 24/7 crypto clock and refresh per-strategy allocations on soft polls.
    if (
        not reset_book
        and not force_price_refresh
        and isinstance(existing, dict)
        and (existing.get("live") or {}).get("source") == "flat_baseline"
    ):
        return _finalize_forward_mark(existing)

    # Always use the repo config — never touch Runs/<id>/config.yaml (iCloud hang).
    cfg_path = PROJECT_ROOT / "config" / "config.yaml"
    set_config(load_config(cfg_path))
    cfg = get_config()
    initial_cash = float(cfg.environment.initial_cash) if reset_book else float(
        (existing or {}).get("initial_cash") or cfg.environment.initial_cash
    )
    # Persistent book start — do NOT snap to today on every refresh.
    holdout_start = _resolve_book_start(existing, stamp, reset_book=reset_book)
    book_start = holdout_start
    holdout_end = (existing or {}).get("holdout_end") or stamp.get("holdout_end")
    cash_yield_bar = _cash_yield_per_bar(
        float(getattr(cfg.reward, "cash_daily_yield", 0.0) or 0.0)
    )
    rl_cash_yield_bar = cash_yield_bar

    latest_w = _latest_shadow_weights(rid) or (existing or {}).get("latest_weights")
    if not isinstance(latest_w, dict) or not latest_w:
        latest_w = {"Cash": 1.0}

    rl_symbols = dict(cfg.universe.assets)
    rl_labels = list(rl_symbols.keys())
    stock_book = _is_external_weight_book(
        latest_w, run_id=rid, cfg_asset_keys=rl_labels
    )
    stock_symbols: dict[str, str] = {}
    if stock_book:
        stock_symbols = _stock_symbols_from_weights(latest_w)
        if not stock_symbols:
            stock_symbols = {"SPY": "SPY"}
        tickers = list(stock_symbols.keys())
        labels = ["Cash"] + list(stock_symbols.keys())
        # Equity paper book: no cash yield (cash account, idle cash = 0).
        cash_yield_bar = 0.0
    else:
        tickers = list(cfg.universe.tickers)
        labels = ["Cash"] + rl_labels

    # Fetch RL sleeve always (EW-10 + LIVE_MODEL); add external book symbols when needed.
    symbols: dict[str, str] = dict(rl_symbols)
    symbols.update(stock_symbols)
    live_vec = _weight_vector(latest_w, labels)
    rl_live_w = _latest_shadow_weights(RL_LIVE_RUN_ID)
    if not isinstance(rl_live_w, dict) or not rl_live_w:
        rl_mark = load_forward_mark(RL_LIVE_RUN_ID)
        rl_live_w = (rl_mark or {}).get("latest_weights") if rl_mark else None
    if not isinstance(rl_live_w, dict) or not rl_live_w:
        rl_live_w = None
    rl_labels_full = ["Cash"] + rl_labels
    rl_vec = (
        _weight_vector(rl_live_w, rl_labels_full)
        if rl_live_w is not None
        else None
    )

    fetched = False
    times: pd.DatetimeIndex
    o = h = l = c = None  # type: ignore[assignment]
    so = sh = sl = sc = None  # type: ignore[assignment]
    fetch_labels = list(symbols.keys())

    def _load_price_cache() -> tuple[
        pd.DatetimeIndex | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        list[str],
    ]:
        if not price_cache.is_file():
            return None, None, None, None, None, None, None, None, None, []
        try:
            blob = np.load(price_cache, allow_pickle=False)
        except OSError:
            return None, None, None, None, None, None, None, None, None, []
        cached_labels = [str(x) for x in blob["labels"].tolist()] if "labels" in blob.files else []
        if cached_labels != fetch_labels:
            return None, None, None, None, None, None, None, None, None, cached_labels
        return (
            pd.DatetimeIndex(blob["times"]),
            np.asarray(blob["open"], dtype=np.float64),
            np.asarray(blob["high"], dtype=np.float64),
            np.asarray(blob["low"], dtype=np.float64),
            np.asarray(blob["close"], dtype=np.float64),
            np.asarray(blob["spy_open"], dtype=np.float64),
            np.asarray(blob["spy_high"], dtype=np.float64),
            np.asarray(blob["spy_low"], dtype=np.float64),
            np.asarray(blob["spy_close"], dtype=np.float64),
            cached_labels,
        )

    def _apply_holdout(
        t: pd.DatetimeIndex,
        oo: np.ndarray,
        hh: np.ndarray,
        ll: np.ndarray,
        cc: np.ndarray,
        so_: np.ndarray,
        sh_: np.ndarray,
        sl_: np.ndarray,
        sc_: np.ndarray,
        *,
        start: Any,
    ) -> tuple[
        pd.DatetimeIndex,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        str,
    ]:
        """Slice to holdout start; if that empties the series, keep all bars."""
        if start:
            mask = t >= pd.Timestamp(start)
            if bool(mask.any()):
                return (
                    t[mask],
                    oo[mask],
                    hh[mask],
                    ll[mask],
                    cc[mask],
                    so_[mask],
                    sh_[mask],
                    sl_[mask],
                    sc_[mask],
                    str(pd.Timestamp(start).date()),
                )
        # No usable start (or start is past last bar): chart whatever we have.
        inferred = str(pd.Timestamp(t[0]).date()) if len(t) else str(pd.Timestamp.utcnow().date())
        return t, oo, hh, ll, cc, so_, sh_, sl_, sc_, inferred

    if need_fetch:
        # Record the attempt up front so a hung Yahoo session does not cause
        # every subsequent poll to re-enter download.
        _write_stamp(
            {
                **stamp,
                "run_id": rid,
                "prices_attempt_at_unix": now,
                "prices_attempt_at_utc": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
            },
            root,
        )
        # Yahoo 5m tops out near 5 trading days per request; local NPZ keeps
        # older bars so the book can grow past a single session. On reset, pull
        # only from book_start (today). Otherwise pull the trailing ~5d and merge.
        book_ts = pd.Timestamp(book_start).date()
        yahoo_floor = (
            pd.Timestamp.now(tz="America/New_York") - pd.Timedelta(days=5)
        ).date()
        since = str(book_ts if reset_book or book_ts >= yahoo_floor else yahoo_floor)
        from rlbot.forward_mark import call_with_timeout

        cached_pack = _load_price_cache()
        fetch_ok = False
        try:
            # Stock book + RL sleeve can be ~40 symbols; allow a longer pull.
            fetch_timeout = 45.0 if (force_price_refresh or stock_book or reset_book) else 15.0
            times, o, h, l, c, so, sh, sl, sc = call_with_timeout(
                _fetch_intraday_ohlc,
                fetch_timeout,
                symbols,
                since=since,
            )
            fetch_ok = True
        except Exception as exc:
            # Weekend / pre-open / holiday: do NOT replay the prior session.
            # Seed a flat $initial_cash baseline until the next cash open.
            if reset_book:
                return seed_flat_forward_baseline(
                    rid,
                    initial_cash=initial_cash,
                    root=root,
                    include_live_model=True,
                )
            if price_cache.is_file():
                # Fall through to cache / empty handling below.
                need_fetch = False
            elif existing is not None and (existing.get("live") or {}).get("source") == "flat_baseline":
                return _finalize_forward_mark(existing)
            elif existing is not None:
                return _finalize_forward_mark(existing)
            else:
                raise RuntimeError(
                    f"no {BAR_INTERVAL} bars on/after {book_start}: {exc}"
                ) from exc
        if fetch_ok:
            (
                c_times,
                c_o,
                c_h,
                c_l,
                c_c,
                c_so,
                c_sh,
                c_sl,
                c_sc,
                _c_labs,
            ) = cached_pack
            if not reset_book:
                times, o, h, l, c, so, sh, sl, sc = _merge_price_history(
                    cached_times=c_times,
                    cached_o=c_o,
                    cached_h=c_h,
                    cached_l=c_l,
                    cached_c=c_c,
                    cached_so=c_so,
                    cached_sh=c_sh,
                    cached_sl=c_sl,
                    cached_sc=c_sc,
                    times=times,
                    o=o,
                    h=h,
                    l=l,
                    c=c,
                    so=so,
                    sh=sh,
                    sl=sl,
                    sc=sc,
                )
            times, o, h, l, c, so, sh, sl, sc, holdout_start = _apply_holdout(
                times, o, h, l, c, so, sh, sl, sc, start=book_start
            )
            if len(times) < 1:
                raise RuntimeError(
                    f"no {BAR_INTERVAL} bars on/after book_start={book_start} for live forward"
                )
            price_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                price_cache,
                times=times.astype("datetime64[ns]").to_numpy(),
                open=o,
                high=h,
                low=l,
                close=c,
                spy_open=so,
                spy_high=sh,
                spy_low=sl,
                spy_close=sc,
                labels=np.asarray(fetch_labels),
            )
            fetched = True
            _write_stamp(
                {
                    "run_id": rid,
                    "bar_interval": BAR_INTERVAL,
                    "book_start": book_start,
                    "holdout_start": book_start,
                    "prices_fetched_at_unix": now,
                    "prices_fetched_at_utc": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                    "prices_attempt_at_unix": now,
                    "n_bars": int(len(times)),
                    "last_bar": pd.Timestamp(times[-1]).isoformat(timespec="minutes"),
                },
                root,
            )

    if not fetched and price_cache.is_file():
        (
            c_times,
            c_o,
            c_h,
            c_l,
            c_c,
            c_so,
            c_sh,
            c_sl,
            c_sc,
            cached_labels,
        ) = _load_price_cache()
        if cached_labels != fetch_labels or c_times is None or c_o is None:
            need_fetch = True
        else:
            times, o, h, l, c, so, sh, sl, sc = (
                c_times,
                c_o,
                c_h,
                c_l,
                c_c,
                c_so,
                c_sh,
                c_sl,
                c_sc,
            )
            times, o, h, l, c, so, sh, sl, sc, holdout_start = _apply_holdout(
                times, o, h, l, c, so, sh, sl, sc, start=book_start
            )
    if (not fetched) and (need_fetch or not price_cache.is_file() or o is None):
        # Cold start / stale label set — fetch once.
        book_ts = pd.Timestamp(book_start).date()
        yahoo_floor = (
            pd.Timestamp.now(tz="America/New_York") - pd.Timedelta(days=5)
        ).date()
        since = str(book_ts if reset_book or book_ts >= yahoo_floor else yahoo_floor)
        from rlbot.forward_mark import call_with_timeout

        times, o, h, l, c, so, sh, sl, sc = call_with_timeout(
            _fetch_intraday_ohlc,
            45.0 if (force_price_refresh or stock_book or reset_book) else 20.0,
            symbols,
            since=since,
        )
        if not reset_book:
            (
                c_times,
                c_o,
                c_h,
                c_l,
                c_c,
                c_so,
                c_sh,
                c_sl,
                c_sc,
                _c_labs,
            ) = _load_price_cache()
            times, o, h, l, c, so, sh, sl, sc = _merge_price_history(
                cached_times=c_times,
                cached_o=c_o,
                cached_h=c_h,
                cached_l=c_l,
                cached_c=c_c,
                cached_so=c_so,
                cached_sh=c_sh,
                cached_sl=c_sl,
                cached_sc=c_sc,
                times=times,
                o=o,
                h=h,
                l=l,
                c=c,
                so=so,
                sh=sh,
                sl=sl,
                sc=sc,
            )
        times, o, h, l, c, so, sh, sl, sc, holdout_start = _apply_holdout(
            times, o, h, l, c, so, sh, sl, sc, start=book_start
        )
        if len(times) < 1:
            return _finalize_forward_mark(existing) if isinstance(existing, dict) else existing
        price_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            price_cache,
            times=times.astype("datetime64[ns]").to_numpy(),
            open=o,
            high=h,
            low=l,
            close=c,
            spy_open=so,
            spy_high=sh,
            spy_low=sl,
            spy_close=sc,
            labels=np.asarray(fetch_labels),
        )
        fetched = True
        _write_stamp(
            {
                "run_id": rid,
                "bar_interval": BAR_INTERVAL,
                "book_start": book_start,
                "holdout_start": book_start,
                "prices_fetched_at_unix": now,
                "prices_fetched_at_utc": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "prices_attempt_at_unix": now,
                "n_bars": int(len(times)),
                "last_bar": pd.Timestamp(times[-1]).isoformat(timespec="minutes"),
            },
            root,
        )

    if o is None or len(times) < 1:
        return _finalize_forward_mark(existing) if isinstance(existing, dict) else existing

    lab_index = {lab: i for i, lab in enumerate(fetch_labels)}

    def _cols(labs: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = [lab_index[x] for x in labs if x in lab_index]
        if not idx:
            empty = np.zeros((len(times), 0), dtype=np.float64)
            return empty, empty, empty, empty
        return o[:, idx], h[:, idx], l[:, idx], c[:, idx]

    if stock_book:
        book_labs = list(stock_symbols.keys())
        o_book, h_book, l_book, c_book = _cols(book_labs)
        model_ohlc = portfolio_ohlc_candles(
            o_book,
            h_book,
            l_book,
            c_book,
            live_vec,
            initial_cash=initial_cash,
            cash_yield_per_bar=cash_yield_bar,
        )
        last_closes_book = c_book[-1] if c_book.shape[1] else None
    else:
        model_ohlc = portfolio_ohlc_candles(
            o, h, l, c, live_vec, initial_cash=initial_cash, cash_yield_per_bar=cash_yield_bar
        )
        last_closes_book = c[-1]

    # Equal-weight is always the original N-asset research sleeve (not stock picks).
    o_rl, h_rl, l_rl, c_rl = _cols(rl_labels)
    ew_ohlc = equal_weight_ohlc_candles(o_rl, h_rl, l_rl, c_rl, initial_cash=initial_cash)
    spy_ohlc = spy_ohlc_candles(so, sh, sl, sc, initial_cash=initial_cash)

    live_model_ohlc = None
    if rl_vec is not None and o_rl.shape[1] == len(rl_labels):
        live_model_ohlc = portfolio_ohlc_candles(
            o_rl,
            h_rl,
            l_rl,
            c_rl,
            rl_vec,
            initial_cash=initial_cash,
            cash_yield_per_bar=rl_cash_yield_bar,
        )

    nav_model = _nav_series_from_ohlc(model_ohlc, initial_cash=initial_cash)
    nav_ew = _nav_series_from_ohlc(ew_ohlc, initial_cash=initial_cash)
    nav_spy = _nav_series_from_ohlc(spy_ohlc, initial_cash=initial_cash)
    nav_live = (
        _nav_series_from_ohlc(live_model_ohlc, initial_cash=initial_cash)
        if live_model_ohlc is not None
        else None
    )

    w_mat = np.tile(live_vec, (len(times), 1))
    candles_payload: dict[str, Any] = {
        "model": _candles_to_payload(times, model_ohlc),
        "equal_weight": _candles_to_payload(times, ew_ohlc),
        "spy": _candles_to_payload(times, spy_ohlc),
    }
    if live_model_ohlc is not None:
        candles_payload["live_model"] = _candles_to_payload(times, live_model_ohlc)

    note = (
        f"GENERAL_EQUITY1 (GeneralEquity1) + LIVE_MODEL RL sleeve since {book_start}. "
        "Equal-weight is the original 10-asset research universe (not algo ETF weights). "
        if stock_book
        else f"Live 5-minute NAV candles on the SPY session clock since {book_start} "
        "from latest target weights + Yahoo OHLC (model / EW / SPY). "
        "Foreign-session assets carry the last print forward. "
    ) + (
        f"History accumulates in execution/ (Yahoo 5m lookback ~5d per pull). "
        f"Prices refresh about every {max(min_refresh_seconds, 60) // 60} min."
    )

    payload = build_forward_mark_payload(
        run_id=rid,
        checkpoint_label=str(
            (existing or {}).get("checkpoint_label")
            or ("locked" if stock_book else "best")
        ),
        dates=times,
        nav_model=nav_model,
        nav_spy=nav_spy,
        nav_ew=nav_ew,
        weights=w_mat,
        asset_labels=labels,
        initial_cash=initial_cash,
        holdout_start=str(book_start),
        holdout_end=str(holdout_end) if holdout_end else None,
        note=note,
        bar_interval=BAR_INTERVAL,
        timestamps=[pd.Timestamp(t).isoformat(timespec="minutes") for t in times],
        candles=candles_payload,
        bars_per_year=BARS_PER_TRADING_DAY * 252,
        nav_live_model=nav_live,
    )
    payload["book_start"] = str(book_start)
    positions = _positions_snapshot(
        latest_w,
        nav=float(nav_model[-1]),
        labels=labels,
        last_closes=last_closes_book,
        tickers=tickers,
    )
    payload["positions"] = positions
    payload["companion_run_id"] = RL_LIVE_RUN_ID if nav_live is not None else None
    payload["live"] = {
        "prices_refreshed": fetched,
        "as_of_bar": pd.Timestamp(times[-1]).isoformat(timespec="minutes"),
        "as_of_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "min_refresh_seconds": int(min_refresh_seconds),
        "bar_interval": BAR_INTERVAL,
        "source": "yfinance_5m",
        "book_start": str(book_start),
        "session_start": str(book_start),
        "reset_book": bool(reset_book),
    }
    payload["latest_weights"] = {
        labels[i]: float(live_vec[i]) for i in range(len(labels))
    }
    return _finalize_forward_mark(
        payload,
        model_positions=positions,
        model_price_source="yahoo",
        write=True,
    )
