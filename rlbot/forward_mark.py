"""Forward-mark artifacts for the ops live dashboard (measurement only).

Writes a JSON series of model / SPY / equal-weight NAVs (and optional weights)
so the developer console can chart post-deploy performance without re-running
torch inference in the API process.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from rlbot.run_artifacts import RunPaths

FORWARD_MARK_NAME = "forward_mark.json"
ACTIVE_POINTER_NAME = "forward_active.json"


def _json_safe(obj: Any) -> Any:
    """Replace NaN/Inf with null so ``json.dumps(..., allow_nan=False)`` succeeds."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    return obj


def call_with_timeout(fn, timeout_s: float, /, *args, **kwargs):
    """Run ``fn`` in a worker thread; never block on shutdown if it overruns.

    ``concurrent.futures.ThreadPoolExecutor``'s context manager defaults to
    ``shutdown(wait=True)``, so a timed-out Yahoo/iCloud worker previously pinned
    the caller forever — which the browser surfaces as a Network/CORS failure.
    """
    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(fn, *args, **kwargs).result(timeout=float(timeout_s))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def forward_mark_path(run_id: str) -> Path:
    return RunPaths(run_id).run_meta_dir / FORWARD_MARK_NAME


def execution_forward_mark_path(run_id: str, root: Path | None = None) -> Path:
    """Local (non-iCloud) mirror — preferred for the operator API to avoid Desktop hangs."""
    from rlbot.run_artifacts import PROJECT_ROOT

    base = root or PROJECT_ROOT
    return base / "execution" / f"forward_mark_{run_id}.json"


def active_pointer_path(root: Path | None = None) -> Path:
    from rlbot.run_artifacts import PROJECT_ROOT

    base = root or PROJECT_ROOT
    return base / "execution" / ACTIVE_POINTER_NAME


def _max_drawdown(navs: np.ndarray) -> float:
    x = np.asarray(navs, dtype=np.float64).reshape(-1)
    if x.size < 2:
        return 0.0
    peak = np.maximum.accumulate(x)
    dd = x / np.maximum(peak, 1e-12) - 1.0
    return float(dd.min())


def _series_stats(
    navs: Sequence[float],
    *,
    bars_per_year: float = 252.0,
    timestamps: Sequence[Any] | None = None,
) -> dict[str, float | None]:
    arr = np.asarray(navs, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        return {
            "total_return": 0.0 if arr.size else None,
            "sharpe": None,
            "max_drawdown": 0.0 if arr.size else None,
            "nav": float(arr[-1]) if arr.size else None,
        }

    sharpe: float | None = None
    # Intraday grids: annualizing raw 5m log-returns over a few sessions produces
    # absurd Sharpes (often 5–10×). Resample to daily last-NAV and require ~1
    # trading month before reporting an annualized figure.
    if float(bars_per_year) > 1000 and timestamps is not None and len(timestamps) == arr.size:
        daily_navs: list[float] = []
        last_day: str | None = None
        for ts, nav in zip(timestamps, arr):
            day = str(pd.Timestamp(ts).date())
            if day != last_day:
                daily_navs.append(float(nav))
                last_day = day
            else:
                daily_navs[-1] = float(nav)
        daily = np.asarray(daily_navs, dtype=np.float64)
        if daily.size >= 21:
            log_rets = np.diff(np.log(np.maximum(daily, 1e-12)))
            std = float(np.std(log_rets, ddof=1)) if log_rets.size > 1 else 0.0
            if std > 1e-5:
                cand = float(np.mean(log_rets) / std * np.sqrt(252.0))
                if np.isfinite(cand) and abs(cand) <= 5.0:
                    sharpe = cand
    elif float(bars_per_year) <= 1000:
        log_rets = np.diff(np.log(np.maximum(arr, 1e-12)))
        if log_rets.size >= 19:
            std = float(np.std(log_rets, ddof=1)) if log_rets.size > 1 else 0.0
            if std > 1e-5:
                cand = float(np.mean(log_rets) / std * np.sqrt(252.0))
                if np.isfinite(cand) and abs(cand) <= 5.0:
                    sharpe = cand

    return {
        "total_return": float(arr[-1] / max(arr[0], 1e-12) - 1.0),
        "sharpe": sharpe,
        "max_drawdown": _max_drawdown(arr),
        "nav": float(arr[-1]),
    }


def _format_bar_label(d: Any, *, bar_interval: str | None) -> str:
    ts = pd.Timestamp(d)
    if bar_interval and bar_interval != "1d":
        return ts.isoformat(timespec="minutes")
    return str(ts.date())


def build_forward_mark_payload(
    *,
    run_id: str,
    checkpoint_label: str,
    dates: Sequence[Any],
    nav_model: np.ndarray,
    nav_spy: np.ndarray,
    nav_ew: np.ndarray,
    weights: np.ndarray | None,
    asset_labels: Sequence[str],
    initial_cash: float,
    holdout_start: str | None,
    holdout_end: str | None,
    note: str = "",
    bar_interval: str | None = None,
    timestamps: Sequence[str] | None = None,
    candles: dict[str, Any] | None = None,
    bars_per_year: float = 252.0,
    nav_live_model: np.ndarray | None = None,
    nav_crypto: np.ndarray | None = None,
    nav_durable: np.ndarray | None = None,
    nav_core_equity: np.ndarray | None = None,
) -> dict[str, Any]:
    """Assemble a browser-friendly forward-mark payload (NAVs start at ``initial_cash``)."""
    model = np.asarray(nav_model, dtype=np.float64).reshape(-1)
    spy = np.asarray(nav_spy, dtype=np.float64).reshape(-1)
    ew = np.asarray(nav_ew, dtype=np.float64).reshape(-1)
    live = (
        np.asarray(nav_live_model, dtype=np.float64).reshape(-1)
        if nav_live_model is not None
        else None
    )
    crypto = (
        np.asarray(nav_crypto, dtype=np.float64).reshape(-1)
        if nav_crypto is not None
        else None
    )
    durable = (
        np.asarray(nav_durable, dtype=np.float64).reshape(-1)
        if nav_durable is not None
        else None
    )
    core_eq = (
        np.asarray(nav_core_equity, dtype=np.float64).reshape(-1)
        if nav_core_equity is not None
        else None
    )
    n = int(min(model.size, spy.size, ew.size, len(dates)))
    if live is not None:
        n = int(min(n, live.size))
    if crypto is not None:
        n = int(min(n, crypto.size))
    if durable is not None:
        n = int(min(n, durable.size))
    if core_eq is not None:
        n = int(min(n, core_eq.size))
    if n < 1:
        raise ValueError("forward mark requires at least one NAV point")

    if candles is not None:
        # Intraday candle builders already emit cash-unit OHLC / closes.
        model_s = model[:n].tolist()
        spy_s = spy[:n].tolist()
        ew_s = ew[:n].tolist()
        live_s = live[:n].tolist() if live is not None else None
        crypto_s = crypto[:n].tolist() if crypto is not None else None
        durable_s = durable[:n].tolist() if durable is not None else None
        core_eq_s = core_eq[:n].tolist() if core_eq is not None else None
    else:
        scale = float(initial_cash) / max(float(model[0]), 1e-12)
        model_s = (model[:n] * scale).tolist()
        # Rebase SPY/EW to the same starting cash for apples-to-apples charting.
        spy_s = (spy[:n] / max(float(spy[0]), 1e-12) * float(initial_cash)).tolist()
        ew_s = (ew[:n] / max(float(ew[0]), 1e-12) * float(initial_cash)).tolist()
        live_s = (
            (live[:n] / max(float(live[0]), 1e-12) * float(initial_cash)).tolist()
            if live is not None and live.size
            else None
        )
        crypto_s = (
            (crypto[:n] / max(float(crypto[0]), 1e-12) * float(initial_cash)).tolist()
            if crypto is not None and crypto.size
            else None
        )
        durable_s = (
            (durable[:n] / max(float(durable[0]), 1e-12) * float(initial_cash)).tolist()
            if durable is not None and durable.size
            else None
        )
        core_eq_s = (
            (core_eq[:n] / max(float(core_eq[0]), 1e-12) * float(initial_cash)).tolist()
            if core_eq is not None and core_eq.size
            else None
        )
    if timestamps is not None and len(timestamps) >= n:
        date_strs = [str(timestamps[i]) for i in range(n)]
    else:
        date_strs = [
            _format_bar_label(d, bar_interval=bar_interval) for d in list(dates)[:n]
        ]

    weight_rows: list[dict[str, float]] | None = None
    latest_weights: dict[str, float] | None = None
    if weights is not None and np.asarray(weights).size > 0:
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim == 1:
            w = w.reshape(1, -1)
        m = min(n, w.shape[0], len(asset_labels))
        labels = list(asset_labels)[: w.shape[1]]
        weight_rows = [
            {labels[j]: float(w[i, j]) for j in range(min(len(labels), w.shape[1]))}
            for i in range(m)
        ]
        # Align length with dates (pad/truncate).
        if len(weight_rows) < n:
            weight_rows.extend([weight_rows[-1]] * (n - len(weight_rows)))
        else:
            weight_rows = weight_rows[:n]
        latest_weights = weight_rows[-1]

    payload: dict[str, Any] = {
        "schema": "markettrainer.forward_mark.v2"
        if bar_interval
        else "markettrainer.forward_mark.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "checkpoint_label": checkpoint_label,
        "initial_cash": float(initial_cash),
        "holdout_start": holdout_start,
        "holdout_end": holdout_end,
        "n_bars": n,
        "dates": date_strs,
        "nav": {
            "model": model_s,
            "spy": spy_s,
            "equal_weight": ew_s,
        },
        "stats": {
            "model": _series_stats(
                model_s, bars_per_year=bars_per_year, timestamps=date_strs
            ),
            "spy": _series_stats(
                spy_s, bars_per_year=bars_per_year, timestamps=date_strs
            ),
            "equal_weight": _series_stats(
                ew_s, bars_per_year=bars_per_year, timestamps=date_strs
            ),
        },
        "latest_weights": latest_weights,
        "weights": weight_rows,
        "asset_labels": list(asset_labels),
        "note": note
        or (
            "Forward OOS measurement with frozen VecNormalize; linear costs only "
            "(no market-impact / capacity model). % returns are scale-invariant "
            "under this cost model (10k vs 100k)."
        ),
    }
    if live_s is not None:
        payload["nav"]["live_model"] = live_s
        payload["stats"]["live_model"] = _series_stats(
            live_s, bars_per_year=bars_per_year, timestamps=date_strs
        )
    if crypto_s is not None:
        payload["nav"]["crypto"] = crypto_s
        payload["stats"]["crypto"] = _series_stats(
            crypto_s, bars_per_year=bars_per_year, timestamps=date_strs
        )
    if durable_s is not None:
        payload["nav"]["durable"] = durable_s
        payload["stats"]["durable"] = _series_stats(
            durable_s, bars_per_year=bars_per_year, timestamps=date_strs
        )
    if core_eq_s is not None:
        payload["nav"]["core_equity"] = core_eq_s
        payload["stats"]["core_equity"] = _series_stats(
            core_eq_s, bars_per_year=bars_per_year, timestamps=date_strs
        )
    if bar_interval:
        payload["bar_interval"] = bar_interval
    if timestamps is not None:
        payload["timestamps"] = [str(timestamps[i]) for i in range(min(n, len(timestamps)))]
    if candles is not None:
        payload["candles"] = candles
    return payload


CRYPTO_COMPANION_RUN_ID = "CREST_DAY"
DURABLE_COMPANION_RUN_ID = "DURABLE_V1"


def _merge_companion_nav(
    mark: dict[str, Any],
    *,
    nav_key: str,
    run_id: str,
    companion_field: str,
) -> dict[str, Any]:
    """Attach a disk companion ``nav.<nav_key>`` when missing (no Yahoo)."""
    if not isinstance(mark, dict):
        return mark
    nav = mark.get("nav") if isinstance(mark.get("nav"), dict) else {}
    existing = nav.get(nav_key)
    n = int(mark.get("n_bars") or len(mark.get("dates") or []) or 0)
    if isinstance(existing, list) and len(existing) >= max(2, min(n, 2)):
        return mark
    companion_mark = load_forward_mark(run_id)
    if not isinstance(companion_mark, dict):
        return mark
    companion_nav = (companion_mark.get("nav") or {}).get("model")
    if not isinstance(companion_nav, list) or len(companion_nav) < 1:
        return mark
    initial = float(mark.get("initial_cash") or companion_mark.get("initial_cash") or 100_000.0)
    src = np.asarray(companion_nav, dtype=np.float64)
    if src.size < 1 or not np.isfinite(src[0]) or src[0] <= 0:
        return mark
    rebased = src / float(src[0]) * initial
    if n < 1:
        n = int(rebased.size)
    out = np.empty(n, dtype=np.float64)
    if rebased.size >= n:
        out[:] = rebased[-n:]
    else:
        out[: rebased.size] = rebased
        out[rebased.size :] = rebased[-1]
    dates = mark.get("dates") or mark.get("timestamps") or []
    date_strs = [str(d) for d in list(dates)[:n]] if dates else None
    bars_per_year = 78.0 * 252.0 if mark.get("bar_interval") in ("5m", "30m") else 252.0
    series = out.tolist()
    next_nav = {**nav, nav_key: series}
    next_stats = dict(mark.get("stats") or {})
    next_stats[nav_key] = _series_stats(
        series, bars_per_year=bars_per_year, timestamps=date_strs
    )
    return {
        **mark,
        "nav": next_nav,
        "stats": next_stats,
        companion_field: run_id,
    }


def merge_crypto_companion(mark: dict[str, Any]) -> dict[str, Any]:
    """Attach CrestDay companion when missing (disk-only). Durable.v1 is retired."""
    return _merge_companion_nav(
        mark,
        nav_key="crypto",
        run_id=CRYPTO_COMPANION_RUN_ID,
        companion_field="companion_crypto_run_id",
    )


def write_forward_mark(payload: dict[str, Any], path: Path | None = None) -> Path:
    """Persist the mark to the local ``execution/`` mirror only (API-safe).

    Deliberately skips ``Runs/<id>/forward_mark.json`` — on iCloud Desktop that
    path routinely hangs forever. Pass ``path`` only for explicit offline copies
    to a known-local directory (tests / non-iCloud roots).
    """
    run_id = str(payload["run_id"])
    # allow_nan=False: browsers reject literal NaN; coerce non-finite floats first.
    safe = _json_safe(payload)
    text = json.dumps(safe, indent=2, default=str, allow_nan=False)
    local = execution_forward_mark_path(run_id)
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(text, encoding="utf-8")
    if path is not None and path.resolve() != local.resolve():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path
    return local


def set_active_forward_run(run_id: str, *, root: Path | None = None) -> Path:
    ptr = active_pointer_path(root)
    ptr.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    ptr.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return ptr


def resolve_active_forward_run_id(root: Path | None = None) -> str | None:
    """Resolve active LIVE run without scanning iCloud-backed ``Runs/``."""
    ptr = active_pointer_path(root)
    if ptr.is_file():
        try:
            data = json.loads(ptr.read_text(encoding="utf-8"))
            rid = str(data.get("run_id") or "").strip()
            if rid:
                return rid
        except (OSError, json.JSONDecodeError):
            pass
    # Fallback: newest local execution mirror (never touch Runs/ here).
    from rlbot.run_artifacts import PROJECT_ROOT

    base = root or PROJECT_ROOT
    exec_dir = base / "execution"
    if not exec_dir.is_dir():
        return None
    candidates: list[tuple[float, str]] = []
    for path in exec_dir.glob("forward_mark_*.json"):
        name = path.name.removeprefix("forward_mark_").removesuffix(".json")
        # LIVE_* RL deploy marks, GeneralEquity1 / CrestDay, or legacy algo ids.
        if name.startswith("LIVE_") or name in {
            "RLModel",
            "CORE_EQUITY",
            "GENERAL_EQUITY1",
            "GENERAL_EQUITY",
            "CREST_DAY",
            "DURABLE_V1",
            "PROD_RETURN_ALPHA",
            "FINALMODEL",
        }:
            try:
                candidates.append((path.stat().st_mtime, name))
            except OSError:
                continue
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def load_forward_mark(run_id: str) -> dict[str, Any] | None:
    """Load mark from the local ``execution/`` mirror only.

    Deliberately does **not** read ``Runs/<id>/forward_mark.json`` — on iCloud
    Desktop that path routinely hangs and takes down the operator API.
    """
    path = execution_forward_mark_path(run_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
