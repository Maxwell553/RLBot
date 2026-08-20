#!/usr/bin/env python3
"""Stdlib-only operator API fallback (no FastAPI / heavy rlbot imports).

Serves health / runs / summary / forward from ``execution/`` plus bounded reads
of ``Runs/*/backtest_summary*.json`` (subprocess + timeout) so the UI keeps
OOS metrics even when the full FastAPI app cannot import under iCloud load.

    python3 scripts/frontend_api_lite.py --port 8787
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlencode, urlparse
from urllib.request import Request, urlopen

ROOT = Path(os.environ.get("MARKETTRAINER_ROOT") or Path(__file__).resolve().parent.parent).resolve()
EXEC = ROOT / "execution"
RUNS = ROOT / "Runs"
RUNS_CACHE = EXEC / "api_runs_cache.json"
OOS_CACHE = EXEC / "api_oos_cache.json"
ACTIVE_PTR = EXEC / "forward_active.json"
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
# W{window}_{cohort}[letter] or W{window}_{cohort}_s{seed} (811a/b variants, seed ensembles).
_COHORT_RUN_RE = re.compile(r"^(W\d+)_(\d+[a-z]*)(?:_[A-Za-z0-9]+)?$", re.IGNORECASE)
_WINDOW_COHORT_RE = re.compile(r"^W(\d+)_(\d+)([a-z]*)(?:_[A-Za-z0-9]+)?$", re.IGNORECASE)
_CKPT_STEP_RE = re.compile(r"^ppo_(\d+)_steps\.zip$")
_DEFAULT_NOMINAL_STEPS = 50_000_000
# No checkpoint/manifest updates for this long → treat as interrupted (not active).
_STALE_ACTIVE_S = 3 * 3600

_CACHE_TTL_S = 60.0
_rows_lock = threading.Lock()
_rows_cache: list[dict[str, Any]] | None = None
_rows_at = 0.0
_enrich_lock = threading.Lock()
_enriching = False
_forward_refresh_lock = threading.Lock()
_forward_refreshing = False
_PUBLIC_DATA = ROOT / "frontend" / "public" / "data"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_sort_key(run_id: str) -> tuple[Any, ...]:
    """Newest numeric cohort first, then W1…W5 (seed suffixes sort after)."""
    m = _WINDOW_COHORT_RE.match(run_id)
    if m is None:
        return (1, 0, 0, "", run_id)
    window = int(m.group(1))
    cohort = int(m.group(2))
    letter = (m.group(3) or "").lower()
    return (0, -cohort, window, letter, run_id)


def _sort_run_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda r: _run_sort_key(str(r.get("run_id") or "")))


def _listdir_timed(path: Path, timeout_s: float = 0.6) -> list[str]:
    try:
        proc = subprocess.run(
            ["/bin/ls", "-1", str(path)],
            capture_output=True,
            timeout=float(timeout_s),
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.decode("utf-8", errors="replace").splitlines() if line]


def _path_mtime(path: Path, timeout_s: float = 0.4) -> float | None:
    try:
        proc = subprocess.run(
            ["/usr/bin/stat", "-f", "%m", str(path)],
            capture_output=True,
            timeout=float(timeout_s),
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.decode().strip())
    except ValueError:
        return None


def _checkpoint_progress(run_id: str) -> tuple[int | None, float | None]:
    """Return (max_checkpoint_steps, newest_mtime) from models/checkpoints."""
    base = RUNS / run_id / "models" / "checkpoints"
    names = _listdir_timed(base)
    best: int | None = None
    newest: float | None = None
    for name in names:
        m = _CKPT_STEP_RE.match(name)
        if m is None:
            continue
        step = int(m.group(1))
        best = step if best is None else max(best, step)
        mt = _path_mtime(base / name)
        if mt is not None:
            newest = mt if newest is None else max(newest, mt)
    return best, newest


def _has_final_model(run_id: str) -> bool:
    models = _listdir_timed(RUNS / run_id / "models")
    for name in models:
        lower = name.lower()
        if "final" in lower and lower.endswith(".zip"):
            return True
    return False


def _infer_run_status(
    *,
    explicit: str | None,
    has_backtest: bool,
    has_final: bool,
    checkpoint_steps: int | None,
    checkpoint_mtime: float | None,
    finished_at: Any,
) -> str:
    status = (explicit or "").strip()
    # A scored backtest / final weights / finished stamp means training is done
    # for the UI — even if the manifest briefly still says ``active`` during the
    # post-train backtest race (otherwise Runs shows "active" + OOS Sharpe).
    if status == "interrupted" and not (has_backtest or has_final):
        return "interrupted"
    if has_backtest or has_final or finished_at or status == "completed":
        return "completed"
    if status == "active":
        return "active"
    if checkpoint_steps is None and not has_final:
        # Bare run dir / abandoned early — not actively training.
        if checkpoint_mtime is None:
            return "interrupted"
    if checkpoint_mtime is not None and (time.time() - checkpoint_mtime) > _STALE_ACTIVE_S:
        return "interrupted"
    if checkpoint_steps is not None:
        return "active"
    return "interrupted" if explicit in ("", None) else "active"


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _timed_read_text(path: Path, timeout_s: float = 2.5) -> str | None:
    """Read a file with a hard timeout (survives iCloud stalls).

    Prefer a direct ``read_text`` in a worker thread — ``/bin/cat`` at ≤1s was
    falsely missing ~2KB ``backtest_summary.json`` files that take ~1–1.3s to
    hydrate from iCloud Desktop.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return None

    if size <= 1_000_000:
        box: dict[str, Any] = {"text": None}

        def _worker() -> None:
            try:
                box["text"] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                box["text"] = None

        thread = threading.Thread(target=_worker, name="lite-read", daemon=True)
        thread.start()
        thread.join(timeout=float(timeout_s))
        if thread.is_alive():
            return None
        text = box.get("text")
        return text if isinstance(text, str) else None

    try:
        proc = subprocess.run(
            ["/bin/cat", str(path)],
            capture_output=True,
            timeout=float(timeout_s),
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    try:
        return proc.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _timed_read_json(path: Path, timeout_s: float = 2.5) -> Any | None:
    text = _timed_read_text(path, timeout_s=timeout_s)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _read_json(path: Path) -> Any | None:
    try:
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _active_run_id() -> str | None:
    data = _read_json(ACTIVE_PTR)
    if isinstance(data, dict):
        rid = str(data.get("run_id") or "").strip()
        if rid:
            return rid
    marks = sorted(
        EXEC.glob("forward_mark_LIVE_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if marks:
        return marks[0].name.removeprefix("forward_mark_").removesuffix(".json")
    return None


def _load_mark(run_id: str) -> dict[str, Any] | None:
    data = _read_json(EXEC / f"forward_mark_{run_id}.json")
    return data if isinstance(data, dict) else None


def _mark_age_s(run_id: str) -> float | None:
    path = EXEC / f"forward_mark_{run_id}.json"
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        return None


def _prices_age_s(run_id: str) -> float | None:
    """Seconds since the last successful Yahoo price fetch (stamp), not mark mtime.

    Clock-touch rewrites the mark file every poll, so mtime cannot mean "fresh prices".
    """
    stamp = _read_json(EXEC / "forward_live_stamp.json")
    if not isinstance(stamp, dict):
        return None
    if str(stamp.get("run_id") or "") and str(stamp.get("run_id")) != run_id:
        # Stamp is for another book — treat as unknown / stale.
        return None
    fetched = stamp.get("prices_fetched_at_unix")
    try:
        ts = float(fetched)
    except (TypeError, ValueError):
        return None
    if ts <= 0:
        return None
    return max(0.0, time.time() - ts)


def _parse_mark_ts(value: str) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is not None:
            dt = dt.astimezone().replace(tzinfo=None)
        return dt
    except ValueError:
        return None


def _floor_5m(dt: datetime) -> datetime:
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)


def _in_us_cash_session(dt: datetime) -> bool:
    """Mon–Fri 09:30–16:00 ET (naive Eastern wall clock)."""
    if dt.weekday() >= 5:
        return False
    minutes = dt.hour * 60 + dt.minute
    return 9 * 60 + 30 <= minutes < 16 * 60


def _next_rth_open_after(dt: datetime) -> datetime:
    cursor = _floor_5m(dt) + timedelta(minutes=5)
    for _ in range(8 * 24 * 12):
        if _in_us_cash_session(cursor):
            return cursor
        cursor = cursor + timedelta(minutes=5)
    return dt + timedelta(days=5)


def _offhours_extend_until(last: datetime, now: datetime) -> datetime | None:
    """Cap invented bars before the next cash open; never across a missed session."""
    last = _floor_5m(last)
    now = _floor_5m(now)
    if now <= last:
        return None
    if _in_us_cash_session(now):
        return None
    nxt = _next_rth_open_after(last)
    cap = min(now, nxt - timedelta(minutes=5))
    if cap <= last:
        return None
    return cap


def _prices_are_stale(last: datetime, now: datetime) -> bool:
    last = _floor_5m(last)
    now = _floor_5m(now)
    if now <= last:
        return False
    if _in_us_cash_session(now):
        return (now - last) > timedelta(minutes=10)
    return _next_rth_open_after(last) <= now


def _stamp_last_price_bar() -> datetime | None:
    stamp = _read_json(EXEC / "forward_live_stamp.json")
    if not isinstance(stamp, dict):
        return None
    return _parse_mark_ts(str(stamp.get("last_bar") or ""))


def _trim_mark_to_last_price(payload: dict[str, Any], last_price: datetime) -> dict[str, Any]:
    """Drop invented timestamps after the last real Yahoo bar.

    Clock-touch used to append flat equity NAV across missed cash sessions.
    Existing marks keep those bars until a successful refresh; trim them so
    the chart stops lying while Yahoo backfills.
    """
    stamps = payload.get("timestamps") or payload.get("dates") or []
    if not isinstance(stamps, list) or not stamps:
        return payload
    keep = 0
    for i, raw in enumerate(stamps):
        ts = _parse_mark_ts(str(raw))
        if ts is None or ts <= last_price:
            keep = i + 1
        else:
            break
    if keep < 1 or keep >= len(stamps):
        return payload
    payload = dict(payload)
    payload["timestamps"] = stamps[:keep]
    payload["dates"] = list(payload["timestamps"])
    payload["n_bars"] = keep
    nav = dict(payload.get("nav") or {})
    for key, series in list(nav.items()):
        if isinstance(series, list) and series:
            nav[key] = series[:keep]
    payload["nav"] = nav
    candles = payload.get("candles")
    if isinstance(candles, dict):
        trimmed = dict(candles)
        for key, rows in list(trimmed.items()):
            if isinstance(rows, list):
                trimmed[key] = rows[:keep]
        payload["candles"] = trimmed
    weights = payload.get("weights")
    if isinstance(weights, list) and weights:
        payload["weights"] = weights[:keep]
    return payload


def _extend_nav_hold(series: list[Any], n_add: int) -> list[float]:
    vals = [float(v) for v in series if isinstance(v, (int, float))]
    if not vals or n_add < 1:
        return [float(v) for v in series] if isinstance(series, list) else []
    tip = vals[-1]
    return vals + [tip] * int(n_add)


def _touch_forward_clock(run_id: str, mark: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Bump freshness metadata without inventing flat equity bars across RTH.

    Overnight / weekend we may hold the last print so CrestDay's 24/7 tip can
    advance — but only up to the next cash open, and never when the last real
    Yahoo bar is mid-session stale (that was the multi-day flatline bug).
    """
    payload = dict(mark) if isinstance(mark, dict) else _load_mark(run_id)
    if not isinstance(payload, dict):
        return None
    payload = _strip_durable_series(payload)
    stamps = payload.get("timestamps") or payload.get("dates") or []
    if not isinstance(stamps, list) or not stamps:
        return _attach_live_allocations(payload)
    last = _parse_mark_ts(str(stamps[-1]))
    if last is None:
        return _attach_live_allocations(payload)
    try:
        from zoneinfo import ZoneInfo

        now = datetime.now(ZoneInfo("America/New_York")).replace(tzinfo=None)
    except Exception:  # noqa: BLE001
        now = datetime.now()
    now = _floor_5m(now)
    live = dict(payload.get("live") or {})
    live["as_of_utc"] = _now()
    payload["generated_at_utc"] = live["as_of_utc"]
    last_price = (
        _parse_mark_ts(str(live.get("last_price_bar") or ""))
        or _stamp_last_price_bar()
        or last
    )
    live["last_price_bar"] = last_price.isoformat(timespec="minutes")
    live["prices_stale"] = _prices_are_stale(last_price, now)
    if last_price < last:
        payload = _trim_mark_to_last_price(payload, last_price)
        stamps = payload.get("timestamps") or payload.get("dates") or []
        last = _parse_mark_ts(str(stamps[-1])) if stamps else last_price
        if last is None:
            last = last_price

    until = _offhours_extend_until(last_price, now)
    # RTH or missed session: never invent equity bars — wait for Yahoo.
    if until is None or now <= last:
        live["as_of_bar"] = str(stamps[-1])
        live["clock_touch"] = "meta"
        payload["live"] = live
        try:
            path = EXEC / f"forward_mark_{run_id}.json"
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            payload = _attach_live_allocations(payload)
            text = json.dumps(payload, indent=2, default=str, allow_nan=False)
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(path)
        except Exception as exc:  # noqa: BLE001
            print(f"[lite-api] clock-touch write skipped: {exc}", file=sys.stderr, flush=True)
        _write_public_forward(run_id, payload)
        return payload

    extra: list[str] = []
    cursor = last + timedelta(minutes=5)
    max_extra = 3 * 24 * 12
    while cursor <= until and len(extra) < max_extra:
        if not _in_us_cash_session(cursor):
            extra.append(cursor.isoformat(timespec="minutes"))
        cursor = cursor + timedelta(minutes=5)
    if extra:
        new_stamps = [str(s) for s in stamps] + extra
        n_add = len(extra)
        payload["timestamps"] = new_stamps
        payload["dates"] = new_stamps
        payload["n_bars"] = len(new_stamps)
        nav = dict(payload.get("nav") or {})
        for key, series in list(nav.items()):
            if isinstance(series, list) and series:
                nav[key] = _extend_nav_hold(series, n_add)
        payload["nav"] = nav
        stats = dict(payload.get("stats") or {})
        for key, series in nav.items():
            if not isinstance(series, list) or not series:
                continue
            tip = float(series[-1])
            base = float(series[0]) if series[0] else tip
            ret = (tip / base - 1.0) if base else 0.0
            prev = stats.get(key) if isinstance(stats.get(key), dict) else {}
            stats[key] = {
                **prev,
                "nav": tip,
                "total_return": ret,
            }
        payload["stats"] = stats
        candles = payload.get("candles")
        if isinstance(candles, dict):
            new_candles = dict(candles)
            for key, series in nav.items():
                if not isinstance(series, list) or len(series) != len(new_stamps):
                    continue
                rows = []
                for i, ts in enumerate(new_stamps):
                    v = float(series[i])
                    rows.append({"t": ts, "o": v, "h": v, "l": v, "c": v})
                new_candles[key] = rows
            payload["candles"] = new_candles
        live["as_of_bar"] = new_stamps[-1]
        live["crypto_clock"] = "24_7"
        live["clock_touch"] = "lite_offhours"
    payload["live"] = live
    payload = _attach_live_allocations(payload)
    # Persist so /data/forward.json and the next soft poll stay fresh.
    try:
        path = EXEC / f"forward_mark_{run_id}.json"
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        text = json.dumps(payload, indent=2, default=str, allow_nan=False)
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:  # noqa: BLE001
        print(f"[lite-api] clock-touch write skipped: {exc}", file=sys.stderr, flush=True)
    _write_public_forward(run_id, payload)
    return payload


def _normalize_run_row(row: dict[str, Any]) -> dict[str, Any]:
    """Guarantee arrays the React UI indexes with ``.length``."""
    out = dict(row)
    if not isinstance(out.get("labels"), list):
        out["labels"] = []
    if not isinstance(out.get("warnings"), list):
        out["warnings"] = []
    return out


def _list_run_ids() -> list[str]:
    try:
        names = os.listdir(RUNS)
    except OSError:
        return []
    out: list[str] = []
    for name in sorted(names):
        if name.startswith(".") or "." in name:
            continue
        if _RUN_ID_RE.match(name):
            out.append(name)
    return out


def _pick_backtest(run_id: str, *, timeout_s: float = 2.5) -> dict[str, Any] | None:
    base = RUNS / run_id
    for name in (
        "backtest_summary.json",
        "backtest_summary_best.json",
        "backtest_summary_final.json",
        "backtest_summary_latest.json",
    ):
        data = _timed_read_json(base / name, timeout_s=timeout_s)
        if isinstance(data, dict) and (
            data.get("sharpe") is not None or data.get("total_return") is not None
        ):
            return data
    return None


def _pick_manifest(run_id: str, *, timeout_s: float = 2.0) -> dict[str, Any]:
    data = _timed_read_json(RUNS / run_id / "manifest.json", timeout_s=timeout_s)
    return data if isinstance(data, dict) else {}


def _merge_run_row(prev: dict[str, Any] | None, new: dict[str, Any]) -> dict[str, Any]:
    """Keep prior OOS if a timed disk read failed mid-enrich (iCloud false negative)."""
    if not prev:
        out = dict(new)
    else:
        out = dict(new)
        if out.get("oos_sharpe") is None and prev.get("oos_sharpe") is not None:
            for key in (
                "oos_sharpe",
                "oos_deflated_sharpe",
                "oos_return",
                "oos_max_drawdown",
                "ew_excess_return",
                "has_backtest",
                "labels",
            ):
                if prev.get(key) is not None:
                    out[key] = prev[key]
            if prev.get("has_backtest"):
                out["has_backtest"] = True
        # Prefer a more specific completed/interrupted status over a degraded active.
        if prev.get("training_status") in ("completed", "interrupted") and out.get(
            "training_status"
        ) not in ("completed", "interrupted"):
            out["training_status"] = prev["training_status"]
            if out.get("progress_pct") is None and prev.get("progress_pct") is not None:
                out["progress_pct"] = prev["progress_pct"]
    if out.get("has_backtest") or out.get("oos_sharpe") is not None:
        out["training_status"] = "completed"
        out["progress_pct"] = 100.0
        if out.get("nominal_timesteps") is not None:
            try:
                out["elapsed_timesteps"] = int(out["nominal_timesteps"])
            except (TypeError, ValueError):
                pass
    return out


def _progress_pct(elapsed: Any, nominal: Any) -> float | None:
    try:
        e = int(elapsed)
        n = int(nominal)
    except (TypeError, ValueError):
        return None
    if n <= 0 or e <= 0:
        return None
    return round(min(100.0, 100.0 * e / n), 1)


def _row_from_disk(run_id: str) -> dict[str, Any]:
    win_m = re.match(r"^(W\d+)_", run_id)
    bt = _pick_backtest(run_id)
    manifest = _pick_manifest(run_id)
    ckpt_steps, ckpt_mtime = _checkpoint_progress(run_id)
    has_final = _has_final_model(run_id)

    elapsed = manifest.get("elapsed_timesteps") or manifest.get("num_timesteps")
    nominal = manifest.get("nominal_timesteps") or manifest.get("timesteps")
    if nominal is None and isinstance(manifest.get("args"), dict):
        nominal = (manifest.get("args") or {}).get("timesteps")
    if elapsed is None and ckpt_steps is not None:
        elapsed = ckpt_steps
    if nominal is None:
        nominal = _DEFAULT_NOMINAL_STEPS

    explicit = str(manifest.get("training_status") or "").strip() or None
    status = _infer_run_status(
        explicit=explicit,
        has_backtest=bt is not None,
        has_final=has_final,
        checkpoint_steps=ckpt_steps,
        checkpoint_mtime=ckpt_mtime,
        finished_at=manifest.get("finished_at_utc"),
    )

    progress = _progress_pct(elapsed, nominal)
    if progress is None and status == "completed":
        progress = 100.0
        elapsed = int(nominal) if nominal is not None else elapsed
    if status == "interrupted" and progress is None:
        progress = _progress_pct(elapsed, nominal) if elapsed is not None else 0.0
        if elapsed is None:
            elapsed = 0

    row: dict[str, Any] = {
        "run_id": run_id,
        "window": win_m.group(1) if win_m else None,
        "training_status": status,
        "progress_pct": progress,
        "elapsed_timesteps": elapsed,
        "nominal_timesteps": nominal,
        "best_eval_step": manifest.get("best_eval_step"),
        "best_eval_score": manifest.get("best_eval_score"),
        "curriculum_stage_at_best": manifest.get("curriculum_stage_at_best"),
        "early_stop_reason": manifest.get("early_stop_reason"),
        "started_at_utc": manifest.get("started_at_utc"),
        "finished_at_utc": manifest.get("finished_at_utc"),
        "oos_sharpe": None,
        "oos_deflated_sharpe": None,
        "oos_return": None,
        "oos_max_drawdown": None,
        "ew_excess_return": None,
        "has_backtest": False,
        "labels": [],
        "warnings": [],
        "comparable": True,
        "git_dirty": manifest.get("git_dirty"),
    }
    if bt is not None:
        row["oos_sharpe"] = bt.get("sharpe")
        row["oos_deflated_sharpe"] = bt.get("deflated_sharpe")
        row["oos_return"] = bt.get("total_return")
        row["oos_max_drawdown"] = bt.get("max_drawdown")
        row["ew_excess_return"] = bt.get("excess_return_vs_equal_weight")
        row["has_backtest"] = True
        row["training_status"] = "completed"
        if not row["best_eval_step"]:
            row["best_eval_step"] = bt.get("best_eval_step")
        ckpt = bt.get("checkpoint_label")
        if ckpt:
            row["labels"] = [str(ckpt)]
    if row["training_status"] == "completed":
        if row["nominal_timesteps"] is not None:
            row["elapsed_timesteps"] = int(row["nominal_timesteps"])
        row["progress_pct"] = 100.0
    return row


def _float_or_none(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _oos_row_from_backtest(run_id: str, bt: dict[str, Any]) -> dict[str, Any] | None:
    match = _COHORT_RUN_RE.match(run_id)
    if match is None:
        return None
    sharpe = bt.get("sharpe")
    ret = bt.get("total_return")
    if sharpe is None and ret is None:
        return None
    detailed = bt.get("detailed") if isinstance(bt.get("detailed"), dict) else {}
    ew = detailed.get("benchmark_equal_weight_daily") or detailed.get("benchmark_equal_weight")
    spy = detailed.get("benchmark_spy") or detailed.get("benchmark_only")
    ew_ret = _float_or_none(ew.get("total_return")) if isinstance(ew, dict) else None
    ew_sh = _float_or_none(ew.get("sharpe")) if isinstance(ew, dict) else None
    spy_ret = _float_or_none(spy.get("total_return")) if isinstance(spy, dict) else None
    spy_sh = _float_or_none(spy.get("sharpe")) if isinstance(spy, dict) else None
    if ew_ret is None:
        ew_ret = _float_or_none(bt.get("equal_weight_daily_return"))
    if ew_sh is None:
        ew_sh = _float_or_none(bt.get("equal_weight_daily_sharpe"))
    if spy_ret is None:
        spy_ret = _float_or_none(bt.get("spy_return") or bt.get("benchmark_return"))
    if spy_sh is None:
        spy_sh = _float_or_none(bt.get("spy_sharpe"))
    return {
        "run_id": run_id,
        "cohort": match.group(2),
        "window": match.group(1).upper(),
        "model_ret": ret,
        "model_sh": sharpe,
        "ew_ret": ew_ret,
        "ew_sh": ew_sh,
        "spy_ret": spy_ret,
        "spy_sh": spy_sh,
        "has_benchmarks": ew_ret is not None and spy_ret is not None,
    }


def _build_enriched_rows(budget_s: float = 25.0) -> list[dict[str, Any]]:
    t0 = time.time()
    ids = _list_run_ids()
    # Seed with previous cache so we don't blank out known OOS while refreshing.
    prev = _read_json(RUNS_CACHE)
    prev_by_id: dict[str, dict[str, Any]] = {}
    if isinstance(prev, dict):
        for rec in prev.get("records") or []:
            if isinstance(rec, dict) and rec.get("run_id"):
                prev_by_id[str(rec["run_id"])] = rec

    def rank(rid: str) -> tuple[Any, ...]:
        """Newest cohorts + missing-OOS completed runs first (budget truncates)."""
        prev_row = prev_by_id.get(rid) or {}
        missing_oos = 0 if prev_row.get("oos_sharpe") is None else 1
        activeish = 0 if prev_row.get("training_status") in (None, "", "active", "running") else 1
        m = _WINDOW_COHORT_RE.match(rid)
        if m is not None:
            window = int(m.group(1))
            cohort_key = -int(m.group(2))  # 809 before 808
            return (missing_oos, activeish, cohort_key, window, rid)
        if rid.startswith("LIVE_") or rid == "RLModel":
            return (missing_oos, 0, 1, 0, rid)
        return (missing_oos, 2, 0, 0, rid)

    ids = sorted(ids, key=rank)

    rows_by_id: dict[str, dict[str, Any]] = dict(prev_by_id)
    oos_rows: list[dict[str, Any]] = []
    for rid in ids:
        if time.time() - t0 > budget_s:
            break
        row = _merge_run_row(prev_by_id.get(rid), _row_from_disk(rid))
        rows_by_id[rid] = row
        if row["has_backtest"] and row.get("oos_sharpe") is not None:
            # Prefer full backtest JSON (detailed EW/SPY sleeves) when cheap;
            # fall back to run-cache fields so we never invent null benchmarks.
            bt = _pick_backtest(rid, timeout_s=1.5)
            if bt is None:
                bt = {
                    "sharpe": row["oos_sharpe"],
                    "total_return": row["oos_return"],
                    "deflated_sharpe": row.get("oos_deflated_sharpe"),
                    "max_drawdown": row.get("oos_max_drawdown"),
                    "equal_weight_daily_return": row.get("equal_weight_daily_return"),
                    "equal_weight_daily_sharpe": row.get("equal_weight_daily_sharpe"),
                    "spy_return": row.get("spy_return"),
                    "spy_sharpe": row.get("spy_sharpe"),
                    "excess_return_vs_equal_weight": row.get("ew_excess_return"),
                }
            oos = _oos_row_from_backtest(rid, bt)
            if oos is not None:
                oos_rows.append(oos)

    # Second pass: rows still missing OOS — longer timeout, patch in place (no
    # second full _row_from_disk which would double iCloud I/O).
    still_missing = [
        rid
        for rid, row in rows_by_id.items()
        if row.get("oos_sharpe") is None
    ]
    still_missing.sort(key=rank)
    for rid in still_missing:
        if time.time() - t0 > budget_s + 45.0:
            break
        bt = _pick_backtest(rid, timeout_s=3.5)
        if bt is None:
            continue
        filled = dict(rows_by_id.get(rid) or {"run_id": rid})
        filled["oos_sharpe"] = bt.get("sharpe")
        filled["oos_deflated_sharpe"] = bt.get("deflated_sharpe")
        filled["oos_return"] = bt.get("total_return")
        filled["oos_max_drawdown"] = bt.get("max_drawdown")
        filled["ew_excess_return"] = bt.get("excess_return_vs_equal_weight")
        filled["has_backtest"] = True
        filled["training_status"] = "completed"
        filled["progress_pct"] = 100.0
        if filled.get("nominal_timesteps") is not None:
            filled["elapsed_timesteps"] = int(filled["nominal_timesteps"])
        ckpt = bt.get("checkpoint_label")
        if ckpt and not filled.get("labels"):
            filled["labels"] = [str(ckpt)]
        rows_by_id[rid] = filled
        oos = _oos_row_from_backtest(rid, bt)
        if oos is not None:
            oos_rows.append(oos)

    rows = [_normalize_run_row(r) for r in _sort_run_rows(list(rows_by_id.values()))]
    try:
        _atomic_write_json(
            RUNS_CACHE,
            {
                "generated_at_utc": _now(),
                "n": len(rows),
                "records": rows,
                "note": "lite enriched from Runs/*/backtest_summary*.json",
            },
        )
        if oos_rows:
            # Merge into existing OOS cache rather than truncating mid-budget.
            existing = _read_json(OOS_CACHE)
            by_id: dict[str, dict[str, Any]] = {}
            if isinstance(existing, dict):
                for rec in existing.get("rows") or []:
                    if isinstance(rec, dict) and rec.get("run_id"):
                        by_id[str(rec["run_id"])] = rec
            for rec in oos_rows:
                by_id[str(rec["run_id"])] = rec
            merged = list(by_id.values())
            _atomic_write_json(
                OOS_CACHE,
                {"generated_at_utc": _now(), "n": len(merged), "rows": merged},
            )
    except OSError:
        pass
    return rows


def _fill_rows_oos(rows: list[dict[str, Any]], *, budget_s: float = 8.0) -> list[dict[str, Any]]:
    """Synchronously refresh the visible page (OOS + progress/status) with a budget."""
    global _rows_cache, _rows_at
    t0 = time.time()
    out: list[dict[str, Any]] = []
    changed = False
    for row in rows:
        if time.time() - t0 > budget_s:
            out.append(row)
            continue
        rid = str(row.get("run_id") or "")
        if not rid:
            out.append(row)
            continue
        needs_refresh = (
            row.get("oos_sharpe") is None
            or row.get("progress_pct") is None
            or row.get("elapsed_timesteps") is None
            or row.get("training_status") in (None, "", "active")
        )
        if not needs_refresh:
            out.append(row)
            continue
        filled = _row_from_disk(rid)
        changed = True
        out.append(filled)
    if changed:
        with _rows_lock:
            by_id = {str(r.get("run_id")): r for r in (_rows_cache or [])}
            for r in out:
                by_id[str(r.get("run_id"))] = r
            _rows_cache = _sort_run_rows(list(by_id.values()))
            _rows_at = time.monotonic()
    return out


def _publish_frontend_snapshots(*, enrich: bool = False) -> None:
    """Rewrite frontend/public/data/*.json from execution caches (best-effort)."""
    script = ROOT / "scripts" / "publish_frontend_data.py"
    args = [sys.executable, str(script), "--with-details"]
    if enrich:
        args.append("--enrich-details")
    try:
        subprocess.run(
            args,
            cwd=str(ROOT),
            timeout=90 if enrich else 15,
            check=False,
            capture_output=True,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        print(f"[lite-api] snapshot publish skipped: {exc}", file=sys.stderr, flush=True)


def _kick_enrich() -> None:
    global _enriching
    with _enrich_lock:
        if _enriching:
            return
        _enriching = True

    def _worker() -> None:
        global _rows_cache, _rows_at, _enriching
        try:
            rows = _build_enriched_rows(budget_s=45.0)
            with _rows_lock:
                _rows_cache = rows
                _rows_at = time.monotonic()
            _publish_frontend_snapshots(enrich=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[lite-api] enrich failed: {exc}", file=sys.stderr, flush=True)
        finally:
            with _enrich_lock:
                _enriching = False

    threading.Thread(target=_worker, name="lite-runs-enrich", daemon=True).start()


def _run_rows() -> list[dict[str, Any]]:
    global _rows_cache, _rows_at
    now = time.monotonic()
    with _rows_lock:
        cached = _rows_cache
        age = now - _rows_at
    if cached is not None and age <= _CACHE_TTL_S:
        return _sort_run_rows(list(cached))

    # Prefer disk cache immediately (may be stale lightweight index).
    payload = _read_json(RUNS_CACHE)
    records = payload.get("records") if isinstance(payload, dict) else None
    disk_rows: list[dict[str, Any]] = []
    if isinstance(records, list):
        for rec in records:
            if isinstance(rec, dict) and rec.get("run_id"):
                disk_rows.append(_normalize_run_row(rec))
    disk_rows = _sort_run_rows(disk_rows)

    # Cold or empty of OOS: kick background enrich; serve disk / empty now.
    has_oos = any(r.get("oos_sharpe") is not None for r in disk_rows)
    if cached is None or age > _CACHE_TTL_S or not has_oos:
        _kick_enrich()

    if disk_rows:
        with _rows_lock:
            if _rows_cache is None:
                _rows_cache = disk_rows
                _rows_at = now
        return disk_rows
    if cached is not None:
        return _sort_run_rows(list(cached))
    return []


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [r for r in rows if r.get("training_status") == "completed"]
    scored = [r for r in rows if isinstance(r.get("oos_sharpe"), (int, float))]
    best = max(scored, key=lambda r: r["oos_sharpe"]) if scored else None
    return {
        "generated_at_utc": _now(),
        "total_runs": len(rows),
        "completed_runs": len(completed),
        "active_runs": len(
            [r for r in rows if r.get("training_status") not in ("completed", "interrupted")]
        ),
        "runs_with_backtest": len(scored),
        "best_oos": (
            {
                "run_id": best["run_id"],
                "sharpe": best["oos_sharpe"],
                "deflated_sharpe": best.get("oos_deflated_sharpe"),
                "window": best.get("window"),
            }
            if best
            else None
        ),
    }


def _venv_python() -> str:
    cand = ROOT / ".venv" / "bin" / "python"
    if cand.is_file():
        return str(cand)
    return sys.executable


_RL_UNIVERSE: dict[str, str] = {
    "SP500": "SPY",
    "GOLD": "GLD",
    "OIL": "USO",
    "EURUSD": "EURUSD=X",
    "USDJPY": "USDJPY=X",
    "NIKKEI": "^N225",
    "FTSE": "^FTSE",
    "BOND10Y": "IEF",
    "COPPER": "HG=F",
    "EM": "EEM",
}
_RL_CASH_YIELD_PER_BAR = 0.00015 / 78.0
_YAHOO_UA = "Mozilla/5.0 (compatible; MarketTrainerForward/1.0)"


def _yahoo_chart_ohlc(symbol: str, *, range_key: str = "1mo", timeout_s: float = 12.0) -> list[tuple[datetime, float, float, float, float]]:
    """Fetch 5m OHLC via Yahoo chart API (stdlib only — no yfinance / pandas)."""
    params = urlencode({"range": range_key, "interval": "5m", "events": "history"})
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}?{params}"
    try:
        req = Request(url, headers={"User-Agent": _YAHOO_UA})
        with urlopen(req, timeout=float(timeout_s)) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return []
    result = (payload.get("chart", {}).get("result") or [None])[0]
    if not isinstance(result, dict):
        return []
    stamps = result.get("timestamp") or []
    quote_blob = ((result.get("indicators") or {}).get("quote") or [{}])[0] or {}
    opens = quote_blob.get("open") or []
    highs = quote_blob.get("high") or []
    lows = quote_blob.get("low") or []
    closes = quote_blob.get("close") or []
    try:
        from zoneinfo import ZoneInfo

        et = ZoneInfo("America/New_York")
    except Exception:  # noqa: BLE001
        et = timezone(timedelta(hours=-4))
    rows: list[tuple[datetime, float, float, float, float]] = []
    for i, raw in enumerate(stamps):
        try:
            ts = datetime.fromtimestamp(int(raw), tz=timezone.utc).astimezone(et).replace(
                tzinfo=None, second=0, microsecond=0
            )
            c = float(closes[i])
        except (TypeError, ValueError, IndexError):
            continue
        if c != c or c <= 0:
            continue

        def _px(seq: list[Any], fallback: float) -> float:
            try:
                v = float(seq[i])
            except (TypeError, ValueError, IndexError):
                return fallback
            return v if v == v and v > 0 else fallback

        o = _px(opens, c)
        h = _px(highs, max(o, c))
        l = _px(lows, min(o, c))
        rows.append((ts, o, h, l, c))
    rows.sort(key=lambda r: r[0])
    return rows


def _fetch_yahoo_frames(symbols: dict[str, str]) -> dict[str, list[tuple[datetime, float, float, float, float]]]:
    uniq = list(dict.fromkeys(symbols.values()))
    out: dict[str, list[tuple[datetime, float, float, float, float]]] = {}
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(uniq)))) as pool:
        futs = {pool.submit(_yahoo_chart_ohlc, sym, range_key="1mo"): sym for sym in uniq}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                rows = fut.result()
            except Exception:  # noqa: BLE001
                rows = []
            if not rows:
                try:
                    rows = _yahoo_chart_ohlc(sym, range_key="5d")
                except Exception:  # noqa: BLE001
                    rows = []
            out[sym] = rows
    return out


def _align_on_clock(
    clock: list[datetime],
    rows: list[tuple[datetime, float, float, float, float]],
) -> tuple[list[float], list[float], list[float], list[float]]:
    by_ts = {r[0]: r for r in rows}
    o: list[float] = []
    h: list[float] = []
    l: list[float] = []
    c: list[float] = []
    last_c: float | None = None
    for ts in clock:
        row = by_ts.get(ts)
        if row is None:
            if last_c is None:
                o.append(float("nan"))
                h.append(float("nan"))
                l.append(float("nan"))
                c.append(float("nan"))
            else:
                o.append(last_c)
                h.append(last_c)
                l.append(last_c)
                c.append(last_c)
            continue
        last_c = row[4]
        o.append(row[1])
        h.append(row[2])
        l.append(row[3])
        c.append(row[4])
    if any(x != x for x in c):
        nxt = next((x for x in c if x == x), 1.0)
        filled_o: list[float] = []
        filled_h: list[float] = []
        filled_l: list[float] = []
        filled_c: list[float] = []
        for i, val in enumerate(c):
            if val == val:
                nxt = val
                filled_o.append(o[i])
                filled_h.append(h[i])
                filled_l.append(l[i])
                filled_c.append(val)
            else:
                filled_o.append(nxt)
                filled_h.append(nxt)
                filled_l.append(nxt)
                filled_c.append(nxt)
        return filled_o, filled_h, filled_l, filled_c
    return o, h, l, c


def _weight_on_labels(weights: dict[str, float], labels: list[str]) -> list[float]:
    by = {str(k).strip().upper().replace(" ", ""): float(v) for k, v in weights.items()}
    out = [by.get(lab.strip().upper().replace(" ", ""), 0.0) for lab in labels]
    s = sum(out)
    if s <= 1e-12:
        out = [1.0] + [0.0] * (len(labels) - 1)
        return out
    return [x / s for x in out]


def _portfolio_ohlc(
    o: list[list[float]],
    h: list[list[float]],
    l: list[list[float]],
    c: list[list[float]],
    weight: list[float] | list[list[float]],
    *,
    initial_cash: float,
    cash_yield_per_bar: float = 0.0,
) -> list[tuple[float, float, float, float]]:
    t_bars = len(c)
    n = len(c[0]) if t_bars else 0

    def _norm_row(row: list[float]) -> list[float]:
        ww = list(row)
        if len(ww) != n + 1:
            pad = [0.0] * (n + 1)
            for i in range(min(len(ww), n + 1)):
                pad[i] = ww[i]
            ww = pad
        s = sum(ww)
        return [x / s for x in ww] if s > 1e-12 else [1.0] + [0.0] * n

    if weight and t_bars and isinstance(weight[0], (list, tuple)):
        W = [_norm_row(list(row)) for row in weight]  # type: ignore[arg-type]
        if len(W) < t_bars:
            W = W + [list(W[-1] if W else _norm_row([]))] * (t_bars - len(W))
        else:
            W = W[:t_bars]
    else:
        row = _norm_row(list(weight) if weight else [])  # type: ignore[arg-type]
        W = [row] * t_bars

    out: list[tuple[float, float, float, float]] = []
    nav_close = float(initial_cash)
    prev = list(o[0]) if t_bars else []
    y = float(cash_yield_per_bar)
    for t in range(t_bars):
        w = W[t]
        cash_w = w[0]
        risky = w[1:]
        if t == 0:
            base = [max(x, 1e-12) for x in o[t]]
            r_h = [h[t][i] / base[i] - 1.0 for i in range(n)]
            r_l = [l[t][i] / base[i] - 1.0 for i in range(n)]
            r_c = [c[t][i] / base[i] - 1.0 for i in range(n)]
            o_nav = float(initial_cash)
            h_nav = o_nav * (1.0 + sum(risky[i] * r_h[i] for i in range(n)) + cash_w * y)
            l_nav = o_nav * (1.0 + sum(risky[i] * r_l[i] for i in range(n)) + cash_w * y)
            c_nav = o_nav * (1.0 + sum(risky[i] * r_c[i] for i in range(n)) + cash_w * y)
        else:
            base = [max(x, 1e-12) for x in prev]
            r_o = [o[t][i] / base[i] - 1.0 for i in range(n)]
            r_h = [h[t][i] / base[i] - 1.0 for i in range(n)]
            r_l = [l[t][i] / base[i] - 1.0 for i in range(n)]
            r_c = [c[t][i] / base[i] - 1.0 for i in range(n)]
            o_nav = nav_close * (1.0 + sum(risky[i] * r_o[i] for i in range(n)))
            h_nav = nav_close * (1.0 + sum(risky[i] * r_h[i] for i in range(n)) + cash_w * y)
            l_nav = nav_close * (1.0 + sum(risky[i] * r_l[i] for i in range(n)) + cash_w * y)
            c_nav = nav_close * (1.0 + sum(risky[i] * r_c[i] for i in range(n)) + cash_w * y)
        hi = max(o_nav, h_nav, l_nav, c_nav)
        lo = min(o_nav, h_nav, l_nav, c_nav)
        out.append((o_nav, hi, lo, c_nav))
        nav_close = c_nav
        prev = list(c[t])
    return out


def _nav_from_ohlc(ohlc: list[tuple[float, float, float, float]], initial_cash: float) -> list[float]:
    if not ohlc:
        return []
    nav = [row[3] for row in ohlc]
    nav[0] = ohlc[0][0] if ohlc[0][0] == ohlc[0][0] else float(initial_cash)
    return nav


def _max_dd(navs: list[float]) -> float:
    if len(navs) < 2:
        return 0.0
    peak = navs[0]
    worst = 0.0
    for x in navs:
        if x > peak:
            peak = x
        dd = x / max(peak, 1e-12) - 1.0
        if dd < worst:
            worst = dd
    return worst


def _series_tip_stats(navs: list[float]) -> dict[str, float]:
    if not navs:
        return {"total_return": 0.0, "sharpe": None, "max_drawdown": 0.0, "nav": None}  # type: ignore[dict-item]
    ret = navs[-1] / max(navs[0], 1e-12) - 1.0
    return {
        "total_return": ret,
        "sharpe": None,  # type: ignore[dict-item]
        "max_drawdown": _max_dd(navs),
        "nav": navs[-1],
    }


def _candles_rows(times: list[datetime], ohlc: list[tuple[float, float, float, float]]) -> list[dict[str, Any]]:
    return [
        {
            "t": ts.isoformat(timespec="minutes"),
            "o": float(ohlc[i][0]),
            "h": float(ohlc[i][1]),
            "l": float(ohlc[i][2]),
            "c": float(ohlc[i][3]),
        }
        for i, ts in enumerate(times)
    ]


def _jsonl_row_is_reset(rec: dict[str, Any]) -> bool:
    note = str(rec.get("note") or "").lower()
    return "reset to 100k" in note or "flat paper book" in note


def _jsonl_last_weights(run_id: str) -> dict[str, float] | None:
    path = EXEC / f"shadow_ledger_{run_id}.jsonl"
    if not path.is_file():
        return None
    last: dict[str, float] | None = None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec, dict) or _jsonl_row_is_reset(rec):
                continue
            tw = rec.get("target_weights")
            if isinstance(tw, dict) and tw:
                last = {str(k): float(v) for k, v in tw.items()}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return last
    return last


def _as_et_naive(dt: datetime) -> datetime:
    try:
        from zoneinfo import ZoneInfo

        if dt.tzinfo is not None:
            return dt.astimezone(ZoneInfo("America/New_York")).replace(tzinfo=None)
    except Exception:  # noqa: BLE001
        if dt.tzinfo is not None:
            return dt.replace(tzinfo=None)
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


def _ledger_event_ts(rec: dict[str, Any]) -> datetime | None:
    for key in ("recorded_at_utc", "recorded_at", "trade_date", "decision_bar", "as_of"):
        raw = rec.get(key)
        if not raw:
            continue
        text = str(raw).strip().replace("Z", "+00:00")
        try:
            if "T" in text or " " in text[10:]:
                dt = datetime.fromisoformat(text)
                return _as_et_naive(dt)
            return datetime.fromisoformat(text[:10] + "T09:30:00")
        except ValueError:
            continue
    return None


def _jsonl_weight_events(run_id: str) -> list[tuple[datetime, dict[str, float]]]:
    path = EXEC / f"shadow_ledger_{run_id}.jsonl"
    events: list[tuple[datetime, dict[str, float]]] = []
    if not path.is_file():
        return events
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return events
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict) or _jsonl_row_is_reset(rec):
            continue
        tw = rec.get("target_weights")
        ts = _ledger_event_ts(rec)
        if not isinstance(tw, dict) or not tw or ts is None:
            continue
        try:
            weights = {str(k): float(v) for k, v in tw.items()}
        except (TypeError, ValueError):
            continue
        events.append((ts, weights))
    events.sort(key=lambda item: item[0])
    return events


def _weight_rows_on_clock(
    clock: list[datetime],
    events: list[tuple[datetime, dict[str, float]]],
    labels: list[str],
    *,
    start: datetime | None = None,
) -> list[list[float]]:
    cash = _weight_on_labels({"CASH": 1.0}, labels)
    ev = sorted(events, key=lambda item: item[0])
    if start is not None:
        ev = [(t, w) for t, w in ev if t >= start]
    rows: list[list[float]] = []
    j = -1
    cur = cash
    for t in clock:
        if start is not None and t < start:
            rows.append(list(cash))
            continue
        while j + 1 < len(ev) and ev[j + 1][0] <= t:
            j += 1
            cur = _weight_on_labels(ev[j][1], labels)
        rows.append(list(cur))
    return rows


def _live_model_start_ts(existing: dict[str, Any], stamp: dict[str, Any]) -> datetime:
    for src in (existing, stamp):
        if not isinstance(src, dict):
            continue
        live = src.get("live") if isinstance(src.get("live"), dict) else {}
        for raw in (src.get("live_model_start"), live.get("live_model_start")):
            if not raw:
                continue
            text = str(raw).strip().replace("Z", "+00:00")
            try:
                if "T" in text:
                    dt = datetime.fromisoformat(text)
                    return _as_et_naive(dt) if dt.tzinfo else dt.replace(second=0, microsecond=0)
                return datetime.fromisoformat(text[:10] + "T09:30:00")
            except ValueError:
                continue
    man = _read_json(RUNS / "RLModel" / "manifest.json")
    fin = man.get("finished_at_utc") if isinstance(man, dict) else None
    if fin:
        try:
            dt = _as_et_naive(datetime.fromisoformat(str(fin).replace("Z", "+00:00")))
            after_close = dt.weekday() >= 5 or dt.hour >= 16
            if after_close:
                nxt = dt + timedelta(days=1)
                while nxt.weekday() >= 5:
                    nxt = nxt + timedelta(days=1)
                return nxt.replace(hour=9, minute=30, second=0, microsecond=0)
            if dt.hour < 9 or (dt.hour == 9 and dt.minute < 30):
                return dt.replace(hour=9, minute=30, second=0, microsecond=0)
            add = 5 - (dt.minute % 5) or 5
            return dt.replace(second=0, microsecond=0) + timedelta(minutes=add)
        except ValueError:
            pass
    return datetime(2026, 8, 17, 9, 30)


def _paper_state_event(run_id: str) -> tuple[datetime, dict[str, float]] | None:
    w = _paper_state_weights(run_id)
    if not w:
        return None
    names = {
        "GENERAL_EQUITY1": EXEC / "paper_general_equity1" / "state.json",
        "CREST_DAY": EXEC / "paper_crest_day" / "state.json",
    }
    st = _read_json(names.get(str(run_id).upper(), Path()))
    ts = _ledger_event_ts(
        {
            "recorded_at_utc": (st or {}).get("updated_at_utc"),
            "trade_date": (st or {}).get("last_trade_date"),
        }
    )
    return (ts or datetime.now(), w)


def _paper_state_weights(run_id: str) -> dict[str, float] | None:
    names = {
        "GENERAL_EQUITY1": EXEC / "paper_general_equity1" / "state.json",
        "CREST_DAY": EXEC / "paper_crest_day" / "state.json",
    }
    path = names.get(str(run_id).upper())
    if path is None or not path.is_file():
        return None
    st = _read_json(path)
    if not isinstance(st, dict):
        return None
    tw = st.get("target_weights")
    if not isinstance(tw, dict) or not tw:
        return None
    return {str(k): float(v) for k, v in tw.items()}


def _paper_share_book(run_id: str) -> tuple[float, dict[str, float], datetime | None]:
    names = {
        "GENERAL_EQUITY1": EXEC / "paper_general_equity1" / "state.json",
        "CREST_DAY": EXEC / "paper_crest_day" / "state.json",
    }
    path = names.get(str(run_id).upper())
    st = _read_json(path) if path is not None else {}
    if not isinstance(st, dict):
        return 0.0, {}, None
    try:
        cash = float(st.get("cash") or 0.0)
    except (TypeError, ValueError):
        cash = 0.0
    pos: dict[str, float] = {}
    raw = st.get("positions") if isinstance(st.get("positions"), dict) else {}
    for key, val in raw.items():
        try:
            qty = float(val)
        except (TypeError, ValueError):
            continue
        if abs(qty) > 1e-12:
            pos[str(key).strip().upper()] = qty
    trade = str(st.get("last_trade_date") or "").strip()
    updated = str(st.get("updated_at_utc") or "").strip()
    rec: dict[str, Any] = {"trade_date": trade or None}
    if updated and trade and updated[:10] == trade[:10]:
        rec["recorded_at_utc"] = updated
    return cash, pos, _ledger_event_ts(rec)


def _lots_ohlc(
    o: list[list[float]],
    h: list[list[float]],
    l: list[list[float]],
    c: list[list[float]],
    clock: list[datetime],
    *,
    cash: float,
    quantities: list[float],
    start: datetime | None,
    initial_cash: float,
) -> list[tuple[float, float, float, float]]:
    t_bars = len(c)
    n = len(c[0]) if t_bars else 0
    qty = list(quantities) + [0.0] * max(0, n - len(quantities))
    qty = qty[:n]
    out: list[tuple[float, float, float, float]] = []
    cash_f = float(cash)
    init = float(initial_cash)
    for t in range(t_bars):
        if start is not None and clock[t] < start:
            out.append((init, init, init, init))
            continue
        o_nav = cash_f + sum(qty[i] * o[t][i] for i in range(n))
        h_nav = cash_f + sum(qty[i] * h[t][i] for i in range(n))
        l_nav = cash_f + sum(qty[i] * l[t][i] for i in range(n))
        c_nav = cash_f + sum(qty[i] * c[t][i] for i in range(n))
        out.append((o_nav, max(o_nav, h_nav, l_nav, c_nav), min(o_nav, h_nav, l_nav, c_nav), c_nav))
    return out


def _strategy_weights(run_id: str, fallback: dict[str, float] | None = None) -> dict[str, float]:
    for getter in (
        lambda: _jsonl_last_weights(run_id),
        lambda: _paper_state_weights(run_id),
        lambda: fallback,
    ):
        try:
            w = getter()
        except Exception:  # noqa: BLE001
            w = None
        if isinstance(w, dict) and w:
            return {str(k): float(v) for k, v in w.items()}
    return {"Cash": 1.0}


def _norm_weight_key(key: str) -> str:
    nk = str(key).strip().upper().replace(" ", "")
    return "CASH" if nk in {"CASH", "USD"} else nk


def _positions_from_weights_lite(
    weights: dict[str, float],
    *,
    nav: float,
    price_by_ticker: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    price_by_ticker = price_by_ticker or {}
    by: dict[str, float] = {}
    for k, v in weights.items():
        by[_norm_weight_key(k)] = float(v)
    ordered: list[str] = []
    seen: set[str] = set()
    for k in weights:
        nk = _norm_weight_key(k)
        if nk == "CASH" or nk in seen or abs(by.get(nk, 0.0)) <= 1e-12:
            continue
        ordered.append(nk)
        seen.add(nk)
    rows: list[dict[str, Any]] = []
    labels = ["Cash"] + ordered
    tickers = ["CASH"] + ordered
    for lab, ticker in zip(labels, tickers):
        w = float(by.get("CASH", 0.0) if ticker == "CASH" else by.get(ticker, 0.0))
        raw_px = 1.0 if ticker == "CASH" else price_by_ticker.get(ticker)
        try:
            fpx = float(raw_px) if raw_px is not None else None
        except (TypeError, ValueError):
            fpx = None
        if fpx is not None and (fpx != fpx or fpx <= 0):
            fpx = None
        rows.append(
            {
                "label": lab,
                "ticker": ticker,
                "weight": w,
                "value_usd": w * float(nav),
                "price": 1.0 if ticker == "CASH" else fpx,
            }
        )
    return rows


def _tip_nav(payload: dict[str, Any], key: str, fallback: float) -> float:
    stats = payload.get("stats") if isinstance(payload.get("stats"), dict) else {}
    st = stats.get(key) if isinstance(stats.get(key), dict) else None
    if st and st.get("nav") is not None:
        try:
            return float(st["nav"])
        except (TypeError, ValueError):
            pass
    nav = payload.get("nav") if isinstance(payload.get("nav"), dict) else {}
    series = nav.get(key)
    if isinstance(series, list) and series:
        try:
            return float(series[-1])
        except (TypeError, ValueError):
            pass
    return float(fallback)


def _attach_live_allocations(payload: dict[str, Any]) -> dict[str, Any]:
    """Rebuild allocation books from live ledgers so the panel tracks the chart."""
    if not isinstance(payload, dict):
        return payload
    initial_cash = float(payload.get("initial_cash") or 100_000.0)
    as_of = (
        (payload.get("live") or {}).get("as_of_utc")
        or (payload.get("live") or {}).get("as_of_bar")
        or payload.get("generated_at_utc")
    )
    ge_w_payload = payload.get("latest_weights") if isinstance(payload.get("latest_weights"), dict) else None
    ge_w = (
        {str(k): float(v) for k, v in ge_w_payload.items()}
        if ge_w_payload
        else _strategy_weights("GENERAL_EQUITY1", None)
    )
    ge_nav = _tip_nav(payload, "model", initial_cash)
    ge_price = {}
    for row in payload.get("positions") or []:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or row.get("label") or "")
        if ticker and row.get("price") is not None:
            try:
                ge_price[_norm_weight_key(ticker)] = float(row["price"])
            except (TypeError, ValueError):
                pass
    ge_positions = payload.get("positions") if isinstance(payload.get("positions"), list) and payload.get("positions") else None
    rl_w = _strategy_weights("RLModel", None)
    rl_nav = _tip_nav(payload, "live_model", initial_cash)
    crypto_w = _strategy_weights("CREST_DAY", None)
    crypto_nav = _tip_nav(payload, "crypto", initial_cash)
    prev = payload.get("allocations") if isinstance(payload.get("allocations"), dict) else {}
    crypto_prev = prev.get("crypto") if isinstance(prev.get("crypto"), dict) else {}
    payload["allocations"] = {
        "model": {
            "key": "model",
            "label": "GeneralEquity1",
            "run_id": "GENERAL_EQUITY1",
            "nav": ge_nav,
            "as_of": as_of,
            "price_source": "yahoo",
            "positions": ge_positions or _positions_from_weights_lite(ge_w, nav=ge_nav, price_by_ticker=ge_price),
            "latest_weights": {str(k): float(v) for k, v in ge_w.items()},
        },
        "live_model": {
            "key": "live_model",
            "label": "RLModel",
            "run_id": "RLModel",
            "nav": rl_nav,
            "as_of": as_of,
            "price_source": "weights",
            "positions": _positions_from_weights_lite(rl_w, nav=rl_nav),
            "latest_weights": {str(k): float(v) for k, v in rl_w.items()},
        },
        "crypto": {
            "key": "crypto",
            "label": "CrestDay",
            "run_id": "CREST_DAY",
            "nav": crypto_nav,
            "as_of": as_of,
            "price_source": str(crypto_prev.get("price_source") or "weights"),
            "positions": _positions_from_weights_lite(crypto_w, nav=crypto_nav),
            "latest_weights": {str(k): float(v) for k, v in crypto_w.items()},
        },
    }
    return payload


def _cols(labels: list[str], frames: dict[str, tuple[list[float], list[float], list[float], list[float]]]) -> tuple[list[list[float]], list[list[float]], list[list[float]], list[list[float]]]:
    o: list[list[float]] = []
    h: list[list[float]] = []
    l: list[list[float]] = []
    c: list[list[float]] = []
    n = len(next(iter(frames.values()))[3]) if frames else 0
    for i in range(n):
        o.append([frames[lab][0][i] for lab in labels])
        h.append([frames[lab][1][i] for lab in labels])
        l.append([frames[lab][2][i] for lab in labels])
        c.append([frames[lab][3][i] for lab in labels])
    return o, h, l, c


def _refresh_forward_prices_stdlib(run_id: str) -> dict[str, Any] | None:
    """Rebuild 5m NAV from Yahoo chart API without importing rlbot (iCloud-safe)."""
    rid = str(run_id).strip()
    existing = _load_mark(rid) or {}
    stamp = _read_json(EXEC / "forward_live_stamp.json")
    stamp = stamp if isinstance(stamp, dict) else {}
    initial_cash = float(existing.get("initial_cash") or 100_000.0)
    book_start = str(existing.get("book_start") or existing.get("holdout_start") or stamp.get("book_start") or "")
    if not book_start:
        book_start = datetime.now().date().isoformat()
    try:
        start_ts = datetime.fromisoformat(book_start[:10])
    except ValueError:
        start_ts = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    ge_events = _jsonl_weight_events(rid)
    paper_ev = _paper_state_event(rid)
    if paper_ev is not None:
        ge_events.append(paper_ev)
        ge_events.sort(key=lambda item: item[0])
    ge_w = ge_events[-1][1] if ge_events else _strategy_weights(
        rid, existing.get("latest_weights") if isinstance(existing.get("latest_weights"), dict) else None
    )
    paper_cash, paper_lots, lots_start = _paper_share_book(rid)
    rl_events = _jsonl_weight_events("RLModel")
    live_start = _live_model_start_ts(existing, stamp)

    ge_labs: list[str] = []
    seen_ge: set[str] = set()
    for lab in list(paper_lots.keys()) + [
        k for _, book in (ge_events + [(datetime.min, ge_w)]) for k in book
    ]:
        name = str(lab).strip().upper()
        if name in {"CASH", "USD"} or name in seen_ge:
            continue
        seen_ge.add(name)
        ge_labs.append(name)
    symbols: dict[str, str] = dict(_RL_UNIVERSE)
    for lab in ge_labs:
        symbols[lab.upper()] = str(lab).upper().replace(".", "-")
    symbols["SPY"] = "SPY"

    raw = _fetch_yahoo_frames(symbols)
    spy_rows = raw.get("SPY") or []
    spy_rows = [r for r in spy_rows if r[0] >= start_ts]
    if not spy_rows:
        raise RuntimeError("Yahoo returned no 5m SPY bars")
    clock = [r[0] for r in spy_rows]

    aligned: dict[str, tuple[list[float], list[float], list[float], list[float]]] = {}
    for lab, sym in symbols.items():
        aligned[lab] = _align_on_clock(clock, raw.get(sym) or [])

    ge_asset_labs = [lab.upper() for lab in ge_labs]
    ge_labels = ["Cash"] + ge_asset_labs
    ge_rows = _weight_rows_on_clock(clock, ge_events, ge_labels)
    ge_vec = ge_rows[-1] if ge_rows else _weight_on_labels(ge_w, ge_labels)
    o_ge, h_ge, l_ge, c_ge = _cols(ge_asset_labs, aligned) if ge_asset_labs else ([], [], [], [])
    if not ge_asset_labs:
        model_ohlc = [(initial_cash, initial_cash, initial_cash, initial_cash) for _ in clock]
    elif paper_lots:
        qty = [float(paper_lots.get(lab, 0.0)) for lab in ge_asset_labs]
        model_ohlc = _lots_ohlc(
            o_ge,
            h_ge,
            l_ge,
            c_ge,
            clock,
            cash=paper_cash,
            quantities=qty,
            start=lots_start,
            initial_cash=initial_cash,
        )
        last_nav = float(model_ohlc[-1][3]) if model_ohlc else float(initial_cash)
        lot_w = {"CASH": float(paper_cash)}
        last_px = c_ge[-1] if c_ge else []
        for i, lab in enumerate(ge_asset_labs):
            px = float(last_px[i]) if i < len(last_px) else 0.0
            lot_w[lab] = qty[i] * px
        if last_nav > 1e-12:
            ge_w = {k: v / last_nav for k, v in lot_w.items()}
            ge_vec = _weight_on_labels(ge_w, ge_labels)
    else:
        model_ohlc = _portfolio_ohlc(
            o_ge, h_ge, l_ge, c_ge, ge_rows, initial_cash=initial_cash, cash_yield_per_bar=0.0
        )

    rl_labs = list(_RL_UNIVERSE.keys())
    o_rl, h_rl, l_rl, c_rl = _cols(rl_labs, aligned)
    n_rl = len(rl_labs)
    ew_w = [0.0] + [1.0 / n_rl] * n_rl
    ew_ohlc = _portfolio_ohlc(o_rl, h_rl, l_rl, c_rl, ew_w, initial_cash=initial_cash, cash_yield_per_bar=0.0)
    spy_o = [r[1] for r in spy_rows]
    spy_h = [r[2] for r in spy_rows]
    spy_l = [r[3] for r in spy_rows]
    spy_c = [r[4] for r in spy_rows]
    scale = initial_cash / max(spy_o[0], 1e-12)
    spy_ohlc = [
        (spy_o[i] * scale, max(spy_o[i], spy_h[i], spy_c[i]) * scale, min(spy_o[i], spy_l[i], spy_c[i]) * scale, spy_c[i] * scale)
        for i in range(len(clock))
    ]
    rl_labels = ["Cash"] + rl_labs
    rl_rows = _weight_rows_on_clock(clock, rl_events, rl_labels, start=live_start)
    start_i = next((i for i, ts in enumerate(clock) if ts >= live_start), len(clock))
    prefix = [(initial_cash, initial_cash, initial_cash, initial_cash) for _ in range(start_i)]
    if start_i >= len(clock):
        live_ohlc = prefix
    else:
        live_ohlc = prefix + _portfolio_ohlc(
            o_rl[start_i:],
            h_rl[start_i:],
            l_rl[start_i:],
            c_rl[start_i:],
            rl_rows[start_i:],
            initial_cash=initial_cash,
            cash_yield_per_bar=_RL_CASH_YIELD_PER_BAR,
        )

    nav_model = _nav_from_ohlc(model_ohlc, initial_cash)
    nav_ew = _nav_from_ohlc(ew_ohlc, initial_cash)
    nav_spy = _nav_from_ohlc(spy_ohlc, initial_cash)
    nav_live = _nav_from_ohlc(live_ohlc, initial_cash)
    crypto_src = (existing.get("nav") or {}).get("crypto") if isinstance(existing.get("nav"), dict) else None
    if isinstance(crypto_src, list) and crypto_src:
        last_c = float(crypto_src[-1])
        nav_crypto = [float(crypto_src[0])] * min(len(crypto_src), len(clock))
        if len(nav_crypto) < len(clock):
            nav_crypto = nav_crypto + [last_c] * (len(clock) - len(nav_crypto))
        else:
            nav_crypto = nav_crypto[: len(clock)]
            nav_crypto[-1] = last_c
        if nav_crypto[0] > 0:
            nav_crypto = [x / nav_crypto[0] * initial_cash for x in nav_crypto]
    else:
        nav_crypto = [initial_cash] * len(clock)

    iso = [ts.isoformat(timespec="minutes") for ts in clock]
    last_iso = iso[-1]
    nav = {
        "model": nav_model,
        "spy": nav_spy,
        "equal_weight": nav_ew,
        "live_model": nav_live,
        "crypto": nav_crypto,
    }
    stats = {k: _series_tip_stats(v) for k, v in nav.items()}
    candles = {
        "model": _candles_rows(clock, model_ohlc),
        "spy": _candles_rows(clock, spy_ohlc),
        "equal_weight": _candles_rows(clock, ew_ohlc),
        "live_model": _candles_rows(clock, live_ohlc),
        "crypto": _candles_rows(
            clock,
            [(v, v, v, v) for v in nav_crypto],
        ),
    }
    last_closes = {lab: aligned[lab][3][-1] for lab in ge_asset_labs if lab in aligned}
    positions = []
    ge_map = {str(k).strip().upper().replace(" ", ""): float(v) for k, v in ge_w.items()}
    for i, lab in enumerate(ge_labels):
        w = ge_vec[i]
        price = 1.0 if i == 0 else last_closes.get(lab)
        positions.append(
            {
                "label": lab,
                "ticker": "CASH" if i == 0 else lab,
                "weight": w,
                "value_usd": w * nav_model[-1],
                "price": price if price == price else None,
            }
        )
    now_unix = time.time()
    payload = dict(existing) if existing else {}
    payload.update(
        {
            "schema": "markettrainer.forward_mark.v2",
            "generated_at_utc": _now(),
            "run_id": rid,
            "checkpoint_label": str(existing.get("checkpoint_label") or "locked"),
            "initial_cash": initial_cash,
            "holdout_start": book_start[:10],
            "book_start": book_start[:10],
            "live_model_start": live_start.isoformat(timespec="minutes"),
            "n_bars": len(iso),
            "dates": iso,
            "timestamps": iso,
            "bar_interval": "5m",
            "nav": nav,
            "stats": stats,
            "candles": candles,
            "latest_weights": {ge_labels[i]: ge_vec[i] for i in range(len(ge_labels))},
            "asset_labels": ge_labels,
            "positions": positions,
            "companion_run_id": "RLModel",
            "companion_crypto_run_id": "CREST_DAY",
            "live": {
                "prices_refreshed": True,
                "as_of_bar": last_iso,
                "as_of_utc": _now(),
                "min_refresh_seconds": 300,
                "bar_interval": "5m",
                "source": "yahoo_chart_stdlib",
                "book_start": book_start[:10],
                "session_start": book_start[:10],
                "live_model_start": live_start.isoformat(timespec="minutes"),
                "last_price_bar": last_iso,
                "prices_stale": False,
            },
            "note": (
                f"GENERAL_EQUITY1 share lots since {book_start[:10]}; "
                f"RLModel $100k from {live_start.date().isoformat()} then shadow books as recorded. "
                "Prices from Yahoo 5m chart API (stdlib)."
            ),
        }
    )
    _attach_live_allocations(payload)
    path = EXEC / f"forward_mark_{rid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(_json_safe_lite(payload), indent=2, default=str, allow_nan=False), encoding="utf-8")
    tmp.replace(path)
    stamp_path = EXEC / "forward_live_stamp.json"
    stamp_payload = {
        "run_id": rid,
        "bar_interval": "5m",
        "book_start": book_start[:10],
        "holdout_start": book_start[:10],
        "live_model_start": live_start.isoformat(timespec="minutes"),
        "prices_fetched_at_unix": now_unix,
        "prices_fetched_at_utc": _now(),
        "prices_attempt_at_unix": now_unix,
        "n_bars": len(iso),
        "last_bar": last_iso,
    }
    stamp_path.write_text(json.dumps(stamp_payload, indent=2), encoding="utf-8")
    _write_public_forward(rid, payload)
    print(
        f"[lite-api] stdlib Yahoo refresh {rid}: bars={len(iso)} last={last_iso}",
        file=sys.stderr,
        flush=True,
    )
    return payload


def _sync_live_refresh(
    run_id: str,
    *,
    force: bool,
    reset_book: bool = False,
    timeout_s: float = 75.0,
) -> dict[str, Any] | None:
    """Refresh 5m prices. Prefer stdlib Yahoo (no iCloud venv import)."""
    del reset_book  # stdlib path keeps the persistent book_start
    try:
        return _refresh_forward_prices_stdlib(run_id)
    except Exception as exc:  # noqa: BLE001
        print(f"[lite-api] stdlib Yahoo refresh failed: {exc}", file=sys.stderr, flush=True)
    py = _venv_python()
    code = (
        "from rlbot.forward_live import refresh_forward_mark_live;"
        f"p=refresh_forward_mark_live({run_id!r}, force_price_refresh={bool(force)}, "
        "reset_book=False);"
        "print('ok' if p else 'none')"
    )
    try:
        proc = subprocess.run(
            [py, "-c", code],
            cwd="/tmp",
            timeout=float(timeout_s),
            check=False,
            capture_output=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(ROOT)},
        )
        if proc.returncode != 0:
            err = (proc.stderr or b"").decode("utf-8", errors="replace")[-400:]
            print(f"[lite-api] forward refresh rc={proc.returncode}: {err}", file=sys.stderr, flush=True)
    except (subprocess.TimeoutExpired, OSError) as exc:
        print(f"[lite-api] forward refresh timeout/fail: {exc}", file=sys.stderr, flush=True)
    return _load_mark(run_id)


def _kick_live_refresh(run_id: str, *, force: bool = False) -> None:
    """Single-flight background Yahoo refresh so soft polls do not pile up."""
    global _forward_refreshing
    with _forward_refresh_lock:
        if _forward_refreshing:
            return
        _forward_refreshing = True

    def _worker() -> None:
        global _forward_refreshing
        try:
            mark = _sync_live_refresh(
                run_id, force=force, timeout_s=90.0 if force else 75.0
            )
            if mark is not None:
                _write_public_forward(run_id, mark)
        finally:
            with _forward_refresh_lock:
                _forward_refreshing = False

    threading.Thread(target=_worker, name="lite-forward-refresh", daemon=True).start()


def _strip_durable_series(mark: dict[str, Any]) -> dict[str, Any]:
    """Durable.v1 is retired from the ops forward UI — drop leftover series."""
    out = dict(mark)
    for section in ("nav", "stats", "candles", "allocations"):
        blob = out.get(section)
        if isinstance(blob, dict) and "durable" in blob:
            cleaned = dict(blob)
            cleaned.pop("durable", None)
            out[section] = cleaned
    out.pop("companion_durable_run_id", None)
    return out


def _json_safe_lite(value: Any) -> Any:
    """Finite-float JSON sanitizer (no rlbot import — iCloud-safe)."""
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, dict):
        return {str(k): _json_safe_lite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe_lite(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe_lite(v) for v in value]
    return value


def _write_public_forward(run_id: str, mark: dict[str, Any]) -> None:
    """Keep Vite /data/forward.json in sync with execution marks (no full publish)."""
    payload = {
        "generated_at_utc": _now(),
        "available": True,
        "run_id": run_id,
        "mark": _strip_durable_series(mark),
        "message": None,
    }
    tmp: Path | None = None
    try:
        _PUBLIC_DATA.mkdir(parents=True, exist_ok=True)
        path = _PUBLIC_DATA / "forward.json"
        # Unique tmp avoids concurrent clock-touch / refresh replace races.
        tmp = path.with_name(f"forward.{os.getpid()}.{threading.get_ident()}.tmp")
        text = json.dumps(_json_safe_lite(payload), indent=2, default=str, allow_nan=False)
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
        tmp = None
    except Exception as exc:  # noqa: BLE001
        print(f"[lite-api] forward.json publish skipped: {exc}", file=sys.stderr, flush=True)
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

def _run_detail(run_id: str) -> dict[str, Any] | None:
    known = set(_list_run_ids())
    cached_ids = {str(r.get("run_id")) for r in _run_rows()}
    if run_id not in known and run_id not in cached_ids:
        return None
    # Always refresh this run from disk so expand isn't stuck on a lightweight index row.
    audit = _row_from_disk(run_id)
    if audit.get("oos_sharpe") is None:
        # Fall back to cached row if disk timed out.
        audit = next((r for r in _run_rows() if r.get("run_id") == run_id), audit)
    manifest = _pick_manifest(run_id)
    bt = _pick_backtest(run_id)
    if bt is not None and not audit.get("has_backtest"):
        audit = {
            **audit,
            "oos_sharpe": bt.get("sharpe"),
            "oos_deflated_sharpe": bt.get("deflated_sharpe"),
            "oos_return": bt.get("total_return"),
            "oos_max_drawdown": bt.get("max_drawdown"),
            "ew_excess_return": bt.get("excess_return_vs_equal_weight"),
            "has_backtest": True,
            "training_status": audit.get("training_status") or "completed",
            "progress_pct": audit.get("progress_pct")
            if audit.get("progress_pct") is not None
            else 100.0,
        }
    detail: dict[str, Any] = {
        "run_id": run_id,
        "audit": audit,
        "provenance": {
            "git_commit": manifest.get("git_commit"),
            "git_dirty": manifest.get("git_dirty"),
            "config_hash": manifest.get("config_hash"),
            "data_cache_hash": manifest.get("data_cache_hash"),
            "started_at_utc": manifest.get("started_at_utc"),
            "finished_at_utc": manifest.get("finished_at_utc"),
        },
        "holdout": manifest.get("chronological_holdout"),
        "universe": manifest.get("universe"),
        "backtest": None,
    }
    if bt is not None:
        detail["backtest"] = {
            "checkpoint_label": bt.get("checkpoint_label"),
            "oos_window": bt.get("oos_window"),
            "total_return": bt.get("total_return"),
            "sharpe": bt.get("sharpe"),
            "excess_sharpe": bt.get("excess_sharpe"),
            "max_drawdown": bt.get("max_drawdown"),
            "deflated_sharpe": bt.get("deflated_sharpe"),
            "deflated_sharpe_excess": bt.get("deflated_sharpe_excess"),
            "oos_trials_for_window": bt.get("oos_trials_for_window"),
            "oos_trials_conservative": bt.get("oos_trials_conservative"),
            "equal_weight_daily_return": bt.get("equal_weight_daily_return"),
            "excess_return_vs_equal_weight": bt.get("excess_return_vs_equal_weight"),
            "hash_drift": bt.get("hash_drift"),
            "n_bars": bt.get("n_bars"),
            "portfolio_diagnostics": bt.get("portfolio_diagnostics"),
        }
    return detail


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[lite-api] {self.address_string()} {fmt % args}", file=sys.stderr, flush=True)

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "authorization, content-type, accept")

    def _json(self, status: int, payload: Any) -> None:
        try:
            from rlbot.forward_mark import _json_safe

            payload = _json_safe(payload)
        except Exception:  # noqa: BLE001
            pass
        body = json.dumps(payload, default=str, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/api/health":
            self._json(
                200,
                {
                    "status": "ok",
                    "mode": "lite",
                    "runs_root": str(RUNS),
                    "generated_at_utc": _now(),
                    "auth_required": False,
                    "oos_aggregation": "backtest_summaries",
                },
            )
            return

        if path == "/api/summary":
            self._json(200, _summary(_run_rows()))
            return

        if path == "/api/runs":
            # Serve-only: never scan Runs/ on the request thread (UI uses static
            # snapshots; background enrich keeps execution/api_*_cache.json warm).
            rows = _run_rows()
            try:
                offset = max(0, int((qs.get("offset") or ["0"])[0]))
                limit = min(200, max(1, int((qs.get("limit") or ["50"])[0])))
            except ValueError:
                offset, limit = 0, 50
            prefix = (qs.get("prefix") or [""])[0]
            search = (qs.get("search") or [""])[0].casefold()
            status = (qs.get("status") or [""])[0]
            filtered = rows
            if prefix:
                filtered = [r for r in filtered if str(r.get("run_id", "")).startswith(prefix)]
            if search:
                filtered = [
                    r for r in filtered if search in str(r.get("run_id", "")).casefold()
                ]
            if status == "completed":
                filtered = [r for r in filtered if r.get("training_status") == "completed"]
            elif status == "interrupted":
                filtered = [r for r in filtered if r.get("training_status") == "interrupted"]
            elif status == "active":
                filtered = [
                    r
                    for r in filtered
                    if r.get("training_status") not in ("completed", "interrupted")
                ]
            counts = {
                "all": len(rows),
                "completed": sum(r.get("training_status") == "completed" for r in rows),
                "active": sum(
                    r.get("training_status") not in ("completed", "interrupted") for r in rows
                ),
                "interrupted": sum(r.get("training_status") == "interrupted" for r in rows),
                "with_backtest": sum(bool(r.get("has_backtest")) for r in rows),
            }
            page = [_normalize_run_row(r) for r in filtered[offset : offset + limit]]
            self._json(
                200,
                {
                    "generated_at_utc": _now(),
                    "runs": page,
                    "total": len(filtered),
                    "offset": offset,
                    "limit": limit,
                    "counts": counts,
                },
            )
            return

        m = re.match(r"^/api/runs/([^/]+)$", path)
        if m:
            rid = m.group(1)
            if not _RUN_ID_RE.match(rid):
                self._json(400, {"detail": "Invalid run id"})
                return
            detail = _run_detail(rid)
            if detail is None:
                self._json(404, {"detail": "Unknown run id"})
                return
            self._json(200, detail)
            return

        if path == "/api/dashboard":
            rows = _run_rows()
            self._json(
                200,
                {
                    "generated_at_utc": _now(),
                    "summary": _summary(rows),
                    "recent_runs": rows[:6],
                    "window_sharpes": [],
                },
            )
            return

        if path == "/api/results":
            payload = _read_json(OOS_CACHE)
            rows = payload.get("rows") if isinstance(payload, dict) else []
            if not isinstance(rows, list):
                rows = []
            cohort = (qs.get("cohort") or [""])[0]
            if cohort:
                rows = [r for r in rows if isinstance(r, dict) and str(r.get("cohort")) == cohort]
            self._json(
                200,
                {
                    "generated_at_utc": _now(),
                    "rows": rows,
                    "meta": {
                        "source": "execution/api_oos_cache.json (lite)",
                        "published_runs": len(rows),
                        "runs_with_backtest": len(rows),
                        "runs_with_benchmarks": sum(
                            1 for r in rows if isinstance(r, dict) and r.get("has_benchmarks")
                        ),
                        "total_runs": len(_run_rows()),
                    },
                },
            )
            return

        if path == "/api/forward":
            rid = (qs.get("run_id") or [""])[0].strip() or (_active_run_id() or "")
            live = (qs.get("live") or ["1"])[0] not in ("0", "false", "False")
            force = (qs.get("force_refresh") or ["0"])[0] in ("1", "true", "True")
            reset_book = (qs.get("reset_book") or ["0"])[0] in ("1", "true", "True")
            if not rid or not _RUN_ID_RE.match(rid):
                self._json(
                    200,
                    {
                        "generated_at_utc": _now(),
                        "available": False,
                        "run_id": rid or None,
                        "mark": None,
                        "message": "No forward mark yet.",
                    },
                )
                return
            mark = _load_mark(rid)
            # Prefer Yahoo stamp age — mark mtime is bumped by clock-touch every poll.
            price_age = _prices_age_s(rid)
            if live and reset_book:
                # Reset must complete before paint — keep a bounded wait.
                mark = _sync_live_refresh(
                    rid, force=True, reset_book=True, timeout_s=90.0
                ) or mark
            elif live and (force or price_age is None or price_age > 300.0):
                # Kick Yahoo in the background. Do not force=True just because
                # prices are stale — that bypassed the 60s cooldown and hammered
                # Yahoo after a failed pull. refresh_forward_mark_live already
                # fetches when the stamp is older than 5 minutes.
                _kick_live_refresh(rid, force=force)
            if mark is None:
                self._json(
                    200,
                    {
                        "generated_at_utc": _now(),
                        "available": False,
                        "run_id": rid,
                        "mark": None,
                        "message": f"Run {rid} has no execution/forward_mark_{rid}.json",
                    },
                )
                return
            # Instant clock touch so the tip never sticks on yesterday while a
            # Yahoo subprocess is stuck importing under iCloud.
            if live and not reset_book:
                mark = _touch_forward_clock(rid, mark) or mark
            # Avoid importing rlbot on the request path (iCloud hang).
            mark = _strip_durable_series(mark)
            weights = mark.get("weights")
            if isinstance(weights, list) and len(weights) > 400:
                mark = {**mark, "weights": weights[:: max(1, len(weights) // 200)]}
            _write_public_forward(rid, mark)
            self._json(
                200,
                {
                    "generated_at_utc": _now(),
                    "available": True,
                    "run_id": rid,
                    "mark": mark,
                    "message": None,
                },
            )
            return

        self._json(404, {"detail": f"Not found: {path}"})


def _write_loop_status(payload: dict[str, Any]) -> None:
    path = EXEC / "forward_loop_status.json"
    try:
        EXEC.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except OSError as exc:
        print(f"[lite-api] status write skipped: {exc}", file=sys.stderr, flush=True)


def _acquire_collect_lock(*, blocking: bool = True) -> int | None:
    path = EXEC / "forward_loop.lock"
    try:
        EXEC.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        flags = fcntl.LOCK_EX if blocking else (fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, flags)
        return fd
    except (OSError, BlockingIOError):
        return None


def _stamp_collector_mark(run_id: str, mark: dict[str, Any] | None, *, interval_s: int) -> dict[str, Any] | None:
    if not isinstance(mark, dict):
        return mark
    live = dict(mark.get("live") or {})
    live["collector"] = {
        "running": True,
        "last_tick_utc": _now(),
        "interval_s": int(interval_s),
        "mode": "lite",
    }
    mark = {**mark, "live": live}
    mark_path = EXEC / f"forward_mark_{run_id}.json"
    try:
        mark_path.write_text(json.dumps(mark, indent=2, default=str), encoding="utf-8")
    except OSError:
        pass
    _write_public_forward(run_id, mark)
    return mark


def _maybe_paper_once(status: dict[str, Any]) -> dict[str, Any]:
    """Paper + RLModel shadow via the venv. Parent already holds the collect lock."""
    py = _venv_python()
    script = ROOT / "scripts" / "live_forward_loop.py"
    if not script.is_file():
        return {"ok": False, "error": "missing live_forward_loop.py"}
    try:
        proc = subprocess.run(
            [py, str(script), "--once", "--skip-lock", "--no-prices"],
            cwd="/tmp",
            timeout=720.0,
            check=False,
            capture_output=True,
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": str(ROOT),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace")[-300:]
        return {"ok": False, "error": err or f"rc={proc.returncode}"}
    fresh = _read_json(EXEC / "forward_loop_status.json")
    shadow = {}
    if isinstance(fresh, dict):
        sleeves = fresh.get("sleeves") if isinstance(fresh.get("sleeves"), dict) else {}
        shadow = sleeves.get("RLModel") or sleeves.get(
            os.environ.get("MARKETTRAINER_LIVE_RUN_ID") or "RLModel"
        ) or {}
        as_of = fresh.get("paper_ge1_date") or datetime.now().strftime("%Y-%m-%d")
        return {
            "ok": True,
            "as_of": as_of,
            "via": "live_forward_loop --once --skip-lock",
            "rl_shadow": shadow,
            "rl_shadow_date": fresh.get("rl_shadow_date"),
        }
    return {"ok": True, "as_of": datetime.now().strftime("%Y-%m-%d"), "via": "live_forward_loop --once --skip-lock"}


def collect_once(*, interval_s: int = 300, run_paper: bool = True) -> dict[str, Any]:
    """Stdlib Yahoo 5m rewrite + optional paper subprocess. Safe for LaunchAgents."""
    rid = _active_run_id() or "GENERAL_EQUITY1"
    print(f"[lite-api] collect-once run_id={rid}", flush=True)
    mark = _sync_live_refresh(rid, force=True, timeout_s=90.0)
    mark = _stamp_collector_mark(rid, mark, interval_s=interval_s)
    prior = _read_json(EXEC / "forward_loop_status.json")
    if not isinstance(prior, dict):
        prior = {}
    live = (mark or {}).get("live") if isinstance(mark, dict) else {}
    status = {
        "schema": "markettrainer.forward_loop.v1",
        "pid": os.getpid(),
        "run_id": rid,
        "last_tick_utc": _now(),
        "interval_s": int(interval_s),
        "n_bars": (mark or {}).get("n_bars") if isinstance(mark, dict) else None,
        "last_price_bar": (live or {}).get("last_price_bar") if isinstance(live, dict) else None,
        "prices_stale": (live or {}).get("prices_stale") if isinstance(live, dict) else None,
        "mode": "lite",
        "paper_attempt_date": prior.get("paper_attempt_date"),
        "paper_ge1_date": prior.get("paper_ge1_date"),
        "sleeves": {"paper": {"ok": True, "skipped": "pending"}},
    }
    _write_loop_status(status)
    paper: dict[str, Any] = {"ok": True, "skipped": "disabled"}
    if run_paper:
        paper = _maybe_paper_once(prior)
        paper_ok = bool(paper.get("ok"))
        fresh = _read_json(EXEC / "forward_loop_status.json")
        if isinstance(fresh, dict) and (
            fresh.get("rl_shadow_date") or (fresh.get("sleeves") or {}).get("RLModel")
        ):
            status["paper_ge1_date"] = fresh.get("paper_ge1_date") or paper.get("as_of")
            status["paper_crest_date"] = fresh.get("paper_crest_date")
            status["rl_shadow_date"] = fresh.get("rl_shadow_date") or paper.get("rl_shadow_date")
            status["sleeves"] = fresh.get("sleeves") or {"paper": paper}
        else:
            status["paper_attempt_date"] = datetime.now().strftime("%Y-%m-%d")
            if paper_ok and paper.get("as_of"):
                status["paper_ge1_date"] = paper.get("as_of")
            status["sleeves"] = {"paper": paper}
            if paper.get("rl_shadow_date"):
                status["rl_shadow_date"] = paper.get("rl_shadow_date")
        status["pid"] = os.getpid()
        status["mode"] = "lite"
        if isinstance(mark, dict):
            mark = _attach_live_allocations(mark)
            mark = _stamp_collector_mark(rid, mark, interval_s=interval_s)
        _write_loop_status(status)
    print(
        f"[lite-api] collect-once done bars={status.get('n_bars')} last={status.get('last_price_bar')}",
        flush=True,
    )
    return status


def collect_loop(*, interval_s: int = 300) -> None:
    print(f"[lite-api] collect-loop interval={interval_s}s pid={os.getpid()}", flush=True)
    while True:
        lock_fd = _acquire_collect_lock(blocking=True)
        if lock_fd is None:
            print(
                "[lite-api] collect-loop lock unavailable (retry in 30s)",
                flush=True,
            )
            time.sleep(30)
            continue
        try:
            while True:
                t0 = time.time()
                try:
                    collect_once(interval_s=interval_s, run_paper=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"[lite-api] collect-once failed: {exc}", file=sys.stderr, flush=True)
                elapsed = time.time() - t0
                time.sleep(max(5.0, float(interval_s) - elapsed))
        finally:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
            except OSError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--collect-once",
        action="store_true",
        help="One stdlib Yahoo 5m rewrite + paper tick, then exit (LaunchAgent)",
    )
    parser.add_argument(
        "--collect-loop",
        action="store_true",
        help="KeepAlive loop: collect-once every --interval seconds",
    )
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()
    if args.collect_once:
        collect_once(interval_s=max(30, int(args.interval)))
        return
    if args.collect_loop:
        collect_loop(interval_s=max(30, int(args.interval)))
        return
    ThreadingHTTPServer.allow_reuse_address = True
    server = None
    last_exc: BaseException | None = None
    for attempt in range(3):
        try:
            server = ThreadingHTTPServer((args.host, args.port), Handler)
            break
        except OSError as exc:
            last_exc = exc
            print(
                f"[lite-api] bind failed on {args.host}:{args.port} (attempt {attempt + 1}/3): {exc}",
                flush=True,
            )
            time.sleep(0.6 + attempt * 0.4)
    if server is None:
        raise SystemExit(1) from last_exc
    print(
        f"[lite-api] serving on http://{args.host}:{args.port} "
        f"(cache-first; background enrich → execution/ + public/data)",
        flush=True,
    )
    # Immediate snapshot from disk caches, then deeper enrich in background.
    threading.Thread(target=_publish_frontend_snapshots, name="lite-publish-boot", daemon=True).start()
    _kick_enrich()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[lite-api] shutdown", flush=True)
    finally:
        try:
            server.server_close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
