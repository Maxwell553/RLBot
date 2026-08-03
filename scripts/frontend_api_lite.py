#!/usr/bin/env python3
"""Stdlib-only operator API fallback (no FastAPI / heavy rlbot imports).

Serves health / runs / summary / forward from ``execution/`` plus bounded reads
of ``Runs/*/backtest_summary*.json`` (subprocess + timeout) so the UI keeps
OOS metrics even when the full FastAPI app cannot import under iCloud load.

    python3 scripts/frontend_api_lite.py --port 8787
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
EXEC = ROOT / "execution"
RUNS = ROOT / "Runs"
RUNS_CACHE = EXEC / "api_runs_cache.json"
OOS_CACHE = EXEC / "api_oos_cache.json"
ACTIVE_PTR = EXEC / "forward_active.json"
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
_COHORT_RUN_RE = re.compile(r"^(W\d+)_(.+)$", re.IGNORECASE)
_WINDOW_COHORT_RE = re.compile(r"^W(\d+)_(.+)$", re.IGNORECASE)
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_sort_key(run_id: str) -> tuple[Any, ...]:
    """Newest cohort first, then W1…W5 within the cohort."""
    m = _WINDOW_COHORT_RE.match(run_id)
    if m is None:
        return (1, 0, 0, run_id)
    window = int(m.group(1))
    cohort = m.group(2)
    try:
        return (0, -int(cohort), window, run_id)
    except ValueError:
        return (0, 0, window, run_id.lower())


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
        m = re.match(r"^W(\d+)_(.+)$", rid, re.IGNORECASE)
        if m is not None:
            window = int(m.group(1))
            cohort = m.group(2)
            try:
                cohort_key = -int(cohort)  # 804 before 803
            except ValueError:
                cohort_key = 0
            return (missing_oos, activeish, cohort_key, window, rid)
        if rid.startswith("LIVE_"):
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


def _sync_live_refresh(
    run_id: str,
    *,
    force: bool,
    reset_book: bool = False,
    timeout_s: float = 90.0,
) -> dict[str, Any] | None:
    """Block until Yahoo refresh writes execution/forward_mark_*.json (or timeout)."""
    py = _venv_python()
    # Chart-API fetch is fast; the cost is importing rlbot under iCloud load.
    code = (
        "from rlbot.forward_live import refresh_forward_mark_live;"
        f"p=refresh_forward_mark_live({run_id!r}, force_price_refresh={bool(force)}, "
        f"reset_book={bool(reset_book)});"
        "print('ok' if p else 'none')"
    )
    try:
        proc = subprocess.run(
            [py, "-c", code],
            cwd=str(ROOT),
            timeout=float(timeout_s),
            check=False,
            capture_output=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        if proc.returncode != 0:
            err = (proc.stderr or b"").decode("utf-8", errors="replace")[-400:]
            print(f"[lite-api] forward refresh rc={proc.returncode}: {err}", file=sys.stderr, flush=True)
    except (subprocess.TimeoutExpired, OSError) as exc:
        print(f"[lite-api] forward refresh timeout/fail: {exc}", file=sys.stderr, flush=True)
    return _load_mark(run_id)


def _kick_live_refresh(run_id: str) -> None:
    def _worker() -> None:
        _sync_live_refresh(run_id, force=False, timeout_s=50.0)

    threading.Thread(target=_worker, name="lite-forward-refresh", daemon=True).start()


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
        body = json.dumps(payload, default=str).encode("utf-8")
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
            age = _mark_age_s(rid)
            stale = age is None or age > 180.0  # Yahoo 5m marks go stale after ~3 minutes
            if live and (force or reset_book):
                # Only force_refresh / reset_book blocks; soft polls never wait on Yahoo.
                mark = _sync_live_refresh(
                    rid, force=True, reset_book=reset_book, timeout_s=120.0
                ) or mark
            elif live and (mark is None or stale):
                _kick_live_refresh(rid)
                mark = _load_mark(rid) or mark
            elif live:
                _kick_live_refresh(rid)
                mark = _load_mark(rid) or mark
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
            try:
                from rlbot.forward_mark import merge_crypto_companion

                mark = merge_crypto_companion(mark)
            except Exception:  # noqa: BLE001
                pass
            weights = mark.get("weights")
            if isinstance(weights, list) and len(weights) > 400:
                mark = {**mark, "weights": weights[:: max(1, len(weights) // 200)]}
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
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
