#!/usr/bin/env python3
"""Publish page-ready JSON snapshots for the Vite UI (no request-path Runs/ scans).

Reads ``execution/api_runs_cache.json``, ``execution/api_oos_cache.json``, and
``execution/forward_mark_*.json``, then writes:

  frontend/public/data/
    meta.json
    summary.json
    dashboard.json
    runs.json
    results.json
    forward.json
    details/<run_id>.json   (optional, from existing caches / timed disk reads)

The SPA loads these as static files (milliseconds). Re-run after train/backtest
or from ``frontend/scripts/dev.mjs`` on boot.

    python3 scripts/publish_frontend_data.py
    python3 scripts/publish_frontend_data.py --with-details          # cache-only detail stubs
    python3 scripts/publish_frontend_data.py --enrich-details        # timed Runs/ reads
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
EXEC = ROOT / "execution"
RUNS = ROOT / "Runs"
OUT = ROOT / "frontend" / "public" / "data"
RUNS_CACHE = EXEC / "api_runs_cache.json"
OOS_CACHE = EXEC / "api_oos_cache.json"
ACTIVE_PTR = EXEC / "forward_active.json"

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
# W{window}_{cohort}[letter] or W{window}_{cohort}_s{seed} (811a/b variants, seed ensembles).
_WINDOW_COHORT_RE = re.compile(r"^W(\d+)_(\d+)([a-z]*)(?:_[A-Za-z0-9]+)?$", re.IGNORECASE)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Any | None:
    try:
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":"), default=str), encoding="utf-8")
    tmp.replace(path)


def _run_sort_key(run_id: str) -> tuple[Any, ...]:
    m = _WINDOW_COHORT_RE.match(run_id)
    if m is None:
        return (1, 0, 0, "", run_id)
    window = int(m.group(1))
    cohort = int(m.group(2))
    letter = (m.group(3) or "").lower()
    return (0, -cohort, window, letter, run_id)


def _normalize_run(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if not isinstance(out.get("labels"), list):
        out["labels"] = []
    if not isinstance(out.get("warnings"), list):
        out["warnings"] = []
    return out


def _load_runs() -> list[dict[str, Any]]:
    payload = _read_json(RUNS_CACHE)
    records = payload.get("records") if isinstance(payload, dict) else None
    rows: list[dict[str, Any]] = []
    if isinstance(records, list):
        for rec in records:
            if isinstance(rec, dict) and rec.get("run_id"):
                rows.append(_normalize_run(rec))
    rows.sort(key=lambda r: _run_sort_key(str(r.get("run_id") or "")))
    return rows


def _load_oos() -> list[dict[str, Any]]:
    payload = _read_json(OOS_CACHE)
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict) and r.get("run_id")]


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [r for r in rows if r.get("training_status") == "completed"]
    scored = [r for r in rows if isinstance(r.get("oos_sharpe"), (int, float))]
    best = max(scored, key=lambda r: float(r["oos_sharpe"])) if scored else None
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


def _window_sharpes(oos_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_window: dict[str, list[float]] = {}
    for row in oos_rows:
        window = row.get("window")
        sharpe = row.get("model_sh")
        if isinstance(window, str) and isinstance(sharpe, (int, float)) and math.isfinite(float(sharpe)):
            by_window.setdefault(window, []).append(float(sharpe))
    return [
        {"window": window, "sharpe": round(_median(values), 2)}
        for window, values in sorted(by_window.items())
        if values
    ]


def _cohort_sort_key(cohort: str) -> tuple[Any, ...]:
    m = re.match(r"^(\d+)([a-z]*)$", str(cohort), re.IGNORECASE)
    if m:
        return (0, -int(m.group(1)), (m.group(2) or "").lower())
    return (1, 0, str(cohort))


def _active_run_id() -> str | None:
    data = _read_json(ACTIVE_PTR)
    if isinstance(data, dict):
        rid = str(data.get("run_id") or "").strip()
        if rid:
            return rid
    marks = sorted(EXEC.glob("forward_mark_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in marks:
        name = path.name.removeprefix("forward_mark_").removesuffix(".json")
        if name:
            return name
    return None


def _load_forward() -> dict[str, Any]:
    rid = _active_run_id() or ""
    if not rid or not _RUN_ID_RE.match(rid):
        return {
            "generated_at_utc": _now(),
            "available": False,
            "run_id": rid or None,
            "mark": None,
            "message": "No forward mark yet.",
        }
    mark = _read_json(EXEC / f"forward_mark_{rid}.json")
    if not isinstance(mark, dict):
        return {
            "generated_at_utc": _now(),
            "available": False,
            "run_id": rid,
            "mark": None,
            "message": f"Run {rid} has no execution/forward_mark_{rid}.json",
        }
    # Ensure CrestDay companion series is present when its mark exists.
    try:
        from rlbot.forward_mark import merge_crypto_companion

        mark = merge_crypto_companion(mark)
    except Exception:  # noqa: BLE001
        pass
    # Durable.v1 retired from the ops forward surface.
    for section in ("nav", "stats", "candles", "allocations"):
        blob = mark.get(section)
        if isinstance(blob, dict) and "durable" in blob:
            cleaned = dict(blob)
            cleaned.pop("durable", None)
            mark[section] = cleaned
    mark.pop("companion_durable_run_id", None)
    weights = mark.get("weights")
    if isinstance(weights, list) and len(weights) > 400:
        mark = {**mark, "weights": weights[:: max(1, len(weights) // 200)]}
    return {
        "generated_at_utc": _now(),
        "available": True,
        "run_id": rid,
        "mark": mark,
        "message": None,
    }


def _timed_read_json(path: Path, timeout_s: float = 2.5) -> Any | None:
    """Read small JSON with a hard timeout (iCloud Desktop can take >1s)."""
    try:
        size = path.stat().st_size
    except OSError:
        return None

    if size <= 1_000_000:
        import threading

        box: dict[str, Any] = {"data": None}

        def _worker() -> None:
            try:
                box["data"] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                box["data"] = None

        thread = threading.Thread(target=_worker, name="publish-read", daemon=True)
        thread.start()
        thread.join(timeout=float(timeout_s))
        if thread.is_alive():
            return None
        data = box.get("data")
        return data if isinstance(data, dict) or isinstance(data, list) else data

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
        return json.loads(proc.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


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


def _pick_manifest(run_id: str) -> dict[str, Any]:
    data = _timed_read_json(RUNS / run_id / "manifest.json", timeout_s=2.0)
    return data if isinstance(data, dict) else {}


def _patch_missing_oos(rows: list[dict[str, Any]], *, budget_s: float = 60.0) -> int:
    """Fill completed rows that still lack OOS from Runs/*/backtest_summary*.json."""
    t0 = time.time()
    patched = 0
    # Newest completed holes first — otherwise a 20s budget dies on old missing files
    # and never reaches a just-scored 817 cell.
    ordered = sorted(
        rows,
        key=lambda r: (
            0 if r.get("oos_sharpe") is None else 1,
            0 if r.get("training_status") == "completed" else 1,
            _run_sort_key(str(r.get("run_id") or "")),
        ),
    )
    for row in ordered:
        if time.time() - t0 > budget_s:
            break
        if row.get("oos_sharpe") is not None:
            continue
        rid = str(row.get("run_id") or "")
        if not rid or not _RUN_ID_RE.match(rid):
            continue
        bt = _pick_backtest(rid, timeout_s=3.0)
        if bt is None:
            continue
        row["oos_sharpe"] = bt.get("sharpe")
        row["oos_deflated_sharpe"] = bt.get("deflated_sharpe")
        row["oos_return"] = bt.get("total_return")
        row["oos_max_drawdown"] = bt.get("max_drawdown")
        row["ew_excess_return"] = bt.get("excess_return_vs_equal_weight")
        row["has_backtest"] = True
        row["training_status"] = "completed"
        row["progress_pct"] = 100.0
        if row.get("nominal_timesteps") is not None:
            try:
                row["elapsed_timesteps"] = int(row["nominal_timesteps"])
            except (TypeError, ValueError):
                pass
        ckpt = bt.get("checkpoint_label")
        if ckpt:
            labels = row.get("labels") if isinstance(row.get("labels"), list) else []
            if str(ckpt) not in labels:
                row["labels"] = [str(ckpt), *labels]
        patched += 1
    return patched


def _backtest_payload_from_summary(bt: dict[str, Any]) -> dict[str, Any]:
    return {
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


def _backtest_payload_from_audit(row: dict[str, Any]) -> dict[str, Any] | None:
    """Build a detail ``backtest`` block from runs-index OOS fields (no Runs/ I/O)."""
    if not row.get("has_backtest") and row.get("oos_sharpe") is None and row.get("oos_return") is None:
        return None
    labels = row.get("labels") if isinstance(row.get("labels"), list) else []
    return {
        "checkpoint_label": labels[0] if labels else None,
        "oos_window": row.get("window"),
        "total_return": row.get("oos_return"),
        "sharpe": row.get("oos_sharpe"),
        "excess_sharpe": None,
        "max_drawdown": row.get("oos_max_drawdown"),
        "deflated_sharpe": row.get("oos_deflated_sharpe"),
        "deflated_sharpe_excess": None,
        "oos_trials_for_window": None,
        "oos_trials_conservative": None,
        "equal_weight_daily_return": row.get("equal_weight_daily_return"),
        "excess_return_vs_equal_weight": row.get("ew_excess_return"),
        "hash_drift": None,
        "n_bars": None,
        "portfolio_diagnostics": None,
    }


def _detail_from_caches(
    run_id: str,
    audit: dict[str, Any] | None,
    *,
    read_disk: bool,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    bt: dict[str, Any] | None = None
    if read_disk:
        manifest = _pick_manifest(run_id)
        bt = _pick_backtest(run_id)
    row = dict(audit or {"run_id": run_id})
    if bt is not None:
        row = {
            **row,
            "oos_sharpe": bt.get("sharpe"),
            "oos_deflated_sharpe": bt.get("deflated_sharpe"),
            "oos_return": bt.get("total_return"),
            "oos_max_drawdown": bt.get("max_drawdown"),
            "ew_excess_return": bt.get("excess_return_vs_equal_weight"),
            "has_backtest": True,
            "training_status": row.get("training_status") or "completed",
            "progress_pct": row.get("progress_pct") if row.get("progress_pct") is not None else 100.0,
        }
    detail: dict[str, Any] = {
        "run_id": run_id,
        "audit": _normalize_run(row),
        "provenance": {
            "git_commit": manifest.get("git_commit"),
            "git_dirty": manifest.get("git_dirty") if manifest else row.get("git_dirty"),
            "config_hash": manifest.get("config_hash"),
            "data_cache_hash": manifest.get("data_cache_hash"),
            "started_at_utc": manifest.get("started_at_utc") or row.get("started_at_utc"),
            "finished_at_utc": manifest.get("finished_at_utc") or row.get("finished_at_utc"),
        },
        "holdout": manifest.get("chronological_holdout"),
        "universe": manifest.get("universe"),
        "backtest": None,
    }
    if bt is not None:
        detail["backtest"] = _backtest_payload_from_summary(bt)
    else:
        # Cache-only stubs still carry OOS on the audit row; expose them in the
        # backtest block so expand doesn't say "no OOS" while the list shows Sharpe.
        detail["backtest"] = _backtest_payload_from_audit(row)
    return detail


def publish(
    *,
    with_details: bool = True,
    enrich_details: bool = False,
    details_budget_s: float = 20.0,
    max_details: int = 80,
) -> dict[str, Any]:
    t0 = time.time()
    rows = _load_runs()
    # Boot/publish path: heal cache holes where backtest files exist but the
    # lite enrich timed out (common on iCloud Desktop).
    patched = _patch_missing_oos(rows, budget_s=max(15.0, details_budget_s))
    if patched:
        try:
            _atomic_write_json(
                RUNS_CACHE,
                {
                    "generated_at_utc": _now(),
                    "n": len(rows),
                    "records": rows,
                    "note": f"publish patched {patched} missing OOS from backtest_summary",
                },
            )
        except OSError:
            pass
    oos_rows = _load_oos()
    # Also merge freshly patched sharpes into the OOS results list when absent.
    if patched:
        oos_by = {
            str(r.get("run_id")): r
            for r in oos_rows
            if isinstance(r, dict) and r.get("run_id")
        }
        for row in rows:
            rid = str(row.get("run_id") or "")
            if not rid or row.get("oos_sharpe") is None or rid in oos_by:
                continue
            m = _WINDOW_COHORT_RE.match(rid)
            if m is None:
                continue
            oos_by[rid] = {
                "run_id": rid,
                "window": f"W{m.group(1)}",
                "cohort": m.group(2),
                "model_ret": row.get("oos_return"),
                "model_sh": row.get("oos_sharpe"),
                "ew_ret": None,
                "ew_sh": None,
                "spy_ret": None,
                "spy_sh": None,
                "has_benchmarks": False,
            }
        oos_rows = list(oos_by.values())
        try:
            _atomic_write_json(
                OOS_CACHE,
                {"generated_at_utc": _now(), "n": len(oos_rows), "rows": oos_rows},
            )
        except OSError:
            pass
    summary = _summary(rows)
    dashboard = {
        "generated_at_utc": _now(),
        "summary": summary,
        "recent_runs": rows[:6],
        "window_sharpes": _window_sharpes(oos_rows),
    }
    counts = {
        "all": len(rows),
        "completed": sum(r.get("training_status") == "completed" for r in rows),
        "active": sum(r.get("training_status") not in ("completed", "interrupted") for r in rows),
        "interrupted": sum(r.get("training_status") == "interrupted" for r in rows),
        "with_backtest": sum(bool(r.get("has_backtest")) for r in rows),
    }
    runs_payload = {
        "generated_at_utc": _now(),
        "runs": rows,
        "total": len(rows),
        "offset": 0,
        "limit": len(rows),
        "counts": counts,
    }
    cohorts = sorted({str(r.get("cohort")) for r in oos_rows if r.get("cohort")}, key=_cohort_sort_key)
    results = {
        "generated_at_utc": _now(),
        "available": bool(oos_rows) or bool(cohorts),
        "cohorts": cohorts,
        "rows": oos_rows,
        "coverage": {
            "source": "execution/api_oos_cache.json (published snapshot)",
            "published_rows": len(oos_rows),
            "published_runs": len({str(r.get("run_id")) for r in oos_rows}),
            "runs_with_backtest": len(oos_rows),
            "runs_with_benchmarks": sum(1 for r in oos_rows if r.get("has_benchmarks")),
            "total_runs": len(rows),
        },
    }
    forward = _load_forward()
    meta = {
        "generated_at_utc": _now(),
        "source": "scripts/publish_frontend_data.py",
        "n_runs": len(rows),
        "n_oos_rows": len(oos_rows),
        "forward_run_id": forward.get("run_id"),
        "elapsed_ms": None,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(OUT / "summary.json", summary)
    _atomic_write_json(OUT / "dashboard.json", dashboard)
    _atomic_write_json(OUT / "runs.json", runs_payload)
    _atomic_write_json(OUT / "results.json", results)
    _atomic_write_json(OUT / "forward.json", forward)

    details_written = 0
    if with_details:
        details_dir = OUT / "details"
        details_dir.mkdir(parents=True, exist_ok=True)
        by_id = {str(r.get("run_id")): r for r in rows}
        if enrich_details:
            ordered = sorted(
                by_id.keys(),
                key=lambda rid: (0 if by_id[rid].get("has_backtest") else 1, _run_sort_key(rid)),
            )
            for rid in ordered[: max(0, max_details)]:
                if time.time() - t0 > details_budget_s:
                    break
                if not _RUN_ID_RE.match(rid):
                    continue
                detail = _detail_from_caches(rid, by_id.get(rid), read_disk=True)
                _atomic_write_json(details_dir / f"{rid}.json", detail)
                details_written += 1
        # Cache-only stubs for remaining runs (no Runs/ I/O). Prefer synthesizing
        # backtest from audit OOS so expand matches the list Sharpe/DSR.
        for rid, audit in by_id.items():
            path = details_dir / f"{rid}.json"
            if enrich_details and path.is_file():
                continue  # keep disk-enriched file written above
            detail = _detail_from_caches(rid, audit, read_disk=False)
            if not enrich_details and path.is_file():
                existing = _read_json(path)
                # Don't discard a richer on-disk backtest block from a prior enrich.
                if (
                    isinstance(existing, dict)
                    and isinstance(existing.get("backtest"), dict)
                    and existing["backtest"].get("n_bars") is not None
                    and (
                        not isinstance(detail.get("backtest"), dict)
                        or detail["backtest"].get("n_bars") is None
                    )
                ):
                    detail["backtest"] = existing["backtest"]
                    if isinstance(existing.get("holdout"), dict) and not detail.get("holdout"):
                        detail["holdout"] = existing["holdout"]
                    if isinstance(existing.get("provenance"), dict):
                        prov = dict(detail.get("provenance") or {})
                        for key, val in existing["provenance"].items():
                            if prov.get(key) in (None, "") and val not in (None, ""):
                                prov[key] = val
                        detail["provenance"] = prov
            _atomic_write_json(path, detail)
            details_written += 1

    meta["details_written"] = details_written
    meta["elapsed_ms"] = round((time.time() - t0) * 1000, 1)
    _atomic_write_json(OUT / "meta.json", meta)
    return meta


def _rebuild_oos_cache(*, budget_s: float = 120.0) -> int:
    """Re-parse Runs/*/backtest_summary*.json into execution/api_oos_cache.json.

    Uses the lite API parser (reads ``detailed`` EW/SPY sleeves) so older
    ``--detailed`` summaries populate the OOS UI without re-backtesting.
    """
    # Import lazily — keep publish boot light when not rebuilding.
    sys.path.insert(0, str(ROOT / "scripts"))
    import frontend_api_lite as lite  # type: ignore

    t0 = time.time()
    ids = lite._list_run_ids()
    by_id: dict[str, dict[str, Any]] = {}
    existing = _read_json(OOS_CACHE)
    if isinstance(existing, dict):
        for row in existing.get("rows") or []:
            if isinstance(row, dict) and row.get("run_id"):
                by_id[str(row["run_id"])] = row
    upgraded = 0
    for rid in ids:
        if time.time() - t0 > budget_s:
            break
        bt = lite._pick_backtest(rid, timeout_s=3.5)
        if bt is None:
            continue
        oos = lite._oos_row_from_backtest(rid, bt)
        if oos is None:
            continue
        prev = by_id.get(rid)
        by_id[rid] = oos
        if not prev or prev.get("has_benchmarks") != oos.get("has_benchmarks") or (
            prev.get("ew_sh") is None and oos.get("ew_sh") is not None
        ) or (prev.get("spy_sh") is None and oos.get("spy_sh") is not None):
            upgraded += 1
    rows = list(by_id.values())
    _atomic_write_json(
        OOS_CACHE,
        {
            "generated_at_utc": _now(),
            "n": len(rows),
            "rows": rows,
            "note": "rebuild-oos from backtest_summary (detailed EW/SPY sleeves)",
        },
    )
    print(
        f"[publish-frontend-data] rebuild-oos: {len(rows)} rows, "
        f"{upgraded} upgraded/added, {sum(1 for r in rows if r.get('has_benchmarks'))} with benchmarks "
        f"({round((time.time() - t0) * 1000, 1)}ms)",
        flush=True,
    )
    return upgraded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-details",
        action="store_true",
        default=True,
        help="Write details/<run_id>.json stubs from the runs cache (default on)",
    )
    parser.add_argument(
        "--no-details",
        action="store_true",
        help="Skip details/ entirely (page index JSON only)",
    )
    parser.add_argument(
        "--enrich-details",
        action="store_true",
        help="Timed Reads of Runs/*/manifest + backtest_summary into details/",
    )
    parser.add_argument(
        "--details-budget-s",
        type=float,
        default=20.0,
        help="Wall-clock budget for --enrich-details (default 20)",
    )
    parser.add_argument(
        "--max-details",
        type=int,
        default=80,
        help="Max run ids to enrich from disk (rest get cache-only stubs)",
    )
    parser.add_argument(
        "--rebuild-oos",
        action="store_true",
        help="Re-parse backtest summaries into api_oos_cache.json (EW/SPY sleeves)",
    )
    parser.add_argument(
        "--rebuild-oos-budget-s",
        type=float,
        default=180.0,
        help="Wall-clock budget for --rebuild-oos (default 180)",
    )
    args = parser.parse_args()
    if args.rebuild_oos:
        _rebuild_oos_cache(budget_s=float(args.rebuild_oos_budget_s))
    meta = publish(
        with_details=not bool(args.no_details),
        enrich_details=bool(args.enrich_details),
        details_budget_s=float(args.details_budget_s),
        max_details=int(args.max_details),
    )
    print(
        f"[publish-frontend-data] wrote {OUT} "
        f"runs={meta['n_runs']} oos={meta['n_oos_rows']} "
        f"details={meta.get('details_written', 0)} in {meta['elapsed_ms']}ms",
        flush=True,
    )
    if meta["n_runs"] == 0:
        print(
            "[publish-frontend-data] warning: no runs in execution/api_runs_cache.json",
            file=sys.stderr,
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
