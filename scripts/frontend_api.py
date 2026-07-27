#!/usr/bin/env python3
"""Read-only local API for the MarketTrainer operator console (``frontend/``).

Serves real ``Runs/`` artifacts (audit records, manifests, backtest summaries,
cohort tables) and an engine-backed config preflight to the browser UI. It is a
local operator console API, not a multi-tenant product backend:

- Read-only over run artifacts; the only POST endpoint parses YAML with the
  real ``load_config`` parser and never executes anything.
- Binds to 127.0.0.1 by default. Optional shared-secret auth via
  ``MARKETTRAINER_API_TOKEN`` (send ``Authorization: Bearer <token>``).
- No filesystem paths are exposed to the browser — run ids are validated
  against the discovered run set.

Usage:
    pip install -e ".[api]"
    python scripts/frontend_api.py --port 8787
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path as _Path
from typing import Any

_bootstrap_path = _Path(__file__).resolve().parent / "_bootstrap.py"
_bootstrap_spec = importlib.util.spec_from_file_location("_rlbot_repo_bootstrap", _bootstrap_path)
assert _bootstrap_spec is not None and _bootstrap_spec.loader is not None
_bootstrap_mod = importlib.util.module_from_spec(_bootstrap_spec)
_bootstrap_spec.loader.exec_module(_bootstrap_mod)

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rlbot.curriculum_preflight import build_curriculum_preflight
from rlbot.rl_config import load_config, validate_config_for_universe
from rlbot.run_artifacts import (
    PROJECT_ROOT,
    RUNS_ROOT,
    RunPaths,
    read_run_manifest,
    resolve_backtest_summary_path,
)
from rlbot.run_audit import audit_runs, discover_audit_run_ids

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
_TICKER_RE = re.compile(r"^[A-Z0-9^][A-Z0-9.\-=^]{0,14}$")
_AUDIT_CACHE_TTL_SECONDS = 60.0
_MAX_PREFLIGHT_YAML_BYTES = 256 * 1024
_INSTRUMENT_LOOKUP_CACHE_TTL_SECONDS = 300.0
_YAHOO_USER_AGENT = "Mozilla/5.0 (compatible; MarketTrainer/1.0)"

app = FastAPI(title="MarketTrainer local API", version="0.1.0", docs_url="/docs")


def _require_token(request: Request) -> None:
    token = os.environ.get("MARKETTRAINER_API_TOKEN", "").strip()
    if not token:
        return
    supplied = request.headers.get("authorization", "")
    if supplied != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token")


class _AuditCache:
    def __init__(self) -> None:
        self._records: list[Any] | None = None
        self._by_id: dict[str, Any] = {}
        self._signatures: dict[str, tuple[tuple[int, int], ...]] = {}
        self._at = 0.0

    @staticmethod
    def _signature(run_id: str) -> tuple[tuple[int, int], ...]:
        paths = RunPaths(run_id=run_id, root=PROJECT_ROOT)
        watched = (
            paths.run_meta_dir / "manifest.json",
            paths.config_snapshot,
            paths.run_meta_dir / "backtest_summary.json",
            paths.run_meta_dir / "backtest_summary_final.json",
            paths.run_meta_dir / "backtest_summary_latest.json",
            paths.eval_nav_history,
            paths.eval_log_dir / "reward_decomp.jsonl",
            paths.eval_log_dir / "reward_decomp.json",
            paths.models_dir / "checkpoints",
        )
        signature: list[tuple[int, int]] = []
        for path in watched:
            try:
                stat = path.stat()
                signature.append((stat.st_mtime_ns, stat.st_size))
            except OSError:
                signature.append((0, 0))
        return tuple(signature)

    def get(self) -> list[Any]:
        now = time.monotonic()
        ids = discover_audit_run_ids(root=PROJECT_ROOT)
        id_set = set(ids)
        for stale_id in set(self._by_id) - id_set:
            self._by_id.pop(stale_id, None)
            self._signatures.pop(stale_id, None)

        # Always re-audit runs whose watched artifacts changed (manifest, backtest
        # summaries, checkpoints). Full rediscover also runs on TTL so new run dirs
        # appear promptly for the ops dashboard auto-refresh.
        need_full = self._records is None or (now - self._at) > _AUDIT_CACHE_TTL_SECONDS
        for run_id in ids:
            signature = self._signature(run_id)
            if (
                not need_full
                and self._signatures.get(run_id) == signature
                and run_id in self._by_id
            ):
                continue
            records = audit_runs([run_id], root=PROJECT_ROOT)
            if records:
                self._by_id[run_id] = records[0]
                self._signatures[run_id] = signature
            else:
                self._by_id.pop(run_id, None)
                self._signatures.pop(run_id, None)
        self._records = [self._by_id[run_id] for run_id in ids if run_id in self._by_id]
        if need_full:
            self._at = now
        return self._records

    def get_by_id(self, run_id: str) -> Any | None:
        self.get()
        return self._by_id.get(run_id)


_audit_cache = _AuditCache()


class _InstrumentLookupCache:
    def __init__(self) -> None:
        self._entries: dict[str, tuple[float, dict[str, Any]]] = {}

    def get(self, symbol: str) -> dict[str, Any] | None:
        entry = self._entries.get(symbol)
        if entry is None:
            return None
        cached_at, payload = entry
        if (time.monotonic() - cached_at) > _INSTRUMENT_LOOKUP_CACHE_TTL_SECONDS:
            self._entries.pop(symbol, None)
            return None
        return payload

    def set(self, symbol: str, payload: dict[str, Any]) -> None:
        self._entries[symbol] = (time.monotonic(), payload)


_instrument_cache = _InstrumentLookupCache()


def _quote_type_to_group(quote_type: str | None) -> str:
    mapping = {
        "EQUITY": "Equity",
        "ETF": "Equity",
        "MUTUALFUND": "Equity",
        "INDEX": "Equity",
        "CURRENCY": "FX",
        "CRYPTOCURRENCY": "Alternative",
        "FUTURE": "Commodity",
        "COMMODITY": "Commodity",
    }
    return mapping.get((quote_type or "").upper(), "Alternative")


def _instrument_record(
    *,
    found: bool,
    symbol: str,
    name: str,
    group: str,
    exchange: str | None = None,
    currency: str | None = None,
) -> dict[str, Any]:
    return {
        "found": found,
        "symbol": symbol,
        "name": name,
        "group": group,
        "exchange": exchange,
        "currency": currency,
    }


def _yahoo_json(url: str) -> dict[str, Any]:
    try:
        from curl_cffi import requests as curl_requests

        response = curl_requests.get(url, impersonate="chrome", timeout=15)
        if response.status_code != 200:
            raise urllib.error.HTTPError(url, response.status_code, "non-200", None, None)
        payload = response.json()
        return payload if isinstance(payload, dict) else {}
    except ImportError:
        request = urllib.request.Request(url, headers={"User-Agent": _YAHOO_USER_AGENT})
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload if isinstance(payload, dict) else {}


def _quote_row(record: dict[str, Any]) -> dict[str, Any]:
    symbol = str(record.get("symbol") or "").upper()
    return _instrument_record(
        found=True,
        symbol=symbol,
        name=str(record.get("longname") or record.get("shortname") or symbol),
        group=_quote_type_to_group(str(record.get("quoteType")) if record.get("quoteType") is not None else None),
        exchange=str(record.get("exchange") or record.get("exchDisp") or "") or None,
        currency=str(record.get("currency")) if record.get("currency") else None,
    )


def _lookup_yfinance(symbol: str) -> dict[str, Any]:
    sym = symbol.strip().upper()
    if not _TICKER_RE.match(sym):
        return _instrument_record(found=False, symbol=sym, name=sym, group="Alternative")

    cached = _instrument_cache.get(sym)
    if cached is not None:
        return cached

    params = urllib.parse.urlencode({"q": sym, "quotesCount": 8, "newsCount": 0})
    try:
        data = _yahoo_json(f"https://query2.finance.yahoo.com/v1/finance/search?{params}")
        quotes = [quote for quote in (data.get("quotes") or []) if isinstance(quote, dict)]
        match = next((quote for quote in quotes if str(quote.get("symbol", "")).upper() == sym), None)
        if match is None and len(quotes) == 1:
            match = quotes[0]
        payload = _quote_row(match) if match is not None else _instrument_record(
            found=False, symbol=sym, name=sym, group="Alternative"
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        payload = _instrument_record(found=False, symbol=sym, name=sym, group="Alternative")

    _instrument_cache.set(sym, payload)
    return payload


def _search_yfinance(query: str, limit: int) -> list[dict[str, Any]]:
    cleaned = query.strip()
    if not cleaned:
        return []

    if cleaned == cleaned.upper() and _TICKER_RE.match(cleaned):
        exact = _lookup_yfinance(cleaned)
        return [exact] if exact["found"] else []

    params = urllib.parse.urlencode({"q": cleaned, "quotesCount": limit, "newsCount": 0})
    try:
        data = _yahoo_json(f"https://query2.finance.yahoo.com/v1/finance/search?{params}")
        quotes = data.get("quotes") or []
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return []

    rows: list[dict[str, Any]] = []
    for quote in quotes:
        if not isinstance(quote, dict):
            continue
        symbol = quote.get("symbol")
        if not isinstance(symbol, str) or not symbol.strip():
            continue
        rows.append(_quote_row(quote))
        if len(rows) >= limit:
            break
    return rows


def _load_json(path: _Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _window_of(run_id: str) -> str | None:
    m = re.match(r"^(W\d+)_", run_id)
    return m.group(1) if m else None


def _progress_pct(elapsed: int | None, nominal: int | None) -> float | None:
    if not elapsed or not nominal or nominal <= 0:
        return None
    return round(min(100.0, 100.0 * elapsed / nominal), 1)


def _run_record(rec: Any) -> dict[str, Any]:
    d = rec.to_dict()
    return {
        "run_id": d["run_id"],
        "window": _window_of(d["run_id"]),
        "training_status": d.get("training_status"),
        "progress_pct": _progress_pct(d.get("elapsed_timesteps"), d.get("nominal_timesteps")),
        "elapsed_timesteps": d.get("elapsed_timesteps"),
        "nominal_timesteps": d.get("nominal_timesteps"),
        "best_eval_step": d.get("best_eval_step"),
        "best_eval_score": d.get("best_eval_score"),
        "curriculum_stage_at_best": d.get("curriculum_stage_at_best"),
        "early_stop_reason": d.get("early_stop_reason"),
        "started_at_utc": d.get("started_at_utc"),
        "finished_at_utc": d.get("finished_at_utc"),
        "oos_sharpe": d.get("oos_sharpe"),
        "oos_deflated_sharpe": d.get("oos_deflated_sharpe"),
        "oos_return": d.get("oos_return"),
        "oos_max_drawdown": d.get("oos_max_dd"),
        "ew_excess_return": d.get("ew_excess_return"),
        "has_backtest": d.get("oos_sharpe") is not None or d.get("oos_return") is not None,
        "labels": d.get("labels") or [],
        "warnings": d.get("warnings") or [],
        "comparable": d.get("comparable", True),
        "git_dirty": d.get("git_dirty"),
    }


def _run_rows() -> list[dict[str, Any]]:
    return [_run_record(record) for record in _audit_cache.get()]


def _summary_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row["training_status"] == "completed"]
    scored = [row for row in rows if isinstance(row["oos_sharpe"], (int, float))]
    comparable_scored = [row for row in scored if row["comparable"]]
    best = max(comparable_scored, key=lambda row: row["oos_sharpe"]) if comparable_scored else None
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_runs": len(rows),
        "completed_runs": len(completed),
        "active_runs": len(
            [row for row in rows if row["training_status"] not in ("completed", "interrupted")]
        ),
        "runs_with_backtest": len(scored),
        "best_oos": (
            {
                "run_id": best["run_id"],
                "sharpe": best["oos_sharpe"],
                "deflated_sharpe": best["oos_deflated_sharpe"],
                "window": best["window"],
            }
            if best
            else None
        ),
    }


def _run_order(row: dict[str, Any]) -> tuple[bool, str]:
    active = row.get("training_status") not in ("completed", "interrupted")
    timestamp = str(row.get("started_at_utc") or row.get("finished_at_utc") or "")
    return active, timestamp


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


_COHORT_RUN_RE = re.compile(r"^(W\d+)_(\d+)", re.IGNORECASE)


def _published_benchmark_index() -> dict[str, dict[str, Any]]:
    table = _load_json(RUNS_ROOT / "cohort_vs_benchmark.json") or {}
    rows = table.get("rows") if isinstance(table.get("rows"), list) else []
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("run_id"), str):
            out[str(row["run_id"])] = row
    return out


def _float_or_none(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _oos_row_from_backtest(run_id: str, backtest: dict[str, Any], published: dict[str, Any] | None) -> dict[str, Any] | None:
    match = _COHORT_RUN_RE.match(run_id)
    if match is None:
        return None
    model_ret = _float_or_none(backtest.get("total_return"))
    model_sh = _float_or_none(backtest.get("sharpe"))
    if model_ret is None or model_sh is None:
        return None

    detailed = backtest.get("detailed") if isinstance(backtest.get("detailed"), dict) else {}
    ew = detailed.get("benchmark_equal_weight_daily") or detailed.get("benchmark_equal_weight")
    spy = detailed.get("benchmark_spy")
    ew_ret = _float_or_none(ew.get("total_return")) if isinstance(ew, dict) else None
    ew_sh = _float_or_none(ew.get("sharpe")) if isinstance(ew, dict) else None
    spy_ret = _float_or_none(spy.get("total_return")) if isinstance(spy, dict) else None
    spy_sh = _float_or_none(spy.get("sharpe")) if isinstance(spy, dict) else None

    if ew_ret is None:
        ew_ret = _float_or_none(backtest.get("equal_weight_daily_return"))
    if published:
        if ew_ret is None:
            ew_ret = _float_or_none(published.get("ew_ret"))
        if ew_sh is None:
            ew_sh = _float_or_none(published.get("ew_sh"))
        if spy_ret is None:
            spy_ret = _float_or_none(published.get("spy_ret"))
        if spy_sh is None:
            spy_sh = _float_or_none(published.get("spy_sh"))

    return {
        "run_id": run_id,
        "cohort": match.group(2),
        "window": match.group(1).upper(),
        "model_ret": model_ret,
        "model_sh": model_sh,
        "ew_ret": ew_ret,
        "ew_sh": ew_sh,
        "spy_ret": spy_ret,
        "spy_sh": spy_sh,
        "has_benchmarks": ew_ret is not None and spy_ret is not None,
    }


def _oos_result_rows() -> list[dict[str, Any]]:
    """Build OOS comparison rows from every local backtest summary.

    Prefer live backtest summaries (canonical ``backtest_summary.json``, else
    final/latest). Fall back to ``cohort_vs_benchmark.json`` only for missing
    benchmark fields on older summaries.
    """
    published = _published_benchmark_index()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    if not RUNS_ROOT.is_dir():
        return []
    for path in sorted(RUNS_ROOT.iterdir()):
        if not path.is_dir():
            continue
        run_id = path.name
        if not _RUN_ID_RE.match(run_id):
            continue
        summary_path = resolve_backtest_summary_path(path)
        summary = _load_json(summary_path) if summary_path is not None else None
        if not isinstance(summary, dict):
            continue
        row = _oos_row_from_backtest(run_id, summary, published.get(run_id))
        if row is None:
            continue
        rows.append(row)
        seen.add(run_id)
    # Keep published-only rows if a summary disappeared but the table remains.
    for run_id, published_row in published.items():
        if run_id in seen:
            continue
        match = _COHORT_RUN_RE.match(run_id)
        if match is None:
            continue
        model_ret = _float_or_none(published_row.get("model_ret"))
        model_sh = _float_or_none(published_row.get("model_sh"))
        if model_ret is None or model_sh is None:
            continue
        rows.append(
            {
                "run_id": run_id,
                "cohort": match.group(2),
                "window": match.group(1).upper(),
                "model_ret": model_ret,
                "model_sh": model_sh,
                "ew_ret": _float_or_none(published_row.get("ew_ret")),
                "ew_sh": _float_or_none(published_row.get("ew_sh")),
                "spy_ret": _float_or_none(published_row.get("spy_ret")),
                "spy_sh": _float_or_none(published_row.get("spy_sh")),
                "has_benchmarks": True,
            }
        )
    rows.sort(key=lambda row: (row["cohort"], row["window"], row["run_id"]))
    return rows


def _cohort_sort_key(cohort: str) -> tuple[int, str]:
    return (int(cohort), cohort) if cohort.isdigit() else (10**9, cohort)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "runs_root_exists": RUNS_ROOT.is_dir(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "auth_required": bool(os.environ.get("MARKETTRAINER_API_TOKEN", "").strip()),
        # Bumped when /api/results aggregation contract changes; used by frontend/scripts/dev.mjs
        "oos_aggregation": "backtest_summaries",
    }


@app.get("/api/summary", dependencies=[Depends(_require_token)])
def summary() -> dict[str, Any]:
    return _summary_from_rows(_run_rows())


@app.get("/api/runs", dependencies=[Depends(_require_token)])
def runs(
    prefix: str = "",
    search: str = Query(default="", max_length=80),
    status: str = Query(default="", pattern=r"^(|completed|active|interrupted)$"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    rows = _run_rows()
    counts = {
        "all": len(rows),
        "completed": sum(row["training_status"] == "completed" for row in rows),
        "active": sum(
            row["training_status"] not in ("completed", "interrupted") for row in rows
        ),
        "interrupted": sum(row["training_status"] == "interrupted" for row in rows),
        "with_backtest": sum(bool(row["has_backtest"]) for row in rows),
    }
    if prefix:
        rows = [row for row in rows if row["run_id"].startswith(prefix)]
    if search:
        needle = search.casefold()
        rows = [row for row in rows if needle in row["run_id"].casefold()]
    if status == "completed":
        rows = [row for row in rows if row["training_status"] == "completed"]
    elif status == "interrupted":
        rows = [row for row in rows if row["training_status"] == "interrupted"]
    elif status == "active":
        rows = [
            row for row in rows if row["training_status"] not in ("completed", "interrupted")
        ]
    rows.sort(key=_run_order, reverse=True)
    total = len(rows)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": rows[offset : offset + limit],
        "total": total,
        "offset": offset,
        "limit": limit,
        "counts": counts,
    }


@app.get("/api/dashboard", dependencies=[Depends(_require_token)])
def dashboard() -> dict[str, Any]:
    """One narrow startup payload for the operator dashboard."""
    rows = _run_rows()
    rows.sort(key=_run_order, reverse=True)
    result_rows = _oos_result_rows()
    by_window: dict[str, list[float]] = {}
    for row in result_rows:
        window = row.get("window")
        sharpe = row.get("model_sh")
        if isinstance(window, str) and isinstance(sharpe, (int, float)):
            by_window.setdefault(window, []).append(float(sharpe))
    window_sharpes = [
        {"window": window, "sharpe": round(_median(values), 2)}
        for window, values in sorted(by_window.items())
        if values
    ]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": _summary_from_rows(rows),
        "recent_runs": rows[:6],
        "window_sharpes": window_sharpes,
    }


@app.get("/api/runs/{run_id}", dependencies=[Depends(_require_token)])
def run_detail(run_id: str) -> dict[str, Any]:
    if not _RUN_ID_RE.match(run_id):
        raise HTTPException(status_code=400, detail="Invalid run id")
    rec = _audit_cache.get_by_id(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Unknown run id")

    manifest = read_run_manifest(run_id) or {}
    paths = RunPaths(run_id=run_id, root=PROJECT_ROOT)
    summary_path = resolve_backtest_summary_path(paths)
    backtest = _load_json(summary_path) if summary_path is not None else None

    audit_row = _run_record(rec)

    detail: dict[str, Any] = {
        "run_id": run_id,
        "audit": audit_row,
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
    if backtest:
        detail["backtest"] = {
            "checkpoint_label": backtest.get("checkpoint_label"),
            "oos_window": backtest.get("oos_window"),
            "total_return": backtest.get("total_return"),
            "sharpe": backtest.get("sharpe"),
            "excess_sharpe": backtest.get("excess_sharpe"),
            "max_drawdown": backtest.get("max_drawdown"),
            "deflated_sharpe": backtest.get("deflated_sharpe"),
            "deflated_sharpe_excess": backtest.get("deflated_sharpe_excess"),
            "oos_trials_for_window": backtest.get("oos_trials_for_window"),
            "oos_trials_conservative": backtest.get("oos_trials_conservative"),
            "equal_weight_daily_return": backtest.get("equal_weight_daily_return"),
            "excess_return_vs_equal_weight": backtest.get("excess_return_vs_equal_weight"),
            "hash_drift": backtest.get("hash_drift"),
            "n_bars": backtest.get("n_bars"),
            "portfolio_diagnostics": backtest.get("portfolio_diagnostics"),
        }
    return detail


@app.get("/api/results", dependencies=[Depends(_require_token)])
def results(cohort: str = "") -> dict[str, Any]:
    """OOS comparison rows across every local backtest, filterable by cohort.

    Built from ``Runs/*/backtest_summary.json``. ``cohort_vs_benchmark.json`` is
    only used to fill missing EW/SPY fields on older summaries.
    """
    rows = _oos_result_rows()
    cohorts = sorted({str(row["cohort"]) for row in rows}, key=_cohort_sort_key)
    if cohort:
        rows = [row for row in rows if str(row.get("cohort")) == cohort]
    with_benchmarks = sum(1 for row in rows if row.get("has_benchmarks"))
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "available": bool(rows) or bool(cohorts),
        "cohorts": cohorts,
        "rows": rows,
        "coverage": {
            "source": "Runs/*/backtest_summary*.json (best preferred; final/latest fallback)",
            "published_rows": len(rows),
            "published_runs": len({str(row["run_id"]) for row in rows}),
            "runs_with_backtest": len(rows),
            "runs_with_benchmarks": with_benchmarks,
            "total_runs": len(_run_rows()),
        },
    }


class PreflightRequest(BaseModel):
    yaml_text: str = Field(..., description="Full config.yaml contents to validate")


@app.post("/api/preflight", dependencies=[Depends(_require_token)])
def preflight(body: PreflightRequest) -> dict[str, Any]:
    raw = body.yaml_text
    if len(raw.encode("utf-8")) > _MAX_PREFLIGHT_YAML_BYTES:
        raise HTTPException(status_code=413, detail="Config too large")

    errors: list[str] = []
    warnings: list[str] = []
    n_assets: int | None = None
    milestones: dict[str, Any] | None = None

    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    try:
        tmp.write(raw)
        tmp.close()
        try:
            cfg = load_config(tmp.name)
            n_assets = len(cfg.universe.assets)
            validate_config_for_universe(cfg, n_assets)
            pf = build_curriculum_preflight(cfg)
            milestones = pf.to_dict()
            warnings.extend(str(w) for w in pf.warnings)
            if not pf.early_stop_reachable and int(cfg.training.early_stop_patience) > 0:
                warnings.append("Early stop is unreachable under this curriculum schedule.")
        except Exception as exc:  # noqa: BLE001 — surface parser/validator message verbatim
            errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        os.unlink(tmp.name)

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "n_assets": n_assets,
        "milestones": milestones,
        "validated_with": "rlbot.rl_config.load_config + validate_config_for_universe + curriculum preflight",
    }


@app.get("/api/instruments/lookup", dependencies=[Depends(_require_token)])
def instrument_lookup(symbol: str = Query(..., min_length=1, max_length=20)) -> dict[str, Any]:
    return _lookup_yfinance(symbol)


@app.get("/api/instruments/search", dependencies=[Depends(_require_token)])
def instrument_search(
    q: str = Query(..., min_length=1, max_length=40),
    limit: int = Query(default=8, ge=1, le=20),
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "query": q,
        "results": _search_yfinance(q, limit),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--cors-origin",
        action="append",
        default=None,
        help="Allowed browser origin (repeatable). Default: http://localhost:5173",
    )
    args = parser.parse_args()

    origins = args.cors_origin or [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["authorization", "content-type"],
    )

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
