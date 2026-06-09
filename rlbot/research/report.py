"""Generate human-readable research tables from the JSONL run registry, so RESEARCH.md
stops being hand-maintained."""

from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import Iterable, Mapping


def _fmt(x, pct: bool = False) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return f"{v * 100:.2f}%" if pct else f"{v:.2f}"


def _median(values: list) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return statistics.median(vals) if vals else None


# Statuses that mark an OOS read with no usable score (crash between read and scoring).
_UNSCORED_STATUSES = {"oos_read_attempt", "failed"}


def _is_scored(r: Mapping) -> bool:
    return str(r.get("status", "ok")) not in _UNSCORED_STATUSES


def _ci_cell(rows: list[Mapping]) -> str:
    cis = [r.get("oos_sharpe_ci") for r in rows if r.get("oos_sharpe_ci")]
    if not cis:
        return "—"
    lo = _median([c[0] for c in cis])
    hi = _median([c[2] for c in cis])
    return f"[{_fmt(lo)}, {_fmt(hi)}]"


def _render(by_group: dict[str, list[Mapping]]) -> str:
    lines = [
        "| variant (across seeds) | tier | seeds | best eval NAV (med) | OOS return (med) | OOS Sharpe (med) | Sharpe 95% CI | max DD (med) | split |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for group_id in sorted(by_group):
        rows = [r for r in by_group[group_id] if _is_scored(r)]
        if not rows:
            rows = by_group[group_id]  # only attempt/failure records: still list the group
            tier = max(int(r.get("evaluation_tier", 0)) for r in rows)
            lines.append(f"| {group_id} | {tier} | 0 | — | — | — | — | — | — |")
            continue
        # One run can hold several scored records (tier-3 launch + tier-4 promote).
        # Dedupe per run_id keeping the highest-tier record — otherwise the promoted
        # (best) seed enters every median twice, biasing best_eval_nav upward.
        by_run: dict[str, Mapping] = {}
        for i, r in enumerate(rows):
            rid = str(r.get("run_id") or f"__row{i}")  # legacy records without run_id
            if rid not in by_run or int(r.get("evaluation_tier", 0)) > int(
                by_run[rid].get("evaluation_tier", 0)
            ):
                by_run[rid] = r
        runs = list(by_run.values())
        tier = max(int(r.get("evaluation_tier", 0)) for r in runs)
        seeds = {r.get("seed") for r in runs if r.get("seed") is not None}
        n_seeds = len(seeds) if seeds else len(runs)
        eval_nav = _median([r.get("best_eval_nav") for r in runs])
        # The OOS firewall: only promoted (tier >= 4) scored records may surface holdout
        # metrics, regardless of what a hand-run backtest wrote into a low-tier summary.
        oos_rows = [r for r in runs if int(r.get("evaluation_tier", 0)) >= 4]
        ret = _median([r.get("oos_total_return") for r in oos_rows])
        sharpe = _median([r.get("oos_sharpe") for r in oos_rows])
        dd = _median([r.get("oos_max_drawdown") for r in oos_rows])
        splits = sorted({str(r.get("feature_split_mode")) for r in runs if r.get("feature_split_mode")})
        split = "/".join(splits) if splits else "—"
        oos_n = f" (n={len(oos_rows)})" if oos_rows and len(oos_rows) < len(runs) else ""
        lines.append(
            f"| {group_id} | {tier} | {n_seeds} | {_fmt(eval_nav)} | {_fmt(ret, pct=True)}{oos_n} | "
            f"{_fmt(sharpe)} | {_ci_cell(oos_rows)} | {_fmt(dd, pct=True)} | {split} |"
        )
    return "\n".join(lines)


def write_report(records: Iterable[Mapping], path: str | Path, *, title: str = "Research cohort") -> Path:
    records = list(records)
    by_group = _group(records)
    n_variants = len({str(r.get("variant_id")) for r in records})
    n_groups = len(by_group)
    # One holdout read = one "oos_read_attempt" record (written before the backtest).
    # Scored tier>=4 records without a matching attempt are legacy single-record reads.
    tier4 = [r for r in records if int(r.get("evaluation_tier", 0)) >= 4]
    attempts = [r for r in tier4 if str(r.get("status", "ok")) == "oos_read_attempt"]
    attempted_runs = {str(r.get("run_id")) for r in attempts}
    legacy_scored = [
        r for r in tier4
        if _is_scored(r) and str(r.get("run_id")) not in attempted_runs
    ]
    n_oos_reads = len(attempts) + len(legacy_scored)
    body = [
        f"# {title}",
        "",
        f"{len(records)} run record(s) across {n_variants} variant(s) "
        f"({n_groups} seed-group(s)); "
        f"{n_oos_reads} holdout read(s) recorded. OOS metrics shown only for promoted "
        "(tier ≥ 4) runs; checkpoint rule = eval-NAV-best. Any OOS number below was "
        f"selected from {n_variants} variant(s) — interpret with that multiplicity in mind. "
        "Generated from registry.jsonl — do not edit by hand.",
        "",
        _render(by_group),
        "",
    ]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(body), encoding="utf-8")
    return out


_SEED_PART = re.compile(r"__seed\d+")


def _group_key(r: Mapping) -> str:
    """Cross-seed aggregation key: explicit group_id, else variant_id minus its
    __seedN component (legacy records predate group_id)."""
    gid = r.get("group_id")
    if gid:
        return str(gid)
    return _SEED_PART.sub("", str(r.get("variant_id")))


def _group(records: Iterable[Mapping]) -> dict[str, list[Mapping]]:
    by_group: dict[str, list[Mapping]] = {}
    for r in records:
        by_group.setdefault(_group_key(r), []).append(r)
    return by_group
