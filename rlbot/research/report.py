"""Generate human-readable research tables from the JSONL run registry, so RESEARCH.md
stops being hand-maintained."""

from __future__ import annotations

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


def _render(by_variant: dict[str, list[Mapping]]) -> str:
    lines = [
        "| variant | tier | seeds | OOS return (med) | OOS Sharpe (med) | max DD (med) | split |",
        "|---|---|---|---|---|---|---|",
    ]
    for variant_id in sorted(by_variant):
        rows = by_variant[variant_id]
        tier = max(int(r.get("evaluation_tier", 0)) for r in rows)
        ret = _median([r.get("oos_total_return") for r in rows])
        sharpe = _median([r.get("oos_sharpe") for r in rows])
        dd = _median([r.get("oos_max_drawdown") for r in rows])
        split = rows[0].get("feature_split_mode") or "—"
        lines.append(
            f"| {variant_id} | {tier} | {len(rows)} | {_fmt(ret, pct=True)} | "
            f"{_fmt(sharpe)} | {_fmt(dd, pct=True)} | {split} |"
        )
    return "\n".join(lines)


def write_report(records: Iterable[Mapping], path: str | Path, *, title: str = "Research cohort") -> Path:
    records = list(records)
    body = [
        f"# {title}",
        "",
        f"{len(records)} run record(s). OOS metrics shown only for promoted (tier ≥ 4) runs; "
        "checkpoint rule = eval-NAV-best. Generated from registry.jsonl — do not edit by hand.",
        "",
        _render(_group(records)),
        "",
    ]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(body), encoding="utf-8")
    return out


def _group(records: Iterable[Mapping]) -> dict[str, list[Mapping]]:
    by_variant: dict[str, list[Mapping]] = {}
    for r in records:
        by_variant.setdefault(str(r.get("variant_id")), []).append(r)
    return by_variant
