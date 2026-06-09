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


def dedupe_scored_by_run(rows: Iterable[Mapping]) -> list[Mapping]:
    """One record per run_id, keeping the highest tier.

    A variant launched at tier 3 and later promoted holds TWO scored records for
    the same run; any median over raw records double-counts the promoted (best)
    run. This bug was reintroduced three times in three aggregations — every
    consumer must go through this helper."""
    by_run: dict[str, Mapping] = {}
    for i, r in enumerate(rows):
        rid = str(r.get("run_id") or f"__row{i}")
        if rid not in by_run or int(r.get("evaluation_tier", 0)) > int(
            by_run[rid].get("evaluation_tier", 0)
        ):
            by_run[rid] = r
    return list(by_run.values())


def count_oos_reads(records: Iterable[Mapping]) -> int:
    """Holdout reads = attempt records + legacy scored tier>=4 records that predate
    attempt-writing. Shared by the per-cohort and global reports (they must agree)."""
    records = list(records)
    tier4 = [r for r in records if int(r.get("evaluation_tier", 0)) >= 4]
    attempts = [r for r in tier4 if str(r.get("status", "ok")) == "oos_read_attempt"]
    attempted_runs = {str(r.get("run_id")) for r in attempts}
    legacy_scored = [
        r for r in tier4
        if _is_scored(r) and str(r.get("run_id")) not in attempted_runs
    ]
    return len(attempts) + len(legacy_scored)


def _md_cell(text: object, limit: int = 60) -> str:
    """Sanitize free text for a markdown table cell (hypotheses come from YAML)."""
    out = str(text).replace("\n", " ").replace("|", "\\|").strip()
    return out[: limit - 1] + "…" if len(out) > limit else out


def _norm_value(value: object) -> str:
    """Canonical display for a patched value: 1000 and 1000.0 are one cell."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):g}"
    return str(value)


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
        runs = dedupe_scored_by_run(rows)
        # Tier-1 rows are smoke/screen evidence (tiny budgets, e.g. `research.py
        # screen` runs sharing this group_id) — they never feed decision metrics
        # unless they are ALL the group has.
        decision = [r for r in runs if int(r.get("evaluation_tier", 0)) >= 2] or runs
        runs = decision
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
    n_oos_reads = count_oos_reads(records)
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


# ── Cross-cohort memory: global aggregation + knob sensitivity ────────────


def knob_sensitivity(records: Iterable[Mapping]) -> dict[str, list[dict]]:
    """Per patched config key: how does eval NAV move with the knob's value?

    Groups scored records by (cohort, key, value) using the concrete ``patch``
    carried on each record, and reports the median best_eval_nav per value plus
    its delta vs the cohort-wide median (a cheap normalization so cohorts trained
    under different budgets/windows can sit in one table). This is the input a
    hypothesis proposer reads: which knobs have moved the needle, where, and by
    how much. Tier-1 (smoke/screen) rows are excluded.
    """
    scored = [
        r
        for r in dedupe_scored_by_run(r for r in records if _is_scored(r))
        if int(r.get("evaluation_tier", 0)) >= 2 and r.get("best_eval_nav") is not None
    ]
    cohort_navs: dict[str, list[float]] = {}
    for r in scored:
        cohort_navs.setdefault(str(r.get("cohort")), []).append(float(r["best_eval_nav"]))
    cohort_median = {c: statistics.median(v) for c, v in cohort_navs.items()}

    cells: dict[tuple[str, str, str], list[float]] = {}
    for r in scored:
        patch = r.get("patch") or {}
        if not isinstance(patch, Mapping):
            continue
        for key, value in patch.items():
            cells.setdefault(
                (str(key), str(r.get("cohort")), _norm_value(value)), []
            ).append(float(r["best_eval_nav"]))

    # values-per-(key,cohort): a single value means no within-cohort contrast — its
    # delta is structurally 0 and must not read as "knob has no effect".
    n_values: dict[tuple[str, str], set[str]] = {}
    for key, cohort, value in cells:
        n_values.setdefault((key, cohort), set()).add(value)

    def _sort_key(item):
        (key, cohort, value), _ = item
        try:
            return (key, cohort, 0, float(value), "")
        except ValueError:
            return (key, cohort, 1, 0.0, value)

    out: dict[str, list[dict]] = {}
    for (key, cohort, value), navs in sorted(cells.items(), key=_sort_key):
        med = statistics.median(navs)
        has_contrast = len(n_values[(key, cohort)]) > 1
        out.setdefault(key, []).append(
            {
                "cohort": cohort,
                "value": value,
                "n_runs": len(navs),
                "median_best_eval_nav": med,
                "delta_vs_cohort_median": (
                    med - cohort_median.get(cohort, med) if has_contrast else None
                ),
            }
        )
    return out


def write_global_report(
    cohorts: Mapping[str, list[Mapping]],
    cohort_meta: Mapping[str, Mapping],
    path: str | Path,
) -> Path:
    """All-cohort view: per-cohort summary with parent lineage + the cross-cohort
    knob-sensitivity table. ``cohorts`` maps cohort id → its registry records;
    ``cohort_meta`` maps cohort id → its cohort.json dict (may be empty)."""
    lines = [
        "# Research memory — all cohorts",
        "",
        "Generated from Runs/*/registry.jsonl — do not edit by hand.",
        "",
        "| cohort | parent | hypothesis | records | groups | tier(max) | OOS reads |",
        "|---|---|---|---|---|---|---|",
    ]
    all_records: list[Mapping] = []
    for cohort in sorted(cohorts):
        records = cohorts[cohort]
        all_records.extend(records)
        meta = cohort_meta.get(cohort) or {}
        groups = _group(records)
        tiers = [int(r.get("evaluation_tier", 0)) for r in records] or [0]
        oos_reads = count_oos_reads(records)  # same counting as the per-cohort report
        hyp = _md_cell(meta.get("hypothesis") or next(
            (r.get("hypothesis") for r in records if r.get("hypothesis")), ""
        ))
        lines.append(
            f"| {_md_cell(cohort)} | {_md_cell(meta.get('parent') or '—')} | {hyp} | "
            f"{len(records)} | {len(groups)} | {max(tiers)} | {oos_reads} |"
        )

    lines += ["", "## Knob sensitivity (median best_eval_nav by patched value)", ""]
    sens = knob_sensitivity(all_records)
    if not sens:
        lines.append("_No scored records with patches yet._")
    else:
        lines += [
            "| config key | cohort | value | runs | median NAV | Δ vs cohort median |",
            "|---|---|---|---|---|---|",
        ]
        for key in sorted(sens):
            for cell in sens[key]:
                delta = cell["delta_vs_cohort_median"]
                delta_cell = f"{delta:+.2f}" if delta is not None else "— (no contrast)"
                lines.append(
                    f"| {_md_cell(key)} | {_md_cell(cell['cohort'])} | "
                    f"{_md_cell(cell['value'])} | {cell['n_runs']} | "
                    f"{_fmt(cell['median_best_eval_nav'])} | {delta_cell} |"
                )
    lines.append("")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
