"""Global holdout-burn ledger: one JSONL record per OOS read, across ALL cohorts.

The per-cohort registry firewall only sees reads that go through ``research.py``.
This ledger is written by ``scripts/backtest.py`` (every backtest read) and
``scripts/infer_weights.py`` (audited weight emissions over recent bars), so manual
backtests, cohort launches, promotes, and inference all land in one place. Reads
that bypass these CLIs (calling ``rollout_policy_on_slice`` directly) are NOT
captured — the CLIs are the supported surface. It is the source of truth for:

- the cumulative per-window read budget (``assert_window_budget``), and
- the trial count ``N`` for the deflated Sharpe ratio (selection-aware
  significance: "best of N models on this window").

Burn semantics: re-running the SAME run_id on the same window does not add
selection pressure (same model, same answer), so budgets and DSR trial counts use
**distinct run_ids per window**, not raw read counts. Raw reads are still recorded.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from rlbot.run_artifacts import RUNS_ROOT

from . import registry as _registry

# Per-window cumulative budget of DISTINCT models that may read a holdout window
# before further research-loop reads are refused. Deliberately small: every extra
# read deflates what any result on that window can claim.
DEFAULT_WINDOW_BUDGET = 10


def ledger_path(root: Path | None = None) -> Path:
    return (root or RUNS_ROOT) / "oos_ledger.jsonl"


def window_key(holdout_start: Any, holdout_end: Any) -> str:
    """Canonical key for a holdout window: 'YYYY-MM-DD..YYYY-MM-DD' (dates only —
    the calendar window is what gets burned, regardless of cache vintage)."""
    import pandas as pd

    s = pd.Timestamp(holdout_start).date() if holdout_start else "?"
    e = pd.Timestamp(holdout_end).date() if holdout_end else "?"
    return f"{s}..{e}"


def window_key_for_read(
    registered_start: Any,
    registered_end: Any,
    realized_start: Any,
    realized_end: Any,
) -> str:
    """The key a backtest must record under: the REGISTERED calendar window when the
    manifest carries one, else the realized trading-day span (tail-mode runs).

    Spec windows and manifests speak calendar dates ('2022-01-01'); panels speak
    trading days ('2022-01-03'). Keying reads on realized days while budgets check
    calendar dates would make every budget lookup miss — the two sides of the
    launch→backtest seam MUST share this function's output."""
    if registered_start and registered_end:
        return window_key(registered_start, registered_end)
    return window_key(realized_start, realized_end)


def record_oos_read(
    *,
    run_id: str,
    window: str,
    checkpoint: str = "",
    data_cache_hash: str | None = None,
    context: str = "",
    root: Path | None = None,
    enforce_budget: int | None = None,
) -> dict:
    """Append one read record. Called by backtest BEFORE the rollout starts, so a
    crash mid-rollout still counts as a burn (fail-closed accounting).

    With ``enforce_budget`` set (research-driven reads), the read is checked against
    the budget ATOMICALLY with the append under the ledger lock — closing the gap
    between a launch-time budget check and a read that happens hours later."""
    rec = {
        "window": window,
        "run_id": str(run_id),
        "checkpoint": checkpoint,
        "data_cache_hash": data_cache_hash,
        "context": context,  # e.g. 'research:cohort', 'infer_weights', or 'manual'
        "read_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = ledger_path(root)
    with _registry.registry_lock(path):
        if enforce_budget is not None:
            records = _registry.read_records(path, on_corrupt="raise")
            assert_window_budget(records, window, [str(run_id)], budget=enforce_budget)
        _registry.append_record(path, rec)
    return rec


def read_ledger(root: Path | None = None, *, on_corrupt: str = "skip") -> list[dict]:
    return _registry.read_records(ledger_path(root), on_corrupt=on_corrupt)


def distinct_models_for_window(
    records: list[Mapping], window: str
) -> set[str]:
    return {
        str(r.get("run_id"))
        for r in records
        if str(r.get("window")) == window and r.get("run_id")
    }


def window_burn_counts(records: list[Mapping]) -> dict[str, int]:
    """window key → number of distinct models that have read it."""
    out: dict[str, set[str]] = {}
    for r in records:
        w = str(r.get("window"))
        out.setdefault(w, set()).add(str(r.get("run_id")))
    return {w: len(s) for w, s in out.items()}


def assert_window_budget(
    records: list[Mapping],
    window: str,
    new_run_ids: list[str],
    budget: int = DEFAULT_WINDOW_BUDGET,
) -> None:
    """Refuse reads that would push a window's distinct-model count past ``budget``.

    Models that already read this window do not re-burn (re-scoring the same model
    is not new selection pressure)."""
    seen = distinct_models_for_window(records, window)
    new = [r for r in dict.fromkeys(new_run_ids) if str(r) not in seen]
    if len(seen) + len(new) > budget:
        raise PermissionError(
            f"OOS window {window} budget exhausted: {len(seen)} distinct model(s) have "
            f"already read it and this would add {len(new)} more (budget {budget}). "
            "Every additional read deflates what any result on this window can claim "
            "(see deflated Sharpe in docs/RESEARCH.md). Use in-training eval (tier ≤ 3) "
            "for iteration, or raise --window-budget explicitly if this is deliberate."
        )


def trials_for_window(
    window: str, root: Path | None = None, *, on_corrupt: str = "raise"
) -> int:
    """Ledger-derived trial count N for the deflated Sharpe ratio (≥ 1).

    Strict by default: a corrupt ledger line would UNDERCOUNT N exactly where it
    inflates a published significance number, so the DSR path fails closed too."""
    return max(
        1, len(distinct_models_for_window(read_ledger(root, on_corrupt=on_corrupt), window))
    )
