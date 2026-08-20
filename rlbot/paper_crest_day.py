"""Paper / forward companion for CrestDay (locked pack, read-only).

Writes ``execution/shadow_ledger_CREST_DAY.jsonl`` + a forward mark so
``/ops/forward`` can show CrestDay NAV beside GeneralEquity1. Soft Yahoo MTM on
the equity book still attaches this series via ``pack_crestday.simulate_nav_series``.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rlbot.forward_mark import (
    build_forward_mark_payload,
    load_forward_mark,
    set_active_forward_run,
    write_forward_mark,
)
from rlbot.pack_crestday import (
    DEFAULT_INITIAL_CASH,
    PAPER_RUN_ID,
    STRATEGY_ID,
    latest_intents,
    simulate_nav_series,
)
from rlbot.run_artifacts import PROJECT_ROOT

EXECUTION_DIR = PROJECT_ROOT / "execution"
PAPER_DIR = EXECUTION_DIR / "paper_crest_day"
STATE_PATH = PAPER_DIR / "state.json"
ORDERS_PATH = PAPER_DIR / "order_intents.jsonl"
BAR_INTERVAL = "1h"  # pack panel bar size


def ledger_path(run_id: str = PAPER_RUN_ID) -> Path:
    return EXECUTION_DIR / f"shadow_ledger_{run_id}.jsonl"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _append_jsonl(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def default_state(initial_cash: float = DEFAULT_INITIAL_CASH) -> dict[str, Any]:
    return {
        "strategy_id": STRATEGY_ID,
        "run_id": PAPER_RUN_ID,
        "equity": float(initial_cash),
        "cash": float(initial_cash),
        "peak_equity": float(initial_cash),
        "positions": {},
        "target_weights": {"CASH": 1.0},
        "last_signal_date": None,
        "last_trade_date": None,
        "updated_at_utc": None,
        "book_start": None,
    }


def load_state(initial_cash: float = DEFAULT_INITIAL_CASH) -> dict[str, Any]:
    st = default_state(initial_cash)
    st.update(_read_json(STATE_PATH))
    st.setdefault("positions", {})
    st.setdefault("target_weights", {"CASH": 1.0})
    return st


def save_state(state: dict[str, Any]) -> None:
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_json(STATE_PATH, state)


def _weights_from_intents(intents_payload: dict[str, Any]) -> dict[str, float]:
    intents = intents_payload.get("intents") or []
    if not intents:
        return {"CASH": 1.0}
    # Notional risk fractions → gross weights; residual cash.
    by_sym: dict[str, float] = {}
    for it in intents:
        sym = str(it.get("symbol") or it.get("asset") or "").upper()
        if not sym:
            continue
        side = float(it.get("side") or it.get("dir") or 1.0)
        frac = abs(float(it.get("risk_frac") or it.get("size") or 0.0))
        if frac <= 0:
            continue
        by_sym[sym] = by_sym.get(sym, 0.0) + frac * (1.0 if side >= 0 else -1.0)
    gross = sum(abs(v) for v in by_sym.values())
    if gross <= 1e-12:
        return {"CASH": 1.0}
    # Long-only display weights for the chart (shorts → cash).
    tw = {k: max(0.0, v) / max(gross, 1e-12) * min(gross, 1.0) for k, v in by_sym.items()}
    tw = {k: v for k, v in tw.items() if v > 1e-8}
    tw["CASH"] = max(0.0, 1.0 - sum(tw.values()))
    return tw


def build_tip_forward_mark(
    *,
    book_start: str,
    weights: dict[str, float],
    intents_meta: dict[str, Any],
    initial_cash: float,
) -> dict[str, Any]:
    """Single-bar today@$initial_cash tip — live equity forward overlays CrestDay NAV."""
    ts = pd.Timestamp(f"{book_start} 09:30:00")
    flat = np.asarray([float(initial_cash)], dtype=np.float64)
    labels = ["Cash"] + [
        k for k in weights if str(k).upper() != "CASH" and float(weights[k]) > 0
    ]
    if len(labels) == 1:
        labels = ["Cash"]
        w_row = np.asarray([1.0], dtype=np.float64)
    else:
        w_row = np.asarray(
            [float(weights.get("CASH", 0.0))]
            + [float(weights.get(lab, 0.0)) for lab in labels[1:]],
            dtype=np.float64,
        )
        s = float(w_row.sum())
        w_row = w_row / s if s > 1e-12 else np.asarray([1.0] + [0.0] * (len(labels) - 1))
    payload = build_forward_mark_payload(
        run_id=PAPER_RUN_ID,
        checkpoint_label="locked",
        dates=[ts.date()],
        nav_model=flat,
        nav_spy=flat,
        nav_ew=flat,
        weights=np.asarray([w_row], dtype=np.float64),
        asset_labels=labels,
        initial_cash=float(initial_cash),
        holdout_start=str(book_start),
        holdout_end=str(book_start),
        note=(
            f"{STRATEGY_ID} (CrestDay pack): tip mark @ ${initial_cash:,.0f} on {book_start}. "
            f"asof={intents_meta.get('asof')} intents={len(intents_meta.get('intents') or [])}."
        ),
        bar_interval="5m",
        timestamps=[ts.isoformat(timespec="minutes")],
        bars_per_year=78 * 252,
    )
    payload["book_start"] = str(book_start)
    payload["companion_crypto_run_id"] = PAPER_RUN_ID
    payload["strategy_id"] = STRATEGY_ID
    return payload


def run_paper_day(
    *,
    as_of: date | None = None,
    force_refresh: bool = False,
    set_active: bool = False,
    initial_cash: float = DEFAULT_INITIAL_CASH,
    dry_run: bool = False,
) -> dict[str, Any]:
    del as_of
    state = load_state(initial_cash=initial_cash)
    if not state.get("book_start"):
        state["book_start"] = str(date.today())
        state["equity"] = float(initial_cash)
        state["cash"] = float(initial_cash)

    book_start = str(state["book_start"])
    # Warm pack NAV cache for soft companion overlays (not written as chart history).
    try:
        simulate_nav_series(
            force_refresh=bool(force_refresh),
            initial_cash=float(initial_cash),
            since=book_start,
        )
    except Exception:  # noqa: BLE001
        pass

    intents = latest_intents(aum=float(initial_cash), equity=float(initial_cash))
    weights = _weights_from_intents(intents)
    asof = str(intents.get("asof") or "")
    already = bool(asof) and str(state.get("last_signal_date") or "") == asof
    tip_nav = float(initial_cash)
    state["equity"] = tip_nav
    state["cash"] = tip_nav * float(weights.get("CASH", 1.0))
    state["target_weights"] = weights
    state["last_signal_date"] = asof or str(state.get("last_signal_date") or "")
    state["positions"] = {
        k: float(v) for k, v in weights.items() if str(k).upper() != "CASH" and float(v) > 0
    }

    actions = ["crestday_pack"]
    if already:
        actions.append("already_logged")

    if not dry_run:
        save_state(state)
        if not already:
            _append_jsonl(
                ledger_path(),
                {
                    "run_id": PAPER_RUN_ID,
                    "strategy_id": STRATEGY_ID,
                    "decision_bar": intents.get("asof"),
                    "trade_date": str(intents.get("asof") or "")[:10],
                    "as_of": intents.get("asof"),
                    "target_weights": {
                        ("Cash" if k.upper() == "CASH" else k): float(v)
                        for k, v in weights.items()
                    },
                    "sleeve_meta": {
                        "n_intents": len(intents.get("intents") or []),
                        "venue": intents.get("venue"),
                        "note": intents.get("note"),
                    },
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                    "bar_interval": "5m",
                    "pack": "CrestDay",
                },
            )
            _append_jsonl(
                ORDERS_PATH,
                {"as_of": intents.get("asof"), "intents": intents.get("intents")},
            )
        existing = load_forward_mark(PAPER_RUN_ID)
        n_existing = int((existing or {}).get("n_bars") or 0)
        # Do not clobber a live 5m overlay with a single-bar tip mark.
        if n_existing <= 1:
            payload = build_tip_forward_mark(
                book_start=book_start,
                weights=weights,
                intents_meta=intents,
                initial_cash=float(initial_cash),
            )
            write_forward_mark(payload)
        if set_active:
            set_active_forward_run(PAPER_RUN_ID)

    return {
        "strategy_id": STRATEGY_ID,
        "run_id": PAPER_RUN_ID,
        "as_of": intents.get("asof"),
        "actions": actions,
        "target_weights": weights,
        "equity": tip_nav,
        "n_bars": 1,
        "n_intents": len(intents.get("intents") or []),
        "dry_run": bool(dry_run),
        "set_active": bool(set_active),
        "initial_cash": float(initial_cash),
        "book_start": book_start,
        "pack": "CrestDay",
    }
