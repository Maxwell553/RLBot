"""Headless forward collector: 5m MTM + paper/shadow logs without the UI.

The ops dashboard used to be the only Yahoo / paper trigger (``/api/forward``
polls). This loop writes the same ``execution/`` caches on a timer so charts
and ledgers keep moving when the browser is closed.

Per-strategy caches:
  - GeneralEquity1 — ``execution/paper_general_equity1/`` + ``shadow_ledger_GENERAL_EQUITY1.jsonl``
  - CoreEquity — ``execution/paper_core_equity/`` + ``shadow_ledger_CORE_EQUITY.jsonl``
  - CrestDay — ``execution/paper_crest_day/`` + ``shadow_ledger_CREST_DAY.jsonl``
  - RLModel — ``execution/shadow_ledger_RLModel.jsonl`` (daily after the cash
    close; immediately if the ledger is still a cash-reset stub)
  - Shared 5m book — ``execution/forward_prices_5m_*.npz`` + ``forward_mark_*.json``
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from rlbot.forward_live import (
    ALGO_LIVE_RUN_ID,
    CORE_EQUITY_LIVE_RUN_ID,
    CRYPTO_LIVE_RUN_ID,
    DEFAULT_MIN_REFRESH_SECONDS,
    RL_LIVE_RUN_ID,
    canonical_forward_run_id,
    refresh_forward_mark_live,
)
from rlbot.forward_mark import (
    call_with_timeout,
    load_forward_mark,
    resolve_active_forward_run_id,
    write_forward_mark,
)
from rlbot.run_artifacts import PROJECT_ROOT

DEFAULT_INTERVAL_S = int(DEFAULT_MIN_REFRESH_SECONDS)
STATUS_NAME = "forward_loop_status.json"
LOCK_NAME = "forward_loop.lock"
PUBLIC_FORWARD = PROJECT_ROOT / "frontend" / "public" / "data" / "forward.json"
# RL shadow needs a settled daily bar; match the old 18:15 ET LaunchAgent.
RL_SHADOW_AFTER_MINUTES = 18 * 60
_FLAT_LEDGER_NOTE_MARKERS = ("reset to 100k", "flat paper book")


def _exec_dir(root: Path | None = None) -> Path:
    return (root or PROJECT_ROOT) / "execution"


def _status_path(root: Path | None = None) -> Path:
    return _exec_dir(root) / STATUS_NAME


def _lock_path(root: Path | None = None) -> Path:
    return _exec_dir(root) / LOCK_NAME


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    text = json.dumps(payload, indent=2, default=str)
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def read_status(root: Path | None = None) -> dict[str, Any]:
    return _read_json(_status_path(root))


def write_status(payload: dict[str, Any], root: Path | None = None) -> Path:
    path = _status_path(root)
    _atomic_write_json(path, payload)
    return path


def now_et() -> pd.Timestamp:
    return pd.Timestamp.now(tz="America/New_York")


def session_date(now: pd.Timestamp | None = None) -> str:
    ts = now if now is not None else now_et()
    if ts.tzinfo is not None:
        ts = ts.tz_convert("America/New_York")
    return str(ts.date())


def paper_day_due(last_date: str | None, *, session: str) -> bool:
    return str(last_date or "") != str(session)


def last_shadow_record(path: Path) -> dict[str, Any] | None:
    """Last JSON object in a shadow ledger, or ``None`` if missing/empty."""
    if not path.is_file():
        return None
    last: dict[str, Any] | None = None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if isinstance(rec, dict):
                last = rec
    except (OSError, json.JSONDecodeError):
        return last
    return last


def shadow_ledger_needs_reopen(path: Path) -> bool:
    """True when RLModel has never recorded, or the last row is a cash-reset stub.

    A real high-cash policy (no reset note) is left alone so we do not re-infer
    every 5 minutes. Missing ledgers and explicit 100k flats must re-open.
    """
    rec = last_shadow_record(path)
    if rec is None:
        return True
    note = str(rec.get("note") or "").lower()
    if any(m in note for m in _FLAT_LEDGER_NOTE_MARKERS):
        return True
    tw = rec.get("target_weights")
    if not isinstance(tw, dict) or not tw:
        return True
    risky = 0.0
    for key, val in tw.items():
        if str(key).strip().upper() in {"CASH", "USD"}:
            continue
        try:
            risky += abs(float(val))
        except (TypeError, ValueError):
            continue
    return risky < 1e-6 and ("reset" in note or "flat" in note)


def _shadow_ledger_path(run_id: str, root: Path | None = None) -> Path:
    return _exec_dir(root) / f"shadow_ledger_{run_id}.jsonl"


def rl_shadow_due(
    now: pd.Timestamp,
    last_shadow_date: str | None,
    *,
    after_minutes: int = RL_SHADOW_AFTER_MINUTES,
    force: bool = False,
) -> bool:
    """True once per weekday session after ``after_minutes`` Eastern.

    ``force`` (cash-reset / empty ledger) ignores the clock and today's
    ``last_shadow_date`` so a flattened paper book reopens before 18:00.
    """
    ts = pd.Timestamp(now)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("America/New_York")
    if int(ts.dayofweek) >= 5:
        return False
    if force:
        return True
    if str(last_shadow_date or "") == str(ts.date()):
        return False
    minutes = int(ts.hour) * 60 + int(ts.minute)
    return minutes >= int(after_minutes)


class LoopLock:
    """Exclusive flock so launchd + lite-API + CLI cannot double-collect."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd: int | None = None

    def acquire(self, *, blocking: bool = False) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        flags = fcntl.LOCK_EX if blocking else (fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            fcntl.flock(fd, flags)
        except BlockingIOError:
            os.close(fd)
            return False
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode("ascii"))
        self._fd = fd
        return True

    def release(self) -> None:
        fd = self._fd
        if fd is None:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass
        self._fd = None

    def __enter__(self) -> "LoopLock":
        if not self.acquire(blocking=False):
            raise BlockingIOError("forward loop already running")
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def publish_public_forward(run_id: str, mark: dict[str, Any]) -> Path | None:
    """Keep Vite ``/data/forward.json`` in sync with the execution mark."""
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "available": True,
        "run_id": run_id,
        "mark": mark,
        "message": None,
    }
    try:
        _atomic_write_json(PUBLIC_FORWARD, payload)
        return PUBLIC_FORWARD
    except OSError:
        return None


def _stamp_collector(mark: dict[str, Any], *, interval_s: int) -> dict[str, Any]:
    out = dict(mark)
    live = dict(out.get("live") or {})
    live["collector"] = {
        "running": True,
        "last_tick_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "interval_s": int(interval_s),
    }
    out["live"] = live
    return out


def _soft_paper_ge1(*, session: str, last_date: str | None) -> dict[str, Any]:
    if not paper_day_due(last_date, session=session):
        return {"ok": True, "skipped": "already_today", "as_of": last_date}
    try:
        from rlbot.pack_general_equity1 import PACK_DIR
        from rlbot.paper_prod_return_alpha import run_paper_day

        if not PACK_DIR.is_dir():
            return {"ok": True, "skipped": "pack_missing"}
        result = call_with_timeout(
            run_paper_day, 180.0, set_active=False, force_refresh=False
        )
        return {
            "ok": True,
            "as_of": result.get("as_of"),
            "actions": result.get("actions"),
            "n_orders": len(result.get("orders") or []),
            "run_id": ALGO_LIVE_RUN_ID,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _paper_core_needs_reopen(*, session: str, root: Path | None = None) -> bool:
    """True when CoreEquity is still a cash stub from an earlier session."""
    from rlbot.paper_core_equity import paper_book_needs_reopen

    st = _read_json(_exec_dir(root) / "paper_core_equity" / "state.json")
    return paper_book_needs_reopen(st, session=session)


def _soft_paper_core(
    *,
    session: str,
    last_date: str | None,
    root: Path | None = None,
) -> dict[str, Any]:
    force = _paper_core_needs_reopen(session=session, root=root)
    if not paper_day_due(last_date, session=session) and not force:
        return {"ok": True, "skipped": "already_today", "as_of": last_date}
    try:
        from rlbot.pack_core_equity import PACK_DIR
        from rlbot.paper_core_equity import run_paper_day

        if not PACK_DIR.is_dir():
            return {"ok": True, "skipped": "pack_missing"}
        result = call_with_timeout(
            run_paper_day, 180.0, set_active=False, force_refresh=False
        )
        return {
            "ok": True,
            "as_of": result.get("as_of"),
            "bar_date": result.get("bar_date"),
            "actions": result.get("actions"),
            "n_orders": len(result.get("orders") or []),
            "n_positions": result.get("n_positions"),
            "run_id": CORE_EQUITY_LIVE_RUN_ID,
            "force_reopen": force,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _soft_paper_crest(*, session: str, last_date: str | None) -> dict[str, Any]:
    # Crypto is 24/7; still skip a second pass on the same calendar date unless
    # the pack as-of changed (handled inside run_paper_day ledger dedupe).
    del session
    try:
        from rlbot.pack_crestday import PACK_DIR
        from rlbot.paper_crest_day import run_paper_day

        if not PACK_DIR.is_dir():
            return {"ok": True, "skipped": "pack_missing"}
        result = call_with_timeout(
            run_paper_day, 45.0, set_active=False, force_refresh=False
        )
        return {
            "ok": True,
            "as_of": result.get("as_of"),
            "actions": result.get("actions"),
            "n_intents": result.get("n_intents"),
            "run_id": CRYPTO_LIVE_RUN_ID,
            "prior_as_of": last_date,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _soft_rl_shadow(
    *,
    run_id: str,
    now: pd.Timestamp,
    last_shadow_date: str | None,
    root: Path | None = None,
) -> dict[str, Any]:
    force = shadow_ledger_needs_reopen(_shadow_ledger_path(run_id, root))
    if not rl_shadow_due(now, last_shadow_date, force=force):
        return {
            "ok": True,
            "skipped": "not_due",
            "last_shadow_date": last_shadow_date,
            "force_reopen": force,
        }
    weights = (root or PROJECT_ROOT) / "Runs" / run_id / "models" / "best" / "best_model.zip"
    if not weights.is_file():
        return {"ok": True, "skipped": "missing_checkpoint", "run_id": run_id}
    py = sys.executable
    script = PROJECT_ROOT / "scripts" / "shadow_trade.py"
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    try:
        rec = subprocess.run(
            [py, str(script), "record", "--run-id", run_id, "--checkpoint", "best", "--refresh-data"],
            cwd=str(PROJECT_ROOT),
            timeout=600.0,
            check=False,
            capture_output=True,
            env=env,
        )
        recon = subprocess.run(
            [py, str(script), "reconcile", "--run-id", run_id, "--checkpoint", "best"],
            cwd=str(PROJECT_ROOT),
            timeout=180.0,
            check=False,
            capture_output=True,
            env=env,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "run_id": run_id}
    if rec.returncode != 0:
        err = (rec.stderr or b"").decode("utf-8", errors="replace")[-400:]
        return {"ok": False, "error": err or f"record rc={rec.returncode}", "run_id": run_id}
    return {
        "ok": True,
        "recorded": True,
        "reconcile_ok": recon.returncode == 0,
        "as_of": session_date(now),
        "run_id": run_id,
    }


def tick(
    *,
    run_id: str | None = None,
    root: Path | None = None,
    now: pd.Timestamp | None = None,
    interval_s: int = DEFAULT_INTERVAL_S,
    refresh_prices: bool = True,
    run_paper: bool = True,
    run_rl_shadow: bool = True,
) -> dict[str, Any]:
    """One collector pass. Torch-free except the optional RL shadow subprocess."""
    ts = now if now is not None else now_et()
    session = session_date(ts)
    prior = read_status(root)
    rid = canonical_forward_run_id(
        (run_id or "").strip() or (resolve_active_forward_run_id(root) or ALGO_LIVE_RUN_ID)
    )
    sleeves: dict[str, Any] = {}

    mark: dict[str, Any] | None = None
    price_error: str | None = None
    if refresh_prices:
        try:
            mark = refresh_forward_mark_live(
                rid,
                min_refresh_seconds=int(interval_s),
                force_price_refresh=False,
                root=root,
            )
        except Exception as exc:  # noqa: BLE001
            price_error = f"{type(exc).__name__}: {exc}"
            mark = load_forward_mark(rid)

    if isinstance(mark, dict):
        mark = _stamp_collector(mark, interval_s=int(interval_s))
        write_forward_mark(mark)
        publish_public_forward(str(mark.get("run_id") or rid), mark)

    if run_paper:
        sleeves[ALGO_LIVE_RUN_ID] = _soft_paper_ge1(
            session=session,
            last_date=str(prior.get("paper_ge1_date") or "") or None,
        )
        sleeves[CORE_EQUITY_LIVE_RUN_ID] = _soft_paper_core(
            session=session,
            last_date=str(prior.get("paper_core_date") or "") or None,
            root=root,
        )
        sleeves[CRYPTO_LIVE_RUN_ID] = _soft_paper_crest(
            session=session,
            last_date=str(prior.get("paper_crest_date") or "") or None,
        )

    shadow: dict[str, Any] = {"ok": True, "skipped": "disabled"}
    shadow_id = (
        (os.environ.get("MARKETTRAINER_LIVE_RUN_ID") or "").strip() or RL_LIVE_RUN_ID
    )
    if run_rl_shadow:
        shadow = _soft_rl_shadow(
            run_id=shadow_id,
            now=ts,
            last_shadow_date=str(prior.get("rl_shadow_date") or "") or None,
            root=root,
        )
    sleeves[shadow_id] = shadow

    ge1 = sleeves.get(ALGO_LIVE_RUN_ID) or {}
    core = sleeves.get(CORE_EQUITY_LIVE_RUN_ID) or {}
    crest = sleeves.get(CRYPTO_LIVE_RUN_ID) or {}
    paper_ge1_date = str(prior.get("paper_ge1_date") or "")
    paper_core_date = str(prior.get("paper_core_date") or "")
    # Calendar session, not the pack bar date — weekend ticks would otherwise
    # see Friday's as_of and re-enter paper_day every 5 minutes.
    if ge1.get("ok") and not ge1.get("skipped"):
        paper_ge1_date = session
    if core.get("ok") and not core.get("skipped"):
        leftover = _paper_core_needs_reopen(session=session, root=root)
        n_pos = core.get("n_positions")
        if leftover and n_pos is not None and int(n_pos) == 0:
            # Cash-reset leftover: do not mark the session done so the next
            # 5m tick retries after Yahoo publishes today's bar / session flags.
            pass
        else:
            paper_core_date = session
    paper_crest_date = str(crest.get("as_of") or prior.get("paper_crest_date") or "")
    rl_shadow_date = str(prior.get("rl_shadow_date") or "")
    if shadow.get("recorded") and shadow.get("as_of"):
        rl_shadow_date = str(shadow["as_of"])

    live = (mark or {}).get("live") if isinstance(mark, dict) else {}
    status = {
        "schema": "markettrainer.forward_loop.v1",
        "pid": os.getpid(),
        "run_id": rid,
        "session_date": session,
        "last_tick_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "last_tick_et": str(ts),
        "interval_s": int(interval_s),
        "n_bars": (mark or {}).get("n_bars") if isinstance(mark, dict) else None,
        "last_price_bar": (live or {}).get("last_price_bar") if isinstance(live, dict) else None,
        "prices_stale": (live or {}).get("prices_stale") if isinstance(live, dict) else None,
        "price_error": price_error,
        "paper_ge1_date": paper_ge1_date or None,
        "paper_core_date": paper_core_date or None,
        "paper_crest_date": paper_crest_date or None,
        "rl_shadow_date": rl_shadow_date or None,
        "sleeves": sleeves,
    }
    write_status(status, root)
    return status


def run_loop(
    *,
    interval_s: int = DEFAULT_INTERVAL_S,
    run_id: str | None = None,
    root: Path | None = None,
    once: bool = False,
    exit_if_locked: bool = True,
    refresh_prices: bool = True,
    run_paper: bool = True,
    run_rl_shadow: bool = True,
    take_lock: bool = True,
) -> dict[str, Any] | None:
    """Acquire the collector lock and tick until killed (or once).

    ``take_lock=False`` is for a nested call from the stdlib lite collector,
    which already holds ``forward_loop.lock``.
    """
    lock = LoopLock(_lock_path(root))
    if take_lock:
        if not lock.acquire(blocking=False):
            if exit_if_locked:
                print("[forward-loop] another collector holds the lock; exiting", flush=True)
                return None
            if not lock.acquire(blocking=True):
                return None
    try:
        print(
            f"[forward-loop] starting interval={interval_s}s pid={os.getpid()} once={once}",
            flush=True,
        )
        last: dict[str, Any] | None = None
        while True:
            t0 = time.time()
            try:
                last = tick(
                    run_id=run_id,
                    root=root,
                    interval_s=int(interval_s),
                    refresh_prices=refresh_prices,
                    run_paper=run_paper,
                    run_rl_shadow=run_rl_shadow,
                )
                n_bars = last.get("n_bars")
                last_bar = last.get("last_price_bar")
                print(
                    f"[forward-loop] tick run_id={last.get('run_id')} "
                    f"bars={n_bars} last={last_bar} "
                    f"stale={last.get('prices_stale')}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[forward-loop] tick failed: {exc}", flush=True)
                last = {"ok": False, "error": str(exc)}
            if once:
                return last
            elapsed = time.time() - t0
            time.sleep(max(5.0, float(interval_s) - elapsed))
    finally:
        if take_lock:
            lock.release()
