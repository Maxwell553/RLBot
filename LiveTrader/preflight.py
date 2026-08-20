"""Pre-live checks: data freshness, account, foreign names, port/mode, fractionals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np

from book import foreign_symbols, needs_fractional, session_rebalance_flags
from config import LiveConfig
from ibkr_client import AccountSnapshot, ib_insync_available


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    blocking: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stale_session(asof: date, today: date | None = None) -> bool:
    today = today or datetime.now(timezone.utc).date()
    return asof < today - timedelta(days=4)


def evaluate(
    *,
    cfg: LiveConfig,
    dates: list[date],
    px: dict[str, np.ndarray],
    ohlc: dict[str, tuple[np.ndarray, ...]],
    marks: dict[str, float],
    weights: dict[str, float],
    snapshot: AccountSnapshot | None,
    orders: list[Any] | None = None,
    want_connect: bool = False,
    want_submit: bool = False,
    arm_live: bool = False,
    confirm_env: str = "",
    already_traded: bool = False,
    skip_reason: str = "",
    flatten: bool = False,
) -> list[Check]:
    del ohlc, marks
    checks: list[Check] = []
    asof = dates[-1] if dates else None
    checks.append(
        Check(
            "panel_asof",
            bool(asof) and not _stale_session(asof),
            f"asof={asof} n_bars={len(dates)}",
        )
    )
    tqqq = px.get("TQQQ")
    last252 = tqqq[-252:] if tqqq is not None and len(tqqq) >= 252 else tqqq
    tqqq_ok = last252 is not None and int(np.isfinite(last252).sum()) == len(last252)
    checks.append(
        Check(
            "tqqq_recent_finite",
            bool(tqqq_ok),
            "last 252 TQQQ closes are finite" if tqqq_ok else "TQQQ recent bars have NaNs",
        )
    )
    wsum = float(sum(weights.values())) if weights else 0.0
    checks.append(
        Check("weights_simplex", abs(wsum - 1.0) < 1e-6, f"sum={wsum:.8f}"),
    )
    wk, me = session_rebalance_flags(dates, len(dates) - 1) if dates else (False, False)
    checks.append(
        Check(
            "rebalance_calendar",
            True,
            f"week_end={wk} month_end={me} skip={skip_reason or 'none'}",
            blocking=False,
        )
    )
    if cfg.mode == "paper" and cfg.is_live_port:
        checks.append(
            Check("port_mode", False, f"paper mode on live port {cfg.port}"),
        )
    elif cfg.mode == "live" and cfg.is_paper_port:
        checks.append(
            Check("port_mode", False, f"live mode on paper port {cfg.port}"),
        )
    else:
        checks.append(
            Check("port_mode", True, f"mode={cfg.mode} port={cfg.port}"),
        )

    if want_submit and cfg.mode == "dry_run":
        checks.append(Check("submit_mode", False, "dry_run cannot submit orders"))
    else:
        checks.append(Check("submit_mode", True, f"mode={cfg.mode} want_submit={want_submit}"))

    if want_submit and cfg.mode == "live":
        env_ok = str(confirm_env or "").strip() == str(cfg.confirm_phrase)
        checks.append(
            Check(
                "live_arm",
                bool(cfg.allow_live and arm_live and env_ok),
                "need config allow_live=true, --arm-live, and LIVE_TRADER_CONFIRM="
                f"{cfg.confirm_phrase!r} (got allow_live={cfg.allow_live} arm={arm_live} "
                f"env_set={bool(str(confirm_env or '').strip())})",
            )
        )

    if want_submit and cfg.require_account and not str(cfg.account or "").strip():
        checks.append(
            Check(
                "account_pin",
                False,
                "set IBKR_ACCOUNT (or ibkr.account) before paper/live submit",
            )
        )
    else:
        checks.append(
            Check(
                "account_pin",
                True,
                f"account={cfg.account or '(blank, allowed for dry-run)'}",
                blocking=False,
            )
        )

    if want_submit and already_traded and not flatten:
        checks.append(
            Check("already_traded", False, f"last_trade_date already {asof}; refuse duplicate submit"),
        )
    else:
        checks.append(
            Check("already_traded", True, skip_reason or "ok", blocking=False),
        )

    if want_connect or snapshot is not None:
        checks.append(
            Check(
                "ib_insync",
                ib_insync_available() or snapshot is not None,
                "ib_insync installed" if ib_insync_available() else "pip install ib_insync",
            )
        )
    else:
        checks.append(
            Check(
                "ib_insync",
                True,
                "not required for offline plan",
                blocking=False,
            )
        )

    if snapshot is None:
        if want_connect or want_submit:
            checks.append(Check("ib_snapshot", False, "no IBKR snapshot (TWS/Gateway not connected)"))
        else:
            checks.append(
                Check("ib_snapshot", True, "skipped (offline)", blocking=False),
            )
        return checks

    if cfg.account and snapshot.account and cfg.account != snapshot.account:
        checks.append(
            Check(
                "account_match",
                False,
                f"config account {cfg.account} != snapshot {snapshot.account}",
            )
        )
    else:
        checks.append(
            Check("account_match", True, f"snapshot={snapshot.account}", blocking=False),
        )

    nlv = float(snapshot.net_liquidation or 0.0)
    checks.append(
        Check(
            "account_nlv",
            nlv > 0,
            f"account={snapshot.account} NLV={nlv:.2f} cash={snapshot.cash:.2f}",
        )
    )
    acct = str(snapshot.account or "")
    paperish = acct.upper().startswith("DU")
    liveish = acct.upper().startswith("U") and not paperish
    if cfg.mode == "paper" and liveish:
        checks.append(
            Check("account_kind", False, f"paper mode but account {acct} looks live (U…, not DU…)"),
        )
    elif cfg.mode == "live" and paperish:
        checks.append(
            Check("account_kind", False, f"live mode but account {acct} looks paper (DU…)"),
        )
    else:
        checks.append(
            Check("account_kind", True, f"account={acct} paperish={paperish}", blocking=False),
        )

    foreign = foreign_symbols(snapshot.positions)
    if foreign and not cfg.allow_foreign_positions and not flatten:
        checks.append(
            Check(
                "foreign_positions",
                False,
                "flatten these before GE1 can own the book: " + ", ".join(foreign),
            )
        )
    else:
        checks.append(
            Check(
                "foreign_positions",
                True,
                "none" if not foreign else "allowed: " + ", ".join(foreign),
            )
        )

    order_syms = {str(getattr(o, "symbol", "")).upper() for o in (orders or [])}
    open_syms = {str(s).upper() for s in (snapshot.open_order_symbols or [])}
    clash = sorted(order_syms & open_syms)
    if want_submit and clash and not flatten:
        checks.append(
            Check("open_orders", False, "working IB orders already on " + ", ".join(clash)),
        )
    else:
        checks.append(
            Check(
                "open_orders",
                True,
                "none" if not snapshot.open_order_symbols else ",".join(snapshot.open_order_symbols),
                blocking=False,
            )
        )

    frac_needed = needs_fractional(orders or [])
    if not cfg.allow_fractional and frac_needed:
        checks.append(
            Check(
                "fractional",
                False,
                "QQQ/GLD legs need fractionals at this AUM: " + ", ".join(frac_needed),
            )
        )
    else:
        checks.append(
            Check(
                "fractional",
                True,
                "allow_fractional=true"
                + (f"; sub-share qty on {', '.join(frac_needed)}" if frac_needed else ""),
            )
        )

    if nlv > 0 and nlv < 2000:
        checks.append(
            Check(
                "small_account",
                True,
                f"NLV ${nlv:.0f} is below typical Reg-T $2k; expect cash-like constraints",
                blocking=False,
            )
        )
    return checks


def blocking_failures(checks: list[Check]) -> list[Check]:
    return [c for c in checks if c.blocking and not c.ok]
