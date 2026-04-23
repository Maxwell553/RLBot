"""Connect to TWS, read NLV and positions, place market orders toward target weight dollar notionals."""

from __future__ import annotations

import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from ib_insync import IB, MarketOrder, PortfolioItem, Ticker

# Paper (default) / live; warn if 7496
DEFAULT_PORT = 7497
LIVE_PORT = 7496


@dataclass
class RebalanceConfig:
    min_leg_dollars: float = 100.0
    forex_lot: float = 1000.0
    market_data_timeout: float = 5.0


def _parse_managed_account_ids(ib: IB) -> list[str]:
    """Return linked account name(s) from TWS (e.g. paper DU...). ``managedAccounts`` is a method."""
    try:
        m = ib.managedAccounts
        out = m() if callable(m) else m
    except Exception:
        return []
    if not out:
        return []
    if isinstance(out, str):
        for sep in (";", ","):
            if sep in out:
                return [a.strip() for a in out.split(sep) if a.strip()]
        return [out.strip()] if out.strip() else []
    return [str(a).strip() for a in out if str(a).strip()]


def _av_ok_for_account(v, primary_accounts: list[str]) -> bool:
    if not primary_accounts:
        return True
    ac = getattr(v, "account", None) or ""
    return (not ac) or (ac in primary_accounts)


def _merge_account_value_rows(ib: IB) -> list:
    """
    ``reqAccountSummary`` → :meth:`ib.accountSummary` (wrapper.acctSummary) contains
    **NetLiquidation** and the usual summary tags. ``ib.accountValues`` is filled by
    **reqAccountUpdates** and may be empty until that runs — do not use summary-only tags from there.
    """
    out: list = []
    try:
        out.extend(ib.accountSummary())
    except Exception:
        pass
    try:
        extra = list(ib.accountValues())
    except Exception:
        extra = []
    seen = {(v.account, v.tag, v.currency) for v in out}
    for v in extra:
        k = (v.account, v.tag, v.currency)
        if k not in seen:
            seen.add(k)
            out.append(v)
    return out


def _max_positive_for_tag_rows(
    value_rows: list,
    tag: str,
    primary_accounts: list[str],
) -> tuple[float, str]:
    """
    Return (value, detail) for the best *positive* value for this tag
    (IB often reports NetLiquidation in USD, BASE, or empty currency).
    """
    def _collect(use_acct_filter: bool) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        for v in value_rows:
            if v.tag != tag:
                continue
            if use_acct_filter and not _av_ok_for_account(v, primary_accounts):
                continue
            try:
                x = float(v.value)
            except (TypeError, ValueError):
                continue
            if x <= 0:
                continue
            out.append((v.currency, x))
        return out

    rows = _collect(True)
    if not rows and primary_accounts:
        rows = _collect(False)
    if not rows:
        return 0.0, ""
    # Prefer BASE before USD: for many accounts the consolidated total is in BASE;
    # a separate USD line can be a partial / segment (e.g. ~1M vs ~5M in TWS).
    for cur in ("BASE", "USD"):
        for c, x in rows:
            if c == cur:
                return x, f"{tag} {c}=${x:,.2f}"
    for c, x in rows:
        if c in ("", "NONE"):
            return x, f"{tag} {c!r}=${x:,.2f}"
    c, x = rows[0]
    return x, f"{tag} {c}=${x:,.2f}"


def _aggregate_net_liquidation(
    merged: list, primary_accounts: list[str]
) -> tuple[float, str]:
    """
    IB may send *NetLiquidation* per sub-account. Taking one *USD* / *BASE* across *all* rows
    returns whichever row appears first (often a single ~$1M leg). For each acct, pick the
    best currency (BASE before USD), then **sum** — matches a family of paper accounts
    and fixes single-acct when multiple currency rows exist.
    """
    by_acct: dict[str, list] = defaultdict(list)
    for v in merged:
        if v.tag != "NetLiquidation":
            continue
        if not _av_ok_for_account(v, primary_accounts):
            continue
        ac = (getattr(v, "account", None) or "").strip()
        by_acct[ac].append(v)
    if not by_acct:
        return 0.0, ""
    total = 0.0
    parts: list[str] = []
    for ac in sorted(by_acct.keys(), key=str):
        sub = by_acct[ac]
        prim: list[str] = [ac] if ac else primary_accounts
        n, _s = _max_positive_for_tag_rows(sub, "NetLiquidation", prim)
        if n > 0:
            total += n
            label = ac if ac else "default"
            parts.append(f"{label} ${n:,.0f}")
    if total <= 0:
        return 0.0, ""
    return total, f"NetLiquidation Σaccts [ {' | '.join(parts)} ] total ≈ ${total:,.2f}"


def _portfolio_gross_mkt_value(
    ib: IB, primary_accounts: list[str]
) -> tuple[float, float, str, str]:
    """
    Return (sum_primary, sum_all, desc_primary, desc_all) for portfolio line market value,
    with per-(account, conId) de-dupe. TWS *Net Liq* for cash-heavy books often matches
    this sum, while *NetLiquidation* in account tags can be a different / stale segment
    (e.g. ~1M vs ~5M in the same session).
    """
    seen_p: set[tuple[str, int]] = set()
    seen_a: set[tuple[str, int]] = set()
    sum_p = 0.0
    sum_a = 0.0
    n_p = 0
    n_a = 0
    for p in ib.portfolio():
        acc = getattr(p, "account", None) or ""
        if primary_accounts and acc and acc not in primary_accounts:
            pass_ok = False
        else:
            pass_ok = True
        try:
            cid = int(getattr(p.contract, "conId", 0) or 0)
        except (TypeError, ValueError):
            cid = 0
        key = (acc, cid)
        try:
            mv = float(p.marketValue)
        except (TypeError, ValueError):
            continue
        if key not in seen_a:
            seen_a.add(key)
            sum_a += mv
            n_a += 1
        if pass_ok and key not in seen_p:
            seen_p.add(key)
            sum_p += mv
            n_p += 1
    d_p = f"sum(portfolio) primary acct, {n_p} line(s) = ${sum_p:,.2f}"
    d_a = f"sum(portfolio) all accts, {n_a} line(s) = ${sum_a:,.2f}"
    return sum_p, sum_a, d_p, d_a


def _nlv_from_portfolio_plus_cash(ib: IB, primary_accounts: list[str]) -> tuple[float, str]:
    """
    If NetLiquidation is missing, approximate: sum(portfolio mkt) + TotalCashValue (BASE/USD),
    for paper this usually matches the account window.
    """
    mv = 0.0
    for p in ib.portfolio():
        acc = getattr(p, "account", None) or ""
        if primary_accounts and acc and acc not in primary_accounts:
            continue
        try:
            mv += float(p.marketValue)
        except (TypeError, ValueError):
            pass
    merged = _merge_account_value_rows(ib)
    cash, _ = _max_positive_for_tag_rows(merged, "TotalCashValue", primary_accounts)
    s = float(mv) + float(cash)
    if s > 0:
        return s, f"sum(portfolio mkt) + TotalCashValue ≈ ${s:,.2f}"
    return 0.0, ""


def nlv_usd(ib: IB) -> tuple[float, str]:
    """
    Net liquidation in US dollars, plus a short string describing the source.
    Uses **ib.accountSummary()** first — that is how IB delivers NetLiquidation
    (``accountValues`` alone is often empty on API connect).
    """
    primary = _parse_managed_account_ids(ib)
    # Portfolio (incl. CASH lines) can lag a moment after connect; do not call
    # reqAccountUpdates here — connect() already subscribes; re-requesting for every
    # sub-account can time out and duplicate load (see ib_insync connect logs).
    sum_pv_p = 0.0
    sum_pv_a = 0.0
    desc_p = desc_a = ""
    for _ in range(5):
        sum_pv_p, sum_pv_a, desc_p, desc_a = _portfolio_gross_mkt_value(ib, primary)
        if sum_pv_p > 0.0 or sum_pv_a > 0.0:
            break
        ib.sleep(0.1)
    merged = _merge_account_value_rows(ib)
    nlv_tag, src_tag = _aggregate_net_liquidation(merged, primary)
    nlv0, _src0 = _max_positive_for_tag_rows(merged, "NetLiquidation", primary)
    if nlv_tag > 0.0:
        nlv, src = nlv_tag, src_tag
    else:
        nlv, src = nlv0, _src0
    tcv, _tcv_src = _max_positive_for_tag_rows(merged, "TotalCashValue", primary)
    gpv, _gpv_src = _max_positive_for_tag_rows(merged, "GrossPositionValue", primary)
    # Prefer the larger portfolio aggregate when tag NLV is clearly a partial / wrong segment
    # (e.g. NetLiquidation USD ≈1M but TWS and portfolio CASH / lines ≈5M).
    sum_pv = max(sum_pv_p, sum_pv_a)
    pv_line = desc_a if sum_pv_a > sum_pv_p + 1.0 else desc_p
    if sum_pv > 0.0 and (nlv <= 0.0 or sum_pv > nlv * 1.01 + 1.0):
        return float(sum_pv), f"{pv_line} (TWS line-up; acct tag had ${nlv:,.0f} in NetLiquidation)"
    # All-cash (or tiny positions): TWS "Net Liq" often tracks TotalCash; API NetLiquidation
    # can lag or report a partial USD line — prefer cash when it clearly dominates.
    if tcv > 0 and nlv > 0 and tcv > nlv * 1.05 and float(gpv) < 10_000.0:
        return tcv, (
            f"TotalCashValue ${tcv:,.2f} (cash book; NetLiquidation was ${nlv:,.2f}, "
            f"GrossPositionValue ${gpv:,.2f})"
        )
    if nlv > 0:
        return nlv, src or "NetLiquidation (account summary)"
    if tcv > 0 and float(gpv) < 10_000.0:
        return tcv, f"TotalCashValue ${tcv:,.2f} (no NetLiquidation row)"
    nlv2, src2 = _max_positive_for_tag_rows(merged, "EquityWithLoanValue", primary)
    if nlv2 > 0:
        return nlv2, src2 or "EquityWithLoanValue"
    nlv3, src3 = _nlv_from_portfolio_plus_cash(ib, primary)
    if nlv3 > 0:
        return nlv3, src3
    if sum_pv > 0.0:
        return float(sum_pv), pv_line or "sum(portfolio marketValue)"
    return 0.0, "no positive NetLiquidation / equity / (portfolio+cash) found"


def connect_ib(host: str, port: int, client_id: int) -> IB:
    if port == LIVE_PORT:
        print(
            f"WARNING: IB_PORT={port} is the usual *live* TWS port. "
            f"For paper, use {DEFAULT_PORT} (typical) unless you intend live."
        )
    # Default connect sync (positions, many sub-accounts' updates) can exceed 4s; raise cap.
    try:
        tmo = float(os.environ.get("IB_CONNECT_TIMEOUT", "25"))
    except ValueError:
        tmo = 25.0
    if tmo < 0:
        tmo = 0.0
    ib = IB()
    ib.connect(host, port, clientId=client_id, readonly=False, timeout=tmo)
    return ib


def qualify_legs(ib: IB, contracts: list) -> list:
    return list(ib.qualifyContracts(*contracts))


def _con_to_logical(qualified: list, logical_order: list[str]) -> dict[int, str]:
    return {c.conId: logical_order[i] for i, c in enumerate(qualified)}


def mkt_dollar_by_logical(
    ib: IB, qualified: list, logical_order: list[str]
) -> dict[str, float]:
    c2l = _con_to_logical(qualified, logical_order)
    mkt: dict[str, float] = {n: 0.0 for n in logical_order}
    for p in ib.portfolio():
        name = c2l.get(p.contract.conId)
        if name is not None:
            mkt[name] += float(p.marketValue)
    return mkt


def _item_by_logical(
    ib: IB, qualified: list, logical_order: list[str]
) -> dict[str, PortfolioItem | None]:
    c2l = _con_to_logical(qualified, logical_order)
    out: dict[str, PortfolioItem | None] = {n: None for n in logical_order}
    for p in ib.portfolio():
        name = c2l.get(p.contract.conId)
        if name is not None and out[name] is None:
            out[name] = p
    return out


def _wait_ticker(
    t: Ticker, timeout: float, ib: IB
) -> Ticker:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if t.bid and t.ask and t.bid > 0 and t.ask > 0:
            return t
        if t.last and t.last > 0:
            return t
        ib.sleep(0.1)
    return t


def _yf_last_close(yahoo_symbol: str) -> float:
    import yfinance as yf

    df = yf.Ticker(yahoo_symbol).history(period="10d", interval="1d", auto_adjust=False)
    if df is None or df.empty or "Close" not in df.columns:
        raise RuntimeError(f"No yfinance history for {yahoo_symbol!r}")
    s = float(df["Close"].dropna().iloc[-1])
    if s <= 0:
        raise RuntimeError(f"Invalid yfinance close for {yahoo_symbol!r}")
    return s


def _price_for_leg(
    ib: IB,
    c,
    item: PortfolioItem | None,
    timeout: float,
    yf_symbol_for_fallback: str | None = None,
) -> tuple[float, str]:
    """Return (price, source) where source is 'ib' or 'yfinance'."""
    t = ib.reqMktData(c, "", False, False)
    try:
        t = _wait_ticker(t, timeout, ib)
        if t.bid and t.ask and t.bid > 0 and t.ask > 0:
            return (t.bid + t.ask) / 2.0, "ib"
        if t.last and t.last > 0:
            return float(t.last), "ib"
        if t.close and t.close > 0:
            return float(t.close), "ib"
    finally:
        ib.cancelMktData(c)
    if item and abs(item.position) > 1e-9:
        return float(item.marketValue) / float(item.position), "ib_position"
    if yf_symbol_for_fallback:
        return _yf_last_close(yf_symbol_for_fallback), "yfinance"
    raise RuntimeError(
        f"No market price for {c} (IB has no data; subscribe to market data in TWS, "
        f"or enable delayed data under Global Config → API / Market data.)"
    )


def _round_shares(
    dollar_delta: float, price: float, max_sell: float | None
) -> int:
    if price <= 0:
        return 0
    n = int(abs(dollar_delta) / price)
    if n < 1 and abs(dollar_delta) >= 1.0:
        n = 1
    if dollar_delta < 0:
        n = -n
    if n < 0 and max_sell is not None and abs(n) > max_sell + 1e-6:
        n = -int(max_sell)
    return n


def _round_fx(
    dollar_delta: float, price: float, item: PortfolioItem | None, lot: float
) -> int:
    if price <= 0:
        return 0
    if item and abs(item.position) > 1e-9:
        usd_per_base = float(item.marketValue) / float(item.position)
    else:
        usd_per_base = float(price)
    d_base = dollar_delta / usd_per_base
    step = max(lot, 1.0)
    m = d_base / step
    n = int(math.copysign(int(round(abs(m))), m))
    q = int(n * int(step)) if n != 0 else 0
    if q == 0 and abs(d_base) * usd_per_base > 1.0:
        q = int(math.copysign(int(step), d_base)) if d_base > 0 else 0
    if q < 0 and item and abs(item.position) > 0:
        pos = int(abs(item.position))
        if abs(q) > pos:
            q = -pos
    return q


def rebalance_toward_targets(
    ib: IB,
    target_w: np.ndarray,
    nlv: float,
    logical_order: list[str],
    qualified: list,
    config: RebalanceConfig | None = None,
    yf_ticker_by_logical: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """
    Rebalance to ``target_w[1:11] * nlv`` notional per leg (model weights, including cash w[0]).

    If IB market data is not subscribed, pass ``yf_ticker_by_logical`` so share sizing can
    use yfinance last closes for the *executed* symbol (e.g. EWJ for NIKKEI).
    """
    cfg = config or RebalanceConfig()
    yf_map = yf_ticker_by_logical or {}
    assert len(target_w) == 11, "11 slots (cash + 10)"
    assert len(qualified) == 10, "10 risky legs"
    mkt0 = mkt_dollar_by_logical(ib, qualified, logical_order)
    items = _item_by_logical(ib, qualified, logical_order)
    log: list[dict[str, Any]] = []
    yf_used_once = False

    for i, name in enumerate(logical_order):
        c = qualified[i]
        t_usd = float(target_w[i + 1]) * float(nlv)
        cur = mkt0.get(name, 0.0)
        delta = t_usd - cur
        item = items.get(name)
        if abs(delta) < cfg.min_leg_dollars:
            log.append(
                {
                    "logical": name,
                    "action": "skip",
                    "reason": f"|delta| ${abs(delta):.2f} < min_leg {cfg.min_leg_dollars}",
                }
            )
            continue
        yf_sym = yf_map.get(name)
        p, px_src = _price_for_leg(
            ib, c, item, cfg.market_data_timeout, yf_symbol_for_fallback=yf_sym
        )
        if px_src == "yfinance" and yf_sym:
            yf_used_once = True
        max_sell: float | None = None
        if c.secType == "STK" and item is not None and item.position > 0 and delta < 0:
            max_sell = float(item.position)
        if c.secType == "STK":
            sh = _round_shares(delta, p, max_sell)
            if sh == 0:
                log.append({"logical": name, "action": "skip", "reason": "zero shares after round"})
                continue
            o = MarketOrder("BUY" if sh > 0 else "SELL", abs(sh))
        else:
            qfx = _round_fx(delta, p, item, cfg.forex_lot)
            if qfx == 0:
                log.append({"logical": name, "action": "skip", "reason": "zero FX qty after round"})
                continue
            o = MarketOrder("BUY" if qfx > 0 else "SELL", abs(qfx))
        ib.placeOrder(c, o)
        log.append(
            {
                "logical": name,
                "action": o.action,
                "qty": o.totalQuantity,
                "conId": c.conId,
                "price_ref": p,
                "price_source": px_src,
            }
        )
    if yf_used_once:
        print(
            "  (Some legs used yfinance last close for order sizing: IB real-time / delayed API "
            "data not available. Subscribe in Account Management to use IB quotes for sizing.)"
        )
    return log


def dry_run_deltas(
    nlv: float,
    target_w: np.ndarray,
    current_mkt: dict[str, float],
    logical_order: list[str],
    min_leg: float = 100.0,
) -> list[dict[str, Any]]:
    rows = []
    for i, name in enumerate(logical_order):
        t_usd = float(target_w[i + 1]) * nlv
        cur = current_mkt.get(name, 0.0)
        delta = t_usd - cur
        rows.append(
            {
                "logical": name,
                "target_usd": t_usd,
                "current_usd": cur,
                "delta_usd": delta,
                "skip": abs(delta) < min_leg,
            }
        )
    return rows
