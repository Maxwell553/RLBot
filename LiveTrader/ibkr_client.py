"""Interactive Brokers adapter (ib_insync). Read-only unless submit() is called."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from book import OrderIntent
from config import LiveConfig
from ge1_strategy import BOOK_SYMBOLS


DONE_STATUSES = frozenset(
    {"Filled", "Cancelled", "ApiCancelled", "Inactive", "PendingCancel"}
)
ACCEPTED_STATUSES = frozenset(
    {"Submitted", "PreSubmitted", "Filled", "PendingSubmit", "ApiPending"}
)
REJECT_STATUSES = frozenset({"Cancelled", "ApiCancelled", "Inactive"})


@dataclass
class Position:
    symbol: str
    qty: float
    avg_cost: float = 0.0
    market_price: float = 0.0


@dataclass
class AccountSnapshot:
    account: str
    net_liquidation: float
    cash: float
    buying_power: float
    positions: dict[str, float]
    position_rows: list[Position] = field(default_factory=list)
    managed_accounts: list[str] = field(default_factory=list)
    last_prices: dict[str, float] = field(default_factory=dict)
    open_order_symbols: list[str] = field(default_factory=list)
    port: int = 0
    server_name: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "account": self.account,
            "net_liquidation": self.net_liquidation,
            "cash": self.cash,
            "buying_power": self.buying_power,
            "positions": dict(self.positions),
            "last_prices": dict(self.last_prices),
            "open_order_symbols": list(self.open_order_symbols),
            "managed_accounts": list(self.managed_accounts),
            "port": self.port,
            "server_name": self.server_name,
        }


class BrokerError(RuntimeError):
    pass


def ib_insync_available() -> bool:
    """True when the package is installed. Do not import it (eventkit needs a loop)."""
    import importlib.util

    return importlib.util.find_spec("ib_insync") is not None


def _ensure_asyncio_loop() -> None:
    import asyncio

    try:
        asyncio.get_running_loop()
        return
    except RuntimeError:
        pass
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


def _usd_tag(rows: list[Any], tag: str) -> float:
    for row in rows:
        if str(getattr(row, "tag", "")) == tag and str(getattr(row, "currency", "USD")) in (
            "USD",
            "",
        ):
            try:
                return float(row.value)
            except (TypeError, ValueError):
                continue
    for row in rows:
        if str(getattr(row, "tag", "")) == tag:
            try:
                return float(row.value)
            except (TypeError, ValueError):
                continue
    return 0.0


def _ticker_px(ticker: Any) -> float:
    for attr in ("last", "close", "bid", "ask"):
        v = getattr(ticker, attr, None)
        try:
            px = float(v)
        except (TypeError, ValueError):
            continue
        if px > 0:
            return px
    try:
        mid = ticker.marketPrice()
        px = float(mid)
        if px > 0:
            return px
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def _build_ib_order(intent: OrderIntent, *, default_type: str, tif: str):
    from ib_insync import Order

    qty = round(float(intent.qty), 6)
    order = Order()
    order.action = "BUY" if intent.side == "buy" else "SELL"
    order.totalQuantity = qty
    order.orderType = str(intent.order_type or default_type).upper()
    order.tif = str(tif).upper()
    # ib_insync defaults these True on some versions; IBKR then rejects 10268.
    order.eTradeOnly = False
    order.firmQuoteOnly = False
    if order.orderType == "MKT" and abs(qty - round(qty)) > 1e-9:
        order.tif = "DAY"
    return order, qty


class FakeBroker:
    """In-process broker for tests and offline dry-run."""

    def __init__(self, snapshot: AccountSnapshot):
        self._snap = snapshot
        self.submitted: list[OrderIntent] = []
        self.connected = False
        self.what_if_ok = True
        self.what_if_detail = "ok"

    def connect(self) -> FakeBroker:
        self.connected = True
        return self

    def disconnect(self) -> None:
        self.connected = False

    def snapshot(self) -> AccountSnapshot:
        return self._snap

    def qualify_book(self) -> list[str]:
        return list(BOOK_SYMBOLS)

    def last_prices(self, symbols: list[str]) -> dict[str, float]:
        out = dict(self._snap.last_prices)
        for s in symbols:
            su = str(s).upper()
            if su not in out:
                out[su] = float(self._snap.last_prices.get(su) or 0.0)
        return out

    def open_order_symbols(self) -> list[str]:
        return list(self._snap.open_order_symbols)

    def what_if(self, orders: list[OrderIntent], *, order_type: str, tif: str) -> dict[str, Any]:
        del order_type, tif
        return {"ok": bool(self.what_if_ok), "detail": self.what_if_detail, "n": len(orders)}

    def submit(self, orders: list[OrderIntent], *, order_type: str, tif: str) -> list[dict[str, Any]]:
        del order_type, tif
        self.submitted.extend(orders)
        pos = dict(self._snap.positions)
        cash = float(self._snap.cash)
        for o in orders:
            qty = float(o.qty)
            if o.side == "sell":
                have = float(pos.get(o.symbol, 0.0))
                sell = min(have, qty)
                pos[o.symbol] = have - sell
                if pos[o.symbol] <= 1e-12:
                    pos.pop(o.symbol, None)
                cash += sell * float(o.mark)
                status = "Filled"
            else:
                pos[o.symbol] = pos.get(o.symbol, 0.0) + qty
                cash -= qty * float(o.mark)
                status = "Filled"
        self._snap.positions = pos
        self._snap.cash = cash
        self._snap.open_order_symbols = []
        return [
            {
                "symbol": o.symbol,
                "side": o.side,
                "qty": o.qty,
                "order_type": o.order_type,
                "status": "Filled",
                "filled": o.qty,
                "remaining": 0.0,
                "avg_fill_price": o.mark,
            }
            for o in orders
        ]

    def wait_for_fills(
        self,
        reports: list[dict[str, Any]],
        *,
        timeout_s: float,
    ) -> tuple[list[dict[str, Any]], bool]:
        del timeout_s
        return reports, all(str(r.get("status")) == "Filled" for r in reports)

    def cancel_open_book_orders(self) -> list[str]:
        had = list(self._snap.open_order_symbols)
        self._snap.open_order_symbols = []
        return had


class IBKRBroker:
    def __init__(self, cfg: LiveConfig):
        if not ib_insync_available():
            raise BrokerError(
                "ib_insync is not installed. pip install ib_insync  "
                "(or pip install -e '.[live]')"
            )
        self.cfg = cfg
        self.ib = None
        self._last_trades: list[Any] = []

    def connect(self) -> IBKRBroker:
        _ensure_asyncio_loop()
        from ib_insync import IB

        ib = IB()
        try:
            ib.connect(
                self.cfg.host,
                int(self.cfg.port),
                clientId=int(self.cfg.client_id),
                timeout=float(self.cfg.connect_timeout_s),
            )
        except Exception as exc:  # noqa: BLE001
            raise BrokerError(
                f"could not connect to IBKR at {self.cfg.host}:{self.cfg.port} "
                f"(start TWS/Gateway, enable API, check port). {exc}"
            ) from exc
        if int(self.cfg.market_data_type) in {1, 2, 3, 4}:
            try:
                ib.reqMarketDataType(int(self.cfg.market_data_type))
            except Exception:  # noqa: BLE001
                pass
        self.ib = ib
        return self

    def disconnect(self) -> None:
        if self.ib is not None and self.ib.isConnected():
            self.ib.disconnect()
        self.ib = None

    def _require(self):
        if self.ib is None or not self.ib.isConnected():
            raise BrokerError("not connected")
        return self.ib

    def _stock(self, symbol: str):
        from ib_insync import Stock

        ib = self._require()
        contract = Stock(str(symbol).upper(), "SMART", "USD")
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            raise BrokerError(f"could not qualify {symbol}")
        return qualified[0]

    def last_prices(self, symbols: list[str]) -> dict[str, float]:
        ib = self._require()
        tickers = []
        for sym in symbols:
            try:
                contract = self._stock(sym)
            except BrokerError:
                continue
            tickers.append((str(sym).upper(), ib.reqMktData(contract, "", False, False)))
        if tickers:
            ib.sleep(max(0.2, float(self.cfg.mkt_data_wait_s)))
        out: dict[str, float] = {}
        for sym, ticker in tickers:
            px = _ticker_px(ticker)
            if px > 0:
                out[sym] = px
            try:
                ib.cancelMktData(ticker.contract)
            except Exception:  # noqa: BLE001
                pass
        return out

    def open_order_symbols(self) -> list[str]:
        ib = self._require()
        try:
            ib.reqOpenOrders()
            ib.sleep(0.3)
        except Exception:  # noqa: BLE001
            pass
        found: set[str] = set()
        for trade in list(ib.openTrades() or []):
            st = str(getattr(trade.orderStatus, "status", "") or "")
            if st in REJECT_STATUSES:
                continue
            sym = str(getattr(getattr(trade, "contract", None), "symbol", "") or "").upper()
            if sym:
                found.add(sym)
        return sorted(found)

    def snapshot(self) -> AccountSnapshot:
        ib = self._require()
        managed = [str(a) for a in (ib.managedAccounts() or [])]
        account = self.cfg.account or (managed[0] if managed else "")
        if self.cfg.account and account != self.cfg.account:
            account = self.cfg.account
        values = list(ib.accountValues(account) if account else ib.accountValues())
        positions: dict[str, float] = {}
        rows: list[Position] = []
        for pos in ib.positions(account) if account else ib.positions():
            sym = str(getattr(pos.contract, "symbol", "") or "").upper()
            if not sym:
                continue
            qty = float(pos.position)
            if abs(qty) <= 1e-12:
                continue
            positions[sym] = positions.get(sym, 0.0) + qty
            rows.append(
                Position(
                    symbol=sym,
                    qty=qty,
                    avg_cost=float(getattr(pos, "avgCost", 0.0) or 0.0),
                )
            )
        want = sorted(set(BOOK_SYMBOLS) | set(positions))
        last = self.last_prices(want)
        for row in rows:
            row.market_price = float(last.get(row.symbol) or 0.0)
        server = ""
        try:
            server = str(ib.client.serverVersion())
        except Exception:  # noqa: BLE001
            server = ""
        return AccountSnapshot(
            account=account,
            net_liquidation=_usd_tag(values, "NetLiquidation"),
            cash=_usd_tag(values, "TotalCashValue") or _usd_tag(values, "CashBalance"),
            buying_power=_usd_tag(values, "BuyingPower"),
            positions=positions,
            position_rows=rows,
            managed_accounts=managed,
            last_prices=last,
            open_order_symbols=self.open_order_symbols(),
            port=int(self.cfg.port),
            server_name=server,
        )

    def qualify_book(self) -> list[str]:
        ok: list[str] = []
        for sym in BOOK_SYMBOLS:
            try:
                self._stock(sym)
                ok.append(sym)
            except BrokerError:
                continue
        return ok

    def what_if(self, orders: list[OrderIntent], *, order_type: str, tif: str) -> dict[str, Any]:
        ib = self._require()
        details: list[dict[str, Any]] = []
        ok = True
        for intent in orders:
            contract = self._stock(intent.symbol)
            order, qty = _build_ib_order(intent, default_type=order_type, tif=tif)
            order.whatIf = True
            try:
                state = ib.whatIfOrder(contract, order)
            except Exception as exc:  # noqa: BLE001
                ok = False
                details.append({"symbol": intent.symbol, "error": str(exc), "qty": qty})
                continue
            warn = str(getattr(state, "warningText", "") or "")
            rec = {
                "symbol": intent.symbol,
                "qty": qty,
                "init_margin_change": getattr(state, "initMarginChange", None),
                "equity_with_loan_after": getattr(state, "equityWithLoanAfter", None),
                "warning": warn,
            }
            if warn and "error" in warn.lower():
                ok = False
            details.append(rec)
        return {"ok": ok, "detail": details, "n": len(orders)}

    def submit(self, orders: list[OrderIntent], *, order_type: str, tif: str) -> list[dict[str, Any]]:
        ib = self._require()
        self._last_trades = []
        out: list[dict[str, Any]] = []
        for intent in orders:
            contract = self._stock(intent.symbol)
            order, qty = _build_ib_order(intent, default_type=order_type, tif=tif)
            if qty <= 0:
                continue
            trade = ib.placeOrder(contract, order)
            self._last_trades.append(trade)
            status = str(getattr(trade.orderStatus, "status", "") or "Submitted")
            out.append(
                {
                    "symbol": intent.symbol,
                    "side": intent.side,
                    "qty": qty,
                    "order_type": order.orderType,
                    "status": status,
                    "order_id": getattr(trade.order, "orderId", None),
                    "filled": float(getattr(trade.orderStatus, "filled", 0.0) or 0.0),
                    "remaining": float(getattr(trade.orderStatus, "remaining", qty) or qty),
                    "avg_fill_price": float(getattr(trade.orderStatus, "avgFillPrice", 0.0) or 0.0),
                }
            )
        return out

    def wait_for_fills(
        self,
        reports: list[dict[str, Any]],
        *,
        timeout_s: float,
    ) -> tuple[list[dict[str, Any]], bool]:
        ib = self._require()
        trades = list(self._last_trades)
        if not trades:
            return reports, True
        moc = any(str(getattr(t.order, "orderType", "")).upper() == "MOC" for t in trades)
        deadline = time.time() + max(1.0, float(timeout_s))
        while time.time() < deadline:
            ib.sleep(0.4)
            if moc:
                if all(
                    str(getattr(t.orderStatus, "status", "")) in ACCEPTED_STATUSES | DONE_STATUSES
                    for t in trades
                ):
                    break
            elif all(t.isDone() for t in trades):
                break
        out: list[dict[str, Any]] = []
        all_ok = True
        for trade, rec in zip(trades, reports):
            st = str(getattr(trade.orderStatus, "status", "") or rec.get("status") or "")
            filled = float(getattr(trade.orderStatus, "filled", 0.0) or 0.0)
            remaining = float(getattr(trade.orderStatus, "remaining", 0.0) or 0.0)
            avg = float(getattr(trade.orderStatus, "avgFillPrice", 0.0) or 0.0)
            updated = {
                **rec,
                "status": st,
                "filled": filled,
                "remaining": remaining,
                "avg_fill_price": avg,
            }
            if moc:
                if st in REJECT_STATUSES or st not in ACCEPTED_STATUSES | DONE_STATUSES:
                    all_ok = False
            elif st != "Filled":
                all_ok = False
            out.append(updated)
        return out, all_ok

    def cancel_open_book_orders(self) -> list[str]:
        ib = self._require()
        cancelled: list[str] = []
        book = {s.upper() for s in BOOK_SYMBOLS}
        for trade in list(ib.openTrades() or []):
            sym = str(getattr(getattr(trade, "contract", None), "symbol", "") or "").upper()
            if sym not in book:
                continue
            try:
                ib.cancelOrder(trade.order)
                cancelled.append(sym)
            except Exception:  # noqa: BLE001
                continue
        if cancelled:
            ib.sleep(0.5)
        return cancelled
