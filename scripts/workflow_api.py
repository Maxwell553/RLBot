#!/usr/bin/env python3
"""Product-workflow API for investor mandates and Research Operations.

This service is intentionally separate from ``scripts/frontend_api.py``:

* workflow records live in transactional SQLite, never under ``Runs/``;
* every record is tenant-owned;
* IDs, lifecycle state, timestamps, quote, and payment state are server-owned;
* operator transitions are role-checked and append-only audit events are kept;
* payment verification only enters through the webhook endpoint.

Token authentication is suitable for local/internal deployments. A public
deployment should place this service behind an OIDC/session gateway and supply
the same actor claims from that trusted layer.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import os
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

_bootstrap_path = Path(__file__).resolve().parent / "_bootstrap.py"
_bootstrap_spec = importlib.util.spec_from_file_location("_rlbot_repo_bootstrap", _bootstrap_path)
assert _bootstrap_spec is not None and _bootstrap_spec.loader is not None
_bootstrap_mod = importlib.util.module_from_spec(_bootstrap_spec)
_bootstrap_spec.loader.exec_module(_bootstrap_mod)

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from rlbot.rl_config import load_config
from rlbot.run_artifacts import PROJECT_ROOT
from rlbot.workflow_store import (
    WorkflowActor,
    WorkflowConflict,
    WorkflowForbidden,
    WorkflowNotFound,
    WorkflowStore,
)


app = FastAPI(title="MarketTrainer workflow API", version="0.1.0", docs_url="/docs")
_store = WorkflowStore(
    os.environ.get("MARKETTRAINER_WORKFLOW_DB", str(PROJECT_ROOT / ".cache" / "workflow.sqlite3"))
)


_DEFAULT_LOCAL_TOKENS = {
    "investor-local": {"user_id": "investor_1", "org_id": "org_1", "role": "investor"},
    "operator-local": {"user_id": "operator_1", "org_id": "internal", "role": "operator"},
}


def _token_claims() -> dict[str, dict[str, str]]:
    """Bearer token → actor claims.

    When ``MARKETTRAINER_WORKFLOW_TOKENS`` is unset, fall back to the documented
    local-development tokens so investor/ops pages work out of the box.
    """
    raw = os.environ.get("MARKETTRAINER_WORKFLOW_TOKENS", "").strip()
    if not raw:
        return dict(_DEFAULT_LOCAL_TOKENS)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _actor(request: Request) -> WorkflowActor:
    supplied = request.headers.get("authorization", "")
    if not supplied.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required")
    claims = _token_claims().get(supplied.removeprefix("Bearer "))
    if not isinstance(claims, dict):
        raise HTTPException(status_code=401, detail="Invalid workflow token")
    user_id = claims.get("user_id")
    org_id = claims.get("org_id")
    role = claims.get("role")
    if not all(isinstance(value, str) and value for value in (user_id, org_id, role)):
        raise HTTPException(status_code=500, detail="Workflow token claims are malformed")
    return WorkflowActor(str(user_id), str(org_id), str(role))


def _translate_error(exc: Exception) -> HTTPException:
    if isinstance(exc, WorkflowNotFound):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, WorkflowForbidden):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, WorkflowConflict):
        return HTTPException(status_code=409, detail=str(exc))
    return HTTPException(status_code=500, detail="Workflow operation failed")


class InstrumentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=120)
    ticker: str = Field(..., min_length=1, max_length=20)
    group: str = Field(..., min_length=1, max_length=40)


class MandateSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=160)
    instruments: list[InstrumentInput] = Field(..., min_length=5, max_length=55)
    maxWeight: int = Field(..., ge=1, le=100)
    riskPreference: str = Field(..., pattern=r"^(defensive|balanced|growth)$")
    # Approximate portfolio notional used to size transaction-cost / slippage assumptions.
    approximateTradingCapital: int = Field(..., ge=25_000, le=5_000_000_000)


class WorkflowAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str = Field(..., min_length=1, max_length=80)
    detail: dict[str, Any] = Field(default_factory=dict)


class PaymentWebhook(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mandateId: str = Field(..., min_length=1, max_length=80)
    providerEventId: str = Field(..., min_length=1, max_length=160)
    paid: bool


def _approved_symbols() -> set[str]:
    cfg = load_config(PROJECT_ROOT / "config" / "config.yaml")
    return {str(symbol).upper() for symbol in cfg.universe.assets.values()}


def _market_history_bars(symbol: str) -> tuple[bool, int, str | None, str | None]:
    try:
        from curl_cffi import requests as curl_requests

        params = urllib.parse.urlencode({"range": "max", "interval": "1d", "events": "history"})
        response = curl_requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}?{params}",
            impersonate="chrome",
            timeout=15,
        )
        if response.status_code != 200:
            return False, 0, None, None
        result = (response.json().get("chart", {}).get("result") or [None])[0]
        timestamps = result.get("timestamp") if isinstance(result, dict) else None
        if not isinstance(timestamps, list) or not timestamps:
            return False, 0, None, None
        from datetime import datetime, timezone

        first = datetime.fromtimestamp(timestamps[0], timezone.utc).date().isoformat()
        last = datetime.fromtimestamp(timestamps[-1], timezone.utc).date().isoformat()
        return True, len(timestamps), first, last
    except Exception:  # noqa: BLE001 - network/data failures become a failed eligibility check
        return False, 0, None, None


def _quote_group(quote_type: str | None) -> str:
    return {
        "EQUITY": "Equity",
        "ETF": "Equity",
        "MUTUALFUND": "Equity",
        "INDEX": "Equity",
        "CURRENCY": "FX",
        "FUTURE": "Commodity",
        "COMMODITY": "Commodity",
        "CRYPTOCURRENCY": "Alternative",
    }.get((quote_type or "").upper(), "Alternative")


@functools.lru_cache(maxsize=256)
def _market_search(query: str, limit: int) -> tuple[tuple[Any, ...], ...]:
    try:
        from curl_cffi import requests as curl_requests

        params = urllib.parse.urlencode({"q": query, "quotesCount": limit, "newsCount": 0})
        response = curl_requests.get(
            f"https://query2.finance.yahoo.com/v1/finance/search?{params}",
            impersonate="chrome",
            timeout=15,
        )
        if response.status_code != 200:
            return ()
        quotes = response.json().get("quotes") or []
    except Exception:  # noqa: BLE001 - external lookup errors return no matches
        return ()
    rows: list[tuple[Any, ...]] = []
    for quote in quotes:
        if not isinstance(quote, dict) or not quote.get("symbol"):
            continue
        symbol = str(quote["symbol"]).upper()
        rows.append(
            (
                True,
                symbol,
                str(quote.get("longname") or quote.get("shortname") or symbol),
                _quote_group(str(quote.get("quoteType") or "")),
                str(quote.get("exchange") or quote.get("exchDisp") or "") or None,
                str(quote.get("currency")) if quote.get("currency") else None,
            )
        )
        if len(rows) >= limit:
            break
    return tuple(rows)


def _run_eligibility(record: dict[str, Any]) -> list[dict[str, Any]]:
    approved = _approved_symbols()
    instruments = list(record["instruments"])

    def check(instrument: dict[str, Any]) -> dict[str, Any]:
        ticker = str(instrument["ticker"]).upper()
        found, bars, first, last = _market_history_bars(ticker)
        policy_approved = ticker in approved
        sufficient_history = bars >= 2_500
        return {
            "ticker": ticker,
            "symbolFound": found,
            "historyBars": bars,
            "firstDate": first,
            "lastDate": last,
            "approvedPolicy": policy_approved,
            "panelCompatible": found,
            "sufficientHistory": sufficient_history,
            "eligible": found and policy_approved and sufficient_history,
        }

    with ThreadPoolExecutor(max_workers=min(8, len(instruments))) as executor:
        return list(executor.map(check, instruments))


@app.get("/api/session")
def session(actor: WorkflowActor = Depends(_actor)) -> dict[str, str]:
    return {"userId": actor.user_id, "organizationId": actor.org_id, "role": actor.role}


@app.get("/api/instruments/search")
def instrument_search(
    q: str = Query(..., min_length=2, max_length=40),
    limit: int = Query(default=8, ge=1, le=20),
    _actor_claims: WorkflowActor = Depends(_actor),
) -> dict[str, Any]:
    rows = _market_search(q.strip(), limit)
    return {
        "results": [
            {
                "found": row[0],
                "symbol": row[1],
                "name": row[2],
                "group": row[3],
                "exchange": row[4],
                "currency": row[5],
            }
            for row in rows
        ]
    }


@app.get("/api/mandates")
def mandates(actor: WorkflowActor = Depends(_actor)) -> dict[str, Any]:
    return {"mandates": _store.list(actor)}


@app.get("/api/mandates/{mandate_id}")
def mandate_detail(mandate_id: str, actor: WorkflowActor = Depends(_actor)) -> dict[str, Any]:
    try:
        return _store.get(actor, mandate_id)
    except Exception as exc:
        raise _translate_error(exc) from exc


@app.post("/api/mandates")
def submit_mandate(
    body: MandateSubmission,
    actor: WorkflowActor = Depends(_actor),
    idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=8, max_length=120),
) -> dict[str, Any]:
    payload = body.model_dump() if hasattr(body, "model_dump") else body.dict()
    configuration = {
        "maxWeight": payload.pop("maxWeight"),
        "riskPreference": payload.pop("riskPreference"),
        "approximateTradingCapital": payload.pop("approximateTradingCapital"),
        "longOnly": True,
        "cashAllowed": True,
        "decisionFrequency": "daily",
    }
    try:
        return _store.submit(
            actor,
            idempotency_key=idempotency_key,
            name=str(payload["name"]),
            instruments=list(payload["instruments"]),
            configuration=configuration,
        )
    except Exception as exc:
        raise _translate_error(exc) from exc


@app.post("/api/mandates/{mandate_id}/actions")
def mandate_action(
    mandate_id: str,
    body: WorkflowAction,
    actor: WorkflowActor = Depends(_actor),
) -> dict[str, Any]:
    try:
        if body.action == "cancel":
            return _store.cancel(actor, mandate_id, body.detail)
        if body.action == "run_preflight":
            record = _store.get(actor, mandate_id)
            return _store.record_preflight(actor, mandate_id, _run_eligibility(record))
        return _store.transition(actor, mandate_id, body.action, body.detail)
    except Exception as exc:
        raise _translate_error(exc) from exc


@app.post("/api/payments/webhook")
def payment_webhook(
    body: PaymentWebhook,
    webhook_secret: str | None = Header(default=None, alias="X-Workflow-Webhook-Secret"),
) -> dict[str, Any]:
    expected = os.environ.get("MARKETTRAINER_PAYMENT_WEBHOOK_SECRET", "")
    if not expected or webhook_secret != expected:
        raise HTTPException(status_code=401, detail="Invalid webhook secret")
    if not body.paid:
        return {"accepted": True, "stateChanged": False}
    try:
        record = _store.verify_payment(body.mandateId, body.providerEventId)
    except Exception as exc:
        raise _translate_error(exc) from exc
    return {"accepted": True, "stateChanged": True, "mandate": record}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--cors-origin", action="append", default=None)
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
        allow_headers=["authorization", "content-type", "idempotency-key", "Idempotency-Key"],
        expose_headers=["*"],
    )

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
