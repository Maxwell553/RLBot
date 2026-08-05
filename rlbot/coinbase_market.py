"""Coinbase market data for crypto forward marks.

Public Exchange ticker / candles work without credentials.

Supported credentials (never commit these):

* **CDP / Advanced Trade** (what Coinbase Pro developer portal issues today):
  ``COINBASE_API_KEY_NAME`` + ``COINBASE_API_PRIVATE_KEY`` (EC PEM).
* **Legacy Exchange** HMAC: ``COINBASE_API_KEY`` + ``COINBASE_API_SECRET`` +
  ``COINBASE_API_PASSPHRASE``.

Equity ETFs (TQQQ, QQQ, GLD, SPY, …) are not Coinbase products; those marks
stay on Yahoo. CrestDay / Durable.v1 allocation prices use this module.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from base64 import b64decode, b64encode
from typing import Any

import pandas as pd

from rlbot.run_artifacts import PROJECT_ROOT

EXCHANGE_REST = "https://api.exchange.coinbase.com"
ADVANCED_REST = "https://api.coinbase.com"
DEFAULT_TIMEOUT_S = 8.0

# Common pack / intent symbols → Coinbase product ids.
_PRODUCT_ALIASES: dict[str, str] = {
    "BTC": "BTC-USD",
    "BTCUSD": "BTC-USD",
    "BTCUSDT": "BTC-USD",
    "XBT": "BTC-USD",
    "ETH": "ETH-USD",
    "ETHUSD": "ETH-USD",
    "ETHUSDT": "ETH-USD",
    "SOL": "SOL-USD",
    "SOLUSD": "SOL-USD",
    "SOLUSDT": "SOL-USD",
    "XRP": "XRP-USD",
    "DOGE": "DOGE-USD",
    "ADA": "ADA-USD",
    "LINK": "LINK-USD",
    "AVAX": "AVAX-USD",
    "MATIC": "MATIC-USD",
    "DOT": "DOT-USD",
    "LTC": "LTC-USD",
    "BCH": "BCH-USD",
}

_ENV_LOADED = False


def _load_dotenv_once() -> None:
    """Pull Coinbase keys from repo ``.env`` if the process wasn't started with them."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    path = PROJECT_ROOT / ".env"
    if not path.is_file():
        return
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            if not key.startswith("COINBASE_"):
                continue
            if os.environ.get(key, "").strip():
                continue
            val = val.strip().strip('"').strip("'")
            # PEM private keys often stored with literal \n
            if "PRIVATE_KEY" in key:
                val = val.replace("\\n", "\n")
            os.environ[key] = val
    except OSError:
        return


def exchange_auth_configured() -> bool:
    _load_dotenv_once()
    return bool(
        os.environ.get("COINBASE_API_KEY", "").strip()
        and os.environ.get("COINBASE_API_SECRET", "").strip()
        and os.environ.get("COINBASE_API_PASSPHRASE", "").strip()
    )


def cdp_auth_configured() -> bool:
    _load_dotenv_once()
    return bool(
        os.environ.get("COINBASE_API_KEY_NAME", "").strip()
        and os.environ.get("COINBASE_API_PRIVATE_KEY", "").strip()
    )


def auth_configured() -> bool:
    return exchange_auth_configured() or cdp_auth_configured()


def to_product_id(symbol: str) -> str | None:
    raw = str(symbol or "").strip().upper().replace("/", "").replace("-", "")
    if not raw or raw in {"CASH", "USD", "USDT", "USDC"}:
        return None
    if raw in _PRODUCT_ALIASES:
        return _PRODUCT_ALIASES[raw]
    if raw.endswith("USDT") and len(raw) > 4:
        return f"{raw[:-4]}-USD"
    if raw.endswith("USD") and len(raw) > 3:
        return f"{raw[:-3]}-USD"
    if "-" in str(symbol):
        return str(symbol).strip().upper()
    if raw.isalpha() and 2 <= len(raw) <= 10:
        return f"{raw}-USD"
    return None


def _exchange_signed_headers(method: str, request_path: str, body: str = "") -> dict[str, str]:
    key = os.environ.get("COINBASE_API_KEY", "").strip()
    secret = os.environ.get("COINBASE_API_SECRET", "").strip()
    passphrase = os.environ.get("COINBASE_API_PASSPHRASE", "").strip()
    if not (key and secret and passphrase):
        return {}
    timestamp = str(int(time.time()))
    message = f"{timestamp}{method.upper()}{request_path}{body}".encode("utf-8")
    digest = hmac.new(b64decode(secret), message, hashlib.sha256).digest()
    return {
        "CB-ACCESS-KEY": key,
        "CB-ACCESS-SIGN": b64encode(digest).decode("utf-8"),
        "CB-ACCESS-TIMESTAMP": timestamp,
        "CB-ACCESS-PASSPHRASE": passphrase,
        "Content-Type": "application/json",
    }


def _cdp_jwt(method: str, host: str, path: str) -> str | None:
    """Build a short-lived CDP JWT for Advanced Trade REST (ES256)."""
    _load_dotenv_once()
    key_name = os.environ.get("COINBASE_API_KEY_NAME", "").strip()
    private_key = os.environ.get("COINBASE_API_PRIVATE_KEY", "").strip().replace("\\n", "\n")
    if not (key_name and private_key):
        return None
    try:
        import jwt  # PyJWT
    except ImportError:
        return None
    now = int(time.time())
    uri = f"{method.upper()} {host}{path}"
    payload = {
        "sub": key_name,
        "iss": "cdp",
        "nbf": now,
        "exp": now + 120,
        "uri": uri,
    }
    headers = {"kid": key_name, "nonce": secrets.token_hex(16), "typ": "JWT"}
    try:
        return jwt.encode(payload, private_key, algorithm="ES256", headers=headers)
    except Exception:  # noqa: BLE001
        return None


def _http_get_json(
    base: str,
    path: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    headers: dict[str, str] | None = None,
) -> Any:
    url = f"{base}{path}"
    hdrs = {"User-Agent": "MarketTrainer-forward/1.0", "Accept": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def fetch_ticker_price(product_id: str, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> float | None:
    _load_dotenv_once()
    # 1) Public Exchange ticker (no auth required).
    path = f"/products/{urllib.parse.quote(product_id)}/ticker"
    headers = _exchange_signed_headers("GET", path) if exchange_auth_configured() else {}
    payload = _http_get_json(EXCHANGE_REST, path, timeout_s=timeout_s, headers=headers or None)
    if isinstance(payload, dict):
        try:
            px = float(payload.get("price") or payload.get("last") or 0.0)
            if px > 0:
                return px
        except (TypeError, ValueError):
            pass

    # 2) Advanced Trade public/auth product ticker via CDP JWT when configured.
    adv_path = f"/api/v3/brokerage/market/products/{urllib.parse.quote(product_id)}"
    adv_headers: dict[str, str] = {}
    token = _cdp_jwt("GET", "api.coinbase.com", adv_path) if cdp_auth_configured() else None
    if token:
        adv_headers["Authorization"] = f"Bearer {token}"
    adv = _http_get_json(ADVANCED_REST, adv_path, timeout_s=timeout_s, headers=adv_headers or None)
    if isinstance(adv, dict):
        try:
            # market product endpoint returns price on the product object
            px = float(
                adv.get("price")
                or (adv.get("product") or {}).get("price")
                or 0.0
            )
            if px > 0:
                return px
        except (TypeError, ValueError):
            pass
    return None


def fetch_last_prices(
    symbols: list[str],
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, float]:
    """Map original symbols → last USD spot."""
    out: dict[str, float] = {}
    seen: dict[str, float] = {}
    for sym in symbols:
        product = to_product_id(sym)
        if not product:
            continue
        if product in seen:
            out[str(sym)] = seen[product]
            continue
        px = fetch_ticker_price(product, timeout_s=timeout_s)
        if px is None:
            continue
        seen[product] = px
        out[str(sym)] = px
    return out


def fetch_candles_5m(
    product_id: str,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> pd.DataFrame:
    """Return OHLC indexed by naive UTC bar open (granularity 300s)."""
    _load_dotenv_once()
    params: dict[str, str] = {"granularity": "300"}
    if start is not None:
        params["start"] = pd.Timestamp(start, tz="UTC").isoformat()
    if end is not None:
        params["end"] = pd.Timestamp(end, tz="UTC").isoformat()
    qs = urllib.parse.urlencode(params)
    path = f"/products/{urllib.parse.quote(product_id)}/candles?{qs}"
    headers = _exchange_signed_headers("GET", path) if exchange_auth_configured() else {}
    payload = _http_get_json(EXCHANGE_REST, path, timeout_s=timeout_s, headers=headers or None)
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame()
    rows = []
    for row in payload:
        if not isinstance(row, (list, tuple)) or len(row) < 5:
            continue
        try:
            ts = pd.Timestamp(int(row[0]), unit="s", tz="UTC").tz_localize(None)
            rows.append(
                {
                    "_t": ts,
                    "Open": float(row[3]),
                    "High": float(row[2]),
                    "Low": float(row[1]),
                    "Close": float(row[4]),
                }
            )
        except (TypeError, ValueError):
            continue
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).set_index("_t").sort_index()
    return frame[["Open", "High", "Low", "Close"]]
