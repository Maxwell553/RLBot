"""
Yahoo symbols for *executed* instruments (dollar/price = shares) when IB market data
is not subscribed. Model features still use training tickers; proxies match contracts.json.
"""

from __future__ import annotations

# Logical name -> yfinance symbol (same as paper for direct maps; EWJ/EWU/CPER for execution).
SIZING_YF: dict[str, str] = {
    "SP500": "SPY",
    "GOLD": "GLD",
    "OIL": "USO",
    "EURUSD": "EURUSD=X",
    "USDJPY": "USDJPY=X",
    "NIKKEI": "EWJ",
    "FTSE": "EWU",
    "BOND10Y": "IEF",
    "COPPER": "CPER",
    "EM": "EEM",
}
