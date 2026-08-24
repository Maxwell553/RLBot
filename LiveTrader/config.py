"""LiveTrader config: YAML + env. Live submit stays off unless every gate is set."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from book import LIVE_PORTS, PAPER_PORTS

LIVE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = LIVE_DIR / "config.yaml"


@dataclass(frozen=True)
class LiveConfig:
    host: str
    port: int
    client_id: int
    account: str
    mode: str
    order_type: str
    fractional_order_type: str
    tif: str
    allow_live: bool
    allow_fractional: bool
    min_notional: float
    allow_foreign_positions: bool
    seed_if_flat: bool
    sleeve_split: bool
    cap_buys_to_cash: bool
    require_account: bool
    fill_timeout_s: float
    confirm_phrase: str
    aum_override: float | None
    market_data_type: int
    connect_timeout_s: float
    mkt_data_wait_s: float
    data_source: str
    yahoo_fallback: bool

    @property
    def is_paper_port(self) -> bool:
        return int(self.port) in PAPER_PORTS

    @property
    def is_live_port(self) -> bool:
        return int(self.port) in LIVE_PORTS

    @property
    def whole_share_order_type(self) -> str:
        return str(self.order_type or "MOC").upper()


def _float_or_none(v: Any) -> float | None:
    if v is None or v == "":
        return None
    return float(v)


def load_config(path: Path | None = None) -> LiveConfig:
    cfg_path = path or Path(os.environ.get("LIVE_TRADER_CONFIG") or DEFAULT_CONFIG)
    raw: dict[str, Any] = {}
    if cfg_path.is_file():
        loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"config must be a mapping: {cfg_path}")
        raw = loaded
    ib = dict(raw.get("ibkr") or {})
    ex = dict(raw.get("execution") or {})
    safety = dict(raw.get("safety") or {})
    data = dict(raw.get("data") or {})
    mode = str(os.environ.get("LIVE_TRADER_MODE") or ex.get("mode") or "dry_run").strip().lower()
    if mode not in {"dry_run", "paper", "live"}:
        raise ValueError(f"execution.mode must be dry_run|paper|live, got {mode!r}")
    source = str(os.environ.get("LIVE_TRADER_DATA") or data.get("source") or "ibkr").strip().lower()
    if source not in {"ibkr", "yahoo"}:
        raise ValueError(f"data.source must be ibkr|yahoo, got {source!r}")
    fb_env = os.environ.get("LIVE_TRADER_YAHOO_FALLBACK")
    if fb_env is None or fb_env == "":
        yahoo_fallback = bool(data.get("yahoo_fallback", True))
    else:
        yahoo_fallback = str(fb_env).strip().lower() not in {"0", "false", "no"}
    aum = _float_or_none(os.environ.get("LIVE_TRADER_AUM") or raw.get("aum_override"))
    return LiveConfig(
        host=str(os.environ.get("IBKR_HOST") or ib.get("host") or "127.0.0.1"),
        port=int(os.environ.get("IBKR_PORT") or ib.get("port") or 7497),
        client_id=int(os.environ.get("IBKR_CLIENT_ID") or ib.get("client_id") or 17),
        account=str(os.environ.get("IBKR_ACCOUNT") or ib.get("account") or "").strip(),
        mode=mode,
        order_type=str(ex.get("order_type") or "MOC").strip().upper(),
        fractional_order_type=str(ex.get("fractional_order_type") or "MKT").strip().upper(),
        tif=str(ex.get("tif") or "DAY").strip().upper(),
        allow_live=bool(ex.get("allow_live", False)),
        allow_fractional=bool(ex.get("allow_fractional", True)),
        min_notional=float(ex.get("min_notional") or 1.0),
        allow_foreign_positions=bool(ex.get("allow_foreign_positions", False)),
        seed_if_flat=bool(ex.get("seed_if_flat", True)),
        sleeve_split=bool(ex.get("sleeve_split", True)),
        cap_buys_to_cash=bool(ex.get("cap_buys_to_cash", True)),
        require_account=bool(ex.get("require_account", True)),
        fill_timeout_s=float(ex.get("fill_timeout_s") or 120.0),
        confirm_phrase=str(
            os.environ.get("LIVE_TRADER_CONFIRM") or safety.get("confirm_phrase") or "CORE"
        ).strip(),
        aum_override=aum,
        market_data_type=int(ib.get("market_data_type") or 1),
        connect_timeout_s=float(ib.get("connect_timeout_s") or 10.0),
        mkt_data_wait_s=float(ib.get("mkt_data_wait_s") or 1.5),
        data_source=source,
        yahoo_fallback=yahoo_fallback,
    )
