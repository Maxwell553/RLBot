#!/usr/bin/env python3
"""
Daily paper-trading pipeline for the RecurrentPPO (LSTM) portfolio model.

Run once after market close:
    cd paper_trade && python paper_trade.py

By default uses ``paper_trade/model/ppo_portfolio_final.zip`` and
``paper_trade/model/vec_normalize.pkl`` (deployed **25M_4_16_26_e**). Override with
``--run-id``, ``--model``, or ``--vec-normalize``. If those files are missing, falls back to
``runs/LATEST.txt``.

First run initialises a $10,000 cash portfolio. Each subsequent run:
  1. Fetches the latest 90 days of daily OHLCV from yfinance
  2. Builds the exact 98-dim observation the model was trained on
  3. Normalizes observations with VecNormalize stats (must match training)
  4. Runs the policy for target portfolio weights
  5. Simulates trade execution with realistic costs
  6. Persists state to state.json, appends to trade_journal.txt
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Repo root for importing shared env helpers (must match training)
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from data_utils import compute_fracdiff_panel
from run_artifacts import RunPaths, read_latest_run_id
from trading_env import portfolio_weights_from_action

# ---------------------------------------------------------------------------
# Constants (must match training exactly)
# ---------------------------------------------------------------------------

TICKERS = ["SP500", "GOLD", "OIL", "EURUSD", "USDJPY",
           "NIKKEI", "FTSE", "BOND10Y", "COPPER", "EM"]

YF_SYMBOLS: Dict[str, str] = {
    "SP500": "SPY", "GOLD": "GLD", "OIL": "USO",
    "EURUSD": "EURUSD=X", "USDJPY": "USDJPY=X",
    "NIKKEI": "^N225", "FTSE": "^FTSE",
    "BOND10Y": "IEF", "COPPER": "HG=F", "EM": "EEM",
}

MACRO_SYMBOLS: Dict[str, str] = {"DXY": "DX-Y.NYB", "TNX": "^TNX"}
MACRO_TICKERS = list(MACRO_SYMBOLS.keys())
N_MACRO = len(MACRO_TICKERS)

N_ASSETS = 10
N_ACTIONS = N_ASSETS + 1
LOOKBACK = 20
RETURN_HORIZONS = (1, 5, 10, 20)
FFILL_LIMIT = 5
OBS_LAG = 1
INITIAL_CASH = 10_000.0

ASSET_SLIPPAGE = np.array([
    0.0001, 0.0002, 0.0003, 0.0001, 0.0001,
    0.0005, 0.0005, 0.0001, 0.0008, 0.0002,
], dtype=np.float64)

ASSET_TX_FEE = np.array([
    0.0001, 0.0002, 0.0002, 0.00005, 0.00005,
    0.0010, 0.0010, 0.0001, 0.0005, 0.0002,
], dtype=np.float64)

ANNUAL_HOLDING_COST = np.array([
    0.0009, 0.0040, 0.0083, 0.0000, 0.0000,
    0.0010, 0.0010, 0.0015, 0.0060, 0.0067,
], dtype=np.float64)

DAILY_HOLDING_COST = ANNUAL_HOLDING_COST / 252.0

HERE = Path(__file__).resolve().parent
MODEL_ZIP = HERE / "model" / "ppo_portfolio_final.zip"
VN_PKL = HERE / "model" / "vec_normalize.pkl"
STATE_FILE = HERE / "state.json"
JOURNAL_FILE = HERE / "trade_journal.txt"

# Default bundle in ``model/`` is currently **25M_4_16_26_e** (best_model + vec_normalize).


def resolve_trade_artifacts(
    run_id: str,
    model_path: str,
    vec_path: str,
) -> tuple[Path, Path]:
    """``--model``; else ``--run-id``; else ``paper_trade/model/``; else ``runs/LATEST.txt``."""
    mp, vp = model_path.strip(), vec_path.strip()
    if vp and not mp:
        raise ValueError("Pass --model when using --vec-normalize, or use --run-id / defaults.")
    if mp:
        m = Path(mp).expanduser().resolve()
        if vp:
            v = Path(vp).expanduser().resolve()
        else:
            v = m.parent / "vec_normalize.pkl"
            if not v.is_file() and m.parent.name == "best":
                v = m.parent.parent / "vec_normalize.pkl"
        return m, v

    rid = run_id.strip()
    if rid:
        rp = RunPaths(rid)
        mz = rp.best_model_dir / "best_model.zip"
        vz = rp.models_dir / "vec_normalize.pkl"
        if not vz.is_file():
            alt = rp.best_model_dir / "vec_normalize.pkl"
            if alt.is_file():
                vz = alt
        if mz.is_file() and vz.is_file():
            return mz, vz
        raise FileNotFoundError(
            f"Missing model or vec_normalize for run {rid!r} (expected under models/{rid}/)."
        )

    # Prefer shipped bundle in paper_trade/model/ over LATEST.txt (stable paper-trading deploy)
    if MODEL_ZIP.is_file() and VN_PKL.is_file():
        return MODEL_ZIP.resolve(), VN_PKL.resolve()

    rid = read_latest_run_id() or ""
    if rid:
        rp = RunPaths(rid)
        mz = rp.best_model_dir / "best_model.zip"
        vz = rp.models_dir / "vec_normalize.pkl"
        if not vz.is_file():
            alt = rp.best_model_dir / "vec_normalize.pkl"
            if alt.is_file():
                vz = alt
        if mz.is_file() and vz.is_file():
            return mz, vz

    return MODEL_ZIP.resolve(), VN_PKL.resolve()

# ---------------------------------------------------------------------------
# Data fetching (mirrors data_utils.py)
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd_line(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def _yf_daily_history(yf_sym: str, since: str) -> pd.DataFrame:
    """
    Yahoo often returns empty JSON (then yfinance logs 'possibly delisted') if we hammer
    chart/v8 in a tight loop — that message is misleading. Space calls + retry.
    """
    import yfinance as yf

    last: pd.DataFrame = pd.DataFrame()
    for attempt in range(4):
        if attempt:
            time.sleep(min(1.5 * (2 ** (attempt - 1)), 12.0))
        t = yf.Ticker(yf_sym)
        last = t.history(start=since, interval="1d", auto_adjust=False, timeout=20)
        if not last.empty:
            return last
    return last


def fetch_recent(days: int = 90) -> Tuple[
    pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    """Fetch recent daily data for all assets + macro, return aligned arrays."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    per_ticker: Dict[str, pd.DataFrame] = {}

    pairs = list({**YF_SYMBOLS, **MACRO_SYMBOLS}.items())
    for i, (ticker, yf_sym) in enumerate(pairs):
        if i > 0:
            time.sleep(0.45)
        df = _yf_daily_history(yf_sym, since)
        if df.empty:
            print(f"  WARNING: no data for {ticker} ({yf_sym})")
            continue
        df = df.reset_index()
        time_col = df.columns[0]
        dates = pd.to_datetime(df[time_col])
        if dates.dt.tz is not None:
            dates = dates.dt.tz_localize(None)
        dates = dates.dt.normalize()

        if ticker in YF_SYMBOLS:
            asset_df = pd.DataFrame({
                "date": dates,
                f"{ticker}_open": df["Open"].to_numpy(dtype=np.float64),
                f"{ticker}_high": df["High"].to_numpy(dtype=np.float64),
                f"{ticker}_low": df["Low"].to_numpy(dtype=np.float64),
                f"{ticker}_close": df["Close"].to_numpy(dtype=np.float64),
                f"{ticker}_volume": df["Volume"].fillna(0.0).to_numpy(dtype=np.float64),
            })
        else:
            asset_df = pd.DataFrame({
                "date": dates,
                f"{ticker}_close": df["Close"].to_numpy(dtype=np.float64),
            })
        asset_df = asset_df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
        asset_df = asset_df[asset_df["date"].dt.dayofweek < 5].reset_index(drop=True)
        per_ticker[ticker] = asset_df

    merged = per_ticker[TICKERS[0]]
    for t in TICKERS[1:]:
        if t in per_ticker:
            merged = merged.merge(per_ticker[t], on="date", how="outer")
    for t in MACRO_TICKERS:
        if t in per_ticker:
            macro_cols = per_ticker[t][["date", f"{t}_close"]]
            merged = merged.merge(macro_cols, on="date", how="left")

    merged = merged.sort_values("date").reset_index(drop=True)

    for ticker in TICKERS:
        cols = [f"{ticker}_{c}" for c in ("open", "high", "low", "close", "volume")]
        existing = [c for c in cols if c in merged.columns]
        merged[existing] = merged[existing].ffill(limit=FFILL_LIMIT)
        vol_col = f"{ticker}_volume"
        if vol_col in merged.columns:
            merged[vol_col] = merged[vol_col].fillna(0.0)

    for ticker in MACRO_TICKERS:
        col = f"{ticker}_close"
        if col in merged.columns:
            merged[col] = merged[col].ffill().fillna(0.0)

    asset_cols = []
    for ticker in TICKERS:
        asset_cols.extend([f"{ticker}_{c}" for c in ("open", "high", "low", "close")])
    merged = merged.dropna(subset=asset_cols).reset_index(drop=True)

    merged = merged.sort_values("date").reset_index(drop=True)
    idx = pd.DatetimeIndex(merged["date"])

    ohlcv = np.zeros((len(merged), N_ASSETS, 5), dtype=np.float64)
    rsi_arr = np.zeros((len(merged), N_ASSETS), dtype=np.float64)
    macd_arr = np.zeros((len(merged), N_ASSETS), dtype=np.float64)
    for j, ticker in enumerate(TICKERS):
        ohlcv[:, j, 0] = merged[f"{ticker}_open"].to_numpy()
        ohlcv[:, j, 1] = merged[f"{ticker}_high"].to_numpy()
        ohlcv[:, j, 2] = merged[f"{ticker}_low"].to_numpy()
        ohlcv[:, j, 3] = merged[f"{ticker}_close"].to_numpy()
        ohlcv[:, j, 4] = merged[f"{ticker}_volume"].to_numpy()
        close_s = merged[f"{ticker}_close"]
        rsi_arr[:, j] = _rsi(close_s).fillna(50.0).to_numpy()
        ml = _macd_line(close_s)
        macd_arr[:, j] = (ml / (close_s.abs() + 1e-12)).fillna(0.0).to_numpy()
    ohlcv = np.nan_to_num(ohlcv, nan=0.0, posinf=0.0, neginf=0.0)
    ohlcv[:, :, :4] = np.maximum(ohlcv[:, :, :4], 1e-8)

    macro = np.zeros((len(merged), N_MACRO), dtype=np.float64)
    for j, ticker in enumerate(MACRO_TICKERS):
        col = f"{ticker}_close"
        if col in merged.columns:
            macro[:, j] = np.nan_to_num(merged[col].to_numpy(dtype=np.float64), nan=0.0)
            macro[:, j] = np.maximum(macro[:, j], 1e-8)

    fracdiff, fracdiff_macro = compute_fracdiff_panel(ohlcv, macro)
    return idx, ohlcv, rsi_arr, macd_arr, macro, fracdiff, fracdiff_macro


# ---------------------------------------------------------------------------
# Observation construction (mirrors trading_env._build_obs exactly)
# ---------------------------------------------------------------------------

def build_obs(
    ohlcv: np.ndarray,
    rsi: np.ndarray,
    macd: np.ndarray,
    macro: np.ndarray,
    fracdiff: np.ndarray,
    fracdiff_macro: np.ndarray,
    t: int,
    cash: float,
    units: np.ndarray,
    peak_nav: float,
    progress: float = 0.5,
    obs_lag: int = OBS_LAG,
) -> np.ndarray:
    """Build the 98-dim observation vector, identical to MultiAssetPortfolioEnv._build_obs.

    Market features use t_mkt = t - obs_lag so the agent never sees
    same-day close.  Portfolio weights and meta stay at current t.
    """
    t_mkt = max(t - obs_lag, 0)
    close = ohlcv[t, :, 3]
    nav = cash + float(np.dot(units, close))
    parts: List[np.ndarray] = []

    for h in RETURN_HORIZONS:
        t0 = max(t_mkt - h, 0)
        asset_fd = (fracdiff[t_mkt] - fracdiff[t0]).astype(np.float32)
        parts.append(asset_fd * 100.0)
        parts.append(np.array([asset_fd.mean()], dtype=np.float32) * 100.0)

    start = max(t_mkt - LOOKBACK, 1)
    closes_w = ohlcv[start: t_mkt + 1, :, 3]
    if len(closes_w) >= 2:
        vol = np.diff(np.log(closes_w + 1e-12), axis=0).std(axis=0)
    else:
        vol = np.zeros(N_ASSETS)
    parts.append(vol.astype(np.float32) * 100.0)
    parts.append(np.array([vol.mean()], dtype=np.float32) * 100.0)

    rsi_scaled = (rsi[t_mkt] / 50.0 - 1.0).astype(np.float32)
    parts.append(np.clip(rsi_scaled, -2.0, 2.0))

    macd_scaled = np.tanh(macd[t_mkt]).astype(np.float32)
    parts.append(macd_scaled)

    for h in RETURN_HORIZONS:
        t0 = max(t_mkt - h, 0)
        mfd = (fracdiff_macro[t_mkt] - fracdiff_macro[t0]).astype(np.float32)
        parts.append(mfd * 100.0)
    m_start = max(t_mkt - LOOKBACK, 1)
    m_vals = macro[m_start: t_mkt + 1]
    if len(m_vals) >= 2:
        m_vol = np.diff(np.log(m_vals + 1e-12), axis=0).std(axis=0)
    else:
        m_vol = np.zeros(N_MACRO)
    parts.append(m_vol.astype(np.float32) * 100.0)

    w = np.zeros(N_ACTIONS, dtype=np.float32)
    if nav > 1e-12:
        w[0] = cash / nav
        w[1:] = (units * close).astype(np.float32) / np.float32(nav)
    parts.append(w)

    dd = (nav - peak_nav) / max(peak_nav, 1e-12)
    parts.append(np.array([dd, progress], dtype=np.float32))

    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Trade execution (mirrors trading_env._rebalance + _apply_holding_costs)
# ---------------------------------------------------------------------------

def execute_rebalance(
    cash: float,
    units: np.ndarray,
    exec_price: np.ndarray,
    target_w: np.ndarray,
) -> Tuple[float, np.ndarray, List[dict]]:
    """Simulate rebalance at given execution prices with costs.

    In live trading exec_price should be the market open price (the first
    tradeable price after the overnight decision).
    Returns (new_cash, new_units, trades_list).
    """
    nav = cash + float(np.dot(units, exec_price))
    if nav <= 1e-12:
        return cash, units.copy(), []

    units = units.copy()
    target_units = (target_w[1:] * nav) / (exec_price + 1e-12)
    delta = target_units - units
    trades: List[dict] = []

    for i in np.argsort(delta):
        du = delta[i]
        if du >= -1e-12:
            continue
        sell_u = -du
        cost_rate = ASSET_SLIPPAGE[i] + ASSET_TX_FEE[i]
        proceeds = sell_u * exec_price[i] * (1.0 - cost_rate)
        cost = sell_u * exec_price[i] * cost_rate
        cash += proceeds
        units[i] -= sell_u
        trades.append({
            "action": "SELL", "asset": TICKERS[i],
            "units": float(sell_u), "price": float(exec_price[i]),
            "value": float(sell_u * exec_price[i]), "cost": float(cost),
        })

    for i in np.argsort(-delta):
        du = delta[i]
        if du <= 1e-12:
            continue
        cost_rate = ASSET_SLIPPAGE[i] + ASSET_TX_FEE[i]
        unit_cost = exec_price[i] * (1.0 + cost_rate)
        buy_u = min(du, max(0.0, cash / (unit_cost + 1e-12)))
        if buy_u <= 1e-12:
            continue
        spent = buy_u * unit_cost
        cost = buy_u * exec_price[i] * cost_rate
        cash -= spent
        units[i] += buy_u
        trades.append({
            "action": "BUY", "asset": TICKERS[i],
            "units": float(buy_u), "price": float(exec_price[i]),
            "value": float(buy_u * exec_price[i]), "cost": float(cost),
        })

    return cash, units, trades


def apply_holding_costs(cash: float, units: np.ndarray, close: np.ndarray) -> Tuple[float, float]:
    """Deduct daily holding costs. Returns (new_cash, total_cost)."""
    position_values = units * close
    notional = np.abs(position_values)
    daily_costs = notional * DAILY_HOLDING_COST
    total_cost = float(np.maximum(daily_costs, 0.0).sum())
    if total_cost > 0:
        cash -= total_cost
    return cash, total_cost


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_FILE.is_file():
        return json.loads(STATE_FILE.read_text())
    return {
        "start_date": None,
        "cash": INITIAL_CASH,
        "units": [0.0] * N_ASSETS,
        "peak_nav": INITIAL_CASH,
        "nav_history": [],
        "trade_count": 0,
    }


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")


# ---------------------------------------------------------------------------
# VecNormalize observation normalization
# ---------------------------------------------------------------------------

def load_obs_normalizer(vec_pkl: Path) -> Tuple:
    """Load VecNormalize stats and return (mean, var) for observation normalization."""
    if not vec_pkl.is_file():
        raise FileNotFoundError(
            f"VecNormalize stats not found: {vec_pkl}\n"
            "Copy vec_normalize.pkl from models/<run_id>/ after training, or pass --vec-normalize."
        )
    with open(vec_pkl, "rb") as f:
        vn = pickle.load(f)
    return vn.obs_rms.mean, vn.obs_rms.var


def normalize_obs(obs: np.ndarray, mean: np.ndarray, var: np.ndarray,
                  clip: float = 10.0) -> np.ndarray:
    eps = 1e-8
    normed = (obs - mean) / np.sqrt(var + eps)
    return np.clip(normed, -clip, clip).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Daily paper-trading pipeline")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what the model would do without updating state")
    parser.add_argument("--force", action="store_true",
                        help="Run even if today already has a journal entry")
    parser.add_argument(
        "--run-id", default="",
        help="Use models/<ID>/best/best_model.zip and vec_normalize from that training run",
    )
    parser.add_argument("--model", default="", help="Path to policy .zip (overrides --run-id)")
    parser.add_argument(
        "--vec-normalize", default="",
        help="Path to vec_normalize.pkl (overrides --run-id)",
    )
    args = parser.parse_args()

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    state = load_state()
    if state["start_date"] is None:
        state["start_date"] = today

    already_ran = any(
        entry.get("date") == today for entry in state.get("nav_history", [])
    )
    if already_ran and not args.force and not args.dry_run:
        print(f"Already ran for {today}. Use --force to override.")
        return

    model_zip, vn_pkl = resolve_trade_artifacts(args.run_id, args.model, args.vec_normalize)
    print(f"Paper Trade — {today}")
    print(f"  model: {model_zip}")
    print(f"  vec_normalize: {vn_pkl}")
    print(f"{'(DRY RUN) ' if args.dry_run else ''}Fetching market data...")

    idx, ohlcv, rsi, macd, macro, fracdiff, fracdiff_macro = fetch_recent(days=90)
    min_bars = LOOKBACK + OBS_LAG + 1
    if len(idx) < min_bars:
        print(f"ERROR: Only {len(idx)} bars fetched, need at least {min_bars}.")
        sys.exit(1)

    t = len(idx) - 1
    latest_date = str(idx[t].date())
    print(f"  Latest bar: {latest_date} ({len(idx)} bars fetched)")

    close = ohlcv[t, :, 3]
    open_price = ohlcv[t, :, 0]
    cash = state["cash"]
    units = np.array(state["units"], dtype=np.float64)
    peak_nav = state["peak_nav"]
    nav_before = cash + float(np.dot(units, close))

    obs = build_obs(
        ohlcv, rsi, macd, macro, fracdiff, fracdiff_macro,
        t, cash, units, peak_nav,
    )
    assert obs.shape == (98,), f"Observation shape mismatch: {obs.shape}"

    obs_mean, obs_var = load_obs_normalizer(vn_pkl)
    obs_normed = normalize_obs(obs, obs_mean, obs_var)

    from sb3_contrib import RecurrentPPO
    model = RecurrentPPO.load(str(model_zip), device="cpu")
    action, _ = model.predict(
        obs_normed.reshape(1, -1),
        state=None,
        episode_start=np.ones((1,), dtype=bool),
        deterministic=True,
    )
    action = np.asarray(action).reshape(-1)

    target_w = portfolio_weights_from_action(action.astype(np.float64))

    # Execute at open price (mirrors training env: decide after close[t-1],
    # fill at next open).  In live trading this would be today's open.
    new_cash, new_units, trades = execute_rebalance(cash, units, open_price, target_w)
    new_cash, holding_cost = apply_holding_costs(new_cash, new_units, close)

    nav_after = new_cash + float(np.dot(new_units, close))
    new_peak = max(peak_nav, nav_after)
    total_return = (nav_after / INITIAL_CASH - 1.0) * 100
    day_number = len(state["nav_history"]) + 1

    weight_strs = []
    weight_strs.append(f"CASH {target_w[0]*100:5.1f}%")
    for i, ticker in enumerate(TICKERS):
        weight_strs.append(f"{ticker} {target_w[i+1]*100:5.1f}%")

    lines = []
    lines.append(f"{'=' * 60}")
    lines.append(f"=== {latest_date} (Day {day_number}) ===")
    lines.append(f"NAV: ${nav_before:,.2f} | Cash: ${cash:,.2f}")
    lines.append(f"")
    lines.append(f"Current holdings:")
    for i, ticker in enumerate(TICKERS):
        if units[i] > 1e-8:
            val = units[i] * close[i]
            lines.append(f"  {ticker:>8s}: {units[i]:>10.4f} units @ ${close[i]:>10.2f} = ${val:>10.2f}")
    if all(u < 1e-8 for u in units):
        lines.append(f"  (all cash)")
    lines.append(f"")
    lines.append(f"Target allocation: {' | '.join(weight_strs)}")
    lines.append(f"")
    if trades:
        lines.append(f"Trades:")
        total_cost = 0.0
        for tr in trades:
            total_cost += tr["cost"]
            lines.append(
                f"  {tr['action']:4s} {tr['asset']:>8s}  "
                f"${tr['value']:>9,.2f}  "
                f"({tr['units']:.4f} units @ ${tr['price']:.2f}, "
                f"cost ${tr['cost']:.2f})"
            )
        lines.append(f"  Transaction costs: ${total_cost:.2f}")
        lines.append(f"  Holding costs:     ${holding_cost:.4f}")
    else:
        lines.append(f"Trades: NONE (portfolio matches target)")
    lines.append(f"")
    lines.append(f"Post-trade NAV: ${nav_after:,.2f} | Cash: ${new_cash:,.2f}")
    lines.append(f"Cumulative return: {total_return:+.2f}% (from ${INITIAL_CASH:,.0f})")
    lines.append(f"Peak NAV: ${new_peak:,.2f} | Drawdown: {(nav_after - new_peak) / new_peak * 100:.2f}%")
    lines.append(f"{'=' * 60}")

    output = "\n".join(lines)
    print(output)

    if not args.dry_run:
        state["cash"] = new_cash
        state["units"] = new_units.tolist()
        state["peak_nav"] = new_peak
        state["trade_count"] += len(trades)
        state["nav_history"].append({
            "date": latest_date,
            "nav": round(nav_after, 2),
            "cash": round(new_cash, 2),
            "return_pct": round(total_return, 4),
        })
        save_state(state)

        with open(JOURNAL_FILE, "a") as f:
            f.write(output + "\n\n")

        print(f"\nState saved to {STATE_FILE.name}")
        print(f"Journal appended to {JOURNAL_FILE.name}")
    else:
        print(f"\n(Dry run — no state changes written)")


if __name__ == "__main__":
    main()
