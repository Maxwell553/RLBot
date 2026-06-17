"""Causal execution invariants of MultiAssetPortfolioEnv:
observe close[t - obs_lag] → decide → holding cost on pre-rebalance units at close[t]
→ fill at open[t+1] → mark-to-market at close[t+1]. Torch-free."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import pytest

from rlbot.rl_config import get_config, set_config
from rlbot.trading_env import MultiAssetPortfolioEnv, portfolio_weights_from_action

LOOKBACK = 5


@contextmanager
def _installed_config(
    n_assets: int,
    *,
    slippage: float = 0.0,
    tx_fee: float = 0.0,
    annual_holding: float = 0.0,
    cap: float = 1.0,
):
    """Install a config whose per-asset cost arrays match a synthetic n_assets panel."""
    base = get_config()
    patched = replace(
        base,
        environment=replace(
            base.environment,
            max_single_asset_weight=cap,
            action_smoothing_alpha=0.0,
        ),
        reward=replace(
            base.reward,
            benchmark_cap_weights=[1.0 / n_assets] * n_assets,
            cash_daily_yield=0.0,
        ),
        transaction_costs=replace(
            base.transaction_costs,
            slippage=[slippage] * n_assets,
            tx_fee=[tx_fee] * n_assets,
            annual_holding_cost=[annual_holding] * n_assets,
        ),
    )
    set_config(patched)
    try:
        yield patched
    finally:
        set_config(base)


def _feature_panels(t: int, n_assets: int):
    rsi = np.full((t, n_assets), 50.0)
    macd = np.zeros((t, n_assets))
    fracdiff = np.zeros((t, n_assets))
    fracdiff_macro = np.zeros((t, 4))
    trend = np.zeros((t, n_assets))
    macro = np.full((t, 4), 10.0)
    return rsi, macd, fracdiff, fracdiff_macro, trend, macro


def _make_env(ohlcv: np.ndarray, *, obs_lag_default: int = 1, rsi: np.ndarray | None = None):
    t, n_assets = ohlcv.shape[0], ohlcv.shape[1]
    rsi_p, macd, fd, fdm, trend, macro = _feature_panels(t, n_assets)
    return MultiAssetPortfolioEnv(
        ohlcv,
        rsi if rsi is not None else rsi_p,
        macd,
        macro=macro,
        fracdiff=fd,
        fracdiff_macro=fdm,
        trend=trend,
        random_start=False,
        domain_randomize=False,
        lookback=LOOKBACK,
        obs_lag_default=obs_lag_default,
        action_smoothing_alpha=0.0,
    )


def _single_asset_prices(t: int = 20) -> np.ndarray:
    """Deterministic prices where open[k] != close[k] != close[k-1] (model-discriminating)."""
    ohlcv = np.zeros((t, 1, 5), dtype=np.float64)
    close = 100.0 + 2.0 * np.arange(t)
    open_ = close + 1.0
    ohlcv[:, 0, 0] = open_
    ohlcv[:, 0, 1] = np.maximum(open_, close) + 0.5
    ohlcv[:, 0, 2] = np.minimum(open_, close) - 0.5
    ohlcv[:, 0, 3] = close
    ohlcv[:, 0, 4] = 1e6
    return ohlcv


# ── (a) observation lag ──────────────────────────────────────────────────
def test_observation_reflects_market_features_at_t_minus_lag() -> None:
    """rsi block of the obs at time t equals the (scaled) rsi panel row t - obs_lag."""
    n_assets, t_bars = 3, 60
    with _installed_config(n_assets):
        ohlcv = np.zeros((t_bars, n_assets, 5), dtype=np.float64)
        ohlcv[:, :, :4] = 100.0
        ohlcv[:, :, 4] = 1e6
        # encode (bar index, asset) into rsi: scaled value = (t + j/10) / 1000
        rsi = 50.0 * (
            1.0
            + (np.arange(t_bars)[:, None] + np.arange(n_assets)[None, :] / 10.0) / 1000.0
        )
        n_h = len(MultiAssetPortfolioEnv.RETURN_HORIZONS)
        rsi_offset = n_h * (n_assets + 1) + (n_assets + 1)  # fracdiff blocks + vol block

        for lag in (0, 1, 2):
            env = _make_env(ohlcv, obs_lag_default=lag, rsi=rsi)
            obs, _ = env.reset()
            t0 = env._t
            assert env.obs_lag == lag
            expected = (rsi[t0 - lag] / 50.0 - 1.0).astype(np.float32)
            np.testing.assert_allclose(
                obs[rsi_offset : rsi_offset + n_assets], expected, rtol=0, atol=1e-6
            )
            # after one step (t advances by 1) the lagged row advances by 1 too
            obs2, _, _, _, _ = env.step(np.zeros(env.action_space.shape))
            expected2 = (rsi[t0 + 1 - lag] / 50.0 - 1.0).astype(np.float32)
            np.testing.assert_allclose(
                obs2[rsi_offset : rsi_offset + n_assets], expected2, rtol=0, atol=1e-6
            )
            # the lag is real: a lag-L obs at bar t equals neither row t nor row t-L-1
            if lag > 0:
                wrong_now = (rsi[t0] / 50.0 - 1.0).astype(np.float32)
                assert not np.allclose(obs[rsi_offset : rsi_offset + n_assets], wrong_now)


# ── (b) fill at open[t+1], mark at close[t+1] ────────────────────────────
def test_fill_at_next_open_mark_at_next_close() -> None:
    with _installed_config(1):
        ohlcv = _single_asset_prices()
        env = _make_env(ohlcv)
        env.reset()
        t = env._t
        nav0 = env.initial_cash
        action = np.array([-5.0, 5.0])
        live = env.asset_live[max(t - env.obs_lag, 0)]
        w = portfolio_weights_from_action(action, n_actions=2, asset_live=live)
        w1 = float(w[1])
        assert w1 > 0.99  # nearly fully invested

        _, _, _, _, info = env.step(action)

        open_next = ohlcv[t + 1, 0, 0]
        close_next = ohlcv[t + 1, 0, 3]
        close_t = ohlcv[t, 0, 3]
        # buy w1·nav at open[t+1] (zero costs), mark at close[t+1]
        expected = nav0 * (1.0 - w1) + w1 * nav0 * close_next / open_next
        assert info["nav"] == pytest.approx(expected, rel=1e-12)
        # units actually purchased at open[t+1]
        assert env._units[0] == pytest.approx(w1 * nav0 / open_next, rel=1e-12)

        # discriminate against wrong execution models
        fill_at_close_t = nav0 * (1.0 - w1) + w1 * nav0 * close_next / close_t
        mark_at_open_next = nav0  # buy and mark at the same open
        assert abs(info["nav"] - fill_at_close_t) > 1e-3
        assert abs(info["nav"] - mark_at_open_next) > 1e-3


def test_sell_fills_at_next_open() -> None:
    with _installed_config(1):
        ohlcv = _single_asset_prices()
        env = _make_env(ohlcv)
        env.reset()
        t0 = env._t
        buy = np.array([-5.0, 5.0])
        env.step(buy)
        u = float(env._units[0])
        cash_before = float(env._cash)

        # saturated logits: softmax residual on the risky leg is ~1e-26 (true sell-all)
        sell_all = np.array([60.0, -60.0])
        _, _, _, _, info = env.step(sell_all)
        open_next = ohlcv[t0 + 2, 0, 0]  # second step trades at open[t0+2]
        # all units liquidated at open[t+1] with zero costs; NAV is then pure cash
        assert env._units[0] == pytest.approx(0.0, abs=1e-9)
        assert info["nav"] == pytest.approx(cash_before + u * open_next, rel=1e-12)


# ── (c) holding cost on pre-rebalance units at close[t] ──────────────────
def test_holding_cost_on_pre_rebalance_units_at_close_t() -> None:
    days = int(get_config().transaction_costs.trading_days_per_year)
    annual = 0.252
    daily = annual / days
    ohlcv = _single_asset_prices()
    navs = {}
    units = {}
    for label, hold in (("free", 0.0), ("costly", annual)):
        with _installed_config(1, annual_holding=hold):
            env = _make_env(ohlcv)
            env.reset()
            t0 = env._t
            env.step(np.array([-5.0, 5.0]))  # buy: no units yet → no holding cost
            units[label] = float(env._units[0])
            # saturated sell-all so the residual softmax leg (~1e-26) cannot blur the diff
            _, _, _, _, info = env.step(np.array([60.0, -60.0]))
            navs[label] = float(info["nav"])
    # identical fills (holding cost never applied to an empty book on step 1)
    assert units["free"] == pytest.approx(units["costly"], rel=1e-12)
    u = units["free"]
    close_t_step2 = ohlcv[t0 + 1, 0, 3]  # close of the bar at which step 2 begins
    expected_cost = u * close_t_step2 * daily
    assert expected_cost > 0.0
    assert navs["free"] - navs["costly"] == pytest.approx(expected_cost, rel=1e-9)
    # and NOT charged at the rebalance price open[t+1] of step 2
    wrong_cost = u * ohlcv[t0 + 2, 0, 0] * daily
    assert abs((navs["free"] - navs["costly"]) - wrong_cost) > 1e-6


# ── (d) agent / benchmark execution alignment (H1) ───────────────────────
def _multi_asset_prices(t: int = 30, n_assets: int = 4) -> np.ndarray:
    """Distinct per-asset open/close paths so the friction model is discriminating."""
    rng = np.random.default_rng(0)
    ohlcv = np.zeros((t, n_assets, 5), dtype=np.float64)
    for j in range(n_assets):
        steps = 1.0 + 0.3 * (j + 1) + rng.normal(0.0, 0.5, size=t)
        close = 100.0 + np.cumsum(np.abs(steps))
        open_ = close * (1.0 + 0.001 * (j + 1))  # open != prior close, asset-specific gap
        ohlcv[:, j, 0] = open_
        ohlcv[:, j, 1] = np.maximum(open_, close) + 0.5
        ohlcv[:, j, 2] = np.minimum(open_, close) - 0.5
        ohlcv[:, j, 3] = close
        ohlcv[:, j, 4] = 1e6
    return ohlcv


def test_agent_holding_benchmark_weights_has_zero_excess() -> None:
    """An agent that holds the benchmark weights reproduces the benchmark NAV path, so
    per-step excess (log_ret - market_ret) is ~0. This is the H1 alignment guarantee:
    the in-env benchmark uses the same unit-level execution engine as the agent."""
    n_assets = 4
    # nonzero, per-asset-asymmetric costs so any model mismatch would surface
    with _installed_config(n_assets, slippage=0.0008, tx_fee=0.0005, annual_holding=0.01, cap=1.0):
        ohlcv = _multi_asset_prices(n_assets=n_assets)
        env = _make_env(ohlcv, obs_lag_default=1)
        env.reset()
        # cash logit -60 (softmax floor) + equal risky logits → ~0 cash, equal-weight risky,
        # which equals the equal benchmark_cap_weights (1/N) restricted to live assets.
        action = np.concatenate(([-60.0], np.zeros(n_assets)))
        for _ in range(12):
            _, _, term, trunc, info = env.step(action)
            market_ret = env._market_return_buffer[-1]
            assert abs(info["log_ret"] - market_ret) < 1e-9
            if term or trunc:
                break
