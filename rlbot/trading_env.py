"""
Multi-asset portfolio Gymnasium environment (universe size from OHLCV panel / config).

Reward = return + Sortino diff vs cap-weighted benchmark - inactivity - VIX-scaled churn - drawdown
  - return: clipped_log_return * REWARD_SCALE
  - sortino diff: benchmark-relative over last RISK_WINDOW steps (moving window within episode)
  - inactivity: penalty when cash > 50% and extra when >90% (training scale 1.0;
    periodic eval uses ``EVAL_INACTIVITY_PENALTY_SCALE`` so defensive cash is not over-penalized)
  - Soft per-asset long-only cap after softmax (see config max_single_asset_weight)

Execution: trades fill at open[t+1] (next morning after decision), not close[t].
  Combined with obs_lag, the pipeline is:
    observe close[t-obs_lag] → decide overnight → execute at open[t+1] → earn to close[t+1]

Domain randomization (training): ``obs_lag`` and ``fee_scale`` resampled each episode
after the fee curriculum releases; bounds widen progressively (see ``set_randomization_bounds``).
``fee_scale`` uses Beta(5, 5) mapped to the current fee bounds (bell curve centered at 1.0).
Fee curriculum overrides DR until release (see ``set_curriculum_state``).
Fracdiff features replace raw log-return horizons in market observations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rlbot.data_utils import MACRO_VIX_INDEX, N_MACRO
from rlbot.rl_config import get_config

# Churn penalty scales with live ^VIX (macro panel) vs long-run calm baseline (~18).
VIX_CHURN_BASELINE = 18.0
VIX_CHURN_MULT_MIN = 0.75
VIX_CHURN_MULT_MAX = 1.5


def _softmax_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x - np.max(x)
    exp = np.exp(np.clip(x, -60.0, 60.0))
    s = float(exp.sum())
    return exp / (s + 1e-12)


def _enforce_long_only_simplex(w: np.ndarray) -> np.ndarray:
    """Nonnegative weights summing to 1 (no short positions)."""
    w = np.maximum(np.asarray(w, dtype=np.float64).reshape(-1), 0.0)
    s = float(w.sum())
    if s > 1e-12:
        w = w / s
    else:
        w = np.zeros_like(w)
        w[0] = 1.0
    return w


def portfolio_weights_from_action(
    action: np.ndarray,
    *,
    n_actions: int | None = None,
) -> np.ndarray:
    """Map policy logits → portfolio weights via softmax over cash + risky assets.

    Cash competes for probability mass with every asset. Risky legs are long-only with a
    per-asset cap; overflow is redistributed across other active risky assets before cash.
    """
    x = np.asarray(action, dtype=np.float64).reshape(-1)
    n_act = int(n_actions if n_actions is not None else x.shape[0])
    if x.shape[0] != n_act:
        raise ValueError(f"action must have shape ({n_act},), got {x.shape}")
    p = _softmax_1d(x)
    w = np.zeros(n_act, dtype=np.float64)
    w[0] = float(p[0])
    w[1:] = p[1:]

    max_w = get_config().environment.max_single_asset_weight
    risky_w = w[1:].copy()

    for _ in range(5):
        overflow_mask = risky_w > max_w
        if not np.any(overflow_mask):
            break

        overflow = float(np.sum(risky_w[overflow_mask] - max_w))
        risky_w[overflow_mask] = max_w

        underflow_mask = (risky_w < max_w) & (risky_w > 0.0)
        total_underflow = float(np.sum(risky_w[underflow_mask]))
        if total_underflow > 1e-12:
            risky_w[underflow_mask] += (risky_w[underflow_mask] / total_underflow) * overflow
        else:
            w[0] += overflow
            break

    w[1:] = risky_w
    return _enforce_long_only_simplex(w)


class EpisodeEndNavRecorder(gym.Wrapper):
    """Record terminal ``nav`` from each episode for eval NAV tracking (SB3 EvalCallback)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._ending_navs: list[float] = []

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            nav = info.get("nav")
            if nav is not None:
                self._ending_navs.append(float(nav))
        return obs, reward, terminated, truncated, info

    def pop_ending_navs(self) -> list[float]:
        """Return and clear NAVs collected since the last pop (one eval cycle)."""
        navs = list(self._ending_navs)
        self._ending_navs.clear()
        return navs

    def get_segments(self) -> Optional[list]:
        return self.env.get_segments()


class MultiAssetPortfolioEnv(gym.Env):
    """
    Observation size: ``9 * n_assets + 8 + 5 * N_MACRO`` (e.g. 118 when ``n_assets=10``).

    Action: Box(-3,3)^(n_assets+1) → softmax(cash + assets), long-only risky weights, per-asset cap.

    Reward: return + Sortino_bonus + participation - inactivity - VIX-scaled churn - quadratic drawdown penalty.
    Per-step ``info`` includes ``rew_decomp/*`` for each component (see ``config.yaml`` reward section).
    """

    metadata = {"render_modes": []}
    RETURN_HORIZONS = (1, 5, 10, 20)

    def __init__(
        self,
        ohlcv: np.ndarray,
        rsi: np.ndarray,
        macd: np.ndarray,
        fracdiff: np.ndarray,
        fracdiff_macro: np.ndarray,
        trend: np.ndarray,
        macro: Optional[np.ndarray] = None,
        initial_cash: float | None = None,
        lookback: int | None = None,
        random_start: bool = True,
        max_episode_steps: int | None = None,
        obs_noise_std: float = 0.0,
        reseed_on_reset: bool = False,
        block_boundaries: Optional[list] = None,
        obs_lag: int = 0,
        obs_lag_default: int | None = None,
        fee_scale_default: float | None = None,
        domain_randomize: bool = True,
        inactivity_penalty_scale: float = 1.0,
        action_smoothing_alpha: float | None = None,
    ):
        super().__init__()
        cfg = get_config()
        env_cfg = cfg.environment
        tc = cfg.transaction_costs

        assert ohlcv.ndim == 3 and ohlcv.shape[2] == 5
        self.n_assets = int(ohlcv.shape[1])
        self.n_actions = self.n_assets + 1
        assert fracdiff.shape == (ohlcv.shape[0], self.n_assets)
        assert fracdiff_macro.shape == (ohlcv.shape[0], N_MACRO)
        assert trend.shape == (ohlcv.shape[0], self.n_assets)

        self._env_cfg = env_cfg
        self._reward_cfg = cfg.reward
        self._benchmark_weights = cfg.reward.benchmark_cap_weights_array()
        self._asset_slippage = tc.slippage_array()
        self._asset_tx_fee = tc.tx_fee_array()
        self._daily_holding_cost = tc.daily_holding_cost_array()
        if self._asset_slippage.shape[0] != self.n_assets:
            raise ValueError(
                f"transaction_costs.slippage length {self._asset_slippage.shape[0]} "
                f"!= n_assets {self.n_assets}"
            )
        if self._benchmark_weights.shape[0] != self.n_assets:
            raise ValueError(
                f"benchmark_cap_weights length {self._benchmark_weights.shape[0]} "
                f"!= n_assets {self.n_assets}"
            )

        self.ohlcv = ohlcv.astype(np.float64)
        self.fracdiff = fracdiff.astype(np.float64)
        self.fracdiff_macro = fracdiff_macro.astype(np.float64)
        self.rsi = rsi.astype(np.float64)
        self.macd = macd.astype(np.float64)
        self.trend = trend.astype(np.float64)
        if macro is not None:
            self.macro = macro.astype(np.float64)
        else:
            self.macro = np.zeros((ohlcv.shape[0], N_MACRO), dtype=np.float64)
        self.initial_cash = float(
            env_cfg.initial_cash if initial_cash is None else initial_cash
        )
        self.lookback = int(lookback if lookback is not None else env_cfg.lookback)
        self._obs_lag_default = int(
            obs_lag_default if obs_lag_default is not None else env_cfg.obs_lag_default
        )
        self._fee_scale_default = float(
            fee_scale_default if fee_scale_default is not None else env_cfg.fee_scale_default
        )
        self.domain_randomize = bool(domain_randomize)
        self._inactivity_penalty_scale = float(np.clip(inactivity_penalty_scale, 0.0, 1.0))
        self.obs_lag = int(obs_lag)
        self.fee_scale = self._fee_scale_default
        # Training curriculum (fee ramp / churn off early); None = use domain_randomize or default
        self._curriculum_fee_override: Optional[float] = None
        self._churn_scale = 1.0
        self._fee_dr_min = env_cfg.domain_randomize_fee_dr_min
        self._fee_dr_max = env_cfg.domain_randomize_fee_dr_max
        self._obs_lag_dr_min = env_cfg.min_obs_lag
        self._obs_lag_dr_max = env_cfg.max_obs_lag
        self._fee_dr_beta_a = env_cfg.domain_randomize_fee_beta_a
        self._fee_dr_beta_b = env_cfg.domain_randomize_fee_beta_b
        self.random_start = random_start
        self.max_episode_steps = int(
            max_episode_steps if max_episode_steps is not None else env_cfg.max_episode_steps
        )
        self.obs_noise_std = obs_noise_std
        self.reseed_on_reset = reseed_on_reset

        self._t = 0
        self._steps = 0
        self._cash = self.initial_cash
        self._units = np.zeros(self.n_assets, dtype=np.float64)
        self._episode_start_nav = self.initial_cash
        self._episode_peak_nav = self.initial_cash
        self._current_ep_max_steps = self.max_episode_steps
        self._rng = np.random.default_rng()
        self._reset_count = 0
        self._return_buffer: list[float] = []
        self._market_return_buffer: list[float] = []
        self._prev_target_w = np.zeros(self.n_actions, dtype=np.float64)
        self._prev_target_w[0] = 1.0
        alpha = (
            float(action_smoothing_alpha)
            if action_smoothing_alpha is not None
            else float(env_cfg.action_smoothing_alpha)
        )
        self._action_smoothing_alpha = float(np.clip(alpha, 0.0, 1.0))
        self._smoothed_action: np.ndarray | None = None

        n_returns = len(self.RETURN_HORIZONS) * self.n_assets
        n_mkt_returns = len(self.RETURN_HORIZONS)
        n_vol = self.n_assets
        n_mkt_vol = 1
        n_rsi = self.n_assets
        n_macd = self.n_assets
        n_trend = self.n_assets
        n_macro = N_MACRO * (len(self.RETURN_HORIZONS) + 1)
        n_port = self.n_actions
        n_meta = 2
        self._n_market_features = (
            n_returns
            + n_mkt_returns
            + n_vol
            + n_mkt_vol
            + n_rsi
            + n_macd
            + n_trend
            + n_macro
        )
        obs_dim = self._n_market_features + n_port + n_meta

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-3.0, high=3.0, shape=(self.n_actions,), dtype=np.float32
        )

        self._min_t = self.lookback + env_cfg.max_obs_lag
        self._max_t = self.ohlcv.shape[0] - 2

        # Contiguous segments within the concatenated data.  When the data
        # comes from an alternating train/eval split, block_boundaries marks
        # where non-adjacent time periods were joined.  Episodes must stay
        # entirely within one segment so observations never span a gap.
        self._segments = self._build_segments(block_boundaries or [])

        self._noise_scale = self._compute_noise_scale() if obs_noise_std > 0.0 else None

    def set_curriculum_state(self, fee_override: Optional[float], churn_scale: float) -> None:
        """Called by ``TradingCurriculumCallback`` on training envs only.

        ``fee_override``: fixed ``fee_scale`` until next reset when set (0 = frictionless).
        ``None`` = release to domain randomization (or default fee) on reset.
        ``churn_scale``: multiplies configured churn penalty (0 = off).
        """
        self._curriculum_fee_override = fee_override
        self._churn_scale = float(churn_scale)

    def set_randomization_bounds(
        self,
        fee_min: float,
        fee_max: float,
        obs_lag_min: int | None = None,
        obs_lag_max: int | None = None,
    ) -> None:
        """Widen or narrow domain-randomization sampling (training only)."""
        self._fee_dr_min = float(fee_min)
        self._fee_dr_max = float(max(fee_max, fee_min))
        if obs_lag_min is not None and obs_lag_max is not None:
            ec = get_config().environment
            lo = int(np.clip(obs_lag_min, ec.min_obs_lag, ec.max_obs_lag))
            hi = int(np.clip(obs_lag_max, ec.min_obs_lag, ec.max_obs_lag))
            self._obs_lag_dr_min = min(lo, hi)
            self._obs_lag_dr_max = max(lo, hi)

    def _sample_dr_fee_scale(self) -> float:
        """Beta(α, β) on [0, 1], mapped to ``[_fee_dr_min, _fee_dr_max]`` (mode ≈ 1.0)."""
        u = float(self._rng.beta(self._fee_dr_beta_a, self._fee_dr_beta_b))
        return float(self._fee_dr_min + u * (self._fee_dr_max - self._fee_dr_min))

    def _sample_dr_obs_lag(self) -> int:
        ec = get_config().environment
        lo = max(ec.min_obs_lag, self._obs_lag_dr_min)
        hi = min(ec.max_obs_lag, self._obs_lag_dr_max)
        return int(self._rng.integers(lo, hi + 1))

    MIN_SEGMENT_BARS = 30

    def _build_segments(self, block_boundaries: list):
        """Return list of (earliest_start, segment_end) for contiguous segments.

        ``earliest_start`` accounts for the lookback window so that
        ``_build_obs`` never reads across a discontinuity.  ``segment_end``
        is exclusive — the last bar usable by step() is ``segment_end - 2``
        (since step reads ohlcv[t+1]).  Very short segments (< MIN_SEGMENT_BARS
        usable bars) are dropped to avoid noisy micro-episodes.
        """
        if not block_boundaries:
            return None
        bounds = sorted(set(block_boundaries))
        edges = [self._min_t] + bounds + [self._max_t + 2]
        segments = []
        for i in range(len(edges) - 1):
            seg_raw_start = edges[i]
            seg_end = edges[i + 1]
            earliest = max(
                seg_raw_start + self.lookback + self._env_cfg.max_obs_lag,
                seg_raw_start,
            )
            usable = seg_end - earliest - 1
            if usable >= self.MIN_SEGMENT_BARS:
                segments.append((earliest, seg_end))
        return segments if segments else None

    def _compute_noise_scale(self) -> np.ndarray:
        """
        Per-feature std from actual data; noise is proportional to each
        feature's natural variability so noisier assets get more noise.
        """
        n_samples = min(2000, self._max_t - self._min_t)
        indices = np.linspace(self._min_t + 1, self._max_t, n_samples, dtype=int)

        samples = []
        for t_raw in indices:
            t = max(t_raw - self.obs_lag, 0)
            parts = []
            for h in self.RETURN_HORIZONS:
                t0 = max(t - h, 0)
                fd = self.fracdiff[t] - self.fracdiff[t0]
                parts.append(fd.astype(np.float32) * 100.0)
                parts.append(np.array([fd.mean()], dtype=np.float32) * 100.0)

            start = max(t - self.lookback, 1)
            closes_w = self.ohlcv[start : t + 1, :, 3]
            if len(closes_w) >= 2:
                vol = np.diff(np.log(closes_w + 1e-12), axis=0).std(axis=0)
            else:
                vol = np.zeros(self.n_assets)
            parts.append(vol.astype(np.float32) * 100.0)
            parts.append(np.array([vol.mean()], dtype=np.float32) * 100.0)

            parts.append((self.rsi[t] / 50.0 - 1.0).astype(np.float32))
            parts.append(np.tanh(self.macd[t]).astype(np.float32))
            parts.append(np.clip(self.trend[t], -1.0, 1.0).astype(np.float32))

            for h in self.RETURN_HORIZONS:
                t0 = max(t - h, 0)
                mfd = self.fracdiff_macro[t] - self.fracdiff_macro[t0]
                parts.append(mfd.astype(np.float32) * 100.0)
            m_start = max(t - self.lookback, 1)
            m_vals = self.macro[m_start : t + 1]
            if len(m_vals) >= 2:
                m_vol = np.diff(np.log(m_vals + 1e-12), axis=0).std(axis=0)
            else:
                m_vol = np.zeros(N_MACRO)
            parts.append(m_vol.astype(np.float32) * 100.0)

            samples.append(np.concatenate(parts))

        feature_stds = np.stack(samples).std(axis=0)
        feature_stds = np.maximum(feature_stds, 0.01)
        return feature_stds.astype(np.float32)

    # ── helpers ──────────────────────────────────────────────────────────

    def _nav(self, close: np.ndarray) -> float:
        u = np.maximum(self._units, 0.0)
        return float(self._cash + np.dot(u, close))

    def _portfolio_weights(self, close: np.ndarray) -> np.ndarray:
        v = self._nav(close)
        w = np.zeros(self.n_actions, dtype=np.float32)
        if v > 1e-12:
            w[0] = self._cash / v
            # Long-only book: asset legs are nonnegative (no short inventory)
            pos = np.maximum(self._units, 0.0) * close
            w[1:] = (pos.astype(np.float32) / np.float32(v))
        return w

    def _log_returns(self, t: int, horizon: int) -> np.ndarray:
        t0 = max(t - horizon, 0)
        c_now = self.ohlcv[t, :, 3]
        c_prev = self.ohlcv[t0, :, 3]
        return np.log((c_now + 1e-12) / (c_prev + 1e-12))

    def _realized_vol(self, t: int) -> np.ndarray:
        start = max(t - self.lookback, 1)
        closes = self.ohlcv[start : t + 1, :, 3]
        if len(closes) < 2:
            return np.zeros(self.n_assets, dtype=np.float64)
        rets = np.diff(np.log(closes + 1e-12), axis=0)
        return rets.std(axis=0)

    def _macro_log_returns(self, t: int, horizon: int) -> np.ndarray:
        t0 = max(t - horizon, 0)
        c_now = self.macro[t]
        c_prev = self.macro[t0]
        return np.log((c_now + 1e-12) / (c_prev + 1e-12))

    def _macro_realized_vol(self, t: int) -> np.ndarray:
        start = max(t - self.lookback, 1)
        vals = self.macro[start : t + 1]
        if len(vals) < 2:
            return np.zeros(N_MACRO, dtype=np.float64)
        rets = np.diff(np.log(vals + 1e-12), axis=0)
        return rets.std(axis=0)

    def _build_obs(self) -> np.ndarray:
        t = self._t
        # Market features at t_mkt = t - obs_lag (0 → close[t] known before open[t+1] fill).
        # Portfolio weights and meta (drawdown, progress) use live state at t.
        t_mkt = max(t - self.obs_lag, 0)
        parts = []

        for h in self.RETURN_HORIZONS:
            t0 = max(t_mkt - h, 0)
            asset_fd = (self.fracdiff[t_mkt] - self.fracdiff[t0]).astype(np.float32)
            parts.append(asset_fd * 100.0)
            parts.append(np.array([asset_fd.mean()], dtype=np.float32) * 100.0)

        asset_vol = self._realized_vol(t_mkt).astype(np.float32)
        parts.append(asset_vol * 100.0)
        parts.append(np.array([asset_vol.mean()], dtype=np.float32) * 100.0)

        rsi_scaled = (self.rsi[t_mkt] / 50.0 - 1.0).astype(np.float32)
        parts.append(np.clip(rsi_scaled, -2.0, 2.0))

        macd_scaled = np.tanh(self.macd[t_mkt]).astype(np.float32)
        parts.append(macd_scaled)

        trend_scaled = np.clip(self.trend[t_mkt], -1.0, 1.0).astype(np.float32)
        parts.append(trend_scaled)

        for h in self.RETURN_HORIZONS:
            t0 = max(t_mkt - h, 0)
            mfd = (self.fracdiff_macro[t_mkt] - self.fracdiff_macro[t0]).astype(np.float32)
            parts.append(mfd * 100.0)
        macro_vol = self._macro_realized_vol(t_mkt).astype(np.float32)
        parts.append(macro_vol * 100.0)

        close = self.ohlcv[t, :, 3]
        parts.append(self._portfolio_weights(close))

        nav = self._nav(close)
        dd = (nav - self._episode_peak_nav) / max(self._episode_peak_nav, 1e-12)
        progress = self._steps / max(self._current_ep_max_steps, 1)
        parts.append(np.array([dd, progress], dtype=np.float32))

        obs = np.concatenate(parts)

        if self.obs_noise_std > 0.0 and self._noise_scale is not None:
            noise = self._rng.normal(0.0, 1.0, size=self._n_market_features).astype(np.float32)
            noise *= self._noise_scale * self.obs_noise_std
            obs[:self._n_market_features] += noise

        return obs

    # ── execution ────────────────────────────────────────────────────────

    def _rebalance(self, price: np.ndarray, target_w: np.ndarray) -> float:
        """Execute trades at given prices with per-asset slippage and fees.

        Scales transaction costs by self.fee_scale (for curriculum learning).
        Target weights are long-only (cash + nonnegative asset notionals).
        """
        tw = _enforce_long_only_simplex(np.asarray(target_w, dtype=np.float64))
        nav = self._nav(price)
        if nav <= 1e-12:
            return 0.0

        fs = self.fee_scale
        target_units = (tw[1:] * nav) / (price + 1e-12)
        delta = target_units - self._units
        turnover = 0.0

        for i in np.argsort(delta):
            du = delta[i]
            if du >= -1e-12:
                continue
            sell_u = -du
            cost_rate = (self._asset_slippage[i] + self._asset_tx_fee[i]) * fs
            self._cash += sell_u * price[i] * (1.0 - cost_rate)
            self._units[i] -= sell_u
            turnover += sell_u * price[i]

        for i in np.argsort(-delta):
            du = delta[i]
            if du <= 1e-12:
                continue
            cost_rate = (self._asset_slippage[i] + self._asset_tx_fee[i]) * fs
            unit_cost = price[i] * (1.0 + cost_rate)
            buy_u = min(du, max(0.0, self._cash / (unit_cost + 1e-12)))
            if buy_u <= 1e-12:
                continue
            self._cash -= buy_u * unit_cost
            self._units[i] += buy_u
            turnover += buy_u * price[i]

        self._units = np.maximum(self._units, 0.0)
        return turnover / max(nav, 1e-12)

    def _apply_holding_costs(self, close: np.ndarray) -> float:
        """Deduct daily holding costs (expense ratios, roll costs) from cash."""
        position_values = self._units * close
        notional = np.abs(position_values)
        daily_costs = notional * self._daily_holding_cost * self.fee_scale
        total_cost = float(np.maximum(daily_costs, 0.0).sum())
        if total_cost > 0:
            self._cash -= total_cost
        return total_cost

    def get_segments(self) -> Optional[list]:
        """Contiguous eval/train segments as (earliest_start, exclusive_end) pairs."""
        return self._segments

    # ── gym interface ────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if self.reseed_on_reset:
            self._rng = np.random.default_rng()
        elif seed is not None:
            self._rng = np.random.default_rng(seed)

        self._cash = self.initial_cash
        self._units[:] = 0.0
        self._steps = 0
        self._return_buffer = []
        self._market_return_buffer = []
        self._prev_target_w = np.zeros(self.n_actions, dtype=np.float64)
        self._prev_target_w[0] = 1.0
        self._smoothed_action = None

        if self.domain_randomize and self.random_start:
            self.obs_lag = self._sample_dr_obs_lag()
            if self._curriculum_fee_override is not None:
                self.fee_scale = float(self._curriculum_fee_override)
            else:
                self.fee_scale = self._sample_dr_fee_scale()
        else:
            self.obs_lag = self._obs_lag_default
            if self._curriculum_fee_override is not None:
                self.fee_scale = float(self._curriculum_fee_override)
            else:
                self.fee_scale = self._fee_scale_default

        if not self.random_start and self._segments is not None:
            # Deterministic eval: one episode per segment, full contiguous block.
            seg_idx = self._reset_count % len(self._segments)
            earliest, seg_end = self._segments[seg_idx]
            self._t = earliest
            bars_left = seg_end - self._t - 2
            self._current_ep_max_steps = max(bars_left, 1)
            self._reset_count += 1
        elif self.random_start:
            jitter = self._rng.integers(-self.max_episode_steps // 5,
                                         self.max_episode_steps // 5 + 1)
            self._current_ep_max_steps = max(self.max_episode_steps // 2, self.max_episode_steps + int(jitter))
        else:
            self._current_ep_max_steps = self.max_episode_steps

        if self.random_start and self._segments is not None:
            sizes = [max(0, seg_end - earliest - 1) for earliest, seg_end in self._segments]
            total = sum(sizes)
            if total > 0:
                probs = [s / total for s in sizes]
                seg_idx = int(self._rng.choice(len(sizes), p=probs))
                earliest, seg_end = self._segments[seg_idx]
                latest = seg_end - 2
                self._t = int(self._rng.integers(earliest, latest + 1))
                bars_left = seg_end - self._t - 2
                self._current_ep_max_steps = min(self._current_ep_max_steps, bars_left)
                self._current_ep_max_steps = max(self._current_ep_max_steps, 1)
            else:
                self._t = self._min_t
        elif self.random_start:
            max_start = max(self._min_t, self._max_t - self._current_ep_max_steps)
            if max_start > self._min_t:
                self._t = int(self._rng.integers(self._min_t, max_start + 1))
            else:
                self._t = self._min_t
        elif not (not self.random_start and self._segments is not None):
            self._t = self._min_t

        close0 = self.ohlcv[self._t, :, 3]
        self._episode_start_nav = self._nav(close0)
        self._episode_peak_nav = self._episode_start_nav
        return self._build_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if self._action_smoothing_alpha > 0.0:
            if self._smoothed_action is None or self._steps == 0:
                self._smoothed_action = action.copy()
            else:
                a = self._action_smoothing_alpha
                self._smoothed_action = a * action + (1.0 - a) * self._smoothed_action
            action_for_weights = self._smoothed_action
        else:
            action_for_weights = action

        close_t = self.ohlcv[self._t, :, 3]
        v_pre = max(self._nav(close_t), 1e-12)

        open_next = self.ohlcv[self._t + 1, :, 0]
        w = portfolio_weights_from_action(action_for_weights, n_actions=self.n_actions)
        rwd = self._reward_cfg
        # Scale churn with live VIX (same series as obs macro block; undifferenced close).
        current_vix = float(self.macro[self._t, MACRO_VIX_INDEX])
        if current_vix > 1.0:
            vix_multiplier = float(
                np.clip(current_vix / VIX_CHURN_BASELINE, VIX_CHURN_MULT_MIN, VIX_CHURN_MULT_MAX)
            )
        else:
            vix_multiplier = 1.0
        active_churn_lambda = rwd.churn_lambda * vix_multiplier * self._churn_scale
        dw = w[1:] - self._prev_target_w[1:]
        weight_turnover = float(np.sum(np.abs(dw))) / max(self.n_assets, 1)
        churn = active_churn_lambda * weight_turnover
        self._prev_target_w = w.copy()

        turnover_frac = self._rebalance(open_next, w)

        close_next = self.ohlcv[self._t + 1, :, 3]
        self._apply_holding_costs(close_next)
        v_next = max(self._nav(close_next), 1e-12)

        peak_before = self._episode_peak_nav

        log_ret = float(np.log(v_next / v_pre))
        clipped_ret = float(np.clip(log_ret, -rwd.max_step_log_return, rwd.max_step_log_return))

        # Cap-weighted passive benchmark (SPY-heavy; config benchmark_cap_weights)
        asset_log_rets = np.log((close_next + 1e-12) / (close_t + 1e-12))
        market_ret = float(np.dot(self._benchmark_weights, asset_log_rets))
        self._return_buffer.append(log_ret)
        self._market_return_buffer.append(market_ret)

        # ── reward: return + benchmark-relative Sortino - inactivity ─
        return_component = clipped_ret * rwd.reward_scale
        reward = return_component

        sortino_component = 0.0
        if len(self._return_buffer) >= rwd.risk_window:
            def _sortino(rets: np.ndarray) -> float:
                m = float(rets.mean())
                ds = rets[rets < 0]
                dv = float(np.sqrt((ds ** 2).mean())) if len(ds) > 1 else 1e-8
                return m / (dv + 1e-8)

            agent_s = _sortino(np.array(self._return_buffer[-rwd.risk_window:]))
            market_s = _sortino(np.array(self._market_return_buffer[-rwd.risk_window:]))
            diff = float(np.clip(agent_s - market_s, -3.0, 3.0))
            sortino_component = diff * rwd.risk_bonus_scale
            reward += sortino_component

        cash_frac = self._cash / max(v_next, 1e-12)
        inact = self._inactivity_penalty_scale
        inactivity_component = 0.0
        if cash_frac > 0.50:
            inactivity_component += rwd.inactivity_penalty_over_50 * inact
        if cash_frac > 0.90:
            inactivity_component += rwd.inactivity_penalty_over_90 * inact
        reward -= inactivity_component

        gross_exposure = float(np.sum(w[1:]))
        participation_component = (
            gross_exposure * rwd.participation_bonus * rwd.participation_reward_scale
        )
        reward += participation_component

        churn_component = churn * rwd.churn_penalty_scale
        reward -= churn_component

        dd_frac = max(0.0, (peak_before - v_next) / max(peak_before, 1e-12))
        drawdown_component = (dd_frac ** 2) * (
            rwd.drawdown_penalty_scale * rwd.drawdown_quadratic_multiplier
        )
        reward -= drawdown_component
        self._episode_peak_nav = max(peak_before, v_next)

        self._t += 1
        self._steps += 1

        terminated = bool(
            v_next <= self._env_cfg.stop_loss_fraction * self._episode_start_nav
        )
        truncated = bool(self._t >= self._max_t or self._steps >= self._current_ep_max_steps)

        info: Dict[str, Any] = {
            "nav": v_next,
            "turnover": turnover_frac,
            "log_ret": log_ret,
            "rew_decomp/return": return_component,
            "rew_decomp/sortino": sortino_component,
            "rew_decomp/inactivity": -inactivity_component,
            "rew_decomp/participation": participation_component,
            "rew_decomp/churn": -churn_component,
            "rew_decomp/vix_churn_mult": vix_multiplier,
            "rew_decomp/drawdown": -drawdown_component,
        }

        if terminated or truncated or self._t > self._max_t:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._build_obs()

        return obs, reward, terminated, truncated, info
