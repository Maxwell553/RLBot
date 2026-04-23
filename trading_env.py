"""
Multi-asset portfolio Gymnasium environment for 10 global assets.

Assets (in order): SP500, GOLD, OIL, EURUSD, USDJPY, NIKKEI, FTSE, BOND10Y, COPPER, EM

Reward = return + Sortino diff - inactivity - churn - linear drawdown (from peak NAV)
  - return: clipped_log_return * REWARD_SCALE
  - sortino diff: benchmark-relative over last RISK_WINDOW steps (moving window within episode)
  - inactivity: penalty when cash > 50% (scaled vs return) and extra when >90%
  - Soft 40% per-asset long-only cap after softmax

Execution: trades fill at open[t+1] (next morning after decision), not close[t].
  Combined with obs_lag, the pipeline is:
    observe close[t-obs_lag] → decide overnight → execute at open[t+1] → earn to close[t+1]

Domain randomization (training): ``obs_lag`` resampled each episode; ``fee_scale`` can be
overridden by a fee curriculum (see ``set_curriculum_state``).
Fracdiff features replace raw log-return horizons in market observations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from data_utils import N_MACRO

# ── Per-asset transaction costs ─────────────────────────────────────────
# Order matches TICKERS in data_utils:
# SP500, GOLD, OIL, EURUSD, USDJPY, NIKKEI, FTSE, BOND10Y, COPPER, EM

ASSET_SLIPPAGE = np.array([
    0.0001,   # SP500  - SPY extremely liquid, tight spreads
    0.0002,   # GOLD   - GLD liquid ETF
    0.0003,   # OIL    - USO moderate spreads
    0.0001,   # EURUSD - most liquid forex pair
    0.0001,   # USDJPY - very liquid forex
    0.0005,   # NIKKEI - foreign equity index
    0.0005,   # FTSE   - foreign equity index
    0.0001,   # BOND10Y- IEF very liquid bond ETF
    0.0008,   # COPPER - HG=F less liquid commodity
    0.0002,   # EM     - EEM liquid ETF
], dtype=np.float64)

ASSET_TX_FEE = np.array([
    0.0001,   # SP500  - near-zero commission
    0.0002,   # GOLD   - ETF commission
    0.0002,   # OIL    - ETF commission
    0.00005,  # EURUSD - forex broker spread
    0.00005,  # USDJPY - forex broker spread
    0.0010,   # NIKKEI - international market access premium
    0.0010,   # FTSE   - international market access premium
    0.0001,   # BOND10Y- ETF commission
    0.0005,   # COPPER - futures commission
    0.0002,   # EM     - ETF commission
], dtype=np.float64)

ANNUAL_HOLDING_COST = np.array([
    0.0009,   # SP500  - SPY 0.09% expense ratio
    0.0040,   # GOLD   - GLD 0.40% expense ratio
    0.0083,   # OIL    - USO 0.83% expense ratio + roll drag
    0.0000,   # EURUSD - spot forex, no holding cost
    0.0000,   # USDJPY - spot forex, no holding cost
    0.0010,   # NIKKEI - index access/tracking cost
    0.0010,   # FTSE   - index access/tracking cost
    0.0015,   # BOND10Y- IEF 0.15% expense ratio
    0.0060,   # COPPER - HG=F futures roll cost
    0.0067,   # EM     - EEM 0.67% expense ratio
], dtype=np.float64)

DAILY_HOLDING_COST = ANNUAL_HOLDING_COST / 252.0

STOP_LOSS_FRACTION = 0.45

N_ASSETS = 10
N_ACTIONS = N_ASSETS + 1       # cash + 10 assets
LOOKBACK = 20                  # 20 trading days ≈ 1 calendar month

# ── Reward ───────────────────────────────────────────────────────────────
REWARD_SCALE = 2000.0
MAX_STEP_LOG_RETURN = 0.03
STOP_LOSS_TERMINAL_PENALTY = 100.0  # legacy; terminal penalty disabled in favor of smooth DD term
CHURN_LAMBDA = 0.0002        # Penalize |Δw| (× REWARD_SCALE); can be scaled to 0 early in training
MAX_OBS_LAG = 2  # max obs_lag for index safety (domain randomization samples 0..2)
RISK_WINDOW = 21              # ~3 updates per 63-step episode; moving Sortino vs one end-of-episode block
RISK_BONUS_SCALE = 25.0       # Sortino diff scale (lower → weight raw returns more vs risk term)
INACTIVITY_PENALTY_OVER_50 = 5.0   # Per-step when cash/NAV > 50% (comparable to ~0.25% daily return term)
INACTIVITY_PENALTY_OVER_90 = 0.1   # Extra per-step when cash/NAV > 90%
PARTICIPATION_BONUS = 0.1          # reward += gross_exposure * this (gross = sum|w_risky|)
MAX_SINGLE_ASSET_WEIGHT = 0.40  # Soft cap: max weight per asset (long-only after softmax)

DEFAULT_NOISE_SCALES = None


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


def portfolio_weights_from_action(action: np.ndarray) -> np.ndarray:
    """Map policy logits → portfolio weights via **softmax over all 11 slots** (cash + 10 assets).

    Cash competes for probability mass with every asset on the same footing. Resulting
    risky weights are **long-only** (nonnegative); apply a soft cap per asset and return
    freed notional to cash.
    """
    x = np.asarray(action, dtype=np.float64).reshape(-1)
    if x.shape[0] != N_ACTIONS:
        raise ValueError(f"action must have shape ({N_ACTIONS},), got {x.shape}")
    p = _softmax_1d(x)
    w = np.zeros(N_ACTIONS, dtype=np.float64)
    w[0] = float(p[0])
    w[1:] = p[1:]

    # Soft cap on long risky weights; return overflow to cash
    asset_w = w[1:].copy()
    old_sum = float(np.sum(asset_w))
    clipped = np.clip(asset_w, 0.0, MAX_SINGLE_ASSET_WEIGHT)
    new_sum = float(np.sum(clipped))
    w[0] += old_sum - new_sum
    w[1:] = clipped
    return _enforce_long_only_simplex(w)


class MultiAssetPortfolioEnv(gym.Env):
    """
    Observation (98 features):
      - Multi-horizon fracdiff increments (1d,5d,10d,20d) × 10  = 40
      - Market-wide mean returns per horizon                       = 4
      - Per-asset realized volatility (20d)                        = 10
      - Market-wide mean volatility                                = 1
      - RSI per asset (scaled [-1,1])                              = 10
      - MACD per asset (tanh-compressed)                           = 10
      - Macro features (DXY, 10Y Yield): 4-horizon rets + vol × 2 = 10
      - Current portfolio weights (cash + 10 assets)               = 11
      - Drawdown from episode peak                                 = 1
      - Episode progress fraction                                  = 1

    Action: Box(-3,3)^11 → softmax(cash + 10 assets), long-only risky weights, ±40% soft cap

    Reward: return + Sortino_bonus + participation - inactivity - churn - linear drawdown penalty
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
        macro: Optional[np.ndarray] = None,
        initial_cash: float = 100_000.0,
        lookback: int = LOOKBACK,
        random_start: bool = True,
        max_episode_steps: int = 252,
        obs_noise_std: float = 0.0,
        reseed_on_reset: bool = False,
        block_boundaries: Optional[list] = None,
        obs_lag: int = 0,
        obs_lag_default: int = 1,
        fee_scale_default: float = 1.0,
        domain_randomize: bool = True,
    ):
        super().__init__()
        assert ohlcv.ndim == 3 and ohlcv.shape[1] == N_ASSETS and ohlcv.shape[2] == 5
        assert fracdiff.shape == (ohlcv.shape[0], N_ASSETS)
        assert fracdiff_macro.shape == (ohlcv.shape[0], N_MACRO)

        self.ohlcv = ohlcv.astype(np.float64)
        self.fracdiff = fracdiff.astype(np.float64)
        self.fracdiff_macro = fracdiff_macro.astype(np.float64)
        self.rsi = rsi.astype(np.float64)
        self.macd = macd.astype(np.float64)
        if macro is not None:
            self.macro = macro.astype(np.float64)
        else:
            self.macro = np.zeros((ohlcv.shape[0], N_MACRO), dtype=np.float64)
        self.initial_cash = float(initial_cash)
        self.lookback = lookback
        self._obs_lag_default = int(obs_lag_default)
        self._fee_scale_default = float(fee_scale_default)
        self.domain_randomize = bool(domain_randomize)
        self.obs_lag = int(obs_lag)
        self.fee_scale = float(fee_scale_default)
        # Training curriculum (fee ramp / churn off early); None = use domain_randomize or default
        self._curriculum_fee_override: Optional[float] = None
        self._churn_scale = 1.0
        self.random_start = random_start
        self.max_episode_steps = max_episode_steps
        self.obs_noise_std = obs_noise_std
        self.reseed_on_reset = reseed_on_reset

        self._t = 0
        self._steps = 0
        self._cash = self.initial_cash
        self._units = np.zeros(N_ASSETS, dtype=np.float64)
        self._episode_start_nav = self.initial_cash
        self._episode_peak_nav = self.initial_cash
        self._current_ep_max_steps = max_episode_steps
        self._rng = np.random.default_rng()
        self._reset_count = 0
        self._return_buffer: list[float] = []
        self._market_return_buffer: list[float] = []
        self._prev_target_w = np.zeros(N_ACTIONS, dtype=np.float64)
        self._prev_target_w[0] = 1.0

        n_returns = len(self.RETURN_HORIZONS) * N_ASSETS  # 40
        n_mkt_returns = len(self.RETURN_HORIZONS)          # 4
        n_vol = N_ASSETS                                   # 10
        n_mkt_vol = 1                                      # 1
        n_rsi = N_ASSETS                                   # 10
        n_macd = N_ASSETS                                  # 10
        n_macro = N_MACRO * (len(self.RETURN_HORIZONS) + 1)  # 2 × 5 = 10
        n_port = N_ACTIONS                                 # 11
        n_meta = 2                                         # drawdown + progress
        self._n_market_features = (
            n_returns + n_mkt_returns + n_vol + n_mkt_vol + n_rsi + n_macd + n_macro
        )
        obs_dim = self._n_market_features + n_port + n_meta  # 98

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-3.0, high=3.0, shape=(N_ACTIONS,), dtype=np.float32
        )

        self._min_t = lookback + MAX_OBS_LAG
        self._max_t = self.ohlcv.shape[0] - 2

        # Contiguous segments within the concatenated data.  When the data
        # comes from an alternating train/eval split, block_boundaries marks
        # where non-adjacent time periods were joined.  Episodes must stay
        # entirely within one segment so observations never span a gap.
        self._segments = self._build_segments(block_boundaries or [])

        self._noise_scale = self._compute_noise_scale() if obs_noise_std > 0.0 else None

    def set_curriculum(self, obs_lag: int, fee_scale: float) -> None:
        """Legacy: set obs_lag and fee (disables curriculum override for fee)."""
        self.obs_lag = int(obs_lag)
        self.fee_scale = float(fee_scale)
        self._curriculum_fee_override = None

    def set_curriculum_state(self, fee_override: Optional[float], churn_scale: float) -> None:
        """Called by ``TradingCurriculumCallback`` on training envs only.

        ``fee_override``: fixed ``fee_scale`` until next reset when set (0 = frictionless).
        ``None`` = release to domain randomization (or default fee) on reset.
        ``churn_scale``: multiplies ``CHURN_LAMBDA`` (0 = no churn penalty).
        """
        self._curriculum_fee_override = fee_override
        self._churn_scale = float(churn_scale)

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
            earliest = max(seg_raw_start + self.lookback + MAX_OBS_LAG, seg_raw_start)
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
                vol = np.zeros(N_ASSETS)
            parts.append(vol.astype(np.float32) * 100.0)
            parts.append(np.array([vol.mean()], dtype=np.float32) * 100.0)

            parts.append((self.rsi[t] / 50.0 - 1.0).astype(np.float32))
            parts.append(np.tanh(self.macd[t]).astype(np.float32))

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
        w = np.zeros(N_ACTIONS, dtype=np.float32)
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
            return np.zeros(N_ASSETS, dtype=np.float64)
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
        # Market features use lagged time so the agent never sees today's
        # close in its observation.  Portfolio weights and meta (drawdown,
        # progress) still reflect the agent's live state.
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
            cost_rate = (ASSET_SLIPPAGE[i] + ASSET_TX_FEE[i]) * fs
            self._cash += sell_u * price[i] * (1.0 - cost_rate)
            self._units[i] -= sell_u
            turnover += sell_u * price[i]

        for i in np.argsort(-delta):
            du = delta[i]
            if du <= 1e-12:
                continue
            cost_rate = (ASSET_SLIPPAGE[i] + ASSET_TX_FEE[i]) * fs
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
        daily_costs = notional * DAILY_HOLDING_COST * self.fee_scale
        total_cost = float(np.maximum(daily_costs, 0.0).sum())
        if total_cost > 0:
            self._cash -= total_cost
        return total_cost

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
        self._prev_target_w = np.zeros(N_ACTIONS, dtype=np.float64)
        self._prev_target_w[0] = 1.0

        if self.domain_randomize and self.random_start:
            self.obs_lag = int(self._rng.integers(0, MAX_OBS_LAG + 1))
            if self._curriculum_fee_override is not None:
                self.fee_scale = float(self._curriculum_fee_override)
            else:
                self.fee_scale = float(self._rng.uniform(0.5, 1.5))
        else:
            self.obs_lag = self._obs_lag_default
            if self._curriculum_fee_override is not None:
                self.fee_scale = float(self._curriculum_fee_override)
            else:
                self.fee_scale = self._fee_scale_default

        if self.random_start:
            jitter = self._rng.integers(-self.max_episode_steps // 5,
                                         self.max_episode_steps // 5 + 1)
            self._current_ep_max_steps = max(self.max_episode_steps // 2, self.max_episode_steps + int(jitter))
        else:
            self._current_ep_max_steps = self.max_episode_steps

        if not self.random_start and self._segments is not None:
            # Deterministic eval: cycle through segments so each evaluation
            # checkpoint tests the same set of starting points.
            seg_idx = self._reset_count % len(self._segments)
            earliest, seg_end = self._segments[seg_idx]
            self._t = earliest
            bars_left = seg_end - self._t - 2
            self._current_ep_max_steps = min(self._current_ep_max_steps,
                                             max(bars_left, 1))
            self._reset_count += 1
        elif self.random_start and self._segments is not None:
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
        else:
            self._t = self._min_t

        close0 = self.ohlcv[self._t, :, 3]
        self._episode_start_nav = self._nav(close0)
        self._episode_peak_nav = self._episode_start_nav
        return self._build_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        close_t = self.ohlcv[self._t, :, 3]
        v_pre = max(self._nav(close_t), 1e-12)

        open_next = self.ohlcv[self._t + 1, :, 0]
        w = portfolio_weights_from_action(action)
        churn = (
            CHURN_LAMBDA
            * self._churn_scale
            * REWARD_SCALE
            * float(np.sum(np.abs(w - self._prev_target_w)))
        )
        self._prev_target_w = w.copy()

        turnover_frac = self._rebalance(open_next, w)

        self._apply_holding_costs(close_t)

        close_next = self.ohlcv[self._t + 1, :, 3]
        v_next = max(self._nav(close_next), 1e-12)

        peak_before = self._episode_peak_nav

        log_ret = float(np.log(v_next / v_pre))
        clipped_ret = float(np.clip(log_ret, -MAX_STEP_LOG_RETURN, MAX_STEP_LOG_RETURN))

        # Equal-weight market return (frictionless benchmark)
        market_ret = float(np.log((close_next + 1e-12) / (close_t + 1e-12)).mean())
        self._return_buffer.append(log_ret)
        self._market_return_buffer.append(market_ret)

        # ── reward: return + benchmark-relative Sortino - inactivity ─
        reward = clipped_ret * REWARD_SCALE

        if len(self._return_buffer) >= RISK_WINDOW:
            def _sortino(rets: np.ndarray) -> float:
                m = float(rets.mean())
                ds = rets[rets < 0]
                dv = float(np.sqrt((ds ** 2).mean())) if len(ds) > 1 else 1e-8
                return m / (dv + 1e-8)

            agent_s = _sortino(np.array(self._return_buffer[-RISK_WINDOW:]))
            market_s = _sortino(np.array(self._market_return_buffer[-RISK_WINDOW:]))
            diff = float(np.clip(agent_s - market_s, -3.0, 3.0))
            reward += diff * RISK_BONUS_SCALE

        cash_frac = self._cash / max(v_next, 1e-12)
        if cash_frac > 0.50:
            reward -= INACTIVITY_PENALTY_OVER_50
        if cash_frac > 0.90:
            reward -= INACTIVITY_PENALTY_OVER_90

        gross_exposure = float(np.sum(w[1:]))
        reward += gross_exposure * PARTICIPATION_BONUS

        reward -= churn
        dd_frac = max(0.0, (peak_before - v_next) / max(peak_before, 1e-12))
        reward -= dd_frac * 10.0
        self._episode_peak_nav = max(peak_before, v_next)

        self._t += 1
        self._steps += 1

        terminated = bool(v_next <= STOP_LOSS_FRACTION * self._episode_start_nav)
        truncated = bool(self._t >= self._max_t or self._steps >= self._current_ep_max_steps)

        info: Dict[str, Any] = {"nav": v_next, "turnover": turnover_frac, "log_ret": log_ret}

        if terminated or truncated or self._t > self._max_t:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._build_obs()

        return obs, reward, terminated, truncated, info
