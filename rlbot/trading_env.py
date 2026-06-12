"""
Multi-asset portfolio Gymnasium environment (universe size from OHLCV panel / config).

Reward = return (drawdown amp) + benchmark excess + Sortino diff + participation
  - inactivity - cost-linked churn - drawdown penalty - concentration penalty
  - return: clipped_log_return * REWARD_SCALE; negative returns amplified by (1 + gamma * dd_pre)
  - drawdown penalty: dd_increase * reward_scale * drawdown_increase_penalty
    + max(dd_next - drawdown_level_floor, 0) * drawdown_level_penalty
  - concentration: concentration_penalty * max(target_eff_n - eff_n, 0) on risky weights
  - cash accrual (optional): cash *= (1 + cash_daily_yield) when yield > 0
  - benchmark excess: clip(agent_log_ret - cap_weight_bench_log_ret) * benchmark_excess_scale
  - sortino diff: benchmark-relative Sortino over last RISK_WINDOW steps (moving window)
  - Sortino + benchmark capped at a constant |sortino+benchmark| <= benchmark_combined_abs_cap
  - inactivity: linear in cash fraction (plus extra ramp above 90%)
  - churn: realized tx cost (slippage + fees) * churn_penalty * reward_scale * VIX * curriculum scale
  - Soft per-asset long-only cap after softmax (see config max_single_asset_weight)

Execution: trades fill at open[t+1] (next morning after decision), not close[t].
  Combined with obs_lag, the pipeline is:
    observe close[t-obs_lag] → decide overnight → execute at open[t+1] → earn to close[t+1]

Domain randomization (training): ``obs_lag`` and ``fee_scale`` resampled each episode
after the fee curriculum releases; bounds widen progressively (see ``set_randomization_bounds``).
``fee_scale`` uses Beta(5, 5) mapped to the current fee bounds (bell curve centered at 1.0).
Fee curriculum overrides DR until release (see ``set_curriculum_state``).
Fracdiff panel snapshots at RETURN_HORIZONS lags (no second differencing on top of d=0.4 fracdiff).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rlbot.baselines import portfolio_step_nav
from rlbot.reward_terms import (
    concentration_penalty_from_weights,
    drawdown_penalty_from_nav,
    exposure_risk_penalty_from_state,
)
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
    asset_live: np.ndarray | None = None,
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

    if asset_live is not None:
        live = np.asarray(asset_live, dtype=np.float64).reshape(-1)
        if live.shape[0] != n_act - 1:
            raise ValueError(f"asset_live must have length {n_act - 1}, got {live.shape[0]}")
        w[1:] *= np.clip(live, 0.0, 1.0)
        w = _enforce_long_only_simplex(w)

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
    w = _enforce_long_only_simplex(w)
    # Final projection: guarantee the per-asset cap post-condition for arbitrary
    # cap/N (the 5-iteration redistribute above is best-effort). Park any residual
    # excess in cash so the result stays a long-only simplex summing to 1.
    risky = w[1:]
    over = risky > max_w + 1e-9
    if np.any(over):
        excess = float(np.sum(risky[over] - max_w))
        risky[over] = max_w
        w[1:] = risky
        w[0] += excess
    return w


def _cap_benchmark_components(
    *,
    sortino: float,
    benchmark: float,
    cap_abs: float,
) -> tuple[float, float]:
    """Scale Sortino + benchmark excess so |sortino + benchmark| ≤ ``cap_abs`` (constant).

    The cap is deliberately a CONSTANT, never a function of the other reward terms.
    The earlier relative cap (≤ share of |return|+|participation|+|inactivity|+|churn|)
    created a verified reward-hacking gradient: while the combined benchmark term sat at
    a positive cap, every other term's *magnitude* fed the reward with coefficient
    share/(1−share) > 1 — burning transaction costs or taking losses *raised* total
    reward. A constant cap keeps ∂reward/∂churn = −1 and ∂reward/∂|loss| < 0 always.

    ``cap_abs <= 0`` disables both terms (ablation switch).
    """
    if cap_abs <= 0.0:
        return 0.0, 0.0
    bench_abs = abs(sortino + benchmark)
    if bench_abs <= cap_abs or bench_abs < 1e-12:
        return sortino, benchmark
    scale = cap_abs / bench_abs
    return sortino * scale, benchmark * scale


class EpisodeEndNavRecorder(gym.Wrapper):
    """Record eval rollouts: ending NAV, segment paths, weights, and drawdowns."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._ending_navs: list[float] = []
        self._episodes: list[dict] = []
        self._cur_navs: list[float] = []
        self._cur_weights: list[np.ndarray] = []
        self._cur_start_nav: float | None = None
        self._cur_start_bar: int | None = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._cur_navs = []
        self._cur_weights = []
        self._cur_start_nav = None
        self._cur_start_bar = None
        unwrapped = self.env.unwrapped
        start_bar = getattr(unwrapped, "_t", None)
        if start_bar is not None:
            self._cur_start_bar = int(start_bar)
        start_nav = getattr(unwrapped, "_episode_start_nav", None)
        if start_nav is None:
            start_nav = getattr(unwrapped, "initial_cash", None)
        if start_nav is not None and np.isfinite(float(start_nav)):
            start_nav = float(start_nav)
            self._cur_start_nav = start_nav
            self._cur_navs.append(start_nav)
        return obs, info

    def _finalize_episode(self, ending_nav: float) -> dict:
        navs = np.asarray(self._cur_navs, dtype=np.float64)
        if navs.size:
            peak = np.maximum.accumulate(navs)
            dd_nav = peak - navs
            max_dd_frac = float(np.max(dd_nav / np.maximum(peak, 1e-12)))
            max_dd_nav = float(np.max(dd_nav))
            start_nav = float(self._cur_start_nav if self._cur_start_nav is not None else navs[0])
        else:
            max_dd_frac = 0.0
            max_dd_nav = 0.0
            start_nav = float(ending_nav)
        weights = (
            np.stack(self._cur_weights, axis=0)
            if self._cur_weights
            else np.zeros((0, 1), dtype=np.float64)
        )
        return {
            "ending_nav": float(ending_nav),
            "start_nav": start_nav,
            "start_bar": self._cur_start_bar,
            "max_drawdown_frac": max_dd_frac,
            "max_drawdown_nav": max_dd_nav,
            "nav_path": navs.tolist(),
            "weights": weights,
        }

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        nav = info.get("nav")
        tw = info.get("target_weights")
        if nav is not None:
            nav_f = float(nav)
            if self._cur_start_nav is None:
                self._cur_start_nav = nav_f
            self._cur_navs.append(nav_f)
            if tw is not None:
                self._cur_weights.append(np.asarray(tw, dtype=np.float64).reshape(-1))
        if (terminated or truncated) and nav is not None:
            ep = self._finalize_episode(float(nav))
            self._episodes.append(ep)
            self._ending_navs.append(float(nav))
            self._cur_navs = []
            self._cur_weights = []
            self._cur_start_nav = None
            self._cur_start_bar = None
        return obs, reward, terminated, truncated, info

    def pop_ending_navs(self) -> list[float]:
        """Return and clear ending NAVs collected since the last pop (one eval cycle)."""
        navs = list(self._ending_navs)
        self._ending_navs.clear()
        return navs

    def pop_eval_episodes(self) -> list[dict]:
        """Return and clear full eval episode records (NAV paths, weights, drawdowns)."""
        eps = list(self._episodes)
        self._episodes.clear()
        self._ending_navs.clear()
        self._cur_navs = []
        self._cur_weights = []
        self._cur_start_nav = None
        self._cur_start_bar = None
        return eps

    def get_segments(self) -> Optional[list]:
        return self.env.get_segments()


class MultiAssetPortfolioEnv(gym.Env):
    """
    Observation size: ``10 * n_assets + 8 + 5 * N_MACRO`` (e.g. 128 when ``n_assets=10``).

    Action: Box(-3,3)^(n_assets+1) → softmax(cash + assets), long-only risky weights, per-asset cap.

    Reward: return (drawdown-amplified downside) + benchmark excess + Sortino diff
    + participation - inactivity - cost-linked churn - drawdown penalty - concentration.
    Per-step ``info`` includes ``rew_decomp/*`` for each component (see ``config.yaml`` reward section).

    Feature arrays after ``macd`` are keyword-only so walk-forward packs
    ``(ohlcv, rsi, macd, macro, fracdiff, fracdiff_macro, trend, ...)`` cannot be
    misaligned by positional unpacking.
    """

    metadata = {"render_modes": []}
    RETURN_HORIZONS = (1, 5, 10, 20)
    MIN_EPISODE_TRAIN_BARS = 21

    @staticmethod
    def noisy_market_feature_count(n_assets: int, n_macro: int = N_MACRO) -> int:
        """Features in ``_build_obs`` that receive domain-randomization noise (excludes ``asset_live``)."""
        h = len(MultiAssetPortfolioEnv.RETURN_HORIZONS)
        return (
            h * (n_assets + 1)
            + (n_assets + 1)
            + n_assets
            + n_assets
            + n_assets
            + h * n_macro
            + n_macro
        )

    def __init__(
        self,
        ohlcv: np.ndarray,
        rsi: np.ndarray,
        macd: np.ndarray,
        *,
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
        env_seed: int | None = None,
        block_boundaries: Optional[list] = None,
        obs_lag: int = 0,
        obs_lag_default: int | None = None,
        fee_scale_default: float | None = None,
        domain_randomize: bool = True,
        inactivity_penalty_scale: float = 1.0,
        action_smoothing_alpha: float | None = None,
        asset_live: np.ndarray | None = None,
        asset_realized_vol: np.ndarray | None = None,
        macro_realized_vol: np.ndarray | None = None,
        noise_scale: np.ndarray | None = None,
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
        if asset_live is not None:
            al = np.asarray(asset_live, dtype=np.float64)
            if al.shape != (ohlcv.shape[0], self.n_assets):
                raise ValueError(
                    f"asset_live shape {al.shape} != ({ohlcv.shape[0]}, {self.n_assets})"
                )
            self.asset_live = np.clip(al, 0.0, 1.0)
        else:
            self.asset_live = np.ones((ohlcv.shape[0], self.n_assets), dtype=np.float64)
        n_bars = ohlcv.shape[0]
        if asset_realized_vol is not None:
            av = np.asarray(asset_realized_vol, dtype=np.float64)
            if av.shape != (n_bars, self.n_assets):
                raise ValueError(f"asset_realized_vol shape {av.shape} != ({n_bars}, {self.n_assets})")
            self._asset_vol_panel = av
        else:
            self._asset_vol_panel = None
        if macro_realized_vol is not None:
            mv = np.asarray(macro_realized_vol, dtype=np.float64)
            if mv.shape != (n_bars, N_MACRO):
                raise ValueError(f"macro_realized_vol shape {mv.shape} != ({n_bars}, {N_MACRO})")
            self._macro_vol_panel = mv
        else:
            self._macro_vol_panel = None
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
        self._rng = np.random.default_rng(env_seed)
        self._reset_count = 0
        self._return_buffer: list[float] = []
        self._market_return_buffer: list[float] = []
        self._bench_nav = 1.0
        self._bench_w_prev: np.ndarray | None = None
        self._prev_target_w = np.zeros(self.n_actions, dtype=np.float64)
        self._prev_target_w[0] = 1.0
        alpha = (
            float(action_smoothing_alpha)
            if action_smoothing_alpha is not None
            else float(env_cfg.action_smoothing_alpha)
        )
        self._action_smoothing_alpha = float(np.clip(alpha, 0.0, 1.0))
        self._smoothed_action: np.ndarray | None = None

        n_live = self.n_assets
        n_port = self.n_actions
        n_meta = 2
        self._n_noisy_features = self.noisy_market_feature_count(self.n_assets, N_MACRO)
        self._n_market_features = self._n_noisy_features + n_live
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

        # Exclusive segment end (see ``_build_segments``); step() must not read past seg_end - 2.
        self._current_seg_end = self._max_t + 2

        if obs_noise_std > 0.0:
            if noise_scale is not None:
                ns = np.asarray(noise_scale, dtype=np.float32).reshape(-1)
                if ns.shape[0] != self._n_noisy_features:
                    raise ValueError(
                        f"noise_scale length {ns.shape[0]} != {self._n_noisy_features}"
                    )
                self._noise_scale = ns
            else:
                self._noise_scale = self.compute_obs_noise_scale(
                    self.ohlcv,
                    self.rsi,
                    self.macd,
                    self.fracdiff,
                    self.fracdiff_macro,
                    self.trend,
                    self.macro,
                    self._asset_vol_panel,
                    self._macro_vol_panel,
                    n_assets=self.n_assets,
                    n_noisy_features=self._n_noisy_features,
                    lookback=self.lookback,
                    return_horizons=self.RETURN_HORIZONS,
                    min_t=self._min_t,
                    max_t=self._max_t,
                    obs_lag=0,
                )
        else:
            self._noise_scale = None

    def set_curriculum_state(self, fee_override: Optional[float], churn_scale: float) -> None:
        """Called by ``TradingCurriculumCallback`` on train and eval envs.

        ``fee_override``: fixed ``fee_scale`` until next reset when set (0 = frictionless).
        ``None`` = release to domain randomization (or default fee) on reset.
        ``churn_scale``: multiplies churn and turnover penalties (0 = off during fee-free).
        """
        self._curriculum_fee_override = fee_override
        self._churn_scale = float(churn_scale)
        if not self.domain_randomize:
            if fee_override is not None:
                self.fee_scale = float(fee_override)
            else:
                self.fee_scale = self._fee_scale_default

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

    @staticmethod
    def compute_obs_noise_scale(
        ohlcv: np.ndarray,
        rsi: np.ndarray,
        macd: np.ndarray,
        fracdiff: np.ndarray,
        fracdiff_macro: np.ndarray,
        trend: np.ndarray,
        macro: np.ndarray,
        asset_realized_vol: np.ndarray | None,
        macro_realized_vol: np.ndarray | None,
        *,
        n_assets: int,
        n_noisy_features: int,
        lookback: int,
        return_horizons: tuple[int, ...],
        min_t: int,
        max_t: int,
        obs_lag: int = 0,
    ) -> np.ndarray:
        """Per-feature std for domain-randomization noise (compute once per training panel)."""
        n_samples = min(2000, max_t - min_t)
        indices = np.linspace(min_t + 1, max_t, n_samples, dtype=int)

        samples = []
        for t_raw in indices:
            t = max(int(t_raw) - obs_lag, 0)
            parts = []
            for h in return_horizons:
                t0 = max(t - h, 0)
                fd = fracdiff[t0]
                parts.append(fd.astype(np.float32) * 100.0)
                parts.append(np.array([fd.mean()], dtype=np.float32) * 100.0)

            if asset_realized_vol is not None:
                vol = asset_realized_vol[t]
            else:
                start = max(t - lookback, 1)
                closes_w = ohlcv[start : t + 1, :, 3]
                if len(closes_w) >= 2:
                    vol = np.diff(np.log(closes_w + 1e-12), axis=0).std(axis=0)
                else:
                    vol = np.zeros(n_assets)
            parts.append(vol.astype(np.float32) * 100.0)
            parts.append(np.array([vol.mean()], dtype=np.float32) * 100.0)

            parts.append((rsi[t] / 50.0 - 1.0).astype(np.float32))
            parts.append(np.tanh(macd[t]).astype(np.float32))
            parts.append(np.clip(trend[t], -1.0, 1.0).astype(np.float32))

            for h in return_horizons:
                t0 = max(t - h, 0)
                mfd = fracdiff_macro[t0]
                parts.append(mfd.astype(np.float32) * 100.0)
            if macro_realized_vol is not None:
                m_vol = macro_realized_vol[t]
            else:
                m_start = max(t - lookback, 1)
                m_vals = macro[m_start : t + 1]
                if len(m_vals) >= 2:
                    m_vol = np.diff(np.log(m_vals + 1e-12), axis=0).std(axis=0)
                else:
                    m_vol = np.zeros(N_MACRO)
            parts.append(m_vol.astype(np.float32) * 100.0)

            samples.append(np.concatenate(parts))

        feature_stds = np.stack(samples).std(axis=0)
        if feature_stds.shape[0] != n_noisy_features:
            raise ValueError(
                f"noise sample width {feature_stds.shape[0]} != n_noisy_features {n_noisy_features}"
            )
        return np.maximum(feature_stds, 0.01).astype(np.float32)

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
            asset_fd = self.fracdiff[t0].astype(np.float32)
            parts.append(asset_fd * 100.0)
            parts.append(np.array([asset_fd.mean()], dtype=np.float32) * 100.0)

        if self._asset_vol_panel is not None:
            asset_vol = self._asset_vol_panel[t_mkt].astype(np.float32)
        else:
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
            mfd = self.fracdiff_macro[t0].astype(np.float32)
            parts.append(mfd * 100.0)
        if self._macro_vol_panel is not None:
            macro_vol = self._macro_vol_panel[t_mkt].astype(np.float32)
        else:
            macro_vol = self._macro_realized_vol(t_mkt).astype(np.float32)
        parts.append(macro_vol * 100.0)

        parts.append(self.asset_live[t_mkt].astype(np.float32))

        close = self.ohlcv[t, :, 3]
        parts.append(self._portfolio_weights(close))

        nav = self._nav(close)
        dd = (nav - self._episode_peak_nav) / max(self._episode_peak_nav, 1e-12)
        progress = self._steps / max(self._current_ep_max_steps, 1)
        parts.append(np.array([dd, progress], dtype=np.float32))

        obs = np.concatenate(parts)

        if self.obs_noise_std > 0.0 and self._noise_scale is not None:
            noise = self._rng.normal(0.0, 1.0, size=self._n_noisy_features).astype(np.float32)
            noise *= self._noise_scale[: self._n_noisy_features] * self.obs_noise_std
            obs[: self._n_noisy_features] += noise

        return obs

    # ── execution ────────────────────────────────────────────────────────

    def _compute_sortino(self, rets: np.ndarray) -> float:
        m = float(rets.mean())
        downside_elements = np.minimum(rets, 0.0) ** 2
        # The floor is the economic resolution of "downside": without it, any no-loss
        # window (cash returns are exactly 0) divides by ~0 and saturates the clipped
        # Sortino differential, turning the risk bonus into a binary exploit.
        dv = max(
            float(np.sqrt(downside_elements.mean())),
            float(self._reward_cfg.sortino_downside_floor),
        )
        return m / dv

    def _benchmark_weights_live(self, live: np.ndarray) -> np.ndarray:
        w = self._benchmark_weights * np.asarray(live, dtype=np.float64).reshape(-1)
        total = float(w.sum())
        if total > 1e-12:
            return w / total
        n_live = int(np.sum(live > 0.5))
        if n_live < 1:
            return np.zeros(self.n_assets, dtype=np.float64)
        out = np.zeros(self.n_assets, dtype=np.float64)
        out[live > 0.5] = 1.0 / n_live
        return out

    def _benchmark_log_return_step(self, t: int, live: np.ndarray) -> float:
        """Cap-weight benchmark log return with same friction model as ``portfolio_step_nav``."""
        w_tgt = self._benchmark_weights_live(live)
        nav_pre = max(self._bench_nav, 1e-12)
        nav_next = portfolio_step_nav(
            nav_pre,
            self.ohlcv,
            t,
            w_tgt,
            prev_weights=self._bench_w_prev,
            slippage=self._asset_slippage,
            tx_fee=self._asset_tx_fee,
            daily_holding=self._daily_holding_cost,
            fee_scale=self.fee_scale,
            asset_live=self.asset_live,
        )
        self._bench_nav = nav_next
        self._bench_w_prev = w_tgt.copy()
        return float(np.log(max(nav_next, 1e-12) / nav_pre))

    def _rebalance(self, price: np.ndarray, target_w: np.ndarray) -> Tuple[float, float]:
        """Execute trades at given prices with per-asset slippage and fees.

        Scales transaction costs by self.fee_scale (for curriculum learning).
        Target weights are long-only (cash + nonnegative asset notionals).

        Returns ``(turnover_frac, tx_cost_frac)`` where ``tx_cost_frac`` is total
        slippage + fee dollars paid this step divided by pre-rebalance NAV.
        """
        tw = _enforce_long_only_simplex(np.asarray(target_w, dtype=np.float64))
        nav = self._nav(price)
        if nav <= 1e-12:
            return 0.0, 0.0

        fs = self.fee_scale
        target_units = (tw[1:] * nav) / (price + 1e-12)
        delta = target_units - self._units
        turnover = 0.0
        tx_cost = 0.0

        for i in np.argsort(delta):
            du = delta[i]
            if du >= -1e-12:
                continue
            sell_u = -du
            cost_rate = (self._asset_slippage[i] + self._asset_tx_fee[i]) * fs
            notional = sell_u * price[i]
            tx_cost += notional * cost_rate
            self._cash += notional * (1.0 - cost_rate)
            self._units[i] -= sell_u
            turnover += notional

        buy_idxs = np.where(delta > 1e-12)[0]
        if buy_idxs.size > 0:
            du = delta[buy_idxs]
            cost_rate = (self._asset_slippage[buy_idxs] + self._asset_tx_fee[buy_idxs]) * fs
            unit_cost = price[buy_idxs] * (1.0 + cost_rate)
            need = du * unit_cost
            buy_cash_need = float(need.sum())
            scale = (
                min(1.0, max(0.0, self._cash / (buy_cash_need + 1e-12)))
                if buy_cash_need > 1e-12
                else 1.0
            )
            buy_u = du * scale
            filled = buy_u > 1e-12
            if np.any(filled):
                idx_f = buy_idxs[filled]
                bu = buy_u[filled]
                uc = unit_cost[filled]
                cr = cost_rate[filled]
                pr = price[idx_f]
                tx_cost += float((bu * pr * cr).sum())
                self._cash -= float((bu * uc).sum())
                self._units[idx_f] += bu
                turnover += float((bu * pr).sum())

        self._units = np.maximum(self._units, 0.0)
        nav_denom = max(nav, 1e-12)
        return turnover / nav_denom, tx_cost / nav_denom

    def _apply_holding_costs(self, close: np.ndarray) -> float:
        """Deduct daily holding on pre-rebalance units (call before ``_rebalance``)."""
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
        self._bench_nav = 1.0
        self._bench_w_prev = None
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

        panel_end = self._max_t + 2
        self._current_seg_end = panel_end

        if not self.random_start and self._segments is not None:
            # Deterministic eval: one episode per segment, full contiguous block.
            seg_idx = self._reset_count % len(self._segments)
            earliest, seg_end = self._segments[seg_idx]
            self._current_seg_end = seg_end
            self._t = earliest
            bars_left = seg_end - self._t - 1
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
                self._current_seg_end = seg_end
                min_run = self.MIN_EPISODE_TRAIN_BARS
                latest = max(earliest, seg_end - 2 - min_run)
                self._t = int(self._rng.integers(earliest, latest + 1))
                bars_left = seg_end - self._t - 1
                self._current_ep_max_steps = min(self._current_ep_max_steps, bars_left)
                self._current_ep_max_steps = max(self._current_ep_max_steps, min_run)
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
        live_t = self.asset_live[max(self._t - self.obs_lag, 0)]
        w = portfolio_weights_from_action(
            action_for_weights,
            n_actions=self.n_actions,
            asset_live=live_t,
        )
        rwd = self._reward_cfg
        # Scale churn with live VIX (same series as obs macro block; undifferenced close).
        current_vix = float(self.macro[self._t, MACRO_VIX_INDEX])
        if current_vix > 1.0:
            vix_multiplier = float(
                np.clip(current_vix / VIX_CHURN_BASELINE, VIX_CHURN_MULT_MIN, VIX_CHURN_MULT_MAX)
            )
        else:
            vix_multiplier = 1.0
        active_churn_scale = vix_multiplier * self._churn_scale
        # Holding on pre-rebalance units at close[t] (matches portfolio_step_nav w_prev).
        self._apply_holding_costs(close_t)
        turnover_frac, tx_cost_frac = self._rebalance(open_next, w)
        self._prev_target_w = w.copy()

        if rwd.cash_daily_yield > 0.0:
            self._cash *= 1.0 + float(rwd.cash_daily_yield)

        close_next = self.ohlcv[self._t + 1, :, 3]
        v_next = max(self._nav(close_next), 1e-12)

        peak_before = self._episode_peak_nav
        dd_frac_pre = max(0.0, (peak_before - v_pre) / max(peak_before, 1e-12))

        log_ret = float(np.log(v_next / v_pre))
        clipped_ret = float(
            np.clip(
                log_ret,
                rwd.max_step_log_return_downside,
                rwd.max_step_log_return,
            )
        )

        market_ret = self._benchmark_log_return_step(self._t, live_t)
        self._return_buffer.append(log_ret)
        self._market_return_buffer.append(market_ret)

        # ── reward: return (drawdown-amplified downside) + Sortino - inactivity ─
        base_return = clipped_ret * rwd.reward_scale
        if clipped_ret < 0.0:
            amp = 1.0 + rwd.drawdown_downside_gamma * dd_frac_pre
            return_component = base_return * amp
            drawdown_component = base_return * (amp - 1.0)
        else:
            return_component = base_return
            drawdown_component = 0.0

        sortino_component = 0.0
        available_steps = len(self._return_buffer)
        if available_steps >= rwd.sortino_min_steps:
            active_lookback = min(available_steps, rwd.risk_window)
            agent_s = self._compute_sortino(
                np.array(self._return_buffer[-active_lookback:], dtype=np.float64)
            )
            market_s = self._compute_sortino(
                np.array(self._market_return_buffer[-active_lookback:], dtype=np.float64)
            )
            diff = float(np.clip(agent_s - market_s, -3.0, 3.0))
            sortino_component = diff * rwd.risk_bonus_scale

        excess_ret = float(
            np.clip(
                log_ret - market_ret,
                -rwd.benchmark_excess_clip,
                rwd.benchmark_excess_clip,
            )
        )
        benchmark_component = excess_ret * rwd.benchmark_excess_scale

        cash_frac = self._cash / max(v_next, 1e-12)
        inact = self._inactivity_penalty_scale
        inactivity_component = float(cash_frac * rwd.inactivity_penalty_over_50 * inact)
        if cash_frac > 0.90:
            inactivity_component += float(
                ((cash_frac - 0.90) / 0.10) * rwd.inactivity_penalty_over_90 * inact
            )
        gross_exposure = float(np.sum(w[1:]))
        participation_component = (
            gross_exposure * rwd.participation_bonus * rwd.participation_reward_scale
        )

        churn_component = (
            tx_cost_frac * rwd.churn_penalty * rwd.reward_scale * active_churn_scale
        )
        turnover_component = (
            turnover_frac * rwd.turnover_penalty * rwd.reward_scale * active_churn_scale
        )

        sortino_component, benchmark_component = _cap_benchmark_components(
            sortino=sortino_component,
            benchmark=benchmark_component,
            cap_abs=rwd.benchmark_combined_abs_cap,
        )

        drawdown_penalty_component, dd_next, _ = drawdown_penalty_from_nav(
            peak_before=peak_before,
            v_pre=v_pre,
            v_next=v_next,
            dd_frac_pre=dd_frac_pre,
            rwd=rwd,
        )
        concentration_component, eff_n = concentration_penalty_from_weights(w, rwd)
        active_returns = np.asarray(self._return_buffer[-min(len(self._return_buffer), rwd.risk_window) :], dtype=np.float64)
        exposure_risk_component = exposure_risk_penalty_from_state(
            gross_exposure=gross_exposure,
            agent_returns=active_returns,
            vix=current_vix,
            rwd=rwd,
        )

        reward = (
            return_component
            + sortino_component
            + benchmark_component
            + participation_component
            - inactivity_component
            - churn_component
            - turnover_component
            - drawdown_penalty_component
            - concentration_component
            - exposure_risk_component
        )

        self._episode_peak_nav = max(peak_before, v_next)

        self._t += 1
        self._steps += 1

        terminated = bool(
            v_next <= self._env_cfg.stop_loss_fraction * self._episode_start_nav
        )
        seg_bar_limit = self._current_seg_end - 1
        truncated = bool(
            self._t >= self._max_t
            or self._t >= seg_bar_limit
            or self._steps >= self._current_ep_max_steps
        )

        info: Dict[str, Any] = {
            "nav": v_next,
            "target_weights": w.copy(),
            "turnover": turnover_frac,
            "tx_cost_frac": tx_cost_frac,
            "log_ret": log_ret,
            "rew_decomp/return": return_component,
            "rew_decomp/benchmark": benchmark_component,
            "rew_decomp/sortino": sortino_component,
            "rew_decomp/inactivity": -inactivity_component,
            "rew_decomp/participation": participation_component,
            "rew_decomp/churn": -churn_component,
            "rew_decomp/turnover": -turnover_component,
            "rew_decomp/vix_churn_mult": vix_multiplier,
            "rew_decomp/drawdown": drawdown_component,
            "rew_decomp/drawdown_penalty": -drawdown_penalty_component,
            "rew_decomp/concentration": -concentration_component,
            "rew_decomp/exposure_risk": -exposure_risk_component,
            "rew_decomp/effective_n_assets": eff_n,
        }

        obs = self._build_obs()
        if terminated or truncated:
            info["terminal_observation"] = obs

        return obs, reward, terminated, truncated, info
