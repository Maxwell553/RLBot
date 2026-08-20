"""Written inference overlays on an already-trained book's target weights.

These run at backtest / live time only — never during PPO training. 809 already
allocates; it does not time crashes. Reward-shaping crash timing failed, so the
overlays scale or shrink 809's *executed* weights:

- vol-target gross: cut the risky book toward a 10–12% annualized vol *cap*
- 13w / 200d NAV trend: reduce gross only when the book's own NAV trend is down
- EW blend: ``w = (1-α)·w_809 + α·w_EW`` to shrink seed-lucky tilts

Kill an overlay if mean cash ≳ 25% or W1/W5 return collapses vs the raw 809 book.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from rlbot.two_head_actions import project_long_only_capped

TRADING_DAYS_PER_YEAR = 252
# 13 calendar weeks of daily bars; 200d is the slow SMA (GE1-style gate).
TREND_FAST_BARS = 65
TREND_SLOW_BARS = 200
DEFAULT_VOL_TARGET = 0.11  # midpoint of the 10–12% ann. cap
MIN_VOL_BARS = 20
CASH_KILL = 0.25
RETURN_COLLAPSE_FRAC = 0.90


@dataclass(frozen=True)
class OverlaySpec:
    """One written overlay (all fields optional / off when zero or None)."""

    name: str = "custom"
    vol_target: float | None = None  # annualized cap; None / <=0 disables
    vol_window: int = 63
    trend_fast: int | None = None  # SMA bars; None disables with trend_slow
    trend_slow: int | None = None
    trend_min_gross: float = 0.70  # floor when trend is down (not a cash park)
    ew_alpha: float = 0.0  # 0 disables
    max_single_asset_weight: float = 0.20

    def enabled(self) -> bool:
        return (
            (self.vol_target is not None and float(self.vol_target) > 0.0)
            or (
                self.trend_fast is not None
                and self.trend_slow is not None
                and int(self.trend_fast) > 0
                and int(self.trend_slow) > int(self.trend_fast)
            )
            or float(self.ew_alpha) > 1e-12
        )


def default_overlay_grid(*, max_single_asset_weight: float = 0.20) -> tuple[OverlaySpec, ...]:
    """The 809-compatible overlay cells to score on existing rollouts (no new train)."""
    cap = float(max_single_asset_weight)
    vt = DEFAULT_VOL_TARGET
    fast, slow = TREND_FAST_BARS, TREND_SLOW_BARS
    return (
        OverlaySpec(name="vol_target", vol_target=vt, max_single_asset_weight=cap),
        OverlaySpec(
            name="trend_13w_200d",
            trend_fast=fast,
            trend_slow=slow,
            max_single_asset_weight=cap,
        ),
        OverlaySpec(name="ew_blend_0.25", ew_alpha=0.25, max_single_asset_weight=cap),
        OverlaySpec(name="ew_blend_0.50", ew_alpha=0.50, max_single_asset_weight=cap),
        OverlaySpec(
            name="vol+trend",
            vol_target=vt,
            trend_fast=fast,
            trend_slow=slow,
            max_single_asset_weight=cap,
        ),
        OverlaySpec(
            name="vol+trend+ew_0.25",
            vol_target=vt,
            trend_fast=fast,
            trend_slow=slow,
            ew_alpha=0.25,
            max_single_asset_weight=cap,
        ),
        OverlaySpec(
            name="vol+trend+ew_0.50",
            vol_target=vt,
            trend_fast=fast,
            trend_slow=slow,
            ew_alpha=0.50,
            max_single_asset_weight=cap,
        ),
    )


def equal_weight_risky(
    n_assets: int, asset_live: np.ndarray | None = None
) -> np.ndarray:
    """Fully-invested 1/N on live assets (cash = 0)."""
    live = (
        np.clip(np.asarray(asset_live, dtype=np.float64).reshape(-1), 0.0, 1.0)
        if asset_live is not None
        else np.ones(int(n_assets), dtype=np.float64)
    )
    if live.shape[0] != n_assets:
        raise ValueError(f"asset_live must have length {n_assets}, got {live.shape[0]}")
    s = float(np.sum(live))
    if s <= 1e-12:
        return np.zeros(n_assets, dtype=np.float64)
    return live / s


def _scale_gross(w: np.ndarray, gross_cap: float) -> np.ndarray:
    """Cut risky gross to ``gross_cap`` (never lever up). Residual → cash."""
    out = np.asarray(w, dtype=np.float64).reshape(-1).copy()
    if out.size < 2:
        return out
    cap = float(np.clip(gross_cap, 0.0, 1.0))
    risky = out[1:]
    gross = float(np.sum(risky))
    if gross <= cap + 1e-12 or gross <= 1e-12:
        return out
    scaled = risky * (cap / gross)
    out[1:] = scaled
    out[0] = max(0.0, 1.0 - float(np.sum(scaled)))
    return out


def blend_with_equal_weight(
    w: np.ndarray,
    alpha: float,
    *,
    asset_live: np.ndarray | None = None,
) -> np.ndarray:
    """``(1-α)·w + α·w_EW`` then long-only renormalize (no cap yet)."""
    out = np.asarray(w, dtype=np.float64).reshape(-1).copy()
    a = float(np.clip(alpha, 0.0, 1.0))
    if a <= 1e-12 or out.size < 2:
        return out
    n = out.size - 1
    ew = np.zeros_like(out)
    ew[1:] = equal_weight_risky(n, asset_live=asset_live)
    mixed = (1.0 - a) * out + a * ew
    mixed = np.maximum(mixed, 0.0)
    s = float(np.sum(mixed))
    if s <= 1e-12:
        mixed[:] = 0.0
        mixed[0] = 1.0
        return mixed
    return mixed / s


class InferenceOverlay:
    """Causal overlay state: uses the *overlayed* book's own NAV path only."""

    def __init__(self, spec: OverlaySpec):
        self.spec = spec
        self._navs: list[float] = []

    def reset(self) -> None:
        self._navs.clear()

    def apply(
        self,
        weights: np.ndarray,
        *,
        nav: float,
        asset_live: np.ndarray | None = None,
    ) -> np.ndarray:
        """Observe pre-rebalance NAV, then overlay ``weights``. Call once per step."""
        self._navs.append(float(nav))
        w = np.asarray(weights, dtype=np.float64).reshape(-1).copy()
        spec = self.spec
        if spec.ew_alpha > 1e-12:
            w = blend_with_equal_weight(w, spec.ew_alpha, asset_live=asset_live)
        if spec.vol_target is not None and float(spec.vol_target) > 0.0:
            w = self._apply_vol_cap(w, float(spec.vol_target), int(spec.vol_window))
        if (
            spec.trend_fast is not None
            and spec.trend_slow is not None
            and int(spec.trend_fast) > 0
            and int(spec.trend_slow) > int(spec.trend_fast)
        ):
            w = self._apply_trend_gate(
                w, int(spec.trend_fast), int(spec.trend_slow), float(spec.trend_min_gross)
            )
        return project_long_only_capped(w, float(spec.max_single_asset_weight))

    def _apply_vol_cap(self, w: np.ndarray, target: float, window: int) -> np.ndarray:
        navs = np.asarray(self._navs, dtype=np.float64)
        if navs.size < MIN_VOL_BARS + 1:
            return w
        log_rets = np.diff(np.log(np.maximum(navs, 1e-12)))
        look = log_rets[-max(int(window), MIN_VOL_BARS) :]
        if look.size < MIN_VOL_BARS:
            return w
        realized = float(np.std(look, ddof=1)) * np.sqrt(TRADING_DAYS_PER_YEAR)
        if not np.isfinite(realized) or realized <= target + 1e-12:
            return w
        scale = float(target / realized)
        gross = float(np.sum(w[1:]))
        return _scale_gross(w, gross * scale)

    def _apply_trend_gate(
        self, w: np.ndarray, fast: int, slow: int, min_gross: float
    ) -> np.ndarray:
        navs = np.asarray(self._navs, dtype=np.float64)
        if navs.size < int(slow):
            return w
        sma_f = float(np.mean(navs[-int(fast) :]))
        sma_s = float(np.mean(navs[-int(slow) :]))
        if sma_f >= sma_s - 1e-12:
            return w
        return _scale_gross(w, float(np.clip(min_gross, 0.0, 1.0)))


def overlay_kill_reasons(
    *,
    baseline_return: float,
    overlay_return: float,
    mean_cash: float,
    cash_kill: float = CASH_KILL,
    return_floor_frac: float = RETURN_COLLAPSE_FRAC,
) -> list[str]:
    """Kill flags for overlay scoring (mean cash ≳ 25% or W1/W5-style collapse)."""
    reasons: list[str] = []
    if mean_cash >= float(cash_kill) - 1e-12:
        reasons.append(f"mean_cash={mean_cash:.1%} ≥ {cash_kill:.0%}")
    floor = float(baseline_return) * float(return_floor_frac)
    if overlay_return < floor - 1e-12:
        reasons.append(
            f"return={overlay_return:.1%} collapsed vs 809 {baseline_return:.1%} "
            f"(floor {return_floor_frac:.0%} of baseline)"
        )
    return reasons


def overlay_spec_to_dict(spec: OverlaySpec) -> dict:
    return asdict(spec)


def parse_overlay_names(raw: str) -> list[str]:
    """Comma-separated overlay tokens: ``vol_target``, ``trend``, ``ew_blend``."""
    if not raw or not str(raw).strip():
        return []
    allowed = {"vol_target", "trend", "ew_blend"}
    out: list[str] = []
    for tok in str(raw).split(","):
        name = tok.strip().lower().replace("-", "_")
        if not name:
            continue
        if name not in allowed:
            raise ValueError(
                f"unknown overlay {tok!r}; expected comma-separated "
                f"{sorted(allowed)}"
            )
        if name not in out:
            out.append(name)
    return out


def spec_from_cli(
    *,
    names: list[str],
    vol_target: float = DEFAULT_VOL_TARGET,
    trend_fast: int = TREND_FAST_BARS,
    trend_slow: int = TREND_SLOW_BARS,
    ew_alpha: float = 0.25,
    max_single_asset_weight: float = 0.20,
) -> OverlaySpec | None:
    """Build one overlay spec from CLI tokens. ``None`` if nothing enabled."""
    if not names:
        return None
    use_vol = "vol_target" in names
    use_trend = "trend" in names
    use_ew = "ew_blend" in names
    bits = []
    if use_vol:
        bits.append("vol_target")
    if use_trend:
        bits.append("trend_13w_200d")
    if use_ew:
        bits.append(f"ew_{ew_alpha:g}")
    spec = OverlaySpec(
        name="+".join(bits) if bits else "custom",
        vol_target=float(vol_target) if use_vol else None,
        trend_fast=int(trend_fast) if use_trend else None,
        trend_slow=int(trend_slow) if use_trend else None,
        ew_alpha=float(ew_alpha) if use_ew else 0.0,
        max_single_asset_weight=float(max_single_asset_weight),
    )
    return spec if spec.enabled() else None
