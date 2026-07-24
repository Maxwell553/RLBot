"""In-training eval selection score and portfolio diagnostics (torch-free)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from rlbot.baselines import balanced_6040_nav, equal_weight_daily_cost_aware_nav
from rlbot.portfolio_metrics import summarize_weight_panel

EXPOSURE_RISK_MODES = frozenset({"realized_vol", "vix_positive"})
EVAL_BENCHMARK_MODES = frozenset({"balanced_6040", "equal_weight_daily"})
BEST_MODEL_SCORE_MODES = frozenset({"excess_nav", "excess_sharpe", "defensive_sharpe"})
VIX_RISK_BASELINE = 18.0
TRADING_DAYS_PER_YEAR = 252
# Segment Sharpe clip: near-zero-vol excess paths (agent ≈ benchmark) otherwise
# produce unbounded ratios that would dominate the selection score.
EXCESS_SHARPE_CLIP = 10.0


def apply_dd_exposure_taper(
    weights: np.ndarray,
    dd_frac: float,
    *,
    start: float,
    end: float,
    min_gross: float,
) -> np.ndarray:
    """Cap risky gross exposure as episode drawdown deepens.

    Allowed gross goes from 1.0 at ``dd_frac <= start`` down to ``min_gross`` at
    ``dd_frac >= end`` (linear). If current gross already sits under the cap,
    weights are unchanged; otherwise risky weights are scaled to the allowed
    gross and the residual is parked in cash. No-op when ``end <= start``.
    """
    w = np.asarray(weights, dtype=np.float64).reshape(-1).copy()
    if w.size < 2:
        return w
    s = float(start)
    e = float(end)
    floor = float(np.clip(min_gross, 0.0, 1.0))
    if e <= s:
        return w
    dd = float(max(dd_frac, 0.0))
    if dd <= s:
        allowed = 1.0
    elif dd >= e:
        allowed = floor
    else:
        t = (dd - s) / (e - s)
        allowed = 1.0 + t * (floor - 1.0)
    risky = w[1:]
    gross = float(np.sum(risky))
    if gross <= allowed + 1e-12 or gross <= 1e-12:
        return w
    scaled = risky * (allowed / gross)
    out = np.empty_like(w)
    out[0] = max(0.0, 1.0 - float(np.sum(scaled)))
    out[1:] = scaled
    ssum = float(np.sum(out))
    if ssum > 1e-12:
        out /= ssum
    else:
        out[:] = 0.0
        out[0] = 1.0
    return out


def apply_episode_burn_in(
    episode: Mapping[str, Any],
    burn_in_bars: int,
) -> dict[str, Any]:
    """Drop the first ``burn_in_bars`` steps so cash-restart transients do not dominate.

    ``nav_path`` is ``[start_nav, nav_after_step1, ...]``. Skipping ``B`` steps keeps
    ``nav_path[B:]`` (post-burn-in start) and ``weights[B:]`` when present. Drawdowns
    and ``ending_nav`` are recomputed on the trimmed path. Episodes that are too short
    after burn-in are returned unchanged.
    """
    b = int(burn_in_bars)
    if b <= 0:
        return dict(episode)
    nav = [float(x) for x in episode.get("nav_path", [])]
    if len(nav) <= b + 2:
        return dict(episode)
    new_nav = np.asarray(nav[b:], dtype=np.float64)
    peak = np.maximum.accumulate(new_nav)
    dd_nav = peak - new_nav
    out = dict(episode)
    out["nav_path"] = new_nav.tolist()
    out["start_nav"] = float(new_nav[0])
    out["ending_nav"] = float(new_nav[-1])
    out["max_drawdown_nav"] = float(np.max(dd_nav))
    out["max_drawdown_frac"] = float(np.max(dd_nav / np.maximum(peak, 1e-12)))
    if episode.get("start_bar") is not None:
        out["start_bar"] = int(episode["start_bar"]) + b
    weights = episode.get("weights")
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim == 2 and w.shape[0] > b:
            out["weights"] = w[b:]
        else:
            out["weights"] = w
    return out


def apply_episodes_burn_in(
    episodes: list[Mapping[str, Any]],
    burn_in_bars: int,
) -> list[dict[str, Any]]:
    """Apply :func:`apply_episode_burn_in` to each episode."""
    return [apply_episode_burn_in(ep, burn_in_bars) for ep in episodes]


def blend_block_and_oos_aligned_scores(
    block_score: float,
    oos_aligned_score: float | None,
    *,
    weight: float,
) -> float:
    """Blend alternating-block and continuous OOS-aligned selection scores.

    ``weight`` is the continuous-path weight in ``[0, 1]``. When the continuous
    score is missing, returns ``block_score`` unchanged.
    """
    w = float(np.clip(weight, 0.0, 1.0))
    if oos_aligned_score is None or w <= 0.0:
        return float(block_score)
    if w >= 1.0:
        return float(oos_aligned_score)
    return float((1.0 - w) * float(block_score) + w * float(oos_aligned_score))


MULTI_REGIME_SCORE_AGGS = frozenset({"p25", "median", "min", "mean"})


def aggregate_multi_regime_scores(
    scores: list[float] | np.ndarray,
    *,
    agg: str = "p25",
) -> float:
    """Collapse per-regime continuous scores into one selection signal.

    ``p25`` (default for cohort 727) is intentionally pessimistic: a checkpoint
    must hold up on the harder regimes, not just average into a bull slice.
    """
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return float("-inf")
    arr = arr[np.isfinite(arr)]
    mode = str(agg).lower()
    if mode not in MULTI_REGIME_SCORE_AGGS:
        raise ValueError(
            f"multi-regime agg must be one of {sorted(MULTI_REGIME_SCORE_AGGS)}, got {agg!r}"
        )
    if mode == "p25":
        return float(np.percentile(arr, 25))
    if mode == "median":
        return float(np.median(arr))
    if mode == "min":
        return float(np.min(arr))
    return float(np.mean(arr))


def _episode_excess_log_returns(
    episode: Mapping[str, Any],
    ctx: EvalBenchmarkContext,
) -> np.ndarray:
    """Daily excess log returns (agent − benchmark) for one eval segment."""
    nav = np.asarray(episode.get("nav_path", []), dtype=np.float64)
    if nav.size < 3:
        return np.empty(0, dtype=np.float64)
    bench = benchmark_nav_path_for_episode(episode, ctx)
    if bench.size != nav.size:
        return np.empty(0, dtype=np.float64)
    agent_r = np.diff(np.log(np.maximum(nav, 1e-12)))
    bench_r = np.diff(np.log(np.maximum(bench, 1e-12)))
    return agent_r - bench_r


def annualized_sharpe(
    daily_returns: np.ndarray,
    *,
    clip: float = EXCESS_SHARPE_CLIP,
) -> float:
    """Annualized Sharpe of a daily-return series, clipped to ±clip."""
    r = np.asarray(daily_returns, dtype=np.float64)
    if r.size < 2:
        return 0.0
    std = float(np.std(r))
    if std < 1e-12:
        return float(np.clip(np.sign(float(np.mean(r))) * clip, -clip, clip))
    sharpe = float(np.mean(r)) / std * float(np.sqrt(TRADING_DAYS_PER_YEAR))
    return float(np.clip(sharpe, -clip, clip))


@dataclass(frozen=True)
class EvalBenchmarkContext:
    """Panel data for passive benchmark NAV paths on eval segments."""

    ohlcv: np.ndarray
    idx: pd.DatetimeIndex
    tickers: list[str]
    asset_live: np.ndarray | None = None
    mode: str = "balanced_6040"
    fee_scale: float = 1.0


def benchmark_nav_path_for_episode(
    episode: Mapping[str, Any],
    ctx: EvalBenchmarkContext,
) -> np.ndarray:
    """Passive benchmark NAV path aligned with an eval segment's ``nav_path`` length."""
    nav_path = np.asarray(episode.get("nav_path", []), dtype=np.float64)
    if nav_path.size < 2:
        return nav_path.copy()
    start_bar = episode.get("start_bar")
    if start_bar is None:
        return nav_path.copy()
    mode = str(ctx.mode)
    if mode not in EVAL_BENCHMARK_MODES:
        raise ValueError(
            f"eval benchmark mode must be one of {sorted(EVAL_BENCHMARK_MODES)}, got {mode!r}"
        )
    template = nav_path.copy()
    if mode == "balanced_6040":
        return balanced_6040_nav(
            template,
            ctx.ohlcv,
            int(start_bar),
            ctx.idx,
            ctx.tickers,
            asset_live=ctx.asset_live,
            fee_scale=float(ctx.fee_scale),
            apply_costs=True,
        )
    return equal_weight_daily_cost_aware_nav(
        template,
        ctx.ohlcv,
        int(start_bar),
        asset_live=ctx.asset_live,
        fee_scale=float(ctx.fee_scale),
    )


def compute_stitched_eval_metrics(
    episodes: list[Mapping[str, Any]],
    *,
    benchmark_ctx: EvalBenchmarkContext | None = None,
    initial_cash: float = 100_000.0,
) -> dict[str, float | list[float]]:
    """Compound eval-block returns into one continuous validation NAV path."""
    usable = [
        e
        for e in episodes
        if e.get("start_bar") is not None and len(e.get("nav_path", [])) >= 2
    ]
    if not usable:
        return {}
    usable.sort(key=lambda e: int(e["start_bar"]))
    agent = float(initial_cash)
    bench = float(initial_cash)
    path_a = [agent]
    path_b = [bench]
    for ep in usable:
        nav = np.asarray(ep["nav_path"], dtype=np.float64)
        agent *= float(nav[-1] / max(nav[0], 1e-12))
        if benchmark_ctx is not None:
            bnav = benchmark_nav_path_for_episode(ep, benchmark_ctx)
            if bnav.size >= 2:
                bench *= float(bnav[-1] / max(bnav[0], 1e-12))
        path_a.append(agent)
        path_b.append(bench)
    path_a_arr = np.asarray(path_a, dtype=np.float64)
    peak = np.maximum.accumulate(path_a_arr)
    dd_nav = peak - path_a_arr
    return {
        "stitched_agent_nav": float(path_a_arr[-1]),
        "stitched_bench_nav": float(path_b[-1]),
        "stitched_excess_nav": float(path_a_arr[-1] - path_b[-1]),
        "stitched_max_drawdown_frac": float(
            np.max(dd_nav / np.maximum(peak, 1e-12))
        ),
        "stitched_max_drawdown_nav": float(np.max(dd_nav)),
        "stitched_nav_path": path_a,
        "stitched_bench_path": path_b,
    }


def compute_robust_eval_score(
    episodes: list[Mapping[str, Any]],
    *,
    std_coef: float = 0.75,
    dd_coef: float = 2.0,
    stitched_blend: float = 0.5,
    benchmark_ctx: EvalBenchmarkContext | None = None,
    score_mode: str = "excess_nav",
    burn_in_bars: int = 0,
) -> dict[str, float]:
    """Robust checkpoint score from one eval cycle's segment rollouts.

    ``score_mode="excess_nav"`` (legacy), with ``benchmark_ctx`` set:

    return_signal = (1 - stitched_blend) * mean(segment excess ending NAV)
                    + stitched_blend * stitched_excess_nav
    score = return_signal - std_coef * std(segment excess)
            - dd_coef * p75(max_dd_nav)

    ``score_mode="excess_sharpe"`` (requires ``benchmark_ctx``): the return signal is
    the annualized Sharpe of daily excess returns vs the benchmark — a blend of the
    mean per-segment Sharpe and the Sharpe of all segments' daily excess pooled
    chronologically — and the drawdown penalty uses the unitless
    ``p75(max_dd_frac)``:

    return_signal = (1 - stitched_blend) * mean(segment excess Sharpe)
                    + stitched_blend * pooled excess Sharpe
    score = return_signal - std_coef * std(segment excess Sharpe)
            - dd_coef * p75(max_dd_frac)

    ``score_mode="defensive_sharpe"`` (cohort 728+): same excess-Sharpe return signal,
    but the drawdown penalty uses ``max(max_dd_frac)`` instead of p75 so a single
    deep left-tail episode cannot hide behind a milder quartile. Intended to prefer
    shallow-DD checkpoints for walk-forward windows with crash risk (W2–W5).

    ``stitched_excess_nav`` compounds eval blocks chronologically (honest validation path).
    Also returns stitched validation NAV metrics when segment ``start_bar`` is present.

    ``burn_in_bars`` drops the first N steps of each episode before scoring so
    cash-restart deploy transients do not dominate (esp. short alternating blocks).
    """
    if int(burn_in_bars) > 0:
        episodes = apply_episodes_burn_in(episodes, int(burn_in_bars))
    mode = str(score_mode)
    if mode not in BEST_MODEL_SCORE_MODES:
        raise ValueError(
            f"score_mode must be one of {sorted(BEST_MODEL_SCORE_MODES)}, got {mode!r}"
        )
    empty = {
        "score": float("-inf"),
        "mean_ending_nav": float("nan"),
        "mean_excess_nav": float("nan"),
        "std_ending_nav": float("nan"),
        "std_excess_nav": float("nan"),
        "mean_max_drawdown_nav": float("nan"),
        "p75_max_drawdown_nav": float("nan"),
        "mean_max_drawdown_frac": float("nan"),
        "p75_max_drawdown_frac": float("nan"),
        "max_max_drawdown_frac": float("nan"),
    }
    if not episodes:
        return empty

    navs = np.asarray([float(e["ending_nav"]) for e in episodes], dtype=np.float64)
    dd_navs = np.asarray([float(e["max_drawdown_nav"]) for e in episodes], dtype=np.float64)
    dd_fracs = np.asarray([float(e.get("max_drawdown_frac", 0.0)) for e in episodes], dtype=np.float64)
    sharpe_modes = ("excess_sharpe", "defensive_sharpe")

    sharpe_metrics: dict[str, float] = {}
    if benchmark_ctx is not None:
        excess = np.asarray(
            [
                float(np.asarray(e["nav_path"], dtype=np.float64)[-1])
                - float(benchmark_nav_path_for_episode(e, benchmark_ctx)[-1])
                if len(e.get("nav_path", [])) >= 1
                else float(e["ending_nav"])
                for e in episodes
            ],
            dtype=np.float64,
        )
        mean_excess = float(np.mean(excess))
        std_signal = float(np.std(excess)) if excess.size > 1 else 0.0
        stitched = compute_stitched_eval_metrics(episodes, benchmark_ctx=benchmark_ctx)
        stitched_excess = float(stitched.get("stitched_excess_nav", mean_excess))
        blend = float(np.clip(stitched_blend, 0.0, 1.0))
        mean_signal = (1.0 - blend) * mean_excess + blend * stitched_excess
        if mode in sharpe_modes:
            ordered = sorted(
                episodes,
                key=lambda e: int(e["start_bar"]) if e.get("start_bar") is not None else 0,
            )
            per_segment = [
                _episode_excess_log_returns(e, benchmark_ctx) for e in ordered
            ]
            usable = [r for r in per_segment if r.size >= 2]
            if usable:
                seg_sharpes = np.asarray(
                    [annualized_sharpe(r) for r in usable], dtype=np.float64
                )
                pooled_sharpe = annualized_sharpe(np.concatenate(usable))
                mean_signal = (1.0 - blend) * float(np.mean(seg_sharpes)) + blend * pooled_sharpe
                std_signal = float(np.std(seg_sharpes)) if seg_sharpes.size > 1 else 0.0
                sharpe_metrics = {
                    "segment_excess_sharpe_mean": float(np.mean(seg_sharpes)),
                    "segment_excess_sharpe_std": std_signal,
                    "pooled_excess_sharpe": pooled_sharpe,
                }
            # else: no segment has enough bars — fall back to the excess-NAV signal
            # computed above (still with the unitless drawdown penalty below).
    else:
        excess = navs.copy()
        mean_signal = float(np.mean(navs))
        std_signal = float(np.std(navs)) if navs.size > 1 else 0.0
        stitched = compute_stitched_eval_metrics(episodes, benchmark_ctx=benchmark_ctx)

    dd_p75 = float(np.percentile(dd_navs, 75)) if dd_navs.size else 0.0
    dd_frac_p75 = float(np.percentile(dd_fracs, 75)) if dd_fracs.size else 0.0
    dd_frac_max = float(np.max(dd_fracs)) if dd_fracs.size else 0.0
    if mode == "defensive_sharpe":
        dd_term = dd_frac_max
    elif mode == "excess_sharpe":
        dd_term = dd_frac_p75
    else:
        dd_term = dd_p75
    score = mean_signal - float(std_coef) * std_signal - float(dd_coef) * dd_term

    out: dict[str, float] = {
        "score": score,
        "mean_ending_nav": float(np.mean(navs)),
        "mean_excess_nav": float(np.mean(excess)) if benchmark_ctx is not None else float(np.mean(navs)),
        "std_ending_nav": float(np.std(navs)) if navs.size > 1 else 0.0,
        "std_excess_nav": std_signal if benchmark_ctx is not None else float(np.std(navs)) if navs.size > 1 else 0.0,
        "mean_max_drawdown_nav": float(np.mean(dd_navs)),
        "p75_max_drawdown_nav": dd_p75,
        "mean_max_drawdown_frac": float(np.mean(dd_fracs)),
        "p75_max_drawdown_frac": dd_frac_p75,
        "max_max_drawdown_frac": dd_frac_max,
    }
    if benchmark_ctx is not None:
        out["return_signal"] = mean_signal
    out.update(sharpe_metrics)
    for k, v in stitched.items():
        if isinstance(v, (int, float)) and np.isfinite(float(v)):
            out[k] = float(v)
    return out


def aggregate_eval_portfolio_diagnostics(
    episodes: list[Mapping[str, Any]],
    *,
    tickers: list[str],
    max_single_asset_weight: float,
    benchmark_ctx: EvalBenchmarkContext | None = None,
    burn_in_bars: int = 0,
) -> dict[str, Any]:
    """Portfolio panel summary + per-segment NAV stats for one eval cycle."""
    if int(burn_in_bars) > 0:
        episodes = apply_episodes_burn_in(episodes, int(burn_in_bars))
    weights_blocks = [
        np.asarray(e["weights"], dtype=np.float64) for e in episodes if e.get("weights") is not None
    ]
    if weights_blocks:
        weights = np.vstack(weights_blocks)
    else:
        weights = np.zeros((0, 1), dtype=np.float64)

    panel = summarize_weight_panel(
        weights,
        tickers=tickers,
        max_single_asset_weight=max_single_asset_weight,
    )
    segments = []
    for i, ep in enumerate(episodes):
        nav_path = [float(x) for x in ep.get("nav_path", [])]
        seg: dict[str, Any] = {
            "segment_index": i,
            "start_bar": ep.get("start_bar"),
            "start_nav": float(ep.get("start_nav", nav_path[0] if nav_path else 0.0)),
            "ending_nav": float(ep["ending_nav"]),
            "max_drawdown_frac": float(ep.get("max_drawdown_frac", 0.0)),
            "max_drawdown_nav": float(ep.get("max_drawdown_nav", 0.0)),
            "n_bars": len(nav_path),
            "nav_path": nav_path,
        }
        if benchmark_ctx is not None and nav_path:
            bench_path = benchmark_nav_path_for_episode(ep, benchmark_ctx)
            seg["bench_nav_path"] = [float(x) for x in bench_path.tolist()]
            seg["excess_ending_nav"] = float(nav_path[-1] - bench_path[-1])
        segments.append(seg)

    stitched = compute_stitched_eval_metrics(episodes, benchmark_ctx=benchmark_ctx)
    return {"portfolio": panel, "segments": segments, "stitched": stitched}


def append_eval_diagnostics_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def exposure_risk_penalty(
    *,
    gross_exposure: float,
    agent_returns: np.ndarray,
    vix: float,
    mode: str,
    scale: float,
) -> float:
    """Penalize gross exposure in high-vol regimes (realized or VIX-positive)."""
    if scale <= 0.0:
        return 0.0
    m = str(mode)
    if m not in EXPOSURE_RISK_MODES:
        raise ValueError(f"exposure_risk_mode must be one of {sorted(EXPOSURE_RISK_MODES)}, got {m!r}")
    if m == "vix_positive":
        z = max((float(vix) - VIX_RISK_BASELINE) / VIX_RISK_BASELINE, 0.0)
        return float(gross_exposure * z * scale)
    if agent_returns.size < 2:
        return 0.0
    vol = float(np.std(np.asarray(agent_returns, dtype=np.float64)))
    return float(gross_exposure * vol * scale)
