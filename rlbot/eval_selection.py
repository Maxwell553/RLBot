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
VIX_RISK_BASELINE = 18.0


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
) -> dict[str, float]:
    """Robust checkpoint score from one eval cycle's segment rollouts.

    When ``benchmark_ctx`` is set (recommended):

    return_signal = (1 - stitched_blend) * mean(segment excess ending NAV)
                    + stitched_blend * stitched_excess_nav
    score = return_signal - std_coef * std(segment excess)
            - dd_coef * p75(max_dd_nav)

    ``stitched_excess_nav`` compounds eval blocks chronologically (honest validation path).
    Also returns stitched validation NAV metrics when segment ``start_bar`` is present.
    """
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
    }
    if not episodes:
        return empty

    navs = np.asarray([float(e["ending_nav"]) for e in episodes], dtype=np.float64)
    dd_navs = np.asarray([float(e["max_drawdown_nav"]) for e in episodes], dtype=np.float64)
    dd_fracs = np.asarray([float(e.get("max_drawdown_frac", 0.0)) for e in episodes], dtype=np.float64)

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
    else:
        excess = navs.copy()
        mean_signal = float(np.mean(navs))
        std_signal = float(np.std(navs)) if navs.size > 1 else 0.0
        stitched = compute_stitched_eval_metrics(episodes, benchmark_ctx=benchmark_ctx)

    dd_p75 = float(np.percentile(dd_navs, 75)) if dd_navs.size else 0.0
    dd_frac_p75 = float(np.percentile(dd_fracs, 75)) if dd_fracs.size else 0.0
    score = mean_signal - float(std_coef) * std_signal - float(dd_coef) * dd_p75

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
    }
    if benchmark_ctx is not None:
        out["return_signal"] = mean_signal
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
) -> dict[str, Any]:
    """Portfolio panel summary + per-segment NAV stats for one eval cycle."""
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
