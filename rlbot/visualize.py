"""
Training and backtest charts (matplotlib, non-interactive PNG by default).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Headless-safe default; override with MPLBACKEND if you want an interactive GUI.
if os.environ.get("MPLBACKEND") is None:
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:  # Keep pure plotting helpers importable without the RL training stack installed.
    from stable_baselines3.common.callbacks import BaseCallback
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight envs
    class BaseCallback:  # type: ignore[no-redef]
        def __init__(self, *_, **__):
            raise ModuleNotFoundError(
                "stable_baselines3 is required for TrainingVizCallback; "
                "pure plotting helpers remain available."
            )

from rlbot.modal_cloud import mark_plot_saved


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) < window or window < 2:
        return x.astype(np.float64)
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def _k_formatter() -> mticker.FuncFormatter:
    """Format tick values in thousands: -20000 → -20, 4500 → 4.5, 500 → 0.5."""
    def fmt(v, _):
        if abs(v) < 100:
            return f"{v:.1f}"
        kv = v / 1000.0
        return f"{kv:.0f}" if kv == int(kv) else f"{kv:.1f}"
    return mticker.FuncFormatter(fmt)


def _dollar_formatter() -> mticker.FuncFormatter:
    """Format tick values as $100,000 → $100k, etc."""
    def fmt(v, _):
        if abs(v) >= 1_000_000:
            return f"${v / 1_000_000:.1f}M"
        if abs(v) >= 1_000:
            return f"${v / 1_000:.0f}k"
        return f"${v:.0f}"
    return mticker.FuncFormatter(fmt)


def _timestep_formatter() -> mticker.FuncFormatter:
    """Format timestep values: 1e9 → 1B, 1.5e6 → 1.5M, 1e3 → 1k."""
    def fmt(v, _):
        av = abs(v)
        if av >= 1e9:
            s = f"{v / 1e9:.1f}B"
        elif av >= 1e6:
            s = f"{v / 1e6:.1f}M"
        elif av >= 1e3:
            s = f"{v / 1e3:.0f}k"
        else:
            return f"{v:.0f}"
        return s.replace(".0M", "M").replace(".0B", "B")
    return mticker.FuncFormatter(fmt)


def _percent_formatter() -> mticker.FuncFormatter:
    def fmt(v, _):
        return f"{v:.1f}%"
    return mticker.FuncFormatter(fmt)


def robust_score_delta_from_best(scores: np.ndarray) -> np.ndarray:
    """``score[i] - max(score[: i + 1])`` — running best is always 0, later evals ≤ 0."""
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0:
        return s
    return s - np.maximum.accumulate(s)


def robust_score_delta_post_gate(
    scores: np.ndarray,
    timesteps: np.ndarray,
    gate_step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score delta with running max reset at ``gate_step``.

    Returns ``(pre_gate_mask, delta, post_gate_mask)``. Pre-gate entries in ``delta`` are
    NaN; post-gate uses ``score - max(score since gate)``. When ``gate_step <= 0``, all
    points use the full-series delta (no gate split).
    """
    s = np.asarray(scores, dtype=np.float64)
    t = np.asarray(timesteps, dtype=np.int64)
    if s.size == 0:
        empty = np.zeros(0, dtype=bool)
        return empty, s, empty
    gate = int(gate_step)
    if gate <= 0:
        delta = robust_score_delta_from_best(s)
        mask = np.ones(s.shape, dtype=bool)
        return np.zeros(s.shape, dtype=bool), delta, mask
    pre_mask = t < gate
    post_mask = ~pre_mask
    delta = np.full(s.shape, np.nan, dtype=np.float64)
    if post_mask.any():
        post_scores = s[post_mask]
        delta[post_mask] = post_scores - np.maximum.accumulate(post_scores)
    return pre_mask, delta, post_mask


def _scalar_from_npz(z, key: str) -> int | None:
    arr = z.get(key)
    if arr is None:
        return None
    flat = np.asarray(arr).reshape(-1)
    if flat.size == 0:
        return None
    return int(flat[0])


def resolve_eval_plot_milestones(
    hist: dict[str, np.ndarray],
    *,
    run_dir: Path | None = None,
) -> tuple[int | None, int | None]:
    """Return ``(best_model_min_step, best_eval_step)`` for training-plot annotations."""
    best_model_min_step: int | None = None
    best_eval_step: int | None = None
    if "best_model_min_step" in hist:
        best_model_min_step = int(hist["best_model_min_step"])
    if "best_eval_step" in hist:
        best_eval_step = int(hist["best_eval_step"])

    manifest: dict | None = None
    if run_dir is not None:
        manifest_path = Path(run_dir) / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = None

    if best_eval_step is None and manifest is not None:
        raw = manifest.get("best_eval_step")
        if raw is not None:
            best_eval_step = int(raw)

    if best_model_min_step is None:
        if manifest is not None and run_dir is not None:
            args = manifest.get("args") or {}
            learn_budget = args.get("timesteps")
            cfg_path = Path(run_dir) / "config.yaml"
            if learn_budget is not None and cfg_path.is_file():
                try:
                    from rlbot.rl_config import load_config, resolve_best_model_min_step

                    cfg = load_config(cfg_path)
                    best_model_min_step = resolve_best_model_min_step(
                        int(learn_budget), cur=cfg.curriculum
                    )
                except (ValueError, TypeError, OSError):
                    pass

    scores = hist.get("robust_scores")
    ts = hist.get("timesteps")
    if best_eval_step is None and scores is not None and ts is not None and len(scores) == len(ts):
        steps = np.asarray(ts, dtype=np.int64)
        navs = np.asarray(scores, dtype=np.float64)
        gate = int(best_model_min_step or 0)
        post = steps >= gate if gate > 0 else np.ones(len(steps), dtype=bool)
        if post.any():
            j = int(np.argmax(navs[post]))
            best_eval_step = int(steps[post][j])

    return best_model_min_step, best_eval_step


def _shade_best_model_eligible(
    ax: plt.Axes,
    best_model_min_step: int | None,
    xmax: float,
    *,
    labeled: bool,
) -> None:
    if best_model_min_step is None or best_model_min_step <= 0 or xmax <= best_model_min_step:
        return
    ax.axvspan(
        best_model_min_step,
        xmax,
        color="#2ca02c",
        alpha=0.07,
        zorder=0,
        label="best-model eligible" if labeled else "_best-model eligible",
    )


def _mark_saved_best_checkpoint(
    ax: plt.Axes,
    best_eval_step: int | None,
    y: float | None,
) -> None:
    if best_eval_step is None:
        return
    ax.axvline(
        best_eval_step,
        color="#ff7f0e",
        ls="--",
        lw=1.3,
        alpha=0.9,
        zorder=4,
        label="saved best",
    )
    if y is not None and np.isfinite(y):
        ax.plot(
            best_eval_step,
            y,
            marker="*",
            ms=14,
            color="#ff7f0e",
            markeredgecolor="#333333",
            markeredgewidth=0.6,
            zorder=5,
            linestyle="None",
        )


def _score_formula_from_run(run_dir: str | Path | None) -> str | None:
    """Compact eval-score formula from a run-local config snapshot."""
    if run_dir is None:
        return None
    cfg_path = Path(run_dir) / "config.yaml"
    if not cfg_path.is_file():
        return None
    try:
        from rlbot.rl_config import load_config

        cfg = load_config(cfg_path)
        tr = cfg.training
        blend = float(tr.best_model_score_stitched_blend)
        return (
            f"score: {1.0 - blend:g} segment excess + {blend:g} stitched excess "
            f"- {tr.best_model_score_std_coef:g} std - "
            f"{tr.best_model_score_dd_coef:g} p75 DD; bench={tr.best_model_benchmark}"
        )
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def plot_training_progress(
    episode_timesteps: Sequence[int],
    episode_rewards: Sequence[float],
    eval_timesteps: Optional[np.ndarray] = None,
    eval_ending_navs: Optional[np.ndarray] = None,
    eval_std_navs: Optional[np.ndarray] = None,
    eval_robust_scores: Optional[np.ndarray] = None,
    eval_mean_max_dd_pct: Optional[np.ndarray] = None,
    eval_mean_excess_nav: Optional[np.ndarray] = None,
    eval_stitched_excess_nav: Optional[np.ndarray] = None,
    eval_diag: Optional[dict[str, np.ndarray]] = None,
    eval_benchmark_label: str = "eval benchmark",
    eval_score_formula: str | None = None,
    episode_navs: Optional[Sequence[float]] = None,
    episode_nav_ts: Optional[Sequence[int]] = None,
    episode_lengths: Optional[Sequence[int]] = None,
    smooth_window: int = 15,
    title: str = "RL portfolio training",
    save_path: str | Path = "plots/training.png",
    best_model_min_step: int | None = None,
    best_eval_step: int | None = None,
) -> Path:
    """Episode rewards + eval NAV/drawdown + score delta + training episode-end NAV ($).

    Panels (top → bottom): per-step training reward; eval mean ending NAV (green) with
    ±1σ band and p75 max drawdown (%) on secondary axis; post-gate score delta (running
    max resets at fee ramp; pre-gate evals grayed); episode-end training NAV.

    When ``best_model_min_step`` is set (fee ramp end / best-save gate), eval panels and
    the score panel shade the eligible region; ``best_eval_step`` marks the saved checkpoint.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    has_nav = episode_navs is not None and len(episode_navs) > 0
    has_eval = (
        eval_timesteps is not None
        and eval_ending_navs is not None
        and len(eval_ending_navs) > 0
    )
    has_robust = (
        has_eval
        and eval_robust_scores is not None
        and len(eval_robust_scores) == len(eval_ending_navs)
    )
    has_diag = bool(eval_diag) and has_eval
    n_panels = (
        1
        + (1 if has_eval else 0)
        + (1 if has_robust else 0)
        + (1 if has_diag else 0)
        + (1 if has_nav else 0)
    )

    fig, axes = plt.subplots(
        n_panels, 1, figsize=(11, 3.2 * n_panels),
        sharex=False, constrained_layout=True,
    )
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13)

    panel = 0
    score_delta: np.ndarray | None = None
    pre_gate_mask: np.ndarray | None = None
    post_gate_mask: np.ndarray | None = None
    best_delta_y: float | None = None
    if has_robust and eval_robust_scores is not None and eval_timesteps is not None:
        score = np.asarray(eval_robust_scores, dtype=np.float64)
        ts_arr = np.asarray(eval_timesteps, dtype=np.int64)
        gate = int(best_model_min_step or 0)
        pre_gate_mask, score_delta, post_gate_mask = robust_score_delta_post_gate(
            score, ts_arr, gate
        )
        if best_eval_step is not None:
            hits = np.nonzero(ts_arr == int(best_eval_step))[0]
            if hits.size:
                val = float(score_delta[int(hits[0])])
                if np.isfinite(val):
                    best_delta_y = val

    ax_eval_nav: plt.Axes | None = None

    # ── Panel: Training per-step reward ──────────────────────────────
    ax0 = axes[panel]
    panel += 1
    if episode_timesteps and episode_rewards:
        ts = np.asarray(episode_timesteps, dtype=np.int64)
        rew = np.asarray(episode_rewards, dtype=np.float64)
        if episode_lengths is not None and len(episode_lengths) == len(rew):
            lens = np.asarray(episode_lengths, dtype=np.float64)
            lens = np.maximum(lens, 1.0)
            per_step = rew / lens
        else:
            per_step = rew
        ax0.scatter(ts, per_step, s=8, alpha=0.35, c="#1f77b4", label="per-step reward")
        if len(per_step) >= smooth_window:
            sm = _rolling_mean(per_step, smooth_window)
            ax0.plot(ts[smooth_window - 1 :], sm, color="#ff7f0e", lw=2, label=f"rolling-{smooth_window} mean")
        ax0.set_ylabel("reward / step")
        ax0.xaxis.set_major_formatter(_timestep_formatter())
        ax0.legend(loc="upper right", fontsize=8)
    else:
        ax0.text(0.5, 0.5, "No finished episodes yet", ha="center", va="center", transform=ax0.transAxes)
    ax0.set_title("Training episodes — per-step avg reward")
    ax0.grid(True, alpha=0.25)

    # ── Panel: Eval mean NAV + dispersion + drawdown ─────────────────
    if has_eval:
        ax1 = axes[panel]
        ax_eval_nav = ax1
        panel += 1
        ts = np.asarray(eval_timesteps, dtype=np.int64)
        nav = np.asarray(eval_ending_navs, dtype=np.float64)
        ax1.plot(
            ts,
            nav,
            marker="o",
            ms=3,
            color="#2ca02c",
            lw=2.0,
            label="mean ending NAV",
            zorder=3,
        )
        if eval_mean_excess_nav is not None and len(eval_mean_excess_nav) == len(nav):
            ax1.plot(
                ts,
                np.asarray(eval_mean_excess_nav, dtype=np.float64),
                color="#111111",
                ls="--",
                lw=1.2,
                marker=".",
                ms=3,
                alpha=0.85,
                label=f"mean excess vs {eval_benchmark_label}",
                zorder=3,
            )
        if eval_stitched_excess_nav is not None and len(eval_stitched_excess_nav) == len(nav):
            ax1.plot(
                ts,
                np.asarray(eval_stitched_excess_nav, dtype=np.float64),
                color="#6f4e7c",
                ls=":",
                lw=1.5,
                marker=".",
                ms=3,
                alpha=0.9,
                label="stitched excess",
                zorder=3,
            )
        if eval_std_navs is not None and len(eval_std_navs) == len(nav):
            std = np.asarray(eval_std_navs, dtype=np.float64)
            ax1.fill_between(
                ts,
                nav - std,
                nav + std,
                color="#2ca02c",
                alpha=0.18,
                label="mean NAV ± std",
                zorder=1,
            )
        ax1.axhline(100_000, color="gray", ls=":", lw=0.8, alpha=0.6, label="$100k start")
        ax1.set_ylabel("Portfolio value ($)")
        ax1.yaxis.set_major_formatter(_dollar_formatter())
        if eval_mean_max_dd_pct is not None and len(eval_mean_max_dd_pct) == len(nav):
            ax1_dd = ax1.twinx()
            dd_pct = np.asarray(eval_mean_max_dd_pct, dtype=np.float64)
            ax1_dd.plot(
                ts,
                dd_pct,
                color="#d62728",
                ls="-.",
                lw=1.4,
                marker="x",
                ms=3,
                alpha=0.85,
                label="p75 max drawdown (%)",
                zorder=2,
            )
            ax1_dd.set_ylabel("p75 max drawdown (%)", color="#d62728")
            ax1_dd.yaxis.set_major_formatter(_percent_formatter())
            ax1_dd.tick_params(axis="y", labelcolor="#d62728")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_dd.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7)
        else:
            ax1.legend(loc="upper left", fontsize=8)
        ax1.set_xlabel("timesteps")
        ax1.xaxis.set_major_formatter(_timestep_formatter())
        if eval_score_formula:
            ax1.text(
                0.99,
                0.02,
                eval_score_formula,
                transform=ax1.transAxes,
                ha="right",
                va="bottom",
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.85),
            )
        ax1.set_title("Periodic validation diagnostics — NAV, benchmark excess, drawdown")
        ax1.grid(True, alpha=0.25)

    # ── Panel: Post-gate score delta (running max resets at fee ramp) ─
    if has_robust and score_delta is not None and eval_timesteps is not None:
        ax_score = axes[panel]
        panel += 1
        ts = np.asarray(eval_timesteps, dtype=np.int64)
        if pre_gate_mask is not None and np.any(pre_gate_mask):
            ax_score.scatter(
                ts[pre_gate_mask],
                np.zeros(int(np.sum(pre_gate_mask)), dtype=np.float64),
                marker="|",
                s=120,
                color="#999999",
                alpha=0.55,
                linewidths=1.2,
                label="pre-gate eval (excluded)",
                zorder=2,
            )
        if post_gate_mask is not None and np.any(post_gate_mask):
            post_ts = ts[post_gate_mask]
            post_delta = score_delta[post_gate_mask]
            ax_score.plot(
                post_ts,
                post_delta,
                color="#1a4d1a",
                ls="-",
                lw=2.0,
                marker="o",
                ms=3,
                label="post-gate delta from best",
                zorder=3,
            )
        ax_score.axhline(0.0, color="gray", ls=":", lw=0.8, alpha=0.6)
        _mark_saved_best_checkpoint(ax_score, best_eval_step, best_delta_y)
        ax_score.set_ylabel("Score delta ($)")
        ax_score.yaxis.set_major_formatter(_k_formatter())
        ax_score.set_xlabel("timesteps")
        ax_score.xaxis.set_major_formatter(_timestep_formatter())
        gate_label = (
            f"gate {_timestep_formatter()(best_model_min_step, None)}"
            if best_model_min_step and best_model_min_step > 0
            else "no gate"
        )
        ax_score.set_title(
            "Periodic evaluation — post-gate score delta "
            f"(0 = new save-eligible best; {gate_label})"
        )
        ax_score.legend(loc="lower left", fontsize=8)
        ax_score.grid(True, alpha=0.25)

    # ── Panel: Eval portfolio diagnostics ───────────────────────────────
    if has_diag and eval_diag is not None and eval_timesteps is not None:
        ax_diag = axes[panel]
        panel += 1
        ts = np.asarray(eval_timesteps, dtype=np.int64)

        def _arr(key: str) -> np.ndarray | None:
            val = eval_diag.get(key)
            if val is None or len(val) != len(ts):
                return None
            return np.asarray(val, dtype=np.float64)

        pct_series = [
            ("mean_cash_frac", "cash", "#1f77b4"),
            ("mean_gross_exposure", "gross exposure", "#2ca02c"),
            ("mean_turnover", "turnover", "#ff7f0e"),
            ("cap_hit_fraction", "cap hits", "#d62728"),
        ]
        plotted = False
        for key, label, color in pct_series:
            arr = _arr(key)
            if arr is not None:
                ax_diag.plot(ts, 100.0 * arr, lw=1.5, marker=".", ms=3, color=color, label=label)
                plotted = True
        ax_diag.set_ylabel("%")
        ax_diag.yaxis.set_major_formatter(_percent_formatter())
        ax_diag.xaxis.set_major_formatter(_timestep_formatter())
        ax_diag.set_xlabel("timesteps")
        ax_diag.grid(True, alpha=0.25)

        eff = _arr("mean_effective_n_assets")
        if eff is not None:
            ax_eff = ax_diag.twinx()
            ax_eff.plot(ts, eff, lw=1.5, marker=".", ms=3, color="#9467bd", label="effective N")
            ax_eff.set_ylabel("effective N", color="#9467bd")
            ax_eff.tick_params(axis="y", labelcolor="#9467bd")
            lines1, labels1 = ax_diag.get_legend_handles_labels()
            lines2, labels2 = ax_eff.get_legend_handles_labels()
            ax_diag.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7, ncol=2)
        elif plotted:
            ax_diag.legend(loc="upper left", fontsize=7, ncol=2)
        ax_diag.set_title("Periodic validation diagnostics — allocation behavior")

    # ── Panel: Episode-end NAV in dollars ────────────────────────────
    if has_nav:
        ax2 = axes[panel]
        nav_arr = np.asarray(episode_navs, dtype=np.float64)
        nav_ts = np.asarray(episode_nav_ts if episode_nav_ts is not None else episode_timesteps[:len(nav_arr)], dtype=np.int64)
        ax2.scatter(nav_ts, nav_arr, s=8, alpha=0.35, c="#9467bd", label="episode-end NAV")
        if len(nav_arr) >= smooth_window:
            sm_nav = _rolling_mean(nav_arr, smooth_window)
            ax2.plot(nav_ts[smooth_window - 1 :], sm_nav, color="#d62728", lw=2, label=f"rolling-{smooth_window} mean")
        ax2.axhline(100_000, color="gray", ls="--", lw=0.8, alpha=0.6, label="$100k start")
        ax2.set_xlabel("timesteps")
        ax2.set_ylabel("portfolio value ($)")
        ax2.yaxis.set_major_formatter(_dollar_formatter())
        ax2.xaxis.set_major_formatter(_timestep_formatter())
        ax2.legend(loc="upper right", fontsize=8)
        ax2.set_title("Episode-end portfolio value")
        ax2.grid(True, alpha=0.25)

    # Align x across panels (eval can span full run while training dots were only in RAM for a tail segment).
    xmax = 0.0
    if episode_timesteps and len(episode_timesteps):
        xmax = max(xmax, float(np.max(np.asarray(episode_timesteps, dtype=np.float64))))
    if eval_timesteps is not None and len(eval_timesteps):
        xmax = max(xmax, float(np.max(np.asarray(eval_timesteps, dtype=np.float64))))
    if has_nav and episode_nav_ts is not None and len(episode_nav_ts):
        xmax = max(xmax, float(np.max(np.asarray(episode_nav_ts, dtype=np.float64))))
    if xmax > 0:
        pad = max(xmax * 0.01, 1.0)
        x_hi = xmax + pad
        for ax in axes:
            ax.set_xlim(0.0, x_hi)
        shaded = False
        for ax in axes:
            _shade_best_model_eligible(
                ax,
                best_model_min_step,
                x_hi,
                labeled=not shaded,
            )
            if best_model_min_step is not None and best_model_min_step > 0:
                shaded = True
        if ax_eval_nav is not None and best_eval_step is not None:
            best_nav_y: float | None = None
            if eval_timesteps is not None and eval_ending_navs is not None:
                ts_arr = np.asarray(eval_timesteps, dtype=np.int64)
                nav_arr = np.asarray(eval_ending_navs, dtype=np.float64)
                hits = np.nonzero(ts_arr == int(best_eval_step))[0]
                if hits.size:
                    best_nav_y = float(nav_arr[int(hits[0])])
            _mark_saved_best_checkpoint(ax_eval_nav, best_eval_step, best_nav_y)

    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    return save_path


def _normalized_equity(nav: np.ndarray) -> np.ndarray:
    nav = np.asarray(nav, dtype=np.float64)
    return nav / max(nav[0], 1e-12)


def _drawdown_fraction(eq: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(eq)
    return (eq - peak) / np.maximum(peak, 1e-12)


def _period_label(t: Sequence) -> str:
    t0, t1 = t[0], t[-1]
    if hasattr(t0, "date"):
        return f"{t0.date()} → {t1.date()}"
    return f"{t0} → {t1}"


def _plot_equity_drawdown_benchmarks(
    axes: Sequence,
    t: Sequence,
    *,
    nav_model: np.ndarray,
    nav_spy: np.ndarray | None,
    nav_equal_weight: np.ndarray | None,
    nav_balanced_6040: np.ndarray | None = None,
    nav_risk_parity: np.ndarray | None = None,
    nav_stochastic_ensemble: np.ndarray | None = None,
    model_label: str,
) -> None:
    """Fill top two axes: normalized equity and drawdown for model + passive benchmarks."""
    ax_eq, ax_dd = axes[0], axes[1]
    COLOR_MODEL = "#0066FF"
    COLOR_SPY = "#FF1744"
    COLOR_EW = "#00E676"
    COLOR_6040 = "#9C27B0"
    COLOR_RP = "#FF9100"

    bench_series: list[tuple[np.ndarray, str, str, str, float]] = []
    if nav_spy is not None:
        bench_series.append((nav_spy, "SPY buy & hold", COLOR_SPY, "-", 1.6))
    if nav_balanced_6040 is not None:
        bench_series.append((nav_balanced_6040, "60/40 SPY / IEF", COLOR_6040, "-", 1.5))
    if nav_risk_parity is not None:
        bench_series.append((nav_risk_parity, "Naive risk parity", COLOR_RP, "-", 1.5))
    if nav_equal_weight is not None:
        bench_series.append((nav_equal_weight, "Equal-weight daily rebalanced", COLOR_EW, "-", 1.4))

    eq_lines: list[tuple[np.ndarray, str, str, str, float]] = []
    z = 2
    for nav, label, color, ls, lw in bench_series:
        eq = _normalized_equity(nav)
        ret_pct = float(eq[-1] - 1.0) * 100.0
        leg = f"{label} ({ret_pct:+.1f}%)"
        ax_eq.plot(t, eq, color=color, lw=lw, ls=ls, label=leg, zorder=z)
        eq_lines.append((eq, leg, color, ls, lw))
        z += 1

    if nav_stochastic_ensemble is not None and nav_stochastic_ensemble.ndim == 2:
        ens = np.asarray(nav_stochastic_ensemble, dtype=np.float64)
        if ens.shape[1] == len(nav_model):
            eq_paths = ens / np.maximum(ens[:, :1], 1e-12)
            for path_eq in eq_paths:
                ax_eq.plot(
                    t,
                    path_eq,
                    color=COLOR_MODEL,
                    lw=0.45,
                    alpha=0.18,
                    zorder=z,
                )
            p5 = np.percentile(eq_paths, 5, axis=0)
            p50 = np.percentile(eq_paths, 50, axis=0)
            p95 = np.percentile(eq_paths, 95, axis=0)
            ax_eq.fill_between(
                t,
                p5,
                p95,
                color=COLOR_MODEL,
                alpha=0.22,
                label=f"Stochastic paths (n={ens.shape[0]}, 5–95%)",
                zorder=z,
            )
            ax_eq.plot(
                t,
                p50,
                color=COLOR_MODEL,
                lw=1.2,
                ls=":",
                alpha=0.85,
                zorder=z + 1,
            )
            z += 2

    eq_m = _normalized_equity(nav_model)
    ret_m_pct = float(eq_m[-1] - 1.0) * 100.0
    leg_m = f"{model_label} deterministic ({ret_m_pct:+.1f}%)"
    ax_eq.plot(
        t, eq_m, color=COLOR_MODEL, lw=2.6, ls="-", label=leg_m, zorder=z + 1,
    )
    eq_lines.append((eq_m, leg_m, COLOR_MODEL, "-", 2.6))

    ax_eq.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax_eq.set_ylabel("NAV / start")
    ax_eq.legend(loc="upper left", fontsize=9)
    ax_eq.grid(True, alpha=0.25)

    ret_m = float(_normalized_equity(nav_model)[-1] - 1.0)
    excess_spy = (
        (ret_m - float(_normalized_equity(nav_spy)[-1] - 1.0)) * 100.0
        if nav_spy is not None
        else float("nan")
    )
    period = _period_label(t)
    if nav_spy is not None and not np.isnan(excess_spy):
        ax_eq.set_title(f"Equity curve — {period}  |  excess vs SPY {excess_spy:+.1f} pp")
    else:
        ax_eq.set_title(f"Equity curve — {period}")

    for i, (eq, leg, color, ls, lw) in enumerate(eq_lines):
        dd = _drawdown_fraction(eq) * 100.0
        z = 2 + i
        ax_dd.fill_between(t, dd, 0.0, color=color, alpha=0.14, zorder=z)
        ax_dd.plot(t, dd, color=color, lw=lw, ls=ls, label=leg, zorder=z + 1)
    ax_dd.set_ylabel("drawdown %")
    ax_dd.legend(loc="lower left", fontsize=8)
    ax_dd.grid(True, alpha=0.25)
    ax_dd.set_title("Drawdown")


def plot_backtest_dashboard(
    timestamps: Sequence,
    nav: np.ndarray,
    *,
    nav_spy: np.ndarray | None = None,
    nav_equal_weight: np.ndarray | None = None,
    nav_balanced_6040: np.ndarray | None = None,
    nav_risk_parity: np.ndarray | None = None,
    nav_stochastic_ensemble: np.ndarray | None = None,
    weights: Optional[np.ndarray] = None,
    weight_timestamps: Optional[Sequence] = None,
    asset_labels: Optional[Sequence[str]] = None,
    model_label: str = "Model",
    title: str = "OOS backtest vs benchmarks",
    metrics: Optional[dict] = None,
    save_path: str | Path = "plots/backtest.png",
) -> Path:
    """
    Three-row OOS dashboard: model vs benchmarks (equity + drawdown), then target weights.

    Parameters
    ----------
    timestamps : length len(nav)
    nav : model portfolio NAV
    nav_spy, nav_equal_weight, nav_balanced_6040, nav_risk_parity : passive benchmarks (same len as ``nav``)
    weights : shape (n_steps, n_weights) — executed post-rebalance targets from
        ``info["target_weights"]`` (EMA-smoothed logits → softmax → cap), not raw policy logits.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    nav = np.asarray(nav, dtype=np.float64)
    t = np.asarray(timestamps)
    if len(t) != len(nav):
        raise ValueError("timestamps must match nav length")
    for name, bench in (
        ("nav_spy", nav_spy),
        ("nav_equal_weight", nav_equal_weight),
        ("nav_balanced_6040", nav_balanced_6040),
        ("nav_risk_parity", nav_risk_parity),
    ):
        if bench is not None and len(bench) != len(nav):
            raise ValueError(f"{name} must match nav length")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    fig.suptitle(title, fontsize=13)

    _plot_equity_drawdown_benchmarks(
        axes,
        t,
        nav_model=nav,
        nav_spy=nav_spy,
        nav_equal_weight=nav_equal_weight,
        nav_balanced_6040=nav_balanced_6040,
        nav_risk_parity=nav_risk_parity,
        nav_stochastic_ensemble=nav_stochastic_ensemble,
        model_label=model_label,
    )
    if metrics:
        def _pct(v):
            return "n/a" if v is None else f"{float(v) * 100:+.1f}%"

        def _num(v):
            return "n/a" if v is None else f"{float(v):.2f}"

        text = (
            f"Return {_pct(metrics.get('total_return'))}   "
            f"Sharpe {_num(metrics.get('sharpe'))}   "
            f"Max DD {_pct(metrics.get('max_drawdown'))}   "
            f"DSR {_num(metrics.get('deflated_sharpe'))} "
            f"(trials={metrics.get('oos_trials_for_window', 'n/a')})   "
            f"Excess vs EW {_pct(metrics.get('excess_equal_weight'))}"
        )
        axes[0].text(
            0.99,
            0.02,
            text,
            transform=axes[0].transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9),
        )

    if weights is not None and weights.size > 0:
        w = np.asarray(weights, dtype=np.float64)
        n = w.shape[0]
        if asset_labels is None:
            asset_labels = [f"w{i}" for i in range(w.shape[1])]
        if weight_timestamps is not None:
            tw = np.asarray(weight_timestamps)
        else:
            tw = t[:n] if len(t) >= n else t
        if len(tw) != n:
            tw = t[:n]

        plot_w = w.T
        plot_labels = list(asset_labels)
        if w.shape[1] > 15:
            # Cash (col 0) + top-8 risky sleeves; remainder → "Other Assets"
            cash = w[:, 0]
            risky = w[:, 1:]
            n_risky = risky.shape[1]
            top_k = min(8, n_risky)
            mean_risky = risky.mean(axis=0)
            top_idx = np.argsort(-mean_risky)[:top_k]
            stacks = [cash]
            labels = [plot_labels[0] if plot_labels else "Cash"]
            for j in top_idx:
                stacks.append(risky[:, j])
                labels.append(plot_labels[j + 1] if j + 1 < len(plot_labels) else f"Asset{j}")
            other_mask = np.ones(n_risky, dtype=bool)
            other_mask[top_idx] = False
            if np.any(other_mask):
                stacks.append(risky[:, other_mask].sum(axis=1))
                labels.append("Other Assets")
            plot_w = np.vstack(stacks)
            plot_labels = labels

        axes[2].stackplot(
            tw,
            plot_w,
            labels=plot_labels,
            alpha=0.9,
        )
        axes[2].set_ylim(0.0, 1.0)
        ncol = min(4, max(1, len(plot_labels)))
        axes[2].legend(loc="upper left", fontsize=7, ncol=ncol, framealpha=0.9)
        axes[2].set_title(
            "Executed target weights (EMA-smoothed logits → softmax → cap)"
        )
    else:
        axes[2].text(0.5, 0.5, "No weight history", ha="center", va="center", transform=axes[2].transAxes)

    axes[2].set_xlabel("time (UTC)")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=22)

    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    return save_path


def load_eval_nav_npz(path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load ``eval_nav_history.npz`` written by ``EvalNavBestModelCallback`` in train.py.

    Returns ``(timesteps, mean_ending_nav)`` as dollar portfolio values at episode end
    (averaged across eval episodes at each checkpoint). No per-step division.
    """
    hist = load_eval_history_npz(path)
    if hist is None:
        return None, None
    return hist["timesteps"], hist["mean_ending_nav"]


def _mean_max_dd_frac_from_diagnostics_jsonl(path: Path) -> np.ndarray | None:
    """One mean max-drawdown fraction per eval cycle from portfolio diagnostics JSONL."""
    path = Path(path)
    if not path.is_file():
        return None
    fracs: list[float] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            segs = rec.get("segments") or []
            if not segs:
                continue
            fracs.append(float(np.mean([float(s.get("max_drawdown_frac", 0.0)) for s in segs])))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    return np.asarray(fracs, dtype=np.float64) if fracs else None


def _portfolio_diagnostics_from_jsonl(path: Path) -> dict[str, np.ndarray] | None:
    """One portfolio-diagnostic scalar series per eval cycle from diagnostics JSONL."""
    path = Path(path)
    if not path.is_file():
        return None
    keys = (
        "mean_cash_frac",
        "mean_gross_exposure",
        "mean_effective_n_assets",
        "mean_turnover",
        "cap_hit_fraction",
    )
    rows: dict[str, list[float]] = {k: [] for k in keys}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            panel = rec.get("portfolio") or {}
            for k in keys:
                rows[k].append(float(panel.get(k, np.nan)))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if not any(rows.values()):
        return None
    return {k: np.asarray(v, dtype=np.float64) for k, v in rows.items()}


def load_eval_history_npz(path: Path) -> dict[str, np.ndarray] | None:
    """Load full eval history arrays for plotting (backward-compatible with older npz)."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        z = np.load(path, allow_pickle=False)
        ts = z.get("timesteps")
        nav = z.get("mean_ending_nav")
        if ts is None or nav is None:
            return None
        out: dict[str, np.ndarray] = {
            "timesteps": np.asarray(ts, dtype=np.int64).reshape(-1),
            "mean_ending_nav": np.asarray(nav, dtype=np.float64).reshape(-1),
        }
        for key in (
            "robust_scores",
            "std_ending_nav",
            "mean_max_drawdown_nav",
            "mean_max_drawdown_frac",
            "mean_excess_nav",
            "stitched_agent_nav",
            "stitched_excess_nav",
            "stitched_max_drawdown_frac",
        ):
            arr = z.get(key)
            if arr is not None:
                out[key] = np.asarray(arr, dtype=np.float64).reshape(-1)
        frac = out.get("mean_max_drawdown_frac")
        if frac is not None and len(frac) == len(out["timesteps"]):
            out["mean_max_drawdown_pct"] = 100.0 * frac
        else:
            jsonl = path.parent / "eval_portfolio_diagnostics.jsonl"
            from_jsonl = _mean_max_dd_frac_from_diagnostics_jsonl(jsonl)
            if from_jsonl is not None and len(from_jsonl) == len(out["timesteps"]):
                out["mean_max_drawdown_frac"] = from_jsonl
                out["mean_max_drawdown_pct"] = 100.0 * from_jsonl
            elif "mean_max_drawdown_nav" in out:
                dd_nav = out["mean_max_drawdown_nav"]
                out["mean_max_drawdown_pct"] = (
                    100.0 * dd_nav / np.maximum(out["mean_ending_nav"], 1e-12)
                )
        for scalar_key in ("best_eval_step", "best_model_min_step"):
            val = _scalar_from_npz(z, scalar_key)
            if val is not None:
                out[scalar_key] = val
        jsonl = path.parent / "eval_portfolio_diagnostics.jsonl"
        diag = _portfolio_diagnostics_from_jsonl(jsonl)
        if diag is not None:
            n = len(out["timesteps"])
            out["portfolio_diagnostics"] = {
                k: v for k, v in diag.items() if len(v) == n
            }
        return out
    except (OSError, ValueError, KeyError):
        return None


def regenerate_training_plot(run_dir: str | Path, *, title: str | None = None) -> Path | None:
    """Rebuild ``Runs/<id>/plots/training.png`` from persisted eval + episode npz."""
    run_dir = Path(run_dir)
    plot_path = run_dir / "plots" / "training.png"
    hist_path = run_dir / "eval_logs" / "eval_nav_history.npz"
    ep_path = plot_path.with_name("training_episodes.npz")
    hist = load_eval_history_npz(hist_path)
    if hist is None:
        return None
    ep_ts: list[int] = []
    ep_rew: list[float] = []
    ep_len: list[int] = []
    ep_nav: list[float] | None = None
    ep_nav_ts: list[int] | None = None
    if ep_path.is_file():
        try:
            z = np.load(ep_path, allow_pickle=False)
            ep_ts = list(np.asarray(z["episode_ts"], dtype=np.int64))
            ep_rew = list(np.asarray(z["episode_rewards"], dtype=np.float64))
            ep_len = list(np.asarray(z["episode_lengths"], dtype=np.int64))
            nav = z.get("episode_navs")
            nav_t = z.get("episode_nav_ts")
            if nav is not None and nav_t is not None and len(nav) > 0:
                ep_nav = list(np.asarray(nav, dtype=np.float64))
                ep_nav_ts = list(np.asarray(nav_t, dtype=np.int64))
        except (OSError, ValueError, KeyError):
            pass
    run_id = run_dir.name
    gate_step, best_step = resolve_eval_plot_milestones(hist, run_dir=run_dir)
    return plot_training_progress(
        ep_ts,
        ep_rew,
        eval_timesteps=hist["timesteps"],
        eval_ending_navs=hist["mean_ending_nav"],
        eval_std_navs=hist.get("std_ending_nav"),
        eval_robust_scores=hist.get("robust_scores"),
        eval_mean_max_dd_pct=hist.get("mean_max_drawdown_pct"),
        eval_mean_excess_nav=hist.get("mean_excess_nav"),
        eval_stitched_excess_nav=hist.get("stitched_excess_nav"),
        eval_diag=hist.get("portfolio_diagnostics"),
        eval_score_formula=_score_formula_from_run(run_dir),
        episode_navs=ep_nav,
        episode_nav_ts=ep_nav_ts,
        episode_lengths=ep_len if ep_len else None,
        title=title or f"RL portfolio training — {run_id}",
        save_path=plot_path,
        best_model_min_step=gate_step,
        best_eval_step=best_step,
    )


class TrainingVizCallback(BaseCallback):
    """
    Collect Monitor episode stats during PPO rollouts and refresh a PNG on a fixed step cadence.
    Middle panel: eval mean ending NAV (±std, p75 drawdown). When present, robust score
    gets its own panel below eval NAV.
    Also tracks episode-end portfolio NAV for the bottom training panel.

    Episode series are persisted next to the PNG (``training_episodes.npz``) so the training
    and NAV panels still show the **full** run after restarts or ``--resume``.
    """

    def __init__(
        self,
        plot_path: str | Path,
        eval_nav_npz_path: str | Path,
        plot_freq: int = 10_000,
        smooth_window: int = 15,
    ):
        super().__init__()
        self.plot_path = Path(plot_path)
        self.eval_nav_npz_path = Path(eval_nav_npz_path)
        self.plot_freq = int(plot_freq)
        self.smooth_window = int(smooth_window)
        self._history_path = self.plot_path.with_name(self.plot_path.stem + "_episodes.npz")
        self._episode_rewards: List[float] = []
        self._episode_lengths: List[int] = []
        self._episode_ts: List[int] = []
        self._episode_navs: List[float] = []
        self._episode_nav_ts: List[int] = []
        self._last_plot = 0
        self._load_episode_history()

    def _load_episode_history(self) -> None:
        """Restore episode scatter data from a previous session (same run folder)."""
        p = self._history_path
        if not p.is_file():
            return
        try:
            z = np.load(p, allow_pickle=False)
            self._episode_ts = list(np.asarray(z["episode_ts"], dtype=np.int64))
            self._episode_rewards = list(np.asarray(z["episode_rewards"], dtype=np.float64))
            self._episode_lengths = list(np.asarray(z["episode_lengths"], dtype=np.int64))
            nav = z.get("episode_navs")
            nav_t = z.get("episode_nav_ts")
            if nav is not None and nav_t is not None and len(nav) > 0:
                self._episode_navs = list(np.asarray(nav, dtype=np.float64))
                self._episode_nav_ts = list(np.asarray(nav_t, dtype=np.int64))
            n = min(len(self._episode_ts), len(self._episode_rewards), len(self._episode_lengths))
            self._episode_ts = self._episode_ts[:n]
            self._episode_rewards = self._episode_rewards[:n]
            self._episode_lengths = self._episode_lengths[:n]
        except (OSError, ValueError, KeyError):
            pass

    def _save_episode_history(self) -> None:
        """Persist episode data so plots survive process restarts and ``--resume``."""
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self._history_path,
            episode_ts=np.asarray(self._episode_ts, dtype=np.int64),
            episode_rewards=np.asarray(self._episode_rewards, dtype=np.float64),
            episode_lengths=np.asarray(self._episode_lengths, dtype=np.int64),
            episode_navs=np.asarray(self._episode_navs, dtype=np.float64),
            episode_nav_ts=np.asarray(self._episode_nav_ts, dtype=np.int64),
        )

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if infos:
            for info in infos:
                if not isinstance(info, dict):
                    continue
                ep = info.get("episode")
                if ep is not None:
                    self._episode_rewards.append(float(ep["r"]))
                    self._episode_lengths.append(int(ep.get("l", 1)))
                    self._episode_ts.append(int(self.num_timesteps))

                    nav = info.get("nav")
                    if nav is None:
                        ti = info.get("terminal_info")
                        if isinstance(ti, dict):
                            nav = ti.get("nav")
                    if nav is not None:
                        self._episode_navs.append(float(nav))
                        self._episode_nav_ts.append(int(self.num_timesteps))

        if self.num_timesteps - self._last_plot >= self.plot_freq:
            self._render()
            self._last_plot = int(self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        self._render()

    def _render(self) -> None:
        hist = load_eval_history_npz(self.eval_nav_npz_path)
        ev_t = ev_nav = ev_std = ev_score = ev_dd_pct = None
        if hist is not None:
            ev_t = hist["timesteps"]
            ev_nav = hist["mean_ending_nav"]
            ev_std = hist.get("std_ending_nav")
            ev_score = hist.get("robust_scores")
            ev_dd_pct = hist.get("mean_max_drawdown_pct")
            ev_excess = hist.get("mean_excess_nav")
            ev_stitched = hist.get("stitched_excess_nav")
            ev_diag = hist.get("portfolio_diagnostics")
        else:
            ev_excess = ev_stitched = ev_diag = None
        gate_step, best_step = (
            resolve_eval_plot_milestones(hist, run_dir=self.plot_path.parent.parent)
            if hist is not None
            else (None, None)
        )
        plot_training_progress(
            self._episode_ts,
            self._episode_rewards,
            eval_timesteps=ev_t,
            eval_ending_navs=ev_nav,
            eval_std_navs=ev_std,
            eval_robust_scores=ev_score,
            eval_mean_max_dd_pct=ev_dd_pct,
            eval_mean_excess_nav=ev_excess,
            eval_stitched_excess_nav=ev_stitched,
            eval_diag=ev_diag,
            eval_score_formula=_score_formula_from_run(self.plot_path.parent.parent),
            episode_navs=self._episode_navs if self._episode_navs else None,
            episode_nav_ts=self._episode_nav_ts if self._episode_nav_ts else None,
            episode_lengths=self._episode_lengths if self._episode_lengths else None,
            smooth_window=self.smooth_window,
            save_path=self.plot_path,
            best_model_min_step=gate_step,
            best_eval_step=best_step,
        )
        self._save_episode_history()
        mark_plot_saved(self.plot_path)


def open_plot_file(path: str | Path) -> None:
    """Open a PNG/PDF with the OS default viewer (macOS `open`, etc.)."""
    import subprocess
    import sys

    path = Path(path)
    if not path.is_file():
        return
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception:
        pass
