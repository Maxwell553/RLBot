"""
Training and backtest charts (matplotlib, non-interactive PNG by default).
"""

from __future__ import annotations

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
from stable_baselines3.common.callbacks import BaseCallback


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


def plot_training_progress(
    episode_timesteps: Sequence[int],
    episode_rewards: Sequence[float],
    eval_timesteps: Optional[np.ndarray] = None,
    eval_ending_navs: Optional[np.ndarray] = None,
    episode_navs: Optional[Sequence[float]] = None,
    episode_nav_ts: Optional[Sequence[int]] = None,
    episode_lengths: Optional[Sequence[int]] = None,
    smooth_window: int = 15,
    title: str = "RL portfolio training",
    save_path: str | Path = "plots/training.png",
) -> Path:
    """Episode rewards + eval mean ending NAV + training episode-end NAV ($).

    Top panel: per-step training reward. Middle: mean ending portfolio NAV on the
    in-training eval split (from ``eval_nav_history.npz``). Bottom: training NAV.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    has_nav = episode_navs is not None and len(episode_navs) > 0
    n_panels = 3 if has_nav else 2

    fig, axes = plt.subplots(
        n_panels, 1, figsize=(11, 3.5 * n_panels),
        sharex=False, constrained_layout=True,
    )
    fig.suptitle(title, fontsize=13)

    # ── Panel 0: Training per-step reward ────────────────────────────
    ax0 = axes[0]
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

    # ── Panel 1: Eval mean ending NAV ($) ─────────────────────────────
    ax1 = axes[1]
    if eval_timesteps is not None and eval_ending_navs is not None and len(eval_ending_navs) > 0:
        ax1.plot(
            eval_timesteps,
            eval_ending_navs,
            marker="o",
            ms=3,
            color="#2ca02c",
            label="mean ending NAV",
        )
        ax1.axhline(100_000, color="gray", ls="--", lw=0.8, alpha=0.6, label="$100k start")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.yaxis.set_major_formatter(_dollar_formatter())
        ax1.legend(loc="upper right", fontsize=8)
    else:
        ax1.text(0.5, 0.5, "No eval checkpoints yet", ha="center", va="center", transform=ax1.transAxes)
    ax1.set_xlabel("timesteps")
    ax1.xaxis.set_major_formatter(_timestep_formatter())
    ax1.set_title("Periodic evaluation — Validation Ending NAV")
    ax1.grid(True, alpha=0.25)

    # ── Panel 2: Episode-end NAV in dollars ──────────────────────────
    if has_nav:
        ax2 = axes[2]
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
        for ax in axes:
            ax.set_xlim(0.0, xmax + pad)

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
        bench_series.append((nav_equal_weight, "Equal-weight buy & hold", COLOR_EW, "-", 1.4))

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
    save_path: str | Path = "plots/backtest.png",
) -> Path:
    """
    Three-row OOS dashboard: model vs benchmarks (equity + drawdown), then target weights.

    Parameters
    ----------
    timestamps : length len(nav)
    nav : model portfolio NAV
    nav_spy, nav_equal_weight, nav_balanced_6040, nav_risk_parity : passive benchmarks (same len as ``nav``)
    weights : shape (n_steps, n_weights)
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
        axes[2].set_title("Target portfolio weights (softmax of actions)")
    else:
        axes[2].text(0.5, 0.5, "No weight history", ha="center", va="center", transform=axes[2].transAxes)

    axes[2].set_xlabel("time (UTC)")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate(rotation=22)

    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    return save_path


def load_eval_nav_npz(path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load ``eval_nav_history.npz`` written by ``EvalNAVTrackerCallback`` in train.py.

    Returns ``(timesteps, mean_ending_nav)`` as dollar portfolio values at episode end
    (averaged across eval episodes at each checkpoint). No per-step division.
    """
    path = Path(path)
    if not path.is_file():
        return None, None
    try:
        z = np.load(path, allow_pickle=False)
        ts = z.get("timesteps")
        nav = z.get("mean_ending_nav")
        if ts is None or nav is None:
            return None, None
        return np.asarray(ts, dtype=np.int64).reshape(-1), np.asarray(nav, dtype=np.float64).reshape(-1)
    except (OSError, ValueError, KeyError):
        return None, None


class TrainingVizCallback(BaseCallback):
    """
    Collect Monitor episode stats during PPO rollouts and refresh a PNG on a fixed step cadence.
    Middle panel reads ``eval_nav_history.npz`` (mean ending NAV on the eval split).
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
        ev_t, ev_nav = load_eval_nav_npz(self.eval_nav_npz_path)
        plot_training_progress(
            self._episode_ts,
            self._episode_rewards,
            eval_timesteps=ev_t,
            eval_ending_navs=ev_nav,
            episode_navs=self._episode_navs if self._episode_navs else None,
            episode_nav_ts=self._episode_nav_ts if self._episode_nav_ts else None,
            episode_lengths=self._episode_lengths if self._episode_lengths else None,
            smooth_window=self.smooth_window,
            save_path=self.plot_path,
        )
        self._save_episode_history()


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
