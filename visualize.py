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
    eval_means: Optional[np.ndarray] = None,
    episode_navs: Optional[Sequence[float]] = None,
    episode_nav_ts: Optional[Sequence[int]] = None,
    episode_lengths: Optional[Sequence[int]] = None,
    smooth_window: int = 15,
    title: str = "RL portfolio training",
    save_path: str | Path = "plots/training.png",
) -> Path:
    """Episode rewards + eval mean + episode-end NAV (portfolio value in $).

    Both reward panels show **per-step average reward** so that training
    (longer episodes) and eval (shorter episodes) are on comparable scales.
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

    # ── Panel 1: Eval per-step reward ────────────────────────────────
    ax1 = axes[1]
    if eval_timesteps is not None and eval_means is not None and len(eval_means) > 0:
        ax1.plot(eval_timesteps, eval_means, marker="o", ms=3, color="#2ca02c", label="eval per-step reward")
        ax1.set_ylabel("reward / step")
        ax1.legend(loc="upper right", fontsize=8)
    else:
        ax1.text(0.5, 0.5, "No eval checkpoints yet", ha="center", va="center", transform=ax1.transAxes)
    ax1.set_xlabel("timesteps")
    ax1.xaxis.set_major_formatter(_timestep_formatter())
    ax1.set_title("Periodic evaluation — per-step avg reward")
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


def plot_backtest_dashboard(
    timestamps: Sequence,
    nav: np.ndarray,
    weights: Optional[np.ndarray] = None,
    weight_timestamps: Optional[Sequence] = None,
    asset_labels: Optional[Sequence[str]] = None,
    title: str = "Backtest",
    save_path: str | Path = "plots/backtest.png",
) -> Path:
    """
    Equity (normalized), drawdown, and optional target-weight stack.

    Parameters
    ----------
    timestamps : length len(nav), timezone-aware or naive datetimes
    nav : portfolio values
    weights : shape (n_steps, n_weights)
    weight_timestamps : x-axis for weights; defaults to first len(weights) entries of `timestamps`
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    nav = np.asarray(nav, dtype=np.float64)
    t = np.asarray(timestamps)
    if len(t) != len(nav):
        raise ValueError("timestamps must match nav length")

    peak = np.maximum.accumulate(nav)
    dd = (nav - peak) / np.maximum(peak, 1e-12)
    eq_norm = nav / max(nav[0], 1e-12)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    fig.suptitle(title, fontsize=13)

    axes[0].plot(t, eq_norm, color="#1f77b4", lw=1.4, label="equity (normalized)")
    axes[0].axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.6)
    axes[0].set_ylabel("NAV / start")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(True, alpha=0.25)
    axes[0].set_title("Equity curve")

    axes[1].fill_between(t, dd * 100.0, 0.0, color="#d62728", alpha=0.35)
    axes[1].plot(t, dd * 100.0, color="#d62728", lw=1.0)
    axes[1].set_ylabel("drawdown %")
    axes[1].grid(True, alpha=0.25)
    axes[1].set_title("Drawdown")

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
        axes[2].stackplot(
            tw,
            w.T,
            labels=list(asset_labels),
            alpha=0.9,
        )
        axes[2].set_ylim(0.0, 1.0)
        axes[2].legend(loc="upper left", fontsize=7, ncol=4, framealpha=0.9)
        axes[2].set_title("Target portfolio weights (softmax of actions)")
    else:
        axes[2].text(0.5, 0.5, "No weight history", ha="center", va="center", transform=axes[2].transAxes)

    axes[2].set_xlabel("time (UTC)")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate(rotation=22)

    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    return save_path


def load_eval_npz(path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    SB3 EvalCallback saves ``eval_logs/evaluations.npz`` with ``timesteps``,
    ``results`` (episode rewards), and ``ep_lengths`` (episode lengths).

    Returns per-step average reward (total reward / episode length) so that
    the eval graph is directly comparable to the training per-step graph
    regardless of episode length differences.
    """
    path = Path(path)
    if not path.is_file():
        return None, None
    z = np.load(path, allow_pickle=True)
    ts = z.get("timesteps")
    raw = z.get("results")
    raw_lens = z.get("ep_lengths")
    if ts is None or raw is None:
        return None, None
    ts = np.asarray(ts, dtype=np.int64).reshape(-1)

    def _per_step_mean(rewards, lengths):
        """Average per-step reward across episodes at one eval point."""
        r = np.asarray(rewards, dtype=np.float64).ravel()
        if lengths is not None:
            l = np.maximum(np.asarray(lengths, dtype=np.float64).ravel(), 1.0)
            if len(l) == len(r):
                return float(np.mean(r / l))
        return float(np.mean(r))

    if raw.dtype == object:
        if raw_lens is not None and raw_lens.dtype == object:
            means = np.array([_per_step_mean(r, l) for r, l in zip(raw, raw_lens)], dtype=np.float64)
        else:
            means = np.array([_per_step_mean(r, None) for r in raw], dtype=np.float64)
    else:
        arr = np.asarray(raw, dtype=np.float64)
        if raw_lens is not None:
            larr = np.maximum(np.asarray(raw_lens, dtype=np.float64), 1.0)
            if arr.shape == larr.shape:
                means = (arr / larr).mean(axis=-1) if arr.ndim > 1 else arr / larr
            else:
                means = arr.mean(axis=-1) if arr.ndim > 1 else arr
        else:
            means = arr.mean(axis=-1) if arr.ndim > 1 else arr
    return ts, means


class TrainingVizCallback(BaseCallback):
    """
    Collect Monitor episode stats during PPO rollouts and refresh a PNG on a fixed step cadence.
    Optionally overlays EvalCallback curves if `evaluations.npz` exists.
    Also tracks episode-end portfolio NAV for the dollar-value panel.

    Episode series are persisted next to the PNG (``training_episodes.npz``) so the training
    and NAV panels still show the **full** run after restarts or ``--resume``. Eval metrics
    already come from ``evaluations.npz``; without this file, only in-RAM episodes since
    process start would be plotted (often a short tail on the x-axis).
    """

    def __init__(
        self,
        plot_path: str | Path,
        eval_npz_path: str | Path,
        plot_freq: int = 10_000,
        smooth_window: int = 15,
    ):
        super().__init__()
        self.plot_path = Path(plot_path)
        self.eval_npz_path = Path(eval_npz_path)
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
        ev_t, ev_r = load_eval_npz(self.eval_npz_path)
        plot_training_progress(
            self._episode_ts,
            self._episode_rewards,
            eval_timesteps=ev_t,
            eval_means=ev_r,
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
