#!/usr/bin/env python3
"""
Evaluate a trained RecurrentPPO (LSTM) policy on data reserved for OOS backtest only.

By default uses the same **chronological holdout** as training: the last
``holdout_days`` calendar days of the cache, which **must not** appear in
``train.py`` (see ``reserve_chronological_holdout``). This matches
``runs/<id>/manifest.json`` when ``--run-id`` is set.

Use ``--legacy-last-days`` to reproduce the old behavior (last N days from the
full cache, overlapping training data).

10 global assets: S&P 500 (SPY), Gold (GLD), Crude Oil WTI (USO),
EUR/USD, USD/JPY, Nikkei 225, FTSE 100, 10-Year Treasury (IEF),
Copper (HG=F), Emerging Markets (EEM).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from data_utils import (
    TICKERS,
    benchmark_ohlcv_index,
    clip_index_until,
    load_cache,
    reserve_chronological_holdout,
    train_test_split_last_days,
)
from run_artifacts import PROJECT_ROOT, RunPaths, read_latest_run_id, read_run_manifest
from trading_env import MultiAssetPortfolioEnv, portfolio_weights_from_action
from vecnorm_utils import freeze_vec_normalize_for_inference
from visualize import open_plot_file, plot_backtest_dashboard

ROOT = PROJECT_ROOT
DATA_CACHE = ROOT / "data_cache.npz"
LEGACY_MODEL_DIR = ROOT / "models"
PLOTS_DIR = ROOT / "plots"


def _infer_run_id_from_model_path(model_path: Path) -> str | None:
    parts = model_path.resolve().parts
    for i, name in enumerate(parts):
        if name == "models" and i + 1 < len(parts):
            nxt = parts[i + 1]
            if nxt not in ("best",):
                return nxt
    return None


def _resolve_model_path(args: argparse.Namespace) -> tuple[Path, str | None]:
    """Return (model_zip, run_id hint for naming plots, or None)."""
    if args.model:
        p = Path(args.model)
        if not p.is_file():
            raise FileNotFoundError(f"Model not found: {p}")
        return p, _infer_run_id_from_model_path(p)

    rid = args.run_id.strip()
    if rid:
        rp = RunPaths(rid)
        for cand in (rp.best_model_dir / "best_model.zip", rp.final_model):
            if cand.is_file():
                return cand, rid
        raise FileNotFoundError(
            f"No best_model.zip or ppo_portfolio_final.zip under models/{rid}/"
        )

    latest = read_latest_run_id()
    if latest:
        rp = RunPaths(latest)
        for cand in (rp.best_model_dir / "best_model.zip", rp.final_model):
            if cand.is_file():
                return cand, latest

    for cand in (
        LEGACY_MODEL_DIR / "best" / "best_model.zip",
        LEGACY_MODEL_DIR / "ppo_portfolio_final.zip",
    ):
        if cand.is_file():
            return cand, _infer_run_id_from_model_path(cand)

    raise FileNotFoundError(
        "No model found. Train first (writes runs/LATEST.txt), pass --run-id, or --model path/to.zip"
    )


def _latest_checkpoint_vecnormalize(ckpt_dir: Path) -> Path | None:
    """Pick the checkpoint VecNormalize file with the largest timestep suffix."""
    best_p: Path | None = None
    best_step = -1
    for p in ckpt_dir.glob("ppo_vecnormalize_*_steps.pkl"):
        m = re.search(r"vecnormalize_(\d+)_steps", p.name)
        if m:
            s = int(m.group(1))
            if s > best_step:
                best_step = s
                best_p = p
    return best_p


def _find_vec_normalize(
    model_path: Path,
    run_hint: str | None,
    explicit: Path | None = None,
) -> Path:
    """Locate VecNormalize stats (.pkl). Returned path may not exist."""
    if explicit is not None:
        e = Path(explicit).expanduser().resolve()
        if e.is_file():
            return e
        raise FileNotFoundError(f"--vec-normalize not found: {e}")

    # Same directory as the .zip (e.g. checkpoint folder)
    p = model_path.parent / "vec_normalize.pkl"
    if p.is_file():
        return p

    stem = model_path.stem
    parts = stem.split("_", 1)
    if len(parts) == 2:
        ckpt_vn = model_path.parent / f"{parts[0]}_vecnormalize_{parts[1]}.pkl"
        if ckpt_vn.is_file():
            return ckpt_vn

    if run_hint:
        md = RunPaths(run_hint).models_dir
        for candidate in (
            md / "vec_normalize.pkl",
            md / "best" / "vec_normalize.pkl",
        ):
            if candidate.is_file():
                return candidate
        # models/<id>/best/best_model.zip → run-level vec_normalize
        if model_path.parent.name == "best":
            parent_vn = model_path.parent.parent / "vec_normalize.pkl"
            if parent_vn.is_file():
                return parent_vn
        ckpt_dir = md / "checkpoints"
        fallback = _latest_checkpoint_vecnormalize(ckpt_dir)
        if fallback is not None:
            return fallback

    return model_path.parent / "vec_normalize.pkl"


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.maximum(peak, 1e-12)
    return float(dd.min())


def _sharpe_ann_from_log_rets(log_rets: np.ndarray) -> float:
    if log_rets.size < 2:
        return float("nan")
    return float(np.mean(log_rets) / (np.std(log_rets) + 1e-12) * np.sqrt(252))


def rollout_policy_on_slice(
    model: RecurrentPPO,
    *,
    test_idx: pd.DatetimeIndex,
    test_ohlcv: np.ndarray,
    test_rsi: np.ndarray,
    test_macd: np.ndarray,
    test_macro: np.ndarray,
    test_fd: np.ndarray,
    test_fdm: np.ndarray,
    test_trend: np.ndarray,
    obs_lag: int,
    vec_norm_path: Path,
    use_vec_norm: bool,
    deterministic: bool = True,
    collect_weights: bool = False,
) -> tuple[np.ndarray, int, int, np.ndarray | None]:
    """
    One full episode on a contiguous date slice. Returns
    (navs, start_bar, n_rewards, weights|None) where n_rewards == len(navs) - 1.
    Causal: no look-ahead beyond training-time observation pipeline.
    """
    n_bars = len(test_idx)
    if n_bars < 10:
        raise ValueError("Slice too short for a rollout")
    raw_env = MultiAssetPortfolioEnv(
        test_ohlcv,
        test_rsi,
        test_macd,
        fracdiff=test_fd,
        fracdiff_macro=test_fdm,
        trend=test_trend,
        macro=test_macro,
        random_start=False,
        max_episode_steps=n_bars,
        obs_lag=0,
        obs_lag_default=obs_lag,
        fee_scale_default=1.0,
        domain_randomize=False,
    )
    vec_env = None
    if use_vec_norm:
        if not vec_norm_path.is_file():
            raise FileNotFoundError(f"VecNormalize not found: {vec_norm_path}")
        venv = DummyVecEnv([lambda: raw_env])
        try:
            vec_env = VecNormalize.load(str(vec_norm_path), venv)
        except AssertionError as e:
            if "spaces must have the same shape" in str(e):
                cur = int(venv.observation_space.shape[0])
                raise RuntimeError(
                    f"VecNormalize obs dim mismatch for {vec_norm_path.name}: "
                    f"stats were saved for a different feature layout than the current env "
                    f"({cur}-dim, 10-asset universe). "
                    f"Use a model trained on the current codebase (e.g. --run-id 65M_4_20_26_a) "
                    f"or retrain; older 8-asset runs (83-dim) are incompatible."
                ) from e
            raise
        freeze_vec_normalize_for_inference(vec_env)

    obs, _ = raw_env.reset(seed=0)
    if use_vec_norm and vec_env is not None:
        obs = vec_env.normalize_obs(obs)
    start_bar = int(raw_env._t)
    navs: list[float] = [raw_env._nav(test_ohlcv[raw_env._t, :, 3])]
    w_rows: list[np.ndarray] = []
    done = False
    truncated = False
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    while not (done or truncated):
        obs_model = obs.reshape(1, -1) if getattr(obs, "ndim", 1) == 1 else obs
        action, lstm_states = model.predict(
            obs_model,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        episode_starts = np.zeros((1,), dtype=bool)
        if collect_weights:
            w_rows.append(portfolio_weights_from_action(np.asarray(action).reshape(-1)))
        obs, _, done, truncated, info = raw_env.step(action)
        if use_vec_norm and vec_env is not None:
            obs = vec_env.normalize_obs(obs)
        if "nav" in info:
            navs.append(info["nav"])

    out = np.asarray(navs, dtype=np.float64)
    n_rew = len(navs) - 1
    w_arr: np.ndarray | None
    if collect_weights and w_rows:
        w_arr = np.stack(w_rows, axis=0)
    elif collect_weights:
        w_arr = np.zeros((0, 1), dtype=np.float64)
    else:
        w_arr = None
    return out, start_bar, n_rew, w_arr


def _spy_buyhold_nav(
    navs: np.ndarray,
    ohlcv_window: np.ndarray,
    start_bar: int,
    spy_ohlcv_col: int | None = None,
) -> np.ndarray:
    """SPY buy-and-hold NAV aligned to model ``navs`` (same start_bar and length)."""
    spy_col = benchmark_ohlcv_index() if spy_ohlcv_col is None else spy_ohlcv_col
    close_spy = ohlcv_window[:, spy_col, 3].astype(np.float64, copy=False)
    i0, i1 = int(start_bar), int(start_bar + len(navs) - 1)
    s0 = max(close_spy[i0], 1e-12)
    return (close_spy[i0 : i1 + 1] / s0) * float(navs[0])


def _equal_weight_buyhold_nav(
    navs: np.ndarray,
    ohlcv_window: np.ndarray,
    start_bar: int,
) -> np.ndarray:
    """Equal-weight (1/N) daily-rebalanced buy-and-hold on all tradeable assets, no cash.

    Each step applies the cross-sectional mean log return across the N assets — the
    passive benchmark for a fixed 10% sleeve in each risky leg.
    """
    close = ohlcv_window[:, :, 3].astype(np.float64, copy=False)
    i0 = int(start_bar)
    n = len(navs)
    out = np.empty(n, dtype=np.float64)
    out[0] = float(navs[0])
    for k in range(1, n):
        t_prev = i0 + k - 1
        t_curr = i0 + k
        log_rets = np.log((close[t_curr] + 1e-12) / (close[t_prev] + 1e-12))
        out[k] = out[k - 1] * float(np.exp(np.mean(log_rets)))
    return out


def _print_detailed_stats(
    *,
    test_idx: pd.DatetimeIndex,
    navs: np.ndarray,
    log_rets: np.ndarray,
    ohlcv_window: np.ndarray,
    start_bar: int,
    spy_ohlcv_col: int | None = None,
) -> None:
    """Ohlcv_window is the full OOS test slice; prices align with test_idx rows."""
    n = len(test_idx)
    t0, t1 = test_idx[0], test_idx[-1]
    cal_days = (t1 - t0).days if hasattr(t1 - t0, "days") else int(
        (np.datetime64(t1) - np.datetime64(t0)) / np.timedelta64(1, "D")
    )
    print("--- detailed ---")
    print(
        f"OOS window: {t0} .. {t1}  ({n} daily bars, ~{cal_days} calendar days)"
    )
    cagr = float((navs[-1] / max(navs[0], 1e-12)) ** (252.0 / max(len(log_rets), 1)) - 1.0)
    print(f"Compound annualized growth (from daily bars): {cagr * 100:.2f}%")
    mdd = _max_drawdown(navs)
    calmar = float(cagr / max(abs(mdd), 1e-12)) if mdd < 0 else float("nan")
    print(f"Calmar (CAGR / |max DD|): {calmar:.2f}")

    nav_spy = _spy_buyhold_nav(navs, ohlcv_window, start_bar, spy_ohlcv_col)
    lr_spy = np.diff(np.log(np.maximum(nav_spy, 1e-12)))
    sh_spy = _sharpe_ann_from_log_rets(lr_spy)
    print(
        f"SPY 100% buy&hold (same OOS path, {len(lr_spy)} daily rets): "
        f"total {(nav_spy[-1] / nav_spy[0] - 1) * 100:.2f}%, ann. Sharpe {sh_spy:.2f}"
    )
    nav_ew = _equal_weight_buyhold_nav(navs, ohlcv_window, start_bar)
    lr_ew = np.diff(np.log(np.maximum(nav_ew, 1e-12)))
    sh_ew = _sharpe_ann_from_log_rets(lr_ew)
    print(
        f"Equal-weight 10-asset buy&hold (10% each, daily rebal., {len(lr_ew)} daily rets): "
        f"total {(nav_ew[-1] / nav_ew[0] - 1) * 100:.2f}%, ann. Sharpe {sh_ew:.2f}"
    )
    nlr = log_rets.size
    for label, n_parts in (("1st/2nd half of OOS", 2), ("quarters of OOS", 4)):
        if nlr < n_parts * 2:
            continue
        w = nlr // n_parts
        parts: list[str] = []
        for p in range(n_parts):
            sl = log_rets[p * w : (p + 1) * w if p < n_parts - 1 else nlr]
            if sl.size < 2:
                continue
            tr = float(np.expm1(np.sum(sl)) * 100.0)
            sh = _sharpe_ann_from_log_rets(sl)
            parts.append(f"part{p + 1}: {tr:+.1f}% ret, Sh={sh:.2f}")
        if parts:
            print(f"{label}: {', '.join(parts)}")

    if nlr >= 10:
        rng = np.random.default_rng(0)
        b_n = 8000
        sh_boot = np.empty(b_n, dtype=np.float64)
        for b in range(b_n):
            samp = rng.choice(log_rets, size=nlr, replace=True)
            sh_boot[b] = _sharpe_ann_from_log_rets(samp)
        lo, med, hi = (
            float(np.percentile(sh_boot, 2.5)),
            float(np.percentile(sh_boot, 50.0)),
            float(np.percentile(sh_boot, 97.5)),
        )
        print(
            f"Bootstrap Sharpe ({b_n} resamples, i.i.d. days): "
            f"2.5%={lo:.2f}, 50%={med:.2f}, 97.5%={hi:.2f}"
        )
    skew = float(np.mean(((log_rets - np.mean(log_rets)) / (np.std(log_rets) + 1e-12)) ** 3))
    print(
        f"Skew of daily log returns: {skew:.2f}  (strong positive skew can raise sample Sharpe in short windows)"
    )
    print("--- end detailed ---")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Path to a .zip model; if omitted, uses runs/LATEST.txt → models/<id>/…",
    )
    parser.add_argument(
        "--run-id",
        default="",
        metavar="ID",
        help="Use models/<ID>/best_model.zip (or ppo_portfolio_final.zip); plot saved under plots/<ID>/",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=60,
        help="With --legacy-last-days: calendar days of trailing data (may overlap training).",
    )
    parser.add_argument(
        "--holdout-days",
        type=int,
        default=None,
        help=(
            "Reserve the last N calendar days as OOS (must match training). "
            "Ignored when date holdout is set. "
            "Default: read from run manifest when --run-id is set, else 365."
        ),
    )
    parser.add_argument("--until", default=None, help="Clip cache to this date (YYYY-MM-DD); should match training")
    parser.add_argument(
        "--train-end",
        default=None,
        metavar="YYYY-MM-DD",
        help="Last trainable day (must match training). Default: manifest.",
    )
    parser.add_argument(
        "--holdout-start",
        default=None,
        metavar="YYYY-MM-DD",
        help="First OOS day (must match training). Default: manifest.",
    )
    parser.add_argument(
        "--holdout-end",
        default=None,
        metavar="YYYY-MM-DD",
        help="Last OOS day (must match training). Default: manifest or last bar.",
    )
    parser.add_argument(
        "--legacy-last-days",
        action="store_true",
        help="Use last --test-days from the full cache (old behavior; can overlap training blocks).",
    )
    parser.add_argument("--obs-lag", type=int, default=1, help="Market features lag (must match training)")
    parser.add_argument(
        "--vec-normalize",
        type=str,
        default="",
        metavar="PATH",
        help="VecNormalize .pkl (default: auto from run-id / checkpoints next to model)",
    )
    parser.add_argument(
        "--require-vec-normalize",
        action="store_true",
        help="Exit with error if no VecNormalize stats file is found (strict trade validation)",
    )
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--no-viz", action="store_true", help="Skip saving backtest plot PNG")
    parser.add_argument(
        "--show-viz",
        action="store_true",
        help="Open the backtest plot with the default viewer after the run",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print subperiod stats, SPY benchmark, bootstrap Sharpe band, OOS bar/calendar count.",
    )
    parser.add_argument(
        "--plot-tag",
        default="",
        metavar="TAG",
        help="If set, save plots/backtest_TAG.png (e.g. latest, best).",
    )
    args = parser.parse_args()

    idx, ohlcv, rsi, macd, macro, fracdiff, fracdiff_macro, trend = load_cache(str(DATA_CACHE))

    model_path, run_hint = _resolve_model_path(args)
    plot_run_id = args.run_id.strip() or run_hint or read_latest_run_id() or "misc"
    manifest = read_run_manifest(plot_run_id) if plot_run_id != "misc" else None

    until = args.until
    if until is None and manifest:
        until = manifest.get("args", {}).get("until")
    if until:
        idx, (ohlcv, rsi, macd, macro, fracdiff, fracdiff_macro, trend) = clip_index_until(
            idx,
            ohlcv,
            rsi,
            macd,
            macro,
            fracdiff,
            fracdiff_macro,
            trend,
            until=until,
        )

    if args.legacy_last_days:
        _, (test_idx, test_ohlcv, test_rsi, test_macd, test_macro, test_fd, test_fdm, test_trend) = (
            train_test_split_last_days(
                idx,
                ohlcv,
                rsi,
                macd,
                macro,
                fracdiff,
                fracdiff_macro,
                trend,
                test_days=args.test_days,
            )
        )
        print(
            "WARNING: --legacy-last-days uses trailing data from the full cache; "
            "bars may overlap with training (not strict OOS)."
        )
    else:
        holdout_days = args.holdout_days
        train_end = args.train_end
        holdout_start = args.holdout_start
        holdout_end = args.holdout_end
        ch = (manifest or {}).get("chronological_holdout") or {}
        if train_end is None:
            train_end = ch.get("train_end") or (manifest or {}).get("args", {}).get("train_end")
        if holdout_start is None:
            holdout_start = ch.get("holdout_start") or (manifest or {}).get("args", {}).get("holdout_start")
        if holdout_end is None:
            holdout_end = ch.get("holdout_end") or (manifest or {}).get("args", {}).get("holdout_end")

        if holdout_days is None:
            if manifest and ch.get("holdout_days") is not None:
                holdout_days = int(ch["holdout_days"])
            elif manifest and manifest.get("args", {}).get("holdout_days") is not None:
                holdout_days = int(manifest["args"]["holdout_days"])
            else:
                holdout_days = 365

        if train_end and holdout_start:
            print(
                f"Using date holdout from manifest/CLI: train_end={train_end}, "
                f"holdout_start={holdout_start}, holdout_end={holdout_end or '(last bar)'}"
            )
        elif holdout_days is not None:
            print(f"Using holdout_days={holdout_days} (calendar tail)")

        _, (test_idx, test_ohlcv, test_rsi, test_macd, test_macro, test_fd, test_fdm, test_trend) = (
            reserve_chronological_holdout(
                idx,
                ohlcv,
                rsi,
                macd,
                macro,
                fracdiff,
                fracdiff_macro,
                trend,
                holdout_days=holdout_days,
                train_end=train_end,
                holdout_start=holdout_start,
                holdout_end=holdout_end,
            )
        )
        if train_end and holdout_start:
            print(
                f"Strict OOS backtest: {holdout_start} .. {test_idx[-1].date()} "
                f"({test_idx[0]} .. {test_idx[-1]}, {len(test_idx)} bars) — excluded from training."
            )
        else:
            print(
                f"Strict OOS backtest: last {holdout_days} calendar days "
                f"({test_idx[0]} .. {test_idx[-1]}, {len(test_idx)} bars) — excluded from training."
            )

    if len(test_idx) < 10:
        raise RuntimeError(
            "Test window too short; fetch more history or reduce holdout days / --test-days."
        )

    print(f"Model: {model_path}")
    print(f"Backtest plot folder (run tag): plots/{plot_run_id}/")
    model = RecurrentPPO.load(str(model_path), device="auto")

    explicit_vn = Path(args.vec_normalize).expanduser().resolve() if args.vec_normalize.strip() else None
    vec_norm_path = _find_vec_normalize(model_path, run_hint, explicit=explicit_vn)

    use_vec_norm = vec_norm_path.is_file()
    if not use_vec_norm:
        msg = (
            f"No VecNormalize stats at {vec_norm_path}. "
            "Inference will use raw observations (misaligned with training). "
            "Train with current train.py (saves vec_normalize.pkl) or pass --vec-normalize PATH."
        )
        if args.require_vec_normalize:
            raise FileNotFoundError(msg)
        print(f"WARNING: {msg}")
    else:
        if "checkpoints" in vec_norm_path.parts:
            print(f"Loading VecNormalize stats (checkpoint fallback): {vec_norm_path}")
        else:
            print(f"Loading VecNormalize stats: {vec_norm_path}")

    navs, start_bar, n_rew, w_opt = rollout_policy_on_slice(
        model,
        test_idx=test_idx,
        test_ohlcv=test_ohlcv,
        test_rsi=test_rsi,
        test_macd=test_macd,
        test_macro=test_macro,
        test_fd=test_fd,
        test_fdm=test_fdm,
        test_trend=test_trend,
        obs_lag=args.obs_lag,
        vec_norm_path=vec_norm_path,
        use_vec_norm=use_vec_norm,
        deterministic=args.deterministic,
        collect_weights=not args.no_viz,
    )
    weights = w_opt if w_opt is not None else np.zeros((0, 1))
    nav_ix = start_bar + np.arange(len(navs), dtype=np.int64)
    nav_ix = np.clip(nav_ix, 0, len(test_idx) - 1)
    time_nav = test_idx[nav_ix]

    if not args.no_viz:
        plot_dir = PLOTS_DIR / plot_run_id
        plot_dir.mkdir(parents=True, exist_ok=True)
        tag = args.plot_tag.strip()
        dash_name = f"backtest_{tag}.png" if tag else "backtest.png"
        out = plot_dir / dash_name
        time_w = None
        if weights.size > 0 and weights.shape[0] > 0:
            w_ix = start_bar + np.arange(weights.shape[0], dtype=np.int64)
            w_ix = np.clip(w_ix, 0, len(test_idx) - 1)
            time_w = test_idx[w_ix]
        nav_spy = _spy_buyhold_nav(navs, test_ohlcv, start_bar)
        nav_ew = _equal_weight_buyhold_nav(navs, test_ohlcv, start_bar)
        model_label = f"Model ({tag})" if tag else f"Model ({model_path.stem})"
        plot_backtest_dashboard(
            time_nav,
            navs,
            nav_spy=nav_spy,
            nav_equal_weight=nav_ew,
            weights=weights,
            weight_timestamps=time_w,
            asset_labels=["Cash"] + list(TICKERS),
            model_label=model_label,
            title="OOS backtest vs benchmarks",
            save_path=out,
        )
        print(f"Backtest plot: {out}")
        if args.show_viz:
            open_plot_file(out)

    total_return = float(navs[-1] / navs[0] - 1.0)
    log_rets = np.diff(np.log(np.maximum(navs, 1e-12)))
    # Annualized Sharpe from daily log-returns (252 trading days/year)
    sharpe = float(np.mean(log_rets) / (np.std(log_rets) + 1e-12) * np.sqrt(252))

    print(f"Test window: {test_idx[0]} .. {test_idx[-1]}  ({len(test_idx)} bars)")
    print(f"Bars stepped: {n_rew}")
    print(f"Total return: {total_return * 100:.2f}%")
    print(f"Approx. annualized Sharpe (log-ret, daily): {sharpe:.2f}")
    print(f"Max drawdown: {_max_drawdown(navs) * 100:.2f}%")
    if args.detailed:
        _print_detailed_stats(
            test_idx=test_idx,
            navs=navs,
            log_rets=log_rets,
            ohlcv_window=test_ohlcv,
            start_bar=start_bar,
        )


if __name__ == "__main__":
    main()
