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
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from data_utils import (
    TICKERS,
    load_cache,
    reserve_chronological_holdout,
    train_test_split_last_days,
)
from run_artifacts import PROJECT_ROOT, RunPaths, read_latest_run_id, read_run_manifest
from trading_env import MultiAssetPortfolioEnv, portfolio_weights_from_action
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
            "Default: read from run manifest when --run-id is set, else 365."
        ),
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
    args = parser.parse_args()

    idx, ohlcv, rsi, macd, macro, fracdiff, fracdiff_macro = load_cache(str(DATA_CACHE))

    model_path, run_hint = _resolve_model_path(args)
    plot_run_id = args.run_id.strip() or run_hint or read_latest_run_id() or "misc"

    if args.legacy_last_days:
        _, (test_idx, test_ohlcv, test_rsi, test_macd, test_macro, test_fd, test_fdm) = (
            train_test_split_last_days(
                idx, ohlcv, rsi, macd, macro, fracdiff, fracdiff_macro,
                test_days=args.test_days,
            )
        )
        print(
            "WARNING: --legacy-last-days uses trailing data from the full cache; "
            "bars may overlap with training (not strict OOS)."
        )
    else:
        holdout_days = args.holdout_days
        if holdout_days is None:
            manifest = read_run_manifest(plot_run_id) if plot_run_id != "misc" else None
            if manifest and manifest.get("chronological_holdout"):
                holdout_days = int(manifest["chronological_holdout"]["holdout_days"])
                print(f"Using holdout_days={holdout_days} from runs/{plot_run_id}/manifest.json")
            else:
                holdout_days = 365
                print(
                    f"WARNING: No chronological_holdout in manifest (older run?); "
                    f"using holdout_days={holdout_days}. "
                    "Retrain with current train.py for a strict OOS holdout, or pass --holdout-days."
                )
        _, (test_idx, test_ohlcv, test_rsi, test_macd, test_macro, test_fd, test_fdm) = (
            reserve_chronological_holdout(
                idx, ohlcv, rsi, macd, macro, fracdiff, fracdiff_macro,
                holdout_days=holdout_days,
            )
        )
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

    raw_env = MultiAssetPortfolioEnv(
        test_ohlcv,
        test_rsi,
        test_macd,
        fracdiff=test_fd,
        fracdiff_macro=test_fdm,
        macro=test_macro,
        random_start=False,
        max_episode_steps=len(test_idx),
        obs_lag=0,
        obs_lag_default=args.obs_lag,
        fee_scale_default=1.0,
        domain_randomize=False,
    )

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
        vec_env = DummyVecEnv([lambda: raw_env])
        vec_env = VecNormalize.load(str(vec_norm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    obs, _ = raw_env.reset(seed=0)
    if use_vec_norm:
        obs = vec_env.normalize_obs(obs)
    start_bar = int(raw_env._t)
    navs = [raw_env._nav(test_ohlcv[raw_env._t, :, 3])]
    weight_rows: list[np.ndarray] = []
    rewards = []
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
            deterministic=args.deterministic,
        )
        episode_starts = np.zeros((1,), dtype=bool)
        weight_rows.append(portfolio_weights_from_action(np.asarray(action).reshape(-1)))
        obs, r, done, truncated, info = raw_env.step(action)
        if use_vec_norm:
            obs = vec_env.normalize_obs(obs)
        rewards.append(r)
        if "nav" in info:
            navs.append(info["nav"])

    navs = np.array(navs, dtype=np.float64)
    weights = np.stack(weight_rows, axis=0) if weight_rows else np.zeros((0, 1))
    nav_ix = start_bar + np.arange(len(navs), dtype=np.int64)
    nav_ix = np.clip(nav_ix, 0, len(test_idx) - 1)
    time_nav = test_idx[nav_ix]

    if not args.no_viz:
        plot_dir = PLOTS_DIR / plot_run_id
        plot_dir.mkdir(parents=True, exist_ok=True)
        out = plot_dir / "backtest.png"
        time_w = None
        if weights.size > 0 and weights.shape[0] > 0:
            w_ix = start_bar + np.arange(weights.shape[0], dtype=np.int64)
            w_ix = np.clip(w_ix, 0, len(test_idx) - 1)
            time_w = test_idx[w_ix]
        plot_backtest_dashboard(
            time_nav,
            navs,
            weights=weights,
            weight_timestamps=time_w,
            asset_labels=["Cash"] + list(TICKERS),
            title=f"Backtest ({model_path.name})",
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
    print(f"Bars stepped: {len(rewards)}")
    print(f"Total return: {total_return * 100:.2f}%")
    print(f"Approx. annualized Sharpe (log-ret, daily): {sharpe:.2f}")
    print(f"Max drawdown: {_max_drawdown(navs) * 100:.2f}%")


if __name__ == "__main__":
    main()
