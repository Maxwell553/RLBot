#!/usr/bin/env python3
"""
Evaluate a trained RecurrentPPO (LSTM) policy on data reserved for OOS backtest only.

Uses the same **chronological holdout** as training (``reserve_chronological_holdout``),
which **must not** appear in ``scripts/train.py``. Dates default from
``runs/<run-id>/manifest.json`` when ``--run-id`` is set.

Tradeable universe: ``config/config.yaml`` → ``universe.assets`` (5–55). See docs/TRAINING.md.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path as _Path

_bootstrap_path = _Path(__file__).resolve().parent / "_bootstrap.py"
_bootstrap_spec = importlib.util.spec_from_file_location("_rlbot_repo_bootstrap", _bootstrap_path)
assert _bootstrap_spec is not None and _bootstrap_spec.loader is not None
_bootstrap_mod = importlib.util.module_from_spec(_bootstrap_spec)
_bootstrap_spec.loader.exec_module(_bootstrap_mod)

import argparse
import copy
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rlbot.data_utils import (
    clip_index_until,
    load_cache,
    resolve_panel_tickers,
    reserve_chronological_holdout,
)
from rlbot.run_artifacts import (
    PROJECT_ROOT,
    RunPaths,
    read_latest_run_id,
    read_run_manifest,
    resolve_data_cache,
)
from rlbot.baselines import (
    balanced_6040_nav,
    benchmark_buyhold_nav,
    benchmark_metrics,
    equal_weight_buyhold_nav,
    naive_risk_parity_nav,
    portfolio_step_nav,
)
from rlbot.rl_config import observation_dim_for_universe
from rlbot.trading_env import MultiAssetPortfolioEnv, portfolio_weights_from_action
from rlbot.vecnorm_utils import freeze_vec_normalize_for_inference
from rlbot.visualize import open_plot_file, plot_backtest_dashboard

ROOT = PROJECT_ROOT
DATA_CACHE = resolve_data_cache()
PLOTS_DIR = ROOT / "plots"
MODELS_DIR = ROOT / "models"


def _assert_manifest_panel_compatible(
    manifest: dict | None,
    panel_tickers: list[str],
    n_assets: int,
) -> None:
    """Ensure cache/config panel matches the training run recorded in manifest.json."""
    if not manifest:
        return
    uni = manifest.get("universe")
    if not isinstance(uni, dict):
        return
    exp_tickers = uni.get("tickers")
    if exp_tickers and [str(t) for t in exp_tickers] != list(panel_tickers):
        raise ValueError(
            f"Ticker order mismatch: manifest {exp_tickers!r} vs panel {panel_tickers!r}. "
            "Use --refresh-data after editing config.yaml universe.assets."
        )
    if uni.get("n_assets") is not None and int(uni["n_assets"]) != n_assets:
        raise ValueError(
            f"manifest n_assets={uni['n_assets']} but panel has {n_assets} assets"
        )
    want_obs = observation_dim_for_universe(n_assets)
    if uni.get("obs_dim") is not None and int(uni["obs_dim"]) != want_obs:
        raise ValueError(
            f"manifest obs_dim={uni['obs_dim']} but current layout needs {want_obs} "
            f"(N={n_assets})"
        )


@dataclass
class BacktestResult:
    run_id: str
    model_path: Path
    checkpoint_label: str
    total_return: float
    sharpe: float
    max_drawdown: float
    n_bars: int
    seed_label: str = ""


def _infer_run_id_from_model_path(model_path: Path) -> str | None:
    parts = model_path.resolve().parts
    for i, name in enumerate(parts):
        if name == "models" and i + 1 < len(parts):
            nxt = parts[i + 1]
            if nxt not in ("best",):
                return nxt
    return None


def _resolve_model_path(args: argparse.Namespace) -> tuple[Path, str | None]:
    """Return (model_zip, run_id hint for naming plots, or None).

  Ex-ante holdout rule (default): only ``best/best_model.zip`` (max in-training
  eval NAV). Use ``--allow-latest-checkpoint`` to permit ``ppo_portfolio_final.zip``.
    """
    if args.model:
        p = Path(args.model)
        if not p.is_file():
            raise FileNotFoundError(f"Model not found: {p}")
        if not getattr(args, "allow_latest_checkpoint", False):
            name = p.name.lower()
            if name != "best_model.zip" and "best" not in p.parts:
                print(
                    "WARNING: --model overrides ex-ante rule; holdout metrics are "
                    "not pre-registered to eval-NAV-best only."
                )
        return p, _infer_run_id_from_model_path(p)

    rid = args.run_id.strip()
    if rid:
        rp = RunPaths(rid)
        best = rp.best_model_dir / "best_model.zip"
        if getattr(args, "allow_latest_checkpoint", False):
            for cand in (rp.final_model, best):
                if cand.is_file():
                    return cand, rid
        else:
            if best.is_file():
                return best, rid
        raise FileNotFoundError(
            f"No best_model.zip under models/{rid}/ "
            f"(ex-ante holdout rule: eval-NAV-best only; use --allow-latest-checkpoint for final weights)"
        )

    latest = read_latest_run_id()
    if latest:
        rp = RunPaths(latest)
        best = rp.best_model_dir / "best_model.zip"
        if getattr(args, "allow_latest_checkpoint", False):
            for cand in (rp.final_model, best):
                if cand.is_file():
                    return cand, latest
        elif best.is_file():
            return best, latest

    raise FileNotFoundError(
        "No model found. Train first (writes runs/LATEST.txt), pass --run-id, or --model path/to.zip"
    )


def _resolve_model_path_for_run(
    run_id: str,
    *,
    allow_latest_checkpoint: bool = False,
    model_override: Path | None = None,
) -> Path:
    if model_override is not None:
        p = Path(model_override)
        if not p.is_file():
            raise FileNotFoundError(f"Model not found: {p}")
        return p
    rp = RunPaths(run_id)
    best = rp.best_model_dir / "best_model.zip"
    if allow_latest_checkpoint:
        for cand in (rp.final_model, best):
            if cand.is_file():
                return cand
    elif best.is_file():
        return best
    raise FileNotFoundError(
        f"No checkpoint for {run_id} "
        f"(best={'yes' if best.is_file() else 'no'}, "
        f"final={'yes' if rp.final_model.is_file() else 'no'})"
    )


def discover_ensemble_run_ids(prefix: str, seeds: list[int] | None = None) -> list[str]:
    """``models/<prefix>_seed_<n>`` directories, sorted by seed then name."""
    if not prefix:
        return []
    found: list[tuple[int, str]] = []
    pat = re.compile(rf"^{re.escape(prefix)}_seed_(\d+)$")
    for p in MODELS_DIR.iterdir():
        if not p.is_dir():
            continue
        m = pat.match(p.name)
        if not m:
            continue
        seed = int(m.group(1))
        if seeds is not None and seed not in seeds:
            continue
        found.append((seed, p.name))
    found.sort(key=lambda x: x[0])
    return [name for _, name in found]


def _seed_from_run_id(run_id: str, prefix: str) -> str:
    m = re.search(r"_seed_(\d+)$", run_id)
    if m:
        return m.group(1)
    return run_id.removeprefix(prefix + "_seed_") if run_id.startswith(prefix + "_seed_") else run_id


def resolve_oos_holdout(
    args: argparse.Namespace,
    idx: pd.DatetimeIndex,
    ohlcv: np.ndarray,
    rsi: np.ndarray,
    macd: np.ndarray,
    macro: np.ndarray,
    fracdiff: np.ndarray,
    fracdiff_macro: np.ndarray,
    trend: np.ndarray,
    manifest: dict | None,
) -> tuple[
    pd.DatetimeIndex,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
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

    _, holdout = reserve_chronological_holdout(
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
    test_idx = holdout[0]
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
    return holdout


def run_oos_backtest(args: argparse.Namespace) -> BacktestResult:
    """Single OOS deterministic rollout; optional plot via ``args``."""
    run_id = args.run_id.strip()
    if not run_id:
        raise ValueError("run_id required")
    manifest = read_run_manifest(run_id)
    model_path = _resolve_model_path_for_run(
        run_id,
        allow_latest_checkpoint=args.allow_latest_checkpoint,
        model_override=Path(args.model) if args.model.strip() else None,
    )
    ckpt_label = "latest" if args.allow_latest_checkpoint and model_path.name != "best_model.zip" else "best"

    (
        idx,
        ohlcv,
        rsi,
        macd,
        macro,
        fracdiff,
        fracdiff_macro,
        trend,
        cache_tickers,
    ) = load_cache(str(DATA_CACHE))
    panel_tickers = resolve_panel_tickers(manifest, cache_tickers)
    n_assets = int(ohlcv.shape[1])
    _assert_manifest_panel_compatible(manifest, panel_tickers, n_assets)
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

    test_idx, test_ohlcv, test_rsi, test_macd, test_macro, test_fd, test_fdm, test_trend = (
        resolve_oos_holdout(args, idx, ohlcv, rsi, macd, macro, fracdiff, fracdiff_macro, trend, manifest)
    )
    if len(test_idx) < 10:
        raise RuntimeError("Test window too short; fetch more history or reduce holdout days.")

    model = RecurrentPPO.load(str(model_path), device="auto")
    explicit_vn = Path(args.vec_normalize).expanduser().resolve() if args.vec_normalize.strip() else None
    vec_norm_path = _find_vec_normalize(model_path, run_id, explicit=explicit_vn)
    use_vec_norm = vec_norm_path.is_file()
    if not use_vec_norm and args.require_vec_normalize:
        raise FileNotFoundError(f"No VecNormalize stats at {vec_norm_path}")

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
        deterministic=True,
        collect_weights=not args.no_viz,
    )
    nav_ensemble: np.ndarray | None = None
    n_stoch = int(args.stochastic_paths)
    if n_stoch > 0 and not getattr(args, "_ensemble_mode", False):
        print(f"Stochastic ensemble: {n_stoch} paths (deterministic=False, same holdout window)")
    if n_stoch > 0:
        nav_ensemble = rollout_stochastic_ensemble(
            model,
            n_paths=n_stoch,
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
        )
        if nav_ensemble.shape[1] != len(navs):
            m = min(nav_ensemble.shape[1], len(navs))
            navs = navs[:m]
            nav_ensemble = nav_ensemble[:, :m]

    log_rets = np.diff(np.log(np.maximum(navs, 1e-12)))
    total_return = float(navs[-1] / navs[0] - 1.0)
    sharpe = _sharpe_ann_from_log_rets(log_rets)

    if not args.no_viz and not getattr(args, "_ensemble_mode", False):
        plot_dir = PLOTS_DIR / run_id
        plot_dir.mkdir(parents=True, exist_ok=True)
        tag = args.plot_tag.strip()
        dash_name = f"backtest_{tag}.png" if tag else "backtest.png"
        out = plot_dir / dash_name
        nav_ix = start_bar + np.arange(len(navs), dtype=np.int64)
        nav_ix = np.clip(nav_ix, 0, len(test_idx) - 1)
        time_nav = test_idx[nav_ix]
        weights = w_opt if w_opt is not None else np.zeros((0, 1))
        time_w = None
        if weights.size > 0 and weights.shape[0] > 0:
            w_ix = start_bar + np.arange(weights.shape[0], dtype=np.int64)
            w_ix = np.clip(w_ix, 0, len(test_idx) - 1)
            time_w = test_idx[w_ix]
        nav_spy = benchmark_buyhold_nav(
            navs, test_ohlcv, start_bar, tickers=panel_tickers
        )
        nav_ew = equal_weight_buyhold_nav(navs, test_ohlcv, start_bar)
        nav_6040 = balanced_6040_nav(
            navs, test_ohlcv, start_bar, test_idx, tickers=panel_tickers
        )
        nav_rp = naive_risk_parity_nav(navs, test_ohlcv, start_bar)
        model_label = f"Model ({tag})" if tag else f"Model ({model_path.stem})"
        plot_backtest_dashboard(
            time_nav,
            navs,
            nav_spy=nav_spy,
            nav_equal_weight=nav_ew,
            nav_balanced_6040=nav_6040,
            nav_risk_parity=nav_rp,
            nav_stochastic_ensemble=nav_ensemble,
            weights=weights,
            weight_timestamps=time_w,
            asset_labels=["Cash"] + list(panel_tickers),
            model_label=model_label,
            title="OOS backtest vs benchmarks",
            save_path=out,
        )
        print(f"Backtest plot: {out}")
        if args.show_viz:
            open_plot_file(out)

    if args.detailed and not getattr(args, "_ensemble_mode", False):
        _print_detailed_stats(
            test_idx=test_idx,
            navs=navs,
            log_rets=log_rets,
            ohlcv_window=test_ohlcv,
            start_bar=start_bar,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_avg_block=args.bootstrap_avg_block,
            nav_ensemble=nav_ensemble,
        )

    prefix = getattr(args, "ensemble_prefix", "") or ""
    return BacktestResult(
        run_id=run_id,
        model_path=model_path,
        checkpoint_label=ckpt_label,
        total_return=total_return,
        sharpe=sharpe,
        max_drawdown=_max_drawdown(navs),
        n_bars=len(test_idx),
        seed_label=_seed_from_run_id(run_id, prefix) if prefix else "",
    )


def _print_ensemble_summary(prefix: str, checkpoint_label: str, results: list[BacktestResult]) -> None:
    print(f"\n=== Ensemble OOS summary ({prefix}, checkpoint={checkpoint_label}) ===")
    print(f"{'seed':>8}  {'return %':>10}  {'Sharpe':>8}  {'max DD %':>10}")
    rets, sharpes, dds = [], [], []
    for r in results:
        seed = r.seed_label or r.run_id
        print(
            f"{seed:>8}  {r.total_return * 100:>10.2f}  {r.sharpe:>8.2f}  {r.max_drawdown * 100:>10.2f}"
        )
        rets.append(r.total_return)
        sharpes.append(r.sharpe)
        dds.append(r.max_drawdown)
    if len(results) >= 2:
        print(
            f"{'mean':>8}  {np.mean(rets) * 100:>10.2f}  {np.mean(sharpes):>8.2f}  {np.mean(dds) * 100:>10.2f}"
        )
        print(
            f"{'std':>8}  {np.std(rets, ddof=1) * 100:>10.2f}  {np.std(sharpes, ddof=1):>8.2f}  "
            f"{np.std(dds, ddof=1) * 100:>10.2f}"
        )
        print(
            f"{'μ±σ':>8}  "
            f"{np.mean(rets)*100:.2f}±{np.std(rets, ddof=1)*100:.2f}  "
            f"{np.mean(sharpes):.2f}±{np.std(sharpes, ddof=1):.2f}  "
            f"{np.mean(dds)*100:.2f}±{np.std(dds, ddof=1)*100:.2f}"
        )
    print()


def run_ensemble_backtests(args: argparse.Namespace) -> None:
    prefix = args.ensemble_prefix.strip()
    seeds: list[int] | None = None
    if args.ensemble_seeds.strip():
        seeds = [int(s.strip()) for s in args.ensemble_seeds.split(",") if s.strip()]
    run_ids = discover_ensemble_run_ids(prefix, seeds)
    if not run_ids:
        raise SystemExit(
            f"No runs found under models/ matching '{prefix}_seed_*'. "
            f"Train with scripts/run_seed_ensemble.sh --cohort {prefix}"
        )
    print(f"Discovered {len(run_ids)} runs: {', '.join(run_ids)}")

    modes: list[tuple[str, bool]] = []
    ck = args.ensemble_checkpoint
    if ck in ("best", "both"):
        modes.append(("best", False))
    if ck in ("latest", "both"):
        modes.append(("latest", True))

    args._ensemble_mode = True  # type: ignore[attr-defined]
    args.no_viz = True

    summary_root: dict[str, object] = {"prefix": prefix, "checkpoints": {}}
    for label, allow_latest in modes:
        sub_results: list[BacktestResult] = []
        for rid in run_ids:
            print(f"\n--- {rid} ({label}) ---")
            sub = copy.copy(args)
            sub.run_id = rid
            sub.allow_latest_checkpoint = allow_latest
            sub._ensemble_mode = True  # type: ignore[attr-defined]
            try:
                sub_results.append(run_oos_backtest(sub))
            except FileNotFoundError as e:
                print(f"SKIP {rid}: {e}")
        if not sub_results:
            print(f"No successful backtests for checkpoint={label}")
            continue
        _print_ensemble_summary(prefix, label, sub_results)
        summary_root["checkpoints"][label] = [asdict(r) for r in sub_results]

    out_dir = PLOTS_DIR / prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "ensemble_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_root, f, indent=2, default=str)
    print(f"Wrote {out_json}")


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


def block_bootstrap_log_rets(
    log_rets: np.ndarray,
    n_resamples: int = 5000,
    avg_block_size: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """Stationary (Politis–Romano style) block bootstrap Sharpe samples.

    Builds synthetic return series by stitching contiguous blocks; block breaks
    arrive with geometric probability ``1 / avg_block_size``. Indices wrap
    circularly to preserve length without edge truncation.
    """
    log_rets = np.asarray(log_rets, dtype=np.float64).reshape(-1)
    n = log_rets.size
    if n < 2:
        return np.full(n_resamples, np.nan, dtype=np.float64)
    boot_sharpes = np.empty(n_resamples, dtype=np.float64)
    p = 1.0 / max(int(avg_block_size), 1)
    rng = np.random.default_rng(seed)

    for b in range(n_resamples):
        sim_idx = np.empty(n, dtype=np.int64)
        curr_idx = int(rng.integers(0, n))
        for i in range(n):
            sim_idx[i] = curr_idx
            if rng.random() < p:
                curr_idx = int(rng.integers(0, n))
            else:
                curr_idx = (curr_idx + 1) % n
        sample_rets = log_rets[sim_idx]
        boot_sharpes[b] = _sharpe_ann_from_log_rets(sample_rets)
    return boot_sharpes


def block_bootstrap_sharpe_percentiles(
    log_rets: np.ndarray,
    n_resamples: int = 5000,
    avg_block_size: int = 10,
    seed: int = 42,
) -> tuple[float, float, float]:
    """2.5 / 50 / 97.5 percentiles of block-bootstrap annualized Sharpe."""
    sharpes = block_bootstrap_log_rets(
        log_rets, n_resamples=n_resamples, avg_block_size=avg_block_size, seed=seed
    )
    sharpes = sharpes[np.isfinite(sharpes)]
    if sharpes.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.percentile(sharpes, 2.5)),
        float(np.percentile(sharpes, 50.0)),
        float(np.percentile(sharpes, 97.5)),
    )


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
    reset_seed: int = 0,
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
                want = observation_dim_for_universe(raw_env.n_assets)
                raise RuntimeError(
                    f"VecNormalize obs dim mismatch for {vec_norm_path.name}: "
                    f"saved stats expect a different layout than this env "
                    f"({cur}-dim vs {want}-dim for N={raw_env.n_assets} assets). "
                    f"Train a new run with the current config/cache (--run-id) or align "
                    f"universe.assets with the checkpoint manifest."
                ) from e
            raise
        freeze_vec_normalize_for_inference(vec_env)

    obs, _ = raw_env.reset(seed=int(reset_seed))
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
            w_rows.append(
                portfolio_weights_from_action(np.asarray(action).reshape(-1))
            )
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


def rollout_stochastic_ensemble(
    model: RecurrentPPO,
    *,
    n_paths: int,
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
    base_seed: int = 0,
) -> np.ndarray:
    """``n_paths`` stochastic rollouts (``deterministic=False``); shape (n_paths, len(navs))."""
    paths: list[np.ndarray] = []
    for i in range(int(n_paths)):
        navs, _, _, _ = rollout_policy_on_slice(
            model,
            test_idx=test_idx,
            test_ohlcv=test_ohlcv,
            test_rsi=test_rsi,
            test_macd=test_macd,
            test_macro=test_macro,
            test_fd=test_fd,
            test_fdm=test_fdm,
            test_trend=test_trend,
            obs_lag=obs_lag,
            vec_norm_path=vec_norm_path,
            use_vec_norm=use_vec_norm,
            deterministic=False,
            collect_weights=False,
            reset_seed=base_seed + i + 1,
        )
        paths.append(navs)
    min_len = min(len(p) for p in paths)
    if min_len < 2:
        raise RuntimeError("Stochastic ensemble paths too short")
    return np.stack([p[:min_len] for p in paths], axis=0)


def _print_detailed_stats(
    *,
    test_idx: pd.DatetimeIndex,
    navs: np.ndarray,
    log_rets: np.ndarray,
    ohlcv_window: np.ndarray,
    start_bar: int,
    spy_ohlcv_col: int | None = None,
    bootstrap_resamples: int = 8000,
    bootstrap_avg_block: int = 10,
    nav_ensemble: np.ndarray | None = None,
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

    nav_spy = benchmark_buyhold_nav(
        navs, ohlcv_window, start_bar, tickers=None, benchmark_col=spy_ohlcv_col
    )
    lr_spy = np.diff(np.log(np.maximum(nav_spy, 1e-12)))
    sh_spy = _sharpe_ann_from_log_rets(lr_spy)
    print(
        f"SPY 100% buy&hold (same OOS path, {len(lr_spy)} daily rets): "
        f"total {(nav_spy[-1] / nav_spy[0] - 1) * 100:.2f}%, ann. Sharpe {sh_spy:.2f}"
    )
    nav_ew = equal_weight_buyhold_nav(navs, ohlcv_window, start_bar)
    lr_ew = np.diff(np.log(np.maximum(nav_ew, 1e-12)))
    sh_ew = _sharpe_ann_from_log_rets(lr_ew)
    print(
        f"Equal-weight buy&hold (1/N assets, daily rebal., {len(lr_ew)} daily rets): "
        f"total {(nav_ew[-1] / nav_ew[0] - 1) * 100:.2f}%, ann. Sharpe {sh_ew:.2f}"
    )
    nav_6040 = balanced_6040_nav(navs, ohlcv_window, start_bar, test_idx)
    ret_6040, sh_6040, _ = benchmark_metrics(nav_6040)
    print(
        f"60/40 SP500/IEF (monthly rebal., {len(lr_ew)} daily rets): "
        f"total {ret_6040 * 100:.2f}%, ann. Sharpe {sh_6040:.2f}"
    )
    nav_rp = naive_risk_parity_nav(navs, ohlcv_window, start_bar)
    ret_rp, sh_rp, _ = benchmark_metrics(nav_rp)
    print(
        f"Naive risk parity (inverse 20d vol, daily rebal., {len(lr_ew)} daily rets): "
        f"total {ret_rp * 100:.2f}%, ann. Sharpe {sh_rp:.2f}"
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
        lo, med, hi = block_bootstrap_sharpe_percentiles(
            log_rets,
            n_resamples=bootstrap_resamples,
            avg_block_size=bootstrap_avg_block,
            seed=42,
        )
        print(
            f"Block-bootstrap Sharpe ({bootstrap_resamples} resamples, "
            f"avg block ~{bootstrap_avg_block}d, stationary): "
            f"2.5%={lo:.2f}, 50%={med:.2f}, 97.5%={hi:.2f}"
        )
    if nav_ensemble is not None and nav_ensemble.ndim == 2 and nav_ensemble.shape[0] >= 2:
        ens_rets = np.diff(np.log(np.maximum(nav_ensemble, 1e-12)), axis=1)
        ens_sh = np.array([_sharpe_ann_from_log_rets(ens_rets[i]) for i in range(ens_rets.shape[0])])
        ens_sh = ens_sh[np.isfinite(ens_sh)]
        if ens_sh.size:
            print(
                f"Stochastic ensemble ({ens_sh.size} paths, policy sampling): "
                f"Sharpe mean={ens_sh.mean():.2f}, "
                f"5–95%=[{np.percentile(ens_sh, 5):.2f}, {np.percentile(ens_sh, 95):.2f}]"
            )
            tot = nav_ensemble[:, -1] / np.maximum(nav_ensemble[:, 0], 1e-12) - 1.0
            print(
                f"  Total return across paths: "
                f"5%={np.percentile(tot, 5) * 100:.1f}%, "
                f"50%={np.percentile(tot, 50) * 100:.1f}%, "
                f"95%={np.percentile(tot, 95) * 100:.1f}%"
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
    parser.add_argument(
        "--allow-latest-checkpoint",
        action="store_true",
        help="Allow ppo_portfolio_final.zip on holdout (breaks ex-ante eval-NAV-best rule).",
    )
    parser.add_argument(
        "--stochastic-paths",
        type=int,
        default=0,
        metavar="N",
        help="If >0, run N stochastic policy rollouts and plot equity fan (default 0).",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=8000,
        help="Block-bootstrap resamples for Sharpe CI (--detailed).",
    )
    parser.add_argument(
        "--bootstrap-avg-block",
        type=int,
        default=10,
        help="Mean block length (days) for stationary block bootstrap.",
    )
    parser.add_argument(
        "--ensemble-prefix",
        default="",
        metavar="PREFIX",
        help="Aggregate OOS metrics over models/<PREFIX>_seed_* runs (μ±σ table).",
    )
    parser.add_argument(
        "--ensemble-checkpoint",
        default="best",
        choices=("best", "latest", "both"),
        help="Checkpoint type when using --ensemble-prefix (default: best).",
    )
    parser.add_argument(
        "--ensemble-seeds",
        default="",
        metavar="LIST",
        help="Comma-separated seeds to include (default: all matching PREFIX_seed_* dirs).",
    )
    args = parser.parse_args()

    if args.ensemble_prefix.strip():
        run_ensemble_backtests(args)
        return

    if not args.run_id.strip():
        _model_path, run_hint = _resolve_model_path(args)
        if run_hint:
            args.run_id = run_hint
        elif not args.model.strip():
            latest = read_latest_run_id()
            if latest:
                args.run_id = latest

    if not args.run_id.strip() and not args.model.strip():
        raise SystemExit("Pass --run-id, --model, or --ensemble-prefix")

    print(f"Model run: {args.run_id or '(from --model)'}")
    print(f"Backtest plot folder (run tag): plots/{args.run_id}/")

    result = run_oos_backtest(args)
    print(f"Model: {result.model_path}")
    if result.checkpoint_label == "best":
        print("Ex-ante checkpoint: eval-NAV-best (best_model.zip) — holdout not used to pick weights.")
    print(f"OOS bars: {result.n_bars}")
    print(f"Total return: {result.total_return * 100:.2f}%")
    print(f"Approx. annualized Sharpe (log-ret, daily): {result.sharpe:.2f}")
    print(f"Max drawdown: {result.max_drawdown * 100:.2f}%")


if __name__ == "__main__":
    main()
