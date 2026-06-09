#!/usr/bin/env python3
"""
Evaluate a trained RecurrentPPO (LSTM) policy on data reserved for OOS backtest only.

Uses the same **chronological holdout** as training (``reserve_chronological_holdout``),
which **must not** appear in ``scripts/train.py``. Requires ``--run-id``; holdout
dates and universe come from ``Runs/<run-id>/manifest.json`` (PyTorch loads lazily).

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
import os
import copy
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


def _bt_log(msg: str) -> None:
    print(msg, flush=True)


import numpy as np
import pandas as pd

from rlbot.data_utils import (
    clip_index_until,
    load_cache,
    resolve_panel_tickers,
    reserve_chronological_holdout,
)
from rlbot.research import oos_ledger
from rlbot.stats import deflated_sharpe_ratio
from rlbot.run_artifacts import (
    check_holdout_window_against_manifest,
    PROJECT_ROOT,
    RUNS_ROOT,
    RunPaths,
    config_sha256,
    discover_run_ids_with_models,
    git_provenance,
    read_run_manifest,
    resolve_data_cache,
    resolve_run_data_cache,
    sha256_file,
)
from rlbot.baselines import (
    balanced_6040_nav,
    benchmark_buyhold_nav,
    benchmark_metrics,
    benchmark_only_nav,
    cash_nav,
    equal_weight_daily_cost_aware_nav,
    equal_weight_monthly_nav,
    naive_risk_parity_nav,
    portfolio_step_nav,
)
from rlbot.inference_load import (
    load_recurrent_ppo_inference,
    load_vec_normalize_for_inference,
    swap_recurrent_ppo_weights,
)
from rlbot.rl_config import get_config, load_config, observation_dim_for_universe, set_config
from rlbot.stats import (
    block_bootstrap_sharpe_percentiles,
    sharpe_ann_from_log_rets as _sharpe_ann_from_log_rets,
)

# Lazy-loaded in ensure_backtest_dependencies() (once per process).
th = None
RecurrentPPO = None
DummyVecEnv = None
VecNormalize = None
MultiAssetPortfolioEnv = None
portfolio_weights_from_action = None
freeze_vec_normalize_for_inference = None
_DEPS_LOADED = False
_PANEL_CACHE: tuple | None = None
_INFERENCE_POLICY: object | None = None  # RecurrentPPO shell reused across checkpoints in batch
_INFERENCE_POLICY_KEY: str | None = None
ROOT = PROJECT_ROOT
DATA_CACHE = resolve_data_cache()


def ensure_backtest_dependencies() -> None:
    """Import PyTorch/SB3/env once per Python process."""
    global _DEPS_LOADED, th, RecurrentPPO, DummyVecEnv, VecNormalize
    global MultiAssetPortfolioEnv, portfolio_weights_from_action
    global freeze_vec_normalize_for_inference
    if _DEPS_LOADED:
        return
    t0 = time.perf_counter()
    _bt_log("[backtest] Loading dependencies (once per process)...")
    t1 = time.perf_counter()
    import torch as _th

    th = _th
    _bt_log(f"[backtest]   torch ({time.perf_counter() - t1:.1f}s)")
    t1 = time.perf_counter()
    from sb3_contrib import RecurrentPPO as _RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv as _DummyVecEnv, VecNormalize as _VN

    RecurrentPPO = _RecurrentPPO
    DummyVecEnv = _DummyVecEnv
    VecNormalize = _VN
    _bt_log(f"[backtest]   stable-baselines3 ({time.perf_counter() - t1:.1f}s)")
    t1 = time.perf_counter()
    from rlbot.trading_env import MultiAssetPortfolioEnv as _Env, portfolio_weights_from_action as _pwf
    from rlbot.vecnorm_utils import freeze_vec_normalize_for_inference as _freeze
    _bt_log(f"[backtest]   trading env ({time.perf_counter() - t1:.1f}s)")

    MultiAssetPortfolioEnv = _Env
    portfolio_weights_from_action = _pwf
    freeze_vec_normalize_for_inference = _freeze
    _DEPS_LOADED = True
    _bt_log(f"[backtest] Dependencies ready ({time.perf_counter() - t0:.1f}s).")


def _get_shared_panel(cache_path: str | Path | None = None) -> tuple:
    """Load a panel cache once per process, memoized per resolved path."""
    global _PANEL_CACHE
    key = str(cache_path) if cache_path is not None else str(DATA_CACHE)
    if not isinstance(_PANEL_CACHE, dict):
        _PANEL_CACHE = {}
    if key not in _PANEL_CACHE:
        _bt_log(f"[backtest] Loading market cache (once per path): {key}")
        t0 = time.perf_counter()
        _PANEL_CACHE[key] = load_cache(key, expected_fracdiff_d=get_config().data.fracdiff_d)
        _bt_log(f"[backtest] Cache loaded ({time.perf_counter() - t0:.1f}s).")
    return _PANEL_CACHE[key]


def _resolve_run_data_cache(run_id: str, args: argparse.Namespace) -> Path:
    """Cache path for a run: --data-cache > run-local snapshot > global cache."""
    return resolve_run_data_cache(
        run_id, getattr(args, "data_cache", ""), default=DATA_CACHE
    )


def _maybe_load_run_config(run_id: str, args: argparse.Namespace) -> None:
    """Bind the run-local config snapshot for inference unless overridden.

    With no snapshot, rebind to the fresh global default — otherwise, in a batch
    (``--run-ids``), a run without a snapshot would silently inherit the *previous*
    run's run-local config.
    """
    if getattr(args, "use_current_config", False):
        _bt_log("[backtest] Using current global config (--use-current-config).")
        return
    snap = RunPaths(run_id).config_snapshot
    if snap.is_file():
        set_config(load_config(snap))
        _bt_log(f"[backtest] Loaded run-local config snapshot: {snap}")
    else:
        set_config(load_config())  # fresh global default, no cross-run bleed
        _bt_log(f"[backtest] No config snapshot at {snap}; using fresh global config.")


def _clear_inference_policy_cache() -> None:
    global _INFERENCE_POLICY, _INFERENCE_POLICY_KEY
    _INFERENCE_POLICY = None
    _INFERENCE_POLICY_KEY = None


def _load_inference_policy(
    model_path: Path,
    *,
    device: str = "cpu",
    full_reload: bool = False,
) -> object:
    """
    Load policy for rollout. First call builds the LSTM policy only (no AdamW).
    Later calls in the same process swap weights via ``set_parameters``.
    """
    global _INFERENCE_POLICY, _INFERENCE_POLICY_KEY
    ensure_backtest_dependencies()
    path_key = str(model_path.resolve())
    if full_reload:
        _clear_inference_policy_cache()

    if _INFERENCE_POLICY is not None:
        cached_obs_dim = _INFERENCE_POLICY.policy.observation_space.shape[0]
        required_obs_dim = observation_dim_for_universe(get_config().universe.n_assets)
        if cached_obs_dim != required_obs_dim:
            _bt_log(
                f"[backtest] Universe size shift detected ({cached_obs_dim}d → {required_obs_dim}d). "
                "Clearing shell cache..."
            )
            _clear_inference_policy_cache()

    if _INFERENCE_POLICY is not None:
        if _INFERENCE_POLICY_KEY == path_key:
            return _INFERENCE_POLICY
        t0 = time.perf_counter()
        swap_recurrent_ppo_weights(_INFERENCE_POLICY, path_key, device=device)
        _INFERENCE_POLICY_KEY = path_key
        _bt_log(f"[backtest] Swapped policy weights ({time.perf_counter() - t0:.1f}s).")
        return _INFERENCE_POLICY

    t0 = time.perf_counter()
    _bt_log(f"[backtest] Loading policy weights ({model_path.name})...")
    _INFERENCE_POLICY = load_recurrent_ppo_inference(path_key, device=device)
    _INFERENCE_POLICY_KEY = path_key
    _bt_log(f"[backtest] Policy ready ({time.perf_counter() - t0:.1f}s).")
    return _INFERENCE_POLICY


def _latest_step_checkpoint(run_id: str) -> Path | None:
    ckpt_dir = RunPaths(run_id).models_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    best: Path | None = None
    best_step = -1
    for p in ckpt_dir.glob("ppo_*_steps.zip"):
        m = re.search(r"ppo_(\d+)_steps\.zip$", p.name)
        if m and int(m.group(1)) > best_step:
            best_step = int(m.group(1))
            best = p
    return best


def _parse_run_id_list(spec: str) -> list[str]:
    spec = spec.strip()
    if not spec:
        return []
    return [x.strip() for x in spec.split(",") if x.strip()]


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
    # return-distribution stats for selection-aware significance (PSR/DSR)
    n_rets: int = 0
    ret_skew: float = float("nan")
    ret_kurt: float = float("nan")


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
    """``Runs/<prefix>_seed_<n>/`` (or legacy ``models/``), sorted by seed then name."""
    if not prefix:
        return []
    found: list[tuple[int, str]] = []
    pat = re.compile(rf"^{re.escape(prefix)}_seed_(\d+)$")
    for rid in discover_run_ids_with_models():
        m = pat.match(rid)
        if not m:
            continue
        seed = int(m.group(1))
        if seeds is not None and seed not in seeds:
            continue
        found.append((seed, rid))
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
    asset_vol: np.ndarray,
    macro_vol: np.ndarray,
    asset_live: np.ndarray,
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
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if not manifest:
        raise ValueError(
            "Runs/<run-id>/manifest.json is required for OOS dates; pass --run-id from a completed train run."
        )
    holdout_days = args.holdout_days
    train_end = args.train_end
    holdout_start = args.holdout_start
    holdout_end = args.holdout_end
    ch = manifest.get("chronological_holdout") or {}
    margs = manifest.get("args") or {}
    if train_end is None:
        train_end = ch.get("train_end") or margs.get("train_end")
    if holdout_start is None:
        holdout_start = ch.get("holdout_start") or margs.get("holdout_start")
    if holdout_end is None:
        holdout_end = ch.get("holdout_end") or margs.get("holdout_end")
    if holdout_days is None:
        if ch.get("holdout_days") is not None:
            holdout_days = int(ch["holdout_days"])
        elif margs.get("holdout_days") is not None:
            holdout_days = int(margs["holdout_days"])

    if train_end and holdout_start:
        print(
            f"Using date holdout from manifest/CLI: train_end={train_end}, "
            f"holdout_start={holdout_start}, holdout_end={holdout_end or '(last bar)'}"
        )
    elif holdout_days is not None:
        print(f"Using holdout_days={holdout_days} (calendar tail)")
    else:
        raise ValueError(
            "Manifest has no holdout dates (train_end/holdout_start or holdout_days); "
            "re-train or pass --train-end/--holdout-start on the CLI."
        )

    _, holdout = reserve_chronological_holdout(
        idx,
        ohlcv,
        rsi,
        macd,
        macro,
        fracdiff,
        fracdiff_macro,
        trend,
        asset_vol,
        macro_vol,
        asset_live,
        holdout_days=holdout_days,
        train_end=train_end,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
    )
    test_idx = holdout[0]

    # Cross-check the realized window against what training recorded (see
    # rlbot.run_artifacts.check_holdout_window_against_manifest). Explicit CLI window
    # flags (the documented cross-window check) downgrade the failure to a loud warning.
    cli_override = any(
        v is not None
        for v in (args.train_end, args.holdout_start, args.holdout_end, args.holdout_days)
    ) or bool((getattr(args, "until", "") or "").strip())
    warn = check_holdout_window_against_manifest(
        test_idx[0], test_idx[-1], ch, cli_override=cli_override
    )
    if warn:
        print(f"[backtest] WARNING: {warn} (explicit CLI window flags — proceeding)")

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
    ensure_backtest_dependencies()
    progress = not getattr(args, "no_progress", False)
    t_phase = time.perf_counter()

    run_id = args.run_id.strip()
    if not run_id:
        raise ValueError("run_id required")
    manifest = read_run_manifest(run_id)
    if manifest is None:
        raise FileNotFoundError(f"Missing Runs/{run_id}/manifest.json")
    # Bind the run's own config + data snapshot so OOS metrics are reproducible
    # regardless of the current global config / cache (override with flags).
    _maybe_load_run_config(run_id, args)
    cache_path = _resolve_run_data_cache(run_id, args)
    if not (getattr(args, "data_cache", "") or "").strip() and not RunPaths(
        run_id
    ).data_snapshot.is_file():
        print(
            f"[backtest] WARNING: run-local data snapshot Runs/{run_id}/data_cache.npz is "
            f"missing; falling back to {cache_path}. If that cache has newer bars and the "
            "manifest lacks an explicit holdout_end, the OOS window may extend past what "
            "training reserved — results may not be reproducible."
        )
    model_path = _resolve_model_path_for_run(
        run_id,
        allow_latest_checkpoint=args.allow_latest_checkpoint,
        model_override=Path(args.model) if args.model.strip() else None,
    )
    # OOS provenance: the label is derived from the weights actually evaluated, never
    # from --plot-tag (a tag once mislabeled final-model OOS numbers as "best").
    if model_path.name == "best_model.zip":
        ckpt_label = "best"
    elif model_path == RunPaths(run_id).final_model:
        ckpt_label = "final"
    else:
        ckpt_label = "latest"
    tag = (getattr(args, "plot_tag", None) or "").strip()
    if tag and tag != ckpt_label:
        print(
            f"[backtest] NOTE: --plot-tag {tag!r} names the plot only; metrics are "
            f"labeled by the evaluated checkpoint ({ckpt_label!r})."
        )
    _bt_log(f"[backtest] Checkpoint: {model_path} ({ckpt_label})")

    if getattr(args, "reuse_panel", False):
        (
            idx,
            ohlcv,
            rsi,
            macd,
            macro,
            fracdiff,
            fracdiff_macro,
            trend,
            asset_vol,
            macro_vol,
            asset_live,
            cache_tickers,
        ) = _get_shared_panel(cache_path)
    else:
        _bt_log(f"[backtest] Loading market cache: {cache_path}")
        (
            idx,
            ohlcv,
            rsi,
            macd,
            macro,
            fracdiff,
            fracdiff_macro,
            trend,
            asset_vol,
            macro_vol,
            asset_live,
            cache_tickers,
        ) = load_cache(str(cache_path), expected_fracdiff_d=get_config().data.fracdiff_d)
    n_assets = int(ohlcv.shape[1])
    # Compare the manifest against the RAW cache tickers — resolve_panel_tickers
    # prefers the manifest's own list, which would make the order check tautological.
    _assert_manifest_panel_compatible(manifest, cache_tickers, n_assets)
    panel_tickers = resolve_panel_tickers(manifest, cache_tickers)
    until = args.until
    if until is None and manifest:
        until = manifest.get("args", {}).get("until")
    if until:
        idx, (ohlcv, rsi, macd, macro, fracdiff, fracdiff_macro, trend, asset_vol, macro_vol, asset_live) = (
            clip_index_until(
                idx,
                ohlcv,
                rsi,
                macd,
                macro,
                fracdiff,
                fracdiff_macro,
                trend,
                asset_vol,
                macro_vol,
                asset_live,
                until=until,
            )
        )

    (
        test_idx,
        test_ohlcv,
        test_rsi,
        test_macd,
        test_macro,
        test_fd,
        test_fdm,
        test_trend,
        test_avol,
        test_mvol,
        test_live,
    ) = resolve_oos_holdout(
        args,
        idx,
        ohlcv,
        rsi,
        macd,
        macro,
        fracdiff,
        fracdiff_macro,
        trend,
        asset_vol,
        macro_vol,
        asset_live,
        manifest,
    )
    if len(test_idx) < 10:
        raise RuntimeError("Test window too short; fetch more history or reduce holdout days.")

    # Global holdout-burn accounting: EVERY backtest read lands in
    # Runs/oos_ledger.jsonl, recorded before the rollout so a crash still burns
    # (fail-closed). Keyed on the REGISTERED calendar window from the manifest when
    # present (the same key research budgets use), else the realized trading days.
    # Research-driven reads (RLBOT_OOS_CONTEXT=research:*) re-check the cumulative
    # window budget atomically at read time — a launch-time check alone would leave
    # an hours-long gap to concurrent burns.
    _ch = (manifest or {}).get("chronological_holdout") or {}
    _ledger_window = oos_ledger.window_key_for_read(
        _ch.get("holdout_start"), _ch.get("holdout_end"), test_idx[0], test_idx[-1]
    )
    _oos_context = os.environ.get("RLBOT_OOS_CONTEXT", "manual")
    _budget_env = os.environ.get("RLBOT_WINDOW_BUDGET", "")
    _enforce = None
    if _oos_context.startswith("research:"):
        _enforce = int(_budget_env) if _budget_env.strip() else oos_ledger.DEFAULT_WINDOW_BUDGET
    _cache_hash = sha256_file(cache_path)
    args._data_cache_hash = _cache_hash  # type: ignore[attr-defined]
    oos_ledger.record_oos_read(
        run_id=run_id,
        window=_ledger_window,
        checkpoint=getattr(args, "checkpoint", "") or "",
        data_cache_hash=_cache_hash,
        context=_oos_context,
        enforce_budget=_enforce,
    )
    args._ledger_window = _ledger_window  # type: ignore[attr-defined]
    _burn = oos_ledger.trials_for_window(_ledger_window)
    _bt_log(
        f"[backtest] OOS ledger: window {_ledger_window} has now been read by "
        f"{_burn} distinct model(s) — selection-aware significance (deflated Sharpe) "
        "uses this trial count."
    )

    _bt_log(
        f"[backtest] OOS holdout: {test_idx[0].date()} .. {test_idx[-1].date()} "
        f"({len(test_idx)} bars, setup {time.perf_counter() - t_phase:.1f}s)"
    )

    obs_lag = args.obs_lag
    if obs_lag is None:
        margs = manifest.get("args") or {}
        if margs.get("obs_lag") is not None:
            obs_lag = int(margs["obs_lag"])
        else:
            obs_lag = int(get_config().environment.obs_lag_default)
        _bt_log(f"[backtest] obs_lag={obs_lag} (from manifest/run config)")
    obs_lag = int(obs_lag)

    device = getattr(args, "device", "cpu") or "cpu"
    full_reload = bool(getattr(args, "full_policy_load", False))
    model = _load_inference_policy(model_path, device=device, full_reload=full_reload)

    explicit_vn = Path(args.vec_normalize).expanduser().resolve() if args.vec_normalize.strip() else None
    vec_norm_path = _find_vec_normalize(model_path, run_id, explicit=explicit_vn)
    use_vec_norm = vec_norm_path.is_file()
    if not use_vec_norm:
        if not getattr(args, "allow_missing_vec_normalize", False):
            raise FileNotFoundError(
                f"No VecNormalize stats at {vec_norm_path} "
                "(required for OOS backtest; pass --allow-missing-vec-normalize to debug without obs norm)"
            )
        if get_config().vec_normalize.norm_obs and not getattr(args, "allow_raw_obs", False):
            raise FileNotFoundError(
                f"No VecNormalize stats at {vec_norm_path}, but this run trained with "
                "vec_normalize.norm_obs: true — rolling out on raw observations would "
                "produce plausible-looking but meaningless metrics. Restore "
                "models/vec_normalize.pkl (or pass --vec-normalize PATH); "
                "--allow-raw-obs overrides this check explicitly."
            )
        if getattr(args, "allow_raw_obs", False):
            print(
                "[backtest] WARNING: no VecNormalize stats found — rolling out on RAW "
                "observations (--allow-raw-obs). Metrics are not comparable to training."
            )

    _bt_log("[backtest] Deterministic OOS rollout...")
    t_roll = time.perf_counter()
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
        test_asset_vol=test_avol,
        test_macro_vol=test_mvol,
        test_asset_live=test_live,
        obs_lag=obs_lag,
        vec_norm_path=vec_norm_path,
        use_vec_norm=use_vec_norm,
        deterministic=True,
        collect_weights=not args.no_viz,
        progress=progress,
        progress_label="deterministic",
    )
    _bt_log(f"[backtest] Rollout done ({time.perf_counter() - t_roll:.1f}s).")

    nav_ensemble: np.ndarray | None = None
    n_stoch = int(args.stochastic_paths)
    if n_stoch > 0 and not getattr(args, "_ensemble_mode", False):
        _bt_log(
            f"[backtest] Stochastic ensemble: {n_stoch} paths "
            "(deterministic=False, same holdout window)"
        )
    if n_stoch > 0:
        t_stoch = time.perf_counter()
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
            test_asset_vol=test_avol,
            test_macro_vol=test_mvol,
            test_asset_live=test_live,
            obs_lag=obs_lag,
            vec_norm_path=vec_norm_path,
            use_vec_norm=use_vec_norm,
            progress=progress,
        )
        _bt_log(f"[backtest] Stochastic paths done ({time.perf_counter() - t_stoch:.1f}s).")
        if nav_ensemble.shape[1] != len(navs):
            m = min(nav_ensemble.shape[1], len(navs))
            print(
                f"[backtest] WARNING: stochastic-path length ({nav_ensemble.shape[1]}) != "
                f"deterministic NAV length ({len(navs)}); truncating BOTH to {m} bars — "
                "the headline deterministic metrics below cover the truncated window."
            )
            navs = navs[:m]
            nav_ensemble = nav_ensemble[:, :m]

    log_rets = np.diff(np.log(np.maximum(navs, 1e-12)))
    total_return = float(navs[-1] / navs[0] - 1.0)
    sharpe = _sharpe_ann_from_log_rets(log_rets)
    _sd = float(np.std(log_rets)) + 1e-12
    _z = (log_rets - float(np.mean(log_rets))) / _sd
    ret_skew = float(np.mean(_z**3)) if log_rets.size >= 3 else float("nan")
    ret_kurt = float(np.mean(_z**4)) if log_rets.size >= 4 else float("nan")

    if not args.no_viz and not getattr(args, "_ensemble_mode", False):
        _bt_log("[backtest] Building plot...")
        plot_dir = RunPaths(run_id).plots_dir
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
        nav_ew = equal_weight_daily_cost_aware_nav(
            navs, test_ohlcv, start_bar, asset_live=test_live
        )
        try:
            nav_6040 = balanced_6040_nav(
                navs,
                test_ohlcv,
                start_bar,
                test_idx,
                tickers=panel_tickers,
                asset_live=test_live,
            )
        except KeyError:
            nav_6040 = None  # universe missing SP500/BOND10Y sleeve; skip 60/40 on the plot
        nav_rp = naive_risk_parity_nav(
            navs, test_ohlcv, start_bar, asset_live=test_live
        )
        model_label = f"Model ({tag})" if tag else f"Model ({model_path.stem})"
        from rlbot.visualize import open_plot_file, plot_backtest_dashboard

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

    detailed_stats: dict | None = None
    if args.detailed:
        _bt_log(
            f"[backtest] Detailed stats (bootstrap resamples={args.bootstrap_resamples})..."
        )
        t_det = time.perf_counter()
        detailed_stats = _print_detailed_stats(
            test_idx=test_idx,
            navs=navs,
            log_rets=log_rets,
            ohlcv_window=test_ohlcv,
            start_bar=start_bar,
            test_asset_live=test_live,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_avg_block=args.bootstrap_avg_block,
            nav_ensemble=nav_ensemble,
            bootstrap_progress=progress,
        )
        _bt_log(f"[backtest] Detailed stats done ({time.perf_counter() - t_det:.1f}s).")

    prefix = getattr(args, "ensemble_prefix", "") or ""
    result = BacktestResult(
        run_id=run_id,
        model_path=model_path,
        checkpoint_label=ckpt_label,
        total_return=total_return,
        sharpe=sharpe,
        max_drawdown=_max_drawdown(navs),
        n_bars=len(test_idx),
        seed_label=_seed_from_run_id(run_id, prefix) if prefix else "",
        n_rets=int(log_rets.size),
        ret_skew=ret_skew,
        ret_kurt=ret_kurt,
    )
    _write_backtest_summary(result, args, detailed_stats, cache_path, manifest)
    return result


def _write_backtest_summary(
    result: BacktestResult,
    args: argparse.Namespace,
    detailed: dict | None,
    cache_path: Path,
    manifest: dict | None = None,
) -> None:
    """Write a machine-readable per-run backtest summary."""
    from dataclasses import asdict

    cfg = get_config()
    config_hash = config_sha256(cfg.to_dict())
    data_cache_hash = getattr(args, "_data_cache_hash", None) or sha256_file(cache_path)
    # Drift detection: hashes are not just recorded, they are compared to training.
    hash_drift: dict[str, dict] = {}
    for key, now in (("config_hash", config_hash), ("data_cache_hash", data_cache_hash)):
        trained = (manifest or {}).get(key)
        if trained and trained != now:
            hash_drift[key] = {"training": trained, "backtest": now}
            print(
                f"[backtest] WARNING: {key} differs from the training manifest "
                f"({trained[:12]}… → {now[:12]}…); this backtest does not bind the "
                "exact training-time inputs and may not reproduce the run's OOS numbers."
            )
    # Selection-aware significance: the deflated Sharpe deflates the observed Sharpe
    # by the expected max-Sharpe of N zero-skill models, where N = distinct models
    # that have read this window per the global OOS ledger.
    # Use the window ACTUALLY read this invocation (recorded in the ledger at
    # rollout time) — under an explicit CLI cross-window override it differs from
    # the manifest's recorded window.
    window = getattr(args, "_ledger_window", None)
    if window is None:
        ch = (manifest or {}).get("chronological_holdout") or {}
        if ch.get("holdout_start") and ch.get("holdout_end"):
            window = oos_ledger.window_key(ch["holdout_start"], ch["holdout_end"])
    n_trials = 1
    if window:
        # A batch/ensemble is ONE selection event: every seed's DSR must use the same
        # trial count (prior burns ∪ the whole invocation), not an order-dependent
        # running count.
        prior = oos_ledger.distinct_models_for_window(
            oos_ledger.read_ledger(on_corrupt="raise"), window
        )
        invocation = {str(r) for r in getattr(args, "_invocation_run_ids", [])}
        n_trials = max(1, len(prior | invocation))
    dsr = None
    if result.n_rets >= 4 and np.isfinite(result.sharpe):
        dsr = float(
            deflated_sharpe_ratio(
                result.sharpe,
                n_obs=result.n_rets,
                n_trials=n_trials,
                skew=result.ret_skew if np.isfinite(result.ret_skew) else 0.0,
                kurt=result.ret_kurt if np.isfinite(result.ret_kurt) else 3.0,
            )
        )
        if not np.isfinite(dsr):
            dsr = None
        else:
            print(
                f"[backtest] Deflated Sharpe (vs best of {n_trials} model(s) on this "
                f"window): {dsr:.3f} (>0.95 = significant after selection)"
            )
    payload = {
        **asdict(result),
        "config_path": str(cfg.path),
        "config_hash": config_hash,
        "data_cache_path": str(cache_path),
        "data_cache_hash": data_cache_hash,
        "hash_drift": hash_drift or None,
        "oos_window": window,
        "oos_trials_for_window": n_trials,
        "deflated_sharpe": dsr,
        "feature_split_mode": cfg.data.feature_split_mode,
        **git_provenance(),
        "detailed": detailed,
    }
    override = (getattr(args, "summary_json", "") or "").strip()
    if override:
        out = Path(override)
        if getattr(args, "_multi_run_summary", False):
            # Batch/ensemble: one fixed path would be silently clobbered per run.
            out = out.with_name(f"{out.stem}_{result.run_id}{out.suffix or '.json'}")
            print(f"[backtest] NOTE: --summary-json with multiple runs → per-run file {out.name}")
    else:
        label = result.checkpoint_label
        name = "backtest_summary.json" if label == "best" else f"backtest_summary_{label}.json"
        out = RunPaths(result.run_id).run_meta_dir / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Backtest summary: {out}")


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
    ensure_backtest_dependencies()
    prefix = args.ensemble_prefix.strip()
    seeds: list[int] | None = None
    if args.ensemble_seeds.strip():
        seeds = [int(s.strip()) for s in args.ensemble_seeds.split(",") if s.strip()]
    run_ids = discover_ensemble_run_ids(prefix, seeds)
    if not run_ids:
        raise SystemExit(
            f"No runs found under Runs/ (or legacy models/) matching '{prefix}_seed_*'. "
            f"Train with scripts/run_seed_ensemble.sh --cohort {prefix}"
        )
    print(f"Discovered {len(run_ids)} runs: {', '.join(run_ids)}")

    modes: list[tuple[str, bool]] = []
    ck = args.ensemble_checkpoint
    if ck in ("best", "both"):
        modes.append(("best", False))
    if ck in ("latest", "both"):
        # Ensemble 'latest' resolves ppo_portfolio_final.zip (run-level final weights),
        # not the newest step checkpoint like single-run --checkpoint latest — label
        # the rows by what is actually evaluated.
        print("[backtest] NOTE: ensemble 'latest' evaluates each run's FINAL model "
              "(end-of-run weights); rows are labeled 'final'.")
        modes.append(("final", True))

    args._ensemble_mode = True  # type: ignore[attr-defined]
    args._multi_run_summary = True  # type: ignore[attr-defined]
    args._invocation_run_ids = list(run_ids)  # type: ignore[attr-defined]
    if not args.no_viz:
        _bt_log("[backtest] Ensemble mode: per-run plots are skipped (μ±σ table + ensemble_summary.json).")
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

    out_dir = RUNS_ROOT / prefix / "plots"
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
    test_asset_vol: np.ndarray,
    test_macro_vol: np.ndarray,
    test_asset_live: np.ndarray,
    obs_lag: int,
    vec_norm_path: Path,
    use_vec_norm: bool,
    deterministic: bool = True,
    collect_weights: bool = False,
    reset_seed: int = 0,
    progress: bool = False,
    progress_label: str = "rollout",
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
        macro=test_macro,
        fracdiff=test_fd,
        fracdiff_macro=test_fdm,
        trend=test_trend,
        asset_realized_vol=test_asset_vol,
        macro_realized_vol=test_macro_vol,
        random_start=False,
        max_episode_steps=n_bars,
        obs_lag=0,
        obs_lag_default=obs_lag,
        fee_scale_default=1.0,
        domain_randomize=False,
        asset_live=test_asset_live,
    )
    vec_env = None
    if use_vec_norm:
        if not vec_norm_path.is_file():
            raise FileNotFoundError(f"VecNormalize not found: {vec_norm_path}")
        venv = DummyVecEnv([lambda: raw_env])
        try:
            vec_env = load_vec_normalize_for_inference(vec_norm_path, venv)
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
    log_every = max(n_bars // 10, 50) if progress else 0
    step_i = 0

    with th.inference_mode():
        while not (done or truncated):
            obs_model = obs.reshape(1, -1) if getattr(obs, "ndim", 1) == 1 else obs
            action, lstm_states = model.predict(
                obs_model,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=deterministic,
            )
            episode_starts = np.zeros((1,), dtype=bool)
            obs, _, done, truncated, info = raw_env.step(action)
            if collect_weights:
                tw = info.get("target_weights")
                if tw is not None:
                    w_rows.append(np.asarray(tw, dtype=np.float64).reshape(-1))
            if use_vec_norm and vec_env is not None:
                obs = vec_env.normalize_obs(obs)
            if "nav" in info:
                navs.append(info["nav"])
            step_i += 1
            if log_every and step_i % log_every == 0:
                pct = min(100, int(100 * step_i / max(n_bars - 1, 1)))
                _bt_log(f"[backtest] {progress_label}: ~{pct}% ({step_i} steps)")

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
    test_asset_vol: np.ndarray,
    test_macro_vol: np.ndarray,
    test_asset_live: np.ndarray,
    obs_lag: int,
    vec_norm_path: Path,
    use_vec_norm: bool,
    base_seed: int = 0,
    progress: bool = False,
) -> np.ndarray:
    """``n_paths`` stochastic rollouts (``deterministic=False``); shape (n_paths, len(navs))."""
    paths: list[np.ndarray] = []
    n_paths = int(n_paths)
    for i in range(n_paths):
        if progress:
            _bt_log(f"[backtest] stochastic path {i + 1}/{n_paths}...")
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
            test_asset_vol=test_asset_vol,
            test_macro_vol=test_macro_vol,
            test_asset_live=test_asset_live,
            obs_lag=obs_lag,
            vec_norm_path=vec_norm_path,
            use_vec_norm=use_vec_norm,
            deterministic=False,
            collect_weights=False,
            reset_seed=base_seed + i + 1,
            progress=progress,
            progress_label=f"stochastic {i + 1}/{n_paths}",
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
    test_asset_live: np.ndarray | None = None,
    spy_ohlcv_col: int | None = None,
    bootstrap_resamples: int = 8000,
    bootstrap_avg_block: int = 10,
    nav_ensemble: np.ndarray | None = None,
    bootstrap_progress: bool = False,
) -> dict:
    """Compute + print detailed OOS stats; return them as a machine-readable dict.

    Ohlcv_window is the full OOS test slice; prices align with test_idx rows.
    """
    stats: dict = {}
    n = len(test_idx)
    t0, t1 = test_idx[0], test_idx[-1]
    cal_days = (t1 - t0).days if hasattr(t1 - t0, "days") else int(
        (np.datetime64(t1) - np.datetime64(t0)) / np.timedelta64(1, "D")
    )
    print("--- detailed ---")
    print(
        f"OOS window: {t0} .. {t1}  ({n} daily bars, ~{cal_days} calendar days)"
    )
    stats["oos_window"] = {"start": str(t0), "end": str(t1), "n_bars": n, "calendar_days": cal_days}
    cagr = float((navs[-1] / max(navs[0], 1e-12)) ** (252.0 / max(len(log_rets), 1)) - 1.0)
    print(f"Compound annualized growth (from daily bars): {cagr * 100:.2f}%")
    mdd = _max_drawdown(navs)
    calmar = float(cagr / max(abs(mdd), 1e-12)) if mdd < 0 else float("nan")
    print(f"Calmar (CAGR / |max DD|): {calmar:.2f}")
    stats["cagr"] = cagr
    stats["max_drawdown"] = float(mdd)
    stats["calmar"] = calmar

    nav_cash = cash_nav(navs, ohlcv_window, start_bar)
    print(
        f"100% cash / no-trade (flat NAV): "
        f"total {(nav_cash[-1] / nav_cash[0] - 1) * 100:.2f}%"
    )
    stats["benchmark_cash"] = {"total_return": float(nav_cash[-1] / nav_cash[0] - 1), "sharpe": float("nan")}

    nav_bench = benchmark_only_nav(
        navs, ohlcv_window, start_bar, tickers=None, benchmark_col=spy_ohlcv_col,
        asset_live=test_asset_live,
    )
    lr_bench = np.diff(np.log(np.maximum(nav_bench, 1e-12)))
    sh_bench = _sharpe_ann_from_log_rets(lr_bench)
    print(
        f"Benchmark-only buy&hold (config benchmark sleeve, {len(lr_bench)} daily rets): "
        f"total {(nav_bench[-1] / nav_bench[0] - 1) * 100:.2f}%, ann. Sharpe {sh_bench:.2f}"
    )
    stats["benchmark_only"] = {
        "total_return": float(nav_bench[-1] / nav_bench[0] - 1),
        "sharpe": sh_bench,
    }
    stats["benchmark_spy"] = stats["benchmark_only"]

    nav_ew = equal_weight_daily_cost_aware_nav(
        navs, ohlcv_window, start_bar, asset_live=test_asset_live
    )
    lr_ew = np.diff(np.log(np.maximum(nav_ew, 1e-12)))
    sh_ew = _sharpe_ann_from_log_rets(lr_ew)
    print(
        f"Equal-weight daily (tx-cost-aware, {len(lr_ew)} daily rets): "
        f"total {(nav_ew[-1] / nav_ew[0] - 1) * 100:.2f}%, ann. Sharpe {sh_ew:.2f}"
    )
    stats["benchmark_equal_weight_daily"] = {
        "total_return": float(nav_ew[-1] / nav_ew[0] - 1),
        "sharpe": sh_ew,
    }
    stats["benchmark_equal_weight"] = stats["benchmark_equal_weight_daily"]

    nav_ew_m = equal_weight_monthly_nav(
        navs, ohlcv_window, start_bar, test_idx, asset_live=test_asset_live
    )
    lr_ew_m = np.diff(np.log(np.maximum(nav_ew_m, 1e-12)))
    sh_ew_m = _sharpe_ann_from_log_rets(lr_ew_m)
    print(
        f"Equal-weight monthly (tx-cost-aware, {len(lr_ew_m)} daily rets): "
        f"total {(nav_ew_m[-1] / nav_ew_m[0] - 1) * 100:.2f}%, ann. Sharpe {sh_ew_m:.2f}"
    )
    stats["benchmark_equal_weight_monthly"] = {
        "total_return": float(nav_ew_m[-1] / nav_ew_m[0] - 1),
        "sharpe": sh_ew_m,
    }
    try:
        nav_6040 = balanced_6040_nav(
            navs, ohlcv_window, start_bar, test_idx, asset_live=test_asset_live
        )
        ret_6040, sh_6040, _ = benchmark_metrics(nav_6040)
        print(
            f"60/40 SP500/BOND10Y (monthly rebal., {len(lr_ew)} daily rets): "
            f"total {ret_6040 * 100:.2f}%, ann. Sharpe {sh_6040:.2f}"
        )
        stats["benchmark_6040"] = {"total_return": float(ret_6040), "sharpe": float(sh_6040)}
    except KeyError as e:
        print(f"60/40 SP500/BOND10Y: skipped (universe missing required sleeve: {e})")
        stats["benchmark_6040"] = None
    nav_rp = naive_risk_parity_nav(
        navs, ohlcv_window, start_bar, asset_live=test_asset_live
    )
    ret_rp, sh_rp, _ = benchmark_metrics(nav_rp)
    print(
        f"Naive risk parity (inverse 20d vol, daily rebal., {len(lr_ew)} daily rets): "
        f"total {ret_rp * 100:.2f}%, ann. Sharpe {sh_rp:.2f}"
    )
    stats["benchmark_risk_parity"] = {"total_return": float(ret_rp), "sharpe": float(sh_rp)}
    nlr = log_rets.size
    subperiods: dict = {}
    for label, n_parts in (("1st/2nd half of OOS", 2), ("quarters of OOS", 4)):
        if nlr < n_parts * 2:
            continue
        w = nlr // n_parts
        parts: list[str] = []
        part_stats: list[dict] = []
        for p in range(n_parts):
            sl = log_rets[p * w : (p + 1) * w if p < n_parts - 1 else nlr]
            if sl.size < 2:
                continue
            tr = float(np.expm1(np.sum(sl)) * 100.0)
            sh = _sharpe_ann_from_log_rets(sl)
            parts.append(f"part{p + 1}: {tr:+.1f}% ret, Sh={sh:.2f}")
            part_stats.append({"total_return_pct": tr, "sharpe": sh})
        if parts:
            print(f"{label}: {', '.join(parts)}")
            subperiods[str(n_parts)] = part_stats
    stats["subperiods"] = subperiods

    if nlr >= 10:
        lo, med, hi = block_bootstrap_sharpe_percentiles(
            log_rets,
            n_resamples=bootstrap_resamples,
            avg_block_size=bootstrap_avg_block,
            seed=42,
            progress=bootstrap_progress,
        )
        print(
            f"Block-bootstrap Sharpe ({bootstrap_resamples} resamples, "
            f"avg block ~{bootstrap_avg_block}d, stationary): "
            f"2.5%={lo:.2f}, 50%={med:.2f}, 97.5%={hi:.2f}"
        )
        stats["bootstrap_sharpe"] = {"p2_5": lo, "p50": med, "p97_5": hi,
                                     "resamples": bootstrap_resamples, "avg_block": bootstrap_avg_block}
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
            stats["stochastic_ensemble"] = {
                "n_paths": int(ens_sh.size),
                "sharpe_mean": float(ens_sh.mean()),
                "sharpe_p5": float(np.percentile(ens_sh, 5)),
                "sharpe_p95": float(np.percentile(ens_sh, 95)),
                "total_return_p5": float(np.percentile(tot, 5)),
                "total_return_p50": float(np.percentile(tot, 50)),
                "total_return_p95": float(np.percentile(tot, 95)),
            }
    skew = float(np.mean(((log_rets - np.mean(log_rets)) / (np.std(log_rets) + 1e-12)) ** 3))
    print(
        f"Skew of daily log returns: {skew:.2f}  (strong positive skew can raise sample Sharpe in short windows)"
    )
    stats["skew_daily_log_returns"] = skew
    print("--- end detailed ---")
    return stats


def _print_backtest_summary(results: list[BacktestResult]) -> None:
    if not results:
        print("No backtest results.")
        return
    print("\n=== OOS summary ===")
    print(f"{'run_id':<16} {'ckpt':<8} {'return%':>10} {'Sharpe':>8} {'maxDD%':>10} {'bars':>6}")
    for r in results:
        print(
            f"{r.run_id:<16} {r.checkpoint_label:<8} "
            f"{r.total_return * 100:>10.2f} {r.sharpe:>8.2f} "
            f"{r.max_drawdown * 100:>10.2f} {r.n_bars:>6}"
        )


def _checkpoint_modes(checkpoint: str) -> list[str]:
    if checkpoint in ("latest", "both"):
        print(
            "[backtest] WARNING: evaluating 'latest'/'both' weights on the OOS holdout — "
            "only 'best' (eval-NAV-selected) is the ex-ante checkpoint. Use --checkpoint best "
            "for the published metric."
        )
    if checkpoint == "both":
        return ["best", "latest"]
    return [checkpoint]


def _configure_checkpoint_subargs(sub: argparse.Namespace, run_id: str, ck: str) -> bool:
    """Set model path / tags for best vs latest. Return False to skip (no latest ckpt)."""
    sub.run_id = run_id
    sub.reuse_panel = True
    sub.model = ""
    sub.allow_latest_checkpoint = False
    if ck == "best":
        sub.plot_tag = "best"
        return True
    latest = _latest_step_checkpoint(run_id)
    if latest is None:
        _bt_log(f"[backtest] {run_id} latest: no checkpoint — skip")
        return False
    sub.model = str(latest)
    sub.plot_tag = "latest"
    return True


def run_backtest_batch(args: argparse.Namespace, run_ids: list[str]) -> list[BacktestResult]:
    """Backtest multiple run ids in one process (one PyTorch import, shared cache loads)."""
    ensure_backtest_dependencies()
    try:
        _get_shared_panel()  # warm the global cache when runs share it
    except FileNotFoundError:
        pass  # runs may use run-local snapshots; each resolves its own cache lazily

    checkpoints = _checkpoint_modes(args.checkpoint)
    results: list[BacktestResult] = []
    t_batch = time.perf_counter()
    for run_id in run_ids:
        if not read_run_manifest(run_id):
            raise SystemExit(f"No manifest for run id {run_id!r} (Runs/{run_id}/manifest.json)")
        for ck in checkpoints:
            sub = copy.copy(args)
            sub.full_policy_load = bool(getattr(args, "full_policy_load", False))
            if not _configure_checkpoint_subargs(sub, run_id, ck):
                continue
            _bt_log(f"[backtest] === {run_id} {ck} ===")
            results.append(run_oos_backtest(sub))

    _print_backtest_summary(results)
    _bt_log(f"[backtest] Batch finished ({len(results)} runs, {time.perf_counter() - t_batch:.1f}s).")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "OOS backtest for a trained run. Holdout dates and universe come from "
            "Runs/<run-id>/manifest.json unless overridden on the CLI."
        ),
    )
    parser.add_argument(
        "--run-ids",
        default="",
        metavar="LIST",
        help="Comma-separated run ids (e.g. W1,W2,W3). One process, shared cache load.",
    )
    parser.add_argument(
        "--checkpoint",
        default="best",
        choices=("best", "latest", "both"),
        help="Which weights to evaluate: eval-NAV-best (default; holdout not used to pick "
        "weights), latest step checkpoint, or both.",
    )
    parser.add_argument(
        "--use-current-config",
        action="store_true",
        help="Use the current global config.yaml instead of Runs/<id>/config.yaml (stress test).",
    )
    parser.add_argument(
        "--data-cache",
        type=str,
        default="",
        metavar="PATH",
        help="Override the panel cache (default: Runs/<id>/data_cache.npz if present, else global).",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="",
        metavar="PATH",
        help="Override path for the per-run backtest_summary.json (default: Runs/<id>/).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Path to a .zip model; if omitted, uses Runs/<run-id>/models/ (best or final).",
    )
    parser.add_argument(
        "--run-id",
        default="",
        metavar="ID",
        help="Run id (required unless --run-ids or --ensemble-prefix). Dates from Runs/<ID>/manifest.json.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda", "auto"),
        help="Torch device for inference (default: cpu — fastest for this model size on Apple Silicon).",
    )
    parser.add_argument(
        "--full-policy-load",
        action="store_true",
        help="Rebuild policy from each checkpoint zip (slower; avoids reusing W1 policy shell in batch).",
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
        "--obs-lag", type=int, default=None,
        help="Market features lag (must match training). Default: manifest args.obs_lag, "
        "else the run config's environment.obs_lag_default.",
    )
    parser.add_argument(
        "--vec-normalize",
        type=str,
        default="",
        metavar="PATH",
        help="VecNormalize .pkl (default: auto from run-id / checkpoints next to model)",
    )
    parser.add_argument(
        "--allow-missing-vec-normalize",
        action="store_true",
        help="Allow backtest without VecNormalize stats (debug only; training uses obs norm by default)",
    )
    parser.add_argument(
        "--allow-raw-obs",
        action="store_true",
        help="Permit a rollout without VecNormalize stats even though the run trained "
        "with norm_obs: true (refused by default — raw-obs metrics are meaningless).",
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
        help="If set, save Runs/<id>/plots/backtest_TAG.png (e.g. latest, best).",
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
        "--fast",
        action="store_true",
        help="Faster run: stochastic-paths=0 and bootstrap-resamples=2000 (unless overridden on CLI).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable [backtest] rollout/bootstrap progress logs.",
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
        help="Aggregate OOS metrics over Runs/<PREFIX>_seed_* runs (μ±σ table).",
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

    if args.run_ids.strip():
        run_ids = _parse_run_id_list(args.run_ids)
        if not run_ids:
            raise SystemExit("--run-ids is empty")
        args._multi_run_summary = len(run_ids) > 1  # type: ignore[attr-defined]
        args._invocation_run_ids = list(run_ids)  # type: ignore[attr-defined]
        full_batch = args.detailed or int(args.stochastic_paths) > 0
        if not full_batch:
            args.no_viz = True
            args.detailed = False
            args.stochastic_paths = 0
            if args.fast:
                args.bootstrap_resamples = 500
        elif args.no_viz:
            _bt_log("[backtest] --no-viz set; skipping plots despite --detailed/--stochastic-paths.")
        run_backtest_batch(args, run_ids)
        return

    if args.fast:
        args.stochastic_paths = 0
        if not any(
            a == "--bootstrap-resamples" or a.startswith("--bootstrap-resamples=")
            for a in sys.argv[1:]
        ):
            args.bootstrap_resamples = 2000
        _bt_log(
            "[backtest] --fast: stochastic-paths=0, "
            f"bootstrap-resamples={args.bootstrap_resamples}"
        )

    if args.ensemble_prefix.strip():
        run_ensemble_backtests(args)
        return

    if not args.run_id.strip():
        raise SystemExit("--run-id is required (holdout dates from Runs/<id>/manifest.json)")

    if not read_run_manifest(args.run_id.strip()):
        raise SystemExit(f"No manifest for {args.run_id!r} (Runs/{args.run_id}/manifest.json)")

    checkpoints = _checkpoint_modes(args.checkpoint)
    if len(checkpoints) == 1:
        ck = checkpoints[0]
        sub = args
        if ck == "latest":
            if not _configure_checkpoint_subargs(sub, args.run_id.strip(), "latest"):
                raise SystemExit(f"No step checkpoint under Runs/{args.run_id}/models/checkpoints/")
        else:
            sub.model = ""
            sub.plot_tag = sub.plot_tag or "best"
        ensure_backtest_dependencies()
        print(f"Model run: {args.run_id}")
        print(f"Backtest plot folder: Runs/{args.run_id}/plots/")
        result = run_oos_backtest(sub)
        _print_single_result(result)
        return

    args.no_viz = True
    run_backtest_batch(args, [args.run_id.strip()])


def _print_single_result(result: BacktestResult) -> None:
    print(f"Model: {result.model_path}")
    if result.checkpoint_label == "best":
        print("Ex-ante checkpoint: eval-NAV-best (best_model.zip) — holdout not used to pick weights.")
    print(f"OOS bars: {result.n_bars}")
    print(f"Total return: {result.total_return * 100:.2f}%")
    print(f"Approx. annualized Sharpe (log-ret, daily): {result.sharpe:.2f}")
    print(f"Max drawdown: {result.max_drawdown * 100:.2f}%")


if __name__ == "__main__":
    main()
