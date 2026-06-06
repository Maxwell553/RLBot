"""Audited target-weight inference: given a run id and an as-of date, warm the recurrent
state over recent history (frozen VecNormalize) and emit today's target portfolio weights
with full provenance. No broker calls.

    python scripts/infer_weights.py --run-id <ID> --checkpoint best --as-of 2022-12-31

Reuses the proven backtest rollout (rollout_policy_on_slice) so the observation pipeline,
recurrent warmup, and normalization match training/backtest exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from rlbot.data_utils import clip_index_until, load_cache, resolve_panel_tickers  # noqa: E402
from rlbot.inference_load import load_recurrent_ppo_inference  # noqa: E402
from rlbot.inference_output import build_weights_payload  # noqa: E402
from rlbot.rl_config import get_config, load_config, set_config  # noqa: E402
from rlbot.run_artifacts import (  # noqa: E402
    RunPaths,
    config_sha256,
    git_provenance,
    read_run_manifest,
    resolve_data_cache,
    resolve_run_data_cache,
    sha256_file,
)


def _resolve_model_and_vecnorm(run_id: str, checkpoint: str) -> tuple[Path, Path]:
    rp = RunPaths(run_id)
    if checkpoint == "best":
        model = rp.best_model_dir / "best_model.zip"
        vn = rp.best_model_dir / "vec_normalize.pkl"
        if not vn.is_file():
            vn = rp.models_dir / "vec_normalize.pkl"
    else:
        model = rp.final_model
        vn = rp.models_dir / "vec_normalize.pkl"
    if not model.is_file():
        raise FileNotFoundError(f"model not found: {model}")
    return model, vn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", default="best", choices=("best", "final"))
    parser.add_argument("--as-of", default="", help="As-of date (default: last cache bar).")
    parser.add_argument("--warmup", type=int, default=252, help="Recurrent warmup bars.")
    parser.add_argument("--obs-lag", type=int, default=1, help="Market feature lag (match training).")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--data-cache", default="", metavar="PATH")
    parser.add_argument("--use-current-config", action="store_true")
    parser.add_argument("--out", default="", metavar="PATH", help="Output JSON path.")
    args = parser.parse_args()

    run_id = args.run_id.strip()
    manifest = read_run_manifest(run_id)
    if manifest is None:
        raise FileNotFoundError(f"Missing Runs/{run_id}/manifest.json")

    # Bind the run's own config snapshot (costs/cap/env) unless overridden.
    snap = RunPaths(run_id).config_snapshot
    if snap.is_file() and not args.use_current_config:
        set_config(load_config(snap))
        print(f"[infer] loaded run-local config: {snap}")
    cfg = get_config()

    cache_path = resolve_run_data_cache(run_id, args.data_cache, default=resolve_data_cache())
    print(f"[infer] cache: {cache_path}")

    # rollout_policy_on_slice lives in backtest.py (imports torch); import lazily.
    from scripts.backtest import ensure_backtest_dependencies, rollout_policy_on_slice

    ensure_backtest_dependencies()
    (idx, ohlcv, rsi, macd, macro, fd, fdm, trend, avol, mvol, live, cache_tickers) = load_cache(
        str(cache_path)
    )
    panel_tickers = resolve_panel_tickers(manifest, cache_tickers)

    as_of = args.as_of.strip() or str(idx[-1].date())
    idx, (ohlcv, rsi, macd, macro, fd, fdm, trend, avol, mvol, live) = clip_index_until(
        idx, ohlcv, rsi, macd, macro, fd, fdm, trend, avol, mvol, live, until=as_of
    )
    warm = int(min(max(args.warmup, 10), len(idx)))
    sl = slice(len(idx) - warm, len(idx))

    model_path, vn_path = _resolve_model_and_vecnorm(run_id, args.checkpoint)
    t0 = time.perf_counter()
    model = load_recurrent_ppo_inference(model_path, device=args.device)
    print(f"[infer] loaded model {model_path.name} ({time.perf_counter() - t0:.1f}s)")

    print(f"[infer] warming recurrent state over {warm} bars ending {as_of} ...")
    t0 = time.perf_counter()
    _, _, _, weights = rollout_policy_on_slice(
        model,
        test_idx=idx[sl],
        test_ohlcv=ohlcv[sl],
        test_rsi=rsi[sl],
        test_macd=macd[sl],
        test_macro=macro[sl],
        test_fd=fd[sl],
        test_fdm=fdm[sl],
        test_trend=trend[sl],
        test_asset_vol=avol[sl],
        test_macro_vol=mvol[sl],
        test_asset_live=live[sl],
        obs_lag=args.obs_lag,
        vec_norm_path=vn_path,
        use_vec_norm=vn_path.is_file(),
        deterministic=True,
        collect_weights=True,
    )
    if weights is None or len(weights) == 0:
        raise RuntimeError("rollout produced no weights")
    print(f"[infer] rollout done ({time.perf_counter() - t0:.1f}s)")
    target = np.asarray(weights[-1], dtype=np.float64)

    provenance = {
        "config_path": str(cfg.path),
        "config_hash": config_sha256(cfg.to_dict()),
        "data_cache_path": str(cache_path),
        "data_cache_hash": sha256_file(cache_path),
        "model_path": str(model_path),
        "vec_normalize_path": str(vn_path) if vn_path.is_file() else None,
        "feature_split_mode": cfg.data.feature_split_mode,
        "warmup_bars": warm,
        "obs_lag": args.obs_lag,
        **git_provenance(),
    }
    payload = build_weights_payload(
        run_id=run_id,
        checkpoint=args.checkpoint,
        as_of=as_of,
        weights=target,
        tickers=panel_tickers,
        cap=cfg.environment.max_single_asset_weight,
        asset_live=live[sl][-1],
        provenance=provenance,
    )

    out = Path(args.out) if args.out.strip() else (
        RunPaths(run_id).run_meta_dir / f"target_weights_{as_of}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))
    print(f"[infer] wrote {out}")


if __name__ == "__main__":
    main()
