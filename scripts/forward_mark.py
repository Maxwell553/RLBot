#!/usr/bin/env python3
"""Refresh data and export a forward-mark JSON for the ops live dashboard.

Wraps ``scripts/backtest.py --export-forward-mark`` so a daily cron can:

1. Refresh the global data cache (new bars since deploy),
2. Re-roll the frozen LIVE policy on the chronological holdout,
3. Write ``Runs/<run_id>/forward_mark.json`` + ``execution/forward_active.json``.

Example::

    python scripts/forward_mark.py --run-id RLModel --refresh-data
    # or the daily wrapper (shadow record + reconcile + mark):
    bash scripts/daily_live_forward.sh --run-id RLModel
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlbot.data_utils import fetch_aligned_daily, save_cache
from rlbot.forward_mark import load_forward_mark, resolve_active_forward_run_id
from rlbot.rl_config import get_config, load_config, set_config
from rlbot.run_artifacts import PROJECT_ROOT, resolve_data_cache


def _refresh_global_cache() -> Path:
    cfg = get_config()
    cache_path = resolve_data_cache()
    print(f"[forward_mark] refreshing global cache → {cache_path}")
    idx, ohlcv, rsi, macd, macro, fd, fdm, trend, avol, mvol, live = fetch_aligned_daily(
        symbols_dict=cfg.universe.assets,
        since=cfg.data.since,
        until=None,
        fracdiff_d=cfg.data.fracdiff_d,
    )
    save_cache(
        str(cache_path),
        idx,
        ohlcv,
        rsi,
        macd,
        macro,
        fd,
        fdm,
        trend,
        avol,
        mvol,
        asset_live=live,
        fracdiff_d=cfg.data.fracdiff_d,
        tickers=list(cfg.universe.tickers),
    )
    return Path(cache_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default="",
        help="LIVE run id (default: execution/forward_active.json or newest LIVE_* mark)",
    )
    parser.add_argument("--checkpoint", default="best", choices=("best", "latest", "final"))
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Fetch new bars into the global cache before the forward rollup",
    )
    parser.add_argument(
        "--config",
        default="",
        help="Optional config for cache refresh universe (default: current config.yaml)",
    )
    args = parser.parse_args()

    cfg_path = args.config.strip() or str(PROJECT_ROOT / "config" / "config.yaml")
    set_config(load_config(cfg_path))

    run_id = args.run_id.strip() or (resolve_active_forward_run_id() or "")
    if not run_id:
        raise SystemExit(
            "No --run-id and no active LIVE forward mark. Train a LIVE_* run first, "
            "then re-run with --run-id."
        )

    if args.refresh_data:
        cache = _refresh_global_cache()
    else:
        cache = Path(resolve_data_cache())
        if not cache.is_file():
            raise SystemExit(f"Missing cache at {cache}; pass --refresh-data")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "backtest.py"),
        "--run-id",
        run_id,
        "--checkpoint",
        "latest" if args.checkpoint == "final" else args.checkpoint,
        "--data-cache",
        str(cache),
        "--export-forward-mark",
        "--fast",
        "--no-progress",
    ]
    if args.checkpoint == "final":
        cmd.append("--allow-latest-checkpoint")
    print(f"[forward_mark] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    mark = load_forward_mark(run_id)
    if mark:
        stats = (mark.get("stats") or {}).get("model") or {}
        print(
            f"[forward_mark] {run_id}: bars={mark.get('n_bars')} "
            f"ret={stats.get('total_return')} sharpe={stats.get('sharpe')}"
        )


if __name__ == "__main__":
    main()
