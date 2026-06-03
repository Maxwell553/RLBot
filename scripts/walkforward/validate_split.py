#!/usr/bin/env python3
"""Print trainable vs OOS bar counts for a walk-forward window (no training)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rlbot.data_utils import (
    clip_index_until,
    load_cache,
    reserve_chronological_holdout,
    train_test_split_alternating,
)
from rlbot.run_artifacts import resolve_data_cache

DATA_CACHE = resolve_data_cache()

WINDOWS = {
    1: dict(
        until="2017-12-31",
        train_end="2015-12-31",
        holdout_start="2016-01-01",
        holdout_end="2017-12-31",
        holdout_days=365,
    ),
    2: dict(
        until="2019-12-31",
        train_end="2017-12-31",
        holdout_start="2018-01-01",
        holdout_end="2019-12-31",
        holdout_days=365,
    ),
    3: dict(
        until="2021-06-30",
        train_end="2019-12-31",
        holdout_start="2020-01-01",
        holdout_end="2021-06-30",
        holdout_days=365,
    ),
    4: dict(
        until="2022-12-31",
        train_end="2021-06-30",
        holdout_start="2021-07-01",
        holdout_end="2022-12-31",
        holdout_days=365,
    ),
    5: dict(
        until="2024-12-31",
        train_end="2022-12-31",
        holdout_start="2023-01-01",
        holdout_end="2024-12-31",
        holdout_days=365,
    ),
    6: dict(
        until=None,
        train_end="2024-12-31",
        holdout_start="2025-01-01",
        holdout_end=None,
        holdout_days=365,
    ),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, choices=tuple(WINDOWS), required=True)
    args = p.parse_args()
    cfg = WINDOWS[args.window]

    idx, ohlcv, rsi, macd, macro, fd, fdm, _trend, _tickers = load_cache(str(DATA_CACHE))
    if cfg["until"]:
        idx, (ohlcv, rsi, macd, macro, fd, fdm) = clip_index_until(
            idx, ohlcv, rsi, macd, macro, fd, fdm, until=cfg["until"],
        )
    del rsi, macd, fd, fdm  # train/eval split uses isolated per-block features

    (fit_idx, ohlcv_fit, macro_fit), (hold_idx, _, _) = reserve_chronological_holdout(
        idx, ohlcv, macro,
        holdout_days=cfg["holdout_days"],
        train_end=cfg["train_end"],
        holdout_start=cfg["holdout_start"],
        holdout_end=cfg["holdout_end"],
    )

    (tr_idx, *_, tr_b), (ev_idx, *_, ev_b) = train_test_split_alternating(
        fit_idx,
        ohlcv_fit,
        macro_fit,
        block_size=126,
        eval_stride=4,
    )

    print(f"Window {args.window}")
    print(f"  cache clip: {idx[0].date()} .. {idx[-1].date()} ({len(idx)} bars)")
    print(
        f"  trainable:  {fit_idx[0].date()} .. {fit_idx[-1].date()} ({len(fit_idx)} bars)"
    )
    print(
        f"  OOS holdout: {hold_idx[0].date()} .. {hold_idx[-1].date()} ({len(hold_idx)} bars)"
    )
    print(f"  in-train split: train={len(tr_idx)} eval={len(ev_idx)} (alternating 126/stride 4)")


if __name__ == "__main__":
    main()
